import json
from amemgym.utils import call_llm
from loguru import logger
from backoff import on_exception, expo


def sample_initial_state(llm_config, start_date, user_profile, num_total_months, state_schema):
    """
    Sample initial state values for a user's personal state variables based on their current profile and a predefined schema.

    Args:
        llm_config (dict): Configuration for the LLM, including model, temperature, etc
        start_date (str): The current date in YYYY-MM-DD format.
        user_profile (dict): The user's profile containing personal information.
        num_total_months (int): The total number of months to consider for future state evolution.
        state_schema (dict): A schema defining the state variables and their possible values.
    """
    prompt = f"""
You are tasked with selecting initial values for a user's personal state variables. The goal is to choose values that:
1. Are consistent with the user's current profile
2. Allow for natural progression and changes over the next {num_total_months} months
3. Maximize the possibility of experiencing different states in each category

User Profile (on the current date {start_date}):
{user_profile}

State Schema (each key represents a state variable with possible values):
{json.dumps(state_schema, indent=2, ensure_ascii=False)}

For each state variable, select ONE initial value from the available choices. Consider:
- The user's current profile and background
- Values that are neither at the extreme beginning nor end of ranges (to allow growth in both directions)
- Realistic starting points that could naturally evolve in future updates

Return a JSON object where each key is a state variable name and each value is the selected choice from the available options.
"""
    
    messages = [{"role": "user", "content": prompt}]
    response = call_llm(messages, llm_config, json=True)
    
    try:
        initial_state = json.loads(response)
        
        # Validate that all keys exist in schema and values are valid choices
        for key, value in initial_state.items():
            if key not in state_schema:
                raise ValueError(f"Invalid state variable: {key}")
            if value not in state_schema[key]:
                raise ValueError(f"Invalid choice '{value}' for state variable '{key}'. Valid choices: {state_schema[key]}")
        
        # Ensure all schema keys are present
        for key in state_schema:
            if key not in initial_state:
                raise ValueError(f"Missing state variable: {key}")
        
        return initial_state
    
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Error parsing initial state: {e}\nRaw response: {response}")
        raise e


@on_exception(expo, Exception, max_tries=5)
def sample_state_updates(
    llm_config, start_date, user_profile, num_months, current_date, end_date,
    num_changes_per_period, max_changes_per_state,
    state_schema, latest_state, prior_updates, update_cnts,
    remaining_steps=10, total_steps=10, error_hist=()
):
    """
    Sample updates to the user's state variables for a preriod based on their current profile and the current state.
    
    Args:
        llm_config (dict): Configuration for the LLM, including model, temperature, etc
        start_date (str): The start date of the simulation in YYYY-MM-DD format.
        user_profile (str): The user's profile containing personal information at the start of the simulation.
        num_months (int): The number of months in the current period for which updates are to be made.
        current_date (datetime): The current date at the start of the period.
        end_date (datetime): The end date of the current period.
        num_changes_per_period (int): The expected number of state variables to change in this period.
        max_changes_per_state (int): The maximum number of times a single state variable can be changed across all periods.
        state_schema (dict): A schema defining the state variables and their possible values.
        latest_state (dict): The most recent state of the user's personal information variables.
        prior_updates (list): A list of prior updates made to the user's state variables.
        update_cnts (dict): A dictionary tracking how many times each state variable has been updated.
        remaining_steps (int): The remaining number of steps to simulate updates for.
        total_steps (int): The total number of steps in the simulation.
        error_hist (tuple): A tuple containing any previous error information for reflection.

    Returns:
        dict: A dictionary containing updates and a summary of the corresponding period.
            - "updates": A dictionary where each key is an updated state variable and each value is the updated value.
            - "period_start": A string representing the start date of the period (YYYY-MM-DD) for which updates are made.
            - "period_end": A string representing the end date of the period (YYYY-MM-DD) for which updates are made (3 months after the current date).
            - "period_summary": A string summarizing the changes and context for the updates in the specified period.
    """
    # Calculate period end date (3 months later)
    current_date_str = current_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
    
    prompt = f"""
Generate realistic state updates for a user over the next {num_months}-month period.

**Context:**
- Step {total_steps - remaining_steps + 1} of {total_steps} (remaining: {remaining_steps - 1})
- Current: {current_date_str} → Target: {end_date_str}

**User Profile (on the start date {start_date}, step 0):**
{user_profile}

**State Schema:**
{json.dumps(state_schema, indent=2, ensure_ascii=False)}

**Current State:**
{json.dumps(latest_state, indent=2, ensure_ascii=False)}

**Prior Updates:**
{json.dumps(prior_updates, indent=2, ensure_ascii=False)}

**Update Counts (prioritize variables with <{max_changes_per_state} updates):**
{json.dumps(update_cnts, indent=2, ensure_ascii=False)}

**REQUIREMENTS:**
1. Update ~{num_changes_per_period} state variables only
2. **Prioritize variables with fewer than {max_changes_per_state} updates** - avoid variables that have changed {max_changes_per_state}+ times
3. Changes must be realistic and gradual
4. States with strong dependencies should be updated together (e.g., `experience` affects `team_size`)
5. Values must be different from the current state and selected from corresponding valid choices
6. Leave room for future progression

**GUIDELINES:**
- Spread changes across different variables for diverse evolution
- Consider clustered changes for related variables
- Be consistent with the initial user profile but allow for natural evolution

Return JSON format:
{{
  "period_summary": "Brief explanation of changes and context for updates in the period",
  "updated": {{
    "state_variable": "new_value"
  }}
}}
"""
    messages = [{"role": "user", "content": prompt}]
    if len(error_hist) >= 4:
        raise ValueError("Too many errors encountered, stopping further updates.")
    if error_hist:
        error = error_hist[-1]
        messages.extend([
            {"role": "assistant", "content": error["response"]},
            {"role": "user", "content": f"Please try again to fix the error {error['info']} in your response."}
        ])
    response = call_llm(messages, llm_config, json=True)
    
    try:
        error_info = None
        update_info = json.loads(response)
        updates = update_info["updated"]
        # check number of changes
        if not (num_changes_per_period - 1 <= len(updates) <= num_changes_per_period + 1):
            error_info = {
                "response": json.dumps(update_info, indent=2, ensure_ascii=False),
                "info": f"Number of changes {len(updates)} not in expected range [{num_changes_per_period - 1}, {num_changes_per_period + 1}]"
            }
            update_info = sample_state_updates(
                llm_config, start_date, user_profile, num_months, current_date, end_date,
                num_changes_per_period, max_changes_per_state,
                state_schema, latest_state, prior_updates, update_cnts,
                remaining_steps, total_steps, error_hist + (error_info,)
            )

        # Validate each update
        for state_var, new_value in updates.items():
            if state_var not in state_schema:
                error_info = {
                    "response": json.dumps(update_info, indent=2, ensure_ascii=False),
                    "info": f"Invalid state variable '{state_var}' in updates"
                }                
            if new_value not in state_schema[state_var]:
                error_info = {
                    "response": json.dumps(update_info, indent=2, ensure_ascii=False),
                    "info": f"Invalid value '{new_value}' for state variable '{state_var}'. Valid choices: {state_schema[state_var]}"
                }
            if latest_state[state_var] == new_value:
                error_info = {
                    "response": json.dumps(update_info, indent=2, ensure_ascii=False),
                    "info": f"State variable '{state_var}' is not actually changing from '{new_value}'"
                }
        if error_info:
            update_info = sample_state_updates(
                llm_config, start_date, user_profile, num_months, current_date, end_date,
                num_changes_per_period, max_changes_per_state,
                state_schema, latest_state, prior_updates, update_cnts,
                remaining_steps, total_steps, error_hist + (error_info,)
            )
        update_info["period_end"] = end_date_str
        update_info["period_start"] = current_date_str
        return update_info
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Error parsing state updates: {e}\nRaw response: {response}")
        raise e


@on_exception(expo, Exception, max_tries=5)
def elaborate_state_updates(
    llm_config, start_date, user_profile, current_state, updates, state_schema
):
    """Elaborate on the state updates by providing triggers for each change.

    Args:
        llm_config (dict): Configuration for the LLM, including model, temperature, etc.
        start_date (str): The date when the user's profile was created, in YYYY-MM-DD format.
        user_profile (str): The user's profile containing personal information at the start of the simulation.
        current_state (dict): The current state of the user's personal variables.
        updates (dict): A dictionary containing the latest state updates.
            - "period_start": The start date of the period for which updates are made (YYYY-MM-DD).
            - "period_end": The end date of the period for which updates are made (YYYY-MM-DD).
            - "period_summary": A string summarizing the changes and context for the updates.
            - "updated": The latest state updates, where each key is an updated state variable and each value is the updated value.
            - "old": The previous state of the user's personal variables before the latest updates.
        state_schema (dict): A schema defining the state variables and their possible values.

    Returns:
        list: A list of dictionaries representing the elaborated events and their relationships to state changes. Each dictionary contains:
            - "event": A description of the event that serves as a trigger/implication for the state change.
            - "states": A list of state variables that are affected by this event.
    """
    states_not_updated = {k: v for k, v in current_state.items() if k not in updates["updated"]}
    
    # Extract state changes for context
    state_changes = []
    for state_var, new_value in updates["updated"].items():
        old_value = updates["old"][state_var]
        state_changes.append({
            "variable": state_var,
            "from": old_value,
            "to": new_value,
            "possible_values": state_schema[state_var]
        })
    
    prompt = f"""
Generate realistic life events that serve as triggers or implications for the user's state changes during the specified period.

**User Profile (on the start date {start_date}):**
{user_profile}

**Period:** {updates["period_start"]} to {updates["period_end"]}
**Period Context:** {updates["period_summary"]}

**State Changes:**
{json.dumps(state_changes, indent=2, ensure_ascii=False)}

**States NOT Updated (should remain unchanged):**
{json.dumps(states_not_updated, indent=2, ensure_ascii=False)}

**REQUIREMENTS:**
1. Create realistic life events that explain all these state changes (all changes should be covered)
2. Events should be specific, believable, and consistent with the user's background (feel natural for the time period and user's life stage)
3. **Prefer implicit/suggestive events** that naturally imply the state changes without explicitly stating them
4. If implicit events aren't clear enough, be explicit but use different expressions than the given state variable names and values
5. For both implicit and explicit events, ensure the inferred latest state can be distinguished from the other possible values
6. Group related state changes under single events when logical
7. **Events should NOT affect or imply changes to states that weren't updated** - be careful not to suggest changes to unchanged states

**EVENT GUIDELINES:**
- Use concrete, specific scenarios (e.g., "Started leading a cross-functional project targeting ..." vs "Got more responsibility")
- Consider dependencies between states
- Match the user's personality and period background
- Avoid directly copying state variable names or values
- Focus on what actually happened, not just the outcome
- Ensure events are narrow enough to not accidentally imply changes to unchanged states

Return JSON format:
{{
  "events": [
    {{
      "states": ["list", "of", "affected", "state", "variables"],
      "event": "Specific description of what happened"
    }}
  ]
}}
"""
    messages = [{"role": "user", "content": prompt}]
    response = call_llm(messages, llm_config, json=True)
    
    try:
        result = json.loads(response)
        events = result["events"]
        covered_states = set()

        
        # Validate that all mentioned states are in the updates
        for event_info in events:
            for state_var in event_info["states"]:
                if state_var not in updates["updated"]:
                    # logger.warning(f"Event mentions state variable '{state_var}' that wasn't updated")
                    raise ValueError("Event mentions state variable that wasn't updated")
                covered_states.add(state_var)
        
        if len(covered_states) != len(state_changes):
            raise ValueError(
                f"Not all state changes are covered in the events. Covered: {covered_states}, Expected: {set(updates['updated'].keys())}"
            )
        
        return events
    
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Error parsing event elaboration: {e}\nRaw response: {response}")
        # Return a fallback structure
        raise e
