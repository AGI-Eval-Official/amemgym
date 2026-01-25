import json
from amemgym.utils import call_llm
from loguru import logger


def sample_update_queries(llm_config, start_date, user_profile, state_schema, updates):
    """
    Sample user queries that can be used to update the user's personal state.

    Args:
        llm_config (dict): Configuration for the LLM.
        start_date (str): The start date in YYYY-MM-DD format.
        user_profile (str): The user's profile containing background information as a formatted string on start_date.
        state_schema (dict): A schema defining the state variables and their possible values.
        updates (dict): A dictionary of state variables to be updated with their new values.

    Returns:
        list: A list of sampled queries that can be used to update the user's personal state.
    """
    old_state = updates["old"]
    new_state = updates["updated"]
    covered = {k: False for k in updates["updated"]}
    context = []
    for event in updates["events"]:
        context_i = {"background": event["event"], "state_transition": {}}
        for k in event["states"]:
            covered[k] = True
            context_i["state_transition"][k] = {
                "old": old_state[k],
                "new": new_state[k]
            }
        context.append(context_i)
    assert all(covered.values()), "Not all state variables are covered in the updates"
    prompt = f"""
You are helping to generate queries that a user would naturally ask you in their daily life. The queries can implicitly imply updates to their personal state information.

Initial User Profile on ({start_date}):
{json.dumps(user_profile, indent=2, ensure_ascii=False)}

State Updates Context ({updates["period_start"]} to {updates["period_end"]}):
{json.dumps(context, indent=2, ensure_ascii=False)}

Available State Schema:
{json.dumps(state_schema, indent=2, ensure_ascii=False)}

Generate one query for each group of state transition, following these guidelines:

1. Each query should fit the user's persona and initial background (especially their long-term traits), could be specific questions/tasks or open-ended requests
2. Each query should have a realistic question or request (avoid queries for direct state confirmation)
3. Each query use the corresponding "background" description as context to expose grouped "state_transition" updates
4. Ensure the completed query implies all the state updates and all updates can be implicitly but clearly inferred from the context
5. Remove details in background text if they reflect other state variables in the schema that are not being updated
5. Ensure the queries are natual and contextual to the user's situation

Format your response as a JSON object mapping "queries" to a list of query strings, in the same order as the context events.
"""
    messages = [{"role": "user", "content": prompt}]
    response = call_llm(messages, llm_config, json=True)
    
    try:
        queries = json.loads(response)["queries"]
        if len(queries) != len(updates["events"]):
            raise ValueError(f"Number of queries {len(queries)} does not match number of updates {len(updates['events'])}")
        outputs = []
        for query, event in zip(queries, updates["events"]):
            exposed_states = {k: new_state[k] for k in event["states"]}
            outputs.append({
                "query": query,
                "exposed_states": exposed_states
            })
        return outputs
    except Exception as e:
        print(f"Error processing LLM response: {e}")
        print("Response content:", response)
        raise e


def sample_init_queries(llm_config, start_date, user_profile, state_schema, initial_state):
    """
    Sample several queries that can be used to expose the user's initial state.

    Args:
        llm_config (dict): Configuration for the LLM.
        start_date (str): The start date in YYYY-MM-DD format.
        user_profile (str): The user's profile containing background information as a formatted string on start_date.
        state_schema (dict): A schema defining the state variables and their possible values.
        initial_state (dict): The current initial state of the user's personal variables.

    Returns:
        list: A list of sampled queries that can be used to expose the user's personal state. Each contains:
            - "query": The query string.
            - "exposed_states": A dictionary mapping state variable names to their values. 
    """
    prompt = f"""
You are helping to generate natural queries that a user would ask, which can indirectly reveal their personal state information.

User Profile (on the current date {start_date}):
{user_profile}

User's Current State (to be exposed through queries):
{json.dumps(initial_state, indent=2, ensure_ascii=False)}

Available State Schema:
{json.dumps(state_schema, indent=2, ensure_ascii=False)}

Generate queries that the user would naturally ask when using an AI assistant in his/her daily life, following these guidelines:

1. Each query should fit the user's persona and background
2. Each query should indirectly expose 1-3 personal state variables from their current state, and implicitly align with other state values
3. Ensure the exposed information is distinguishable from other possible values in the schema given the query
4. Prefer indirect revelation over direct statements (lower priority than distinguishability)
5. Make queries sound natural and contextual to the user's situation
6. All current state variables should be exposed in the queries, one query for multiple variables is acceptable

For each query, specify:
- "exposed_states": A dictionary mapping state variable names to their current values that would be revealed
- "query": The natural language query the user would ask

Format your response as a JSON list of query objects.

Example format:
{{
    "queries": [
        {{
            "exposed_states": {{
                "work_location": "home",
                "work_schedule": "flexible"
            }},
            "query": "What's the best way to stay productive when I can set my own hours and don't have to commute to an office?"
        }},
        ...
    ]
}}
"""
    messages = [{"role": "user", "content": prompt}]
    response = call_llm(messages, llm_config, json=True)
    try:
        queries = json.loads(response)["queries"]
        exposed_states = set()
        for state in queries:
            for key, value in state["exposed_states"].items():
                assert value == initial_state[key], f"Exposed state {key} has unexpected value {value}, expected {initial_state[key]}"
                exposed_states.add(key)
        assert len(exposed_states) == len(initial_state), "Not all initial states were exposed in the queries"
        return queries
    except Exception as e:
        print(f"Error processing LLM response: {e}")
        print("Response content:", response)
        raise e


def check_query_state_exposure(llm_config, query, exposed_states, state_schema):
    """
    Check if a query correctly exposes the intended state variables.

    Args:
        llm_config (dict): Configuration for the LLM.
        query (str): The user's query.
        exposed_states (dict): A dictionary mapping state variable names to their intended values.
        state_schema (dict): A schema defining the state variables and their possible values.

    Returns:
        bool: True if the query correctly exposes the intended states, False otherwise.
    """
    
    state_choices = {k: state_schema[k] for k in exposed_states}
    
    prompt = f"""\
Given the following user query and state schema, predict the most likely values for the specified state variables based on what can be inferred from the query.

User Query: "{query}"

State Variables to Predict:
{json.dumps(state_choices, indent=2, ensure_ascii=False)}

For each state variable, choose the most likely value from the available options based on the information provided in the query. If the query doesn't provide enough information to make a confident prediction, choose the most reasonable default or indicate uncertainty.

Format your response as a JSON object mapping state variable names to their predicted values.

Example format:
{{
    "state_variable_1": "predicted_value_1",
    "state_variable_2": "predicted_value_2"
}}
"""
    messages = [{"role": "user", "content": prompt}]
    response = call_llm(messages, llm_config, json=True) 
    try:
        predicted_states = json.loads(response)
        # Compare predicted states with expected exposed states
        for state_var, expected_value in exposed_states.items():
            if state_var not in predicted_states:
                logger.warning(f"State variable '{state_var}' not predicted")
                return False
            if predicted_states[state_var] != expected_value:
                logger.warning(f"State variable '{state_var}': predicted '{predicted_states[state_var]}', expected '{expected_value}'")
                return False
        return True
        
    except Exception as e:
        print(f"Error processing LLM response: {e}")
        print("Response content:", response)
        return False


def refine_query(llm_config, query, exposed_states, state_schema):
    """
    Refine a user query to better expose the intended state variables.

    Args:
        llm_config (dict): Configuration for the LLM.
        query (str): The original user query.
        exposed_states (dict): A dictionary mapping state variable names to their intended values.
        state_schema (dict): A schema defining the state variables and their possible values.

    Returns:
        str: The refined query that better exposes the intended states.
    """
    
    state_choices = {k: state_schema[k] for k in exposed_states}
    
    prompt = f"""\
You are helping to refine a user query to better expose specific personal state information.

Original Query: "{query}"

Intended State Variables to Expose:
{json.dumps(exposed_states, indent=2, ensure_ascii=False)}

Available State Schema:
{json.dumps(state_choices, indent=2, ensure_ascii=False)}

Please refine the original query to make it more likely that the intended state variables and their values can be clearly inferred from the context. The refined query should:

1. Maintain the natural tone and user persona
2. Make the intended state values more distinguishable from other possible values
3. Include sufficient context clues to expose the target states
4. Still sound like a natural request a user would make

Format your response as a JSON object with the refined query.

Example format:
{{
    "query": "Your refined query text here"
}}
"""
    
    messages = [{"role": "user", "content": prompt}]
    response = call_llm(messages, llm_config, json=True)
    
    try:
        refined_query = json.loads(response)["query"]
        return refined_query.strip()
    except Exception as e:
        logger.error(f"Error processing LLM response: {e}")
        logger.error("Response content:", response)
        raise e
