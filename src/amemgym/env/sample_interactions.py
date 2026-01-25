import json
from amemgym.utils import call_llm


def _generate_user_followup(llm_config, query, user_profile, state_schema, start_date, current_date, conversation_history):
    """
    Generate a follow-up response from the simulated user based on the agent's response.
    
    Args:
        llm_config (dict): Configuration for the LLM
        query (str): The original query that started the conversation
        user_profile (dict): User profile information
        state_schema (dict): Schema for state variables
        start_date (str): Start date of the simulation in YYYY-MM format
        current_date (str): Current date in YYYY-MM format
        conversation_history (list): Previous messages in the conversation
    
    Returns:
        str: User's follow-up message
    """
    # Build conversation context
    context = "\n".join([f"{msg['role'].title()}: {msg['content']}" for msg in conversation_history[-2:]])  # Last 2 messages
    
    followup_prompt = f"""
You are simulating a user in a conversation with an AI assistant. You must continue the conversation - early stopping is not allowed.

Initial User Profile on ({start_date}):
{user_profile['formatted_str']}

Current Date: {current_date}

Initial Query: {query}

Recent Conversation (including the latest assistant response):
{context}

Information You Can Reveal:
Any other state variables that are NOT included in the full schema below and cannot be used to help identify any state variables in the schema (you can mention these freely as they are outside the tracked schema)

Full Schema (DO NOT reveal values for variables in this schema):
{json.dumps(state_schema, indent=2, ensure_ascii=False)}

Instructions:
1. You MUST continue the conversation - do not end it
2. If the assistant asked for clarification, provide a helpful response using information you can reveal as specified above
    - Don't provide further personal information if not asked
    - Don't repeat information already provided in the initial query
3. If your initial query seems addressed, ask a relevant follow-up question that naturally extends the conversation
4. Consider asking about related topics, implementation details, alternatives, or seeking clarification on specific points
5. Keep responses conversational and natural to your persona
6. You can mention any state variables that are NOT in the schema above, but ensure they cannot help identify values of variables in the schema
    - DO NOT reveal specific values for any state variables that are in the schema
7. Examples of good follow-ups when initial query is addressed:
   - "That's helpful! Could you also tell me about..."
   - "Thanks for that information. I'm also curious about..."
   - "That makes sense. What about..."
   - "Good to know. Is there anything else I should consider regarding..."

You must respond with a natural follow-up response that continues the conversation. Return only the response text, no additional formatting or explanation.
"""
    messages = [{"role": "user", "content": followup_prompt}]
    response = call_llm(messages, llm_config, json=False).strip()
    
    return response


def sample_session_given_query(
    llm_config, query, agent, start_date, user_profile, current_date, state_schema, hist=None, max_rounds=10
):
    """
    Sample a session in a specific period between a simulated user and an AI agent, given a query.

    Args:
        llm_config (dict): Configuration for the LLM, including model, temperature, etc.
        query (str): The query to be answered by the agent, which identifies the user's intent and main topic of the session.
        agent (Agent): The AI agent that will interact with the user.
        start_date (str): The start date of the simulation.
        user_profile (dict): The user profile containing background information.
        current_date (str): The current date in YYYY-MM format.
        state_schema (dict): Schema defining the structure of all state variables and possible values.
        hist (list, optional): Optional history of previous messages in the session. Defaults to None.
        max_rounds (int): Maximum number of rounds in the session.
    
    Returns:
        list: A list of messages representing the session, including user queries, agent responses, and follow-up interactions.
    """
    # Initialize session
    if hist is None:
        session_messages = []
        current_user_input = query
        init_num_rounds = 0
    else:
        session_messages = hist.copy()
        init_num_rounds = len(session_messages) // 2  # Each round has a user and agent message
        if init_num_rounds >= max_rounds:
            return session_messages
        current_user_input = _generate_user_followup(
            llm_config, query, user_profile, state_schema,
            start_date, current_date, session_messages
        )

    for num_rounds in range(init_num_rounds, max_rounds):
        # Add user input to session
        session_messages.append({"role": "user", "content": current_user_input})

        # Get agent's response
        agent_response = agent.act(current_user_input)
        session_messages.append({"role": "assistant", "content": agent_response})
        
        # Check if we should continue the conversation
        if num_rounds < max_rounds - 1:  # Don't generate follow-up on last round
            # Generate user follow-up (always continues, no early stopping)
            user_followup = _generate_user_followup(
                llm_config, query, user_profile, state_schema,
                start_date, current_date, session_messages
            )
            current_user_input = user_followup
    
    return session_messages
