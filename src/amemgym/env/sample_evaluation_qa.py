import json
import itertools
import random

from amemgym.utils.llm_utils import call_llm
from loguru import logger


def sample_personalized_answers(llm_config, question, state_variants):
    """
    Sample personalized answers for a question based on state variants.

    Args:
        llm_config (dict): Configuration for the LLM.
        question (dict): The question to answer, containing 'question' and 'required_info'.
            - 'question' (str): The question text.
            - 'required_info' (list): List of required information items, each with 'info_type' and 'info_choices'.
                - 'info_type' (str): The type of information required.
                - 'info_choices' (list): List of choices for this information type.
        state_variants (list): List of state variants to consider.

    Returns:
        list: A list of dictionaries, each containing:
            - 'variant': The state variant as a list of values (for json compatibility).
            - 'answer': The personalized answer for that variant.
    """
    # Extract required info types for context
    required_info_types = [info['info_type'] for info in question['required_info']]
    
    # Format state variants for the prompt
    variants_text = ""
    for i, variant in enumerate(state_variants, 1):
        variant_info = []
        for info_type, choice in zip(required_info_types, variant):
            variant_info.append(f"{info_type}: {choice}")
        variants_text += f"Variant {i}: {', '.join(variant_info)}\n"
    
    prompt = f"""\
You are an expert advisor providing personalized recommendations. Answer the following question for each state variant provided. Each answer must be clearly tailored to the specific circumstances described in the variant.

**Question:** {question['question']}

**Required Information Types:** {', '.join(required_info_types)}

**State Variants to Answer For:**
{variants_text}

**Instructions:**
1. Provide a distinct, personalized answer for each variant
2. Each answer should be 2-3 sentences long
3. Clearly reflect the specific values in each variant
4. Make the differences between answers evident and meaningful
5. Use practical, actionable advice
6. Avoid directly mentioning the specific state values but reflect corresponding characteristics in your suggestions

Return your response in JSON format:
{{
  "variant_1": "personalized answer for variant 1",
  "variant_2": "personalized answer for variant 2",
  ...
}}

Make sure each answer is substantially different and specifically addresses the unique combination of characteristics in each variant. Ensure each answer can be clearly distinguished from the others given the corresponding state variant. Write the answers in the same language as the question.
"""

    messages = [{"role": "user", "content": prompt}]
    response = call_llm(messages, llm_config, json=True)
    
    try:
        parsed_response = json.loads(response)
        
        # Map the responses back to the actual state variants
        answers = []
        for i, variant in enumerate(state_variants, 1):
            variant_key = f"variant_{i}"
            if variant_key not in parsed_response:
                logger.warning(f"Missing answer for {variant_key}")
            answers.append({
                "variant": list(variant),
                "answer": parsed_response[variant_key]
            })
        return answers
    
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Error parsing personalized answers: {e}\nRaw response: {response}")
        raise e


def check_personalized_answer(llm_config, question, answer, variants, matched_index):
    """
    Validate a personalized answer to ensure it aligns with the question and required information.

    Args:
        llm_config (dict): Configuration for the LLM.
        question (dict): The question to answer, containing 'question' and 'required_info'.
            - 'question' (str): The question text.
            - 'required_info' (list): List of required information items, each with 'info_type' and 'info_choices'.
                - 'info_type' (str): The type of information required.
                - 'info_choices' (list): List of choices for this information type.
        answer (str): The personalized answer to validate.
        variants (list): List of state variants considered.
        matched_index (int): The index of the variant that this answer is supposed to correspond to.

    Returns:
        str: The validated answer, potentially revised for clarity or correctness.
    """
    required_info_types = [info['info_type'] for info in question['required_info']]

    choices = "\n".join([
        f"{i+1}. " + json.dumps({k: v for k, v in zip(required_info_types, variant)})
        for i, variant in enumerate(variants)
    ])
    
    prompt = f"""\
You are an expert evaluator. Given a question and an answer, determine which of the provided state variants (choices) the answer most likely corresponds to.

**Question:** {question['question']}

**Answer to Evaluate:** {answer}

**Available State Variants (Choices):**
{choices}

**Instructions:**
1. Analyze the answer to understand what specific characteristics or circumstances it addresses
2. Compare these characteristics with each state variant
3. Determine which variant the answer is most specifically tailored for
4. Return only the number (1, 2, 3, etc.) of the best matching choice

Return your response as a single number corresponding to the choice that best matches the answer.
"""

    messages = [{"role": "user", "content": prompt}]
    response = call_llm(messages, llm_config, json=False)
    try:
        choice_index = int(response.strip()) - 1  # Convert to 0-based index
        if choice_index == matched_index:
            return True
    except (ValueError, TypeError):
        pass
    return False


def refine_personalized_answer(llm_config, question, answer, variants, matched_index):
    """
    Refine a personalized answer to better align with the question and required information.

    Args:
        llm_config (dict): Configuration for the LLM.
        question (dict): The question to answer, containing 'question' and 'required_info'.
            - 'question' (str): The question text.
            - 'required_info' (list): List of required information items, each with 'info_type' and 'info_choices'.
                - 'info_type' (str): The type of information required.
                - 'info_choices' (list): List of choices for this information type.
        answer (str): The personalized answer to refine.
        variants (list): List of state variants considered.
        matched_index (int): The index of the variant that this answer is supposed to correspond to.

    Returns:
        str: The refined answer.
    """
    required_info_types = [info['info_type'] for info in question['required_info']]
    matched_state = json.dumps({k: v for k, v in zip(required_info_types, variants[matched_index])})
    other_states = [
        {k: v for k, v in zip(required_info_types, variant)}
        for i, variant in enumerate(variants) if i != matched_index
    ]
    other_states_text = "\n".join([json.dumps(state) for state in other_states])

    prompt = f"""\
You are an expert advisor providing personalized recommendations. Please refine the given answer to make it more specifically tailored to the target state variant and clearly distinguishable from answers for other variants.

**Question:** {question['question']}

**Target State Variant (the answer should correspond to this):**
{matched_state}

**Other State Variants (the answer should be distinguishable from these):**
{other_states_text}

**Current Answer to Refine:**
{answer}

**Instructions:**
1. Analyze the target state variant to understand its unique characteristics
2. Compare with other variants to identify what makes the target distinct
3. Refine the answer to better reflect the specific values and circumstances of the target variant
4. Ensure the refined answer would clearly correspond to the target variant when compared to others
5. Keep the answer 2-3 sentences long and practical
6. Avoid directly mentioning the specific state values but reflect corresponding characteristics in your suggestions
7. Make the differences more evident and meaningful

Return your response in JSON format:
{{
  "answer": "the refined answer text here"
}}

Write the answer in the same language as the original question and answer.
"""
    messages = [{"role": "user", "content": prompt}]
    response = call_llm(messages, llm_config, json=True)
    try:
        parsed_response = json.loads(response)
        return parsed_response.get("answer", answer)
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Error parsing refined answer: {e}\nRaw response: {response}")
        return answer


def get_state_variants(states, questions, min_variants=4):
    """
    Generate a list of state variants based on the evolution of state variables and the questions asked.

    Args:
        states (list): A list of state snapshots, where each snapshot is a dictionary of state variables and their values.
        questions (list): A list of questions to be asked, answers to which will depend on the state variables. Each question is a dictionary containing:
            - 'question': The question text.
            - 'required_info': A list of dictionaries, each representing a piece of information required to answer the question. Each dictionary contains:
                - 'info_type': The name of the information required.
                - 'info_choices': A list of possible choices for that information.
        min_variants (int): Minimum number of state variants to ensure diversity in answers.

    Returns:
        list: A dictionary mapping each question to a list of unique state variants
    """
    state_variants = {}
    for question in questions:
        info_types = [info["info_type"] for info in question["required_info"]]
        variants = set()
        for state in states:
            variant = tuple(state[info_type] for info_type in info_types)
            variants.add(variant)

        variant_list = list(variants)

        # If we don't have enough variants, sample from all valid combinations
        if len(variant_list) < min_variants:
            # Generate all possible combinations from info_choices
            info_choices_lists = [info["info_choices"] for info in question["required_info"]]
            all_combinations = list(itertools.product(*info_choices_lists))
            remaining_combinations = list(set(all_combinations) - variants)
            additional_needed = min_variants - len(variant_list)
            assert len(remaining_combinations) >= additional_needed, \
                "Not enough unique combinations available to fit the minimum variants requirement"
            additional_variants = random.sample(remaining_combinations, additional_needed)
            variant_list.extend(additional_variants)

        state_variants[question["question"]] = variant_list
    return state_variants
