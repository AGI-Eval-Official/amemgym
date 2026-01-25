import json
from amemgym.utils import call_llm
from loguru import logger
    

def sample_user_questions(
    llm_config, start_date, user_profile, num_questions=10, num_states_per_question=2, num_choices_per_state=3, num_total_months=30, language="en"
):
    """
    Samples a set of potential questions from the user with the given user profile. 
    The questions are designed to be asked by the user for suggestions or advice, 
    and require specific personal information to answer. They can be asked at any time in
    several years, regardless of the user's development and experience at that time.
    
    Args:
        llm_config (dict): Configuration for the LLM, including model, temperature, max tokens, etc.
        start_date (datetime): start date as a datetime object
        user_profile (str): User profile as a formatted string
        num_questions (int): Number of questions to generate (default: 10)
        num_states_per_question (int): Number of required_info items per question (default: 2)
        num_choices_per_state (int): Number of choices per required_info item (default: 3)
        num_total_months (int): Total months to consider for user development (default: 30)
        language (str): Language to generate questions in (default: "en")
    
    Returns:
        list[dict]: List of question dictionaries with required_info
    """
    # Set language-specific prompts
    if language.lower() in ["chinese", "zh", "中文"]:
        prompt_lang = "Chinese (中文)"
        example_format = """
[
    {
        "question": "我应该如何制定职业发展规划？",
        "required_info": [
            {
                "info_type": "当前经验水平",
                "info_choices": ["初级(0-2年)", "中级(3-5年)"]
            },
            {
                "info_type": "家庭状况",
                "info_choices": ["单身", "已婚无子女", "已婚有子女"]
            }
        ]
    }
]"""
    else:
        prompt_lang = "English"
        example_format = """
[
    {
        "question": "How should I plan my career development strategy?",
        "required_info": [
            {
                "info_type": "current_experience_level",
                "info_choices": ["junior_0_2_years", "mid_level_3_5_years"]
            },
            {
                "info_type": "family_status",
                "info_choices": ["single", "married_no_children", "married_with_children"]
            }
        ]
    }
]"""
    
    detailed_prompt = f"""You are a helpful assistant that generates realistic questions that users would ask an AI assistant for suggestions or advice.

Given the following context:
- User Profile (on current date {start_date}):\n{user_profile}

Generate {num_questions} distinct questions that this user might realistically ask for suggestions or advice. Each question should:

1. Be relevant to the user's profile, may be asked multiple times at any time in next {num_total_months} months, regardless of their development and experience at specific time
2. Require specific personal information to provide a good answer
3. Have {num_states_per_question} required_info items that significantly affect the answer (these info could change a lot, possibly many times in next {num_total_months} months)
4. Cover both user-specific and general life topics

For each question, specify the required_info with:
- **info_type**: A specific type of information needed (e.g., experience_level, budget, team_size)
- **info_choices**: {num_choices_per_state} mutually exclusive choices that would lead to different advice, the choices should be specific and cover potential variantions in next {num_total_months} months

**Important Guidelines:**
- Make questions natural and conversational, also coherent with the user's long-term traits reflected in the profile
- Avoid info_types that are changing too frequently or too static
- Avoid info_types irrelevant to the user's personal situation (that can be easily inferred without asking)
- Ensure info_choices are comprehensive, mutually exclusive, and unambiguous (can be clearly distinguished with indirect context or relevant daily dialogue)
- Avoid info_choices that are too specific to a single moment in time
- Focus on actionable advice scenarios
- Vary the scope and perspective of questions

Generate all content in {prompt_lang}. Field names must remain in English.
Return as JSON object with "questions" as the key.

Example format:
{example_format}"""

    response = call_llm([{"role": "user", "content": detailed_prompt}], llm_config, json=True)
    
    # Handle response parsing
    try:
        questions = json.loads(response)["questions"]
        return questions
    except Exception as e:
        logger.error(f"Failed to parse questions response: {e}\nRaw response: {response}")
        return []


def refine_state_schema(llm_config, user_profile, questions, language="en"):
    """
    Refines the user profile schema based on the sampled questions.
    
    Args:
        llm_config (dict): Configuration for the LLM, including model, temperature, max tokens, etc.
        user_profile (str): user profile as a formatted string
        questions (list[dict]): List of question dictionaries with required_info
            - question (str): The question text
            - required_info (list[dict]): List of required info items
                - info_type (str): Specific type of information needed
                - info_choices (list[str]): Choices for this info type
        language (str): Language to generate schema in (default: "en")
        
    Returns:
        dict: Refined persona schema with exclusive info types mapped to original info types.
            - key (str): new exclusive and unambiguous info type
            - value (list[str]): original info types that map to this key
    """
    # Set language-specific prompts and examples
    if language.lower() in ["chinese", "zh", "中文"]:
        prompt_lang = "Chinese (中文)"
        example_format = """
{
    "职业经验年数": ["当前经验水平", "经验级别年数"],
    "团队管理规模": ["团队规模"]
}"""
        guidelines_text = "使用清晰、描述性的中文名称来命名精炼的信息类型"
    else:
        prompt_lang = "English"
        example_format = """
{
    "professional_experience_years": ["current_experience_level", "experience_level_years"],
    "team_management_size": ["team_size"]
}"""
        guidelines_text = "Use clear, descriptive names for refined info types"
    
    # Build the prompt for LLM to refine schema
    detailed_prompt = f"""You are a helpful assistant that refines persona schemas by making info types unambiguous and resolving conflicts.

Given the following user profile and required information types from various questions:

Initial User Profile:\n{user_profile}

Required Information Types:
{json.dumps(questions, ensure_ascii=False, indent=2)}

Your task is to:
1. **Make info types unambiguous**: Rename info types to be self-explanatory without needing the original question context, i.e., add necessary context from the questions
2. **Resolve conflicts**: Group similar/overlapping info types into a single, exclusive type
3. **Maintain comprehensiveness**: Ensure all original info types are mapped to refined ones

Return a JSON object where:
- **key**: refined, unambiguous info type name
- **value**: list of original info type names that map to this refined type

Generate all content in {prompt_lang}.

Example format:
{example_format}

**Guidelines:**
- {guidelines_text}
- Ensure new info types are mutually exclusive
- Consolidate similar concepts (e.g., "team size" and "subordinate count" into a single "team_management_size")
- Maintain the language style consistent with the original content"""
    response = call_llm([{"role": "user", "content": detailed_prompt}], llm_config, json=True)
    
    # Handle response parsing
    try:
        refined_schema = json.loads(response)
        # ensure all info_types in questions are covered in the refined schema
        orig2new = {}
        for new_type, orig_types in refined_schema.items():
            for orig_type in orig_types:
                orig2new[orig_type] = new_type
        for question in questions:
            for info in question["required_info"]:
                info_type = info["info_type"]
                assert info_type in orig2new, f"Info type '{info_type}' not found in refined schema"
        return refined_schema
    except Exception as e:
        logger.error(f"Failed to parse refined schema response: {e}\nRaw response: {response}")
        return {}


def fix_schema_inconsistencies(
    llm_config, start_date, user_profile, num_total_months,
    num_choices_per_state, questions, refined_schema, language="en"
):
    """
    Fix inconsistencies in the user profile based on the refined schema.
    
    Args:
        llm_config (dict): Configuration for the LLM, including model, temperature, max tokens, etc.
        start_date (str): Start date in format "YYYY-MM-DD"
        user_profile (str): User profile as a formatted string
        num_total_months (int): Total months to consider for user development
        num_choices_per_state (int): Number of choices per required_info item
        questions (list[dict]): List of original question dictionaries with required_info
            - question (str): The question text
            - required_info (list[dict]): List of required info items
                - info_type (str): Specific type of information needed
                - info_choices (list[str]): Choices for this info type
        refined_schema (dict): Refined persona schema with exclusive info types
        language (str): Language to generate fixes in (default: "en")
        
    Returns:
        dict: Updated questions with updated info_types and choices based on the refined schema.
            - question (str): The question text
            - required_info (list[dict]): List of required info items
                - info_type (str): Updated specific type of information needed
                - info_choices (list[str]): Updated choices for this info type
        dict: Updated state schema with new exclusive info types and their choices.
            - key (str): new exclusive and unambiguous info type
            - value (list[str]): choices for this info type
    """
    # update questions based on refined schema and check possible inconsistencies
    origtype2choices = {}
    for question in questions:
        q = question["question"]
        for info in question["required_info"]:
            info_type = info["info_type"]
            if info_type not in origtype2choices:
                origtype2choices[info_type] = []
            origtype2choices[info_type].append([q, info["info_choices"]])
    
    state_schema, orig2newtype = {}, {}
    conflict_groups = {}
    for new_type, orig_types in refined_schema.items():
        choices = []
        for orig_type in orig_types:
            choices.extend(origtype2choices[orig_type])
            orig2newtype[orig_type] = new_type
        if len(choices) == 1:
            state_schema[new_type] = choices[0][1]
        else:
            conflict_groups[new_type] = [
                {"question": item[0], "choices": item[1]} for item in choices
            ]
    
    # Set language-specific prompts
    if language.lower() in ["chinese", "zh", "中文"]:
        prompt_lang = "Chinese (中文)"
        example_format = """
{
    "职业经验年数": ["初级(0-2年)", "中级(3-5年)", "高级(6-10年)", "专家级(10年以上)"],
    "团队管理规模": ["无管理职责", "小团队(2-5人)", "中等团队(6-15人)", "大团队(15人以上)"]
}"""
        guidelines_text = f"创建涵盖各种情况（问题）的统一选项，确保选项在未来{num_total_months}个月可能发生选项间多次合理变化"
    else:
        prompt_lang = "English"
        example_format = """
{
    "professional_experience_years": ["junior_0_2_years", "mid_level_3_5_years", "senior_6_10_years", "expert_10_plus_years"],
    "team_management_size": ["no_management", "small_team_2_5", "medium_team_6_15", "large_team_15_plus"]
}"""
        guidelines_text = f"Create unified choices that cover all scenarios (questions) and allow for multiple reasonable changes in next {num_total_months} months"
    
    # Resolve all conflicts in a single LLM call
    newtype2choices = {}
    
    if conflict_groups:        
        detailed_prompt = f"""You are a helpful assistant that resolves conflicts in persona schema by creating unified choice sets.

Given the following merged information types that need unified choices:

User Profile (on current date {start_date}):\n{user_profile}

Conflicting Information Types and their contexts:
{json.dumps(conflict_groups, ensure_ascii=False, indent=2)}

Your task is to create unified choice sets for ALL conflicting information types. For each type, create choices that:
1. **Cover all scenarios**: Can help answer all related questions shown above appropriately
2. **Mutually exclusive**: Each choice is distinct and non-overlapping
3. **Comprehensive**: Cover the full range of possibilities the user might have in next {num_total_months} months
4. **Progressive**: Allow for natural progression/changes over time
5. **Personalized**: Enable different advice for different choices

Requirements:
- Create {num_choices_per_state} choices for each information type that work for ALL questions listed for that type
- Ensure choices allow for multiple reasonable changes in next {num_total_months} months
- Make choices specific enough to enable personalized advice
- {guidelines_text}

Generate all content in {prompt_lang}.
Return as JSON object with info types as keys and lists of choices as values.

Example format:
{example_format}"""

        response = call_llm([{"role": "user", "content": detailed_prompt}], llm_config, json=True)
        
        try:
            newtype2choices = json.loads(response)
        except Exception as e:
            logger.error(f"Failed to parse conflict resolution response: {e}\nRaw response: {response}")
            raise ValueError("Failed to parse conflict resolution response.")

    # Validate that all conflict types have been resolved
    for new_type, q_choices in conflict_groups.items():
        if new_type not in newtype2choices:
            logger.error(f"Failed to get new choices for type '{new_type}' from batch resolution")
            raise ValueError(f"Failed to resolve conflict for type '{new_type}'.")        
        state_schema[new_type] = newtype2choices[new_type]
    
    # update questions with new choices
    updated_questions = []
    for question in questions:
        new_question = {
            "question": question["question"],
            "required_info": []
        }
        for info in question["required_info"]:
            info_type = info["info_type"]
            new_type = orig2newtype[info_type]
            new_question["required_info"].append({
                "info_type": new_type,
                "info_choices": state_schema[new_type]
            })
        updated_questions.append(new_question)
    
    return updated_questions, state_schema


if __name__ == "__main__":
    pass
