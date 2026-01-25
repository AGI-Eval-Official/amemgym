import json
import pandas as pd
from amemgym.utils import call_llm
from loguru import logger


def check_nemotron_occupation_dist(file_path="./data/nemotron-personas.parquet"):
    """Check distinct values and counts for the occupation column in the personas dataset.
    
    Args:
        file_path (str): Path to the parquet file containing personas.
    """
    df = pd.read_parquet(file_path, engine="pyarrow")
    logger.info(f"Loaded {len(df)} personas from {file_path}")

    occupation_counts = df['occupation'].value_counts()
    logger.info(f"Found {len(occupation_counts)} distinct occupations")
    print("\nTop 20 Occupation distribution:")
    print("=" * 50)
    for occupation, count in occupation_counts.head(20).items():
        print(f"{occupation}: {count}({count / len(df) * 100:.2f}%)")
    print("=" * 50)
    print(f"Total personas: {occupation_counts.sum()}")


def sample_nemotron_persona(
    file_path="./data/personas/nemotron.parquet", num_samples=20,
    exclude_ids=None, random_state=None
):
    """Sample personas from the given parquet file.
    Data source: https://huggingface.co/datasets/nvidia/Nemotron-Personas
    
    Args:
        file_path (str): Path to the parquet file containing personas.
        num_samples (int): Number of personas to sample.
        exclude_ids (list): List of UUIDs to exclude from sampling.
        random_state (int, optional): Random seed for reproducibility.
        
    Returns:
        list[dict]: List of sampled persona dictionaries.
    """
    df = pd.read_parquet(file_path, engine="pyarrow")
    logger.info(f"Loaded {len(df)} personas from {file_path}")
    
    # Exclude personas with matching UUIDs if exclude_ids is provided
    if exclude_ids is not None and len(exclude_ids) > 0:
        if 'uuid' in df.columns:
            df = df[~df['uuid'].isin(exclude_ids)]
            logger.info(f"After excluding {len(exclude_ids)} UUIDs, {len(df)} personas remain")
    
    # Ensure we don't sample more than available
    actual_samples = min(num_samples, len(df))
    if actual_samples < num_samples:
        logger.warning(f"Requested {num_samples} samples but only {actual_samples} available")
    
    # Sample the specified number of personas
    sampled_df = df.sample(n=actual_samples, random_state=random_state)
    
    # Convert to list of dictionaries using column names as keys
    return sampled_df.to_dict(orient="records")


def format_nemotron_persona(persona, llm_config):
    """
    Format a Nemotron persona dictionary into a well-structured user profile
    suitable for role-play LLMs.
    
    Args:
        persona (dict): Nemotron persona dictionary containing various persona fields
        llm_config (dict): Configuration for the LLM to use
        
    Returns:
        dict: Key information preserved in a dictionary with 'uuid', 'name', 'age', 'gender',
        'marital_status', 'education_level', 'occupation', 'hobbies_and_interests',
        'skills_and_expertise', and 'complementary_info' keys.
    """
    # get basic profile
    education_level = persona["education_level"]
    bachelors_field = persona["bachelors_field"]
    if bachelors_field is not None:
        education_level += f" ({bachelors_field.replace('_', ' ')})"

    hobbies_and_interests = ", ".join(eval(persona["hobbies_and_interests_list"]))
    if not hobbies_and_interests:
        hobbies_and_interests = "N/A"
    
    skills_and_expertise = ", ".join(eval(persona["skills_and_expertise_list"]))
    if not skills_and_expertise:
        skills_and_expertise = "N/A"


    basic_profile = {
        "age": persona["age"],
        "gender": persona["sex"],
        "marital_status": persona["marital_status"],
        "education_level": education_level,
        "occupation": persona["occupation"],
        "hobbies_and_interests": hobbies_and_interests,
        "skills_and_expertise": skills_and_expertise
    }

    basic_profile_str = "\n".join([
        f"{k}: {basic_profile[k]}" for k in ['age', 'gender', 'marital_status', 'education_level', 'occupation', 'hobbies_and_interests', 'skills_and_expertise']
    ])

    # get complementary info
    complementary_info = {k: persona[k] for k in ["persona", "professional_persona", "sports_persona", "arts_persona", "travel_persona", "culinary_persona", "career_goals_and_ambitions", "skills_and_expertise", "hobbies_and_interests"]}
    
    # call llm to format
    completion_prompt = (
        f"You have two tasks:\n"
        f"1. Extract the full name from the complementary information below\n"
        f"2. Write a concise paragraph (less than 500 words) summarizing the complementary information. Include only details that cannot be derived from the basic profile.\n\n"
        f"Basic Profile:\n{basic_profile_str}\n\n"
        f"Complementary Information:\n{json.dumps(complementary_info, indent=2, ensure_ascii=False)}\n\n"
        f"Keep the summary professional and suitable for role-play scenarios. Make it informative but concise. Respond in JSON format with `name` and `profile` as keys."
    )
    ret = call_llm(
        messages=[
            {"role": "system", "content": "You are a helpful assistant that extracts names and summarizes user profiles."},
            {"role": "user", "content": completion_prompt}
        ],
        llm_config=llm_config,
        json=True
    )
    ret = json.loads(ret)
    name = ret.get("name", "N/A")

    user_profile = {"uuid": persona["uuid"], "name": name} | basic_profile | {"complementary_info": ret.get("profile", "").strip()}
    formatted_str = "\n".join([
        f"{k}: {user_profile[k]}" for k in ['name', 'age', 'gender', 'marital_status', 'education_level', 'occupation', 'hobbies_and_interests', 'skills_and_expertise', 'complementary_info']
    ])
    user_profile["formatted_str"] = formatted_str

    return user_profile
