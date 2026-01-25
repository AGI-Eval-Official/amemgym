from argparse import ArgumentParser
import os
import random
import sys
import shutil
from tqdm import tqdm
from loguru import logger
from dotenv import load_dotenv

from .sample_user_profile import sample_nemotron_persona, format_nemotron_persona
from .sample_state_schema import refine_state_schema, fix_schema_inconsistencies, sample_user_questions
from .sample_state_evolution import sample_initial_state, sample_state_updates, elaborate_state_updates
from .sample_evaluation_qa import sample_personalized_answers, get_state_variants, check_personalized_answer, refine_personalized_answer
from .sample_session_query import sample_init_queries, sample_update_queries, check_query_state_exposure, refine_query
from amemgym.utils import (
    save_json, load_json, load_date, date_plus_months,
    sample_session_timestamps, setup_logger
)


def sample_env_data_given_profile(user_dir, user_profile, llm_config_high, llm_config_low, config):
    user_profile_path = os.path.join(user_dir, "user_profile.json")
    if not os.path.exists(user_profile_path):
        save_json(user_profile_path, user_profile)

    start_date = load_date(config["start_date"])
    num_total_months = config["num_periods"] * config["num_months_per_period"]

    # sample questions and state dependencies given user profile
    questions_path = os.path.join(user_dir, "questions.json")
    schema_path = os.path.join(user_dir, "schema.json")

    if not os.path.exists(questions_path) or not os.path.exists(schema_path):
        questions = sample_user_questions(
            llm_config_high, config["start_date"], user_profile["formatted_str"],
            config["num_questions"], config["num_states_per_question"], config["num_choices_per_state"],
            num_total_months, config["lang"]
        )  # sample questions given user profile
        refined_schema = refine_state_schema(
            llm_config_low, user_profile["formatted_str"], questions, config["lang"]
        )  # refine state schema by grouping similar info types
        questions, schema = fix_schema_inconsistencies(
            llm_config_low, config["start_date"], user_profile["formatted_str"],
            num_total_months, config["num_choices_per_state"], questions, refined_schema, config["lang"]
        )  # update data to ensure consistency
        
        save_json(questions_path, questions)
        save_json(schema_path, schema)
        logger.info(f"Sampled and saved {len(questions)} questions and schema to {user_dir}")
    else:
        questions = load_json(questions_path)
        schema = load_json(schema_path)
        logger.info(f"Loaded existing {len(questions)} questions and schema from {user_dir}")
    
    # sample user states evolution
    transition_path = os.path.join(user_dir, "state_transition.json")
    if not os.path.exists(transition_path):
        logger.info(f"Sampling state transitions for user {user_profile['uuid']}")
        prev_state = sample_initial_state(
            llm_config_low, config["start_date"], user_profile["formatted_str"],
            num_total_months, schema
        )
        current_date = load_date(config["start_date"])
        states = [prev_state]
        updates = [None]
        update_cnts = {k: 0 for k in prev_state}
        for pi in tqdm(range(config["num_periods"]), desc="Sampling state updates", ncols=100, leave=False):
            end_date = date_plus_months(current_date, config["num_months_per_period"])
            update = sample_state_updates(
                llm_config_high, config["start_date"], user_profile["formatted_str"],
                config["num_months_per_period"], current_date, end_date,
                config["num_changes_per_period"], config["max_changes_per_state"],
                schema, prev_state, updates, update_cnts,
                remaining_steps=config["num_periods"]-pi,
                total_steps=config["num_periods"]
            )
            update["old"] = {k: prev_state[k] for k in update["updated"]}
            explanation = elaborate_state_updates(
                llm_config_low, config["start_date"], user_profile["formatted_str"],
                prev_state, update, schema
            )
            update["events"] = explanation
            # update loop variables
            new_state = prev_state | update["updated"]
            states.append(new_state)
            updates.append(update)
            prev_state = new_state
            current_date = end_date
            for var in update["updated"].keys():
                update_cnts[var] += 1

        state_transition = {"states": states, "updates": updates, "update_cnts": update_cnts}
        save_json(transition_path, state_transition)
    else:
        state_transition = load_json(transition_path)
        states, updates = state_transition["states"], state_transition["updates"]
        logger.info(f"Loaded existing state transitions from {transition_path}")

    # sample personalized answers & reflection
    answer_path = os.path.join(user_dir, "personalized_answers.json")
    if not os.path.exists(answer_path):
        logger.info(f"Sampling personalized answers for user {user_profile['uuid']}")
        state_variants = get_state_variants(
            states, questions, min_variants=config["min_state_variants"]
        )
        all_answers = []
        for question in questions:
            q = question["question"]
            variants = state_variants[q]
            answers = sample_personalized_answers(
                llm_config_low, question, variants
            )
            for ai, answer in enumerate(answers):
                answer_text = answer["answer"]
                check_ret = check_personalized_answer(
                    llm_config_low, question, answer_text, variants, ai
                )
                retry_cnt = 20
                while not check_ret and retry_cnt > 0:
                    logger.warning(f"Answer validation failed for question '{question['question']}' and variant {answer['variant']}.\nCurrent Answer:{answer_text}\nRetrying...")
                    answer_text = refine_personalized_answer(
                        llm_config_low, question, answer_text, variants, ai
                    )
                    logger.info(f"After refinement: {answer_text}")
                    check_ret = check_personalized_answer(
                        llm_config_low, question, answer_text, variants, ai
                    )
                    logger.info(f"After refinement, validation result: {check_ret}")
                    retry_cnt -= 1
                if not check_ret:
                    logger.error(f"Failed to generate a valid answer for question '{question['question']}' and variant {answer['variant']} after 10 retries. Exiting.")
                    exit(1)  # fail to generate valid answer after retries
                answer["answer"] = answer_text
            all_answers.append(answers)
        save_json(answer_path, all_answers)

    # sample session start user queries & reflection
    session_path = os.path.join(user_dir, "sessions.json")
    if os.path.exists(session_path):
        return
    sessions = []
    init_queries = sample_init_queries(
        llm_config_low, config["start_date"], user_profile["formatted_str"],
        schema, states[0]
    )
    for query in init_queries:
        query_text = query["query"]
        query_exposed_states = query["exposed_states"]
        check_ret = check_query_state_exposure(
            llm_config_low, query_text, query_exposed_states, schema
        )
        retry_cnt = 20
        while not check_ret and retry_cnt > 0:
            logger.warning(f"Query validation failed for initial query '{query_text}'. Retrying...")
            query_text = refine_query(
                llm_config_low, query_text, query_exposed_states, schema
            )
            check_ret = check_query_state_exposure(
                llm_config_low, query_text, query_exposed_states, schema
            )
            retry_cnt -= 1
        if not check_ret:
            logger.error(f"Failed to generate a valid initial query '{query_text}' after retries. Exiting.")
            exit(1)  # fail to generate valid query after retries
        query["query"] = query_text
    timestamps = sample_session_timestamps(None, start_date, len(init_queries))
    for query, ts in zip(init_queries, timestamps):
        query["timestamp"] = ts.strftime("%Y-%m-%d %H:%M:%S")
    sessions.append(init_queries)

    assert updates[0] is None
    for pi in range(config["num_periods"]):
        update_queries = sample_update_queries(
            llm_config_low, config["start_date"], user_profile["formatted_str"],
            schema, updates[pi+1]
        )
        for query in update_queries:
            query_text = query["query"]
            query_exposed_states = query["exposed_states"]
            check_ret = check_query_state_exposure(
                llm_config_low, query_text, query_exposed_states, schema
            )
            retry_cnt = 20
            while not check_ret and retry_cnt > 0:
                logger.warning(f"Query validation failed for update query '{query_text}'. Retrying...")
                query_text = refine_query(
                    llm_config_low, query_text, query_exposed_states, schema
                )
                check_ret = check_query_state_exposure(
                    llm_config_low, query_text, query_exposed_states, schema
                )
                retry_cnt -= 1
            if not check_ret:
                logger.error(f"Failed to generate a valid update query '{query_text}' after retries. Exiting.")
                exit(1)  # fail to generate valid query after retries
            query["query"] = query_text
        start_date = load_date(updates[pi+1]["period_start"])
        end_date = load_date(updates[pi+1]["period_end"])
        timestamps = sample_session_timestamps(start_date, end_date, len(update_queries))
        for query, ts in zip(update_queries, timestamps):
            query["timestamp"] = ts.strftime("%Y-%m-%d %H:%M:%S")
        sessions.append(update_queries)
    save_json(session_path, sessions)


def convert_raw_data(data_dir):
    """Convert a single user's data to the target format"""
    # Load all data files
    schema_data = load_json(os.path.join(data_dir, "schema.json"))
    sessions_data = load_json(os.path.join(data_dir, "sessions.json"))
    state_data = load_json(os.path.join(data_dir, "state_transition.json"))
    questions_data = load_json(os.path.join(data_dir, "questions.json"))
    answers_data = load_json(os.path.join(data_dir, "personalized_answers.json"))
    user_profile = load_json(os.path.join(data_dir, "user_profile.json"))
    
    period_data = []
    for sessions, state, updates in zip(sessions_data, state_data["states"], state_data["updates"]):
        if updates is None:
            state_updates = {k: {"old": None, "new": state[k]} for k in state}
            period_start, period_end = None, state_data["updates"][1]["period_start"]
            period_summary = None
            events = [None] * len(sessions)
            start_time = period_end
        else:
            state_updates = {k: {"old": updates["old"][k], "new": updates["updated"][k]} for k in updates["updated"]}
            period_start, period_end = updates["period_start"], updates["period_end"]
            period_summary = updates["period_summary"]
            events = [e["event"] for e in updates["events"]]
        
        period_sessions = []

        for session, event in zip(sessions, events):
            period_session = {
                "event": event,
                "exposed_states": session["exposed_states"],
                "query": session["query"],
                "messages": session.get("messages", []),
            }
            period_sessions.append(period_session)
            period_session["session_time"] = session["timestamp"]
        period = {
            "period_start": period_start,
            "period_end": period_end,
            "period_summary": period_summary,
            "sessions": period_sessions,
            "state": state,
            "updates": state_updates,
            "update_cnts": state_data["update_cnts"]
        }
        period_data.append(period)

    qas = []
    for question, answers in zip(questions_data, answers_data):
        question_data = {
            "query": question["question"],
            "required_info": [info["info_type"] for info in question["required_info"]],
            "answer_choices": []
        }
        exp_states = set()
        for state in state_data["states"]:
            variant = tuple(state[info["info_type"]] for info in question["required_info"])
            exp_states.add(variant)
        
        for answer in answers:
            answer_choice = {
                "state": answer["variant"],
                "answer": answer["answer"],
                "type": "experience" if tuple(answer["variant"]) in exp_states else "random"
            }
            question_data["answer_choices"].append(answer_choice)
        qas.append(question_data)
    
    # Build final structure
    result = {
        "id": user_profile["uuid"],
        "start_time": start_time,
        "user_profile": user_profile,
        "state_schema": schema_data,
        "periods": period_data,
        "qas": qas
    }
    return result


def main():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/v1/base")
    parser.add_argument("--config_path", type=str, default="./configs/env/v1.base.json")
    parser.add_argument("--persona_path", type=str, default="./data/personas/nemotron.parquet")
    parser.add_argument("--reset", action="store_true", help="Whether to clean partially sampled data")
    args = parser.parse_args()
    load_dotenv()
    config = load_json(args.config_path)
    # set base url and api key from environment variable
    llm_config_high = config["llm_config_high_temp"] | {"base_url": os.getenv("LLM_BASE_URL"), "api_key": os.getenv("LLM_API_KEY")}
    llm_config_low = config["llm_config_low_temp"] | {"base_url": os.getenv("LLM_BASE_URL"), "api_key": os.getenv("LLM_API_KEY")}

    raw_data_dir = os.path.join(args.data_dir, "raw")
    if args.reset and os.path.exists(raw_data_dir):
        shutil.rmtree(raw_data_dir)
        logger.info(f"Removed existing data directory {raw_data_dir}")
    os.makedirs(raw_data_dir, exist_ok=True)

    # sample user profiles
    user_profile_path = os.path.join(raw_data_dir, "user_profiles.json")
    if not os.path.exists(user_profile_path):
        personas = sample_nemotron_persona(
            file_path=args.persona_path,
            num_samples=config["num_user_profiles"],
            exclude_ids=config["exclude_ids"],
            random_state=config["seed"]
        )
        personas = [
            format_nemotron_persona(persona, llm_config=llm_config_low)
            for persona in tqdm(personas, ncols=100, desc="Formatting user profiles", leave=False)
        ]
        save_json(user_profile_path, personas)
        logger.info(f"Sampled and saved {len(personas)} user profiles to {user_profile_path}")
    user_profiles = load_json(user_profile_path)
    
    random.seed(config["seed"])
    all_data = []
    for ui, user_profile in enumerate(user_profiles):
        try:
            logger.info(f"Processing user {user_profile['uuid']} ({ui+1}/{len(user_profiles)})")
            user_dir = os.path.join(raw_data_dir, user_profile["uuid"])
            os.makedirs(user_dir, exist_ok=True)
            setup_logger(os.path.join(user_dir, "data.log"))
            sample_env_data_given_profile(
                user_dir, user_profile, llm_config_high, llm_config_low, config
            )
            item = convert_raw_data(user_dir)
            all_data.append(item)
            save_json(os.path.join(user_dir, "item.json"), item)
        except:
            continue
    save_json(os.path.join(args.data_dir, "data.json"), all_data)


if __name__ == "__main__":
    main()
