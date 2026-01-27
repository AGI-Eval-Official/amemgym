"""Evaluation module for on-policy agent performance assessment.

This module provides functionality to evaluate agents using environment data
and generated conversations, measuring their ability to track and respond to
user preferences over time.
"""

import os
from dotenv import load_dotenv
import random
import shutil
from argparse import ArgumentParser

import numpy as np
from loguru import logger
from tqdm import tqdm

from amemgym.assistants import create_agent, Mem0Agent
from amemgym.env.sample_interactions import sample_session_given_query
from amemgym.utils import save_json, load_json, parse_json, count_tokens, setup_logger
from .metric import METRICS, state_similarity


random.seed(42)  # For reproducibility


# Template for overall evaluation - asks agent to select best answer from multiple choices
OVERALL_PROMPT = """\
{query}

Please select the most suitable answer for my current situation from the following options:
(considering my current relevant preferences and state information)

{choices}

Express your choice with a number and output in the following JSON format:
```json
{{
    "answer": int
}}
```
Only keep the JSON format output, do not include any other content.
"""


def evaluate_item(item, agent, output_dir, env_config, off_policy=False) -> None:
    """Perform overall evaluation of an agent on a single evaluation item.

    This function evaluates an agent's ability to answer questions based on tracked preferences
    by asking the agent to select the best answer from a set of choices given the question.
    The evaluation is performed across multiple time periods, with the agent's state being
    updated through on-policy interactions before each evaluation.

    Args:
        item: Evaluation item containing periods, questions, and user profile information.
        agent: The agent instance to evaluate.
        output_dir: Directory path where evaluation results will be saved.
        env_config: Environment configuration containing LLM settings and round limits.

    Note:
        Results are saved incrementally to allow for resuming interrupted evaluations.
        Agent states are cached between periods to avoid redundant computations.
    """
    num_questions, num_periods = len(item["qas"]), len(item["periods"])
    results_path = os.path.join(output_dir, "overall_results.json")
    
    if not os.path.exists(results_path):
        results = [[None for _ in range(num_questions)] for _ in range(num_periods)]
    else:
        results = load_json(results_path)
    
    for pi, period in enumerate(tqdm(item["periods"], desc="Evaluating overall questions", ncols=80)):
        agent_state_dir = os.path.join(output_dir, f"agent_states/period_{pi:02d}")
        interactions_path = os.path.join(output_dir, f"interactions/period_{pi:02d}.json")
        max_rounds = env_config["num_rounds_init"] if pi == 0 else env_config["num_rounds_update"]
        
        if os.path.exists(agent_state_dir):
            agent.load_state(agent_state_dir)
        else:
            sessions = period["sessions"]  # load environment data
            if off_policy:
                interactions = load_json(interactions_path)
                for session, session_msgs in tqdm(zip(sessions, interactions), desc="Off-policy Interactions", leave=False):
                    query_with_time = f"[Current Time: {session['session_time']}]\n" + session["query"]
                    assert query_with_time == session_msgs[0]["content"]
                    assert len(session_msgs) == 2 * max_rounds, f"Loaded {len(session_msgs)} messages, expected {2 * max_rounds}"
                    for i in tqdm(range(0, 2 * max_rounds, 2), ncols=80, leave=False):  # load external messages
                        # truncate messages for Mem0Agent (RAG)
                        if isinstance(agent, Mem0Agent) and not agent.config["agent_config"]["enable_llm_mem_policy"]:
                            num_tokens = count_tokens(session_msgs[i: i+2])
                            if num_tokens >= 8192:
                                logger.warning("Truncating messages for proper embeddings...")
                                limit = len(session_msgs[i+1]["content"]) - 10 * (num_tokens - 8192)
                                session_msgs[i+1]["content"] = session_msgs[i+1]["content"][:limit] + "..."
                        agent.add_msgs(session_msgs[i:i+2])
            else:
                interactions = []
                for session in tqdm(sessions, ncols=100, leave=False, desc="On-policy interactions"):
                    query_with_time = f"[Current Time: {session['session_time']}]\n" + session["query"]
                    session_msgs = sample_session_given_query(
                        env_config["llm_config_low_temp"], query_with_time, agent, item["start_time"],
                        item["user_profile"], period["period_end"],
                        item["state_schema"], hist=None, max_rounds=max_rounds
                    )
                    interactions.append(session_msgs)
                # save interactions
                save_json(interactions_path, interactions)
            agent.save_state(agent_state_dir)

        for qi, qa in enumerate(tqdm(item["qas"], desc="Asking questions", ncols=80, leave=False)):
            if results[pi][qi] is not None:
                continue
            
            choices_text = '\n'.join(['{}: {}'.format(
                i + 1, choice['answer']) for i, choice in enumerate(qa["answer_choices"])])
            query = OVERALL_PROMPT.format(query=qa["query"], choices=choices_text)
            response, usage_statistics = agent.answer_question(query)
            
            try:
                response_answer = parse_json(response)["answer"]
                response_choice = qa["answer_choices"][response_answer - 1]
                json_error = False
            except Exception as e:
                response_answer = random.randint(1, len(qa["answer_choices"]))  # Fallback to random choice
                response_choice = qa["answer_choices"][response_answer - 1]
                json_error = True
                
            # Retrieve golden answer
            golden_state = [period["state"][info_type] for info_type in qa["required_info"]]
            ci = 0
            for ci, choice in enumerate(qa["answer_choices"]):
                if choice["state"] == golden_state:
                    break
            choice = qa["answer_choices"][ci]

            scores = {}
            for metric in METRICS:
                scores[metric] = state_similarity(choice["state"], response_choice["state"], metric)

            result = {
                "query": qa["query"],
                "answer": ci,
                "answer_state": choice["state"],
                "answer_choice": choice["answer"],
                "raw_response": response,
                "response": response_answer,
                "response_state": response_choice["state"],
                "response_choice": response_choice["answer"],
                "json_error": json_error,
                "llm_usage_statistics": usage_statistics,
                "scores": scores,
            }
            results[pi][qi] = result
            save_json(results_path, results)
    
    metric_path = os.path.join(output_dir, "overall_metrics.json")
    if os.path.exists(metric_path):
        return
        
    overall_metrics = {metric: np.zeros((num_periods, num_questions)) for metric in METRICS}
    num_json_errors = 0
    
    for pi in range(num_periods):
        for qi in range(num_questions):
            result = results[pi][qi]
            if result["json_error"]:
                num_json_errors += 1
            for metric in METRICS:
                overall_metrics[metric][pi, qi] = result["scores"][metric]
                
    for metric, scores in overall_metrics.items():
        logger.info(f"Overall metric: {metric} {scores.mean()} {scores.mean(axis=1)}")
        overall_metrics[metric] = scores.tolist()
        
    save_json(metric_path, overall_metrics)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Evaluate agents using environment data and user simulation.")
    parser.add_argument("--env_data", type=str, default="data/v1.base/data.json",
                        help="Environment data file")
    parser.add_argument("--env_config", type=str, default="configs/env/v1.base.json",
                        help="Environment configuration file")
    parser.add_argument("--agent_config", type=str, required=True,
                        help="Agent configuration file")
    parser.add_argument("--output_dir", type=str, default="eval-output/v1.base/native",
                        help="Output directory for evaluation results")
    parser.add_argument("--off_policy_dir", type=str, default="",
                        help="Directory to load off-policy interactions.")
    parser.add_argument("--reset", action="store_true",
                        help="Overwrite output")
    args = parser.parse_args()
    load_dotenv()

    # Load configurations
    agent_config = load_json(args.agent_config)
    env_config = load_json(args.env_config)
    for key in ["llm_config_low_temp", "llm_config_high_temp"]:
        env_config[key] |= {
            "base_url": env_config[key].get("base_url") or os.environ.get("OPENAI_BASE_URL"),
            "api_key": env_config[key].get("api_key") or os.environ.get("OPENAI_API_KEY"),
            "source": "env:interaction"
        }  # fill in from environment variables if not provided

    output_dir = os.path.join(args.output_dir, agent_config["name"])
    if args.reset and os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    shutil.copy(args.agent_config, output_dir)  # save agent config

    data = load_json(args.env_data)
    for item in data:
        item_dir = os.path.join(output_dir, item["id"])
        os.makedirs(item_dir, exist_ok=True)
        if agent_config["type"] == "awi-hack":
            agent = create_agent(agent_config, output_dir=item_dir, item=item)
        else:
            agent = create_agent(agent_config, output_dir=item_dir)
        # save agent config with filled environment variables
        save_json(os.path.join(item_dir, "agent_config.json"), agent_config)
        if args.off_policy_dir:
            off_policy = True
            shutil.copytree(
                os.path.join(args.off_policy_dir, item["id"], "interactions"),
                os.path.join(item_dir, "interactions"),
                dirs_exist_ok=True
            )
        else:
            off_policy = False
            os.makedirs(os.path.join(item_dir, "interactions"), exist_ok=True)
        log_dir = os.path.join(item_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        setup_logger(os.path.join(log_dir, "evaluate.log"))
        logger.info(item["id"])
        evaluate_item(item, agent, item_dir, env_config, off_policy)
