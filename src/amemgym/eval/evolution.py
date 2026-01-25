"""Evaluation module for on-policy agent performance with evolution support.

This module provides functionality to evaluate evolvable agents, including
the ability to trigger prompt evolution based on evaluation feedback after
each period.
"""

import os
import random
import shutil
import string
from argparse import ArgumentParser

import numpy as np
from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

from amemgym.assistants import create_agent
from amemgym.env.sample_interactions import sample_session_given_query
from amemgym.utils import save_json, load_json, parse_json, setup_logger
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


def format_question_choices(question: str, choices: list, with_choices: bool = True) -> str:
    """Format a question with optional multiple choice answers.

    Args:
        question: The question text.
        choices: List of answer choices.
        with_choices: Whether to include the choices in output.

    Returns:
        Formatted question string.
    """
    labels = list(string.ascii_uppercase)
    q = question.strip()
    lines = [f"Question: {q}"]
    if with_choices:
        for i, choice in enumerate(choices):
            label = labels[i] if i < len(labels) else str(i + 1)
            lines.append(f"({label}) {choice.strip()}")
    return ";\n".join(lines) + ";"


def collect_evolution_feedback(agent, results, item, period, period_idx):
    """Collect feedback from Q&A results for evolution.

    Args:
        agent: The evolvable agent.
        results: Results from the current period's Q&A evaluation.
        item: The evaluation item containing QAs and period information.
        period: The current period data.
        period_idx: Index of the current period.

    Returns:
        Feedback data structure for evolution.
    """
    labels = list(string.ascii_uppercase)
    feedback = []

    for result, qas in zip(results[period_idx], item['qas']):
        if result is None:
            continue

        question = qas['query']
        choices = [item['answer'] for item in qas['answer_choices']]
        ground_truth_choice_id = result['answer']
        response_choice_id = result['response']
        retrieved_memories = result['relevant_memories']

        feedback_info = {}
        feedback_types = agent.evolution_config.get("feedback_types", [])

        if "vanilla_question_only" in feedback_types:
            # Only provide memory + question text (no candidate choices & assistant's response)
            feedback_info.update({
                "question": format_question_choices(question, choices, with_choices=False),
                "retrieved_memories": retrieved_memories
            })
        elif "vanilla" in feedback_types:
            # Memory + question text with candidate choices & assistant's response
            # NOTE: response_choice_id is 1-indexed from LLM output, matching mem-env behavior
            feedback_info.update({
                "question": format_question_choices(question, choices, with_choices=True),
                "assistant_response": labels[response_choice_id],
                "retrieved_memories": retrieved_memories
            })

        if "with_answer" in feedback_types:
            feedback_info.update({
                "ground_truth": labels[ground_truth_choice_id]
            })

        feedback.append(feedback_info)

    if "with_exposed_states" in agent.evolution_config.get("feedback_types", []):
        # States exposed in this period's sessions
        exposed_states = {}
        for session in period['sessions']:
            exposed_states.update(session['exposed_states'])

        feedback = {
            "question_answer_history": feedback,
            "user_information_updates": exposed_states
        }

    return feedback


def evaluate_item_with_evolution(item, agent, output_dir, env_config) -> None:
    """Perform overall evaluation of an evolvable agent on a single evaluation item.

    This function evaluates an agent's ability to answer questions based on tracked preferences.
    After each period's evaluation, it triggers evolution based on Q&A feedback.

    Args:
        item: Evaluation item containing periods, questions, and user profile information.
        agent: The evolvable agent instance to evaluate.
        output_dir: Directory path where evaluation results will be saved.
        env_config: Environment configuration containing LLM settings and round limits.
    """
    num_questions, num_periods = len(item["qas"]), len(item["periods"])

    # Early exit if evaluation is already complete
    metric_path = os.path.join(output_dir, "overall_metrics.json")
    if os.path.exists(metric_path):
        logger.info(f"Overall evaluation already complete for {output_dir}. Skipping.")
        return

    results_path = os.path.join(output_dir, "overall_results.json")

    if not os.path.exists(results_path):
        results = [[None for _ in range(num_questions)] for _ in range(num_periods)]
    else:
        results = load_json(results_path)

    for pi, period in enumerate(tqdm(item["periods"], desc="Evaluating overall questions", ncols=80)):
        agent_state_dir = os.path.join(output_dir, f"agent_states/period_{pi:02d}")
        if os.path.exists(agent_state_dir):
            # Check if all questions for this period are answered
            period_complete = all(results[pi][qi] is not None for qi in range(num_questions))
            if period_complete:
                logger.info(f"Period {pi} already complete. Loading state and continuing.")
                agent.load_state(agent_state_dir)
                continue
            else:
                agent.load_state(agent_state_dir)
        else:
            sessions = period["sessions"]
            for session in tqdm(sessions, ncols=80, leave=False, desc="On-policy interactions"):
                query_with_time = f"[Current Time: {session['session_time']}]\n" + session["query"]
                max_rounds = env_config["num_rounds_init"] if pi == 0 else env_config["num_rounds_update"]

                # NOTE: amemgym's sample_session_given_query doesn't have exposed_states parameter
                # The exposed_states are stored in session data but not used in user followup generation
                _ = sample_session_given_query(
                    env_config["llm_config_low_temp"], query_with_time, agent, item["start_time"],
                    item["user_profile"], period["period_end"],
                    item["state_schema"], hist=None, max_rounds=max_rounds
                )

            agent.save_state(agent_state_dir)

        # Evaluate Q&A
        for qi, qa in enumerate(tqdm(item["qas"], desc="Asking questions", ncols=80, leave=False)):
            if results[pi][qi] is not None:
                continue
            choices_text = '\n'.join(['{}: {}'.format(
                i + 1, choice['answer']) for i, choice in enumerate(qa["answer_choices"])])
            query = OVERALL_PROMPT.format(
                query=qa["query"], choices=choices_text)

            # Handle different return types (evolvable agents return memories_str too)
            answer_result = agent.answer_question(query)
            if isinstance(answer_result, tuple) and len(answer_result) == 2:
                if isinstance(answer_result[1], tuple):
                    # Evolvable agent: (memories_str, (response, usage))
                    retrieved_memories, (response, usage_statistics) = answer_result
                else:
                    # Standard agent: (response, usage)
                    response, usage_statistics = answer_result
                    retrieved_memories = ""
            else:
                response, usage_statistics = answer_result, {}
                retrieved_memories = ""

            try:
                response_answer = parse_json(response)["answer"]
                response_choice = qa["answer_choices"][response_answer - 1]
                json_error = False
            except Exception:
                response_answer = random.randint(1, len(qa["answer_choices"]))
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
                "relevant_memories": retrieved_memories
            }
            results[pi][qi] = result
            save_json(results_path, results)

        # ========== EVOLUTION POINT ==========
        # Now we have fresh Q&A results and can make informed evolution decisions
        cadence = agent.evolution_config.get("cadence", "no_evolution")

        if cadence == "period":
            logger.info(f"Triggering evolution after period {pi} evaluation")

            # Prepare feedback for evolution
            feedback = collect_evolution_feedback(agent, results, item, period, pi)

            # Trigger evolution with feedback
            changes_made = agent._evolve_policy(feedback)

            # Log evolution step
            evolution_step = {
                "step_id": len(agent.evolution_history),
                "feedback": feedback,
                "changes": changes_made
            }
            agent.evolution_history.append(evolution_step)

        elif cadence == "no_evolution":
            logger.info(f"NO evolution (cadence == 'no_evolution') after period {pi} evaluation")
        else:
            raise ValueError(f"Unexpected cadence value in evolution config: {cadence}")

        agent.save_state(agent_state_dir)
        # ========== END EVOLUTION ==========

    # Compute and save metrics
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


def evaluate_item(item, agent, output_dir, env_config):
    """Perform evaluation of an agent on a single evaluation item.

    This is the main entry point that dispatches to the evolution-aware evaluation.

    Args:
        item: Evaluation item with state schema, QAs, and conversation periods.
        agent: Agent instance to evaluate.
        output_dir: Directory to save evaluation results.
        env_config: Environment configuration.
    """
    evaluate_item_with_evolution(item, agent, output_dir, env_config)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Evaluate evolvable agents using environment data and generated conversations.")
    parser.add_argument("--env_data", type=str, default="data/v1.base/data.json",
                        help="Environment data file")
    parser.add_argument("--env_config", type=str, default="configs/env/v1.base.json",
                        help="Environment configuration file")
    parser.add_argument("--agent_config", type=str, required=True,
                        help="Agent configuration file")
    parser.add_argument("--output_dir", type=str, default="eval-output/evolution",
                        help="Output directory for evaluation results")
    parser.add_argument("--reset", action="store_true",
                        help="Overwrite output")
    args = parser.parse_args()
    load_dotenv()

    # Load configurations
    agent_config = load_json(args.agent_config)
    env_config = load_json(args.env_config)

    # Fill in environment variables for env config
    for key in ["llm_config_low_temp", "llm_config_high_temp"]:
        env_config[key] |= {
            "base_url": env_config[key].get("base_url") or os.environ.get("OPENAI_BASE_URL"),
            "api_key": env_config[key].get("api_key") or os.environ.get("OPENAI_API_KEY"),
            "source": "env:interaction"
        }

    output_dir = os.path.join(args.output_dir, agent_config["name"])
    if args.reset and os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    shutil.copy(args.agent_config, output_dir)

    data = load_json(args.env_data)

    for item in data:
        item_dir = os.path.join(output_dir, item["id"])

        # Check if item already completed
        metric_path = os.path.join(item_dir, "overall_metrics.json")
        if os.path.exists(metric_path) and not args.reset:
            logger.info(f"Item {item['id']} already evaluated. Skipping.")
            continue

        agent = create_agent(agent_config, output_dir=item_dir)

        # Setup logging
        log_dir = os.path.join(item_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        setup_logger(os.path.join(log_dir, "evaluate.log"))

        logger.info(f"Evaluating item {item['id']}")
        evaluate_item(item, agent, item_dir, env_config)
