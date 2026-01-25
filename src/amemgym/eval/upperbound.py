import os
import json
from argparse import ArgumentParser
import numpy as np
from loguru import logger
from tqdm import tqdm
from amemgym.utils import load_json, save_json, parse_json, call_llm, setup_logger
from .metric import state_similarity, METRICS
from dotenv import load_dotenv

import random
random.seed(42)  # For reproducibility

import time


# for upperbound - asks agent to select answer given explicit state information
UTILIZATION_PROMPT = """\
{query}

Given that my current relevant preferences and state information are as follows:
{state}

Please select the most suitable answer for my current situation from the following options:

{choices}

Express your choice with a number and output in the following JSON format:
```json
{{
    "answer": int
}}
```
Only keep the JSON format output, do not include any other content.
"""


def evaluate_utilization(data, llm_config, output_dir):
    results_path = os.path.join(output_dir, f"utilization_results.json")
    if not os.path.exists(results_path):
        results = [[[None for _ in qa["answer_choices"]] for qa in item["qas"]] for item in data]
    else:
        results = load_json(results_path)

    # collect results
    for i, item in enumerate(tqdm(data, desc="Evaluating Utilization", ncols=100)):
        for qi, qa in enumerate(
            tqdm(item["qas"], desc=f"Evaluating qas of {item['id']}", ncols=100, leave=False)
        ):
            choices_text = '\n'.join(['{}: {}'.format(ci + 1, choice['answer'])
                                    for ci, choice in enumerate(qa["answer_choices"])])
            for ci in tqdm(range(len(qa["answer_choices"])), ncols=100, leave=False):
                if results[i][qi][ci] is not None:
                    continue
                choice = qa["answer_choices"][ci]
                state = {info_type: value for info_type, value in zip(qa["required_info"], choice["state"])}
                state_text = json.dumps(state, ensure_ascii=False, indent=2)
                query = UTILIZATION_PROMPT.format(
                    query=qa["query"], state=state_text, choices=choices_text
                )
                response, usage_statistics = call_llm(
                    [{"role": "user", "content": query}], llm_config, json=True, return_token_usage=True
                )
                try:
                    response_answer = parse_json(response)["answer"]
                    json_error = False
                except json.JSONDecodeError:
                    logger.warning(f"json decoding error: {response}")
                    response_answer = random.randint(1, len(qa["answer_choices"])) # Fallback to random choice
                    json_error = True

                response_choice = qa["answer_choices"][response_answer - 1]
                scores = {}
                for metric in METRICS:
                    scores[metric] = state_similarity(choice["state"], response_choice["state"], metric)
                result = {
                    "query": qa["query"],
                    "answer": ci + 1,
                    "answer_state": choice["state"],
                    "answer_choice": choice["answer"],
                    "raw_response": response,
                    "response": response_answer,
                    "response_state": response_choice["state"],
                    "response_choice": response_choice["answer"],
                    "json_error": json_error,
                    "llm_usage_statistics": usage_statistics,
                    "scores": scores
                }
                results[i][qi][ci] = result
                save_json(results_path, results)  # periodically save results
                time.sleep(3)

    # compute metrics
    metrics_path = os.path.join(output_dir, f"utilization_metrics.json")
    if os.path.exists(metrics_path):
        utilization_metrics = load_json(metrics_path)
        for k, v in utilization_metrics.items():
            utilization_metrics[k] = np.array(v)
    else:
        N, Np, Nq = len(data), len(data[0]["periods"]), len(data[0]["qas"])
        utilization_metrics = {metric: np.zeros((N, Np, Nq)) for metric in METRICS}
        for i, item in enumerate(data):
            # construct mapping from given state to score
            state2score = {}
            for qa_scores in results[i]:
                for score in qa_scores:
                    state2score[tuple(score["answer_state"])] = score["scores"]

            for pi, period in enumerate(item["periods"]):
                cur_state = period["state"]
                for qi, qa in enumerate(item["qas"]):
                    required_cur_state = tuple([cur_state[info_type] for info_type in qa["required_info"]])
                    scores = state2score[required_cur_state]
                    for metric in METRICS:
                        utilization_metrics[metric][i, pi, qi] = scores[metric]
        utilization_metrics_serializable = {
            k: v.tolist() for k, v in utilization_metrics.items()
        }
        save_json(metrics_path, utilization_metrics_serializable)
    logger.info(f"Utilization Mean: {utilization_metrics['accuracy'].mean()}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluate utilization metrics for given llms.")
    parser.add_argument("--data_file", type=str, default="data/v1.base/data.json")
    parser.add_argument("--output_dir", type=str, default="eval-output/v1.base/ub")
    parser.add_argument("--agent_config", type=str, default="configs/agents/native.json")
    args = parser.parse_args()
    load_dotenv()

    llm_config = load_json(args.agent_config)["llm_config"]
    llm_config |= {
        "base_url": llm_config.get("base_url") or os.environ.get("OPENAI_BASE_URL"),
        "api_key": llm_config.get("api_key") or os.environ.get("OPENAI_API_KEY"),
        "source": "agent:ub-eval"
    }

    data = load_json(args.data_file)

    output_dir = os.path.join(args.output_dir, llm_config["llm_model"])
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "log")
    setup_logger(log_path)
    logger.info(f"Evaluating {llm_config['llm_model']}...")
    evaluate_utilization(data, llm_config, output_dir)
