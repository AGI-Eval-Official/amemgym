"""Diagnosis module for on-policy agent performance assessment.

This module provides a decomposition of the memory lifecycle into write and read phases, and
offers corresponding metrics for diagnosis.
"""

import os
import json
import random
from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

from amemgym.assistants import create_agent
from amemgym.utils import save_json, load_json, parse_json, setup_logger

random.seed(42)  # For reproducibility


# Template for diagnosis - asks agent to select best states
DIAG_PROMPT = """\
{state_schema}

Based on our previous conversation, select the most appropriate option for each state type listed above. The selected option should be as close as possible to my current situation. 
Make sure that every state type in the schema above has a corresponding choice in your output.

Please respond strictly in the following JSON format:
```json
{{
    "info_type1": "choice",
    "info_type2": "choice",
    ...
}}
```
Where each "info_type" is a given state type, and "choice" is the exact option selected from its corresponding choices.

Only keep the JSON format output, do not include any other content.
"""


def diagnose_item(item, agent, output_dir) -> None:
    """Perform diagnostic evaluation of an agent on a single evaluation item.

    Args:
        item: Evaluation item containing periods, questions, and user profile information.
        agent: The agent instance to evaluate.
        output_dir: Directory path where evaluation results will be saved.

    Note:
        Results are saved incrementally to allow for resuming interrupted evaluations.
        Agent states are cached between periods to avoid redundant computations.
    """
    num_questions, num_periods = len(item["qas"]), len(item["periods"])
    results_path = os.path.join(output_dir, "diagnosis_results.json")
    
    if not os.path.exists(results_path):
        results = [[None for _ in range(num_questions)] for _ in range(num_periods)]
    else:
        results = load_json(results_path)
    
    state_schema = item["state_schema"]
    state_latest_update_pos = {k: None for k in state_schema.keys()}
    sub_schema_list = [
        {k: state_schema[k] for k in qa["required_info"]}
        for qa in item["qas"]
    ]
    sub_schema_str_list = [json.dumps(schema, indent=2, ensure_ascii=False) for schema in sub_schema_list]
    for pi, period in enumerate(tqdm(item["periods"], desc="Evaluating Write & Read", ncols=80)):
        agent_state_dir = os.path.join(output_dir, f"agent_states/period_{pi:02d}")
        assert os.path.exists(agent_state_dir), f"Agent state dir {agent_state_dir} does not exist."
        agent.load_state(agent_state_dir)

        latest_state = period["state"]
        for k in period["updates"].keys():
            state_latest_update_pos[k] = pi

        for qi, qa in enumerate(tqdm(item["qas"], desc="Asking States", ncols=80, leave=False)):
            if results[pi][qi] is not None:
                continue
            sub_schema = sub_schema_list[qi]
            sub_schema_str = sub_schema_str_list[qi]
            query = DIAG_PROMPT.format(state_schema=sub_schema_str)
            response, usage_statistics = agent.answer_question(query)
            
            try:
                response_state = parse_json(response)
            except Exception as e:
                response_state = {}
            
            diagnosis_result = {
                "query": query,
                "raw_response": response,
                "results": []
            }
            for info_type in qa["required_info"]:
                if info_type in response_state:
                    result = {
                        "info_type": info_type,
                        "json_error": False,
                        "answer_state": latest_state[info_type],
                        "response_state": response_state[info_type],
                        "write_pos": state_latest_update_pos[info_type],
                        "read_pos": pi
                    }
                else:
                    result = {
                        "info_type": info_type,
                        "json_error": True,
                        "answer_state": latest_state[info_type],
                        "response_state": None,
                        "write_pos": state_latest_update_pos[info_type],
                        "read_pos": pi
                    }
                result["score"] = float(result["answer_state"] == result["response_state"])
                diagnosis_result["results"].append(result)
            results[pi][qi] = diagnosis_result
            save_json(results_path, results)

    metric_path = os.path.join(output_dir, "diagnosis_metrics.json")
    # if os.path.exists(metric_path):
    #     return
    # diagnosis_metric = {k: np.zeros((num_periods, num_periods)) for k in ["success", "num_total"]}
    diagnosis_metric = {
        k: np.zeros((num_periods, num_questions), dtype=np.int32)
        for k in ["write_failure", "read_failure", "memory_success"]
    }

    for pi in range(num_periods):
        for qi in range(num_questions):
            result = results[pi][qi]
            write_failure, read_failure = 0, 0
            for ri, result in enumerate(result["results"]):
                assert result["read_pos"] == pi
                write_pos = result["write_pos"]
                write_score = results[write_pos][qi]["results"][ri]["score"]
                read_score = result["score"]
                if read_score < .5:
                    if write_score < .5:
                        write_failure += 1
                        # diagnosis_metric["write_failure"][pi] += 1
                    else:
                        read_failure += 1
                        # diagnosis_metric["read_failure"][pi] += 1
            diagnosis_metric["write_failure"][pi, qi] = write_failure
            diagnosis_metric["read_failure"][pi, qi] = read_failure
            if write_failure == 0 and read_failure == 0:
                diagnosis_metric["memory_success"][pi, qi] = 1
            else:
                diagnosis_metric["memory_success"][pi, qi] = 0
                # diagnosis_metric["total"][pi] += 1
                # diagnosis_metric["num_total"][result["write_pos"], result["read_pos"]] += 1
                # diagnosis_metric["success"][result["write_pos"], result["read_pos"]] += result["score"]
    
    for k, array in diagnosis_metric.items():
        diagnosis_metric[k] = array.tolist()
    save_json(metric_path, diagnosis_metric)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Diagnose memory failures.")
    parser.add_argument("--env_data", type=str, default="data/v1.base/data.json",
                        help="Environment data file")
    parser.add_argument("--agent_config", type=str, required=True,
                        help="Agent configuration file")
    parser.add_argument("--output_dir", type=str, default="eval-output/v1.base/native",
                        help="Output directory for evaluation results")
    parser.add_argument("--reset", action="store_true", help="Reset diagnosis output.")
    args = parser.parse_args()
    load_dotenv()

    # Load configurations
    agent_config = load_json(args.agent_config)
    agent_config["llm_config"] |= {"source": "agent:diagnosis"}

    args.output_dir = os.path.join(args.output_dir, agent_config["name"])

    data = load_json(args.env_data)
    for item in data:
        item_dir = os.path.join(args.output_dir, item["id"])
        if args.reset:
            log_path = os.path.join(item_dir, "logs/diagnose.log")
            if os.path.exists(log_path):
                os.remove(log_path)
            diagnose_path = os.path.join(item_dir, "diagnosis_results.json")
            if os.path.exists(diagnose_path):
                os.remove(diagnose_path)
            metric_path = os.path.join(item_dir, "diagnosis_metrics.json")
            if os.path.exists(metric_path):
                os.remove(metric_path)
        agent = create_agent(agent_config, output_dir=item_dir)
        setup_logger(log_path=os.path.join(item_dir, "logs/diagnose.log"))
        diagnose_item(item, agent, item_dir)
