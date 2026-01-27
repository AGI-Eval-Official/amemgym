import json
from argparse import ArgumentParser
import numpy as np

from .metric import METRICS, state_similarity


def evaluate_random(data, metric):
    N, Np, Nq = len(data), len(data[0]["periods"]), len(data[0]["qas"])
    random_scores = np.zeros((N, Np, Nq))
    for i, item in enumerate(data):
        for pi, period in enumerate(item["periods"]):
            cur_state = period["state"]
            for qi, qa in enumerate(item["qas"]):
                required_cur_state = [cur_state[info_type] for info_type in qa["required_info"]]
                scores = []
                for answer_choice in qa["answer_choices"]:
                    scores.append(state_similarity(required_cur_state, answer_choice["state"], metric))
                random_scores[i, pi, qi] = np.mean(scores)
    random_scores = np.array(random_scores)
    # print(random_scores.shape)
    # print(metric, random_scores.mean(axis=(0, 2)))
    return random_scores


if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluate random strategy.")
    parser.add_argument("--env_data", type=str, default="data/v1.base/data.json")
    parser.add_argument("--output_file", type=str, default="eval-output/v1.base/random_metrics.json")
    args = parser.parse_args()

    # Load data
    with open(args.env_data, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Evaluate the data
    output_data = {}
    for metric in METRICS:
        random_scores = evaluate_random(data, metric=metric)
        output_data[metric] = random_scores.tolist()
    with open(args.output_file, "w") as fp:
        json.dump(output_data, fp, ensure_ascii=False, indent=2)
