from argparse import ArgumentParser
import json
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


def remove_leading_zeros(x):
    if x < 0:
        return "-" + remove_leading_zeros(-x)
    if x < 1:
        return f"{x:.3f}".lstrip('0')
    if x >= 1:
        return f"{x:.2f}"


def plot_heatmap(data, ordered_model_names, output_file):
    plt.clf()
    plt.figure(figsize=(7.5, 3.1))

    heatmap_array = np.array([data[name] for name in ordered_model_names])
    x_labels = ["UB", "Mean"] + [str(i) for i in range(heatmap_array.shape[1]-2)]
    annot = np.vectorize(remove_leading_zeros)(heatmap_array)
    ax = sns.heatmap(
        heatmap_array, 
        # annot=True,
        # fmt='.3f',
        annot=annot,
        fmt='',
        vmin=data["random"][1:].min(),
        cmap=plt.cm.RdYlGn,
        yticklabels=ordered_model_names,
        xticklabels=x_labels,
        annot_kws={"size": 10}
    )
    cbar = ax.collections[0].colorbar
    cbar.set_label(label='Overall Score', size=12, weight='bold')

    # plt.title('Overall Score', fontsize=14, fontweight='bold')
    plt.xlabel('Period Index', fontsize=12)
    # plt.ylabel('Models', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    # save pdf
    output_file_pdf = output_file.replace(".png", ".pdf")
    plt.savefig(output_file_pdf, dpi=300, bbox_inches='tight')


def plot_heatmap_normalized(data, ordered_model_names, output_file):
    plt.clf()
    plt.figure(figsize=(7.5, 3.1))

    heatmap_array = np.array([data[name] for name in ordered_model_names])
    x_labels = ["Mean"] + [str(i) for i in range(heatmap_array.shape[1]-1)]
    annot = np.vectorize(remove_leading_zeros)(heatmap_array)
    ax = sns.heatmap(
        heatmap_array, 
        # annot=True, 
        # fmt='.3f',
        annot=annot,
        fmt='',
        vmin=0.,
        cmap=plt.cm.RdYlGn,
        yticklabels=ordered_model_names,
        xticklabels=x_labels,
        annot_kws={"size": 10}
    )
    cbar = ax.collections[0].colorbar
    cbar.set_label(label='Memory Score', size=12, weight='bold')

    # plt.title('Memory Score (Normalized)', fontsize=14, fontweight='bold')
    plt.xlabel('Period Index', fontsize=12)
    # plt.ylabel('Models', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    # save pdf
    output_file_pdf = output_file.replace(".png", ".pdf")
    plt.savefig(output_file_pdf, dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/figure/example.json")
    args = parser.parse_args()

    with open(args.config_path) as fp:
        config = json.load(fp)
    os.makedirs(config["output_dir"], exist_ok=True)
    agent_names = [agent["name"] for agent in config["agents"]]

    with open(config["random_path"]) as fp:
        # shape: [num_users, num_periods, num_questions]
        random_metrics = json.load(fp)

    with open(config["env_data_path"]) as f:
        env_data = json.load(f)
    item_ids = [item["id"] for item in env_data]

    METRIC = "accuracy"
    data_overall = {}
    data_memory = {}
    name2score_overall = {}
    name2score_memory = {}
    random_scores = np.array(random_metrics[METRIC]).mean(axis=2)  # [Nu, Np]

    for agent in config["agents"]:
        # collect overall metrics
        overall_metrics = []
        for item_id in item_ids:
            item_dir = os.path.join(agent["output_dir"], item_id)
            with open(os.path.join(item_dir, "overall_metrics.json")) as f:
                # shape: [num_periods, num_questions]
                overall_metrics.append(json.load(f)[METRIC])
        overall_metrics = np.array(overall_metrics).mean(axis=2)  # [Nu, Np]

        # collect upperbound metrics
        with open(agent["upperbound_path"]) as fp:
            # shape: [num_users, num_periods, num_questions]
            ub_metrics = np.array(json.load(fp)[METRIC]).mean(axis=2)  # [Nu, Np]

        memory_scores = (overall_metrics - random_scores) / (ub_metrics - random_scores)  # [Nu, Np]

        name2score_overall[agent["name"]] = float(overall_metrics.mean())
        name2score_memory[agent["name"]] = float(memory_scores.mean())

        data_overall[agent["name"]] = np.array(
            [ub_metrics.mean(), overall_metrics.mean()] + overall_metrics.mean(axis=0).tolist()
        )
        data_memory[agent["name"]] = np.array([memory_scores.mean()] + memory_scores.mean(axis=0).tolist())

    # plot overall scores
    ordered_agent_names = sorted(agent_names, key=lambda x: data_overall[x][1], reverse=True) + ["random"]
    data_overall["random"] = np.array([float('nan'), random_scores.mean()] + random_scores.mean(axis=0).tolist())
    output_path = os.path.join(config["output_dir"], f"overall.png")
    plot_heatmap(data_overall, ordered_agent_names, output_path)

    # plot normalized memory scores
    ordered_agent_names = sorted(agent_names, key=lambda x: data_memory[x][0], reverse=True)
    output_path = os.path.join(config["output_dir"], f"memory.png")
    plot_heatmap_normalized(data_memory, ordered_agent_names, output_path)

    with open(os.path.join(config["output_dir"], f"overall.json"), "w") as fp:
        json.dump(name2score_overall, fp, indent=2, ensure_ascii=False)
    
    with open(os.path.join(config["output_dir"], f"memory.json"), "w") as fp:
        json.dump(name2score_memory, fp, indent=2, ensure_ascii=False)
