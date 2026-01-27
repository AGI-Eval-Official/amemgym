<h1 align="center">AMemGym: Interactive Memory Benchmarking for Assistants in Long-horizon Conversations</h1>

<p align="center">
    <a href="https://agi-eval-official.github.io/amemgym/#/"><img src="https://img.shields.io/badge/Project-Website-blue" alt="Website"></a>
    <a href="https://arxiv.org/abs/coming-soon"><img src="https://img.shields.io/badge/arXiv-xxxx.xxxxx-b31b1b.svg" alt="Paper"></a>
    <a href="https://huggingface.co/datasets/AGI-Eval/AMemGym"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Dataset-green" alt="Dataset"></a>
    <a href="https://github.com/AGI-Eval-Official/amemgym"><img src="https://img.shields.io/badge/-Github-grey?logo=github" alt="Github"></a>
    <a href="https://github.com/AGI-Eval-Official/amemgym/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License"></a>
</p>

This repo contains the code and data for the paper: *[AMemGym: Interactive Memory Benchmarking for Assistants in Long-horizon Conversations](https://openreview.net/forum?id=sfrVLzsmlf)*.

---

## Overview

AMemGym is the first **interactive, on-policy evaluation framework** for conversational memory in LLM-based assistants. Unlike traditional static benchmarks that rely on pre-generated conversations, AMemGym enables realistic evaluation by allowing assistants to generate their own responses and learn from environmental feedback—bridging the gap between evaluation and real-world deployment.

<img src="assets/figures/framework.png" width="800px" alt="AMemGym Framework">

### Key Features

- **Realistic Evaluation**: Assistants actively participate in conversations with simulated users adapting to their responses
- **Fine-Grained Diagnostics**: Pinpoints failures in Write, Read, and Utilization operations
- **Optimization Feedback**: Enables autonomous agent self-evolution through environmental feedback
- **Fully Automated**: Scalable generation of diverse, high-quality scenarios spanning 128K-512K+ context lengths

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/amemgym.git
cd amemgym

# Install with uv (recommended)
uv sync

# Install the package with pip
uv pip install -e .
```

Set up LLM API access by creating a `.env` file or exporting environment variables:

```bash
export OPENAI_API_KEY=your_api_key
export OPENAI_BASE_URL=https://api.openai.com/v1
```

---

## Quick Start

### Running On-Policy Evaluation

#### Prepare Environment Data
First, prepare the environment data. You can use the provided [v1.base](https://huggingface.co/datasets/AGI-Eval/AMemGym) dataset (aligned with the paper) or create your own.

```bash
# Generate environment data using the provided configuration that you can customize
uv run python -m amemgym.env.gen \
    --data_dir data/v1.base \
    --config_path configs/env/v1.base.json \
    --persona_path data/personas/nemotron.parquet
```

Note that `data/personas/nemotron.parquet` contains user personas from [Nemotron-Personas](https://huggingface.co/datasets/nvidia/Nemotron-Personas) used in the environment data generation.

#### Running Main Evaluation

```bash
# Run main evaluation with a specific agent configuration
uv run python -m amemgym.eval.overall \
    --agent_config configs/agent/awi.json \
    --env_data data/v1.base/data.json \
    --output_dir eval-output/v1.base/overall
```

**Available Agent Configurations:**

| Agent Type | Description | Example Config |
|------------|-------------|------|
| **AWI** | Agentic Write In-context | `configs/agent/awi.json` |
| **AWE** | Agentic Write External | `configs/agent/awe-2-4-30.json` |
| **RAG** | Retrieval Augmented Generation | `configs/agent/rag-2-4-30.json` |
| **Native** | Native LLM (no memory system) | `configs/agent/native.json` |


#### Running Upper-Bound and Random Baselines
Use the following commands to run upper-bound and random baseline evaluations for normalized memory scores.

```bash
# Run upper-bound evaluation
uv run python -m amemgym.eval.upperbound \
    --agent_config <a specific agent config, e.g., configs/agent/awi.json> \
    --env_data data/v1.base/data.json \
    --output_dir eval-output/v1.base/upperbound

# Run random baseline evaluation
uv run python -m amemgym.eval.random \
    --env_data data/v1.base/data.json \
    --output_file eval-output/v1.base/random_metrics.json
```

#### (Optional) Running Fine-Grained Diagnostics
```
# Run fine-grained diagnostics for a specific agent configuration
uv run python -m amemgym.eval.diagnosis \
    --agent_config <a specific agent config, e.g., configs/agent/awi.json> \
    --env_data data/v1.base/data.json \
    --output_dir eval-output/v1.base  # consistent with overall evaluation output dir
```

### Running Evolution Experiments

For self-evolution experiments (Table 3 in paper):

```bash
uv run python -m amemgym.eval.evolution \
    --agent_config configs/agent/awi-evolve/complete.json
```

**Available Evolution Configurations:**

| Config | Description | Example Config |
|--------|-------------|------|
| **No Evolution** | Baseline without prompt evolution | `configs/agent/awi-evolve/no-evolution.json` |
| **Question Only** | Evolution with question-only feedback | `configs/agent/awi-evolve/question-only.json` |
| **Complete** | Full evolution with complete feedback | `configs/agent/awi-evolve/complete.json` |

---

## Citation

If you find AMemGym useful for your research, please cite our paper:

```bibtex
@inproceedings{
    jiayang2026amemgym,
    title={{AM}emGym: Interactive Memory Benchmarking for Assistants in Long-Horizon Conversations},
    author={Cheng Jiayang and Dongyu Ru and Lin Qiu and Yiyang Li and Xuezhi Cao and Yangqiu Song and Xunliang Cai},
    booktitle={The Fourteenth International Conference on Learning Representations},
    year={2026},
    url={https://openreview.net/forum?id=sfrVLzsmlf}
}
```

