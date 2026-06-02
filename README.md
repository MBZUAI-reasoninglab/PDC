# Regeneration Consistency for Diffusion Language Models

This repository contains the code used to run regeneration-consistency
experiments for masked-diffusion language models. Given an initial answer, the
pipeline keeps a fixed prefix, re-denoises the remaining suffix, and reports how
stable or useful the regenerated answers are.

## Entry Points

| Script | Model family | Purpose |
| --- | --- | --- |
| `submit_dream_diffusion.sbatch` | Dream | initial generation, multi-cut regeneration, report generation |
| `submit_llada_diffusion.sbatch` | LLaDA | initial generation, multi-cut regeneration, report generation |

## Setup

Requirements:

- Linux GPU node with CUDA.
- Slurm for the provided submission scripts.
- Python `>=3.12,<3.13`.
- [`uv`](https://docs.astral.sh/uv/) for environment setup.
- Dataset JSONL files and model checkpoints visible from the compute node.

Install dependencies from the repository root:

```bash
uv sync --frozen
```

The lock file uses PyTorch CUDA 12.6 wheels. Edit the `#SBATCH` header in each
submission script for your cluster account, partition, QoS, memory, and wall
time.

## Data and Models

Datasets default to:

```text
<repo>/datasets/
```

Override with:

```bash
export DATASETS_DIR=/path/to/datasets
```

Model paths can be supplied with:

```bash
export MODELS_DIR=/path/to/models
export MODEL_PATH=/path/to/model-or-hf-id
```

Dataset schemas and model defaults are configured in
`regen_consistency/core/config.py`.

## Running

Dream:

```bash
sbatch \
  --export=ALL,DATASET=math500,PROBLEMS=0-499 \
  -J dream-math500 \
  submit_dream_diffusion.sbatch
```

LLaDA:

```bash
sbatch \
  --export=ALL,DATASET=math500,PROBLEMS=0-499 \
  -J llada-math500 \
  submit_llada_diffusion.sbatch
```

Common overrides:

```bash
DATASET=math500
PROBLEMS=0-99
STEPS=128
GEN_LENGTH=128
CUTS=10,50,90
TEMPERATURE=0.2
REGEN_TEMP=0.2
NUM_ANSWERS=1
REGEN_NUM_ANSWERS=1
```

LLaDA additionally supports:

```bash
MODEL=LLaDA-1.5
MODEL_PATH=GSAI-ML/LLaDA-1.5
BLOCK_LENGTH=512
REMASKING=low_confidence
```

## Outputs

Runs write outputs under:

```text
regen_consistency/diffusion/<dataset>/
```

Important artifacts:

| Artifact | Pattern |
| --- | --- |
| Initial answers | `*_probP_answerA.txt` |
| Regenerated answers | `*_probP_keepKK.txt`, `*_probP_answerA_keepKK.txt` |
| Optional sidecars | `*.npy`, `*.tok_logprobs.npy`, `*.unmask_step.npy` |
| Report | `reports/*_report.tex` |

The report summarizes initial accuracy, regeneration majority accuracy,
agreement-based fallback metrics, multi-answer majority metrics, and per-cut
retention.

## Repository Layout

```text
.
├── submit_dream_diffusion.sbatch
├── submit_llada_diffusion.sbatch
├── analysis/
│   ├── math_answer_grader.py
│   ├── multiple_choice_grader.py
│   └── write_diffusion_report.py
└── regen_consistency/
    ├── generate_initial_answers.py
    ├── core/
    └── diffusion/
```

## Reproducibility

- `uv.lock` pins the Python package set.
- Complete outputs are skipped, so jobs can be resumed by re-submission.
- Runtime settings are passed through environment variables and recorded in
  `metadata.json`.
- `PROBLEMS=START-END` is useful for smoke tests.

## License

This repository's code is released under the MIT License; see `LICENSE`.
Models, datasets, remote model code, and Python packages remain under their
own upstream licenses.
