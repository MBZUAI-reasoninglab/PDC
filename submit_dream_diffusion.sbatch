#!/bin/bash
#SBATCH -J gsm-dre
#SBATCH -A faculty
#SBATCH --nodes=1
#SBATCH -p long
#SBATCH -q gpu-12
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --chdir=.
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
: "${PROJECT_ROOT:=${SCRIPT_DIR}}"

unset VIRTUAL_ENV
export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
export MODELS_DIR="${MODELS_DIR:-${PROJECT_ROOT}/models}"

(
  cd "${PROJECT_ROOT}" && uv sync --frozen
)

cd "${PROJECT_ROOT}/regen_consistency"

: "${DATASET:=math500}"
: "${MODEL:=Dream-v0-Instruct-7B}"
: "${STEPS:=128}"
: "${GEN_LENGTH:=128}"
: "${CUTS:=10,50,90}"
: "${DIFFUSION_ALG:=entropy}"
: "${REGEN_ALG:=${DIFFUSION_ALG}}"
: "${NUM_ANSWERS:=10}"
: "${REGEN_NUM_ANSWERS:=10}"
: "${REPORT_ANSWER_COUNTS:=4,10}"
: "${TEMPERATURE:=0.2}"
: "${REGEN_TEMP:=0.2}"
: "${PROBLEMS:=}"
: "${INIT_DIR:=${PROJECT_ROOT}/regen_consistency/diffusion/${DATASET}/${MODEL}_init_${DIFFUSION_ALG}_steps${STEPS}_T${TEMPERATURE}}"
: "${REGEN_DIR:=${PROJECT_ROOT}/regen_consistency/diffusion/${DATASET}/${MODEL}_regen_from_${REGEN_ALG}_steps${STEPS}_initT${TEMPERATURE}_regenT${REGEN_TEMP}}"





export DATASET MODEL STEPS GEN_LENGTH CUTS DIFFUSION_ALG REGEN_ALG \
       INIT_DIR REGEN_DIR PROBLEMS TEMPERATURE REGEN_TEMP NUM_ANSWERS \
       REGEN_NUM_ANSWERS REPORT_ANSWER_COUNTS

bash diffusion/run_dream_pipeline.sh

REPORT_MODEL_PATH_ARGS=()
if [ -n "${MODEL_PATH:-}" ]; then
  REPORT_MODEL_PATH_ARGS=(--model-path "${MODEL_PATH}")
fi

echo "[submit] writing diffusion report tex..."
uv run python ../analysis/write_diffusion_report.py \
  --init-dir "${INIT_DIR}" \
  --regen-dir "${REGEN_DIR}" \
  --num-answers "${NUM_ANSWERS}" \
  --report-answer-counts "${REPORT_ANSWER_COUNTS}" \
  --dataset "${DATASET}" \
  --model-name "${MODEL}" \
  "${REPORT_MODEL_PATH_ARGS[@]}" \
  --problems "${PROBLEMS}" \
  --keeps "${CUTS}" \
  --tif-alpha 5 \
  --tif-total-steps "${STEPS}" \
  || echo "[submit] WARNING: report generation failed; pipeline outputs remain in ${INIT_DIR} and ${REGEN_DIR}" >&2
