#!/bin/bash
#SBATCH -J tau512
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
if [ -n "${MODEL_NAME:-}" ]; then
  MODEL="${MODEL_NAME}"
fi
: "${MODEL:=LLaDA-1.5}"
case "${MODEL}" in
  LLaDA-1.5) DEFAULT_MODEL_PATH="GSAI-ML/LLaDA-1.5" ;;
  LLaDA-8B-Instruct) DEFAULT_MODEL_PATH="GSAI-ML/LLaDA-8B-Instruct" ;;
  LLaDA-8B-Base) DEFAULT_MODEL_PATH="GSAI-ML/LLaDA-8B-Base" ;;
  LLaDA-MoE-7B-A1B-Instruct) DEFAULT_MODEL_PATH="inclusionAI/LLaDA-MoE-7B-A1B-Instruct" ;;
  LLaDA2.0-mini-preview) DEFAULT_MODEL_PATH="inclusionAI/LLaDA2.0-mini-preview" ;;
  *) DEFAULT_MODEL_PATH="${MODELS_DIR}/${MODEL}" ;;
esac
: "${MODEL_PATH:=${DEFAULT_MODEL_PATH}}"
: "${STEPS:=512}"
: "${GEN_LENGTH:=512}"
: "${BLOCK_LENGTH:=512}"
: "${CUTS:=10,50,90}"
: "${TEMPERATURE:=0.2}"
: "${REGEN_TEMP:=0.2}"
: "${TOP_P:=1.0}"
: "${REMASKING:=low_confidence}"
: "${CFG_SCALE:=0.0}"
: "${PARALLEL:=1}"
: "${SAVE_LOGPROBS:=0}"
: "${CAPTURE_TIF:=1}"
: "${TIF_SNAP_STRIDE:=1}"
: "${SAVE_TRAJECTORY:=0}"
: "${NUM_ANSWERS:=1}"
: "${REGEN_NUM_ANSWERS:=1}"
: "${REPORT_ANSWER_COUNTS:=}"
: "${PROBLEMS:=}"
: "${INIT_DIR:=${PROJECT_ROOT}/regen_consistency/diffusion/${DATASET}/${MODEL}_init_llada_steps${STEPS}_gen${GEN_LENGTH}_block${BLOCK_LENGTH}_temp${TEMPERATURE}_${REMASKING}}"
: "${REGEN_DIR:=${PROJECT_ROOT}/regen_consistency/diffusion/${DATASET}/${MODEL}_regen_from_llada_steps${STEPS}_gen${GEN_LENGTH}_block${BLOCK_LENGTH}_initT${TEMPERATURE}_regenT${REGEN_TEMP}_${REMASKING}_regenprop}"

TEMP_TAG="${TEMPERATURE//./p}"
TEMP_TAG="${TEMP_TAG//-/m}"
REGEN_TEMP_TAG="${REGEN_TEMP//./p}"
REGEN_TEMP_TAG="${REGEN_TEMP_TAG//-/m}"
REMASK_TAG="${REMASKING}"
RUN_TAG="steps${STEPS}_gen${GEN_LENGTH}_block${BLOCK_LENGTH}_temp${TEMP_TAG}_${REMASK_TAG}"
REGEN_RUN_TAG="steps${STEPS}_gen${GEN_LENGTH}_block${BLOCK_LENGTH}_initT${TEMP_TAG}_regenT${REGEN_TEMP_TAG}_${REMASK_TAG}"
DEFAULT_INIT_DIR="${PROJECT_ROOT}/regen_consistency/diffusion/${DATASET}/${MODEL}_init_llada_${RUN_TAG}"
DEFAULT_REGEN_DIR="${PROJECT_ROOT}/regen_consistency/diffusion/${DATASET}/${MODEL}_regen_from_llada_${REGEN_RUN_TAG}_regenprop"
if [ -z "${INIT_DIR:-}" ] || [ "${INIT_DIR}" = "/etc/qlustar/common/skel/bash/../init" ]; then
  INIT_DIR="${DEFAULT_INIT_DIR}"
fi
if [ -z "${REGEN_DIR:-}" ]; then
  REGEN_DIR="${DEFAULT_REGEN_DIR}"
fi

export DATASET MODEL MODEL_PATH STEPS GEN_LENGTH BLOCK_LENGTH CUTS \
       TEMPERATURE REGEN_TEMP TOP_P REMASKING CFG_SCALE PARALLEL SAVE_LOGPROBS \
       CAPTURE_TIF TIF_SNAP_STRIDE SAVE_TRAJECTORY PROBLEMS INIT_DIR REGEN_DIR \
       NUM_ANSWERS REGEN_NUM_ANSWERS REPORT_ANSWER_COUNTS

bash diffusion/run_llada_pipeline.sh

echo "[submit-llada] writing diffusion report tex..."
uv run python ../analysis/write_diffusion_report.py \
  --init-dir "${INIT_DIR}" \
  --regen-dir "${REGEN_DIR}" \
  --dataset "${DATASET}" \
  --model-name "${MODEL}" \
  --model-path "${MODEL_PATH}" \
  --problems "${PROBLEMS}" \
  --keeps "${CUTS}" \
  --tif-alpha 5 \
  --tif-total-steps "${STEPS}" \
  --num-answers "${NUM_ANSWERS}" \
  --report-answer-counts "${REPORT_ANSWER_COUNTS}" \
  || echo "[submit-llada] WARNING: report generation failed; pipeline outputs remain in ${INIT_DIR} and ${REGEN_DIR}" >&2
