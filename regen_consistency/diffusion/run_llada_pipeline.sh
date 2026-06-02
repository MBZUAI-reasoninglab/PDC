#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RC_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$RC_ROOT/.." && pwd)"
cd "$RC_ROOT"

unset VIRTUAL_ENV

_preflight_cuda_torch() {
  local ec=0
  uv run python - <<'PY' || ec=$?
import sys
try:
    import torch
except Exception as exc:
    print("[llada-pipeline] ERROR: import torch failed:", exc, file=sys.stderr)
    raise SystemExit(1)
if not torch.cuda.is_available():
    print("[llada-pipeline] ERROR: CUDA not available; allocate a GPU job.", file=sys.stderr)
    raise SystemExit(2)
print("[llada-pipeline] PyTorch", torch.__version__,
      "| CUDA:", torch.version.cuda,
      "| GPU:", torch.cuda.get_device_name(0))
PY
  return "${ec}"
}
_preflight_cuda_torch

: "${DATASET:=math500}"
if [ -n "${MODEL_NAME:-}" ]; then
  MODEL="${MODEL_NAME}"
fi
: "${MODEL:=LLaDA-8B-Instruct}"
case "${MODEL}" in
  LLaDA-1.5) DEFAULT_MODEL_PATH="GSAI-ML/LLaDA-1.5" ;;
  LLaDA-8B-Instruct) DEFAULT_MODEL_PATH="GSAI-ML/LLaDA-8B-Instruct" ;;
  LLaDA-8B-Base) DEFAULT_MODEL_PATH="GSAI-ML/LLaDA-8B-Base" ;;
  LLaDA-MoE-7B-A1B-Instruct) DEFAULT_MODEL_PATH="inclusionAI/LLaDA-MoE-7B-A1B-Instruct" ;;
  LLaDA2.0-mini-preview) DEFAULT_MODEL_PATH="inclusionAI/LLaDA2.0-mini-preview" ;;
  *) DEFAULT_MODEL_PATH="${MODELS_DIR:-${REPO_ROOT}/models}/${MODEL}" ;;
esac
: "${MODEL_PATH:=${DEFAULT_MODEL_PATH}}"
: "${PROBLEMS:=}"
: "${STEPS:=512}"
: "${GEN_LENGTH:=512}"
: "${BLOCK_LENGTH:=32}"
: "${CUTS:=10,50,90}"
: "${NUM_ANSWERS:=1}"
: "${REGEN_NUM_ANSWERS:=${NUM_ANSWERS}}"
: "${PARALLEL:=1}"
: "${TEMPERATURE:=0.0}"
: "${REGEN_TEMP:=${TEMPERATURE}}"
: "${TOP_P:=1.0}"
: "${REMASKING:=low_confidence}"
: "${CFG_SCALE:=0.0}"
: "${SAVE_LOGPROBS:=1}"
: "${CAPTURE_TIF:=1}"
: "${TIF_SNAP_STRIDE:=1}"
: "${SAVE_TRAJECTORY:=0}"

_require_int() {
  local name="$1"
  local value="$2"
  if ! [[ "${value}" =~ ^[0-9]+$ ]]; then
    echo "[llada-pipeline] ERROR: ${name} must be an integer, got '${value}'." >&2
    echo "  Example: use GEN_LENGTH=128, not GEN_LENGTH=STEPS." >&2
    exit 2
  fi
}
_require_int STEPS "${STEPS}"
_require_int GEN_LENGTH "${GEN_LENGTH}"
_require_int BLOCK_LENGTH "${BLOCK_LENGTH}"

if [ $(( GEN_LENGTH % BLOCK_LENGTH )) -ne 0 ]; then
  echo "[llada-pipeline] ERROR: GEN_LENGTH (${GEN_LENGTH}) must be divisible by BLOCK_LENGTH (${BLOCK_LENGTH})" >&2
  exit 2
fi
NUM_BLOCKS=$(( GEN_LENGTH / BLOCK_LENGTH ))
if [ $(( STEPS % NUM_BLOCKS )) -ne 0 ]; then
  echo "[llada-pipeline] ERROR: STEPS (${STEPS}) must be divisible by GEN_LENGTH/BLOCK_LENGTH (${NUM_BLOCKS})" >&2
  exit 2
fi

TEMP_TAG="${TEMPERATURE//./p}"
TEMP_TAG="${TEMP_TAG//-/m}"
REGEN_TEMP_TAG="${REGEN_TEMP//./p}"
REGEN_TEMP_TAG="${REGEN_TEMP_TAG//-/m}"
REMASK_TAG="${REMASKING}"
RUN_TAG="steps${STEPS}_gen${GEN_LENGTH}_block${BLOCK_LENGTH}_temp${TEMP_TAG}_${REMASK_TAG}"
REGEN_RUN_TAG="steps${STEPS}_gen${GEN_LENGTH}_block${BLOCK_LENGTH}_initT${TEMP_TAG}_regenT${REGEN_TEMP_TAG}_${REMASK_TAG}"
INIT_DIR="${INIT_DIR:-${SCRIPT_DIR}/${DATASET}/${MODEL}_init_llada_${RUN_TAG}}"
REGEN_DIR="${REGEN_DIR:-${SCRIPT_DIR}/${DATASET}/${MODEL}_regen_from_llada_${REGEN_RUN_TAG}_regenprop}"
PREFIX="${DATASET}_${MODEL}"

INIT_ARGS=(
  --dataset "${DATASET}"
  --model-name "${MODEL}"
  --model-path "${MODEL_PATH}"
  --num-answers "${NUM_ANSWERS}"
  --out-dir "${INIT_DIR}"
  --parallel "${PARALLEL}"
  --diffusion-batch-size 1
  --diffusion-steps "${STEPS}"
  --diffusion-gen-length "${GEN_LENGTH}"
  --diffusion-block-length "${BLOCK_LENGTH}"
  --diffusion-temperature "${TEMPERATURE}"
  --diffusion-remasking "${REMASKING}"
  --diffusion-cfg-scale "${CFG_SCALE}"
)
if [ "${SAVE_LOGPROBS}" = "1" ]; then
  INIT_ARGS+=( --save-logprobs )
fi
if [ "${CAPTURE_TIF}" = "1" ]; then
  INIT_ARGS+=( --diffusion-capture-tif --tif-snap-stride "${TIF_SNAP_STRIDE}" )
fi
if [ "${SAVE_TRAJECTORY}" = "1" ]; then
  INIT_ARGS+=( --save-trajectory )
fi
if [ -n "${PROBLEMS}" ]; then
  INIT_ARGS+=( --problems "${PROBLEMS}" )
fi

echo "[llada-pipeline] Phase A: generate_initial_answers.py -> ${INIT_DIR}"
echo "[llada-pipeline] model=${MODEL} model_path=${MODEL_PATH}"
echo "[llada-pipeline] backend=HF direct"
echo "[llada-pipeline] answers=${NUM_ANSWERS} gen_length=${GEN_LENGTH} steps=${STEPS} block=${BLOCK_LENGTH} remasking=${REMASKING} init_T=${TEMPERATURE} cfg=${CFG_SCALE}"
uv run python generate_initial_answers.py "${INIT_ARGS[@]}"

echo "[llada-pipeline] Phase B: multi-cut LLaDA regen -> ${REGEN_DIR} (answers=${REGEN_NUM_ANSWERS} cuts=${CUTS} regen_T=${REGEN_TEMP})"
REGEN_ARGS=(
  --init-dir "${INIT_DIR}"
  --out-dir "${REGEN_DIR}"
  --dataset "${DATASET}"
  --model-name "${MODEL}"
  --model-path "${MODEL_PATH}"
  --cuts "${CUTS}"
  --num-answers "${REGEN_NUM_ANSWERS}"
  --gen-length "${GEN_LENGTH}"
  --steps "${STEPS}"
  --block-length "${BLOCK_LENGTH}"
  --temperature "${REGEN_TEMP}"
  --top-p "${TOP_P}"
  --remasking "${REMASKING}"
  --cfg-scale "${CFG_SCALE}"
)
if [ -n "${PROBLEMS}" ]; then
  REGEN_ARGS+=( --problems "${PROBLEMS}" )
fi
uv run python "${SCRIPT_DIR}/regen_llada_multi_cut.py" "${REGEN_ARGS[@]}"

echo "[llada-pipeline] done."
