#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RC_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$RC_ROOT"

unset VIRTUAL_ENV

_preflight_cuda_torch() {
  local ec=0
  uv run python - <<'PY' || ec=$?
import sys

try:
    import torch
except Exception as exc:
    print("[pipeline] ERROR: import torch failed:", exc, file=sys.stderr)
    raise SystemExit(1)
if not torch.cuda.is_available():
    print(
        "[pipeline] ERROR: CUDA not available (torch.cuda.is_available()==False). "
        "Run inside a Slurm job with GPU allocation.",
        file=sys.stderr,
    )
    raise SystemExit(2)
print(
    "[pipeline] PyTorch",
    torch.__version__,
    "| CUDA:",
    torch.version.cuda,
    "| GPU:",
    torch.cuda.get_device_name(0),
)
PY
  if [ "${ec}" -eq 139 ] || [ "${ec}" -eq 134 ]; then
    echo "[pipeline] ERROR: Python crashed (exit ${ec}). Likely PyTorch CUDA build vs NVIDIA driver mismatch." >&2
    echo "  Stack uses PyTorch cu126 wheels (see pyproject.toml). If 139 persists: unset VIRTUAL_ENV, rm -rf .venv && uv sync on this arch, then retry." >&2
  fi
  return "${ec}"
}
_preflight_cuda_torch

: "${DATASET:=math500}"
: "${MODEL:=Dream-v0-Instruct-7B}"
: "${PROBLEMS:=}"
: "${STEPS:=256}"
: "${GEN_LENGTH:=256}"
: "${CUTS:=10,50,90}"
: "${NUM_ANSWERS:=1}"
: "${REGEN_NUM_ANSWERS:=${NUM_ANSWERS}}"
: "${CAPTURE_TIF:=1}"
: "${TIF_STRIDE:=1}"
: "${PARALLEL:=2}"
: "${TEMPERATURE:=0.2}"
: "${DIFFUSION_ALG:=entropy}"
: "${ALG_TEMP:=0}"
: "${REGEN_TEMP:=0.2}"
: "${REGEN_TOP_P:=0.95}"
: "${REGEN_TOP_K:=0}"
: "${REGEN_ALG:=${DIFFUSION_ALG}}"
: "${REGEN_ALG_TEMP:=${ALG_TEMP}}"
: "${INIT_DIR:=${SCRIPT_DIR}/${DATASET}/${MODEL}_init_${DIFFUSION_ALG}_steps${STEPS}}"
: "${REGEN_DIR:=${SCRIPT_DIR}/${DATASET}/${MODEL}_regen_from_${REGEN_ALG}_steps${STEPS}_initT${TEMPERATURE}_regenT${REGEN_TEMP}}"

PROBLEMS_ARGS=()
if [ -n "${PROBLEMS}" ]; then
  PROBLEMS_ARGS=(--problems "${PROBLEMS}")
fi
MODEL_PATH_ARGS=()
if [ -n "${MODEL_PATH:-}" ]; then
  MODEL_PATH_ARGS=(--model-path "${MODEL_PATH}")
fi

echo "[pipeline] Phase A: generate_initial_answers.py -> ${INIT_DIR}  (answers=${NUM_ANSWERS} gen_length=${GEN_LENGTH} steps=${STEPS} alg=${DIFFUSION_ALG} T=${TEMPERATURE} alg_temp=${ALG_TEMP})"
if [ -n "${MODEL_PATH:-}" ]; then
  echo "[pipeline] MODEL_PATH=${MODEL_PATH}"
fi
if [ -n "${PROBLEMS}" ]; then
  echo "[pipeline] PROBLEMS=${PROBLEMS}"
else
  echo "[pipeline] PROBLEMS=(unset) -> all problems in dataset"
fi
if [ "${CAPTURE_TIF}" = "1" ]; then
  echo "[pipeline] TiF: ON  ->  each answer also writes  *_answerK_tif.npz  (same NFE as spine)."
  uv run python generate_initial_answers.py \
    --dataset "${DATASET}" \
    --model-name "${MODEL}" \
    "${MODEL_PATH_ARGS[@]}" \
    --num-answers "${NUM_ANSWERS}" \
    --out-dir "${INIT_DIR}" \
    "${PROBLEMS_ARGS[@]}" \
    --parallel "${PARALLEL}" \
    --save-logprobs \
    --diffusion-batch-size 1 \
    --diffusion-steps "${STEPS}" \
    --diffusion-gen-length "${GEN_LENGTH}" \
    --diffusion-temperature "${TEMPERATURE}" \
    --diffusion-alg "${DIFFUSION_ALG}" \
    --diffusion-alg-temp "${ALG_TEMP}" \
    --diffusion-capture-tif \
    --tif-snap-stride "${TIF_STRIDE}"
else
  echo "[pipeline] TiF: OFF  (set CAPTURE_TIF=1 to also save *_tif.npz on init)."
  uv run python generate_initial_answers.py \
    --dataset "${DATASET}" \
    --model-name "${MODEL}" \
    "${MODEL_PATH_ARGS[@]}" \
    --num-answers "${NUM_ANSWERS}" \
    --out-dir "${INIT_DIR}" \
    "${PROBLEMS_ARGS[@]}" \
    --parallel "${PARALLEL}" \
    --save-logprobs \
    --diffusion-batch-size 1 \
    --diffusion-steps "${STEPS}" \
    --diffusion-gen-length "${GEN_LENGTH}" \
    --diffusion-temperature "${TEMPERATURE}" \
    --diffusion-alg "${DIFFUSION_ALG}" \
    --diffusion-alg-temp "${ALG_TEMP}"
fi

echo "[pipeline] Phase B: multi-cut regen -> ${REGEN_DIR}  (answers=${REGEN_NUM_ANSWERS} cuts=${CUTS} gen_length=${GEN_LENGTH} steps=${STEPS} T=${REGEN_TEMP} alg=${REGEN_ALG} alg_temp=${REGEN_ALG_TEMP})"
uv run python "${SCRIPT_DIR}/regen_dream_multi_cut.py" \
  --init-dir "${INIT_DIR}" \
  --out-dir "${REGEN_DIR}" \
  --dataset "${DATASET}" \
  --model-name "${MODEL}" \
  "${MODEL_PATH_ARGS[@]}" \
  "${PROBLEMS_ARGS[@]}" \
  --cuts "${CUTS}" \
  --num-answers "${REGEN_NUM_ANSWERS}" \
  --gen-length "${GEN_LENGTH}" \
  --steps "${STEPS}" \
  --temperature "${REGEN_TEMP}" \
  --alg "${REGEN_ALG}" \
  --alg-temp "${REGEN_ALG_TEMP}" \
  --top-p "${REGEN_TOP_P}" \
  --top-k "${REGEN_TOP_K}"

echo "[pipeline] done."
