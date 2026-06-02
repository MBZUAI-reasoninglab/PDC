"""
regen_consistency/config.py

Dataset and model configuration.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

# =====================================================================
# Model and dataset directories
# =====================================================================
# Auto-detected from repo location: models default next to repo; datasets under <repo>/datasets.
# Override with MODELS_DIR / DATASETS_DIR env vars to change locations.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PARENT = _REPO_ROOT.parent
MODELS_DIR = Path(os.environ.get("MODELS_DIR", str(_DEFAULT_PARENT / "models")))
DATASETS_DIR = Path(os.environ.get("DATASETS_DIR", str(_REPO_ROOT / "datasets")))

# =====================================================================
# Constants
# =====================================================================
BASE_URL = "http://localhost:8100/v1"
MIN_FILE_SIZE_BYTES = 2048  # Skip files >= 2KB (resumability)
TOP_LOGPROBS = 20  # vLLM max allowed value

# =====================================================================
# Dataset configs
# =====================================================================
_DATASET_CONFIGS = {
    # https://huggingface.co/datasets/MathArena/aime_2024_I + aime_2024_II
    "aime2024": {
        "data_file": str(DATASETS_DIR / "AIME2024" / "aime2024.jsonl"),
        "system_prompt": "You are a helpful assistant specialized in solving mathematical problems.",
        "prompt_suffix": "\n Please reason step by step, and put your final answer within \\boxed{}.",
        "complete_max_tokens": 100_000,
        "gold_key": "answer",
        "question_key": "problem",
        "llm_judge": False,
    },
    # https://huggingface.co/datasets/MathArena/aime_2025
    "aime2025": {
        "data_file": str(DATASETS_DIR / "AIME2025" / "aime2025.jsonl"),
        "system_prompt": "You are a helpful assistant specialized in solving mathematical problems.",
        "prompt_suffix": "\n Please reason step by step, and put your final answer within \\boxed{}.",
        "complete_max_tokens": 100_000,
        "gold_key": "answer",
        "question_key": "problem",
        "llm_judge": False,
    },
    # https://huggingface.co/datasets/MathArena/aime_2026
    "aime2026": {
        "data_file": str(DATASETS_DIR / "AIME2026" / "aime2026.jsonl"),
        "system_prompt": "You are a helpful assistant specialized in solving mathematical problems.",
        "prompt_suffix": "\n Please reason step by step, and put your final answer within \\boxed{}.",
        "complete_max_tokens": 100_000,
        "gold_key": "answer",
        "question_key": "problem",
        "llm_judge": False,
    },
    # https://huggingface.co/datasets/HuggingFaceH4/MATH-500
    "math500": {
        "data_file": str(DATASETS_DIR / "MATH500" / "math500.jsonl"),
        "system_prompt": "You are a helpful assistant specialized in solving mathematical problems.",
        "prompt_suffix": "\n Please reason step by step, and put your final answer within \\boxed{}.",
        "complete_max_tokens": 100_000,
        "gold_key": "answer",
        "question_key": "problem",
        "llm_judge": True,
    },
    # https://huggingface.co/datasets/openai/gsm8k (subset "main", split "test")
    "gsm8k": {
        "data_file": str(DATASETS_DIR / "GSM8K" / "gsm8k.jsonl"),
        "system_prompt": "You are a helpful assistant specialized in solving mathematical problems.",
        "prompt_suffix": "\n Please reason step by step, and put your final answer within \\boxed{}.",
        "complete_max_tokens": 100_000,
        "gold_key": "answer",
        "question_key": "question",
        "llm_judge": False,
    },
    # https://huggingface.co/datasets/ChilleD/SVAMP
    "svamp": {
        "data_file": str(DATASETS_DIR / "SVAMP" / "svamp.jsonl"),
        "system_prompt": "You are a helpful assistant specialized in solving mathematical word problems.",
        "prompt_suffix": "\n Please reason step by step, and put your final numeric answer within \\boxed{}.",
        "complete_max_tokens": 100_000,
        "gold_key": "answer",
        "question_key": "question",
        "llm_judge": False,
    },
    # https://huggingface.co/datasets/fingertap/GPQA-Diamond (from Idavidrein/gpqa)
    "gpqa_diamond": {
        "data_file": str(DATASETS_DIR / "GPQA-Diamond" / "gpqa_diamond.jsonl"),
        "system_prompt": "You are a helpful assistant answering advanced science multiple choice questions.",
        "prompt_suffix": "\n Please reason step by step, and put your final answer as the letter choice (A), (B), (C), etc. within \\boxed{}.",
        "complete_max_tokens": 50_000,
        "gold_key": "answer",
        "question_key": "question",
        "llm_judge": False,
    },
    # https://huggingface.co/datasets/openai/frontierscience (olympiad subset)
    "frontierscience_olympiad": {
        "data_file": str(DATASETS_DIR / "FrontierScience" / "frontierscience_olympiad.jsonl"),
        "system_prompt": "You are an expert scientist solving olympiad-level problems in physics, chemistry, and biology.",
        "prompt_suffix": "\n Please reason step by step, and put your final answer within \\boxed{}.",
        "complete_max_tokens": 100_000,
        "gold_key": "answer",
        "question_key": "problem",
        "llm_judge": True,
    },
    # https://huggingface.co/datasets/cais/hle (gated)
    "hle": {
        "data_file": str(DATASETS_DIR / "HLE" / "hle.jsonl"),
        "system_prompt": "You are a helpful assistant with expertise across many domains.",
        "prompt_suffix": (
            "\n\nPlease reason step by step.\n"
            "Your response should be in the following format:\n"
            "Explanation: {your explanation for your answer choice}\n"
            "Answer: \\boxed{{your chosen answer}}\n"
            "Confidence: {your confidence score between 0% and 100% for your answer}"
        ),
        "complete_max_tokens": 100_000,
        "gold_key": "answer",
        "question_key": "question",
        "llm_judge": True,
    },
    # https://huggingface.co/datasets/MathArena/hmmt_feb_2026
    "hmmt": {
        "data_file": str(DATASETS_DIR / "HMMT" / "hmmt.jsonl"),
        "system_prompt": "You are a helpful assistant specialized in solving mathematical problems.",
        "prompt_suffix": "\n Please reason step by step, and put your final answer within \\boxed{}.",
        "complete_max_tokens": 100_000,
        "gold_key": "answer",
        "question_key": "problem",
        "llm_judge": True,
    },
    # https://huggingface.co/datasets/MathArena/brumo_2025
    "brumo": {
        "data_file": str(DATASETS_DIR / "BRUMO" / "brumo.jsonl"),
        "system_prompt": "You are a helpful assistant specialized in solving mathematical problems.",
        "prompt_suffix": "\n Please reason step by step, and put your final answer within \\boxed{}.",
        "complete_max_tokens": 100_000,
        "gold_key": "answer",
        "question_key": "problem",
        "llm_judge": True,
    },
    # https://github.com/google-deepmind/superhuman/tree/main/imobench
    "imo_answerbench": {
        "data_file": str(DATASETS_DIR / "IMO-AnswerBench" / "imo_answerbench.jsonl"),
        "system_prompt": "You are a helpful assistant specialized in solving mathematical problems.",
        "prompt_suffix": "\n Please reason step by step, and put your final answer within \\boxed{}.",
        "complete_max_tokens": 100_000,
        "gold_key": "answer",
        "question_key": "problem",
        "llm_judge": True,
    },
    # https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro
    "mmlu_pro": {
        "data_file": str(DATASETS_DIR / "MMLU-Pro" / "mmlu_pro.jsonl"),
        "system_prompt": "You are a helpful assistant answering multiple choice questions.",
        "prompt_suffix": (
            "\n\nPlease reason step by step. "
            "Select exactly one option and put only its option letter in \\boxed{}."
        ),
        "complete_max_tokens": 100_000,
        "gold_key": "answer",
        "question_key": "question",
        "llm_judge": False,
    },
    # https://huggingface.co/datasets/tau/commonsense_qa
    # Use the validation split locally because CommonsenseQA's official test
    # split is not intended for gold-label evaluation.
    "tau/commonsense_qa": {
        "data_file": str(DATASETS_DIR / "CommonsenseQA" / "commonsense_qa.jsonl"),
        "system_prompt": (
            "You are a helpful assistant for CommonsenseQA-style multiple-choice questions. "
            "You MUST write explicit multi-step reasoning in full sentences before stating any conclusion. "
            "Do not output only the option letter or a one-line answer; that is invalid."
        ),
        "prompt_suffix": (
            "\n\nMandatory format (do not skip):\n"
            "1) Write your reasoning across several sentences (facts, definitions, or everyday knowledge you use).\n"
            "2) Only after that reasoning, end with exactly one line that contains only one capital option letter "
            "(A–E) inside \\boxed{}, e.g. \\boxed{C}.\n"
            "Do not put the option letter before your reasoning. Do not answer with only a bare letter (e.g. \"D\") "
            "or only a line like \"D. ...\" from the choices without the reasoning block above."
        ),
        "complete_max_tokens": 50_000,
        "gold_key": "answer",
        "question_key": "question",
        "llm_judge": False,
    },
    # https://huggingface.co/datasets/ChilleD/StrategyQA
    # Source labels are boolean; the local JSONL maps them to A. Yes / B. No.
    "strategy_qa": {
        "data_file": str(DATASETS_DIR / "StrategyQA" / "strategy_qa.jsonl"),
        "system_prompt": (
            "You are a helpful assistant for StrategyQA-style questions. "
            "You MUST write explicit multi-step reasoning in full sentences before stating any conclusion. "
            "Do not output only the option letter or a one-line verdict; that is invalid."
        ),
        "prompt_suffix": (
            "\n\nMandatory format (do not skip):\n"
            "1) Write your reasoning across several sentences (facts, definitions, or world knowledge you use).\n"
            "2) Only after that reasoning, end with exactly one line that contains only the option letter inside "
            "\\boxed{}, e.g. \\boxed{A} or \\boxed{B}.\n"
            "Do not put the option letter before your reasoning. Do not answer with only \"A\"/\"B\" or "
            "\"A. Yes\"/\"B. No\" without the reasoning block above."
        ),
        "complete_max_tokens": 50_000,
        "gold_key": "answer",
        "question_key": "question",
        "llm_judge": False,
    },
    # https://github.com/pfnet-research/medrect (Japanese, error samples only)
    "medrect": {
        "data_file": str(DATASETS_DIR / "MedRECT" / "medrect.jsonl"),
        # "You are a medical expert who reviews the accuracy of clinical texts."
        "system_prompt": "あなたは臨床テキストの正確性をレビューする医学専門家です。",
        # "The clinical text above contains exactly one medical error related to
        #  treatment, diagnosis, management, or causation. Reason step by step,
        #  and put only the erroneous sentence number within \boxed{}."
        "prompt_suffix": (
            "\n\n上記の臨床テキストには、治療・診断・管理・因果関係に関連する"
            "医学的エラーが正確に1つ含まれています。"
            "ステップバイステップで推論し、エラーのある文番号のみを \\boxed{} 内に入れてください。"
        ),
        "complete_max_tokens": 100_000,
        "gold_key": "answer",
        "question_key": "question",
        "llm_judge": False,
    },
}

_DATASET_CONFIGS["commonsense_qa"] = _DATASET_CONFIGS["tau/commonsense_qa"]
_DATASET_CONFIGS["strategyqa"] = _DATASET_CONFIGS["strategy_qa"]
_DATASET_CONFIGS["StrategyQA"] = _DATASET_CONFIGS["strategy_qa"]
_DATASET_CONFIGS["ChilleD/StrategyQA"] = _DATASET_CONFIGS["strategy_qa"]

DATASET_NAMES = list(_DATASET_CONFIGS.keys())


def get_dataset_config(dataset: str) -> Dict:
    """Return the dataset config dict for the given dataset name."""
    return _DATASET_CONFIGS[dataset]


# =====================================================================
# Model configs
# =====================================================================
_MODEL_TYPE_CONFIGS = {
    # --- OpenAI ---
    # GPT-OSS (custom CoT delimiters, reasoning_effort modes)
    "gpt-oss": {
        "cot_prefix": "<|channel|>analysis<|message|>",
        "cot_suffix": "<|end|>",
        "final_prefix": "<|start|>assistant<|channel|>final<|message|>",
        "max_context_length": 131072,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 40,
        "template_kwargs": {},
        "no_think_overrides": {},
    },
    # --- Qwen ---
    # Qwen3 dense (think/no-think dual mode)
    "qwen3": {
        "cot_prefix": "<think>",
        "cot_suffix": "</think>",
        "final_prefix": None,
        "max_context_length": 40960,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "template_kwargs": {},
        "no_think_overrides": {
            "temperature": 0.7,
            "top_p": 0.8,
        },
    },
    # Qwen3 MoE thinking-only
    "qwen3-moe-thinking": {
        "cot_prefix": "<think>",
        "cot_suffix": "</think>",
        "final_prefix": None,
        "max_context_length": 131072,  # native 256K, capped to 128K
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "template_kwargs": {},
        "no_think_overrides": {},
    },
    # Qwen3 MoE instruct-only (no thinking)
    "qwen3-moe-instruct": {
        "cot_prefix": None,
        "cot_suffix": None,
        "final_prefix": None,
        "max_context_length": 131072,  # native 256K, capped to 128K
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "template_kwargs": {},
        "no_think_overrides": {},
    },
    # Qwen3.5 dense (GDN architecture, thinking-only)
    "qwen3.5": {
        "cot_prefix": "<think>",
        "cot_suffix": "</think>",
        "final_prefix": None,
        "max_context_length": 131072,  # native 256K, capped to 128K
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "presence_penalty": 1.5,
        "template_kwargs": {},
        "no_think_overrides": {},
    },
    # --- DeepSeek ---
    # R1 distilled models (Qwen3 architecture, DeepSeek tokenizer)
    "deepseek-r1-distill": {
        "cot_prefix": "<think>",
        "cot_suffix": "</think>",
        "final_prefix": None,
        "max_context_length": 131072,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "template_kwargs": {},
        "no_think_overrides": {},
    },
    # --- Microsoft ---
    # Phi-4-reasoning (ChatML format)
    "phi-4-reasoning": {
        "cot_prefix": "<think>",
        "cot_suffix": "</think>",
        "final_prefix": None,
        "max_context_length": 32768,
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": 50,
        "template_kwargs": {},
        "no_think_overrides": {},
    },
    # --- Mistral ---
    # Ministral reasoning ([THINK]/[/THINK] delimiters)
    "ministral-reasoning": {
        "cot_prefix": "[THINK]",
        "cot_suffix": "[/THINK]",
        "final_prefix": None,
        "max_context_length": 131072,  # native 256K, capped to 128K
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 20,
        "template_kwargs": {},
        "no_think_overrides": {},
    },
    # --- NVIDIA ---
    # Nemotron-Nano-9B-v2 (Hybrid Mamba-2/Transformer, requires trust_remote_code)
    "nemotron-nano-v2": {
        "cot_prefix": "<think>",
        "cot_suffix": "</think>",
        "final_prefix": None,
        "max_context_length": 131072,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "template_kwargs": {},
        "no_think_overrides": {},
    },
    # Nemotron-3-Nano-30B (Hybrid Mamba-2/MoE/GQA, requires trust_remote_code)
    "nemotron-3-nano": {
        "cot_prefix": "<think>",
        "cot_suffix": "</think>",
        "final_prefix": None,
        "max_context_length": 131072,  # native 256K, capped to 128K
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 20,
        "template_kwargs": {},
        "no_think_overrides": {},
    },
    # Nemotron-3-Super (Hybrid Mamba-2/LatentMoE, requires trust_remote_code)
    "nemotron-3-super": {
        "cot_prefix": "<think>",
        "cot_suffix": "</think>",
        "final_prefix": None,
        "max_context_length": 131072,  # native 256K, capped to 128K
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "template_kwargs": {},
        "no_think_overrides": {},
    },
    # --- LG ---
    # EXAONE-Deep (<thought></thought> delimiters, requires trust_remote_code)
    "exaone-deep": {
        "cot_prefix": "<thought>",
        "cot_suffix": "</thought>",
        "final_prefix": None,
        "max_context_length": 32768,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "template_kwargs": {},
        "no_think_overrides": {},
    },
    # =================================================================
    # Diffusion LLMs
    # =================================================================
    # These are masked discrete diffusion models. They do NOT generate
    # autoregressively: they fill a fixed-length mask sequence by iterative
    # denoising. The "CoT" concept does not apply natively, so cot_prefix /
    # cot_suffix / final_prefix are all None and the entire generation is
    # treated as the final output (with \boxed{} parsed at the end).
    #
    # Extra keys (non-API):
    #   backend         : "hf-direct" (Dream) or "llada-hf" (LLaDA family)
    #   gen_length      : default mask sequence length
    #   block_length    : semi-AR block size (== gen_length means "full")
    #   denoise_steps   : number of denoising steps (NFE)
    #   remasking       : "low_confidence" or "random"
    #   cfg_scale       : classifier-free guidance scale (LLaDA only)
    #   alg             : Dream sampling algorithm ("entropy", "maskgit_plus",
    #                     "topk_margin")
    #   alg_temp        : temperature for the sampling algorithm
    #   mask_token      : explicit mask token string used for re-masking
    #                     in the position-remask regen mode
    # ---------------------------------------------------------------
    # Dream-v0 (HKUNLP, 7B). ChatML chat template; HF transformers direct.
    "diffusion-dream": {
        "cot_prefix": None,
        "cot_suffix": None,
        "final_prefix": None,
        "max_context_length": 4096,
        # SC-friendly defaults. Two empirical pitfalls observed on Dream:
        # (a) High temperature (>=0.6) causes EOS to be sampled at
        #     gen-position 0 because the early-unmask context is empty,
        #     collapsing 100% of samples to 1 token.
        # (b) alg_temp > 0 randomizes the unmask order which has the same
        #     effect (~50% truncation at alg_temp=0.3) by letting position
        #     0 be unmasked under empty surrounding context.
        # So we keep the original T=0.2 / alg_temp=0.0 settings and rely
        # ENTIRELY on a per-sample torch.manual_seed (set in generate_initial_answers.py)
        # to produce diverse samples. Even at T=0.2 the categorical sample
        # at each position uses the global RNG, so distinct seeds will
        # diverge after the first position whose top-2 candidate has
        # non-negligible probability.
        "temperature": 0.3,
        "top_p": 0.95,
        "top_k": None,
        "template_kwargs": {},
        "no_think_overrides": {},
        "backend": "hf-direct",
        "gen_length": 512,
        "block_length": 512,  # full = no semi-AR
        # Dream paper uses NFE=256 as the standard eval setting; quality vs
        # NFE=512 is essentially indistinguishable while cutting wall-time in half.
        "denoise_steps": 256,
        "remasking": "low_confidence",
        "cfg_scale": 0.0,
        # 'origin' = MaskGIT-style random commit schedule (each position is
        # independently committed with probability 1-s/t at step i). This is
        # the only Dream alg whose unmask order is RNG-driven, so per-sample
        # torch.manual_seed actually produces diverse commit patterns ->
        # diverse samples. 'entropy' (the paper default) gives the best
        # single-shot quality but its order is deterministic for a fixed
        # prompt, and the model is so confident (top-1 logprob ~ -1e-4) that
        # different seeds collapse to identical samples even at T=0.3-0.5.
        "alg": "origin",
        "alg_temp": 0.0,
        "mask_token": "<|mask|>",
    },
    # LLaDA family (GSAI-ML / inclusionAI). HF direct so TiF/unmask artifacts
    # are available from the in-process denoising loop.
    "diffusion-llada": {
        "cot_prefix": None,
        "cot_suffix": None,
        "final_prefix": None,
        "max_context_length": 4096,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": None,
        "template_kwargs": {},
        "no_think_overrides": {},
        "backend": "llada-hf",
        "gen_length": 512,
        "block_length": 32,  # LLaDA paper default
        "denoise_steps": 512,
        "remasking": "low_confidence",
        "cfg_scale": 0.0,
        "alg": None,
        "alg_temp": 0.0,
        "mask_token": "<|mdm_mask|>",  # token id 126336 in LLaDA tokenizer
    },
}

_MODEL_NAME_TO_TYPE = {
    "gpt-oss-20b": "gpt-oss",
    "gpt-oss-120b": "gpt-oss",
    "Qwen3-4B": "qwen3",
    "Qwen3-8B": "qwen3",
    "Qwen3-14B": "qwen3",
    "Qwen3-32B": "qwen3",
    "Qwen3.5-4B": "qwen3.5",
    "DeepSeek-R1-0528-Qwen3-8B": "deepseek-r1-distill",
    "Phi-4-reasoning": "phi-4-reasoning",
    "Qwen3-30B-A3B-Thinking-2507": "qwen3-moe-thinking",
    "Nemotron-3-Nano-30B-A3B": "nemotron-3-nano",
    "Nemotron-3-Super-120B-A12B": "nemotron-3-super",
    "Nemotron-Nano-9B-v2": "nemotron-nano-v2",
    "Ministral-3-8B-Reasoning-2512": "ministral-reasoning",
    "Ministral-3-14B-Reasoning-2512": "ministral-reasoning",
    "EXAONE-Deep-32B": "exaone-deep",
    "Qwen3-30B-A3B-Instruct-2507": "qwen3-moe-instruct",
    # --- Diffusion LLMs ---
    "Dream-v0-Instruct-7B": "diffusion-dream",
    "Dream-v0-Base-7B": "diffusion-dream",
    "LLaDA-1.5": "diffusion-llada",
    "LLaDA-8B-Instruct": "diffusion-llada",
    "LLaDA-8B-Base": "diffusion-llada",
    "LLaDA-MoE-7B-A1B-Instruct": "diffusion-llada",
    "LLaDA2.0-mini-preview": "diffusion-llada",
    "DiffuCoder-7B-cpGRPO": "diffusion-dream",
    "MMaDA-8B-MixCoT": "diffusion-llada",
}

_MODEL_CONFIGS = {
    name: {**_MODEL_TYPE_CONFIGS[model_type], "default_model_path": str(MODELS_DIR / name)}
    for name, model_type in _MODEL_NAME_TO_TYPE.items()
}

MODEL_NAMES = list(_MODEL_CONFIGS.keys())


# =====================================================================
# Model config builder
# =====================================================================

# Keys that go into api_params (used by call_completions / call_chat_completions)
_SAMPLING_KEYS = {"temperature", "top_p", "top_k", "presence_penalty"}

# Keys that are diffusion-only and surfaced into model_info
_DIFFUSION_KEYS = {
    "backend", "gen_length", "block_length", "denoise_steps",
    "remasking", "cfg_scale", "alg", "alg_temp", "mask_token",
}


def build_model_config(
    model_name: str,
    no_think: bool = False,
    reasoning_effort: Optional[str] = None,
) -> Tuple[Dict, Dict]:
    """Build model config with runtime overrides applied.

    Returns (model_info, api_params). Does not mutate _MODEL_CONFIGS.

    model_info: model metadata (paths, CoT delimiters, template_kwargs, max_context_length).
    api_params: pre-built kwargs for API calls. Spread directly into
        client.completions.create / client.chat.completions.create via **api_params.
    """
    base = _MODEL_CONFIGS[model_name]

    # Resolve sampling overrides
    sampling = {k: base[k] for k in _SAMPLING_KEYS if k in base}
    if no_think:
        for k, v in base["no_think_overrides"].items():
            sampling[k] = v

    # Build api_params (ready to unpack into API call kwargs)
    api_params = {
        "temperature": sampling["temperature"],
        "top_p": sampling["top_p"],
        "extra_body": {"skip_special_tokens": False, "top_k": sampling["top_k"]},
    }
    if "presence_penalty" in sampling:
        api_params["presence_penalty"] = sampling["presence_penalty"]

    # Build model_info (everything else)
    template_kwargs = dict(base["template_kwargs"])
    if no_think:
        template_kwargs["enable_thinking"] = False
    if reasoning_effort is not None:
        template_kwargs["reasoning_effort"] = reasoning_effort

    model_info = {
        "default_model_path": base["default_model_path"],
        "max_context_length": base["max_context_length"],
        "cot_prefix": None if no_think else base["cot_prefix"],
        "cot_suffix": None if no_think else base["cot_suffix"],
        "final_prefix": base["final_prefix"],
        "template_kwargs": template_kwargs,
        # Default backend is autoregressive ("vllm"); diffusion model types
        # override this via _DIFFUSION_KEYS below.
        "backend": "vllm",
    }
    # Forward diffusion-specific keys (if present on this model type)
    for k in _DIFFUSION_KEYS:
        if k in base:
            model_info[k] = base[k]

    return model_info, api_params
