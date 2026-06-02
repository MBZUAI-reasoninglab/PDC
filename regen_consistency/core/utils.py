"""
regen_consistency/utils.py

Shared utilities for generate_initial_answers.py and the diffusion pipelines.

API calls, prompt rendering, file I/O, CoT parsing, logprobs.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from openai import OpenAI


def hf_pretrained_local_kw(model_path: str) -> Dict[str, bool]:
    """Extra kwargs for ``transformers`` when ``model_path`` is a local filesystem path.

    Recent ``huggingface_hub`` validates repo ids and rejects absolute paths unless
    ``local_files_only=True``. We key off *absolute* paths (not ``is_dir()``): on some
    batch nodes the model tree is visible only after automount latency, and ``is_dir()``
    would skip the flag and reproduce HFValidationError.
    """
    try:
        if Path(model_path).expanduser().is_absolute():
            return {"local_files_only": True}
    except OSError:
        pass
    return {}


# =====================================================================
# Data loading
# =====================================================================
def load_problems(data_file: str) -> List[Dict]:
    """Load problems from a JSONL file."""
    problems = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                problems.append(json.loads(line))
    print(f"  Loaded {data_file}: {len(problems)} problems")
    return problems


def create_prompt(problem: Dict, dataset_cfg: Dict) -> str:
    question_key = dataset_cfg["question_key"]
    return problem[question_key] + dataset_cfg["prompt_suffix"]


# =====================================================================
# Metadata
# =====================================================================
def write_metadata(out_dir: Path, dataset: str, model_name: str, **extra):
    """Write metadata.json to out_dir.

    If the file already exists, check that existing values are consistent
    with the new values. Raises ValueError on conflict.

    Pass any experiment-specific parameters via **extra so that runs are
    reproducible from metadata alone.  Examples:
        reasoning_effort, num_answers, insert_cot_closing, remove_pct, regen_count
    """
    meta_path = out_dir / "metadata.json"
    new_meta = {"dataset": dataset, "model": model_name}
    new_meta.update(extra)

    if meta_path.exists():
        existing = json.loads(meta_path.read_text())
        # regen_count naturally increases when extending experiments
        SKIP_KEYS = {"regen_count"}
        conflicts = {
            k: (existing[k], new_meta[k])
            for k in new_meta
            if k in existing and existing[k] != new_meta[k]
            and k not in SKIP_KEYS
        }
        if conflicts:
            detail = ", ".join(f"{k}: {old!r} vs {new!r}"
                               for k, (old, new) in conflicts.items())
            raise ValueError(
                f"metadata.json conflict in {out_dir}: {detail}. "
                f"Use a different output directory for different settings."
            )
        existing.update(new_meta)
        new_meta = existing

    meta_path.write_text(json.dumps(new_meta, indent=2, ensure_ascii=False) + "\n")


# =====================================================================
# Prompt rendering
# =====================================================================
def build_rendered_prompt(
    problem: Dict, dataset_cfg: Dict, model_cfg: Dict, tokenizer,
) -> Tuple[str, str]:
    prompt_text = create_prompt(problem, dataset_cfg)
    messages = [
        {"role": "system", "content": dataset_cfg["system_prompt"]},
        {"role": "user", "content": prompt_text},
    ]
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        **model_cfg["template_kwargs"],
    )
    return rendered, prompt_text


# =====================================================================
# Token budget helpers
# =====================================================================
def clamp_max_tokens(max_tokens: int, max_context_length: int, prompt: str, tokenizer) -> int:
    """Clamp max_tokens so that prompt + generation fits within the model context."""
    prompt_tokens = len(tokenizer.encode(prompt))
    return min(max_tokens, max_context_length - prompt_tokens - 256)


# =====================================================================
# vLLM completions API call
# =====================================================================
_MAX_RETRIES = 3


def call_completions(
    client: OpenAI, model_name: str, prompt: str, max_tokens: int,
    max_context_length: int, api_params: Dict,
    tokenizer, top_logprobs: int = 0,
) -> Tuple[str, int, Optional[list], Optional[list]]:
    """Call vLLM completions API.

    Returns (text, completion_tokens, top_logprobs_list, token_logprobs_list).
    top_logprobs_list is a list of dicts (one per token) when top_logprobs > 0,
    or None when top_logprobs == 0.
    token_logprobs_list is a list of floats (logprob of each generated token),
    or None when top_logprobs == 0.
    """
    max_tokens = clamp_max_tokens(max_tokens, max_context_length, prompt, tokenizer)
    kwargs = dict(model=model_name, prompt=prompt, max_tokens=max_tokens, **api_params)
    if top_logprobs > 0:
        kwargs["logprobs"] = top_logprobs
    for attempt in range(_MAX_RETRIES):
        try:
            resp = client.completions.create(**kwargs)
            break
        except Exception as e:
            if attempt < _MAX_RETRIES - 1:
                wait = 2 ** attempt
                print(f"  Retry {attempt + 1}/{_MAX_RETRIES} after {wait}s: {e}")
                time.sleep(wait)
            else:
                raise
    if not resp or not hasattr(resp, "choices") or not resp.choices:
        raise ValueError("Invalid API response")
    choice = resp.choices[0]
    if hasattr(choice, "text"):
        text = choice.text
    elif hasattr(choice, "message") and choice.message:
        text = choice.message.content
    else:
        raise ValueError(f"Cannot extract answer: {choice}")
    usage = getattr(resp, "usage", None)
    tokens = usage.completion_tokens if usage else 0

    raw_top_logprobs = None
    token_logprobs = None
    if top_logprobs > 0 and hasattr(choice, "logprobs") and choice.logprobs:
        raw_top_logprobs = choice.logprobs.top_logprobs
        if hasattr(choice.logprobs, "token_logprobs") and choice.logprobs.token_logprobs:
            token_logprobs = list(choice.logprobs.token_logprobs)
    return text, tokens, raw_top_logprobs, token_logprobs


def call_completions_echo(
    client: OpenAI, model_name: str, prompt: str, top_logprobs: int = 20,
):
    """Call vLLM completions API in echo mode (max_tokens=0, echo=True).

    Returns the logprobs object from the response choice.
    """
    for attempt in range(_MAX_RETRIES):
        try:
            resp = client.completions.create(
                model=model_name,
                prompt=prompt,
                max_tokens=0,
                temperature=0,
                logprobs=top_logprobs,
                echo=True,
            )
            return resp.choices[0].logprobs
        except Exception as e:
            if attempt < _MAX_RETRIES - 1:
                wait = 2 ** attempt
                print(f"  Retry {attempt + 1}/{_MAX_RETRIES} after {wait}s: {e}")
                time.sleep(wait)
            else:
                raise


def call_chat_completions(
    client: OpenAI, model_name: str, messages: List[Dict],
    max_tokens: int, max_context_length: int, api_params: Dict,
    tokenizer,
    extra_body: Optional[Dict] = None,
) -> str:
    prompt_text = " ".join(m["content"] for m in messages)
    max_tokens = clamp_max_tokens(max_tokens, max_context_length, prompt_text, tokenizer)
    kwargs = dict(model=model_name, messages=messages, max_tokens=max_tokens,
                  stream=False, **api_params)
    if extra_body:
        kwargs["extra_body"] = {**kwargs.get("extra_body", {}), **extra_body}
    for attempt in range(_MAX_RETRIES):
        try:
            resp = client.chat.completions.create(**kwargs)
            break
        except Exception as e:
            if attempt < _MAX_RETRIES - 1:
                wait = 2 ** attempt
                print(f"  Retry {attempt + 1}/{_MAX_RETRIES} after {wait}s: {e}")
                time.sleep(wait)
            else:
                raise
    if not resp or not resp.choices:
        raise ValueError("Invalid chat API response")
    return resp.choices[0].message.content


# =====================================================================
# CoT boundary helpers
# =====================================================================
def get_cot_to_final(model_cfg: Dict) -> Optional[str]:
    """Build the tag sequence that separates CoT from the final answer.

    For gpt-oss: cot_suffix (<|end|>) + final_prefix.
    For qwen3:   cot_suffix (</think>).
    For no-think models (cot_suffix is None): returns None.
    """
    if model_cfg["cot_suffix"] is None:
        return None
    if model_cfg["final_prefix"] is not None:
        return model_cfg["cot_suffix"] + model_cfg["final_prefix"]
    return model_cfg["cot_suffix"]


def split_cot_and_final(
    text: str, model_cfg: Dict, start_from_cot: bool = True,
) -> Tuple[Optional[str], Optional[str]]:
    """Split generated text into CoT content and Final content, excluding delimiters.

    Returns (cot_content, final_content):
      - Normal:    ("CoT text", "Final text")
      - Truncated: ("unfinished CoT text", None)   -- cot_suffix not found
      - No-think:  (None, "Final text")             -- model has no CoT phase

    *start_from_cot*: if False, treat the text as purely final answer when
    the CoT-to-final delimiter is not found (instead of assuming truncated CoT).
    Use this for continuations that are known to start from the final answer
    phase (e.g. truncation preserved the CoT boundary, or --insert-cot-closing was used).

    Works regardless of whether cot_prefix appears in the text (completions
    API may or may not include it).
    """
    cot_to_final = get_cot_to_final(model_cfg)

    # No-think model: entire text is final answer
    if cot_to_final is None:
        return None, text

    cot_prefix = model_cfg["cot_prefix"]

    # Skip cot_prefix if present, otherwise start from beginning
    prefix_pos = text.find(cot_prefix)
    after_prefix = text[prefix_pos + len(cot_prefix):] if prefix_pos != -1 else text

    # Find cot_to_final marker and split
    #   gpt-oss:
    #     <|channel|>analysis<|message|>...CoT...<|end|><|start|>assistant<|channel|>final<|message|>...
    #     cot_prefix + CoT + cot_to_final + Final
    #   qwen3 / qwen3.5:
    #     <think>...CoT...</think>\n\nThe answer is \boxed{42}.
    #     cot_prefix + CoT + cot_to_final + Final
    marker_pos = after_prefix.find(cot_to_final)
    if marker_pos == -1:
        if start_from_cot:
            # Truncated: generation ended before cot_suffix
            return after_prefix, None
        else:
            # No CoT in this text, entire content is final answer
            return None, text

    cot_content = after_prefix[:marker_pos]
    final_content = after_prefix[marker_pos + len(cot_to_final):]
    return cot_content, final_content


def count_cot_and_final_tokens(
    text: str, model_cfg: Dict, tokenizer, start_from_cot: bool = True,
) -> Tuple[Optional[int], Optional[int]]:
    """Count CoT and Final tokens (delimiter-excluded) in generated text.

    Returns (cot_tokens, final_tokens). Both are None when cot_suffix is
    not found (e.g. truncated generation).

    See split_cot_and_final() for the meaning of *start_from_cot*.
    """
    cot_content, final_content = split_cot_and_final(text, model_cfg, start_from_cot)
    cot_tokens = None
    final_tokens = None
    if cot_content is not None:
        cot_tokens = len(tokenizer.encode(cot_content, add_special_tokens=False))
    if final_content is not None:
        final_tokens = len(tokenizer.encode(final_content, add_special_tokens=False))
    return cot_tokens, final_tokens


# =====================================================================
# File reading
# =====================================================================
def read_answer_text(file_path: Path) -> str:
    """Read a saved answer file and return the content after the === delimiter."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    delim = "=" * 50
    pos = content.find(delim)
    if pos == -1:
        return content
    start = pos + len(delim)
    if start < len(content) and content[start] == "\n":
        start += 1
    return content[start:]


# =====================================================================
# File save
# =====================================================================
def save_answer(
    file_path: Path,
    dataset: str,
    problem_index: int,
    answer_index: int,
    gold_answer: str,
    prompt: str,
    answer: str,
    completion_tokens: int,
    cot_tokens: Optional[int] = None,
    final_tokens: Optional[int] = None,
    extra_headers: Optional[Dict[str, object]] = None,
):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    header_lines = [
        f"Dataset: {dataset}",
        f"Problem Number: {problem_index}",
        f"Answer Index: {answer_index}",
        f"Gold Answer: {gold_answer}",
        f"Generated Tokens: {completion_tokens}",
    ]
    if cot_tokens is not None:
        header_lines.append(f"CoT Tokens: {cot_tokens}")
    if final_tokens is not None:
        header_lines.append(f"Final Tokens: {final_tokens}")
    if extra_headers:
        for key, val in extra_headers.items():
            header_lines.append(f"{key}: {val}")
    header_lines.append(f"Prompt: {prompt}")
    header = "\n".join(header_lines)
    content = f"{header}\n{'=' * 50}\n{answer}"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)


# =====================================================================
# Logprobs
# =====================================================================
def logprobs_to_array(raw_top_logprobs: list, top_k: int) -> np.ndarray:
    """Convert vLLM top_logprobs (list of dicts) to numpy array (T, top_k)."""
    rows = []
    for top_lp_dict in raw_top_logprobs:
        if top_lp_dict is None:
            rows.append([float("-inf")] * top_k)
            continue
        d = dict(top_lp_dict)
        values = sorted(d.values(), reverse=True)
        if len(values) < top_k:
            values.extend([float("-inf")] * (top_k - len(values)))
        rows.append(values[:top_k])
    return np.array(rows, dtype=np.float32)


def save_logprobs_npy(npy_path: Path, raw_top_logprobs: list,
                      top_k: int) -> None:
    """Convert vLLM top_logprobs to numpy array and save as .npy."""
    arr = logprobs_to_array(raw_top_logprobs, top_k)
    np.save(str(npy_path), arr)


TOKEN_LOGPROBS_SUFFIX = ".tok_logprobs.npy"
UNMASK_STEP_SUFFIX = ".unmask_step.npy"
TRAJECTORY_SUFFIX = ".traj.npy"


def save_token_logprobs_npy(txt_path: Path, token_logprobs: list) -> None:
    """Save per-token logprobs (of actually generated tokens) as 1D .npy."""
    arr = np.array(token_logprobs, dtype=np.float64)
    np.save(str(txt_path) + TOKEN_LOGPROBS_SUFFIX, arr)


def save_unmask_step_npy(txt_path: Path, unmask_step) -> None:
    """Save per-position first-unmask step as 1D int32 .npy.

    Diffusion-only. Shape (gen_length,). For position i, value = first
    denoise step at which the model unmasked that position. Positions that
    remained masked at the end carry the value `steps` (= total step count).
    Used by (B) step-wise confidence, (C) unmask-order ranking,
    (D) step-stopping SC analyses.
    """
    arr = np.asarray(unmask_step, dtype=np.int32)
    np.save(str(txt_path) + UNMASK_STEP_SUFFIX, arr)


def save_trajectory_npy(txt_path: Path, trajectory) -> None:
    """Save full denoise trajectory as 2D int32 .npy.

    Diffusion-only. Shape (steps, gen_length). Cell (s, i) holds the token
    id at position i after denoise step s. Used by (E) trajectory-diversity
    weighting, and as raw data for arbitrary step-level analyses.
    """
    arr = np.asarray(trajectory, dtype=np.int32)
    np.save(str(txt_path) + TRAJECTORY_SUFFIX, arr)


# =====================================================================
# Utilities
# =====================================================================
def file_prefix(dataset: str, model_name: str) -> str:
    """Build filename prefix from dataset and model name."""
    dataset_stem = dataset.replace("/", "__")
    return f"{dataset_stem}_{model_name}"


def parse_int_list(s: Optional[str]) -> Optional[List[int]]:
    if s is None:
        return None
    result = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            result.extend(range(int(a.strip()), int(b.strip()) + 1))
        else:
            result.append(int(part))
    return sorted(set(result))


# =====================================================================
# .txt header I/O
# =====================================================================
def parse_header(file_path: Path) -> Dict[str, str]:
    """Parse all key-value pairs from a .txt file header (before === separator)."""
    header = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("=" * 10):
                break
            if ":" in line:
                key, _, val = line.partition(":")
                header[key.strip()] = val.strip()
    return header


def has_header_field(file_path: Path, key: str) -> bool:
    """Check if a .txt file header contains a given key."""
    prefix = f"{key}:"
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith(prefix):
                return True
            if line.startswith("=" * 10):
                return False
    return False


def write_header_fields(file_path: Path, fields: Dict[str, str]) -> None:
    """Insert key-value pairs into a .txt file header.

    Fields are inserted before the "Prompt:" line if present (keeping Prompt
    as the last header before the === separator). Falls back to inserting
    before the === separator if no Prompt line exists.
    """
    lines = file_path.read_text(encoding="utf-8").splitlines(keepends=True)
    new_lines = []
    inserted = False
    for line in lines:
        if not inserted and (line.startswith("Prompt:") or line.startswith("=" * 10)):
            for key, value in fields.items():
                new_lines.append(f"{key}: {value}\n")
            inserted = True
        new_lines.append(line)
    file_path.write_text("".join(new_lines), encoding="utf-8")
