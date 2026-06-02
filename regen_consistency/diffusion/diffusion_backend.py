"""
regen_consistency/diffusion/diffusion_backend.py

Common interface for the diffusion backends used by the generation and
regeneration pipeline scripts.

The two public functions return the same tuple shape as
:func:`core.utils.call_completions`::

    (text, completion_tokens, raw_top_logprobs, token_logprobs)

so the caller passes the result straight through to ``save_answer`` and
``save_logprobs_npy``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from core.utils import hf_pretrained_local_kw


# =====================================================================
# Lazy imports so importing this module never forces torch / transformers
# (the autoregressive code path does not need them).
# =====================================================================
def _get_dream(model_path: str):
    from .dream_engine import DreamEngine
    return DreamEngine.get(model_path)


def _get_llada_hf(model_path: str):
    from .llada_engine import LLaDAEngine
    return LLaDAEngine.get(model_path)


# =====================================================================
# Helpers
# =====================================================================
def _pick_backend(model_info: Dict) -> str:
    """Resolve the concrete backend name.

    diffusion-dream    -> Dream HF direct
    diffusion-llada    -> LLaDA HF direct
    """
    backend = model_info.get("backend", "vllm")
    if backend == "hf-direct":
        return "dream-hf"
    if backend == "llada-hf":
        return "llada-hf"
    raise ValueError(f"Unsupported diffusion backend: {backend}")


def _hyperparams(model_info: Dict, api_params: Dict, override_steps: Optional[int] = None,
                 override_gen_length: Optional[int] = None) -> Dict:
    """Pull diffusion-specific knobs from model_info / api_params."""
    extra_body = api_params.get("extra_body", {}) or {}
    top_k = extra_body.get("top_k", 0) or 0
    return dict(
        gen_length=override_gen_length or model_info["gen_length"],
        steps=override_steps or model_info["denoise_steps"],
        block_length=model_info["block_length"],
        temperature=api_params.get("temperature", 0.0),
        top_p=api_params.get("top_p", 1.0),
        top_k=top_k,
        cfg_scale=model_info.get("cfg_scale", 0.0),
        remasking=model_info.get("remasking", "low_confidence"),
        alg=model_info.get("alg", "entropy"),
        alg_temp=model_info.get("alg_temp", 0.0),
    )


# =====================================================================
# Initial generation
# =====================================================================
def do_generate_diffusion(
    *,
    model_name: str,
    model_info: Dict,
    api_params: Dict,
    prompt: str,
    save_top_logprobs: int = 0,
    save_trajectory: bool = False,
    save_unmask_step: bool = True,
    seed: Optional[int] = None,
    tif_npz_path: Optional[object] = None,
    tif_snap_stride: int = 8,
) -> Tuple[str, int,
           Optional[List[Dict[int, float]]],
           Optional[List[float]],
           Optional[object],   # unmask_step  (gen_length,) int32 ndarray
           Optional[object]]:  # trajectory   (steps, gen_length) int32 ndarray
    """Initial answer generation.

    ``seed`` is forwarded to backends that support deterministic per-call
    seeding (currently dream-hf and llada-hf). It is essential for
    self-consistency: without it, repeated calls share the global RNG
    state and Dream/LLaDA collapse to identical samples (see
    DreamEngine.generate docstring for details).

    Returns a 6-tuple (text, completion_tokens, top_logprobs, tok_logprobs,
    unmask_step, trajectory). The last two are populated by the HF-direct
    diffusion backends that support trajectory recording (Dream and LLaDA).
    """
    backend = _pick_backend(model_info)
    hp = _hyperparams(model_info, api_params)
    if tif_npz_path is not None and backend not in ("dream-hf", "llada-hf"):
        tif_npz_path = None

    if backend == "dream-hf":
        eng = _get_dream(model_info["default_model_path"])
        gen_kw = dict(
            prompt_text=prompt,
            gen_length=hp["gen_length"],
            steps=hp["steps"],
            temperature=hp["temperature"],
            top_p=hp["top_p"],
            top_k=hp["top_k"],
            alg=hp["alg"],
            alg_temp=hp["alg_temp"],
            save_top_logprobs=save_top_logprobs,
            save_trajectory=save_trajectory,
            save_unmask_step=save_unmask_step,
            seed=seed,
        )
        if tif_npz_path is not None:
            gen_kw["tif_out"] = tif_npz_path
            gen_kw["tif_snap_stride"] = tif_snap_stride
        out = eng.generate(**gen_kw)
    elif backend == "llada-hf":
        eng = _get_llada_hf(model_info["default_model_path"])
        # LLaDA's HF engine accepts seed when available; pass via kwargs so
        # older signatures without `seed` remain compatible.
        llada_kwargs = dict(
            prompt_text=prompt,
            gen_length=hp["gen_length"],
            steps=hp["steps"],
            block_length=hp["block_length"],
            temperature=hp["temperature"],
            cfg_scale=hp["cfg_scale"],
            remasking=hp["remasking"],
            save_top_logprobs=save_top_logprobs,
            save_trajectory=save_trajectory,
            save_unmask_step=save_unmask_step,
            tif_out=tif_npz_path,
            tif_snap_stride=tif_snap_stride,
        )
        if seed is not None:
            llada_kwargs["seed"] = seed
        out = eng.generate(**llada_kwargs)
    else:
        raise ValueError(f"Unknown diffusion backend: {backend}")

    raw_top = out.top_logprobs if save_top_logprobs > 0 else None
    raw_tok = out.token_logprobs if save_top_logprobs > 0 else None
    unmask_step = getattr(out, "unmask_step", None)
    trajectory = getattr(out, "trajectory", None)
    return (out.text, out.completion_tokens, raw_top, raw_tok,
            unmask_step, trajectory)


# =====================================================================
# Batched initial generation (same prompt, many seeds in one forward)
# =====================================================================
def do_generate_diffusion_batch(
    *,
    model_name: str,
    model_info: Dict,
    api_params: Dict,
    prompt: str,
    batch_size: int,
    seed: Optional[int] = None,
    save_top_logprobs: int = 0,
    save_trajectory: bool = False,
    save_unmask_step: bool = True,
) -> List[Tuple[str, int,
                Optional[List[Dict[int, float]]],
                Optional[List[float]],
                Optional[object],
                Optional[object]]]:
    """Run ``batch_size`` independent samples of the same prompt in one
    GPU forward stack.

    Currently only the ``dream-hf`` backend supports true batched generation;
    other backends fall back to sequential ``do_generate_diffusion`` calls
    so the API stays uniform.

    Returns a list of length ``batch_size``, each element being the same
    6-tuple as :func:`do_generate_diffusion`.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    backend = _pick_backend(model_info)
    hp = _hyperparams(model_info, api_params)

    if backend == "dream-hf":
        eng = _get_dream(model_info["default_model_path"])
        outs = eng.generate_batch(
            prompt_text=prompt,
            gen_length=hp["gen_length"],
            steps=hp["steps"],
            batch_size=batch_size,
            seed=seed,
            temperature=hp["temperature"],
            top_p=hp["top_p"],
            top_k=hp["top_k"],
            alg=hp["alg"],
            alg_temp=hp["alg_temp"],
            save_top_logprobs=save_top_logprobs,
            save_trajectory=save_trajectory,
            save_unmask_step=save_unmask_step,
        )
        results = []
        for out in outs:
            raw_top = out.top_logprobs if save_top_logprobs > 0 else None
            raw_tok = out.token_logprobs if save_top_logprobs > 0 else None
            results.append((
                out.text, out.completion_tokens, raw_top, raw_tok,
                getattr(out, "unmask_step", None),
                getattr(out, "trajectory", None),
            ))
        return results

    # ---- Fallback for non-dream backends: loop sequentially. We bump the
    #      seed by ``i`` so each call still gets a distinct RNG state.
    fallback = []
    for i in range(batch_size):
        sub_seed = (seed + i) if seed is not None else None
        fallback.append(do_generate_diffusion(
            model_name=model_name,
            model_info=model_info,
            api_params=api_params,
            prompt=prompt,
            save_top_logprobs=save_top_logprobs,
            save_trajectory=save_trajectory,
            save_unmask_step=save_unmask_step,
            seed=sub_seed,
        ))
    return fallback


# =====================================================================
# Regeneration (truncate-and-resample)
# =====================================================================
def do_regen_diffusion(
    *,
    model_name: str,
    model_info: Dict,
    api_params: Dict,
    prompt: str,
    source_text: str,
    diffusion_mode: str,            # "position" | "block"
    remove_pct: Optional[int],      # 0..100, fraction of generated tokens to drop
    seed: int = 0,
    save_top_logprobs: int = 0,
) -> Tuple[str, int, Optional[List[Dict[int, float]]], Optional[List[float]]]:
    """Regenerate after truncating ``source_text`` by ``remove_pct``.

    For ``position``/``block`` modes, the kept portion is the leading
    ``(100 - remove_pct)%`` of the source's generated tokens.
    """
    if diffusion_mode not in ("position", "block"):
        raise ValueError(f"diffusion-mode must be one of position|block, got {diffusion_mode}")

    backend = _pick_backend(model_info)
    hp = _hyperparams(model_info, api_params)
    gen_length = hp["gen_length"]
    block_length = hp["block_length"]

    # Compute kept-token / kept-block against the actual saved source length,
    # not the fixed diffusion mask budget. Many diffusion generations terminate
    # before gen_length via EOS, so gen_length-based truncation would keep too
    # much of short answers.
    pct = remove_pct if remove_pct is not None else 50
    pct = max(0, min(100, pct))
    keep_pct = (100 - pct) / 100.0
    tok = _get_tokenizer(model_info["default_model_path"])
    source_ids = tok.encode(source_text, add_special_tokens=False)
    source_token_count = len(source_ids)
    desired_kept_tokens = _kept_token_count_for_pct(source_token_count, keep_pct)

    if diffusion_mode == "position":
        kept_token_count = desired_kept_tokens
        kept_text = _decode_prefix_tokens(tok, source_text, source_ids, kept_token_count)
    else:  # block
        num_blocks = max(1, gen_length // block_length)
        source_blocks = ((source_token_count + block_length - 1) // block_length
                         if source_token_count > 0 else 0)
        if desired_kept_tokens <= 0 or source_blocks <= 0:
            kept_block_count = 0
            kept_token_count = 0
        else:
            kept_block_count = max(1, desired_kept_tokens // block_length)
            kept_block_count = min(kept_block_count, source_blocks, num_blocks)
            kept_token_count = min(source_token_count,
                                   kept_block_count * block_length)
        kept_text = _decode_prefix_tokens(tok, source_text, source_ids, kept_token_count)

    if backend == "dream-hf":
        eng = _get_dream(model_info["default_model_path"])
        if diffusion_mode == "block":
            out = eng.regen_block(
                prompt_text=prompt,
                kept_text=kept_text,
                gen_length=gen_length,
                block_length=block_length,
                kept_block_count=kept_block_count,
                steps=hp["steps"],
                temperature=hp["temperature"],
                top_p=hp["top_p"],
                top_k=hp["top_k"],
                alg=hp["alg"],
                alg_temp=hp["alg_temp"],
                save_top_logprobs=save_top_logprobs,
                kept_token_ids=source_ids[:kept_token_count],
            )
        else:
            out = eng.regen_position(
                prompt_text=prompt,
                kept_text=kept_text,
                gen_length=gen_length,
                kept_token_count=kept_token_count,
                steps=hp["steps"],
                temperature=hp["temperature"],
                top_p=hp["top_p"],
                top_k=hp["top_k"],
                alg=hp["alg"],
                alg_temp=hp["alg_temp"],
                save_top_logprobs=save_top_logprobs,
                kept_token_ids=source_ids[:kept_token_count],
            )

    elif backend == "llada-hf":
        eng = _get_llada_hf(model_info["default_model_path"])
        common = dict(
            prompt_text=prompt, gen_length=gen_length,
            steps=hp["steps"], block_length=block_length,
            temperature=hp["temperature"], cfg_scale=hp["cfg_scale"],
            remasking=hp["remasking"], save_top_logprobs=save_top_logprobs,
            seed=seed,
            proportional_regen_steps=True,
        )
        if diffusion_mode == "block":
            out = eng.regen_block(kept_text=kept_text,
                                  kept_block_count=kept_block_count,
                                  kept_token_ids=source_ids[:kept_token_count],
                                  **common)
        else:
            out = eng.regen_position(kept_text=kept_text,
                                     kept_token_count=kept_token_count,
                                     kept_token_ids=source_ids[:kept_token_count],
                                     **common)

    else:
        raise ValueError(f"Unknown diffusion backend: {backend}")

    raw_top = out.top_logprobs if save_top_logprobs > 0 else None
    raw_tok = out.token_logprobs if save_top_logprobs > 0 else None
    return out.text, out.completion_tokens, raw_top, raw_tok


# =====================================================================
# Tokenizer-aware text truncation
# =====================================================================
_TOKENIZER_CACHE: Dict[str, object] = {}


def _get_tokenizer(model_path: str):
    if model_path in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[model_path]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, **hf_pretrained_local_kw(model_path))
    _TOKENIZER_CACHE[model_path] = tok
    return tok


def _kept_token_count_for_pct(total_tokens: int, keep_pct: float) -> int:
    """Return leading-token count for a keep fraction over source tokens."""
    if total_tokens <= 0 or keep_pct <= 0:
        return 0
    return min(total_tokens, max(1, int(total_tokens * keep_pct)))


def _decode_prefix_tokens(tok, original_text: str, token_ids: List[int],
                          keep_tokens: int) -> str:
    """Decode the first ``keep_tokens`` ids, preserving exact full text."""
    if keep_tokens <= 0:
        return ""
    if len(token_ids) <= keep_tokens:
        return original_text
    return tok.decode(token_ids[:keep_tokens], skip_special_tokens=False)


def _truncate_text_by_tokens(model_info: Dict, text: str,
                             keep_tokens: int) -> str:
    """Decode the first ``keep_tokens`` tokens of ``text``."""
    tok = _get_tokenizer(model_info["default_model_path"])
    ids = tok.encode(text, add_special_tokens=False)
    return _decode_prefix_tokens(tok, text, ids, keep_tokens)
