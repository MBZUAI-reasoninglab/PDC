#!/usr/bin/env python3
"""
regen_consistency/generate_initial_answers.py

Initial answer generation via vLLM completions API.
For each problem, generate N answers and save them as individual files.

Usage:
    uv run python generate_initial_answers.py \
        --dataset aime2025 --model-name gpt-oss-20b \
        --num-answers 100 --out-dir regen_consistency/data/initial \
        --parallel 32
"""

import argparse
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from openai import OpenAI

from core.config import (
    BASE_URL, MIN_FILE_SIZE_BYTES, TOP_LOGPROBS,
    DATASET_NAMES, MODEL_NAMES,
    get_dataset_config, build_model_config,
)
from core.utils import (
    build_rendered_prompt, call_completions, count_cot_and_final_tokens, save_token_logprobs_npy,
    file_prefix, hf_pretrained_local_kw, load_problems, parse_int_list,
    save_answer, save_logprobs_npy, save_unmask_step_npy, save_trajectory_npy, write_metadata,
)
from diffusion.diffusion_backend import do_generate_diffusion, do_generate_diffusion_batch


def _load_hf_tokenizer(model_path: str, *, use_fast: bool, trust_remote_code: bool):
    """Defer transformers until tokenization; keeps ``import generate`` free of torch."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        model_path,
        use_fast=use_fast,
        trust_remote_code=trust_remote_code,
        **hf_pretrained_local_kw(model_path),
    )


# =====================================================================
# Worker function
# =====================================================================
def do_generate(
    client: OpenAI,
    dataset: str,
    model_name: str,
    problems: List[Dict],
    rendered_prompts: Dict[int, Tuple[str, str]],
    problem_index: int,
    answer_index: int,
    out_dir: Path,
    model_info: Dict,
    api_params: Dict,
    tokenizer=None,
    save_logprobs: bool = False,
    save_trajectory: bool = False,
    capture_tif: bool = False,
    tif_snap_stride: int = 1,
) -> Tuple[bool, int, int, Optional[str]]:
    """Generate one answer for a problem. Returns (success, prob, ans, error)."""
    try:
        problem = problems[problem_index]
        dataset_cfg = get_dataset_config(dataset)

        # Output path
        prefix = file_prefix(dataset, model_name)
        fp = out_dir / f"{prefix}_prob{problem_index}_answer{answer_index}.txt"
        npy_path = Path(str(fp) + ".npy")
        unmask_path = Path(str(fp) + ".unmask_step.npy")

        # Skip if file exists and is large enough (resumability).
        # AR backends produce long CoT (>2KB), so MIN_FILE_SIZE_BYTES (=2KB)
        # is the right floor. Diffusion outputs are typically <1KB on Dream,
        # so we use a smaller floor + a "Generated Tokens:" header check.
        is_diffusion = model_info.get("backend", "vllm") != "vllm"
        tif_npz = (fp.parent / f"{fp.stem}_tif.npz") if (
            capture_tif and is_diffusion) else None
        size_floor = 100 if is_diffusion else MIN_FILE_SIZE_BYTES
        if fp.exists() and fp.stat().st_size >= size_floor:
            header_ok = True
            if is_diffusion:
                try:
                    head = fp.read_text(encoding="utf-8", errors="ignore")[:512]
                    header_ok = "Generated Tokens:" in head
                except Exception:
                    header_ok = False
            tif_ok = tif_npz is None or tif_npz.exists()
            unmask_ok = (not is_diffusion) or unmask_path.exists()
            if (header_ok and tif_ok and unmask_ok
                    and (not save_logprobs or npy_path.exists())):
                return True, problem_index, answer_index, 0, None

        # Get rendered prompt
        rendered, prompt_text = rendered_prompts[problem_index]

        # Call generation backend (autoregressive vLLM or diffusion).
        unmask_step_arr = None
        trajectory_arr = None
        if model_info.get("backend", "vllm") == "vllm":
            text, comp_tokens, raw_top_logprobs, token_logprobs = call_completions(
                client, model_name, rendered, dataset_cfg["complete_max_tokens"],
                model_info["max_context_length"], api_params,
                tokenizer, top_logprobs=TOP_LOGPROBS if save_logprobs else 0,
            )
        else:
            # Per-sample seed: makes (problem_index, answer_index) the only
            # source of randomness so reruns are bit-reproducible AND distinct
            # answer indices use disjoint RNG streams. Without this Dream
            # collapses to N identical samples (see DreamEngine.generate
            # docstring); the 10_000 multiplier just keeps the per-problem
            # seed bands non-overlapping for reasonable num-answers values.
            sample_seed = problem_index * 10_000 + answer_index
            (text, comp_tokens, raw_top_logprobs, token_logprobs,
             unmask_step_arr, trajectory_arr) = do_generate_diffusion(
                model_name=model_name,
                model_info=model_info,
                api_params=api_params,
                prompt=rendered,
                save_top_logprobs=TOP_LOGPROBS if save_logprobs else 0,
                save_trajectory=save_trajectory,
                save_unmask_step=True,
                seed=sample_seed,
                tif_npz_path=tif_npz,
                tif_snap_stride=tif_snap_stride,
            )

        # Extract CoT / Final token counts (delimiter-excluded, for cost accounting)
        # gen_tokens >= cot_tokens + final_tokens (difference = delimiter overhead)
        cot_tokens, final_tokens = count_cot_and_final_tokens(text, model_info, tokenizer)

        gold = problem[dataset_cfg["gold_key"]]
        save_answer(fp, dataset, problem_index, answer_index, gold, prompt_text, text, comp_tokens,
                    cot_tokens=cot_tokens, final_tokens=final_tokens)

        if save_logprobs and raw_top_logprobs:
            save_logprobs_npy(npy_path, raw_top_logprobs, TOP_LOGPROBS)
        if save_logprobs and token_logprobs:
            save_token_logprobs_npy(fp, token_logprobs)
        # Diffusion-only side artifacts (always written when produced; the
        # engine only produces them for diffusion backends, so the autoregressive
        # path naturally keeps these as None).
        if unmask_step_arr is not None:
            save_unmask_step_npy(fp, unmask_step_arr)
        if trajectory_arr is not None:
            save_trajectory_npy(fp, trajectory_arr)

        return True, problem_index, answer_index, comp_tokens, None

    except Exception as e:
        error_msg = f"prob{problem_index}_ans{answer_index}: {e}"
        traceback.print_exc()
        return False, problem_index, answer_index, 0, error_msg


# =====================================================================
# CLI
# =====================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Initial answer generation via vLLM completions API")
    p.add_argument("--dataset", required=True, choices=DATASET_NAMES)
    p.add_argument("--model-name", required=True, choices=MODEL_NAMES)
    p.add_argument("--model-path", default=None, help="Tokenizer path (default: auto-detected from config)")
    p.add_argument("--num-answers", required=True, type=int, help="Number of answers to generate per problem")
    p.add_argument("--out-dir", required=True, type=Path, help="Output directory")
    p.add_argument("--parallel", type=int, default=32, help="Max parallel workers")
    p.add_argument("--timeout", type=int, default=1800, help="OpenAI client timeout in seconds")
    p.add_argument("--problems", type=str, default=None, help="Problem indices filter (e.g. 0,1,2 or 0-4)")
    p.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default=None,
                   help="Reasoning effort level (gpt-oss); omit to use model default")
    p.add_argument("--no-think", action="store_true", help="Disable CoT (no-think mode)")
    p.add_argument("--save-logprobs", action="store_true",
                   help="Save top-20 logprobs as .npy alongside each answer file. "
                        "For diffusion backends, also writes *.unmask_step.npy "
                        "(step at which each position was first unmasked) used "
                        "by step-wise/order-based confidence analyses.")
    p.add_argument("--save-trajectory", action="store_true",
                   help="Diffusion-only: also save the full denoise trajectory "
                        "as *.traj.npy with shape (steps, gen_length) int32. "
                        "Cost: roughly steps*gen_length*4 bytes per sample "
                        "(~512 KB at the default Dream config). Required for "
                        "(E) trajectory-diversity self-consistency experiments.")
    # ----- Diffusion-specific runtime overrides (no config edit needed) -----
    p.add_argument("--diffusion-alg", default=None,
                   choices=["origin", "entropy", "maskgit_plus", "topk_margin"],
                   help="Diffusion-only: override model_info['alg']. Use "
                        "'origin' for SC (sample diversity) or 'entropy' for "
                        "Dream's paper-best single-shot baseline. Ignored for "
                        "non-diffusion backends.")
    p.add_argument("--diffusion-temperature", type=float, default=None,
                   help="Diffusion-only: override sampling temperature. Pair "
                        "with --diffusion-alg=entropy --diffusion-temperature=0.2 "
                        "to reproduce the Dream paper's greedy baseline.")
    p.add_argument("--diffusion-alg-temp", type=float, default=None,
                   help="Diffusion-only: override alg_temp. Setting > 0 "
                        "randomizes the unmask order for entropy/maskgit_plus/"
                        "topk_margin (NB: causes EOS collapse on Dream).")
    p.add_argument("--diffusion-batch-size", type=int, default=1,
                   help="Diffusion-only: number of samples to denoise in a "
                        "single GPU forward. With B>1 the per-step cost "
                        "amortises across the batch (4-8x faster on A100 at "
                        "B=4-8 for Dream-7B). Setting B=1 falls back to the "
                        "old per-sample path. AR (vLLM) backends ignore this "
                        "flag.")
    p.add_argument("--diffusion-steps", type=int, default=None,
                   help="Diffusion-only: override denoise_steps (NFE). Default "
                        "256 for Dream. Set to 512 to double the denoising "
                        "schedule (2x compute, often improves quality on "
                        "harder problems).")
    p.add_argument("--diffusion-gen-length", type=int, default=None,
                   help="Diffusion-only: override model_info['gen_length'] "
                        "(number of mask slots to fill). If the current "
                        "block_length equals the pre-override gen_length "
                        "(Dream full-block default), block_length is updated to "
                        "match; otherwise (e.g. LLaDA) block_length is left as-is.")
    p.add_argument("--diffusion-block-length", type=int, default=None,
                   help="Diffusion-only: override semi-AR block_length. "
                        "Useful for LLaDA (official/default block_length=32).")
    p.add_argument("--diffusion-remasking", default=None,
                   choices=["low_confidence", "random"],
                   help="Diffusion-only: override LLaDA remasking strategy.")
    p.add_argument("--diffusion-cfg-scale", type=float, default=None,
                   help="Diffusion-only: override LLaDA classifier-free "
                        "guidance scale.")
    p.add_argument("--diffusion-capture-tif", action="store_true",
                   help="Dream/LLaDA HF-direct only: during init generation, also "
                        "save TiF-style per-step argmax snapshots next to each "
                        "answer as <stem>_tif.npz. No extra forward passes. "
                        "Requires --diffusion-batch-size 1.")
    p.add_argument("--tif-snap-stride", type=int, default=1,
                   help="With --diffusion-capture-tif: save logits snapshot "
                        "every N denoise steps.")
    return p.parse_args()


# =====================================================================
# Diffusion batched main loop (alternative to the per-task executor)
# =====================================================================
def _save_diffusion_sample(
    *,
    fp: Path,
    npy_path: Path,
    dataset: str,
    problem_index: int,
    answer_index: int,
    problem: Dict,
    dataset_cfg: Dict,
    prompt_text: str,
    text: str,
    comp_tokens: int,
    raw_top_logprobs,
    token_logprobs,
    unmask_step_arr,
    trajectory_arr,
    model_info: Dict,
    tokenizer,
    save_logprobs: bool,
) -> int:
    """Persist one diffusion sample's outputs to disk. Returns comp_tokens."""
    cot_tokens, final_tokens = count_cot_and_final_tokens(text, model_info, tokenizer)
    gold = problem[dataset_cfg["gold_key"]]
    save_answer(fp, dataset, problem_index, answer_index, gold,
                prompt_text, text, comp_tokens,
                cot_tokens=cot_tokens, final_tokens=final_tokens)
    if save_logprobs and raw_top_logprobs:
        save_logprobs_npy(npy_path, raw_top_logprobs, TOP_LOGPROBS)
    if save_logprobs and token_logprobs:
        save_token_logprobs_npy(fp, token_logprobs)
    if unmask_step_arr is not None:
        save_unmask_step_npy(fp, unmask_step_arr)
    if trajectory_arr is not None:
        save_trajectory_npy(fp, trajectory_arr)
    return comp_tokens


def run_diffusion_batched(
    *,
    args,
    dataset_cfg: Dict,
    model_info: Dict,
    api_params: Dict,
    problems: List[Dict],
    rendered_prompts: Dict[int, Tuple[str, str]],
    problem_indices: List[int],
    tokenizer,
) -> None:
    """Sequential per-problem driver that batches ``args.diffusion_batch_size``
    answers of the SAME prompt into one Dream forward stack.

    Compared to the executor path this trades thread parallelism (which is
    irrelevant for the in-process Dream engine because of its RLock) for
    GPU batch parallelism, which IS what scales on the A100.
    """
    prefix = file_prefix(args.dataset, args.model_name)
    B = max(1, args.diffusion_batch_size)
    save_logprobs = args.save_logprobs
    save_traj = args.save_trajectory

    # Diffusion outputs are short (Dream often <1KB) so the AR-tuned 2KB
    # threshold rejects perfectly valid samples. Use the existence of the
    # companion ``Generated Tokens:`` header line as the "complete" signal
    # instead, with a tiny size floor just to skip true zero-byte files.
    MIN_DIFF_FILE_BYTES = 100

    def _sample_is_complete(fp: Path, npy_path: Path,
                            traj_path: Path, unmask_path: Path) -> bool:
        if not fp.exists() or fp.stat().st_size < MIN_DIFF_FILE_BYTES:
            return False
        # Quick header sanity check: any successful save_answer() call writes
        # "Generated Tokens:" near the top, so its absence means the file is
        # truncated/garbage.
        try:
            head = fp.read_text(encoding="utf-8", errors="ignore")[:512]
        except Exception:
            return False
        if "Generated Tokens:" not in head:
            return False
        if save_logprobs and not npy_path.exists():
            return False
        if not unmask_path.exists():
            return False
        if save_traj and not traj_path.exists():
            return False
        return True

    # Build per-problem missing-answer lists (resumability)
    pending: Dict[int, List[int]] = {}
    total_pending = 0
    n_skipped = 0
    for prob_idx in problem_indices:
        miss = []
        for ans_idx in range(args.num_answers):
            fp = args.out_dir / f"{prefix}_prob{prob_idx}_answer{ans_idx}.txt"
            npy_path = Path(str(fp) + ".npy")
            traj_path = Path(str(fp) + ".traj.npy")
            unmask_path = Path(str(fp) + ".unmask_step.npy")
            if _sample_is_complete(fp, npy_path, traj_path, unmask_path):
                n_skipped += 1
            else:
                miss.append(ans_idx)
        if miss:
            pending[prob_idx] = miss
            total_pending += len(miss)
    if n_skipped > 0:
        print(f"  Skipped {n_skipped} already-complete samples")

    print(f"\n{'='*60}")
    print(f"  Diffusion batched: {len(pending)} problems pending, "
          f"{total_pending} samples remaining, batch_size={B}")
    print(f"{'='*60}")

    t0 = time.time()
    success = 0
    errors = 0
    done = 0
    total_tokens = 0

    for prob_idx in problem_indices:
        miss = pending.get(prob_idx)
        if not miss:
            continue
        rendered, prompt_text = rendered_prompts[prob_idx]
        problem = problems[prob_idx]

        for chunk_start in range(0, len(miss), B):
            chunk = miss[chunk_start:chunk_start + B]
            # Per-batch seed: still derived from (problem, first answer in
            # chunk) so a rerun with the same chunking is bit-reproducible.
            # Per-sample reproducibility is sacrificed (the entire batch
            # shares one seed), but in-batch diversity is provided by the
            # alg='origin' RNG draws being independent across the batch dim.
            batch_seed = prob_idx * 10_000 + chunk[0]
            try:
                results = do_generate_diffusion_batch(
                    model_name=args.model_name,
                    model_info=model_info,
                    api_params=api_params,
                    prompt=rendered,
                    batch_size=len(chunk),
                    seed=batch_seed,
                    save_top_logprobs=TOP_LOGPROBS if save_logprobs else 0,
                    save_trajectory=args.save_trajectory,
                    save_unmask_step=True,
                )
            except Exception as e:
                errors += len(chunk)
                done += len(chunk)
                traceback.print_exc()
                print(f"  ERROR prob{prob_idx} chunk {chunk}: {e}")
                continue

            for ans_idx, res in zip(chunk, results):
                fp = args.out_dir / f"{prefix}_prob{prob_idx}_answer{ans_idx}.txt"
                npy_path = Path(str(fp) + ".npy")
                try:
                    (text, comp_tokens, raw_top, raw_tok,
                     unmask_step_arr, trajectory_arr) = res
                    tokens = _save_diffusion_sample(
                        fp=fp, npy_path=npy_path,
                        dataset=args.dataset,
                        problem_index=prob_idx, answer_index=ans_idx,
                        problem=problem, dataset_cfg=dataset_cfg,
                        prompt_text=prompt_text,
                        text=text, comp_tokens=comp_tokens,
                        raw_top_logprobs=raw_top, token_logprobs=raw_tok,
                        unmask_step_arr=unmask_step_arr,
                        trajectory_arr=trajectory_arr,
                        model_info=model_info, tokenizer=tokenizer,
                        save_logprobs=save_logprobs,
                    )
                    success += 1
                    total_tokens += tokens
                except Exception as e:
                    errors += 1
                    traceback.print_exc()
                    print(f"  ERROR save prob{prob_idx} ans{ans_idx}: {e}")
                done += 1

            elapsed = time.time() - t0
            tps = total_tokens / elapsed if elapsed > 0 else 0
            print(f"  Progress: {done}/{total_pending} "
                  f"(success={success}, errors={errors}, "
                  f"{total_tokens:,} tokens, {tps:.0f} tok/s, "
                  f"{elapsed:.0f}s)")

    print(f"\n  Done: success={success}, errors={errors}, total={total_pending}")


# =====================================================================
# Main
# =====================================================================
def main():
    args = parse_args()
    dataset_cfg = get_dataset_config(args.dataset)
    model_info, api_params = build_model_config(args.model_name,
                                                no_think=args.no_think,
                                                reasoning_effort=args.reasoning_effort)
    model_path = args.model_path or model_info["default_model_path"]
    if args.model_path is not None:
        # Keep downstream diffusion engines / token truncation on the same
        # checkpoint/tokenizer that the CLI explicitly requested.
        model_info["default_model_path"] = args.model_path

    # Apply diffusion-specific runtime overrides into model_info / api_params
    # so the override value flows through do_generate_diffusion -> engine
    # without needing to edit core/config.py.
    if args.diffusion_alg is not None:
        model_info["alg"] = args.diffusion_alg
    if args.diffusion_alg_temp is not None:
        model_info["alg_temp"] = args.diffusion_alg_temp
    if args.diffusion_temperature is not None:
        api_params["temperature"] = args.diffusion_temperature
    if args.diffusion_steps is not None:
        model_info["denoise_steps"] = args.diffusion_steps
    if args.diffusion_gen_length is not None:
        if "gen_length" not in model_info:
            print("ERROR: --diffusion-gen-length applies only to diffusion backends.",
                  file=sys.stderr)
            sys.exit(1)
        _old_gl = model_info["gen_length"]
        model_info["gen_length"] = args.diffusion_gen_length
        if model_info.get("block_length") == _old_gl:
            model_info["block_length"] = args.diffusion_gen_length
    if args.diffusion_block_length is not None:
        model_info["block_length"] = args.diffusion_block_length
    if args.diffusion_remasking is not None:
        model_info["remasking"] = args.diffusion_remasking
    if args.diffusion_cfg_scale is not None:
        model_info["cfg_scale"] = args.diffusion_cfg_scale

    is_diff = model_info.get("backend", "vllm") != "vllm"
    if args.diffusion_capture_tif:
        if not is_diff:
            print("  WARNING: --diffusion-capture-tif applies only to diffusion "
                  "backends; ignored.")
        elif args.diffusion_batch_size > 1:
            print("ERROR: --diffusion-capture-tif requires --diffusion-batch-size 1 "
                  "(TiF hook is per forward).", file=sys.stderr)
            sys.exit(1)

    print(f"  dataset:     {args.dataset}")
    print(f"  model-name:  {args.model_name}")
    print(f"  model-path:  {model_path}")
    print(f"  num-answers: {args.num_answers}")
    print(f"  out-dir:     {args.out_dir}")
    print(f"  parallel:    {args.parallel}")
    if model_info.get("backend", "vllm") != "vllm":
        print(f"  gen_length:  {model_info.get('gen_length')}")
        print(f"  block_len:   {model_info.get('block_length')}")
        print(f"  alg:         {model_info.get('alg')}")
        print(f"  alg_temp:    {model_info.get('alg_temp')}")
        print(f"  temperature: {api_params.get('temperature')}")
        print(f"  batch-size:  {args.diffusion_batch_size}")
        if args.diffusion_capture_tif:
            print(f"  capture-tif:  yes (stride={args.tif_snap_stride})")

    # Load tokenizer
    print("  Loading tokenizer...")
    _mp = Path(model_path).expanduser()
    if _mp.is_absolute() and not _mp.is_dir():
        print(
            f"ERROR: model path is not a directory on this host: {_mp}\n"
            "  GPU nodes must see the same files (shared HOME, NFS, or set MODELS_DIR in core/config). "
            "Otherwise Hugging Face falls back to Hub and rejects absolute paths.",
            file=sys.stderr,
        )
        sys.exit(3)
    # Diffusion models (Dream, LLaDA, ...) ship custom tokenizer code and
    # require trust_remote_code; the autoregressive vLLM models do not need
    # it but it is harmless to enable for already-downloaded local weights.
    needs_trust = model_info.get("backend", "vllm") != "vllm"
    tokenizer = _load_hf_tokenizer(
        model_path, use_fast=False, trust_remote_code=needs_trust)

    # Load problems
    problems = load_problems(dataset_cfg["data_file"])

    # Filter problems
    problem_filter = parse_int_list(args.problems)
    if problem_filter is not None:
        problem_indices = [i for i in problem_filter if 0 <= i < len(problems)]
    else:
        problem_indices = list(range(len(problems)))

    # Pre-build rendered prompts
    rendered_prompts: Dict[int, Tuple[str, str]] = {}
    for i in problem_indices:
        rendered_prompts[i] = build_rendered_prompt(
            problems[i], dataset_cfg, model_info, tokenizer
        )

    # Build task list
    tasks = []
    for prob_idx in problem_indices:
        for ans_idx in range(args.num_answers):
            tasks.append((prob_idx, ans_idx))

    total = len(tasks)
    print(f"\n{'='*60}")
    print(f"  Initial generation: {len(problem_indices)} problems x {args.num_answers} answers = {total} tasks")
    print(f"{'='*60}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _meta = dict(
        reasoning_effort=args.reasoning_effort,
        num_answers=args.num_answers,
        no_think=args.no_think,
        diffusion_capture_tif=args.diffusion_capture_tif,
        tif_snap_stride=args.tif_snap_stride,
    )
    if model_info.get("backend", "vllm") != "vllm":
        _meta["gen_length"] = model_info.get("gen_length")
        _meta["block_length"] = model_info.get("block_length")
        _meta["denoise_steps"] = model_info.get("denoise_steps")
    write_metadata(args.out_dir, args.dataset, args.model_name, **_meta)

    # ---- Branch: diffusion + batch-size > 1 -> dedicated batched driver.
    # The executor path serialises on the engine RLock anyway, so for the
    # in-process diffusion engines batching is the only way to scale.
    is_diffusion = model_info.get("backend", "vllm") != "vllm"
    if is_diffusion and args.diffusion_batch_size > 1:
        run_diffusion_batched(
            args=args,
            dataset_cfg=dataset_cfg,
            model_info=model_info,
            api_params=api_params,
            problems=problems,
            rendered_prompts=rendered_prompts,
            problem_indices=problem_indices,
            tokenizer=tokenizer,
        )
        return

    # Initialize vLLM client (also used as a placeholder for diffusion backends
    # so the executor signature stays uniform; diffusion path ignores it).
    client = OpenAI(base_url=BASE_URL, api_key="dummy-key", timeout=args.timeout)

    success_count = 0
    error_count = 0
    max_workers = min(args.parallel, total) if total > 0 else 1

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for prob_idx, ans_idx in tasks:
            future = executor.submit(
                do_generate,
                client, args.dataset, args.model_name,
                problems, rendered_prompts,
                prob_idx, ans_idx, args.out_dir,
                model_info, api_params, tokenizer, args.save_logprobs,
                args.save_trajectory,
                args.diffusion_capture_tif,
                args.tif_snap_stride,
            )
            futures[future] = (prob_idx, ans_idx)

        t0 = time.time()
        done = 0
        total_tokens = 0
        for future in as_completed(futures):
            ok, pi, ai, tokens, err = future.result()
            done += 1
            total_tokens += tokens
            if ok:
                success_count += 1
            else:
                error_count += 1
                if err:
                    print(f"  ERROR: {err}")
            if done % 100 == 0 or done == total:
                elapsed = time.time() - t0
                tps = total_tokens / elapsed if elapsed > 0 else 0
                print(f"  Progress: {done}/{total} (success={success_count}, errors={error_count}, "
                      f"{total_tokens:,} tokens, {tps:.0f} tok/s, {elapsed:.0f}s)")

    print(f"\n  Done: success={success_count}, errors={error_count}, total={total}")


if __name__ == "__main__":
    main()
