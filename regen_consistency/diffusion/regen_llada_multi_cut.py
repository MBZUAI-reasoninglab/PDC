#!/usr/bin/env python3
"""Multi-cut regeneration for LLaDA-family diffusion models.

This is the LLaDA counterpart of ``regen_dream_multi_cut.py``.
It reads init ``*_answer{A}.txt`` files, keeps the first K% of the generated
body, and re-denoises the remaining positions with LLaDA.

Unlike the Dream-specific script, this goes through ``diffusion.diffusion_backend``. For the
TiF experiment it uses HF direct official-style local/Hub generation so the
in-process denoising loop is available.

``answer0`` keeps the existing regen file format:
``<dataset>_<model>_prob{P}_keep{K}.txt``.
Additional answers are written as:
``<dataset>_<model>_prob{P}_answer{A}_keep{K}.txt``.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import List


def parse_problem_range(spec: str) -> List[int]:
    out = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(chunk))
    return sorted(out)


def read_body(fp: Path) -> str:
    raw = fp.read_text(encoding="utf-8", errors="ignore")
    sep = "\n" + ("=" * 50) + "\n"
    if sep in raw:
        return raw.split(sep, 1)[1]
    return raw.split("==================================================\n", 1)[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--dataset", default="math500")
    ap.add_argument("--model-name", default="LLaDA-1.5")
    ap.add_argument("--model-path", default=None,
                    help="Tokenizer/HF path. Defaults to config path. "
                         "For Hub use e.g. GSAI-ML/LLaDA-1.5.")
    ap.add_argument("--problems", default=None)
    ap.add_argument("--cuts", default="10,50,90")
    ap.add_argument("--num-answers", type=int, default=1)
    ap.add_argument("--gen-length", type=int, default=512)
    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument("--block-length", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--cfg-scale", type=float, default=0.0)
    ap.add_argument("--remasking", choices=["low_confidence", "random"],
                    default="low_confidence")
    ap.add_argument("--save-top-logprobs", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.gen_length % args.block_length != 0:
        print(f"ERROR: gen-length ({args.gen_length}) must be divisible by "
              f"block-length ({args.block_length})", file=sys.stderr)
        return 2
    if args.steps % (args.gen_length // args.block_length) != 0:
        print("ERROR: steps must be divisible by gen_length/block_length "
              f"({args.steps} vs {args.gen_length // args.block_length})",
              file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cuts = [int(x) for x in args.cuts.split(",") if x.strip()]

    _root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_root))
    from core.utils import file_prefix  # type: ignore  # noqa: E402

    prefix = file_prefix(args.dataset, args.model_name)

    pat = re.compile(rf"^{re.escape(prefix)}_prob(\d+)_answer0\.txt$")
    avail = sorted({int(m.group(1)) for f in args.init_dir.iterdir()
                    if (m := pat.match(f.name))})
    if args.problems:
        wanted = set(parse_problem_range(args.problems))
        targets = [p for p in avail if p in wanted]
    else:
        targets = avail

    print(f"[llada-regen] init-dir : {args.init_dir}")
    print(f"[llada-regen] out-dir  : {args.out_dir}")
    print(f"[llada-regen] model    : {args.model_name}")
    print(f"[llada-regen] model-path: {args.model_path or '(config default)'}")
    print("[llada-regen] backend  : hf-direct")
    print(f"[llada-regen] cuts     : {cuts}")
    print(f"[llada-regen] answers  : {args.num_answers}")
    print(f"[llada-regen] problems : {len(targets)} (avail={len(avail)})")
    print(f"[llada-regen] gen_len={args.gen_length} steps={args.steps} "
          f"block={args.block_length} T={args.temperature} "
          f"remasking={args.remasking} cfg={args.cfg_scale}")

    from core.config import build_model_config, get_dataset_config  # type: ignore  # noqa: E402
    from core.utils import build_rendered_prompt, load_problems, save_answer  # type: ignore  # noqa: E402
    from diffusion.diffusion_backend import do_regen_diffusion  # type: ignore  # noqa: E402
    from transformers import AutoTokenizer  # type: ignore  # noqa: E402
    from core.utils import hf_pretrained_local_kw  # type: ignore  # noqa: E402

    dataset_cfg = get_dataset_config(args.dataset)
    model_info, api_params = build_model_config(args.model_name)
    if args.model_path is not None:
        model_info["default_model_path"] = args.model_path
    model_info["gen_length"] = args.gen_length
    model_info["denoise_steps"] = args.steps
    model_info["block_length"] = args.block_length
    model_info["remasking"] = args.remasking
    model_info["cfg_scale"] = args.cfg_scale
    api_params["temperature"] = args.temperature
    api_params["top_p"] = args.top_p

    model_path = model_info["default_model_path"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=False,
        **hf_pretrained_local_kw(model_path),
    )
    problems = load_problems(dataset_cfg["data_file"])

    total_done = 0
    total_skipped = 0
    t_start = time.time()

    for prob_idx in targets:
        if prob_idx < 0 or prob_idx >= len(problems):
            print(f"[prob {prob_idx}] out of range, skip")
            continue

        problem = problems[prob_idx]
        rendered, prompt_text = build_rendered_prompt(
            problem, dataset_cfg, model_info, tokenizer)
        gold = str(problem[dataset_cfg["gold_key"]])

        t0 = time.time()
        for answer_idx in range(args.num_answers):
            spine_fp = args.init_dir / f"{prefix}_prob{prob_idx}_answer{answer_idx}.txt"
            if not spine_fp.exists():
                print(f"[prob {prob_idx}] answer{answer_idx} no spine file, skip")
                continue
            body = read_body(spine_fp)
            body_ids = tokenizer.encode(body, add_special_tokens=False)
            body_len = len(body_ids)
            # if body_len < 5:
            #     print(f"[prob {prob_idx}] answer{answer_idx} spine too short ({body_len} toks), skip")
            #     continue

            print(f"[prob {prob_idx:>5} ans {answer_idx:>2}] "
                  f"gold={gold[:24]!r:<26} spine_len={body_len}")
            for keep_pct in cuts:
                if answer_idx == 0:
                    out_txt = args.out_dir / f"{prefix}_prob{prob_idx}_keep{keep_pct:02d}.txt"
                else:
                    out_txt = args.out_dir / (
                        f"{prefix}_prob{prob_idx}_answer{answer_idx}_keep{keep_pct:02d}.txt"
                    )
                if out_txt.exists() and out_txt.stat().st_size > 100 and not args.overwrite:
                    total_skipped += 1
                    continue

                # do_regen_diffusion keeps (100-remove_pct)% of source_text.
                remove_pct = max(0, min(100, 100 - keep_pct))
                try:
                    text, comp_tokens, raw_top, tok_lp = do_regen_diffusion(
                        model_name=args.model_name,
                        model_info=model_info,
                        api_params=api_params,
                        prompt=rendered,
                        source_text=body,
                        diffusion_mode="position",
                        remove_pct=remove_pct,
                        seed=answer_idx,
                        save_top_logprobs=args.save_top_logprobs,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"  answer={answer_idx} keep={keep_pct:>2}% ERROR: {exc}")
                    continue

                clamped_keep_pct = max(0, min(100, keep_pct))
                if body_len <= 0 or clamped_keep_pct <= 0:
                    kept_tokens = 0
                else:
                    kept_tokens = min(
                        body_len,
                        max(1, int(body_len * clamped_keep_pct / 100)),
                    )
                regen_tokens = max(0, args.gen_length - kept_tokens)
                effective_steps = max(1, round(args.steps * regen_tokens / args.gen_length))
                save_answer(
                    out_txt,
                    dataset=args.dataset,
                    problem_index=prob_idx,
                    answer_index=keep_pct if answer_idx == 0 else answer_idx,
                    gold_answer=gold,
                    prompt=prompt_text,
                    answer=text,
                    completion_tokens=comp_tokens,
                    extra_headers={
                        "Source": "llada_multi_cut_regen_from_spine",
                        "Spine File": spine_fp.name,
                        "Spine Answer Index": answer_idx,
                        "Spine Total Tokens": body_len,
                        "Kept Tokens": kept_tokens,
                        "Regen Tokens": regen_tokens,
                        "Keep Pct": keep_pct,
                        "Steps": args.steps,
                        "Effective Regen Steps": effective_steps,
                        "Gen Length": args.gen_length,
                        "Block Length": args.block_length,
                        "Remasking": args.remasking,
                        "Cfg Scale": args.cfg_scale,
                        "Backend": "hf-direct",
                    },
                )
                # We deliberately skip logprob side files here unless future LLaDA
                # experiments need them.
                _ = raw_top, tok_lp
                total_done += 1

        print(f"          ({time.time()-t0:.0f}s for {len(cuts) * args.num_answers} cuts; "
              f"done={total_done} skipped={total_skipped})")

    print(f"\n[done] total_done={total_done} skipped={total_skipped} "
          f"elapsed={time.time()-t_start:.0f}s "
          f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','?')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
