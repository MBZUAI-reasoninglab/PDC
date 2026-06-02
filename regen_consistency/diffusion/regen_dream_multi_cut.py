#!/usr/bin/env python3
"""
multi_cut_entropy_regen.py
==========================

For each spine in --init-dir, re-roll from N truncation points with Dream's
unmasking ``--alg`` (default: ``entropy``). ``answer0`` keeps the legacy
output names ``..._prob{P}_keep{K}.txt``; additional answers are written as
``..._prob{P}_answer{A}_keep{K}.txt`` so reports can majority-vote across
all ``num_answers * cuts`` regen samples without breaking older scripts.

The model prompt (chat template) is built with
``core.utils.build_rendered_prompt`` — the same function and ``model_info``
as ``generate_initial_answers.py``, so conditioning matches init when ``--dataset`` and
``--model-name`` align.

``entropy`` with ``--alg-temp 0`` is deterministic (one sample per cut
suffices). ``origin`` / non-zero ``alg-temp`` are stochastic; use distinct
seeds or multiple samples if you need diversity.

CLI
---
    uv run python diffusion/regen_dream_multi_cut.py \
        --init-dir /workspace/cot/regen_consistency/diffusion/init_entropy_steps512 \
        --out-dir  /workspace/cot/regen_consistency/diffusion/regen_from_entropy_steps512 \
        --problems 0-249 \
        --cuts 10,20,30,40,50,60,70,80,90 \
        --gen-length 512 --steps 512

Skips files that already exist (--overwrite to force).
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import List

import numpy as np


def parse_problem_range(spec: str) -> List[int]:
    """Parse '0-9,20-29,42' to a sorted unique list of ints."""
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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--init-dir", required=True, type=Path,
                     help="Directory containing entropy spine answer0 files")
    ap.add_argument("--out-dir", required=True, type=Path,
                     help="Where to write regen .txt and .npy files")
    ap.add_argument("--problems", type=str, default=None,
                     help="Range like '0-249' or '0-9,20-29' (default: ALL "
                          "answer0 problems found in --init-dir)")
    ap.add_argument("--dataset", default="math500")
    ap.add_argument("--model-name", default="Dream-v0-Instruct-7B")
    ap.add_argument(
        "--model-path", default=None,
        help="HF weights path (default: same as core/config build_model_config).",
    )
    ap.add_argument("--cuts", default="10,20,30,40,50,60,70,80,90",
                     help="Comma-separated keep%% values (default: 10..90 step 10)")
    ap.add_argument("--num-answers", type=int, default=1,
                    help="Number of init answers per problem to regen.")
    ap.add_argument("--gen-length", type=int, default=512)
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.2,
                     help="Token sampling temperature. Default 0.2 matches "
                          "Dream paper baseline (and existing init_entropy/ "
                          "spines on this project).")
    ap.add_argument(
        "--alg", default="entropy",
        choices=["origin", "entropy", "maskgit_plus", "topk_margin"],
        help="Dream unmasking schedule (same as generate_initial_answers.py --diffusion-alg).",
    )
    ap.add_argument(
        "--alg-temp", type=float, default=0.0,
        help="Unmask-order temperature. Keep 0 for stable entropy order; "
             ">0 randomizes (can hurt quality on Dream if too large).",
    )
    ap.add_argument("--top-p", type=float, default=0.95, help="Diffusion top_p.")
    ap.add_argument("--top-k", type=int, default=0, help="0 = disabled.")
    ap.add_argument("--save-top-logprobs", type=int, default=20)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    cuts = [int(x) for x in args.cuts.split(",")]

    _root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_root))
    from core.utils import file_prefix

    prefix = file_prefix(args.dataset, args.model_name)

    # Discover problems
    pat = re.compile(rf"^{re.escape(prefix)}_prob(\d+)_answer0\.txt$")
    avail = sorted({int(m.group(1)) for f in args.init_dir.iterdir()
                    if (m := pat.match(f.name))})
    if args.problems:
        wanted = set(parse_problem_range(args.problems))
        targets = [p for p in avail if p in wanted]
    else:
        targets = avail

    print(f"[setup] init-dir : {args.init_dir}")
    print(f"[setup] out-dir  : {args.out_dir}")
    print(f"[setup] cuts     : {cuts}")
    print(f"[setup] problems : {len(targets)} (avail={len(avail)})")
    print(f"[setup] answers  : {args.num_answers}")
    print(f"[setup] gen_len  : {args.gen_length}, steps={args.steps}")
    print(f"[setup] alg        : {args.alg}  alg_temp={args.alg_temp}  "
          f"top_p={args.top_p}  top_k={args.top_k}")

    # --- Lazy heavy imports: package root = regen_consistency ---
    import torch
    from core.config import build_model_config, get_dataset_config
    from core.utils import (
        build_rendered_prompt,
        load_problems,
        save_answer,
        save_logprobs_npy,
        save_token_logprobs_npy,
    )
    from diffusion.dream_engine import DreamEngine

    dataset_cfg = get_dataset_config(args.dataset)
    model_info, _api = build_model_config(args.model_name)
    model_path = args.model_path or model_info["default_model_path"]
    problems = load_problems(dataset_cfg["data_file"])

    print(f"[setup] loading {args.model_name} from {model_path} ...")
    eng = DreamEngine.get(model_path=model_path)
    print(f"[setup] ready (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','?')})\n")

    total_done = 0
    total_skipped = 0
    t_start = time.time()

    _body_sep = "\n" + ("=" * 50) + "\n"

    for prob_idx in targets:
        if prob_idx < 0 or prob_idx >= len(problems):
            print(f"[prob {prob_idx}] out of range (dataset has {len(problems)} problems), skip")
            continue

        problem = problems[prob_idx]
        # Same prompt path as generate_initial_answers.py: tokenizer.apply_chat_template via build_rendered_prompt
        rendered, prompt_text = build_rendered_prompt(
            problem, dataset_cfg, model_info, eng.tokenizer
        )
        gold = str(problem[dataset_cfg["gold_key"]])

        t0 = time.time()
        for answer_idx in range(args.num_answers):
            spine_fp = args.init_dir / f"{prefix}_prob{prob_idx}_answer{answer_idx}.txt"
            try:
                raw = spine_fp.read_text(encoding="utf-8", errors="ignore")
                if _body_sep in raw:
                    body = raw.split(_body_sep, 1)[1]
                else:
                    body = raw.split("==================================================\n", 1)[1]
            except Exception as e:
                print(f"[prob {prob_idx}] answer{answer_idx} cannot read spine: {e}")
                continue

            ent_ids = eng.tokenizer.encode(body, add_special_tokens=False)
            body_len = len(ent_ids)
            if body_len < 5:
                print(f"[prob {prob_idx}] answer{answer_idx} spine too short ({body_len} toks), skip")
                continue

            print(f"[prob {prob_idx:>3} ans {answer_idx:>2}] "
                  f"gold={str(gold)[:30]!r:<32} spine_len={body_len}")
            for keep_pct in cuts:
                if answer_idx == 0:
                    out_txt = args.out_dir / f"{prefix}_prob{prob_idx}_keep{keep_pct:02d}.txt"
                else:
                    out_txt = args.out_dir / (
                        f"{prefix}_prob{prob_idx}_answer{answer_idx}_keep{keep_pct:02d}.txt"
                    )
                out_npy = Path(str(out_txt) + ".npy")
                if (out_txt.exists() and out_txt.stat().st_size > 100
                        and out_npy.exists() and not args.overwrite):
                    total_skipped += 1
                    continue

                kept_tokens = max(1, int(body_len * keep_pct / 100))
                kept_text = (body if kept_tokens >= body_len
                             else eng.tokenizer.decode(
                                 ent_ids[:kept_tokens], skip_special_tokens=True))

                torch.manual_seed(answer_idx)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(answer_idx)
                try:
                    out = eng.regen_position(
                        prompt_text=rendered,
                        kept_text=kept_text,
                        gen_length=args.gen_length,
                        kept_token_count=kept_tokens,
                        steps=args.steps,
                        temperature=args.temperature,
                        top_p=args.top_p, top_k=args.top_k,
                        alg=args.alg, alg_temp=args.alg_temp,
                        save_top_logprobs=args.save_top_logprobs,
                        kept_token_ids=ent_ids[:kept_tokens],
                    )
                except Exception as e:
                    print(f"  answer={answer_idx} keep={keep_pct:>2}% ERROR: {e}")
                    continue

                save_answer(
                    out_txt,
                    dataset=args.dataset,
                    problem_index=prob_idx,
                    answer_index=keep_pct if answer_idx == 0 else answer_idx,
                    gold_answer=gold,
                    prompt=prompt_text,
                    answer=out.text,
                    completion_tokens=out.completion_tokens,
                    extra_headers={
                        "Source": "multi_cut_regen_from_spine",
                        "Spine File": spine_fp.name,
                        "Spine Answer Index": answer_idx,
                        "Spine Total Tokens": body_len,
                        "Kept Tokens": kept_tokens,
                        "Keep Pct": keep_pct,
                        "Alg": args.alg,
                        "Alg Temp": args.alg_temp,
                        "Steps": args.steps,
                        "Gen Length": args.gen_length,
                    },
                )
                if args.save_top_logprobs > 0 and out.top_logprobs:
                    save_logprobs_npy(out_npy, out.top_logprobs,
                                      top_k=args.save_top_logprobs)
                    save_token_logprobs_npy(out_txt, out.token_logprobs)
                total_done += 1

        print(f"          ({time.time()-t0:.0f}s for {len(cuts) * args.num_answers} cuts; "
              f"running total: done={total_done}, skipped={total_skipped})")

    dt = time.time() - t_start
    print(f"\n[done] {total_done} new files, {total_skipped} skipped, "
          f"{dt:.0f}s total ({dt/max(1,total_done):.1f}s/sample)")


if __name__ == "__main__":
    main()
