#!/usr/bin/env python3
"""Write a LaTeX summary for diffusion init/regen/TiF runs.

The report contains:

* init accuracy
* regen keep-variant majority accuracy
* proposal accuracy: use answer0 if answer0 and answer0_keep10 agree,
  otherwise fall back to answer1
* TiF snapshot-vote accuracy for fixed / linear / exp(alpha=5)
* optional multi-answer majority metrics for --report-answer-counts:
  - init majority over answer0..answerN
  - regen majority over answer0..answerN x keep variants (total-vote ties:
    prefer more votes at earlier keep rates in ``--keeps`` order, then first
    occurrence in scan order)
  - TiF vote over all snapshots from answer0..answerN
  (Partial vote when some candidate files are missing; if there are no usable
  votes for a problem, it counts as no-vote.)
* per-cut retention:
  - P(keep correct | init correct)
  - P(keep same wrong answer | init wrong)

The output file defaults to:
  <init-dir-parent>/reports/<init-dir-name>__<regen-dir-name>_report.tex
"""
from __future__ import annotations

import argparse
import re
import signal
import sys
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, TypeVar

import numpy as np

T = TypeVar("T")

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
RC_ROOT = REPO_ROOT / "regen_consistency"
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(RC_ROOT))

from math_answer_grader import (  # type: ignore  # noqa: E402
    _fallback_pred,
    _is_equiv,
    _last_boxed,
    _parse_txt,
    _strip,
)
from multiple_choice_grader import (  # type: ignore  # noqa: E402
    _gold_letter,
    _vote_letters,
    pred_to_letter,
)
from core.config import build_model_config  # type: ignore  # noqa: E402
from core.utils import file_prefix, hf_pretrained_local_kw  # type: ignore  # noqa: E402


class _Timeout(Exception):
    pass


def _alarm(signum, frame):  # noqa: ARG001
    raise _Timeout


signal.signal(signal.SIGALRM, _alarm)


@lru_cache(maxsize=200_000)
def _equiv(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return False
    signal.alarm(3)
    try:
        return _is_equiv(a, b)
    except _Timeout:
        return False
    except Exception:
        return False
    finally:
        signal.alarm(0)


def _norm_key(p: str | None) -> str:
    if p is None:
        return "<NONE>"
    try:
        return _strip(p)
    except Exception:
        return p


def _vote(preds: Iterable[str | None]) -> str | None:
    valid = [p for p in preds if p is not None]
    if not valid:
        return None
    buckets: list[tuple[str, str, float]] = []
    for p in valid:
        key = _norm_key(p)
        for i, (k, repr_, c) in enumerate(buckets):
            if k == key:
                buckets[i] = (k, repr_, c + 1.0)
                break
        else:
            buckets.append((key, p, 1.0))
    buckets.sort(key=lambda b: -b[2])
    return buckets[0][1]


def _vote_keyed_with_keep_tiebreak(
    keyed_preds: list[tuple[str | None, str]],
    keeps: list[str],
    mode: EvalMode,
) -> str | None:
    """Plurality over (pred, keep); ties prefer more votes at earlier ``keeps``.

    Totals tie -> lexicographically larger tuple
    ``(count@keep[0], count@keep[1], ...)`` wins; if still tied, earliest first
    appearance in ``keyed_preds`` wins.
    """
    valid = [(p, k) for p, k in keyed_preds if p is not None]
    if not valid:
        return None
    total: dict[str, int] = defaultdict(int)
    per_keep: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    repr_for: dict[str, str] = {}
    first_pos: dict[str, int] = {}
    for i, (p, k) in enumerate(valid):
        key = p if mode.is_choice else _norm_key(p)
        total[key] += 1
        per_keep[key][k] += 1
        if key not in repr_for:
            repr_for[key] = p
        if key not in first_pos:
            first_pos[key] = i
    best = max(total.values())
    tied = [key for key, c in total.items() if c == best]

    def keep_tuple(key: str) -> tuple[int, ...]:
        return tuple(per_keep[key].get(k, 0) for k in keeps)

    best_kt = max(keep_tuple(k) for k in tied)
    still = [k for k in tied if keep_tuple(k) == best_kt]
    winner_key = min(still, key=lambda k: first_pos[k])
    return repr_for[winner_key]


def _weighted_vote(weighted_preds: Iterable[tuple[str, float]]) -> str | None:
    buckets: list[tuple[str, str, float]] = []
    for p, w in weighted_preds:
        key = _norm_key(p)
        for i, (k, repr_, c) in enumerate(buckets):
            if k == key:
                buckets[i] = (k, repr_, c + w)
                break
        else:
            buckets.append((key, p, w))
    if not buckets:
        return None
    buckets.sort(key=lambda b: -b[2])
    return buckets[0][1]


def _extract(body: str) -> str | None:
    pred = _last_boxed(body)
    if pred is None:
        pred = _fallback_pred(body)
    return pred


def _weight(step: int, total_steps: int, method: str, alpha: float) -> float:
    if method == "fixed":
        return 1.0
    if method == "linear":
        return (step + 1) / max(1, total_steps)
    if method == "exp":
        return float(np.exp(step / max(1, total_steps) * alpha))
    raise ValueError(f"unknown vote method: {method}")


def _pct(num: int, den: int) -> float:
    return 100.0 * num / den if den else 0.0


def _ratio(num: int, den: int) -> str:
    return f"{num}/{den} ({_pct(num, den):.2f}\\%)"


def _tex_escape(s: object) -> str:
    text = str(s)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in text)


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_") or "report"


@dataclass
class Accuracy:
    label: str
    correct: int
    total: int
    note: str = ""


@dataclass(frozen=True)
class EvalMode:
    name: str
    choices: set[str]

    @property
    def is_choice(self) -> bool:
        return self.name == "choice"


def _parse_problem_range(spec: str) -> set[int]:
    """Parse ranges like '0-9,20-29,42' into problem indices."""
    out: set[int] = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            start = int(a)
            end = int(b)
            if end < start:
                start, end = end, start
            out.update(range(start, end + 1))
        else:
            out.add(int(chunk))
    return out


def _parse_answer_counts(spec: str | None, max_answers: int) -> list[int]:
    """Parse answer-count checkpoints such as '1,4,10'."""
    if spec is None:
        return [max_answers] if max_answers > 1 else []

    counts: list[int] = []
    seen: set[int] = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        count = int(chunk)
        if count < 1:
            raise ValueError(
                f"--report-answer-counts values must be >= 1, got {count}"
            )
        if count > max_answers:
            raise ValueError(
                f"--report-answer-counts includes {count}, "
                f"but --num-answers is only {max_answers}"
            )
        if count not in seen:
            counts.append(count)
            seen.add(count)
    return counts


def _default_choices(dataset: str) -> set[str]:
    if dataset == "mmlu_pro":
        return set("ABCDEFGHIJ")
    if dataset == "gpqa_diamond":
        return set("ABCD")
    if dataset in {"commonsense_qa", "tau/commonsense_qa"}:
        return set("ABCDE")
    if dataset in {"strategy_qa", "strategyqa", "StrategyQA", "ChilleD/StrategyQA"}:
        return set("AB")
    return set()


def _build_eval_mode(dataset: str, choices_arg: str | None) -> EvalMode:
    if choices_arg:
        choices = {x.strip() for x in choices_arg.split(",") if x.strip()}
    else:
        choices = _default_choices(dataset)
    return EvalMode("choice" if choices else "math", choices)


def _choice_letter(fp: Path, pred: str | None, mode: EvalMode) -> str | None:
    letter, _ = pred_to_letter(fp, pred, mode.choices)
    return letter


def _extract_gold_pred(fp: Path, mode: EvalMode) -> tuple[str | None, str | None]:
    gold, pred = _parse_txt(fp)
    if mode.is_choice:
        gold_letter = _gold_letter(gold, mode.choices)
        pred_letter = _choice_letter(fp, pred, mode)
        return gold_letter, pred_letter
    return gold, pred


def _extract_eval(fp: Path, mode: EvalMode) -> tuple[str | None, str | None, bool]:
    gold, pred = _extract_gold_pred(fp, mode)
    return gold, pred, _same_answer(gold, pred, mode)


def _same_answer(a: str | None, b: str | None, mode: EvalMode) -> bool:
    if mode.is_choice:
        return a is not None and a == b
    return _equiv(a, b)


def _weighted_vote_choice(weighted_letters: Iterable[tuple[str, float]]) -> str | None:
    buckets: dict[str, float] = {}
    order: list[str] = []
    for letter, weight in weighted_letters:
        if letter not in buckets:
            order.append(letter)
            buckets[letter] = 0.0
        buckets[letter] += weight
    if not buckets:
        return None
    best = max(buckets.values())
    for letter in order:
        if buckets[letter] == best:
            return letter
    return None


def _collect_init_answers(init_dir: Path, prefix: str,
                          num_answers: int) -> dict[int, dict[int, Path]]:
    pat = re.compile(rf"^{re.escape(prefix)}_prob(\d+)_answer(\d+)\.txt$")
    out: dict[int, dict[int, Path]] = {}
    for fp in init_dir.glob(f"{prefix}_prob*_answer*.txt"):
        m = pat.match(fp.name)
        if not m:
            continue
        answer_idx = int(m.group(2))
        if answer_idx >= num_answers:
            continue
        out.setdefault(int(m.group(1)), {})[answer_idx] = fp
    return out


def _collect_init(init_dir: Path, prefix: str) -> dict[int, Path]:
    pat = re.compile(rf"^{re.escape(prefix)}_prob(\d+)_answer0\.txt$")
    out: dict[int, Path] = {}
    for fp in init_dir.glob(f"{prefix}_prob*_answer0.txt"):
        m = pat.match(fp.name)
        if m:
            out[int(m.group(1))] = fp
    return out


def _collect_regen(regen_dir: Path, prefix: str,
                   keeps: list[str]) -> dict[int, dict[str, Path]]:
    keep_set = set(keeps)
    pat = re.compile(rf"^{re.escape(prefix)}_prob(\d+)_keep(\d+)\.txt$")
    out: dict[int, dict[str, Path]] = {}
    for fp in regen_dir.glob(f"{prefix}_prob*_keep*.txt"):
        m = pat.match(fp.name)
        if not m:
            continue
        prob = int(m.group(1))
        keep = m.group(2).lstrip("0") or "0"
        if keep in keep_set:
            out.setdefault(prob, {})[keep] = fp
    return out


def _collect_regen_answers(regen_dir: Path, prefix: str,
                           keeps: list[str],
                           num_answers: int) -> dict[int, dict[int, dict[str, Path]]]:
    keep_set = set(keeps)
    out: dict[int, dict[int, dict[str, Path]]] = {}

    legacy_pat = re.compile(rf"^{re.escape(prefix)}_prob(\d+)_keep(\d+)\.txt$")
    for fp in regen_dir.glob(f"{prefix}_prob*_keep*.txt"):
        m = legacy_pat.match(fp.name)
        if not m:
            continue
        keep = m.group(2).lstrip("0") or "0"
        if keep in keep_set:
            out.setdefault(int(m.group(1)), {}).setdefault(0, {})[keep] = fp

    answer_pat = re.compile(
        rf"^{re.escape(prefix)}_prob(\d+)_answer(\d+)_keep(\d+)\.txt$"
    )
    for fp in regen_dir.glob(f"{prefix}_prob*_answer*_keep*.txt"):
        m = answer_pat.match(fp.name)
        if not m:
            continue
        answer_idx = int(m.group(2))
        if answer_idx >= num_answers:
            continue
        keep = m.group(3).lstrip("0") or "0"
        if keep in keep_set:
            out.setdefault(int(m.group(1)), {}).setdefault(answer_idx, {})[keep] = fp
    return out


def _filter_problem_keys(mapping: dict[int, T],
                         wanted: set[int] | None) -> dict[int, T]:
    if wanted is None:
        return mapping
    return {prob: value for prob, value in mapping.items() if prob in wanted}


def _init_accuracy(inits: dict[int, Path],
                   mode: EvalMode) -> tuple[Accuracy, dict[int, tuple[str | None, str | None, bool]]]:
    rows: dict[int, tuple[str | None, str | None, bool]] = {}
    correct = 0
    no_pred = 0
    for prob, fp in sorted(inits.items()):
        gold, pred, ok = _extract_eval(fp, mode)
        rows[prob] = (gold, pred, ok)
        correct += int(ok)
        no_pred += int(pred is None)
    note = f"no-pred={no_pred}" if no_pred else ""
    return Accuracy("init", correct, len(inits), note), rows


def _init_majority_accuracy(init_answers: dict[int, dict[int, Path]],
                            answer_indices: list[int],
                            mode: EvalMode,
                            inits: dict[int, Path]) -> Accuracy:
    total = len(inits)
    correct = 0
    no_vote = 0
    missing_files = 0
    partial_problems = 0
    for prob in sorted(inits.keys()):
        answers = init_answers.get(prob, {})
        preds: list[str | None] = []
        gold: str | None = None
        missing_here = sum(1 for idx in answer_indices if idx not in answers)
        missing_files += missing_here
        if 0 < missing_here < len(answer_indices):
            partial_problems += 1
        for idx in answer_indices:
            if idx not in answers:
                continue
            g, pred = _parse_txt(answers[idx])
            if mode.is_choice:
                gold = gold or _gold_letter(g, mode.choices)
                preds.append(_choice_letter(answers[idx], pred, mode))
            else:
                gold = gold or g
                preds.append(pred)
        if gold is None and prob in inits:
            gold, _, _ = _extract_eval(inits[prob], mode)
        if not preds:
            no_vote += 1
            continue
        chosen = _vote_letters(preds) if mode.is_choice else _vote(preds)
        no_vote += int(chosen is None)
        correct += int(_same_answer(gold, chosen, mode))
    note_parts = [f"answers={len(answer_indices)}"]
    if partial_problems:
        note_parts.append(f"partial-vote-problems={partial_problems}")
    if missing_files:
        note_parts.append(f"missing-files={missing_files}")
    if no_vote:
        note_parts.append(f"no-vote={no_vote}")
    return Accuracy(
        f"init{len(answer_indices)} majority",
        correct,
        total,
        ", ".join(note_parts),
    )


def _regen_majority_accuracy(regens: dict[int, dict[str, Path]],
                             keeps: list[str],
                             mode: EvalMode,
                             inits: dict[int, Path]) -> Accuracy:
    """Majority vote over keep variants (partial: vote over available keeps)."""
    total = len(inits)
    correct = 0
    no_vote = 0
    missing_keep_slots = 0
    partial_problems = 0
    for prob in sorted(inits.keys()):
        gold: str | None = None
        rmap = regens.get(prob, {})
        missing_here = sum(1 for k in keeps if k not in rmap)
        missing_keep_slots += missing_here
        if 0 < missing_here < len(keeps):
            partial_problems += 1
        keyed: list[tuple[str | None, str]] = []
        for keep in keeps:
            if keep not in rmap:
                continue
            g, pred = _parse_txt(rmap[keep])
            if mode.is_choice:
                gold = gold or _gold_letter(g, mode.choices)
                keyed.append((_choice_letter(rmap[keep], pred, mode), keep))
            else:
                gold = gold or g
                keyed.append((pred, keep))
        if gold is None and prob in inits:
            gold, _, _ = _extract_eval(inits[prob], mode)
        if not keyed:
            no_vote += 1
            continue
        chosen = _vote_keyed_with_keep_tiebreak(keyed, keeps, mode)
        no_vote += int(chosen is None)
        correct += int(_same_answer(gold, chosen, mode))
    note_parts = []
    if partial_problems:
        note_parts.append(f"partial-vote-problems={partial_problems}")
    if missing_keep_slots:
        note_parts.append(f"missing-keep-slots={missing_keep_slots}")
    if no_vote:
        note_parts.append(f"no-vote={no_vote}")
    note = ", ".join(note_parts)
    return Accuracy(f"regen{len(keeps)} majority", correct, total, note)


def _regen_answer_majority_accuracy(
    regen_answers: dict[int, dict[int, dict[str, Path]]],
    keeps: list[str],
    answer_indices: list[int],
    mode: EvalMode,
    inits: dict[int, Path],
) -> Accuracy:
    total = len(inits)
    correct = 0
    no_vote = 0
    missing_files = 0
    partial_problems = 0
    expected = len(keeps) * len(answer_indices)
    for prob in sorted(inits.keys()):
        per_answer = regen_answers.get(prob, {})
        gold: str | None = None
        missing_here = 0
        keyed: list[tuple[str | None, str]] = []
        for answer_idx in answer_indices:
            keep_map = per_answer.get(answer_idx, {})
            for keep in keeps:
                if keep not in keep_map:
                    missing_here += 1
                    continue
                fp = keep_map[keep]
                g, pred = _parse_txt(fp)
                if mode.is_choice:
                    gold = gold or _gold_letter(g, mode.choices)
                    keyed.append((_choice_letter(fp, pred, mode), keep))
                else:
                    gold = gold or g
                    keyed.append((pred, keep))
        missing_files += missing_here
        if 0 < missing_here < expected:
            partial_problems += 1
        if gold is None and prob in inits:
            gold, _, _ = _extract_eval(inits[prob], mode)
        if not keyed:
            no_vote += 1
            continue
        chosen = _vote_keyed_with_keep_tiebreak(keyed, keeps, mode)
        no_vote += int(chosen is None)
        correct += int(_same_answer(gold, chosen, mode))

    note_parts = [f"max-votes={expected}"]
    if partial_problems:
        note_parts.append(f"partial-vote-problems={partial_problems}")
    if missing_files:
        note_parts.append(f"missing-files={missing_files}")
    if no_vote:
        note_parts.append(f"no-vote={no_vote}")
    return Accuracy(
        f"regen{len(keeps)}x{len(answer_indices)} majority",
        correct,
        total,
        ", ".join(note_parts),
    )


def _agree_then_answer1_accuracy(
    init_answers: dict[int, dict[int, Path]],
    regens: dict[int, dict[str, Path]],
    keep: str,
    mode: EvalMode,
    inits: dict[int, Path],
) -> Accuracy:
    """Use answer0 when it agrees with answer0_keepK; otherwise use answer1."""
    total = len(inits)
    correct = 0
    no_vote = 0
    agree = 0
    fallback = 0
    missing_keep = 0
    missing_answer1 = 0

    for prob in sorted(inits.keys()):
        answer0_fp = init_answers.get(prob, {}).get(0, inits[prob])
        gold, answer0_pred = _extract_gold_pred(answer0_fp, mode)

        keep_fp = regens.get(prob, {}).get(keep)
        if keep_fp is None:
            missing_keep += 1
            no_vote += 1
            continue

        keep_gold, keep_pred = _extract_gold_pred(keep_fp, mode)
        gold = gold or keep_gold

        if _same_answer(answer0_pred, keep_pred, mode):
            chosen = answer0_pred
            agree += 1
        else:
            answer1_fp = init_answers.get(prob, {}).get(1)
            if answer1_fp is None:
                missing_answer1 += 1
                no_vote += 1
                continue
            answer1_gold, answer1_pred = _extract_gold_pred(answer1_fp, mode)
            gold = gold or answer1_gold
            chosen = answer1_pred
            fallback += 1

        no_vote += int(chosen is None)
        correct += int(_same_answer(gold, chosen, mode))

    note_parts = [f"keep={keep}", f"agree={agree}", f"fallback={fallback}"]
    if missing_keep:
        note_parts.append(f"missing-keep={missing_keep}")
    if missing_answer1:
        note_parts.append(f"missing-answer1={missing_answer1}")
    if no_vote:
        note_parts.append(f"no-vote={no_vote}")
    return Accuracy(
        f"proposal agree{keep} else answer1",
        correct,
        total,
        ", ".join(note_parts),
    )


def _load_tokenizer(model_name: str, model_path: str | None):
    from transformers import AutoTokenizer  # type: ignore

    if model_path is None:
        model_info, _ = build_model_config(model_name)
        model_path = model_info["default_model_path"]
    return AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False,
        **hf_pretrained_local_kw(model_path),
    )


def _mask_ids(tok) -> set[int]:
    ids: set[int] = set()
    unk_id = getattr(tok, "unk_token_id", None)
    for token in ("<|mask|>", "<|mdm_mask|>"):
        try:
            tid = tok.convert_tokens_to_ids(token)
        except Exception:
            tid = None
        if isinstance(tid, int) and tid >= 0 and tid != unk_id:
            ids.add(tid)
    for attr in ("mask_token_id", "pad_token_id"):
        tid = getattr(tok, attr, None)
        if isinstance(tid, int) and tid >= 0:
            ids.add(tid)
    ids.add(126336)  # LLaDA mask id; harmless if absent.
    return ids


def _decode_tif_ids(tok, ids: list[int], mask_ids: set[int]) -> str:
    ids = [t for t in ids if t not in mask_ids]
    eos_id = getattr(tok, "eos_token_id", None)
    if eos_id is not None:
        try:
            ids = ids[:ids.index(eos_id)]
        except ValueError:
            pass
    return tok.decode(ids, skip_special_tokens=True)


def _tif_accuracy(init_dir: Path, prefix: str, inits: dict[int, Path],
                  tok, method: str, alpha: float,
                  total_steps_override: int,
                  mode: EvalMode) -> Accuracy:
    mask_ids = _mask_ids(tok)
    correct = 0
    total = 0
    no_vote = 0
    missing = 0
    for prob, fp in sorted(inits.items()):
        gold, _, _ = _extract_eval(fp, mode)
        npz = init_dir / f"{prefix}_prob{prob}_answer0_tif.npz"
        if not npz.exists():
            missing += 1
            continue
        data = np.load(npz)
        snaps = data["snapshots"]
        steps = data["snap_steps"]
        total_steps = total_steps_override or (int(steps.max()) + 1)
        weighted: list[tuple[str, float]] = []
        for idx, step in enumerate(steps.tolist()):
            text = _decode_tif_ids(tok, snaps[idx].tolist(), mask_ids)
            pred = _extract(text)
            if pred is not None and pred.strip():
                vote_pred = _choice_letter(fp, pred, mode) if mode.is_choice else pred
                if vote_pred is not None:
                    weighted.append((
                        vote_pred,
                        _weight(int(step), total_steps, method, alpha),
                    ))
        chosen = (
            _weighted_vote_choice(weighted)
            if mode.is_choice else
            _weighted_vote(weighted)
        )
        total += 1
        no_vote += int(chosen is None)
        correct += int(_same_answer(gold, chosen, mode))
    label = {
        "fixed": "TiF fixed",
        "linear": "TiF linear",
        "exp": f"TiF exp alpha={alpha:g}",
    }[method]
    note_parts = []
    if missing:
        note_parts.append(f"missing-tif={missing}")
    if no_vote:
        note_parts.append(f"no-vote={no_vote}")
    return Accuracy(label, correct, total, ", ".join(note_parts))


def _tif_answer_majority_accuracy(
    init_dir: Path,
    prefix: str,
    inits: dict[int, Path],
    answer_indices: list[int],
    tok,
    method: str,
    alpha: float,
    total_steps_override: int,
    mode: EvalMode,
) -> Accuracy:
    mask_ids = _mask_ids(tok)
    correct = 0
    total = len(inits)
    no_vote = 0
    missing_files = 0
    partial_problems = 0
    for prob, fp in sorted(inits.items()):
        gold, _, _ = _extract_eval(fp, mode)
        weighted: list[tuple[str, float]] = []
        missing_for_prob = 0
        for answer_idx in answer_indices:
            npz = init_dir / f"{prefix}_prob{prob}_answer{answer_idx}_tif.npz"
            if not npz.exists():
                missing_for_prob += 1
                continue
            data = np.load(npz)
            snaps = data["snapshots"]
            steps = data["snap_steps"]
            total_steps = total_steps_override or (int(steps.max()) + 1)
            for idx, step in enumerate(steps.tolist()):
                text = _decode_tif_ids(tok, snaps[idx].tolist(), mask_ids)
                pred = _extract(text)
                if pred is None or not pred.strip():
                    continue
                vote_pred = _choice_letter(fp, pred, mode) if mode.is_choice else pred
                if vote_pred is not None:
                    weighted.append((
                        vote_pred,
                        _weight(int(step), total_steps, method, alpha),
                    ))
        missing_files += missing_for_prob
        if 0 < missing_for_prob < len(answer_indices):
            partial_problems += 1
        chosen = (
            _weighted_vote_choice(weighted)
            if mode.is_choice else
            _weighted_vote(weighted)
        )
        no_vote += int(chosen is None)
        correct += int(_same_answer(gold, chosen, mode))
    label = {
        "fixed": f"TiF fixed x{len(answer_indices)}",
        "linear": f"TiF linear x{len(answer_indices)}",
        "exp": f"TiF exp alpha={alpha:g} x{len(answer_indices)}",
    }[method]
    note_parts = [f"answers={len(answer_indices)}"]
    if partial_problems:
        note_parts.append(f"partial-vote-problems={partial_problems}")
    if missing_files:
        note_parts.append(f"missing-files={missing_files}")
    if no_vote:
        note_parts.append(f"no-vote={no_vote}")
    return Accuracy(label, correct, total, ", ".join(note_parts))


def _cut_retention(init_rows: dict[int, tuple[str | None, str | None, bool]],
                   regens: dict[int, dict[str, Path]],
                   keeps: list[str],
                   mode: EvalMode) -> list[tuple[str, int, int, int, int]]:
    full = sorted(
        p for p in init_rows
        if p in regens and all(k in regens[p] for k in keeps)
    )
    init_correct = {
        p for p in full
        if init_rows[p][1] is not None and init_rows[p][2]
    }
    init_wrong = {
        p for p in full
        if init_rows[p][1] is not None and not init_rows[p][2]
    }
    out: list[tuple[str, int, int, int, int]] = []
    for keep in keeps:
        keep_correct_given_init_correct = 0
        same_wrong_given_init_wrong = 0
        for prob in init_correct:
            gold, pred = _parse_txt(regens[prob][keep])
            if mode.is_choice:
                gold_eval = _gold_letter(gold, mode.choices)
                pred_eval = _choice_letter(regens[prob][keep], pred, mode)
            else:
                gold_eval = gold
                pred_eval = pred
            keep_correct_given_init_correct += int(
                _same_answer(gold_eval, pred_eval, mode)
            )
        for prob in init_wrong:
            _, init_pred, _ = init_rows[prob]
            _, pred = _parse_txt(regens[prob][keep])
            pred_eval = (
                _choice_letter(regens[prob][keep], pred, mode)
                if mode.is_choice else pred
            )
            same_wrong_given_init_wrong += int(
                _same_answer(init_pred, pred_eval, mode)
            )
        out.append((
            keep,
            keep_correct_given_init_correct,
            len(init_correct),
            same_wrong_given_init_wrong,
            len(init_wrong),
        ))
    return out


def _write_tex(path: Path, *, dataset: str, model_name: str,
               init_dir: Path, regen_dir: Path, keeps: list[str],
               num_answers: int,
               report_answer_counts: list[int],
               problems: str | None,
               mode: EvalMode,
               accuracies: list[Accuracy],
               retention: list[tuple[str, int, int, int, int]]) -> None:
    lines: list[str] = []
    lines.append("% Auto-generated by analysis/write_diffusion_report.py")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Diffusion consistency summary for "
        + _tex_escape(dataset)
        + " / "
        + _tex_escape(model_name)
        + ".}"
    )
    lines.append(r"\begin{tabular}{lrrl}")
    lines.append(r"\hline")
    lines.append(r"Metric & Correct & Total & Accuracy \\")
    lines.append(r"\hline")
    for acc in accuracies:
        note = f" ({acc.note})" if acc.note else ""
        lines.append(
            f"{_tex_escape(acc.label)} & {acc.correct} & {acc.total} & "
            f"{_ratio(acc.correct, acc.total)}{_tex_escape(note)} \\\\"
        )
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append("")
    lines.append(r"\vspace{0.75em}")
    lines.append("")
    lines.append(r"\begin{tabular}{lcc}")
    lines.append(r"\hline")
    lines.append(
        r"Cut & P(correct $\mid$ init correct) & "
        r"P(same wrong $\mid$ init wrong) \\"
    )
    lines.append(r"\hline")
    for keep, c_num, c_den, w_num, w_den in retention:
        lines.append(
            f"keep{_tex_escape(keep)} & {_ratio(c_num, c_den)} & "
            f"{_ratio(w_num, w_den)} \\\\"
        )
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append("")
    lines.append(r"\vspace{0.75em}")
    lines.append("")
    lines.append(r"\footnotesize")
    lines.append(
        "Init dir: "
        + _tex_escape(init_dir)
        + r"\\"
    )
    lines.append(
        "Regen dir: "
        + _tex_escape(regen_dir)
        + r"\\"
    )
    lines.append("Keeps: " + _tex_escape(",".join(keeps)) + r"\\")
    lines.append("Num answers: " + _tex_escape(num_answers) + r"\\")
    if report_answer_counts:
        lines.append(
            "Report answer counts: "
            + _tex_escape(",".join(str(x) for x in report_answer_counts))
            + r"\\"
        )
    if problems:
        lines.append("Problems: " + _tex_escape(problems) + r"\\")
    if mode.is_choice:
        lines.append(
            "Evaluation: choice letters "
            + _tex_escape(",".join(sorted(mode.choices)))
        )
    else:
        lines.append("Evaluation: normalized answer with SymPy fallback")
    lines.append(r"\end{table}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--init-dir", required=True, type=Path)
    p.add_argument("--regen-dir", required=True, type=Path)
    p.add_argument("--dataset", required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--model-path", default=None)
    p.add_argument("--num-answers", type=int, default=1,
                   help="Number of init answers to include in multi-answer majority metrics.")
    p.add_argument("--report-answer-counts", default=None,
                   help="Comma-separated answer counts to report, e.g. '1,4,10'. Defaults to --num-answers when --num-answers > 1.")
    p.add_argument("--problems", default=None,
                   help="Optional problem range like '0-9,20-29,42' to report.")
    p.add_argument("--prefix", default=None)
    p.add_argument("--keeps", default="10,50,90")
    p.add_argument("--proposal-keep", default="10",
                   help=("Keep percentage used by the answer0 agreement "
                         "fallback metric. Empty string disables it."))
    p.add_argument("--choices", default=None,
                   help="Comma-separated choice letters. Defaults to A-J for mmlu_pro, A-D for gpqa_diamond, A-E for commonsense_qa, and A-B for strategy_qa.")
    p.add_argument("--tif-alpha", type=float, default=5.0)
    p.add_argument("--tif-total-steps", type=int, default=0)
    p.add_argument(
        "--skip-tif",
        action="store_true",
        help="Skip TiF snapshot voting (much faster; only init/regen/retention).",
    )
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    keeps = [k.strip().lstrip("0") or "0"
             for k in args.keeps.split(",") if k.strip()]
    proposal_keep = (args.proposal_keep.strip().lstrip("0") or "0"
                     if args.proposal_keep.strip() else "")
    collect_keeps = list(
        dict.fromkeys(keeps + ([proposal_keep] if proposal_keep else []))
    )
    answer_indices = list(range(max(1, args.num_answers)))
    try:
        report_answer_counts = _parse_answer_counts(
            args.report_answer_counts,
            len(answer_indices),
        )
    except ValueError as exc:
        print(f"[report] {exc}", file=sys.stderr)
        return 2
    wanted_probs = _parse_problem_range(args.problems) if args.problems else None
    prefix = args.prefix or file_prefix(args.dataset, args.model_name)
    mode = _build_eval_mode(args.dataset, args.choices)
    inits = _filter_problem_keys(_collect_init(args.init_dir, prefix), wanted_probs)
    init_answers = _filter_problem_keys(
        _collect_init_answers(args.init_dir, prefix, len(answer_indices)),
        wanted_probs,
    )
    regens = _filter_problem_keys(
        _collect_regen(args.regen_dir, prefix, collect_keeps),
        wanted_probs,
    )
    regen_answers = _filter_problem_keys(
        _collect_regen_answers(
            args.regen_dir, prefix, collect_keeps, len(answer_indices)
        ),
        wanted_probs,
    )
    if not inits:
        print(f"[report] no init files found in {args.init_dir} "
              f"with prefix {prefix}", file=sys.stderr)
        return 2

    init_acc, init_rows = _init_accuracy(inits, mode)
    accuracies = [init_acc, _regen_majority_accuracy(regens, keeps, mode, inits)]
    if proposal_keep and len(answer_indices) >= 2:
        accuracies.append(
            _agree_then_answer1_accuracy(
                init_answers, regens, proposal_keep, mode, inits
            )
        )
    multi_answer_accuracies: list[Accuracy] = []
    for count in report_answer_counts:
        count_answer_indices = list(range(count))
        multi_answer_accuracies.extend([
            _init_majority_accuracy(init_answers, count_answer_indices, mode, inits),
            _regen_answer_majority_accuracy(
                regen_answers, keeps, count_answer_indices, mode, inits
            ),
        ])

    if args.skip_tif:
        a = args.tif_alpha
        accuracies.extend([
            Accuracy("TiF fixed", 0, 0, "skipped: --skip-tif"),
            Accuracy("TiF linear", 0, 0, "skipped: --skip-tif"),
            Accuracy(f"TiF exp alpha={a:g}", 0, 0, "skipped: --skip-tif"),
        ])
        for count in report_answer_counts:
            multi_answer_accuracies.extend([
                Accuracy(f"TiF fixed x{count}", 0, 0, "skipped: --skip-tif"),
                Accuracy(f"TiF linear x{count}", 0, 0, "skipped: --skip-tif"),
                Accuracy(f"TiF exp alpha={a:g} x{count}", 0, 0,
                         "skipped: --skip-tif"),
            ])
    else:
        try:
            tok = _load_tokenizer(args.model_name, args.model_path)
            for method in ("fixed", "linear", "exp"):
                accuracies.append(_tif_accuracy(
                    args.init_dir, prefix, inits, tok,
                    method=method,
                    alpha=args.tif_alpha,
                    total_steps_override=args.tif_total_steps,
                    mode=mode,
                ))
                for count in report_answer_counts:
                    count_answer_indices = list(range(count))
                    multi_answer_accuracies.append(
                        _tif_answer_majority_accuracy(
                            args.init_dir, prefix, inits, count_answer_indices, tok,
                            method=method,
                            alpha=args.tif_alpha,
                            total_steps_override=args.tif_total_steps,
                            mode=mode,
                        )
                    )
        except Exception as exc:  # noqa: BLE001
            accuracies.extend([
                Accuracy("TiF fixed", 0, 0, f"skipped: {exc}"),
                Accuracy("TiF linear", 0, 0, f"skipped: {exc}"),
                Accuracy(f"TiF exp alpha={args.tif_alpha:g}", 0, 0,
                         f"skipped: {exc}"),
            ])
            for count in report_answer_counts:
                multi_answer_accuracies.extend([
                    Accuracy(f"TiF fixed x{count}", 0, 0, f"skipped: {exc}"),
                    Accuracy(f"TiF linear x{count}", 0, 0, f"skipped: {exc}"),
                    Accuracy(
                        f"TiF exp alpha={args.tif_alpha:g} x{count}",
                        0,
                        0,
                        f"skipped: {exc}",
                    ),
                ])
    accuracies.extend(multi_answer_accuracies)

    retention = _cut_retention(init_rows, regens, keeps, mode)
    out = args.out
    if out is None:
        name = (
            f"{_safe_name(args.init_dir.name)}__"
            f"{_safe_name(args.regen_dir.name)}_report.tex"
        )
        out = args.init_dir.parent / "reports" / name
    _write_tex(
        out,
        dataset=args.dataset,
        model_name=args.model_name,
        init_dir=args.init_dir,
        regen_dir=args.regen_dir,
        keeps=keeps,
        num_answers=len(answer_indices),
        report_answer_counts=report_answer_counts,
        problems=args.problems,
        mode=mode,
        accuracies=accuracies,
        retention=retention,
    )
    print(f"[report] wrote {out}")
    if mode.is_choice:
        print(f"[report] evaluation: choice letters {','.join(sorted(mode.choices))}")
    else:
        print("[report] evaluation: normalized answer with SymPy fallback")
    for acc in accuracies:
        print(f"[report] {acc.label}: {_ratio(acc.correct, acc.total)}"
              + (f" ({acc.note})" if acc.note else ""))
    for keep, c_num, c_den, w_num, w_den in retention:
        print(f"[report] keep{keep}: correct|init-correct="
              f"{_ratio(c_num, c_den)}, same-wrong|init-wrong="
              f"{_ratio(w_num, w_den)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
