#!/usr/bin/env python3
"""Multiple-choice grader for GPQA/MMLU-style generated txt files.

This differs from ``math_answer_grader.py`` in one important way: predictions are
first normalized to a choice letter (A/B/C/D by default), and majority voting
is performed over those letters, not over raw answer strings.

Letter extraction order:
  1. Extract final answer with ``math_answer_grader._parse_txt``.
  2. If answer text exactly matches an option body, map it to that option.
  3. If answer is a letter form (``C``, ``(C)``, ``C.``, ``\\text{C}``), use it.
  4. Otherwise prediction is ``None``.

The option-body match is intentionally checked before loose letter matching so
that chemical formulas like ``C6H10O`` are not accidentally treated as choice C.
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from math_answer_grader import _parse_txt, _remove_boxed, _strip  # type: ignore  # noqa: E402


OPT_RE = re.compile(r"^([A-Z])\.\s*(.+)$", re.MULTILINE)
BOXED_CHOICE_RE = re.compile(r"^\\text\s*\{\s*([A-Z])\s*\}$")
LETTER_FORM_RE = re.compile(r"^\s*(?:\(?\s*)?([A-Z])(?:\s*\)?|[.):,\s].*)$")


def _norm_text(s: str | None) -> str | None:
    if s is None:
        return None
    try:
        return _strip(s).lower().strip().rstrip(".")
    except Exception:
        return s.lower().strip().rstrip(".")


def _parse_options(fp: Path, choices: set[str]) -> dict[str, str]:
    txt = fp.read_text(encoding="utf-8", errors="ignore")
    header = txt.split("==================================================", 1)[0]
    out: dict[str, str] = {}
    for m in OPT_RE.finditer(header):
        letter = m.group(1)
        if letter in choices:
            out[letter] = m.group(2).strip()
    return out


def pred_to_letter(fp: Path, pred: str | None, choices: set[str]) -> tuple[str | None, str]:
    """Map a raw extracted prediction to a choice letter.

    Returns ``(letter, reason)`` for diagnostics.
    """
    if pred is None:
        return None, "none"

    pred = _remove_boxed(pred).strip()
    options = _parse_options(fp, choices)

    # First, map exact option text to its letter. This must precede letter-form
    # parsing to avoid chemical formulas such as C6H10O being treated as C.
    npred = _norm_text(pred)
    for letter, option_text in options.items():
        if _norm_text(option_text) == npred:
            return letter, "option-text"

    # Common model outputs: \text{C}, C, (C), C., C) or C. explanation.
    m = BOXED_CHOICE_RE.match(pred)
    if m and m.group(1) in choices:
        return m.group(1), "text-letter"

    try:
        stripped = _strip(pred)
    except Exception:
        stripped = pred
    if stripped in choices:
        return stripped, "stripped-letter"

    m = LETTER_FORM_RE.match(pred)
    if m and m.group(1) in choices:
        return m.group(1), "leading-letter"

    return None, "unmapped"


def _vote_letters(letters: list[str | None]) -> str | None:
    valid = [x for x in letters if x is not None]
    if not valid:
        return None
    counts = Counter(valid)
    best = max(counts.values())
    # Tie-break by first occurrence order, matching existing majority graders.
    for x in valid:
        if counts[x] == best:
            return x
    return None


def _gold_letter(gold: str | None, choices: set[str]) -> str | None:
    if gold is None:
        return None
    g = gold.strip()
    if g in choices:
        return g
    m = LETTER_FORM_RE.match(g)
    if m and m.group(1) in choices:
        return m.group(1)
    return None
