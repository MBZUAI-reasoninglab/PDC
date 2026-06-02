#!/usr/bin/env python3
"""math500 accuracy grader.

Reads ``*.txt`` files in DIR (one generation per problem, with a
``Gold Answer:`` header line and a ``\\boxed{...}`` answer somewhere
in the body) and reports accuracy using a Hendrycks-MATH-style
normalization plus a sympy fallback.
"""
from __future__ import annotations

import re
from pathlib import Path

# Hendrycks MATH-style normalization (slightly extended).
# Reference: https://github.com/hendrycks/math/blob/main/modeling/math_equivalence.py


def _last_boxed(text: str) -> str | None:
    idx = text.rfind("\\boxed")
    if idx < 0:
        idx = text.rfind("\\fbox")
        if idx < 0:
            return None
    i = text.find("{", idx)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
    return None


def _remove_boxed(s: str) -> str:
    if s.startswith("\\boxed "):
        return s[len("\\boxed "):]
    left = "\\boxed{"
    if s.startswith(left) and s.endswith("}"):
        return s[len(left):-1]
    return s


def _fix_fracs(s: str) -> str:
    parts = s.split("\\frac")
    out = parts[0]
    for sub in parts[1:]:
        if not sub:
            continue
        if sub[0] == "{":
            out += "\\frac" + sub
        else:
            try:
                a, b, rest = sub[0], sub[1], sub[2:]
                out += f"\\frac{{{a}}}{{{b}}}{rest}"
            except IndexError:
                return s
    return out


def _fix_a_slash_b(s: str) -> str:
    if len(s.split("/")) != 2:
        return s
    a, b = s.split("/")
    try:
        ai = int(a)
        bi = int(b)
        return f"\\frac{{{ai}}}{{{bi}}}"
    except ValueError:
        return s


def _remove_right_units(s: str) -> str:
    return re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", s)


def _strip_outer_text(s: str) -> str:
    s = s.strip()
    m = re.fullmatch(r"\\text\s*\{(.+)\}", s)
    return m.group(1) if m else s


def _fix_sqrt(s: str) -> str:
    if "\\sqrt" not in s:
        return s
    out = ""
    i = 0
    while i < len(s):
        if s.startswith("\\sqrt", i):
            j = i + len("\\sqrt")
            if j < len(s) and s[j] != "{":
                out += "\\sqrt{" + s[j] + "}"
                i = j + 1
                continue
            out += "\\sqrt"
            i += len("\\sqrt")
        else:
            out += s[i]
            i += 1
    return out


def _strip(s: str) -> str:
    s = s.replace("\n", "")
    s = s.replace("\\!", "")
    s = s.replace("\\\\", "\\")
    s = s.replace("tfrac", "frac").replace("dfrac", "frac")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "")
    s = s.replace("\\$", "")
    s = _remove_right_units(s)
    s = s.replace("\\%", "").replace(r"\%", "").replace("%", "")
    s = s.replace(" .", " 0.").replace("{.", "{0.")
    if s.startswith("."):
        s = "0" + s
    if len(s.split("=")) == 2:
        s = s.split("=")[-1]
    s = _fix_sqrt(s)
    s = s.replace(" ", "")
    s = _fix_fracs(s)
    if s == "0.5":
        s = "\\frac{1}{2}"
    s = _fix_a_slash_b(s)
    return s


_MC_LETTER_RE = re.compile(r"^\s*([A-Z])(?:[.):,\s]|$)")


def _mc_letter_match(gold: str, pred: str) -> bool:
    """If gold is a single capital letter (multiple-choice answer),
    accept pred when its leading token is the same letter (e.g.
    ``\\boxed{B. ~ 4.4}`` → ``B``)."""
    g = gold.strip()
    if len(g) == 1 and "A" <= g <= "Z":
        m = _MC_LETTER_RE.match(pred)
        if m and m.group(1) == g:
            return True
    p = pred.strip()
    if len(p) == 1 and "A" <= p <= "Z":
        m = _MC_LETTER_RE.match(gold)
        if m and m.group(1) == p:
            return True
    return False


def _is_equiv(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return False
    if _mc_letter_match(a, b):
        return True
    try:
        na = _strip(a)
        nb = _strip(b)
    except Exception:
        return a == b
    if na == nb:
        return True
    return _sympy_equiv(na, nb)


_LATEX_NUM_RE = re.compile(r"^-?\d+(\.\d+)?$")


def _to_sympy(expr: str):
    import sympy as sp
    from sympy.parsing.latex import parse_latex  # type: ignore

    try:
        return parse_latex(expr)
    except Exception:
        try:
            return sp.sympify(expr)
        except Exception:
            return None


def _sympy_equiv(a: str, b: str) -> bool:
    try:
        import sympy as sp
    except ImportError:
        return False
    if _LATEX_NUM_RE.match(a) and _LATEX_NUM_RE.match(b):
        try:
            return abs(float(a) - float(b)) < 1e-6
        except Exception:
            return False
    ea = _to_sympy(a)
    eb = _to_sympy(b)
    if ea is None or eb is None:
        return False
    try:
        return bool(sp.simplify(ea - eb) == 0)
    except Exception:
        return False


# -----------------------------------------------------------------------
# File parsing
# -----------------------------------------------------------------------
_HEADER_GOLD = re.compile(r"^Gold Answer:\s*(.*)$", re.MULTILINE)
_SEP = "==================================================" + ""


_ANSWER_IS_RE = re.compile(
    r"(?:The\s+answer\s+is|answer\s*[:=]|final\s+answer\s+is|answer\s+is)\s*"
    r"[:=]?\s*\$?(.+?)\$?\s*\.?\s*$",
    re.IGNORECASE,
)


def _fallback_pred(body: str) -> str | None:
    last_lines = [ln.strip() for ln in body.strip().splitlines() if ln.strip()]
    for ln in reversed(last_lines[-6:]):
        m = _ANSWER_IS_RE.search(ln)
        if m:
            cand = m.group(1).strip().rstrip(".")
            cand = cand.strip("$ ").strip()
            if cand:
                return cand
    return None


def _parse_txt(fp: Path) -> tuple[str | None, str | None]:
    txt = fp.read_text(encoding="utf-8", errors="ignore")
    m = _HEADER_GOLD.search(txt)
    gold = m.group(1).strip() if m else None
    body = txt.split(_SEP, 1)[1] if _SEP in txt else txt
    pred = _last_boxed(body)
    if pred is None:
        pred = _fallback_pred(body)
    return gold, pred
