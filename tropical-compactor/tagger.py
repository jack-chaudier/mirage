"""Heuristic role classifier for conversation chunks."""

from __future__ import annotations

import math
import re


PREDECESSOR_SIGNALS = [
    r"\bmust\b",
    r"\brequire[sd]?\b",
    r"\bconstraint\b",
    r"\bdo not\b",
    r"\bdon't\b",
    r"\bshould not\b",
    r"\bshouldn't\b",
    r"\bbreaking change\b",
    r"\bbackward compat\b",
    r"\bapi contract\b",
    r"\btest fail",
    r"\berror:\b",
    r"\btraceback\b",
    r"\bexception\b",
    r"\bassert\b",
    r"\bfixed in\b",
    r"\bdecided\b",
    r"\bagreed\b",
    r"\bconclusion\b",
]

PIVOT_SIGNALS = [
    r"\bimplement\b",
    r"\brefactor\b",
    r"\badd feature\b",
    r"\bbuild\b",
    r"\bcreate\b",
    r"\byour task\b",
    r"\bthe goal\b",
    r"\bobjective\b",
    r"\bplease\b.*\bfix\b",
    r"\bplease\b.*\badd\b",
    r"\bplease\b.*\bimplement\b",
]

_PRED_RE = re.compile("|".join(PREDECESSOR_SIGNALS), re.IGNORECASE)
_PIVOT_RE = re.compile("|".join(PIVOT_SIGNALS), re.IGNORECASE)


def tag_chunk(text: str, role_hint: str | None = None) -> tuple[str, float]:
    """
    Return (role, weight).

    role is one of: pivot, predecessor, noise.
    weight is finite only for pivots.
    """

    if role_hint == "pivot":
        return "pivot", 10.0
    if role_hint == "predecessor":
        return "predecessor", -math.inf
    if role_hint == "noise":
        return "noise", -math.inf

    pivot_matches = len(_PIVOT_RE.findall(text))
    pred_matches = len(_PRED_RE.findall(text))

    if pivot_matches > 0 and pivot_matches >= pred_matches:
        weight = float(min(pivot_matches * 2 + len(text) / 200.0, 20.0))
        return "pivot", weight

    if pred_matches > 0:
        return "predecessor", -math.inf

    return "noise", -math.inf
