"""Heuristic role classifier for conversation chunks."""

from __future__ import annotations

from dataclasses import dataclass
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

ASSISTANT_COMMIT_SIGNALS = [
    r"\bi will\b",
    r"\blet me\b",
    r"\bunderstood\b",
    r"\bi can help\b",
    r"\bnext step\b",
    r"\bstarting with\b",
]

TASK_VERB_SIGNALS = [
    r"\bimplement\b",
    r"\bbuild\b",
    r"\bcreate\b",
    r"\bfix\b",
    r"\brefactor\b",
    r"\bwrite\b",
    r"\bdesign\b",
]

_PRED_RE = re.compile("|".join(PREDECESSOR_SIGNALS), re.IGNORECASE)
_PIVOT_RE = re.compile("|".join(PIVOT_SIGNALS), re.IGNORECASE)
_ASSISTANT_COMMIT_RE = re.compile("|".join(ASSISTANT_COMMIT_SIGNALS), re.IGNORECASE)
_TASK_VERB_RE = re.compile("|".join(TASK_VERB_SIGNALS), re.IGNORECASE)


@dataclass(slots=True)
class TagResult:
    role: str
    weight: float
    confidence: float
    reasons: list[str]


def _confidence(pivot_score: float, pred_score: float) -> float:
    delta = abs(pivot_score - pred_score)
    conf = 0.55 + min(0.35, 0.1 * delta)
    if max(pivot_score, pred_score) >= 3.0:
        conf += 0.05
    return min(0.99, conf)


def tag_chunk_detailed(
    text: str,
    role_hint: str | None = None,
    speaker: str | None = None,
    index: int | None = None,
    total: int | None = None,
) -> TagResult:
    """Return role/weight plus confidence for structural heuristics."""

    text_value = text if isinstance(text, str) else ""

    if role_hint == "pivot":
        return TagResult(role="pivot", weight=10.0, confidence=1.0, reasons=["role_hint:pivot"])
    if role_hint == "predecessor":
        return TagResult(
            role="predecessor",
            weight=-math.inf,
            confidence=1.0,
            reasons=["role_hint:predecessor"],
        )
    if role_hint == "noise":
        return TagResult(role="noise", weight=-math.inf, confidence=1.0, reasons=["role_hint:noise"])

    pivot_matches = len(_PIVOT_RE.findall(text_value))
    pred_matches = len(_PRED_RE.findall(text_value))
    commit_matches = len(_ASSISTANT_COMMIT_RE.findall(text_value))
    task_matches = len(_TASK_VERB_RE.findall(text_value))

    pivot_score = float(pivot_matches)
    pred_score = float(pred_matches)
    reasons: list[str] = []

    speaker_value = speaker.lower().strip() if isinstance(speaker, str) else ""
    if speaker_value in {"user", "system"} and task_matches > 0:
        pivot_score += 0.9
        reasons.append("speaker_task_signal")
    if speaker_value == "assistant" and commit_matches > 0:
        pivot_score += 0.75
        reasons.append("assistant_commit_signal")
    if speaker_value in {"user", "system"} and pred_matches > 0:
        pred_score += 0.4
        reasons.append("speaker_constraint_signal")

    if total is not None and total > 1 and index is not None and index >= 0:
        pos = index / max(1, total - 1)
        if pos <= 0.25 and speaker_value in {"user", "system"} and task_matches > 0:
            pivot_score += 0.4
            reasons.append("early_task_position")
        if pos <= 0.35 and pred_matches > 0:
            pred_score += 0.2
            reasons.append("early_constraint_position")

    if pivot_score > 0.0 and pivot_score >= pred_score:
        weight = float(min(pivot_score * 2.0 + len(text_value) / 200.0, 20.0))
        return TagResult(
            role="pivot",
            weight=weight,
            confidence=_confidence(pivot_score, pred_score),
            reasons=reasons or ["pivot_signal"],
        )

    if pred_score > 0.0:
        return TagResult(
            role="predecessor",
            weight=-math.inf,
            confidence=_confidence(pivot_score, pred_score),
            reasons=reasons or ["predecessor_signal"],
        )

    return TagResult(role="noise", weight=-math.inf, confidence=0.55, reasons=["no_signal"])


def tag_chunk(text: str, role_hint: str | None = None) -> tuple[str, float]:
    """
    Return (role, weight).

    role is one of: pivot, predecessor, noise.
    weight is finite only for pivots.
    """
    tagged = tag_chunk_detailed(text=text, role_hint=role_hint)
    return tagged.role, tagged.weight
