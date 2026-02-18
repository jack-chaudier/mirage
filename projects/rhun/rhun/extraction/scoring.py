"""Scoring functions for extracted sequences."""

from __future__ import annotations

from rhun.schemas import ExtractedSequence


def weight_sum_score(sequence: ExtractedSequence) -> float:
    return float(sum(event.weight for event in sequence.events))


def tp_weighted_score(sequence: ExtractedSequence) -> float:
    base = sum(event.weight for event in sequence.events)
    tp = sequence.turning_point
    tp_bonus = tp.weight * 2.0 if tp else 0.0

    if tp and sequence.events:
        tp_index = next(
            (
                index
                for index, (event, phase) in enumerate(zip(sequence.events, sequence.phases))
                if phase.name == "TURNING_POINT" and event.id == tp.id
            ),
            None,
        )
        if tp_index is None or len(sequence.events) == 1:
            pos = 0.5
        else:
            pos = tp_index / (len(sequence.events) - 1)
        position_penalty = 1.0 - abs(pos - 0.6) * 0.5
    else:
        position_penalty = 0.5

    return float(base + tp_bonus * position_penalty)
