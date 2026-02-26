from __future__ import annotations

import math

from tagger import tag_chunk, tag_chunk_detailed


def test_role_hint_overrides_heuristics() -> None:
    role, weight = tag_chunk("this text would normally be noise", role_hint="pivot")
    assert role == "pivot"
    assert weight == 10.0

    role, weight = tag_chunk("please build a feature", role_hint="predecessor")
    assert role == "predecessor"
    assert math.isinf(weight) and weight < 0


def test_heuristic_pivot_and_predecessor_signals() -> None:
    role, weight = tag_chunk("Please implement the new feature and build migration.")
    assert role == "pivot"
    assert weight > 0

    role, weight = tag_chunk("Error: failing tests and traceback appeared in CI.")
    assert role == "predecessor"
    assert math.isinf(weight) and weight < 0


def test_noise_when_no_signal_present() -> None:
    role, weight = tag_chunk("small neutral note")
    assert role == "noise"
    assert math.isinf(weight) and weight < 0


def test_structural_signals_add_confidence() -> None:
    tagged = tag_chunk_detailed(
        text="Please build the API and implement auth.",
        speaker="user",
        index=0,
        total=8,
    )
    assert tagged.role == "pivot"
    assert tagged.confidence > 0.6
    assert any("signal" in reason or "position" in reason for reason in tagged.reasons)
