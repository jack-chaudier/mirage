from __future__ import annotations

import math

from tagger import tag_chunk


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
