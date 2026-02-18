from __future__ import annotations

from scripts.phase2_experiment import _build_k_sweep_lookup, _interpolate_profile, _k_sweep_key, _validity_adjusted_score


def test_interpolate_profile_endpoints_and_threshold_commitments() -> None:
    baseline = {
        "goals_scalar": {"safety": 0.2, "status": 0.8},
        "closeness": {"elena": 0.4},
        "relationships": {"elena": {"trust": 0.1, "affection": 0.2, "obligation": 0.3}},
        "commitments": ["base"],
    }
    evolved = {
        "goals_scalar": {"safety": 0.8, "status": 0.2},
        "closeness": {"elena": -0.4},
        "relationships": {"elena": {"trust": 0.9, "affection": -0.2, "obligation": 0.1}},
        "commitments": ["evolved"],
    }

    at_zero = _interpolate_profile(baseline, evolved, 0.0)
    assert at_zero["goals_scalar"]["safety"] == 0.2
    assert at_zero["closeness"]["elena"] == 0.4
    assert at_zero["relationships"]["elena"]["trust"] == 0.1
    assert at_zero["commitments"] == ["base"]

    at_one = _interpolate_profile(baseline, evolved, 1.0)
    assert at_one["goals_scalar"]["safety"] == 0.8
    assert at_one["closeness"]["elena"] == -0.4
    assert at_one["relationships"]["elena"]["trust"] == 0.9
    assert at_one["commitments"] == ["evolved"]

    at_half_below = _interpolate_profile(baseline, evolved, 0.49)
    at_half_above = _interpolate_profile(baseline, evolved, 0.5)
    assert at_half_below["commitments"] == ["base"]
    assert at_half_above["commitments"] == ["evolved"]


def test_interpolate_profile_uses_union_for_missing_targets() -> None:
    baseline = {
        "goals_scalar": {"safety": 0.2},
        "closeness": {"elena": 0.2},
        "relationships": {"elena": {"trust": 0.3, "affection": 0.1, "obligation": 0.0}},
        "commitments": [],
    }
    evolved = {
        "goals_scalar": {"safety": 0.6},
        "closeness": {"elena": 0.6, "marcus": 0.5},
        "relationships": {
            "elena": {"trust": 0.7, "affection": 0.2, "obligation": 0.3},
            "marcus": {"trust": 0.4, "affection": 0.4, "obligation": 0.4},
        },
        "commitments": [],
    }

    mid = _interpolate_profile(baseline, evolved, 0.5)
    assert set(mid["closeness"].keys()) == {"elena", "marcus"}
    assert set(mid["relationships"].keys()) == {"elena", "marcus"}
    assert mid["closeness"]["marcus"] == 0.25
    assert mid["relationships"]["marcus"]["trust"] == 0.2


def test_validity_adjusted_score_penalizes_invalid_arcs() -> None:
    assert _validity_adjusted_score(0.9, 6) == 0.9
    assert _validity_adjusted_score(0.9, 3) == 0.45
    assert _validity_adjusted_score(0.9, 0) == 0.0


def test_k_sweep_lookup_keys_are_order_insensitive_for_agents() -> None:
    row = {
        "seed": 11,
        "condition_type": "subset",
        "k": 5,
        "evolved_agents": ["elena", "marcus", "lydia", "diana", "victor"],
        "mean_score": 0.5,
    }
    lookup = _build_k_sweep_lookup([row])

    key = _k_sweep_key(11, "subset", 5, ("victor", "diana", "lydia", "marcus", "elena"))
    assert key in lookup
    assert lookup[key]["mean_score"] == 0.5
