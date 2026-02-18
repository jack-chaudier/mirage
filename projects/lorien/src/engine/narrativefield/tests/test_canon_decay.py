from __future__ import annotations

import copy

import pytest

from narrativefield.schema.canon import CanonTexture, LocationMemory, WorldCanon, decay_canon


def _base_canon() -> WorldCanon:
    return WorldCanon(
        location_memory={
            "dining_table": LocationMemory(tension_residue=0.85),
            "kitchen": LocationMemory(tension_residue=0.04),
        },
        claim_states={
            "secret_affair_01": {
                "elena": "believes_true",
                "thorne": "unknown",
            },
            "secret_blackmail_01": {
                "victor": "suspects",
            },
        },
        claim_confidence={
            "secret_affair_01": {"elena": 1.0, "thorne": 0.5},
            "secret_blackmail_01": {"victor": 0.09},
        },
        texture={
            "run_01__tf_0_0": CanonTexture(
                id="run_01__tf_0_0",
                statement="Victor keeps glancing at the decanter.",
                entity_refs=["victor", "dining_table"],
                detail_type="gesture",
                source_story_id="run_01",
                source_scene_index=0,
                committed_at_canon_version=1,
            )
        },
    )


def test_decay_canon_noop_rates_leave_state_unchanged() -> None:
    canon = _base_canon()
    before = copy.deepcopy(canon.to_dict())

    returned = decay_canon(canon, tension_decay=1.0, belief_decay=1.0)

    assert returned is canon
    assert canon.to_dict() == before


def test_decay_canon_tension_decay_applies_multiplier() -> None:
    canon = _base_canon()

    decay_canon(canon, tension_decay=0.6, belief_decay=1.0)

    assert canon.location_memory["dining_table"].tension_residue == pytest.approx(0.51, abs=1e-3)


def test_decay_canon_belief_confidence_decays() -> None:
    canon = _base_canon()

    decay_canon(canon, tension_decay=1.0, belief_decay=0.85)

    assert canon.claim_confidence["secret_affair_01"]["elena"] == pytest.approx(0.85, abs=1e-6)


def test_decay_canon_prunes_low_confidence_and_zeros_low_tension() -> None:
    canon = WorldCanon(
        location_memory={"kitchen": LocationMemory(tension_residue=0.06)},
        claim_states={"secret_blackmail_01": {"victor": "suspects"}},
        claim_confidence={"secret_blackmail_01": {"victor": 0.11}},
    )

    decay_canon(canon, tension_decay=0.6, belief_decay=0.85)

    assert canon.location_memory["kitchen"].tension_residue == 0.0
    assert canon.claim_states["secret_blackmail_01"]["victor"] == "unknown"
    assert "secret_blackmail_01" not in canon.claim_confidence


def test_decay_canon_does_not_modify_texture_facts() -> None:
    canon = _base_canon()
    before_texture = copy.deepcopy(canon.texture)
    before_count = len(canon.texture)

    decay_canon(canon, tension_decay=0.6, belief_decay=0.85)

    assert len(canon.texture) == before_count
    assert canon.texture == before_texture


def test_decay_canon_compounds_over_multiple_applications() -> None:
    canon = WorldCanon(
        location_memory={"dining_table": LocationMemory(tension_residue=1.0)},
        claim_states={"secret_affair_01": {"elena": "believes_true"}},
        claim_confidence={"secret_affair_01": {"elena": 1.0}},
    )

    for _ in range(3):
        decay_canon(canon, tension_decay=0.6, belief_decay=0.85)

    assert canon.location_memory["dining_table"].tension_residue == pytest.approx(0.216, abs=1e-6)
    assert canon.claim_confidence["secret_affair_01"]["elena"] == pytest.approx(0.614125, abs=1e-6)
