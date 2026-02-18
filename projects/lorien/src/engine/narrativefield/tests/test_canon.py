from __future__ import annotations

from narrativefield.metrics.significance import compute_significance
from narrativefield.schema.canon import CanonArtifact, CanonFaction, LocationMemory, WorldCanon
from narrativefield.schema.events import DeltaKind, Event, EventEntities, EventType


def test_world_canon_roundtrip() -> None:
    canon = WorldCanon(
        world_id="dinner_party_01",
        canon_version=5,
        last_tick=42,
        last_event_id="evt_0042",
        location_memory={
            "kitchen": LocationMemory(
                tension_residue=0.42,
                notable_event_ids=["evt_0010", "evt_0020"],
                visit_count=9,
                last_event_tick=40,
            )
        },
        artifacts={
            "ledger": CanonArtifact(
                id="ledger",
                name="Black Ledger",
                artifact_type="document",
                state={"owner": "marcus", "authentic": True},
                first_seen_event_id="evt_0007",
                history=["evt_0007", "evt_0025"],
            )
        },
        factions={
            "board": CanonFaction(
                id="board",
                name="Investment Board",
                state={"influence": 0.8},
                members=["thorne", "marcus"],
                relationships={"press": -0.2},
                history=["evt_0011"],
            )
        },
        claim_states={
            "secret_affair_01": {
                "elena": "believes_true",
                "thorne": "suspects",
            }
        },
    )

    assert WorldCanon.from_dict(canon.to_dict()).to_dict() == canon.to_dict()


def test_location_memory_roundtrip() -> None:
    mem = LocationMemory(
        tension_residue=0.9,
        notable_event_ids=["evt_0002"],
        visit_count=4,
        last_event_tick=12,
    )
    assert LocationMemory.from_dict(mem.to_dict()).to_dict() == mem.to_dict()


def test_canon_artifact_roundtrip() -> None:
    artifact = CanonArtifact(
        id="knife_01",
        name="Carving Knife",
        artifact_type="weapon",
        state={"condition": "stained", "owner": "thorne"},
        first_seen_event_id="evt_0009",
        history=["evt_0009", "evt_0012"],
    )
    assert CanonArtifact.from_dict(artifact.to_dict()).to_dict() == artifact.to_dict()


def test_canon_faction_roundtrip() -> None:
    faction = CanonFaction(
        id="household",
        name="Thorne Household",
        state={"reputation": 0.3, "wealth": 0.6},
        members=["thorne", "elena"],
        relationships={"press": -0.4},
        history=["evt_0005"],
    )
    assert CanonFaction.from_dict(faction.to_dict()).to_dict() == faction.to_dict()


def test_event_entities_roundtrip() -> None:
    entities = EventEntities(
        locations=["kitchen", "balcony"],
        artifacts=["ledger"],
        factions=["board"],
        institutions=["court"],
        claims=["secret_affair_01"],
        concepts=["betrayal", "romance"],
    )
    assert EventEntities.from_dict(entities.to_dict()).to_dict() == entities.to_dict()


def test_event_entities_empty_serialization() -> None:
    assert EventEntities().to_dict() == {}


def test_event_entities_from_none() -> None:
    entities = EventEntities.from_dict(None)
    assert entities.total_refs == 0
    assert entities.to_dict() == {}


def test_event_backward_compat() -> None:
    raw = {
        "id": "evt_legacy",
        "sim_time": 1.0,
        "tick_id": 1,
        "order_in_tick": 0,
        "type": "chat",
        "source_agent": "thorne",
        "target_agents": ["elena"],
        "location_id": "dining_table",
        "causal_links": [],
        "deltas": [],
        "description": "legacy event",
        "dialogue": None,
        "content_metadata": None,
        "beat_type": None,
        "metrics": {
            "tension": 0.0,
            "irony": 0.0,
            "significance": 0.0,
            "thematic_shift": {},
            "tension_components": {},
            "irony_collapse": None,
        },
    }

    event = Event.from_dict(raw)
    assert event.entities.total_refs == 0
    assert "entities" not in event.to_dict()


def test_new_delta_kinds_exist() -> None:
    assert DeltaKind.ARTIFACT_STATE.value == "artifact_state"
    assert DeltaKind.FACTION_STATE.value == "faction_state"
    assert DeltaKind.INSTITUTION_STATE.value == "institution_state"
    assert DeltaKind.LOCATION_MEMORY.value == "location_memory"


def test_significance_entity_breadth() -> None:
    plain = Event(
        id="evt_plain",
        sim_time=1.0,
        tick_id=1,
        order_in_tick=0,
        type=EventType.CHAT,
        source_agent="thorne",
        target_agents=["elena"],
        location_id="dining_table",
        causal_links=[],
        deltas=[],
        description="small chat",
    )
    with_entities = Event(
        id="evt_rich",
        sim_time=1.0,
        tick_id=1,
        order_in_tick=0,
        type=EventType.CHAT,
        source_agent="thorne",
        target_agents=["elena"],
        location_id="dining_table",
        causal_links=[],
        deltas=[],
        description="small chat",
        entities=EventEntities(
            locations=["dining_table", "kitchen"],
            claims=["secret_affair_01"],
            concepts=["betrayal", "romance"],
        ),
    )

    score_plain = compute_significance(plain, [plain])
    score_rich = compute_significance(with_entities, [with_entities])
    assert score_rich > score_plain
