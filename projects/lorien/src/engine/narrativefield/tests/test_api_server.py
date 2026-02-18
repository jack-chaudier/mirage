from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

import narrativefield.extraction.api_server as api_server
from narrativefield.llm.config import PipelineConfig
from narrativefield.schema.events import Event, EventMetrics, EventType
from narrativefield.schema.world import Location, SecretDefinition
from narrativefield.storyteller.narrator import SequentialNarrator

app = api_server.app


def _mock_generation_result() -> SimpleNamespace:
    return SimpleNamespace(
        prose="Mock generated story.",
        word_count=3,
        scenes_generated=1,
        usage={
            "total_input_tokens": 100,
            "total_output_tokens": 200,
            "estimated_cost_usd": 0.01,
            "continuity_report": [],
            "llm_response_metadata": {},
            "model_history": [],
        },
        generation_time_seconds=0.1,
    )


def _assert_lorebook_ready_world_data(world_data: dict[str, Any]) -> None:
    assert world_data is not None
    assert "world_definition" in world_data
    assert "agents" in world_data
    assert "secrets" in world_data

    lorebook = SequentialNarrator._build_lorebook(world_data)
    assert lorebook is not None

    system_prompt = SequentialNarrator(config=PipelineConfig())._build_system_prompt(lorebook)
    assert "<full_cast>" in system_prompt
    assert "You are a literary fiction writer. Generate vivid prose for " not in system_prompt

    raw_agents = world_data.get("agents") or []
    character_ids: list[str] = []
    for item in raw_agents:
        if isinstance(item, dict) and item.get("id"):
            character_ids.append(str(item["id"]))
        elif hasattr(item, "id"):
            character_ids.append(str(item.id))
        if len(character_ids) >= 2:
            break

    world_definition = world_data["world_definition"]
    locations = world_definition.locations if hasattr(world_definition, "locations") else {}
    location_id = next(iter(locations.keys()), "dining_table")
    scene_context = lorebook.get_context_for_scene(character_ids, location_id)

    assert "<location " in scene_context
    assert "Relationships:" in scene_context


def test_extract_endpoint_smoke_with_inline_context() -> None:
    client = TestClient(app)

    events: list[Event] = []
    for i, (et, tension) in enumerate(
        [
            (EventType.CHAT, 0.1),
            (EventType.OBSERVE, 0.2),
            (EventType.CONFLICT, 0.7),
            (EventType.CATASTROPHE, 1.0),
            (EventType.CHAT, 0.3),
        ],
        1,
    ):
        events.append(
            Event(
                id=f"e{i:03d}",
                sim_time=float(i * 3),
                tick_id=i,
                order_in_tick=0,
                type=et,
                source_agent="a",
                target_agents=["b"] if et in {EventType.CHAT, EventType.CONFLICT} else [],
                location_id="dining_table",
                causal_links=[f"e{i-1:03d}"] if i > 1 else [],
                deltas=[],
                description=f"{et.value} {i}",
                metrics=EventMetrics(
                    tension=float(tension),
                    irony=0.2,
                    significance=0.0,
                    thematic_shift={},
                    tension_components={
                        "danger": 0.0,
                        "time_pressure": 0.0,
                        "goal_frustration": 0.0,
                        "relationship_volatility": 0.0,
                        "information_gap": 0.0,
                        "resource_scarcity": 0.0,
                        "moral_cost": 0.0,
                        "irony_density": 0.0,
                    },
                    irony_collapse=None,
                ),
            )
        )

    payload = {
        "selection_type": "region",
        "event_ids": [e.id for e in events],
        "tension_weights": {
            "danger": 0.125,
            "time_pressure": 0.125,
            "goal_frustration": 0.125,
            "relationship_volatility": 0.125,
            "information_gap": 0.125,
            "resource_scarcity": 0.125,
            "moral_cost": 0.125,
            "irony_density": 0.125,
        },
        "genre_preset": "default",
        "selected_events": [e.to_dict() for e in events],
        "context": {
            "metadata": {"total_sim_time": 120.0},
            "agents": [{"id": "a", "name": "A", "initial_location": "dining_table", "goal_summary": "", "primary_flaw": ""}],
            "scenes": [],
            "secrets": [],
        },
    }

    res = client.post("/extract", json=payload)
    assert res.status_code == 200, res.text
    data = res.json()
    assert "validation" in data
    assert "beats" in data
    assert "beat_sheet" in data or data["validation"]["valid"] is False


def test_extract_endpoint_auto_search_when_too_many_events() -> None:
    """
    If a region selection passes >50 events, the API should auto-switch to search mode
    and return a protagonist-centric arc with <= max_events beats.
    """
    client = TestClient(app)

    # Load a real simulation artifact and send the events inline so the test does not depend
    # on server-side payload preload.
    repo_root = Path(__file__).resolve().parents[4]
    sim_path = repo_root / "data" / "dinner_party_001.nf-sim.json"
    raw = json.loads(sim_path.read_text(encoding="utf-8"))
    events = [Event.from_dict(e) for e in (raw.get("events") or [])]
    meta = raw.get("metadata") or {}

    payload = {
        "selection_type": "region",
        "event_ids": [e.id for e in events],  # triggers auto-search
        "max_events": 20,
        "selected_events": [e.to_dict() for e in events],
        "context": {
            "metadata": {"total_sim_time": float(meta.get("total_sim_time") or 0.0)},
            "agents": [],
            "scenes": [],
            "secrets": [],
        },
    }

    res = client.post("/extract", json=payload)
    assert res.status_code == 200, res.text
    data = res.json()
    assert "validation" in data
    assert "beats" in data
    assert len(data["beats"]) <= 20, "Auto-search should downsample to <= max_events"
    if data["validation"]["valid"] is False:
        assert "diagnostics" in data
        diagnostics = data["diagnostics"]
        assert "rule_failure_counts" in diagnostics
        assert "best_candidate_violation_count" in diagnostics
        assert "candidates_evaluated" in diagnostics
        assert "best_candidate_violations" in diagnostics


def test_generate_prose_world_data_inline_path_builds_lorebook() -> None:
    client = TestClient(app)

    events: list[Event] = []
    for i in range(4):
        events.append(
            Event(
                id=f"inline_e{i:03d}",
                sim_time=float(i * 3),
                tick_id=i,
                order_in_tick=0,
                type=EventType.CHAT if i < 2 else EventType.CONFLICT,
                source_agent="thorne",
                target_agents=["victor"],
                location_id="dining_table",
                causal_links=[f"inline_e{i-1:03d}"] if i > 0 else [],
                deltas=[],
                description=f"Inline test event {i}",
                metrics=EventMetrics(tension=0.2 + i * 0.15, irony=0.2, significance=0.1),
            )
        )

    payload = {
        "selected_events": [e.to_dict() for e in events],
        "protagonist_agent_id": "thorne",
        "context": {
            "metadata": {
                "simulation_id": "inline_test_world",
                "scenario": "dinner_party",
                "total_sim_time": 12.0,
            },
            "agents": [
                {
                    "id": "thorne",
                    "name": "James Thorne",
                    "initial_location": "dining_table",
                    "goal_summary": "Maintain status and social order.",
                    "primary_flaw": "pride",
                },
                {
                    "id": "victor",
                    "name": "Victor Lang",
                    "initial_location": "dining_table",
                    "goal_summary": "Uncover hidden truths and contradictions.",
                    "primary_flaw": "obsession",
                },
            ],
            "locations": [
                {
                    "id": "dining_table",
                    "name": "Dining Table",
                    "privacy": 0.1,
                    "capacity": 6,
                    "adjacent": ["kitchen"],
                    "overhear_from": [],
                    "overhear_probability": 0.0,
                    "description": "The social center of the evening.",
                }
            ],
            "secrets": [
                {
                    "id": "secret_inline_01",
                    "holder": ["thorne"],
                    "about": "thorne",
                    "content_type": "business",
                    "description": "Thorne is hiding a financial crisis.",
                    "truth_value": True,
                    "initial_knowers": ["thorne"],
                    "initial_suspecters": ["victor"],
                    "dramatic_weight": 0.8,
                    "reveal_consequences": "Public humiliation and loss of control.",
                }
            ],
            "scenes": [],
        },
    }

    captured: dict[str, Any] = {}
    with patch("narrativefield.storyteller.narrator.SequentialNarrator") as MockNarrator:
        mock_instance = MagicMock()

        async def _mock_gen(*args, **kwargs):
            captured["world_data"] = kwargs.get("world_data")
            return _mock_generation_result()

        mock_instance.generate = _mock_gen
        MockNarrator.return_value = mock_instance

        res = client.post("/api/generate-prose", json=payload)

    assert res.status_code == 200, res.text
    _assert_lorebook_ready_world_data(captured["world_data"])


def test_generate_prose_world_data_loaded_payload_path_builds_lorebook() -> None:
    events: list[Event] = []
    for i in range(4):
        events.append(
            Event(
                id=f"loaded_e{i:03d}",
                sim_time=float(i * 3),
                tick_id=i,
                order_in_tick=0,
                type=EventType.CHAT if i < 2 else EventType.CONFLICT,
                source_agent="thorne",
                target_agents=["victor"],
                location_id="dining_table",
                causal_links=[f"loaded_e{i-1:03d}"] if i > 0 else [],
                deltas=[],
                description=f"Loaded test event {i}",
                metrics=EventMetrics(tension=0.2 + i * 0.15, irony=0.2, significance=0.1),
            )
        )

    payload_obj = api_server.LoadedPayload(
        metadata={
            "simulation_id": "loaded_test_world",
            "scenario": "dinner_party",
            "total_sim_time": 12.0,
            "total_ticks": 24,
        },
        agents_manifest={
            "thorne": {
                "id": "thorne",
                "name": "James Thorne",
                "initial_location": "dining_table",
                "goal_summary": "Maintain status and social order.",
                "primary_flaw": "pride",
            },
            "victor": {
                "id": "victor",
                "name": "Victor Lang",
                "initial_location": "dining_table",
                "goal_summary": "Uncover hidden truths and contradictions.",
                "primary_flaw": "obsession",
            },
        },
        agent_states={},
        events_by_id={e.id: e for e in events},
        scenes=[],
        secrets={
            "secret_loaded_01": SecretDefinition.from_dict(
                {
                    "id": "secret_loaded_01",
                    "holder": ["thorne"],
                    "about": "thorne",
                    "content_type": "business",
                    "description": "Thorne is hiding a financial crisis.",
                    "truth_value": True,
                    "initial_knowers": ["thorne"],
                    "initial_suspecters": ["victor"],
                    "dramatic_weight": 0.8,
                    "reveal_consequences": "Public humiliation and loss of control.",
                }
            )
        },
        locations={
            "dining_table": Location.from_dict(
                {
                    "id": "dining_table",
                    "name": "Dining Table",
                    "privacy": 0.1,
                    "capacity": 6,
                    "adjacent": ["kitchen"],
                    "overhear_from": [],
                    "overhear_probability": 0.0,
                    "description": "The social center of the evening.",
                }
            )
        },
    )

    captured: dict[str, Any] = {}
    original_loaded_payload = api_server._loaded_payload
    try:
        with TestClient(app) as client:
            api_server._loaded_payload = payload_obj
            with patch("narrativefield.storyteller.narrator.SequentialNarrator") as MockNarrator:
                mock_instance = MagicMock()

                async def _mock_gen(*args, **kwargs):
                    captured["world_data"] = kwargs.get("world_data")
                    return _mock_generation_result()

                mock_instance.generate = _mock_gen
                MockNarrator.return_value = mock_instance

                res = client.post(
                    "/api/generate-prose",
                    json={"event_ids": [e.id for e in events], "protagonist_agent_id": "thorne"},
                )
    finally:
        api_server._loaded_payload = original_loaded_payload

    assert res.status_code == 200, res.text
    _assert_lorebook_ready_world_data(captured["world_data"])
