from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import narrativefield.extraction.api_server as api_server
from narrativefield.llm.config import PipelineConfig
from narrativefield.schema.events import Event, EventMetrics, EventType

app = api_server.app


@pytest.fixture(autouse=True)
def _reset_prose_results() -> None:
    api_server._prose_results.clear()
    yield
    api_server._prose_results.clear()


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


def _sample_events() -> list[Event]:
    events: list[Event] = []
    for i in range(4):
        events.append(
            Event(
                id=f"security_e{i:03d}",
                sim_time=float(i * 3),
                tick_id=i,
                order_in_tick=0,
                type=EventType.CHAT if i < 2 else EventType.CONFLICT,
                source_agent="victor",
                target_agents=["marcus"],
                location_id="dining_table",
                causal_links=[f"security_e{i-1:03d}"] if i > 0 else [],
                deltas=[],
                description=f"Security test event {i}",
                metrics=EventMetrics(tension=0.2 + i * 0.15, irony=0.2, significance=0.1),
            )
        )
    return events


def _generate_payload(
    *,
    run_id: str | None = None,
    config_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "selected_events": [e.to_dict() for e in _sample_events()],
        "protagonist_agent_id": "victor",
        "context": {
            "metadata": {"total_sim_time": 12.0},
            "agents": [],
            "scenes": [],
            "secrets": [],
        },
    }
    if run_id is not None:
        payload["resume_run_id"] = run_id
    if config_overrides is not None:
        payload["config_overrides"] = config_overrides
    return payload


def _call_generate_prose(
    client: TestClient,
    payload: dict[str, Any],
) -> tuple[Any, Any]:
    with patch("narrativefield.storyteller.narrator.SequentialNarrator") as mock_narrator_cls:
        mock_instance = MagicMock()

        async def _mock_gen(*args, **kwargs):
            return _mock_generation_result()

        mock_instance.generate = _mock_gen
        mock_narrator_cls.return_value = mock_instance
        response = client.post("/api/generate-prose", json=payload)
    return response, mock_narrator_cls


def test_generate_prose_rejects_disallowed_config_overrides(caplog: pytest.LogCaptureFixture) -> None:
    client = TestClient(app)
    payload = _generate_payload(
        config_overrides={
            "phase2_max_words_per_chunk": 1700,
            "checkpoint_dir": "/tmp/unsafe",
            "retry_max_attempts": 999,
        }
    )

    with caplog.at_level(logging.WARNING, logger=api_server.logger.name):
        response, mock_narrator_cls = _call_generate_prose(client, payload)

    assert response.status_code == 200, response.text
    config = mock_narrator_cls.call_args.kwargs["config"]
    defaults = PipelineConfig()

    assert config.phase2_max_words_per_chunk == 1700
    assert config.checkpoint_dir == defaults.checkpoint_dir
    assert config.retry_max_attempts == defaults.retry_max_attempts

    warning_text = "\n".join(record.getMessage() for record in caplog.records)
    assert "Rejected config override: checkpoint_dir" in warning_text
    assert "Rejected config override: retry_max_attempts" in warning_text


def test_generate_prose_rejects_out_of_range_allowed_override(caplog: pytest.LogCaptureFixture) -> None:
    client = TestClient(app)
    payload = _generate_payload(
        config_overrides={"phase2_creative_deep_max_tokens": 9_999_999}
    )

    with caplog.at_level(logging.WARNING, logger=api_server.logger.name):
        response, mock_narrator_cls = _call_generate_prose(client, payload)

    assert response.status_code == 200, response.text
    config = mock_narrator_cls.call_args.kwargs["config"]

    assert (
        config.phase2_creative_deep_max_tokens
        == PipelineConfig().phase2_creative_deep_max_tokens
    )

    warning_text = "\n".join(record.getMessage() for record in caplog.records)
    assert "Rejected config override: phase2_creative_deep_max_tokens" in warning_text
    assert "out of range" in warning_text


def test_generate_prose_applies_valid_allowed_config_overrides() -> None:
    client = TestClient(app)
    payload = _generate_payload(
        config_overrides={
            "phase2_max_words_per_chunk": 1800,
            "phase2_creative_max_tokens": 5000,
            "phase2_creative_deep_max_tokens": 6000,
            "phase2_use_extended_thinking_for_pivotal": False,
            "max_summary_words": 900,
        }
    )

    response, mock_narrator_cls = _call_generate_prose(client, payload)

    assert response.status_code == 200, response.text
    config = mock_narrator_cls.call_args.kwargs["config"]

    assert config.phase2_max_words_per_chunk == 1800
    assert config.phase2_creative_max_tokens == 5000
    assert config.phase2_creative_deep_max_tokens == 6000
    assert config.phase2_use_extended_thinking_for_pivotal is False
    assert config.max_summary_words == 900


def test_generate_prose_results_cache_eviction_keeps_latest_entries() -> None:
    client = TestClient(app)
    total_runs = api_server._MAX_PROSE_RESULTS + 10

    with patch("narrativefield.storyteller.narrator.SequentialNarrator") as mock_narrator_cls:
        mock_instance = MagicMock()

        async def _mock_gen(*args, **kwargs):
            return _mock_generation_result()

        mock_instance.generate = _mock_gen
        mock_narrator_cls.return_value = mock_instance

        for i in range(total_runs):
            run_id = f"run_{i:03d}"
            payload = _generate_payload(run_id=run_id)
            response = client.post("/api/generate-prose", json=payload)
            assert response.status_code == 200, response.text

    assert len(api_server._prose_results) == api_server._MAX_PROSE_RESULTS
    expected_ids = [f"run_{i:03d}" for i in range(10, total_runs)]
    assert list(api_server._prose_results.keys()) == expected_ids
