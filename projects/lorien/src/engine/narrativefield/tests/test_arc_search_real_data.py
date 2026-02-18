from __future__ import annotations

import json
from pathlib import Path

from narrativefield.extraction.arc_search import search_arc
from narrativefield.schema.events import Event


def _repo_root() -> Path:
    # /src/engine/narrativefield/tests -> repo root is parents[4]
    return Path(__file__).resolve().parents[4]


def test_arc_search_on_real_sim_artifact_returns_subset_or_diagnostics() -> None:
    """
    ArcSearch should be robust on a real sim artifact.

    The artifact has 200 events; search should downsample to <=20 and either:
    - return a valid arc, or
    - return diagnostics explaining why the best candidate failed.
    """
    sim_path = _repo_root() / "data" / "dinner_party_001.nf-sim.json"
    raw = json.loads(sim_path.read_text(encoding="utf-8"))
    events = [Event.from_dict(e) for e in (raw.get("events") or [])]
    meta = raw.get("metadata") or {}

    result = search_arc(
        all_events=events,
        time_start=0.0,
        time_end=float(meta.get("total_sim_time") or 0.0) or None,
        max_events=20,
        total_sim_time=float(meta.get("total_sim_time") or 0.0) or None,
    )

    assert 0 < len(result.events) <= 20
    assert result.protagonist
    assert len(result.events) == len(result.beats)

    if result.validation.valid:
        assert result.diagnostics is None
    else:
        assert result.diagnostics is not None
        assert result.diagnostics.violations
        assert result.diagnostics.suggested_keep_ids

