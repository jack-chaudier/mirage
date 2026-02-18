from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from random import Random

import pytest

from narrativefield.simulation import run as run_cli
from narrativefield.simulation.scenarios import create_dinner_party_world
from narrativefield.simulation.tick_loop import SimulationConfig, run_simulation


# Fields that must be identical for two runs to be considered deterministic.
_DETERMINISM_FIELDS = (
    "id",
    "type",
    "source_agent",
    "target_agents",
    "location_id",
    "causal_links",
    "deltas",
    "tick_id",
    "order_in_tick",
    "sim_time",
)


def _run_subprocess(seed: int, time_scale: float, output_path: str, hash_seed: str) -> None:
    """Run the simulation in a subprocess with a specific PYTHONHASHSEED."""
    import os

    env = {**os.environ, "PYTHONHASHSEED": hash_seed}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "narrativefield.simulation.run",
            "--seed",
            str(seed),
            "--output",
            output_path,
            "--time-scale",
            str(time_scale),
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(Path(__file__).resolve().parents[2]),
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Subprocess failed (seed={seed}, PYTHONHASHSEED={hash_seed}):\n{result.stderr}")


def _extract_determinism_fields(event: dict) -> dict:
    return {k: event[k] for k in _DETERMINISM_FIELDS}


def test_cross_process_determinism() -> None:
    """Run sim twice in separate subprocesses with same seed but different
    PYTHONHASHSEED values, and assert identical event logs.

    Sorted dict iterations in tick_loop.py and decision_engine.py ensure
    determinism regardless of hash seed. This test validates that guarantee.
    """
    for seed in (42, 99):
        with tempfile.TemporaryDirectory() as tmpdir:
            out1 = str(Path(tmpdir) / f"run1_seed{seed}.json")
            out2 = str(Path(tmpdir) / f"run2_seed{seed}.json")

            _run_subprocess(seed, time_scale=1.0, output_path=out1, hash_seed="12345")
            _run_subprocess(seed, time_scale=1.0, output_path=out2, hash_seed="67890")

            data1 = json.loads(Path(out1).read_text())
            data2 = json.loads(Path(out2).read_text())

            events1 = data1["events"]
            events2 = data2["events"]

            assert len(events1) == len(events2), (
                f"seed={seed}: event count mismatch: {len(events1)} vs {len(events2)}"
            )

            for i, (e1, e2) in enumerate(zip(events1, events2)):
                f1 = _extract_determinism_fields(e1)
                f2 = _extract_determinism_fields(e2)
                assert f1 == f2, (
                    f"seed={seed}, event index {i}: mismatch\n"
                    f"  run1: {f1}\n"
                    f"  run2: {f2}"
                )


def test_in_process_determinism() -> None:
    """Verify that two in-process runs with the same seed produce identical events."""
    for seed in (42, 77):
        events_a = _run_in_process(seed)
        events_b = _run_in_process(seed)

        assert len(events_a) == len(events_b), (
            f"seed={seed}: event count mismatch: {len(events_a)} vs {len(events_b)}"
        )

        for i, (a, b) in enumerate(zip(events_a, events_b)):
            da = a.to_dict()
            db = b.to_dict()
            for field in _DETERMINISM_FIELDS:
                assert da[field] == db[field], (
                    f"seed={seed}, event {i}, field '{field}': {da[field]} != {db[field]}"
                )


def test_time_scale_preserves_event_count() -> None:
    """time_scale should stretch sim_time without changing tick count or event count."""
    seed = 42
    events_1x = _run_in_process(seed, time_scale=1.0)
    events_4x = _run_in_process(seed, time_scale=4.0)

    assert len(events_1x) == len(events_4x), (
        f"Event count changed with time_scale: {len(events_1x)} vs {len(events_4x)}"
    )

    # Same tick structure.
    ticks_1x = [e.tick_id for e in events_1x]
    ticks_4x = [e.tick_id for e in events_4x]
    assert ticks_1x == ticks_4x, "Tick IDs should be identical across time_scale values"

    # Sim times should scale proportionally.
    for e1, e4 in zip(events_1x, events_4x):
        if e1.sim_time == 0.0:
            assert e4.sim_time == 0.0
        else:
            ratio = e4.sim_time / e1.sim_time
            assert abs(ratio - 4.0) < 0.01, (
                f"sim_time ratio should be ~4.0, got {ratio} "
                f"(event {e1.id}: {e1.sim_time} vs {e4.sim_time})"
            )


def test_time_scale_must_be_positive() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "out.json"
        with pytest.raises(ValueError, match=r"time_scale must be positive, got 0\.0"):
            run_cli.main(["--scenario", "dinner_party", "--output", str(out), "--time-scale", "0"])


def test_run_metadata_contains_deterministic_id_and_truncation_flag() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "out.json"
        run_cli.main(
            [
                "--scenario",
                "dinner_party",
                "--seed",
                "42",
                "--event-limit",
                "1",
                "--output",
                str(out),
            ]
        )
        payload = json.loads(out.read_text(encoding="utf-8"))
        metadata = payload["metadata"]
        assert metadata["deterministic_id"] == "dinner_party_seed_42"
        assert isinstance(metadata["truncated"], bool)


def _run_in_process(seed: int, time_scale: float = 1.0):
    world = create_dinner_party_world()
    rng = Random(seed)
    cfg = SimulationConfig(
        tick_limit=300,
        event_limit=200,
        max_sim_time=world.definition.sim_duration_minutes * time_scale,
        snapshot_interval_events=world.definition.snapshot_interval,
        time_scale=time_scale,
    )
    events, _ = run_simulation(world, rng, cfg)
    return events
