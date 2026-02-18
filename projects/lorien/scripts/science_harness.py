from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FORMAT_VERSION = "1.0.0"

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


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _engine_root(repo_root: Path) -> Path:
    return repo_root / "src" / "engine"


def _find_engine_python(repo_root: Path) -> str:
    override = os.environ.get("NARRATIVEFIELD_ENGINE_PYTHON")
    if override and Path(override).exists():
        return override

    candidate = _engine_root(repo_root) / ".venv" / "bin" / "python"
    if candidate.exists():
        return str(candidate)

    # Fallback: allow running inside an activated engine venv.
    return sys.executable


def _to_repo_relative_path(path_str: str, repo_root: Path) -> str:
    """Prefer repo-relative paths in serialized reports."""
    p = Path(path_str)
    if not p.is_absolute():
        return p.as_posix()
    repo_abs = repo_root.resolve()
    try:
        # Do not resolve symlinks on p; keep repo-local virtualenv paths stable.
        return p.relative_to(repo_abs).as_posix()
    except ValueError:
        # If the interpreter is outside the repo, keep an absolute path.
        return p.as_posix()


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Command failed:\n  cmd={cmd}\n  stderr={result.stderr}")


def _event_stream_fingerprint(events: list[dict[str, Any]]) -> str:
    h = hashlib.sha256()
    for e in events:
        snippet = {k: e.get(k) for k in _DETERMINISM_FIELDS}
        h.update(
            json.dumps(snippet, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        )
        h.update(b"\n")
    return h.hexdigest()


def _location_bucket(event: dict[str, Any]) -> str:
    """Bucket events by where they 'happen' for coarse location-share reporting.

    SOCIAL_MOVE events encode destination in content_metadata["destination"]; for
    all other events we use location_id directly.
    """

    etype = event.get("type")
    if etype == "social_move":
        meta = event.get("content_metadata") or {}
        dest = meta.get("destination")
        if isinstance(dest, str) and dest:
            return dest
    loc = event.get("location_id")
    return str(loc) if loc else ""


def _location_distribution(events: list[dict[str, Any]]) -> dict[str, float]:
    counts: dict[str, int] = {}
    for e in events:
        loc = _location_bucket(e)
        if not loc:
            continue
        counts[loc] = counts.get(loc, 0) + 1
    total = sum(counts.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in sorted(counts.items(), key=lambda kv: kv[0])}


def _catastrophes_by_agent(events: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for e in events:
        if e.get("type") != "catastrophe":
            continue
        src = e.get("source_agent")
        if not isinstance(src, str) or not src:
            continue
        out[src] = out.get(src, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def _rank_representative_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank runs for portfolio/review presentation.

    Ranking priority:
    1) catastrophe_count (desc)
    2) catastrophe_agent_diversity (desc)
    3) locations_used_count (desc)
    4) total_sim_time (desc)
    5) seed (asc) for deterministic tie-breaking
    """

    scored: list[tuple[tuple[int, int, int, float, int], dict[str, Any]]] = []
    for run in runs:
        seed = int(run.get("seed") or 0)
        catastrophe_count = int(run.get("catastrophe_count") or 0)
        by_agent = run.get("catastrophes_by_agent")
        catastrophe_agent_diversity = len(by_agent) if isinstance(by_agent, dict) else 0
        locations_used_count = int(run.get("locations_used_count") or 0)
        total_sim_time = float(run.get("total_sim_time") or 0.0)
        # Invert seed in the score tuple because we sort reverse=True below.
        score = (
            catastrophe_count,
            catastrophe_agent_diversity,
            locations_used_count,
            total_sim_time,
            -seed,
        )
        scored.append((score, run))

    ranked = sorted(scored, key=lambda item: item[0], reverse=True)
    out: list[dict[str, Any]] = []
    for rank, (score, run) in enumerate(ranked, start=1):
        out.append(
            {
                "rank": rank,
                "seed": int(run.get("seed") or 0),
                "score": {
                    "catastrophe_count": score[0],
                    "catastrophe_agent_diversity": score[1],
                    "locations_used_count": score[2],
                    "total_sim_time": score[3],
                },
            }
        )
    return out


def _parse_explicit_seeds(raw: str) -> list[int]:
    parts = [part.strip() for part in raw.split(",")]
    if not parts or (len(parts) == 1 and parts[0] == ""):
        raise SystemExit('--seeds requires a comma-separated list of integers, e.g. "1,2,3"')

    seeds: list[int] = []
    for part in parts:
        if not part:
            raise SystemExit('--seeds cannot contain empty items, e.g. use "1,2,3"')
        try:
            seeds.append(int(part))
        except ValueError as exc:
            raise SystemExit(f'Invalid seed "{part}" in --seeds; expected an integer') from exc

    # Keep first occurrence order while dropping duplicates for deterministic runs.
    deduped: list[int] = []
    seen: set[int] = set()
    for seed in seeds:
        if seed in seen:
            continue
        seen.add(seed)
        deduped.append(seed)
    return deduped


def _simulate_one(
    *,
    engine_python: str,
    engine_root: Path,
    scenario: str,
    seed: int,
    event_limit: int,
    tick_limit: int,
    time_scale: float,
    output_path: Path,
    pyhashseed: str | None = None,
) -> dict[str, Any]:
    env = os.environ.copy()
    if pyhashseed is not None:
        env["PYTHONHASHSEED"] = pyhashseed

    _run(
        [
            engine_python,
            "-m",
            "narrativefield.simulation.run",
            "--scenario",
            scenario,
            "--output",
            str(output_path),
            "--seed",
            str(seed),
            "--event-limit",
            str(event_limit),
            "--tick-limit",
            str(tick_limit),
            "--time-scale",
            str(time_scale),
        ],
        cwd=engine_root,
        env=env,
    )

    return json.loads(output_path.read_text(encoding="utf-8"))


def _bundle_one(
    *,
    engine_python: str,
    engine_root: Path,
    sim_path: Path,
    viz_path: Path,
) -> dict[str, Any]:
    _run(
        [
            engine_python,
            "-m",
            "narrativefield.metrics.pipeline",
            "--input",
            str(sim_path),
            "--output",
            str(viz_path),
        ],
        cwd=engine_root,
    )
    return json.loads(viz_path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="science_harness")
    parser.add_argument("--scenario", default="dinner_party", choices=["dinner_party"])
    parser.add_argument("--seeds", default=None, help='Explicit comma-separated seeds, e.g. "1,2,3".')
    parser.add_argument("--seed-from", type=int, default=50)
    parser.add_argument("--seed-to", type=int, default=64)
    parser.add_argument("--event-limit", type=int, default=200)
    parser.add_argument("--tick-limit", type=int, default=300)
    parser.add_argument("--time-scale", type=float, default=4.0)
    parser.add_argument("--output", default="data/science_report.json")
    parser.add_argument("--skip-metrics", action="store_true", help="Only run simulation, not metrics/bundling.")
    args = parser.parse_args(argv)

    repo_root = _repo_root()
    engine_root = _engine_root(repo_root)
    engine_python = _find_engine_python(repo_root)

    explicit_seed_mode = args.seeds is not None
    if explicit_seed_mode:
        seeds = _parse_explicit_seeds(str(args.seeds))
    else:
        if int(args.seed_to) < int(args.seed_from):
            raise SystemExit("--seed-to must be >= --seed-from")
        seeds = list(range(int(args.seed_from), int(args.seed_to) + 1))

    if not seeds:
        raise SystemExit("No seeds requested")

    generated_at = datetime.now(timezone.utc).isoformat()

    runs: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Cross-process determinism probe (one seed, two different PYTHONHASHSEED values).
        det_seed = seeds[0]
        det_out_a = tmp / f"det_seed{det_seed}_a.nf-sim.json"
        det_out_b = tmp / f"det_seed{det_seed}_b.nf-sim.json"

        det_a = _simulate_one(
            engine_python=engine_python,
            engine_root=engine_root,
            scenario=args.scenario,
            seed=det_seed,
            event_limit=args.event_limit,
            tick_limit=args.tick_limit,
            time_scale=args.time_scale,
            output_path=det_out_a,
            pyhashseed="12345",
        )
        det_b = _simulate_one(
            engine_python=engine_python,
            engine_root=engine_root,
            scenario=args.scenario,
            seed=det_seed,
            event_limit=args.event_limit,
            tick_limit=args.tick_limit,
            time_scale=args.time_scale,
            output_path=det_out_b,
            pyhashseed="67890",
        )

        det_hash_a = _event_stream_fingerprint(list(det_a.get("events") or []))
        det_hash_b = _event_stream_fingerprint(list(det_b.get("events") or []))

        determinism_check = {
            "seed": det_seed,
            "pyhashseeds": ["12345", "67890"],
            "event_stream_sha256": [det_hash_a, det_hash_b],
            "matches": det_hash_a == det_hash_b,
        }

        for seed in seeds:
            sim_path = tmp / f"seed{seed}.nf-sim.json"
            viz_path = tmp / f"seed{seed}.nf-viz.json"

            sim = _simulate_one(
                engine_python=engine_python,
                engine_root=engine_root,
                scenario=args.scenario,
                seed=seed,
                event_limit=args.event_limit,
                tick_limit=args.tick_limit,
                time_scale=args.time_scale,
                output_path=sim_path,
            )

            viz: dict[str, Any] | None = None
            if not args.skip_metrics:
                viz = _bundle_one(
                    engine_python=engine_python,
                    engine_root=engine_root,
                    sim_path=sim_path,
                    viz_path=viz_path,
                )

            events = list(sim.get("events") or [])
            meta = dict(sim.get("metadata") or {})

            catastrophe_count = sum(1 for e in events if e.get("type") == "catastrophe")
            locations_used = sorted({str(e.get("location_id")) for e in events if e.get("location_id")})
            location_distribution = _location_distribution(events)
            catastrophes_by_agent = _catastrophes_by_agent(events)
            run_summary: dict[str, Any] = {
                "seed": seed,
                "time_scale": float(meta.get("time_scale") or args.time_scale),
                "total_sim_time": float(meta.get("total_sim_time") or 0.0),
                "total_ticks": int(meta.get("total_ticks") or 0),
                "event_count": int(meta.get("event_count") or len(events)),
                "catastrophe_count": catastrophe_count,
                "catastrophes_by_agent": catastrophes_by_agent,
                "locations_used_count": len(locations_used),
                "locations_used": locations_used,
                "location_distribution": location_distribution,
                "event_stream_sha256": _event_stream_fingerprint(events),
                "determinism_metadata": {
                    "python_version": str(meta.get("python_version") or ""),
                    "git_commit": meta.get("git_commit"),
                    "config_hash": str(meta.get("config_hash") or ""),
                },
            }

            if viz is not None:
                run_summary["scene_count"] = len(list(viz.get("scenes") or []))

            runs.append(run_summary)

    mean_total_sim_time = sum(r["total_sim_time"] for r in runs) / max(1, len(runs))

    runs_with_catastrophe = sum(1 for r in runs if int(r.get("catastrophe_count") or 0) > 0)
    total_catastrophes = sum(int(r.get("catastrophe_count") or 0) for r in runs)
    catastrophe_sources: dict[str, int] = {}
    for r in runs:
        for aid, n in (r.get("catastrophes_by_agent") or {}).items():
            catastrophe_sources[str(aid)] = catastrophe_sources.get(str(aid), 0) + int(n)
    dominant_catastrophe_agent = None
    dominant_catastrophe_share = 0.0
    if total_catastrophes > 0 and catastrophe_sources:
        dominant_catastrophe_agent = max(catastrophe_sources.items(), key=lambda kv: kv[1])[0]
        dominant_catastrophe_share = catastrophe_sources[dominant_catastrophe_agent] / total_catastrophes

    # Mean location share across runs (missing keys treated as 0).
    all_locations: set[str] = set()
    for r in runs:
        all_locations.update((r.get("location_distribution") or {}).keys())
    mean_location_share: dict[str, float] = {}
    for loc in sorted(all_locations):
        mean_location_share[loc] = sum(float((r.get("location_distribution") or {}).get(loc, 0.0)) for r in runs) / max(
            1, len(runs)
        )

    representative_seed_ranking = _rank_representative_runs(runs)
    representative_seed_top3 = representative_seed_ranking[:3]

    parameters: dict[str, Any] = {
        "scenario": args.scenario,
        "seeds": seeds,
        "explicit_seed_mode": explicit_seed_mode,
        "event_limit": int(args.event_limit),
        "tick_limit": int(args.tick_limit),
        "time_scale": float(args.time_scale),
    }
    if not explicit_seed_mode:
        parameters["seed_from"] = int(args.seed_from)
        parameters["seed_to"] = int(args.seed_to)

    report: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "generated_at": generated_at,
        "parameters": parameters,
        "environment": {
            "engine_python": _to_repo_relative_path(engine_python, repo_root),
        },
        "determinism_check": determinism_check,
        "aggregate": {
            "run_count": len(runs),
            "mean_total_sim_time": mean_total_sim_time,
            "min_total_sim_time": min(r["total_sim_time"] for r in runs),
            "max_total_sim_time": max(r["total_sim_time"] for r in runs),
            "runs_with_catastrophe": runs_with_catastrophe,
            "total_catastrophes": total_catastrophes,
            "dominant_catastrophe_agent": dominant_catastrophe_agent,
            "dominant_catastrophe_share": dominant_catastrophe_share,
            "catastrophes_by_agent": dict(sorted(catastrophe_sources.items(), key=lambda kv: kv[0])),
            "mean_location_share": mean_location_share,
            "representative_seed_ranking": representative_seed_ranking,
            "representative_seed_top3": representative_seed_top3,
        },
        "runs": runs,
    }

    out_path = repo_root / str(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
