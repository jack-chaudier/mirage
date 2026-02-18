"""Goal evolution experiment: test whether evolving stale agent goals restores arc quality.

Run:
    cd src/engine && ./.venv/bin/python -m scripts.test_goal_evolution
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from random import Random
from typing import Any, Iterable

from narrativefield.extraction.rashomon import RashomonSet, extract_rashomon_set
from narrativefield.metrics.pipeline import parse_simulation_output, run_metrics_pipeline
from narrativefield.schema.agents import BeliefState, RelationshipState
from narrativefield.schema.canon import WorldCanon
from narrativefield.schema.events import BeatType, Event
from narrativefield.simulation.scenarios import create_dinner_party_world
from narrativefield.simulation.tick_loop import SimulationConfig, init_canon_from_world, run_simulation

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
WOUND_ANALYSIS_PATH = OUTPUT_DIR / "wound_analysis_1_50.json"
CANON_AFTER_B_PATH = OUTPUT_DIR / "canon_after_B.json"
DINNER_PARTY_AGENTS = ["thorne", "elena", "marcus", "lydia", "diana", "victor"]
REPORT_EVENT_TYPES = [
    "chat",
    "conflict",
    "reveal",
    "lie",
    "confide",
    "catastrophe",
    "physical",
    "observe",
]
EPSILON = 1e-12


@dataclass(frozen=True)
class StoryRun:
    label: str
    seed: int
    payload: dict[str, Any]
    events: list[Event]
    start_beliefs: dict[str, dict[str, str]]
    start_location_memory: dict[str, float]
    final_claim_states: dict[str, dict[str, str]]
    final_location_memory: dict[str, float]


@dataclass(frozen=True)
class WoundPattern:
    pattern: str
    agent_pair: tuple[str, str]
    location_id: str
    frequency: float


@dataclass(frozen=True)
class StoryMetrics:
    mean_score: float
    valid_count: int
    total_arcs: int
    per_agent_scores: dict[str, float | None]
    overlap_matrix: dict[str, float]
    wound_presence: dict[str, bool]
    wound_present_set: set[str]


@dataclass(frozen=True)
class Divergence:
    index: int
    tick_id: int | None
    baseline_event_id: str | None
    candidate_event_id: str | None


@dataclass(frozen=True)
class AgentEvolution:
    goals_scalar: dict[str, float]
    closeness: dict[str, float]
    relationships: dict[str, dict[str, float]]
    commitments: tuple[str, ...] | None = None


@dataclass(frozen=True)
class ConditionSpec:
    key: str
    label: str
    loaded_canon: WorldCanon | None
    evolutions: dict[str, AgentEvolution]


@dataclass(frozen=True)
class ConditionResult:
    key: str
    label: str
    story: StoryRun
    metrics: StoryMetrics
    event_type_counts: dict[str, int]
    mutation_details: dict[str, list[dict[str, Any]]]


def _resolve_output_path(path_arg: str) -> Path:
    path = Path(path_arg)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _pair_key(agent_a: str, agent_b: str) -> tuple[str, str]:
    left, right = sorted((str(agent_a), str(agent_b)))
    return (left, right)


def _parse_wound_pattern(raw_pattern: str) -> tuple[tuple[str, str], str]:
    if " @ " not in raw_pattern:
        raise ValueError(f"Invalid wound pattern '{raw_pattern}': expected '<a>-<b> @ <location>'.")
    pair_text, location_id = raw_pattern.split(" @ ", 1)
    pair_parts = pair_text.split("-", 1)
    if len(pair_parts) != 2:
        raise ValueError(f"Invalid wound pair in '{raw_pattern}'.")
    left, right = pair_parts[0].strip(), pair_parts[1].strip()
    if not left or not right:
        raise ValueError(f"Invalid wound pair in '{raw_pattern}'.")
    return _pair_key(left, right), location_id.strip()


def _load_wound_patterns_or_fail(path: Path) -> list[WoundPattern]:
    if not path.exists():
        raise FileNotFoundError(
            "Required wound baseline file missing: "
            f"{path}\nRun `python -m scripts.analyze_wounds --input scripts/output/rashomon_sweep_1_50.json "
            "--output scripts/output/wound_analysis_1_50.json` first."
        )

    raw = json.loads(path.read_text(encoding="utf-8"))
    candidates = list(raw.get("wound_candidates") or [])
    if not candidates:
        raise ValueError(f"Wound baseline file has no wound_candidates: {path}")

    out: list[WoundPattern] = []
    for row in candidates:
        pattern = str(row.get("pattern") or "").strip()
        if not pattern:
            continue
        pair, location = _parse_wound_pattern(pattern)
        out.append(
            WoundPattern(
                pattern=pattern,
                agent_pair=pair,
                location_id=location,
                frequency=float(row.get("frequency", 0.0) or 0.0),
            )
        )
    if not out:
        raise ValueError(f"No parseable wound patterns found in {path}")
    return out


def _detect_wound_presence(rashomon_set: RashomonSet, wounds: list[WoundPattern]) -> dict[str, bool]:
    structural_hits: set[tuple[tuple[str, str], str]] = set()
    for arc in rashomon_set.arcs:
        if not arc.valid:
            continue
        for event, beat in zip(arc.events, arc.beats):
            if beat not in (BeatType.ESCALATION, BeatType.TURNING_POINT):
                continue
            target_head = event.target_agents[0] if event.target_agents else "(none)"
            pair = _pair_key(event.source_agent, target_head)
            structural_hits.add((pair, str(event.location_id)))

    return {
        wound.pattern: (wound.agent_pair, wound.location_id) in structural_hits
        for wound in wounds
    }


def _apply_claim_state_overrides(world, canon: WorldCanon) -> None:
    for claim_id, beliefs_by_agent in sorted(canon.claim_states.items()):
        if claim_id not in world.definition.all_claims:
            continue
        for agent_id, belief_state in sorted((beliefs_by_agent or {}).items()):
            agent = world.agents.get(agent_id)
            if agent is None:
                continue
            try:
                agent.beliefs[claim_id] = BeliefState(str(belief_state))
            except ValueError:
                continue


def _capture_agent_beliefs(world) -> dict[str, dict[str, str]]:
    claim_ids = sorted(world.definition.all_claims)
    beliefs: dict[str, dict[str, str]] = {}
    for agent_id in sorted(world.agents):
        beliefs[agent_id] = {
            claim_id: world.agents[agent_id].beliefs.get(claim_id, BeliefState.UNKNOWN).value
            for claim_id in claim_ids
        }
    return beliefs


def _capture_location_memory(world) -> dict[str, float]:
    out: dict[str, float] = {}
    canon = world.canon
    for location_id in sorted(world.definition.locations):
        memory = canon.location_memory.get(location_id) if canon is not None else None
        out[location_id] = float(memory.tension_residue) if memory is not None else 0.0
    return out


def _sorted_dict_values(mapping: dict[str, Any]) -> list[Any]:
    return [mapping[key] for key in sorted(mapping)]


def _build_payload(world, events: list[Event], snapshots: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    initial_state = snapshots[0] if snapshots else {}
    periodic_snapshots = snapshots[1:] if len(snapshots) > 1 else []
    metadata: dict[str, Any] = {
        "scenario": "dinner_party",
        "deterministic_id": f"dinner_party_seed_{seed}",
        "seed": int(seed),
        "total_ticks": int(world.tick_id),
        "total_sim_time": float(world.sim_time),
        "agent_count": int(len(world.agents)),
        "event_count": int(len(events)),
        "snapshot_interval": int(world.definition.snapshot_interval),
        "truncated": bool(world.truncated),
    }
    return {
        "format_version": "1.0.0",
        "metadata": metadata,
        "initial_state": initial_state,
        "snapshots": periodic_snapshots,
        "events": [event.to_dict() for event in events],
        "secrets": [secret.to_dict() for secret in _sorted_dict_values(world.definition.secrets)],
        "claims": [claim.to_dict() for claim in _sorted_dict_values(world.definition.claims)],
        "locations": [location.to_dict() for location in _sorted_dict_values(world.definition.locations)],
        "world_canon": world.canon.to_dict() if world.canon is not None else WorldCanon().to_dict(),
    }


def _agent_snapshot(agent) -> dict[str, Any]:
    return {
        "goals": {
            "safety": float(agent.goals.safety),
            "status": float(agent.goals.status),
            "secrecy": float(agent.goals.secrecy),
            "truth_seeking": float(agent.goals.truth_seeking),
            "autonomy": float(agent.goals.autonomy),
            "loyalty": float(agent.goals.loyalty),
            "closeness": {k: float(v) for k, v in sorted(agent.goals.closeness.items())},
        },
        "relationships": {
            other: {
                "trust": float(rel.trust),
                "affection": float(rel.affection),
                "obligation": float(rel.obligation),
            }
            for other, rel in sorted(agent.relationships.items())
        },
        "commitments": list(agent.commitments),
    }


def _append_diff_rows(
    out: list[dict[str, Any]],
    old_value: Any,
    new_value: Any,
    *,
    field_prefix: str,
) -> None:
    if isinstance(old_value, dict) and isinstance(new_value, dict):
        keys = sorted(set(old_value) | set(new_value))
        for key in keys:
            prefix = f"{field_prefix}.{key}" if field_prefix else str(key)
            _append_diff_rows(out, old_value.get(key), new_value.get(key), field_prefix=prefix)
        return
    if old_value != new_value:
        out.append(
            {
                "field": field_prefix,
                "old": old_value,
                "new": new_value,
            }
        )


def _diff_agent_snapshot(before: dict[str, Any], after: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    _append_diff_rows(rows, before, after, field_prefix="")
    rows.sort(key=lambda row: str(row["field"]))
    return rows


def _apply_agent_evolutions(world, evolutions: dict[str, AgentEvolution]) -> dict[str, list[dict[str, Any]]]:
    details: dict[str, list[dict[str, Any]]] = {}
    for agent_id, evolution in sorted(evolutions.items()):
        agent = world.agents.get(agent_id)
        if agent is None:
            raise ValueError(f"Unknown agent override id '{agent_id}'.")

        before = _agent_snapshot(agent)

        for attr, value in sorted(evolution.goals_scalar.items()):
            if not hasattr(agent.goals, attr):
                raise ValueError(f"Unknown goal dimension '{attr}' for agent '{agent_id}'.")
            setattr(agent.goals, attr, float(value))

        if evolution.closeness:
            closeness = dict(agent.goals.closeness)
            closeness.pop(agent_id, None)
            for target_id, value in sorted(evolution.closeness.items()):
                if str(target_id) == agent_id:
                    continue
                closeness[str(target_id)] = float(value)
            agent.goals.closeness = closeness

        for target_id, rel_updates in sorted(evolution.relationships.items()):
            rel = agent.relationships.get(target_id)
            if rel is None:
                rel = RelationshipState()
                agent.relationships[target_id] = rel
            for attr, value in sorted(rel_updates.items()):
                if not hasattr(rel, attr):
                    raise ValueError(
                        f"Unknown relationship attribute '{attr}' for agent '{agent_id}' target '{target_id}'."
                    )
                setattr(rel, attr, float(value))

        if evolution.commitments is not None:
            agent.commitments = list(evolution.commitments)

        after = _agent_snapshot(agent)
        details[agent_id] = _diff_agent_snapshot(before, after)
    return details


def _simulate_story(
    *,
    label: str,
    seed: int,
    loaded_canon: WorldCanon | None,
    tick_limit: int,
    event_limit: int,
    evolutions: dict[str, AgentEvolution] | None = None,
) -> tuple[StoryRun, dict[str, list[dict[str, Any]]]]:
    world = create_dinner_party_world()
    mutation_details: dict[str, list[dict[str, Any]]] = {}
    if evolutions:
        mutation_details = _apply_agent_evolutions(world, evolutions)

    canon_copy = WorldCanon.from_dict(loaded_canon.to_dict()) if loaded_canon is not None else None
    world.canon = init_canon_from_world(world.definition, canon_copy)
    if canon_copy is not None:
        _apply_claim_state_overrides(world, world.canon)

    start_beliefs = _capture_agent_beliefs(world)
    start_location_memory = _capture_location_memory(world)

    cfg = SimulationConfig(
        tick_limit=int(tick_limit),
        event_limit=int(event_limit),
        max_sim_time=world.definition.sim_duration_minutes,
        snapshot_interval_events=world.definition.snapshot_interval,
    )
    events, snapshots = run_simulation(world, Random(seed), cfg)

    final_claim_states = {
        claim_id: {agent_id: str(state) for agent_id, state in sorted(states.items())}
        for claim_id, states in sorted((world.canon.claim_states if world.canon is not None else {}).items())
    }
    final_location_memory = _capture_location_memory(world)
    payload = _build_payload(world, events, snapshots, seed)

    return (
        StoryRun(
            label=label,
            seed=seed,
            payload=payload,
            events=events,
            start_beliefs=start_beliefs,
            start_location_memory=start_location_memory,
            final_claim_states=final_claim_states,
            final_location_memory=final_location_memory,
        ),
        mutation_details,
    )


def _analyze_story(payload: dict[str, Any], seed: int, wounds: list[WoundPattern]) -> StoryMetrics:
    parsed = parse_simulation_output(payload)
    metrics_output = run_metrics_pipeline(parsed)
    total_sim_time = float(parsed.metadata.get("total_sim_time") or 0.0) or None
    rashomon = extract_rashomon_set(
        events=metrics_output.events,
        seed=seed,
        agents=list(DINNER_PARTY_AGENTS),
        total_sim_time=total_sim_time,
    )

    valid_scores: list[float] = []
    per_agent_scores: dict[str, float | None] = {agent: None for agent in DINNER_PARTY_AGENTS}
    for arc in rashomon.arcs:
        score = float(arc.arc_score.composite) if arc.arc_score is not None else None
        per_agent_scores[str(arc.protagonist)] = score
        if arc.valid and score is not None:
            valid_scores.append(score)

    mean_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
    overlap = {str(k): float(v) for k, v in rashomon.overlap_matrix.items()}
    wound_presence = _detect_wound_presence(rashomon, wounds)
    wound_present_set = {pattern for pattern, present in wound_presence.items() if present}

    return StoryMetrics(
        mean_score=float(mean_score),
        valid_count=int(rashomon.valid_count),
        total_arcs=int(len(rashomon.arcs)),
        per_agent_scores=per_agent_scores,
        overlap_matrix=overlap,
        wound_presence=wound_presence,
        wound_present_set=wound_present_set,
    )


def _event_signature(event: Event) -> tuple[Any, ...]:
    return (
        int(event.tick_id),
        int(event.order_in_tick),
        event.type.value,
        str(event.source_agent),
        tuple(sorted(str(agent_id) for agent_id in event.target_agents)),
        str(event.location_id),
        json.dumps(event.content_metadata or {}, sort_keys=True),
    )


def _first_divergence(baseline_events: list[Event], candidate_events: list[Event]) -> Divergence | None:
    limit = min(len(baseline_events), len(candidate_events))
    for idx in range(limit):
        if _event_signature(baseline_events[idx]) != _event_signature(candidate_events[idx]):
            return Divergence(
                index=idx,
                tick_id=int(baseline_events[idx].tick_id),
                baseline_event_id=str(baseline_events[idx].id),
                candidate_event_id=str(candidate_events[idx].id),
            )

    if len(baseline_events) != len(candidate_events):
        tick = None
        baseline_id = None
        candidate_id = None
        if limit < len(baseline_events):
            tick = int(baseline_events[limit].tick_id)
            baseline_id = str(baseline_events[limit].id)
        if limit < len(candidate_events):
            tick = int(candidate_events[limit].tick_id) if tick is None else tick
            candidate_id = str(candidate_events[limit].id)
        return Divergence(index=limit, tick_id=tick, baseline_event_id=baseline_id, candidate_event_id=candidate_id)

    return None


def _overlap_top(matrix: dict[str, float], limit: int = 5) -> list[dict[str, Any]]:
    rows = sorted(matrix.items(), key=lambda item: (-float(item[1]), item[0]))[:limit]
    return [{"pair": pair, "jaccard": float(value)} for pair, value in rows]


def _overlap_largest_shifts(baseline: dict[str, float], candidate: dict[str, float], limit: int = 5) -> list[dict[str, Any]]:
    keys = sorted(set(baseline) | set(candidate))
    rows: list[tuple[str, float, float, float]] = []
    for key in keys:
        b = float(baseline.get(key, 0.0))
        c = float(candidate.get(key, 0.0))
        rows.append((key, b, c, c - b))
    rows.sort(key=lambda row: (-abs(row[3]), row[0]))
    return [
        {
            "pair": pair,
            "baseline": b,
            "candidate": c,
            "delta": d,
        }
        for pair, b, c, d in rows[:limit]
    ]


def _present_wounds_with_frequency(wounds: Iterable[WoundPattern], present_set: set[str]) -> list[dict[str, Any]]:
    rows = [
        {
            "pattern": wound.pattern,
            "population_frequency": float(wound.frequency),
        }
        for wound in wounds
        if wound.pattern in present_set
    ]
    rows.sort(key=lambda row: (-row["population_frequency"], row["pattern"]))
    return rows


def _set_delta_rows(target: set[str], baseline: set[str], freq_by_pattern: dict[str, float]) -> list[dict[str, Any]]:
    out = [
        {
            "pattern": pattern,
            "population_frequency": float(freq_by_pattern.get(pattern, 0.0)),
        }
        for pattern in sorted(target - baseline)
    ]
    out.sort(key=lambda row: (-row["population_frequency"], row["pattern"]))
    return out


def _event_type_counts(events: list[Event]) -> dict[str, int]:
    counts = Counter(event.type.value for event in events)
    return {k: int(v) for k, v in sorted(counts.items())}


def _report_event_type_counts(events: list[Event]) -> dict[str, int]:
    counts = Counter(event.type.value for event in events)
    return {event_type: int(counts.get(event_type, 0)) for event_type in REPORT_EVENT_TYPES}


def _fmt_score(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def _fmt_divergence(div: Divergence | dict[str, Any] | None) -> str:
    if div is None:
        return "identical"
    if isinstance(div, dict):
        return f"tick {div.get('tick_id')} (index={div.get('index')})"
    return f"tick {div.tick_id} (index={div.index})"


def _base_world_agent_snapshot() -> dict[str, dict[str, Any]]:
    world = create_dinner_party_world()
    return {agent_id: _agent_snapshot(agent) for agent_id, agent in sorted(world.agents.items())}


def _scalar_goals_from_snapshot(snapshot: dict[str, Any]) -> dict[str, float]:
    goals = snapshot["goals"]
    return {
        "safety": float(goals["safety"]),
        "status": float(goals["status"]),
        "secrecy": float(goals["secrecy"]),
        "truth_seeking": float(goals["truth_seeking"]),
        "autonomy": float(goals["autonomy"]),
        "loyalty": float(goals["loyalty"]),
    }


def _closeness_from_snapshot(snapshot: dict[str, Any]) -> dict[str, float]:
    return {k: float(v) for k, v in sorted((snapshot["goals"]["closeness"] or {}).items())}


def _merge_relationships(
    source: dict[str, dict[str, float]],
    updates: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    merged = {k: dict(v) for k, v in sorted(source.items())}
    for target, delta in sorted(updates.items()):
        row = merged.get(target, {"trust": 0.0, "affection": 0.0, "obligation": 0.0})
        for attr, value in sorted(delta.items()):
            row[attr] = float(value)
        merged[target] = row
    return merged


def _evolution_profiles() -> dict[str, dict[str, AgentEvolution]]:
    base = _base_world_agent_snapshot()

    thorne_target_scalar = _scalar_goals_from_snapshot(base["victor"])
    thorne_target_scalar.update(
        {
            "safety": 0.50,
            "status": 0.72,
            "secrecy": 0.20,
            "truth_seeking": 0.95,
            "autonomy": 0.86,
            "loyalty": 0.55,
        }
    )
    thorne_target_closeness = _closeness_from_snapshot(base["victor"])
    thorne_target_closeness.update(
        {
            "elena": -0.35,
            "marcus": -0.55,
            "lydia": 0.60,
            "diana": 0.15,
            "victor": 0.65,
        }
    )
    thorne_target_relationships = _merge_relationships(
        copy.deepcopy(base["thorne"]["relationships"]),
        {
            "elena": {"trust": -0.25, "affection": -0.10, "obligation": 0.05},
            "marcus": {"trust": -0.35, "affection": -0.15, "obligation": 0.05},
            "lydia": {"trust": 0.55},
            "victor": {"trust": 0.70},
        },
    )
    thorne_evolution = AgentEvolution(
        goals_scalar=thorne_target_scalar,
        closeness=thorne_target_closeness,
        relationships=thorne_target_relationships,
        commitments=None,
    )

    elena_target_scalar = _scalar_goals_from_snapshot(base["diana"])
    elena_target_scalar.update(
        {
            "safety": 0.85,
            "status": 0.30,
            "secrecy": 0.45,
            "truth_seeking": 0.30,
            "autonomy": 0.90,
            "loyalty": 0.55,
        }
    )
    elena_target_closeness = _closeness_from_snapshot(base["elena"])
    elena_target_closeness.update(
        {
            "thorne": -0.25,
            "marcus": 0.20,
            "lydia": 0.55,
            "diana": 0.80,
            "victor": 0.15,
        }
    )
    elena_target_relationships = _merge_relationships(
        copy.deepcopy(base["elena"]["relationships"]),
        {
            "thorne": {"trust": 0.15, "affection": 0.0},
            "marcus": {"trust": 0.25, "affection": 0.25},
            "lydia": {"trust": 0.45, "affection": 0.25},
            "diana": {"trust": 0.80, "affection": 0.60},
        },
    )
    elena_evolution = AgentEvolution(
        goals_scalar=elena_target_scalar,
        closeness=elena_target_closeness,
        relationships=elena_target_relationships,
        commitments=(),
    )

    marcus_scalar = _scalar_goals_from_snapshot(base["marcus"])
    marcus_scalar.update(
        {
            "safety": 0.92,
            "status": 0.62,
            "secrecy": 1.00,
            "truth_seeking": 0.05,
            "autonomy": 0.78,
            "loyalty": 0.10,
        }
    )
    marcus_closeness = _closeness_from_snapshot(base["marcus"])
    marcus_closeness.update(
        {
            "thorne": -0.55,
            "elena": 0.15,
            "lydia": -0.45,
            "diana": 0.25,
            "victor": -0.60,
        }
    )
    marcus_relationships = _merge_relationships(
        copy.deepcopy(base["marcus"]["relationships"]),
        {
            "thorne": {"trust": -0.30},
            "elena": {"trust": 0.20, "affection": 0.10},
            "lydia": {"trust": -0.35},
            "diana": {"trust": 0.35},
            "victor": {"trust": -0.50, "affection": -0.35},
        },
    )
    marcus_evolution = AgentEvolution(
        goals_scalar=marcus_scalar,
        closeness=marcus_closeness,
        relationships=marcus_relationships,
        commitments=("cover_embezzlement",),
    )

    victor_scalar = _scalar_goals_from_snapshot(base["victor"])
    victor_scalar.update(
        {
            "safety": 0.35,
            "status": 0.65,
            "secrecy": 0.40,
            "truth_seeking": 0.95,
            "autonomy": 0.88,
            "loyalty": 0.35,
        }
    )
    victor_closeness = _closeness_from_snapshot(base["victor"])
    victor_closeness.update(
        {
            "thorne": 0.35,
            "elena": 0.00,
            "marcus": -0.60,
            "lydia": 0.55,
            "diana": 0.15,
        }
    )
    victor_relationships = _merge_relationships(
        copy.deepcopy(base["victor"]["relationships"]),
        {
            "marcus": {"trust": -0.35},
            "lydia": {"trust": 0.55, "affection": 0.20},
            "thorne": {"trust": 0.45},
        },
    )
    victor_evolution = AgentEvolution(
        goals_scalar=victor_scalar,
        closeness=victor_closeness,
        relationships=victor_relationships,
        commitments=("write_expose_on_marcus",),
    )

    lydia_scalar = _scalar_goals_from_snapshot(base["lydia"])
    lydia_scalar.update(
        {
            "safety": 0.85,
            "status": 0.25,
            "secrecy": 0.30,
            "truth_seeking": 0.72,
            "autonomy": 0.55,
            "loyalty": 0.95,
        }
    )
    lydia_closeness = _closeness_from_snapshot(base["lydia"])
    lydia_closeness.update(
        {
            "thorne": 0.65,
            "elena": 0.35,
            "marcus": -0.45,
            "diana": 0.40,
            "victor": 0.35,
        }
    )
    lydia_relationships = _merge_relationships(
        copy.deepcopy(base["lydia"]["relationships"]),
        {
            "thorne": {"trust": 0.75},
            "elena": {"trust": 0.35},
            "marcus": {"trust": -0.35},
            "diana": {"trust": 0.45},
        },
    )
    lydia_evolution = AgentEvolution(
        goals_scalar=lydia_scalar,
        closeness=lydia_closeness,
        relationships=lydia_relationships,
        commitments=None,
    )

    diana_scalar = _scalar_goals_from_snapshot(base["diana"])
    diana_scalar.update(
        {
            "safety": 0.75,
            "status": 0.45,
            "secrecy": 0.82,
            "truth_seeking": 0.38,
            "autonomy": 0.68,
            "loyalty": 0.60,
        }
    )
    diana_closeness = _closeness_from_snapshot(base["diana"])
    diana_closeness.update(
        {
            "thorne": 0.15,
            "elena": 0.65,
            "marcus": -0.35,
            "lydia": 0.45,
            "victor": 0.25,
        }
    )
    diana_relationships = _merge_relationships(
        copy.deepcopy(base["diana"]["relationships"]),
        {
            "marcus": {"trust": -0.20, "obligation": 0.25},
            "lydia": {"trust": 0.45},
            "victor": {"trust": 0.40},
            "elena": {"trust": 0.80},
        },
    )
    diana_evolution = AgentEvolution(
        goals_scalar=diana_scalar,
        closeness=diana_closeness,
        relationships=diana_relationships,
        commitments=(),
    )

    targeted = {
        "thorne": thorne_evolution,
        "elena": elena_evolution,
    }
    full = {
        "thorne": thorne_evolution,
        "elena": elena_evolution,
        "marcus": marcus_evolution,
        "victor": victor_evolution,
        "lydia": lydia_evolution,
        "diana": diana_evolution,
    }
    thorne_only = {
        "thorne": thorne_evolution,
    }
    return {
        "targeted": targeted,
        "full": full,
        "thorne_only": thorne_only,
    }


def _load_world_canon_from_file(path: Path) -> WorldCanon:
    if not path.exists():
        raise FileNotFoundError(f"Input canon file not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "world_canon" in raw:
        payload = raw.get("world_canon") or {}
    else:
        payload = raw
    return WorldCanon.from_dict(payload)


def _score_delta_rows(
    left_scores: dict[str, float | None],
    right_scores: dict[str, float | None],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for agent in DINNER_PARTY_AGENTS:
        left = left_scores.get(agent)
        right = right_scores.get(agent)
        delta = None
        if left is not None and right is not None:
            delta = float(right - left)
        rows.append(
            {
                "agent": agent,
                "left": left,
                "right": right,
                "delta": delta,
            }
        )
    return rows


def _max_delta_agent(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid = [row for row in rows if row.get("delta") is not None]
    if not valid:
        return None
    return max(valid, key=lambda row: (float(row["delta"]), row["agent"]))


def _contains_any_shift(rows: list[dict[str, Any]]) -> bool:
    return any(abs(float(row["delta"])) > EPSILON for row in rows if row.get("delta") is not None)


def _agent_definition_discovery_summary() -> dict[str, Any]:
    return {
        "agent_structure_class": "narrativefield.schema.agents.AgentState",
        "agent_source_path": "src/engine/narrativefield/simulation/scenarios/dinner_party.py",
        "agent_fields_influencing_behavior": [
            "goals.safety",
            "goals.status",
            "goals.closeness[target_agent_id]",
            "goals.secrecy",
            "goals.truth_seeking",
            "goals.autonomy",
            "goals.loyalty",
            "relationships[target_agent_id].trust",
            "relationships[target_agent_id].affection",
            "relationships[target_agent_id].obligation",
            "beliefs[claim_id]",
            "flaws[*].{flaw_type,strength,trigger,effect}",
            "pacing.{dramatic_budget,stress,composure,commitment,recovery_timer,suppression_count}",
            "emotional_state[*]",
            "commitments[*]",
            "location",
            "alcohol_level",
        ],
        "goal_representation": (
            "Goals are numeric utilities in GoalVector; the decision engine scores candidate actions "
            "with weighted action effects (base_utility in decision_engine.py)."
        ),
        "action_preferences_representation": (
            "There are no direct per-agent action-type weight fields. Preferences emerge from goal "
            "weights, relationship modifiers, flaw biases, and pacing modifiers."
        ),
        "target_preferences_representation": (
            "Target preference is encoded primarily in goals.closeness per target and relationship state."
        ),
        "instantiation_mechanism": (
            "Agents are hardcoded in AGENT_DICTS and instantiated by create_dinner_party_world(); "
            "not passed as external parameters."
        ),
        "runtime_override_capability": (
            "Agent fields are mutable at runtime on WorldState.agents entries, enabling script-local "
            "goal/relationship/commitment overrides without source changes."
        ),
    }


def _build_conditions(
    *,
    canon_after_b: WorldCanon,
    include_thorne_only: bool,
) -> list[ConditionSpec]:
    profiles = _evolution_profiles()
    conditions = [
        ConditionSpec(
            key="depth0_fresh",
            label="Depth 0 (fresh)",
            loaded_canon=None,
            evolutions={},
        ),
        ConditionSpec(
            key="baseline_depth2",
            label="Baseline (depth 2)",
            loaded_canon=canon_after_b,
            evolutions={},
        ),
        ConditionSpec(
            key="targeted_evolution",
            label="Targeted Evolution",
            loaded_canon=canon_after_b,
            evolutions=profiles["targeted"],
        ),
        ConditionSpec(
            key="full_evolution",
            label="Full Evolution",
            loaded_canon=canon_after_b,
            evolutions=profiles["full"],
        ),
    ]
    if include_thorne_only:
        conditions.append(
            ConditionSpec(
                key="thorne_only_evolution",
                label="Thorne-Only Evolution",
                loaded_canon=canon_after_b,
                evolutions=profiles["thorne_only"],
            )
        )
    return conditions


def _condition_result_payload(condition: ConditionResult, wounds: list[WoundPattern]) -> dict[str, Any]:
    return {
        "key": condition.key,
        "label": condition.label,
        "mean_valid_arc_score": float(condition.metrics.mean_score),
        "valid_arcs": int(condition.metrics.valid_count),
        "total_arcs": int(condition.metrics.total_arcs),
        "per_agent_arc_scores": {
            agent: condition.metrics.per_agent_scores.get(agent)
            for agent in DINNER_PARTY_AGENTS
        },
        "wounds_present": _present_wounds_with_frequency(wounds, condition.metrics.wound_present_set),
        "overlap_top5": _overlap_top(condition.metrics.overlap_matrix),
        "event_type_counts_reported": _report_event_type_counts(condition.story.events),
        "event_type_counts_all": dict(condition.event_type_counts),
        "mutation_details": condition.mutation_details,
    }


def _print_report(report: dict[str, Any]) -> None:
    discovery = report["agent_definition_discovery"]
    conditions = report["conditions"]
    comparisons = report["comparisons"]
    conclusions = report["conclusions"]

    print("GOAL EVOLUTION EXPERIMENT RESULTS")
    print("==================================")
    print()
    print("AGENT DEFINITION DISCOVERY:")
    print(f"- Agent class/structure: {discovery['agent_structure_class']}")
    print(f"- Agent source: {discovery['agent_source_path']}")
    print("- Fields that influence behavior:")
    for field in discovery["agent_fields_influencing_behavior"]:
        print(f"  - {field}")
    print(f"- How goals are represented: {discovery['goal_representation']}")
    print(f"- How action preferences work: {discovery['action_preferences_representation']}")
    print(f"- How target preferences work: {discovery['target_preferences_representation']}")
    print(
        "- How agents are instantiated: "
        f"{discovery['instantiation_mechanism']}"
    )
    print(
        "- Runtime overrides possible: "
        f"{discovery['runtime_override_capability']}"
    )
    print()

    print("MUTATIONS APPLIED:")
    for key in ["targeted_evolution", "full_evolution", "thorne_only_evolution"]:
        row = conditions.get(key)
        if row is None:
            continue
        print()
        print(f"{row['label']}:")
        details = row.get("mutation_details") or {}
        if not details:
            print("  (no agent mutations)")
            continue
        for agent_id in sorted(details):
            print(f"  {agent_id}:")
            for change in details[agent_id]:
                print(f"    - {change['field']}: {change['old']} -> {change['new']}")
    print()

    score_table = comparisons["score_table"]
    print("SCORE COMPARISON TABLE:")
    print()
    print("| Condition | Mean Score | vs Baseline | vs Depth-0 |")
    print("|---|---:|---:|---:|")
    for row in score_table:
        print(
            f"| {row['condition']} | {row['mean_score']:.3f} | "
            f"{row['vs_baseline'] if row['vs_baseline'] is not None else '—'} | "
            f"{row['vs_depth0'] if row['vs_depth0'] is not None else '—'} |"
        )
    print()

    print("PER-AGENT SCORE TABLE:")
    print()
    print("| Agent | Depth 0 | Baseline | Targeted | Full Evo | Thorne-Only |")
    print("|---|---:|---:|---:|---:|---:|")
    for row in comparisons["per_agent_table"]:
        print(
            f"| {row['agent']} | {_fmt_score(row.get('depth0'))} | {_fmt_score(row.get('baseline'))} | "
            f"{_fmt_score(row.get('targeted'))} | {_fmt_score(row.get('full'))} | "
            f"{_fmt_score(row.get('thorne_only'))} |"
        )
    print()

    print("ARC VALIDITY:")
    print()
    print("| Condition | Valid Arcs |")
    print("|---|---:|")
    for row in comparisons["arc_validity_rows"]:
        print(f"| {row['condition']} | {row['valid_arcs']}/6 |")
    print()

    print("WOUND TOPOLOGY:")
    print()
    print(
        "Baseline wounds:",
        [row["pattern"] for row in comparisons["wound_topology"]["baseline_present"]],
    )
    print(
        "Targeted wounds:",
        [row["pattern"] for row in comparisons["wound_topology"]["targeted_present"]],
        f"— appeared: {len(comparisons['wound_topology']['targeted_appeared'])},",
        f"disappeared: {len(comparisons['wound_topology']['targeted_disappeared'])}",
    )
    print(
        "Full Evo wounds:",
        [row["pattern"] for row in comparisons["wound_topology"]["full_present"]],
        f"— appeared: {len(comparisons['wound_topology']['full_appeared'])},",
        f"disappeared: {len(comparisons['wound_topology']['full_disappeared'])}",
    )
    if comparisons["wound_topology"].get("thorne_only_present") is not None:
        print(
            "Thorne-Only wounds:",
            [row["pattern"] for row in comparisons["wound_topology"]["thorne_only_present"]],
            f"— appeared: {len(comparisons['wound_topology']['thorne_only_appeared'])},",
            f"disappeared: {len(comparisons['wound_topology']['thorne_only_disappeared'])}",
        )
    print()

    print("OVERLAP MATRIX SHIFTS:")
    print()
    for label, rows in comparisons["overlap_top5"].items():
        print(f"- {label} top 5: {rows}")
    for label, rows in comparisons["largest_overlap_shifts_vs_baseline"].items():
        print(f"- Largest Jaccard deltas (baseline -> {label}): {rows}")
    print()

    print("EVENT DIVERGENCE:")
    div = comparisons["event_divergence"]
    print(
        f"- Targeted vs Baseline: first divergence at {_fmt_divergence(div['targeted_vs_baseline'])}"
    )
    print(f"- Full Evo vs Baseline: first divergence at {_fmt_divergence(div['full_vs_baseline'])}")
    if div.get("thorne_only_vs_baseline") is not None:
        print(
            f"- Thorne-Only vs Baseline: first divergence at {_fmt_divergence(div['thorne_only_vs_baseline'])}"
        )
    print()

    print("EVENT TYPE DISTRIBUTION:")
    print()
    type_headers = ["Condition", *[name.upper() for name in REPORT_EVENT_TYPES]]
    print("| " + " | ".join(type_headers) + " |")
    print("|" + "|".join(["---"] + ["---:"] * len(REPORT_EVENT_TYPES)) + "|")
    for row in comparisons["event_type_distribution_rows"]:
        cols = [row["condition"], *[str(row["counts"][event_type]) for event_type in REPORT_EVENT_TYPES]]
        print("| " + " | ".join(cols) + " |")
    print()

    print("CONCLUSIONS:")
    print(
        "- Does targeted goal evolution restore arc scores?",
        f"{'YES' if conclusions['targeted_restores'] else 'NO'}",
        f"(delta: {conclusions['targeted_delta_vs_baseline']:+.3f})",
    )
    print(
        "- Does full evolution restore more than targeted?",
        f"{'YES' if conclusions['full_better_than_targeted'] else 'NO'}",
        f"(delta: {conclusions['full_minus_targeted']:+.3f})",
    )
    print(
        "- Do evolved scores approach depth-0 quality?",
        f"{'YES' if conclusions['targeted_approaches_depth0'] or conclusions['full_approaches_depth0'] else 'NO'}",
        f"(targeted gap: {conclusions['targeted_gap_to_depth0']:+.3f}, full gap: {conclusions['full_gap_to_depth0']:+.3f})",
    )
    print(
        "- Do new wound patterns emerge?",
        f"{'YES' if conclusions['new_wounds_emerge'] else 'NO'}",
        conclusions["new_wounds_list"],
    )
    print(
        "- Does the event type distribution shift?",
        f"{'YES' if conclusions['event_type_distribution_shift'] else 'NO'}",
        conclusions["event_type_shift_summary"],
    )
    print(
        "- Which agent benefits most from evolution?",
        conclusions["best_agent_overall"],
    )
    print(
        "- Which agent benefits most from OTHER agents evolving?",
        conclusions["best_agent_from_others"],
    )
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the goal evolution experiment on an exhausted dinner-party canon.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--event-limit", type=int, default=200)
    parser.add_argument("--tick-limit", type=int, default=300)
    parser.add_argument(
        "--canon-after-b",
        type=str,
        default=str(CANON_AFTER_B_PATH),
        help="Path to canon_after_B JSON (wrapper with world_canon or raw WorldCanon dict).",
    )
    parser.add_argument(
        "--wound-analysis",
        type=str,
        default=str(WOUND_ANALYSIS_PATH),
        help="Path to wound analysis JSON.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="scripts/output/goal_evolution_experiment.json",
        help="Path to JSON report output.",
    )
    parser.add_argument(
        "--include-thorne-only",
        action="store_true",
        default=True,
        help="Include sensitivity condition evolving only Thorne (default: on).",
    )
    parser.add_argument(
        "--no-thorne-only",
        dest="include_thorne_only",
        action="store_false",
        help="Disable Thorne-only sensitivity condition.",
    )
    args = parser.parse_args()

    seed = int(args.seed)
    output_path = _resolve_output_path(args.output)
    canon_path = _resolve_output_path(args.canon_after_b)
    wound_path = _resolve_output_path(args.wound_analysis)

    wounds = _load_wound_patterns_or_fail(wound_path)
    wound_freq = {w.pattern: float(w.frequency) for w in wounds}
    canon_after_b = _load_world_canon_from_file(canon_path)

    discovery_summary = _agent_definition_discovery_summary()
    conditions = _build_conditions(canon_after_b=canon_after_b, include_thorne_only=bool(args.include_thorne_only))

    results: dict[str, ConditionResult] = {}
    for condition in conditions:
        story, mutation_details = _simulate_story(
            label=condition.label,
            seed=seed,
            loaded_canon=condition.loaded_canon,
            tick_limit=int(args.tick_limit),
            event_limit=int(args.event_limit),
            evolutions=condition.evolutions,
        )
        metrics = _analyze_story(story.payload, seed, wounds)
        results[condition.key] = ConditionResult(
            key=condition.key,
            label=condition.label,
            story=story,
            metrics=metrics,
            event_type_counts=_event_type_counts(story.events),
            mutation_details=mutation_details,
        )

    baseline = results["baseline_depth2"]
    baseline_repeat_story, _baseline_repeat_mut = _simulate_story(
        label="Baseline (depth 2 repeat)",
        seed=seed,
        loaded_canon=canon_after_b,
        tick_limit=int(args.tick_limit),
        event_limit=int(args.event_limit),
        evolutions={},
    )
    baseline_repeat_metrics = _analyze_story(baseline_repeat_story.payload, seed, wounds)
    determinism_divergence = _first_divergence(baseline.story.events, baseline_repeat_story.events)
    determinism_ok = (
        determinism_divergence is None
        and abs(baseline.metrics.mean_score - baseline_repeat_metrics.mean_score) <= EPSILON
        and baseline.metrics.valid_count == baseline_repeat_metrics.valid_count
    )

    depth0 = results["depth0_fresh"]
    targeted = results["targeted_evolution"]
    full = results["full_evolution"]
    thorne_only = results.get("thorne_only_evolution")

    targeted_div = _first_divergence(baseline.story.events, targeted.story.events)
    full_div = _first_divergence(baseline.story.events, full.story.events)
    thorne_only_div = (
        _first_divergence(baseline.story.events, thorne_only.story.events)
        if thorne_only is not None
        else None
    )

    score_table_rows: list[dict[str, Any]] = []
    for key in ["depth0_fresh", "baseline_depth2", "targeted_evolution", "full_evolution", "thorne_only_evolution"]:
        condition = results.get(key)
        if condition is None:
            continue
        mean = float(condition.metrics.mean_score)
        vs_baseline = None if key == "baseline_depth2" else f"{mean - baseline.metrics.mean_score:+.3f}"
        vs_depth0 = None if key == "depth0_fresh" else f"{mean - depth0.metrics.mean_score:+.3f}"
        score_table_rows.append(
            {
                "condition": condition.label,
                "mean_score": mean,
                "vs_baseline": vs_baseline,
                "vs_depth0": vs_depth0,
            }
        )

    per_agent_rows: list[dict[str, Any]] = []
    for agent in DINNER_PARTY_AGENTS:
        per_agent_rows.append(
            {
                "agent": agent,
                "depth0": depth0.metrics.per_agent_scores.get(agent),
                "baseline": baseline.metrics.per_agent_scores.get(agent),
                "targeted": targeted.metrics.per_agent_scores.get(agent),
                "full": full.metrics.per_agent_scores.get(agent),
                "thorne_only": None if thorne_only is None else thorne_only.metrics.per_agent_scores.get(agent),
            }
        )

    arc_validity_rows: list[dict[str, Any]] = []
    for key in ["depth0_fresh", "baseline_depth2", "targeted_evolution", "full_evolution", "thorne_only_evolution"]:
        condition = results.get(key)
        if condition is None:
            continue
        arc_validity_rows.append({"condition": condition.label, "valid_arcs": int(condition.metrics.valid_count)})

    baseline_present = _present_wounds_with_frequency(wounds, baseline.metrics.wound_present_set)
    targeted_present = _present_wounds_with_frequency(wounds, targeted.metrics.wound_present_set)
    full_present = _present_wounds_with_frequency(wounds, full.metrics.wound_present_set)
    thorne_present = (
        _present_wounds_with_frequency(wounds, thorne_only.metrics.wound_present_set)
        if thorne_only is not None
        else None
    )

    targeted_appeared = _set_delta_rows(targeted.metrics.wound_present_set, baseline.metrics.wound_present_set, wound_freq)
    targeted_disappeared = _set_delta_rows(baseline.metrics.wound_present_set, targeted.metrics.wound_present_set, wound_freq)
    full_appeared = _set_delta_rows(full.metrics.wound_present_set, baseline.metrics.wound_present_set, wound_freq)
    full_disappeared = _set_delta_rows(baseline.metrics.wound_present_set, full.metrics.wound_present_set, wound_freq)
    thorne_appeared = (
        _set_delta_rows(thorne_only.metrics.wound_present_set, baseline.metrics.wound_present_set, wound_freq)
        if thorne_only is not None
        else None
    )
    thorne_disappeared = (
        _set_delta_rows(baseline.metrics.wound_present_set, thorne_only.metrics.wound_present_set, wound_freq)
        if thorne_only is not None
        else None
    )

    overlap_top5 = {
        results[key].label: _overlap_top(results[key].metrics.overlap_matrix)
        for key in ["depth0_fresh", "baseline_depth2", "targeted_evolution", "full_evolution", "thorne_only_evolution"]
        if key in results
    }
    largest_overlap_shifts = {
        "Targeted Evolution": _overlap_largest_shifts(baseline.metrics.overlap_matrix, targeted.metrics.overlap_matrix),
        "Full Evolution": _overlap_largest_shifts(baseline.metrics.overlap_matrix, full.metrics.overlap_matrix),
    }
    if thorne_only is not None:
        largest_overlap_shifts["Thorne-Only Evolution"] = _overlap_largest_shifts(
            baseline.metrics.overlap_matrix, thorne_only.metrics.overlap_matrix
        )

    event_type_distribution_rows: list[dict[str, Any]] = []
    for key in ["depth0_fresh", "baseline_depth2", "targeted_evolution", "full_evolution", "thorne_only_evolution"]:
        condition = results.get(key)
        if condition is None:
            continue
        event_type_distribution_rows.append(
            {
                "condition": condition.label,
                "counts": _report_event_type_counts(condition.story.events),
            }
        )

    targeted_delta = float(targeted.metrics.mean_score - baseline.metrics.mean_score)
    full_minus_targeted = float(full.metrics.mean_score - targeted.metrics.mean_score)
    targeted_gap_to_depth0 = float(targeted.metrics.mean_score - depth0.metrics.mean_score)
    full_gap_to_depth0 = float(full.metrics.mean_score - depth0.metrics.mean_score)

    baseline_to_full_deltas = _score_delta_rows(baseline.metrics.per_agent_scores, full.metrics.per_agent_scores)
    targeted_to_full_deltas = _score_delta_rows(targeted.metrics.per_agent_scores, full.metrics.per_agent_scores)
    best_overall = _max_delta_agent(baseline_to_full_deltas)
    best_from_others = _max_delta_agent(targeted_to_full_deltas)

    baseline_types = _report_event_type_counts(baseline.story.events)
    targeted_types = _report_event_type_counts(targeted.story.events)
    full_types = _report_event_type_counts(full.story.events)
    type_shift_rows: list[str] = []
    for event_type in REPORT_EVENT_TYPES:
        delta_targeted = targeted_types[event_type] - baseline_types[event_type]
        delta_full = full_types[event_type] - baseline_types[event_type]
        if delta_targeted != 0 or delta_full != 0:
            type_shift_rows.append(
                f"{event_type.upper()}: targeted {delta_targeted:+d}, full {delta_full:+d}"
            )
    event_type_shift_summary = "; ".join(type_shift_rows) if type_shift_rows else "no count changes in reported event types"

    conclusions = {
        "targeted_restores": bool(targeted_delta > EPSILON),
        "targeted_delta_vs_baseline": float(targeted_delta),
        "full_better_than_targeted": bool(full_minus_targeted > EPSILON),
        "full_minus_targeted": float(full_minus_targeted),
        "targeted_approaches_depth0": bool(abs(targeted_gap_to_depth0) < abs(baseline.metrics.mean_score - depth0.metrics.mean_score)),
        "full_approaches_depth0": bool(abs(full_gap_to_depth0) < abs(baseline.metrics.mean_score - depth0.metrics.mean_score)),
        "targeted_gap_to_depth0": float(targeted_gap_to_depth0),
        "full_gap_to_depth0": float(full_gap_to_depth0),
        "new_wounds_emerge": bool(targeted_appeared or full_appeared),
        "new_wounds_list": sorted(
            {
                row["pattern"] for row in targeted_appeared
            }
            | {
                row["pattern"] for row in full_appeared
            }
        ),
        "event_type_distribution_shift": bool(_contains_any_shift(
            [
                {"delta": float(targeted_types[event_type] - baseline_types[event_type])}
                for event_type in REPORT_EVENT_TYPES
            ]
            + [
                {"delta": float(full_types[event_type] - baseline_types[event_type])}
                for event_type in REPORT_EVENT_TYPES
            ]
        )),
        "event_type_shift_summary": event_type_shift_summary,
        "best_agent_overall": None
        if best_overall is None
        else {
            "agent": best_overall["agent"],
            "delta_full_vs_baseline": float(best_overall["delta"]),
        },
        "best_agent_from_others": None
        if best_from_others is None
        else {
            "agent": best_from_others["agent"],
            "delta_full_vs_targeted": float(best_from_others["delta"]),
        },
    }

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "seed": seed,
            "event_limit": int(args.event_limit),
            "tick_limit": int(args.tick_limit),
            "canon_after_b_path": str(canon_path),
            "wound_analysis_path": str(wound_path),
            "llm_calls": False,
            "include_thorne_only": bool(args.include_thorne_only),
        },
        "agent_definition_discovery": discovery_summary,
        "conditions": {
            key: _condition_result_payload(result, wounds)
            for key, result in sorted(results.items())
        },
        "comparisons": {
            "score_table": score_table_rows,
            "per_agent_table": per_agent_rows,
            "arc_validity_rows": arc_validity_rows,
            "wound_topology": {
                "baseline_present": baseline_present,
                "targeted_present": targeted_present,
                "targeted_appeared": targeted_appeared,
                "targeted_disappeared": targeted_disappeared,
                "full_present": full_present,
                "full_appeared": full_appeared,
                "full_disappeared": full_disappeared,
                "thorne_only_present": thorne_present,
                "thorne_only_appeared": thorne_appeared,
                "thorne_only_disappeared": thorne_disappeared,
            },
            "overlap_top5": overlap_top5,
            "largest_overlap_shifts_vs_baseline": largest_overlap_shifts,
            "event_divergence": {
                "targeted_vs_baseline": None
                if targeted_div is None
                else {
                    "index": int(targeted_div.index),
                    "tick_id": targeted_div.tick_id,
                    "baseline_event_id": targeted_div.baseline_event_id,
                    "candidate_event_id": targeted_div.candidate_event_id,
                },
                "full_vs_baseline": None
                if full_div is None
                else {
                    "index": int(full_div.index),
                    "tick_id": full_div.tick_id,
                    "baseline_event_id": full_div.baseline_event_id,
                    "candidate_event_id": full_div.candidate_event_id,
                },
                "thorne_only_vs_baseline": None
                if thorne_only_div is None
                else {
                    "index": int(thorne_only_div.index),
                    "tick_id": thorne_only_div.tick_id,
                    "baseline_event_id": thorne_only_div.baseline_event_id,
                    "candidate_event_id": thorne_only_div.candidate_event_id,
                },
            },
            "event_type_distribution_rows": event_type_distribution_rows,
            "per_agent_delta_full_vs_baseline": baseline_to_full_deltas,
            "per_agent_delta_full_vs_targeted": targeted_to_full_deltas,
        },
        "determinism_check": {
            "baseline_repeat_identical": bool(determinism_ok),
            "first_divergence_index": None if determinism_divergence is None else int(determinism_divergence.index),
            "first_divergence_tick": None if determinism_divergence is None else determinism_divergence.tick_id,
        },
        "conclusions": conclusions,
        "artifacts": {
            "report_path": str(output_path),
        },
    }

    _save_json(output_path, report)
    _print_report(report)
    print(f"JSON report written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
