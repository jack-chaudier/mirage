"""Full k-sweep experiment with exhaustive subset enumeration and multi-seed analysis.

Run:
    cd src/engine && ./.venv/bin/python -m scripts.k_sweep_experiment

Notes:
- Uses skip-narration simulation path only (no LLM calls).
- Supports checkpoint+resume for long-running sweeps.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from random import Random
from typing import Any

import numpy as np

from narrativefield.extraction.rashomon import RashomonSet, extract_rashomon_set
from narrativefield.metrics.pipeline import parse_simulation_output, run_metrics_pipeline
from narrativefield.schema.agents import BeliefState
from narrativefield.schema.canon import WorldCanon
from narrativefield.schema.events import BeatType, Event
from narrativefield.simulation.scenarios import create_dinner_party_world
from narrativefield.simulation.tick_loop import SimulationConfig, init_canon_from_world, run_simulation
from scripts.test_goal_evolution import (  # Reuse exact evolution definitions and mutation logic.
    AgentEvolution,
    _apply_agent_evolutions,
    _evolution_profiles,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DEFAULT_OUTPUT_PATH = OUTPUT_DIR / "k_sweep_experiment.json"
DEFAULT_WOUND_PATH = OUTPUT_DIR / "wound_analysis_1_50.json"
DEFAULT_SHARED_CANON_PATH = OUTPUT_DIR / "canon_after_B.json"
DEFAULT_SWEEP_REFERENCE_PATH = OUTPUT_DIR / "rashomon_sweep_1_50.json"

DINNER_PARTY_AGENTS = ["thorne", "elena", "marcus", "lydia", "diana", "victor"]
EVENT_TYPES = [
    "chat",
    "conflict",
    "reveal",
    "lie",
    "confide",
    "catastrophe",
    "physical",
    "observe",
    "social_move",
    "internal",
]
COMPONENT_KEYS = ["M_var", "M_peak", "M_shape", "M_sig", "M_theme", "M_irony", "M_prot"]
CORE_COMPONENT_CONDITIONS = {"fresh", "baseline", "full_evolution"}

EPSILON = 1e-12
DETERMINISM_EPSILON = 1e-10
WARN_TOLERANCE = 0.001
FAIL_TOLERANCE = 0.01

EXPECTED_SEED7 = {
    "baseline": 0.6683585600044828,
    "full_evolution": 0.7153000979723764,
    "thorne_only": 0.6413820420428566,
    "targeted": 0.6829299068601778,
    "fresh": 0.7212762843119922,
}


@dataclass(frozen=True)
class WoundPattern:
    pattern: str
    agent_pair: tuple[str, str]
    location_id: str
    frequency: float


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
    final_relationships: dict[str, dict[str, dict[str, float]]]


@dataclass(frozen=True)
class ConditionSpec:
    condition_type: str
    k: int
    evolved_agents: tuple[str, ...]
    use_canon: bool


@dataclass(frozen=True)
class Divergence:
    index: int
    tick_id: int | None
    baseline_event_id: str | None
    candidate_event_id: str | None


def _resolve_path(path_arg: str) -> Path:
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


def _capture_relationships(world) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    for agent_id, agent in sorted(world.agents.items()):
        out[agent_id] = {
            other: {
                "trust": float(rel.trust),
                "affection": float(rel.affection),
                "obligation": float(rel.obligation),
            }
            for other, rel in sorted(agent.relationships.items())
        }
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


def _simulate_story(
    *,
    label: str,
    seed: int,
    loaded_canon: WorldCanon | None,
    tick_limit: int,
    event_limit: int,
    evolutions: dict[str, AgentEvolution] | None = None,
) -> StoryRun:
    world = create_dinner_party_world()
    if evolutions:
        _apply_agent_evolutions(world, evolutions)

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
    final_relationships = _capture_relationships(world)
    payload = _build_payload(world, events, snapshots, seed)

    return StoryRun(
        label=label,
        seed=seed,
        payload=payload,
        events=events,
        start_beliefs=start_beliefs,
        start_location_memory=start_location_memory,
        final_claim_states=final_claim_states,
        final_location_memory=final_location_memory,
        final_relationships=final_relationships,
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


def _event_type_counts(events: list[Event]) -> dict[str, int]:
    counts = Counter(event.type.value for event in events)
    return {event_type: int(counts.get(event_type, 0)) for event_type in EVENT_TYPES}


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _std(values: list[float]) -> float:
    return float(np.std(np.asarray(values, dtype=float), ddof=0)) if values else 0.0


def _variance(values: list[float]) -> float:
    return float(np.var(np.asarray(values, dtype=float), ddof=0)) if values else 0.0


def _component_vector(arc_score) -> dict[str, float]:
    return {
        "M_var": float(arc_score.tension_variance),
        "M_peak": float(arc_score.peak_tension),
        "M_shape": float(arc_score.tension_shape),
        "M_sig": float(arc_score.significance),
        "M_theme": float(arc_score.thematic_coherence),
        "M_irony": float(arc_score.irony_arc),
        "M_prot": float(arc_score.protagonist_dominance),
    }


def _component_means(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: _mean([float(row[key]) for row in rows if key in row])
        for key in COMPONENT_KEYS
    }


def _relationship_extremes(final_relationships: dict[str, dict[str, dict[str, float]]]) -> dict[str, Any]:
    count_at_boundary = 0
    total = 0
    for rel_map in final_relationships.values():
        for rel_values in rel_map.values():
            for value in rel_values.values():
                total += 1
                if abs(float(value)) > 0.9:
                    count_at_boundary += 1
    fraction = float(count_at_boundary / total) if total else 0.0
    return {
        "count_at_boundary": int(count_at_boundary),
        "total_relationship_fields": int(total),
        "boundary_fraction": fraction,
    }


def _belief_states_from_agent_view(beliefs: dict[str, dict[str, str]]) -> list[str]:
    states: list[str] = []
    for agent_id in sorted(beliefs):
        for claim_id in sorted(beliefs[agent_id]):
            states.append(str(beliefs[agent_id][claim_id]))
    return states


def _belief_states_from_claim_view(claim_states: dict[str, dict[str, str]]) -> list[str]:
    states: list[str] = []
    for claim_id in sorted(claim_states):
        for agent_id in sorted(claim_states[claim_id]):
            states.append(str(claim_states[claim_id][agent_id]))
    return states


def _belief_entropy(states: list[str]) -> float:
    if not states:
        return 0.0
    counts = Counter(states)
    total = float(sum(counts.values()))
    entropy = 0.0
    for count in counts.values():
        p = float(count) / total
        if p > 0.0:
            entropy -= p * math.log2(p)
    return float(entropy)


def _belief_distribution(states: list[str]) -> dict[str, int]:
    counts = Counter(states)
    return {
        "believes_true": int(counts.get(BeliefState.BELIEVES_TRUE.value, 0)),
        "believes_false": int(counts.get(BeliefState.BELIEVES_FALSE.value, 0)),
        "suspects": int(counts.get(BeliefState.SUSPECTS.value, 0)),
        "unknown": int(counts.get(BeliefState.UNKNOWN.value, 0)),
    }


def _analyze_story(
    *,
    payload: dict[str, Any],
    seed: int,
    wounds: list[WoundPattern],
    collect_components: bool,
) -> dict[str, Any]:
    parsed = parse_simulation_output(payload)
    metrics_output = run_metrics_pipeline(parsed)
    total_sim_time = float(parsed.metadata.get("total_sim_time") or 0.0) or None
    rashomon = extract_rashomon_set(
        events=metrics_output.events,
        seed=seed,
        agents=list(DINNER_PARTY_AGENTS),
        total_sim_time=total_sim_time,
    )

    per_agent_scores: dict[str, float | None] = {agent: None for agent in DINNER_PARTY_AGENTS}
    per_agent_components: dict[str, dict[str, float] | None] = {agent: None for agent in DINNER_PARTY_AGENTS}

    valid_scores: list[float] = []
    valid_scores_excl_elena: list[float] = []
    valid_components: list[dict[str, float]] = []
    valid_components_excl_elena: list[dict[str, float]] = []
    invalid_agents: list[str] = []

    for arc in rashomon.arcs:
        protagonist = str(arc.protagonist)
        score = float(arc.arc_score.composite) if arc.arc_score is not None else None
        per_agent_scores[protagonist] = score

        if collect_components and arc.arc_score is not None:
            vector = _component_vector(arc.arc_score)
            per_agent_components[protagonist] = vector

        if arc.valid and score is not None:
            valid_scores.append(score)
            if protagonist != "elena":
                valid_scores_excl_elena.append(score)
            if collect_components and arc.arc_score is not None:
                vector = _component_vector(arc.arc_score)
                valid_components.append(vector)
                if protagonist != "elena":
                    valid_components_excl_elena.append(vector)
        else:
            invalid_agents.append(protagonist)

    overlap_matrix = {str(pair): float(value) for pair, value in rashomon.overlap_matrix.items()}
    overlap_values = [float(value) for value in overlap_matrix.values()]
    overlap_rows = sorted(overlap_matrix.items(), key=lambda item: (-item[1], item[0]))[:3]

    wound_presence = _detect_wound_presence(rashomon, wounds)
    wounds_present = sorted(pattern for pattern, present in wound_presence.items() if present)

    result: dict[str, Any] = {
        "per_agent_scores": per_agent_scores,
        "mean_score": float(_mean(valid_scores)),
        "mean_score_excl_elena": float(_mean(valid_scores_excl_elena)),
        "valid_arc_count": int(rashomon.valid_count),
        "invalid_agents": sorted(invalid_agents),
        "wounds": wounds_present,
        "wound_count": int(len(wounds_present)),
        "max_overlap": float(max(overlap_values) if overlap_values else 0.0),
        "mean_overlap": float(_mean(overlap_values)),
        "overlap_pairs": [
            {
                "pair": str(pair),
                "jaccard": float(value),
            }
            for pair, value in overlap_rows
        ],
    }

    if collect_components:
        result["per_agent_components"] = per_agent_components
        result["component_means_valid"] = _component_means(valid_components) if valid_components else None
        result["component_means_valid_excl_elena"] = (
            _component_means(valid_components_excl_elena) if valid_components_excl_elena else None
        )
    else:
        result["per_agent_components"] = None
        result["component_means_valid"] = None
        result["component_means_valid_excl_elena"] = None

    return result


def _serialize_evolution_profiles(full_profiles: dict[str, AgentEvolution]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for agent_id in sorted(full_profiles):
        row = full_profiles[agent_id]
        out[agent_id] = {
            "goals_scalar": {k: float(v) for k, v in sorted(row.goals_scalar.items())},
            "closeness": {k: float(v) for k, v in sorted(row.closeness.items())},
            "relationships": {
                target: {attr: float(value) for attr, value in sorted(values.items())}
                for target, values in sorted(row.relationships.items())
            },
            "commitments": None if row.commitments is None else [str(x) for x in row.commitments],
        }
    return out


def _default_seeds() -> list[int]:
    if DEFAULT_SWEEP_REFERENCE_PATH.exists():
        raw = json.loads(DEFAULT_SWEEP_REFERENCE_PATH.read_text(encoding="utf-8"))
        per_seed = list(raw.get("per_seed") or [])
        seeds = [int(row["seed"]) for row in per_seed if isinstance(row, dict) and "seed" in row]
        if seeds:
            return sorted(set(seeds))
    return list(range(1, 21))


def _parse_seeds(raw: str | None) -> list[int]:
    if raw is None or raw.strip() == "":
        return _default_seeds()

    text = raw.strip()
    if "," in text:
        out = [int(item.strip()) for item in text.split(",") if item.strip()]
        if not out:
            raise ValueError("--seeds provided but no valid integers were found.")
        return sorted(set(out))

    if "-" in text:
        left, right = text.split("-", 1)
        start = int(left.strip())
        end = int(right.strip())
        if end < start:
            raise ValueError("Seed range must satisfy end >= start.")
        return list(range(start, end + 1))

    return [int(text)]


def _load_world_canon_from_file(path: Path) -> WorldCanon:
    if not path.exists():
        raise FileNotFoundError(f"Input canon file not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    payload = raw.get("world_canon") if isinstance(raw, dict) and "world_canon" in raw else raw
    return WorldCanon.from_dict(payload)


def _generate_canon_after_b_for_seed(seed: int, event_limit: int, tick_limit: int) -> WorldCanon:
    story_a = _simulate_story(
        label=f"Story A (seed {seed})",
        seed=seed,
        loaded_canon=None,
        tick_limit=tick_limit,
        event_limit=event_limit,
        evolutions={},
    )
    canon_a = WorldCanon.from_dict(story_a.payload.get("world_canon") or {})

    story_b = _simulate_story(
        label=f"Story B (seed {seed})",
        seed=seed,
        loaded_canon=canon_a,
        tick_limit=tick_limit,
        event_limit=event_limit,
        evolutions={},
    )
    return WorldCanon.from_dict(story_b.payload.get("world_canon") or {})


def _build_conditions() -> list[ConditionSpec]:
    out: list[ConditionSpec] = [
        ConditionSpec(condition_type="fresh", k=0, evolved_agents=(), use_canon=False),
        ConditionSpec(condition_type="baseline", k=0, evolved_agents=(), use_canon=True),
    ]

    for k in range(1, 6):
        for subset in itertools.combinations(DINNER_PARTY_AGENTS, k):
            out.append(
                ConditionSpec(
                    condition_type="subset",
                    k=k,
                    evolved_agents=tuple(subset),
                    use_canon=True,
                )
            )

    out.append(
        ConditionSpec(
            condition_type="full_evolution",
            k=6,
            evolved_agents=tuple(DINNER_PARTY_AGENTS),
            use_canon=True,
        )
    )
    return out


def _condition_id(seed: int, spec: ConditionSpec) -> str:
    subset = "none" if not spec.evolved_agents else ",".join(spec.evolved_agents)
    return f"seed={seed}|type={spec.condition_type}|k={spec.k}|subset={subset}"


def _run_to_record(
    *,
    seed: int,
    spec: ConditionSpec,
    loaded_canon: WorldCanon | None,
    event_limit: int,
    tick_limit: int,
    full_profiles: dict[str, AgentEvolution],
    wounds: list[WoundPattern],
    collect_components: bool,
) -> dict[str, Any]:
    evolutions = {agent: full_profiles[agent] for agent in spec.evolved_agents}

    story = _simulate_story(
        label=spec.condition_type,
        seed=seed,
        loaded_canon=loaded_canon,
        tick_limit=tick_limit,
        event_limit=event_limit,
        evolutions=evolutions,
    )

    analyzed = _analyze_story(
        payload=story.payload,
        seed=seed,
        wounds=wounds,
        collect_components=collect_components,
    )

    start_states = _belief_states_from_agent_view(story.start_beliefs)
    end_states = _belief_states_from_claim_view(story.final_claim_states)

    record: dict[str, Any] = {
        "seed": int(seed),
        "k": int(spec.k),
        "evolved_agents": list(spec.evolved_agents),
        "condition_id": _condition_id(seed, spec),
        "condition_type": spec.condition_type,
        "per_agent_scores": analyzed["per_agent_scores"],
        "mean_score": float(analyzed["mean_score"]),
        "mean_score_excl_elena": float(analyzed["mean_score_excl_elena"]),
        "valid_arc_count": int(analyzed["valid_arc_count"]),
        "invalid_agents": list(analyzed["invalid_agents"]),
        "event_counts": _event_type_counts(story.events),
        "total_events": int(len(story.events)),
        "wounds": list(analyzed["wounds"]),
        "wound_count": int(analyzed["wound_count"]),
        "max_overlap": float(analyzed["max_overlap"]),
        "mean_overlap": float(analyzed["mean_overlap"]),
        "overlap_pairs": list(analyzed["overlap_pairs"]),
        "relationship_extremes": _relationship_extremes(story.final_relationships),
        "belief_entropy": {
            "start": float(_belief_entropy(start_states)),
            "end": float(_belief_entropy(end_states)),
        },
        "belief_distribution_start": _belief_distribution(start_states),
        "per_agent_components": analyzed["per_agent_components"],
        "component_means_valid": analyzed["component_means_valid"],
        "component_means_valid_excl_elena": analyzed["component_means_valid_excl_elena"],
    }
    return record


def _numerical_skewness(values: list[float]) -> float:
    if len(values) < 3:
        return 0.0
    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr))
    centered = arr - mean
    m2 = float(np.mean(centered**2))
    if m2 <= 0.0:
        return 0.0
    m3 = float(np.mean(centered**3))
    return float(m3 / (m2 ** 1.5))


def _numerical_kurtosis(values: list[float]) -> float:
    if len(values) < 3:
        return 0.0
    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr))
    centered = arr - mean
    m2 = float(np.mean(centered**2))
    if m2 <= 0.0:
        return 0.0
    m4 = float(np.mean(centered**4))
    return float(m4 / (m2**2))


def _bimodality_coefficient(values: list[float]) -> float:
    if len(values) < 3:
        return 0.0
    kurt = _numerical_kurtosis(values)
    if kurt <= 0.0:
        return 0.0
    skew = _numerical_skewness(values)
    return float((skew**2 + 1.0) / kurt)


def _compute_shapley(coalition_values: dict[frozenset[str], float]) -> dict[str, float]:
    n = len(DINNER_PARTY_AGENTS)
    n_fact = math.factorial(n)
    shapley = {agent: 0.0 for agent in DINNER_PARTY_AGENTS}

    for agent in DINNER_PARTY_AGENTS:
        others = [item for item in DINNER_PARTY_AGENTS if item != agent]
        for k in range(n):
            weight = (math.factorial(k) * math.factorial(n - k - 1)) / n_fact
            for subset in itertools.combinations(others, k):
                base = frozenset(subset)
                with_agent = frozenset((*subset, agent))
                if base not in coalition_values or with_agent not in coalition_values:
                    continue
                delta = float(coalition_values[with_agent] - coalition_values[base])
                shapley[agent] += float(weight * delta)

    return {agent: float(value) for agent, value in sorted(shapley.items())}


def _phase_transition_verdict(
    second_differences: dict[str, float],
    variance_by_k: dict[str, float],
    bimodality_by_k: dict[str, float],
) -> str:
    second_vals = list(second_differences.values())
    has_positive_curvature = any(value > 0.0 for value in second_vals)
    has_all_non_positive = all(value <= 0.0 for value in second_vals)

    interior_keys = [str(k) for k in range(1, 6)]
    peak_k = max(interior_keys, key=lambda key: float(variance_by_k.get(key, 0.0)))
    interior_peak = peak_k in interior_keys

    max_bc = max((float(bimodality_by_k.get(key, 0.0)) for key in bimodality_by_k), default=0.0)

    if has_positive_curvature and interior_peak and max_bc > 0.555:
        return "SUPPORTED"
    if has_all_non_positive and max_bc <= 0.555:
        return "NOT SUPPORTED"
    return "INCONCLUSIVE"


def _elena_verdict(elena_diagnostic: dict[str, float]) -> str:
    frac = abs(float(elena_diagnostic.get("elena_shapley_fraction", 0.0)))
    with_elena = float(elena_diagnostic.get("mean_improvement_with_elena", 0.0))
    without_elena = float(elena_diagnostic.get("mean_improvement_without_elena", 0.0))
    ratio = without_elena / with_elena if abs(with_elena) > EPSILON else 0.0

    if frac >= 0.60 or ratio < 0.40:
        return "ELENA-DOMINATED"
    if frac <= 0.35 and ratio >= 0.70:
        return "SYSTEMIC"
    return "HYBRID"


def _mean_std_n(values: list[float]) -> dict[str, float | int]:
    return {
        "mean": float(_mean(values)),
        "std": float(_std(values)),
        "n": int(len(values)),
    }


def _compute_component_recovery(runs: list[dict[str, Any]]) -> dict[str, Any]:
    by_seed_type: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for run in runs:
        by_seed_type[int(run["seed"])][str(run["condition_type"])] = run

    full_with: dict[str, list[float]] = {key: [] for key in COMPONENT_KEYS}
    full_without: dict[str, list[float]] = {key: [] for key in COMPONENT_KEYS}
    fresh_with: dict[str, list[float]] = {key: [] for key in COMPONENT_KEYS}
    fresh_without: dict[str, list[float]] = {key: [] for key in COMPONENT_KEYS}

    for rows in by_seed_type.values():
        baseline = rows.get("baseline")
        full = rows.get("full_evolution")
        fresh = rows.get("fresh")
        if baseline is None:
            continue

        baseline_with = baseline.get("component_means_valid")
        baseline_without = baseline.get("component_means_valid_excl_elena")

        if full is not None:
            full_with_map = full.get("component_means_valid")
            full_without_map = full.get("component_means_valid_excl_elena")
            if isinstance(baseline_with, dict) and isinstance(full_with_map, dict):
                for key in COMPONENT_KEYS:
                    full_with[key].append(float(full_with_map[key]) - float(baseline_with[key]))
            if isinstance(baseline_without, dict) and isinstance(full_without_map, dict):
                for key in COMPONENT_KEYS:
                    full_without[key].append(float(full_without_map[key]) - float(baseline_without[key]))

        if fresh is not None:
            fresh_with_map = fresh.get("component_means_valid")
            fresh_without_map = fresh.get("component_means_valid_excl_elena")
            if isinstance(baseline_with, dict) and isinstance(fresh_with_map, dict):
                for key in COMPONENT_KEYS:
                    fresh_with[key].append(float(fresh_with_map[key]) - float(baseline_with[key]))
            if isinstance(baseline_without, dict) and isinstance(fresh_without_map, dict):
                for key in COMPONENT_KEYS:
                    fresh_without[key].append(float(fresh_without_map[key]) - float(baseline_without[key]))

    return {
        "components": list(COMPONENT_KEYS),
        "baseline_to_full": {
            "with_elena": {key: _mean_std_n(values) for key, values in full_with.items()},
            "without_elena": {key: _mean_std_n(values) for key, values in full_without.items()},
        },
        "baseline_to_fresh": {
            "with_elena": {key: _mean_std_n(values) for key, values in fresh_with.items()},
            "without_elena": {key: _mean_std_n(values) for key, values in fresh_without.items()},
        },
    }


def _compute_analysis(runs: list[dict[str, Any]]) -> dict[str, Any]:
    nonfresh = [run for run in runs if str(run.get("condition_type")) != "fresh"]
    fresh = [run for run in runs if str(run.get("condition_type")) == "fresh"]

    by_k: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for run in nonfresh:
        by_k[int(run["k"])].append(run)

    mean_by_k: dict[str, dict[str, Any]] = {}
    variance_by_k: dict[str, float] = {}
    for k in range(0, 7):
        rows = by_k.get(k, [])
        scores = [float(row["mean_score"]) for row in rows]
        scores_excl = [float(row["mean_score_excl_elena"]) for row in rows]
        mean_by_k[str(k)] = {
            "mean": float(_mean(scores)),
            "std": float(_std(scores)),
            "mean_excl_elena": float(_mean(scores_excl)),
            "n_runs": int(len(rows)),
        }
        variance_by_k[str(k)] = float(_variance(scores))

    second_differences: dict[str, float] = {}
    for k in range(2, 6):
        q_prev = float(mean_by_k[str(k - 1)]["mean"])
        q_now = float(mean_by_k[str(k)]["mean"])
        q_next = float(mean_by_k[str(k + 1)]["mean"])
        second_differences[f"k{k}"] = float(q_next - (2.0 * q_now) + q_prev)

    coalition_scores: dict[frozenset[str], list[float]] = defaultdict(list)
    runs_by_seed_subset: dict[int, dict[frozenset[str], float]] = defaultdict(dict)
    for row in nonfresh:
        subset = frozenset(str(agent) for agent in row.get("evolved_agents") or [])
        score = float(row["mean_score"])
        seed = int(row["seed"])
        coalition_scores[subset].append(score)
        runs_by_seed_subset[seed][subset] = score

    coalition_mean = {
        subset: float(_mean(values))
        for subset, values in coalition_scores.items()
    }
    shapley_population = _compute_shapley(coalition_mean)

    shapley_by_seed: dict[str, dict[str, float]] = {}
    shapley_series: dict[str, list[float]] = {agent: [] for agent in DINNER_PARTY_AGENTS}
    top_agent_counts = Counter()
    elena_fraction_series: list[float] = []

    for seed in sorted(runs_by_seed_subset):
        values = runs_by_seed_subset[seed]
        if len(values) < 2 ** len(DINNER_PARTY_AGENTS):
            continue
        shapley_seed = _compute_shapley(values)
        shapley_by_seed[str(seed)] = shapley_seed
        for agent in DINNER_PARTY_AGENTS:
            shapley_series[agent].append(float(shapley_seed[agent]))

        top_agent = max(DINNER_PARTY_AGENTS, key=lambda agent: (float(shapley_seed[agent]), agent))
        top_agent_counts[top_agent] += 1

        denom = sum(abs(float(value)) for value in shapley_seed.values())
        elena_fraction = abs(float(shapley_seed["elena"])) / denom if denom > EPSILON else 0.0
        elena_fraction_series.append(float(elena_fraction))

    shapley_stability = {
        "agent_summary": {
            agent: {
                "mean": float(_mean(values)),
                "std": float(_std(values)),
                "min": float(min(values) if values else 0.0),
                "max": float(max(values) if values else 0.0),
                "n_seeds": int(len(values)),
            }
            for agent, values in sorted(shapley_series.items())
        },
        "top_agent_frequency": {
            agent: int(top_agent_counts.get(agent, 0))
            for agent in DINNER_PARTY_AGENTS
        },
        "elena_fraction_distribution": {
            "mean": float(_mean(elena_fraction_series)),
            "std": float(_std(elena_fraction_series)),
            "min": float(min(elena_fraction_series) if elena_fraction_series else 0.0),
            "max": float(max(elena_fraction_series) if elena_fraction_series else 0.0),
            "n_seeds": int(len(elena_fraction_series)),
        },
    }

    by_seed_type: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in runs:
        by_seed_type[int(row["seed"])][str(row["condition_type"])] = row

    improvement_with: list[float] = []
    improvement_without: list[float] = []
    for seed in sorted(by_seed_type):
        baseline = by_seed_type[seed].get("baseline")
        full = by_seed_type[seed].get("full_evolution")
        if baseline is None or full is None:
            continue
        improvement_with.append(float(full["mean_score"]) - float(baseline["mean_score"]))
        improvement_without.append(
            float(full["mean_score_excl_elena"]) - float(baseline["mean_score_excl_elena"])
        )

    shapley_abs_total = sum(abs(float(v)) for v in shapley_population.values())
    elena_fraction = (
        abs(float(shapley_population.get("elena", 0.0))) / shapley_abs_total
        if shapley_abs_total > EPSILON
        else 0.0
    )

    elena_diagnostic = {
        "mean_improvement_with_elena": float(_mean(improvement_with)),
        "mean_improvement_without_elena": float(_mean(improvement_without)),
        "elena_shapley_fraction": float(elena_fraction),
    }

    bimodality_by_k = {
        str(k): _bimodality_coefficient(
            [float(row["mean_score"]) for row in by_k.get(k, [])]
        )
        for k in range(1, 6)
    }

    fresh_scores = [float(row["mean_score"]) for row in fresh]
    fresh_scores_excl = [float(row["mean_score_excl_elena"]) for row in fresh]

    phase_verdict = _phase_transition_verdict(second_differences, variance_by_k, bimodality_by_k)
    elena_verdict = _elena_verdict(elena_diagnostic)

    return {
        "mean_by_k": mean_by_k,
        "fresh_depth0": {
            "mean": float(_mean(fresh_scores)),
            "std": float(_std(fresh_scores)),
            "mean_excl_elena": float(_mean(fresh_scores_excl)),
            "n_runs": int(len(fresh_scores)),
        },
        "second_differences": second_differences,
        "variance_by_k": variance_by_k,
        "shapley_values": shapley_population,
        "shapley_values_population": shapley_population,
        "shapley_values_by_seed": shapley_by_seed,
        "shapley_stability": shapley_stability,
        "elena_diagnostic": elena_diagnostic,
        "phase_transition_indicators": {
            "bimodality_coefficient_by_k": bimodality_by_k,
        },
        "component_recovery": _compute_component_recovery(runs),
        "verdicts": {
            "phase_transition": phase_verdict,
            "elena_driven": elena_verdict,
        },
        "verdict_rules": {
            "phase_transition": "SUPPORTED if positive curvature + interior variance peak + BC>0.555",
            "elena_driven": "ELENA-DOMINATED if Elena fraction high or no-Elena improvement collapses",
        },
    }


def _seed7_compatibility_status(value: float, expected: float) -> dict[str, Any]:
    delta = float(value - expected)
    abs_delta = abs(delta)
    if abs_delta <= WARN_TOLERANCE:
        status = "pass"
    elif abs_delta <= FAIL_TOLERANCE:
        status = "warn"
    else:
        status = "fail"
    return {
        "expected": float(expected),
        "observed": float(value),
        "delta": float(delta),
        "abs_delta": float(abs_delta),
        "status": status,
    }


def _run_seed7_compatibility(
    *,
    canon_after_b: WorldCanon,
    event_limit: int,
    tick_limit: int,
    full_profiles: dict[str, AgentEvolution],
    wounds: list[WoundPattern],
) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}

    baseline = _run_to_record(
        seed=7,
        spec=ConditionSpec("baseline", 0, (), True),
        loaded_canon=canon_after_b,
        event_limit=event_limit,
        tick_limit=tick_limit,
        full_profiles=full_profiles,
        wounds=wounds,
        collect_components=False,
    )
    checks["baseline"] = _seed7_compatibility_status(
        float(baseline["mean_score"]), EXPECTED_SEED7["baseline"]
    )

    full = _run_to_record(
        seed=7,
        spec=ConditionSpec("full_evolution", 6, tuple(DINNER_PARTY_AGENTS), True),
        loaded_canon=canon_after_b,
        event_limit=event_limit,
        tick_limit=tick_limit,
        full_profiles=full_profiles,
        wounds=wounds,
        collect_components=False,
    )
    checks["full_evolution"] = _seed7_compatibility_status(
        float(full["mean_score"]), EXPECTED_SEED7["full_evolution"]
    )

    thorne_only = _run_to_record(
        seed=7,
        spec=ConditionSpec("subset", 1, ("thorne",), True),
        loaded_canon=canon_after_b,
        event_limit=event_limit,
        tick_limit=tick_limit,
        full_profiles=full_profiles,
        wounds=wounds,
        collect_components=False,
    )
    checks["thorne_only"] = _seed7_compatibility_status(
        float(thorne_only["mean_score"]), EXPECTED_SEED7["thorne_only"]
    )

    targeted = _run_to_record(
        seed=7,
        spec=ConditionSpec("subset", 2, ("thorne", "elena"), True),
        loaded_canon=canon_after_b,
        event_limit=event_limit,
        tick_limit=tick_limit,
        full_profiles=full_profiles,
        wounds=wounds,
        collect_components=False,
    )
    checks["targeted"] = _seed7_compatibility_status(
        float(targeted["mean_score"]), EXPECTED_SEED7["targeted"]
    )

    fresh = _run_to_record(
        seed=7,
        spec=ConditionSpec("fresh", 0, (), False),
        loaded_canon=None,
        event_limit=event_limit,
        tick_limit=tick_limit,
        full_profiles=full_profiles,
        wounds=wounds,
        collect_components=False,
    )
    checks["fresh"] = _seed7_compatibility_status(
        float(fresh["mean_score"]), EXPECTED_SEED7["fresh"]
    )

    overall = "pass"
    if any(row["status"] == "fail" for row in checks.values()):
        overall = "fail"
    elif any(row["status"] == "warn" for row in checks.values()):
        overall = "warn"

    return {
        "overall_status": overall,
        "checks": checks,
    }


def _run_determinism_gate(
    *,
    canon_after_b: WorldCanon,
    event_limit: int,
    tick_limit: int,
    full_profiles: dict[str, AgentEvolution],
    wounds: list[WoundPattern],
) -> dict[str, Any]:
    spec = ConditionSpec("baseline", 0, (), True)

    run_a = _run_to_record(
        seed=7,
        spec=spec,
        loaded_canon=canon_after_b,
        event_limit=event_limit,
        tick_limit=tick_limit,
        full_profiles=full_profiles,
        wounds=wounds,
        collect_components=False,
    )
    run_b = _run_to_record(
        seed=7,
        spec=spec,
        loaded_canon=canon_after_b,
        event_limit=event_limit,
        tick_limit=tick_limit,
        full_profiles=full_profiles,
        wounds=wounds,
        collect_components=False,
    )

    story_a = _simulate_story(
        label="determinism_a",
        seed=7,
        loaded_canon=canon_after_b,
        tick_limit=tick_limit,
        event_limit=event_limit,
        evolutions={},
    )
    story_b = _simulate_story(
        label="determinism_b",
        seed=7,
        loaded_canon=canon_after_b,
        tick_limit=tick_limit,
        event_limit=event_limit,
        evolutions={},
    )

    divergence = _first_divergence(story_a.events, story_b.events)
    score_delta = abs(float(run_a["mean_score"]) - float(run_b["mean_score"]))
    valid_match = int(run_a["valid_arc_count"]) == int(run_b["valid_arc_count"])

    ok = divergence is None and score_delta <= DETERMINISM_EPSILON and valid_match

    return {
        "ok": bool(ok),
        "first_divergence": None
        if divergence is None
        else {
            "index": int(divergence.index),
            "tick_id": divergence.tick_id,
            "baseline_event_id": divergence.baseline_event_id,
            "candidate_event_id": divergence.candidate_event_id,
        },
        "score_delta": float(score_delta),
        "valid_arc_match": bool(valid_match),
        "epsilon": float(DETERMINISM_EPSILON),
    }


def _print_summary(payload: dict[str, Any]) -> None:
    analysis = payload["analysis"]
    mean_by_k = analysis["mean_by_k"]
    fresh = analysis["fresh_depth0"]
    second = analysis["second_differences"]
    variance = analysis["variance_by_k"]
    bimodality = analysis["phase_transition_indicators"]["bimodality_coefficient_by_k"]
    shapley = analysis["shapley_values_population"]
    elena_diag = analysis["elena_diagnostic"]
    verdicts = analysis["verdicts"]

    print("\n=== K-SWEEP EXPERIMENT RESULTS ===\n")
    print("Mean Q by k (averaged across all seeds and subsets):")
    for k in range(0, 7):
        row = mean_by_k[str(k)]
        print(
            f"k={k}: {float(row['mean']):.3f} +- {float(row['std']):.3f} "
            f"(n={int(row['n_runs'])}, excl_elena={float(row['mean_excl_elena']):.3f})"
        )
    print(
        f"Fresh (depth-0): {float(fresh['mean']):.3f} +- {float(fresh['std']):.3f} "
        f"(n={int(fresh['n_runs'])})"
    )

    print("\nSecond differences (positive = phase transition evidence):")
    for key in ["k2", "k3", "k4", "k5"]:
        print(f"D^2Q({key[1:]}) = {float(second.get(key, 0.0)):+.4f}")

    peak_k = max([str(k) for k in range(0, 7)], key=lambda key: float(variance.get(key, 0.0)))
    interior_peak_k = max([str(k) for k in range(1, 6)], key=lambda key: float(variance.get(key, 0.0)))
    print(
        f"\nVariance peak at k={peak_k}; interior peak at k={interior_peak_k} "
        f"(BC={float(bimodality.get(interior_peak_k, 0.0)):.3f})"
    )

    print("\nShapley values (population-level):")
    abs_mass = sum(abs(float(value)) for value in shapley.values())
    for agent in DINNER_PARTY_AGENTS:
        value = float(shapley.get(agent, 0.0))
        share = abs(value) / abs_mass if abs_mass > EPSILON else 0.0
        print(f"{agent.title():<7}: {value:+.4f} ({share * 100.0:.1f}% abs-mass share)")

    print("\nElena diagnostic:")
    print(f"  With Elena:    {float(elena_diag['mean_improvement_with_elena']):+.4f}")
    print(f"  Without Elena: {float(elena_diag['mean_improvement_without_elena']):+.4f}")
    print(f"  Elena share:   {float(elena_diag['elena_shapley_fraction']) * 100.0:.1f}%")

    print(f"\nPhase transition verdict: {verdicts['phase_transition']}")
    print(f"Elena-driven verdict:     {verdicts['elena_driven']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run exhaustive k-sweep subset experiment.")
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Seed list: comma-separated (e.g., 1,2,3) or inclusive range (e.g., 1-50).",
    )
    parser.add_argument("--event-limit", type=int, default=200)
    parser.add_argument("--tick-limit", type=int, default=300)
    parser.add_argument(
        "--canon-method",
        type=str,
        choices=["per_seed_chain", "shared_canon"],
        default="per_seed_chain",
    )
    parser.add_argument(
        "--shared-canon-path",
        type=str,
        default=str(DEFAULT_SHARED_CANON_PATH),
    )
    parser.add_argument(
        "--wound-analysis",
        type=str,
        default=str(DEFAULT_WOUND_PATH),
    )
    parser.add_argument(
        "--component-scope",
        type=str,
        choices=["core", "all"],
        default="core",
        help="Collect arc component vectors for core conditions only or all conditions.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Save checkpoint every N completed runs.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file if present.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
    )
    args = parser.parse_args()

    start_time = time.monotonic()
    output_path = _resolve_path(args.output)
    wounds = _load_wound_patterns_or_fail(_resolve_path(args.wound_analysis))

    profiles = _evolution_profiles()
    full_profiles = profiles["full"]

    seeds = _parse_seeds(args.seeds)
    if not seeds:
        raise ValueError("No seeds available for sweep.")

    conditions = _build_conditions()
    total_runs_planned = len(seeds) * len(conditions)

    runs: list[dict[str, Any]] = []
    completed: set[str] = set()
    if args.resume and output_path.exists():
        prior = json.loads(output_path.read_text(encoding="utf-8"))
        loaded_runs = list(prior.get("runs") or [])
        runs = loaded_runs
        completed = {str(row.get("condition_id")) for row in loaded_runs if row.get("condition_id")}
        print(f"Resuming from {output_path}: {len(runs)} completed runs found.")

    if args.canon_method == "shared_canon":
        shared_canon = _load_world_canon_from_file(_resolve_path(args.shared_canon_path))
    else:
        shared_canon = None

    print("Running determinism gate...")
    gate_canon = (
        shared_canon
        if shared_canon is not None
        else _generate_canon_after_b_for_seed(7, args.event_limit, args.tick_limit)
    )
    determinism_gate = _run_determinism_gate(
        canon_after_b=gate_canon,
        event_limit=args.event_limit,
        tick_limit=args.tick_limit,
        full_profiles=full_profiles,
        wounds=wounds,
    )
    if not determinism_gate["ok"]:
        raise RuntimeError(f"Determinism gate failed: {determinism_gate}")

    print("Running seed-7 compatibility checks...")
    compatibility = _run_seed7_compatibility(
        canon_after_b=gate_canon,
        event_limit=args.event_limit,
        tick_limit=args.tick_limit,
        full_profiles=full_profiles,
        wounds=wounds,
    )
    if compatibility["overall_status"] == "fail":
        raise RuntimeError(f"Seed-7 compatibility failed: {compatibility}")

    if compatibility["overall_status"] == "warn":
        print("WARNING: seed-7 compatibility drift exceeded 0.001 on one or more checks.")

    checkpoint_counter = 0

    for seed in seeds:
        if args.canon_method == "per_seed_chain":
            canon_after_b = _generate_canon_after_b_for_seed(seed, args.event_limit, args.tick_limit)
        else:
            if shared_canon is None:
                raise RuntimeError("shared_canon mode selected but no shared canon loaded.")
            canon_after_b = shared_canon

        for spec in conditions:
            cid = _condition_id(seed, spec)
            if cid in completed:
                continue

            collect_components = args.component_scope == "all" or spec.condition_type in CORE_COMPONENT_CONDITIONS
            loaded_canon = canon_after_b if spec.use_canon else None
            run_record = _run_to_record(
                seed=seed,
                spec=spec,
                loaded_canon=loaded_canon,
                event_limit=args.event_limit,
                tick_limit=args.tick_limit,
                full_profiles=full_profiles,
                wounds=wounds,
                collect_components=collect_components,
            )

            runs.append(run_record)
            completed.add(cid)
            checkpoint_counter += 1

            if checkpoint_counter >= max(1, int(args.checkpoint_every)):
                checkpoint_counter = 0
                interim_payload = {
                    "experiment": "k_sweep_subset_sampling",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "config": {
                        "seeds": [int(s) for s in seeds],
                        "canon_method": args.canon_method,
                        "canon_seed_scheme": "A=s,B=s,C=s" if args.canon_method == "per_seed_chain" else "shared",
                        "total_runs": int(total_runs_planned),
                        "event_limit": int(args.event_limit),
                        "tick_limit": int(args.tick_limit),
                        "component_scope": args.component_scope,
                        "resume": bool(args.resume),
                        "determinism_gate": determinism_gate,
                        "compatibility": compatibility,
                        "evolution_profiles": _serialize_evolution_profiles(full_profiles),
                    },
                    "runs": runs,
                    "analysis": {},
                }
                _save_json(output_path, interim_payload)
                print(f"Checkpoint: {len(runs)}/{total_runs_planned} runs saved to {output_path}")

    if len(runs) < total_runs_planned:
        print(
            "WARNING: Sweep finished with missing runs: "
            f"{len(runs)}/{total_runs_planned}. Analysis will use available data only."
        )

    analysis = _compute_analysis(runs)

    final_payload = {
        "experiment": "k_sweep_subset_sampling",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "seeds": [int(s) for s in seeds],
            "canon_method": args.canon_method,
            "canon_seed_scheme": "A=s,B=s,C=s" if args.canon_method == "per_seed_chain" else "shared",
            "total_runs": int(total_runs_planned),
            "event_limit": int(args.event_limit),
            "tick_limit": int(args.tick_limit),
            "component_scope": args.component_scope,
            "resume": bool(args.resume),
            "determinism_gate": determinism_gate,
            "compatibility": compatibility,
            "evolution_profiles": _serialize_evolution_profiles(full_profiles),
            "runtime_seconds": float(time.monotonic() - start_time),
        },
        "runs": runs,
        "analysis": analysis,
    }

    _save_json(output_path, final_payload)
    _print_summary(final_payload)
    print(f"\nSaved experiment artifact: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
