"""Canon persistence demo: run Story A then Story B (fresh vs loaded canon).

This script demonstrates storyworld memory carry-over by:
1) Running seed 42 and saving full simulation output (Story A),
2) Running seed 51 fresh (Story B baseline),
3) Running seed 51 with Story A canon loaded,
4) Printing a side-by-side comparison and writing a summary artifact,
5) Producing renderer payloads (.nf-viz.json) for all three runs.

Run:
    cd src/engine && ./.venv/bin/python -m scripts.demo_canon_persistence
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any

from narrativefield.integration.bundler import BundleInputs, bundle_for_renderer
from narrativefield.metrics.pipeline import parse_simulation_output, run_metrics_pipeline
from narrativefield.schema.agents import BeliefState
from narrativefield.schema.canon import WorldCanon
from narrativefield.schema.events import Event, EventType
from narrativefield.simulation.scenarios import create_dinner_party_world
from narrativefield.simulation.tick_loop import SimulationConfig, init_canon_from_world, run_simulation


FORMAT_VERSION = "1.0.0"
STORY_A_SEED = 42
STORY_B_SEED = 51
EPSILON = 1e-9

EXAMPLE_CLAIMS = ("claim_thorne_health", "claim_guild_pressure")

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
STORY_A_JSON = OUTPUT_DIR / "demo_story_a.json"
STORY_B_FRESH_JSON = OUTPUT_DIR / "demo_story_b_fresh.json"
STORY_B_CANON_JSON = OUTPUT_DIR / "demo_story_b_canon.json"

STORY_A_VIZ_JSON = OUTPUT_DIR / "demo_story_a.nf-viz.json"
STORY_B_FRESH_VIZ_JSON = OUTPUT_DIR / "demo_story_b_fresh.nf-viz.json"
STORY_B_CANON_VIZ_JSON = OUTPUT_DIR / "demo_story_b_canon.nf-viz.json"

SUMMARY_TXT = OUTPUT_DIR / "demo_summary.txt"


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
    determinism_ok: bool


@dataclass(frozen=True)
class Divergence:
    index: int
    fresh_event: Event | None
    canon_event: Event | None


def _sorted_dict_values(mapping: dict[str, Any]) -> list[Any]:
    return [mapping[key] for key in sorted(mapping)]


def _apply_claim_state_overrides(world, canon: WorldCanon) -> None:
    """Mirror simulation.run --canon claim override semantics."""
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
        "format_version": FORMAT_VERSION,
        "metadata": metadata,
        "initial_state": initial_state,
        "snapshots": periodic_snapshots,
        "events": [event.to_dict() for event in events],
        "secrets": [secret.to_dict() for secret in _sorted_dict_values(world.definition.secrets)],
        "claims": [claim.to_dict() for claim in _sorted_dict_values(world.definition.claims)],
        "locations": [location.to_dict() for location in _sorted_dict_values(world.definition.locations)],
        "world_canon": world.canon.to_dict() if world.canon is not None else WorldCanon().to_dict(),
    }


def _payload_signature(payload: dict[str, Any]) -> str:
    stable = {
        "events": payload.get("events", []),
        "initial_state": payload.get("initial_state", {}),
        "snapshots": payload.get("snapshots", []),
        "world_canon": payload.get("world_canon", {}),
    }
    return hashlib.sha256(json.dumps(stable, sort_keys=True).encode("utf-8")).hexdigest()


def _simulate_once(seed: int, loaded_canon: WorldCanon | None) -> tuple[
    dict[str, Any],
    list[Event],
    dict[str, dict[str, str]],
    dict[str, float],
    dict[str, dict[str, str]],
    dict[str, float],
]:
    world = create_dinner_party_world()
    canon_copy = WorldCanon.from_dict(loaded_canon.to_dict()) if loaded_canon is not None else None
    world.canon = init_canon_from_world(world.definition, canon_copy)
    if canon_copy is not None:
        _apply_claim_state_overrides(world, world.canon)

    start_beliefs = _capture_agent_beliefs(world)
    start_location_memory = _capture_location_memory(world)

    cfg = SimulationConfig(
        tick_limit=300,
        event_limit=200,
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
        payload,
        events,
        start_beliefs,
        start_location_memory,
        final_claim_states,
        final_location_memory,
    )


def run_story(label: str, seed: int, loaded_canon: WorldCanon | None) -> StoryRun:
    (
        payload,
        events,
        start_beliefs,
        start_location_memory,
        final_claim_states,
        final_location_memory,
    ) = _simulate_once(seed, loaded_canon)

    (
        payload_repeat,
        _events_repeat,
        _start_beliefs_repeat,
        _start_location_memory_repeat,
        _final_claim_states_repeat,
        _final_location_memory_repeat,
    ) = _simulate_once(seed, loaded_canon)

    determinism_ok = _payload_signature(payload) == _payload_signature(payload_repeat)

    return StoryRun(
        label=label,
        seed=seed,
        payload=payload,
        events=events,
        start_beliefs=start_beliefs,
        start_location_memory=start_location_memory,
        final_claim_states=final_claim_states,
        final_location_memory=final_location_memory,
        determinism_ok=determinism_ok,
    )


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _invert_claim_states(agent_beliefs: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
    claim_ids: set[str] = set()
    for beliefs in agent_beliefs.values():
        claim_ids.update(beliefs)

    by_claim: dict[str, dict[str, str]] = {}
    for claim_id in sorted(claim_ids):
        by_claim[claim_id] = {
            agent_id: agent_beliefs.get(agent_id, {}).get(claim_id, BeliefState.UNKNOWN.value)
            for agent_id in sorted(agent_beliefs)
        }
    return by_claim


def _format_claim_state_map(claim_states: dict[str, dict[str, str]]) -> list[str]:
    lines: list[str] = []
    for claim_id in sorted(claim_states):
        states = ", ".join(
            f"{agent_id}={state}" for agent_id, state in sorted((claim_states.get(claim_id) or {}).items())
        )
        lines.append(f"  {claim_id}: {states}")
    return lines


def _format_claim_distribution(claim_id: str, claim_states: dict[str, dict[str, str]]) -> str:
    states_by_agent = claim_states.get(claim_id, {})
    counts = Counter(states_by_agent.values())
    ordered_states = (
        BeliefState.UNKNOWN.value,
        BeliefState.SUSPECTS.value,
        BeliefState.BELIEVES_TRUE.value,
        BeliefState.BELIEVES_FALSE.value,
    )
    formatted = ", ".join(f"{state}={counts.get(state, 0)}" for state in ordered_states)
    return f"  {claim_id}: {formatted}"


def _event_type_counts(events: list[Event]) -> Counter[str]:
    return Counter(event.type.value for event in events)


def _catastrophe_conflict_by_location(events: list[Event]) -> dict[str, list[Event]]:
    grouped: dict[str, list[Event]] = defaultdict(list)
    for event in events:
        if event.type in (EventType.CATASTROPHE, EventType.CONFLICT):
            grouped[event.location_id].append(event)
    return {location_id: grouped[location_id] for location_id in sorted(grouped)}


def _format_location_memory_with_context(
    location_memory: dict[str, float],
    events: list[Event],
    compare_to: dict[str, float] | None = None,
) -> list[str]:
    per_location_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for event in events:
        if event.type in (EventType.CATASTROPHE, EventType.CONFLICT):
            per_location_counts[event.location_id][event.type.value] += 1

    lines: list[str] = []
    for location_id in sorted(location_memory):
        residue = location_memory[location_id]
        cat_count = per_location_counts[location_id]["catastrophe"]
        conflict_count = per_location_counts[location_id]["conflict"]
        details = []
        if cat_count:
            details.append(f"{cat_count} catastrophe{'s' if cat_count != 1 else ''}")
        if conflict_count:
            details.append(f"{conflict_count} conflict{'s' if conflict_count != 1 else ''}")
        detail_text = f" ({', '.join(details)})" if details else ""

        carry_text = ""
        if compare_to is not None:
            prior = float(compare_to.get(location_id, 0.0))
            if abs(prior - residue) <= EPSILON and residue > 0.0:
                carry_text = " <- carried from Story A"

        lines.append(f"  {location_id:<12} tension_residue = {residue:.2f}{detail_text}{carry_text}")
    return lines


def _format_catastrophe_conflict_events(events: list[Event]) -> list[str]:
    grouped = _catastrophe_conflict_by_location(events)
    if not grouped:
        return ["  (none)"]

    lines: list[str] = []
    for location_id, loc_events in grouped.items():
        lines.append(f"  {location_id}:")
        for event in loc_events:
            lines.append(
                f"    - [{event.type.value.upper()}] {event.id} tick={event.tick_id}: {event.description}"
            )
    return lines


def _learned_secrets(
    start_beliefs: dict[str, dict[str, str]],
    final_claim_states: dict[str, dict[str, str]],
) -> dict[str, list[str]]:
    learned: dict[str, list[str]] = {}
    for claim_id, states_by_agent in sorted(final_claim_states.items()):
        if not claim_id.startswith("secret_"):
            continue
        per_secret: list[str] = []
        for agent_id, final_state in sorted(states_by_agent.items()):
            before = start_beliefs.get(agent_id, {}).get(claim_id, BeliefState.UNKNOWN.value)
            if before != final_state and final_state == BeliefState.BELIEVES_TRUE.value:
                per_secret.append(f"{agent_id} ({before} -> {final_state})")
        if per_secret:
            learned[claim_id] = per_secret
    return learned


def _event_signature(event: Event) -> tuple[Any, ...]:
    return (
        event.tick_id,
        event.order_in_tick,
        event.type.value,
        event.source_agent,
        tuple(event.target_agents),
        event.location_id,
        event.description,
        event.dialogue or "",
        json.dumps(event.content_metadata or {}, sort_keys=True),
        tuple(
            (
                delta.kind.value,
                delta.agent,
                delta.agent_b or "",
                delta.attribute,
                delta.op.value,
                str(delta.value),
            )
            for delta in event.deltas
        ),
    )


def _first_divergence(fresh_events: list[Event], canon_events: list[Event]) -> Divergence | None:
    limit = min(len(fresh_events), len(canon_events))
    for idx in range(limit):
        if _event_signature(fresh_events[idx]) != _event_signature(canon_events[idx]):
            return Divergence(index=idx, fresh_event=fresh_events[idx], canon_event=canon_events[idx])
    if len(fresh_events) != len(canon_events):
        return Divergence(
            index=limit,
            fresh_event=fresh_events[limit] if limit < len(fresh_events) else None,
            canon_event=canon_events[limit] if limit < len(canon_events) else None,
        )
    return None


def _starting_belief_diffs(
    fresh_start: dict[str, dict[str, str]],
    canon_start: dict[str, dict[str, str]],
) -> list[tuple[str, str, str, str]]:
    diffs: list[tuple[str, str, str, str]] = []
    for agent_id in sorted(set(fresh_start) | set(canon_start)):
        claim_ids = sorted(set(fresh_start.get(agent_id, {})) | set(canon_start.get(agent_id, {})))
        for claim_id in claim_ids:
            fresh_state = fresh_start.get(agent_id, {}).get(claim_id, BeliefState.UNKNOWN.value)
            canon_state = canon_start.get(agent_id, {}).get(claim_id, BeliefState.UNKNOWN.value)
            if fresh_state != canon_state:
                diffs.append((agent_id, claim_id, fresh_state, canon_state))
    return diffs


def _starting_memory_diffs(
    fresh_start: dict[str, float],
    canon_start: dict[str, float],
) -> list[tuple[str, float, float]]:
    diffs: list[tuple[str, float, float]] = []
    for location_id in sorted(set(fresh_start) | set(canon_start)):
        fresh_value = float(fresh_start.get(location_id, 0.0))
        canon_value = float(canon_start.get(location_id, 0.0))
        if abs(fresh_value - canon_value) > EPSILON:
            diffs.append((location_id, fresh_value, canon_value))
    return diffs


def _save_viz_payload(sim_payload: dict[str, Any], out_path: Path) -> None:
    parsed = parse_simulation_output(sim_payload)
    metrics_output = run_metrics_pipeline(parsed)
    bundled_payload = bundle_for_renderer(
        BundleInputs(
            metadata=parsed.metadata,
            initial_agents=parsed.initial_agents,
            snapshots=metrics_output.belief_snapshots,
            events=metrics_output.events,
            scenes=metrics_output.scenes,
            secrets=parsed.secrets,
            locations=parsed.locations,
            world_canon=parsed.world_canon.to_dict() if parsed.world_canon is not None else None,
        )
    )
    _save_json(out_path, bundled_payload)


def _build_story_a_section(story_a: StoryRun) -> list[str]:
    event_counts = _event_type_counts(story_a.events)
    lines = [
        "STORY A (seed 42) - The First Evening",
        "--------------------------------------",
        f"Events: {len(story_a.events)}",
        f"Catastrophes: {event_counts.get(EventType.CATASTROPHE.value, 0)}",
        f"Conflicts: {event_counts.get(EventType.CONFLICT.value, 0)}",
        "",
        "Locations with CATASTROPHE/CONFLICT events:",
        *_format_catastrophe_conflict_events(story_a.events),
        "",
        "Location Memory After Story A:",
        *_format_location_memory_with_context(story_a.final_location_memory, story_a.events),
        "",
        "Belief States After Story A (canon.claim_states):",
        *_format_claim_state_map(story_a.final_claim_states),
        "",
        "Agents who learned secrets in Story A:",
    ]

    learned = _learned_secrets(story_a.start_beliefs, story_a.final_claim_states)
    if learned:
        for claim_id, agents in learned.items():
            lines.append(f"  {claim_id}: {', '.join(agents)}")
    else:
        lines.append("  (none)")

    lines.extend(
        [
            "",
            "Example claim distributions after Story A:",
            *[_format_claim_distribution(claim_id, story_a.final_claim_states) for claim_id in EXAMPLE_CLAIMS],
            "",
        ]
    )
    return lines


def _build_story_b_fresh_section(story_b_fresh: StoryRun) -> list[str]:
    event_counts = _event_type_counts(story_b_fresh.events)
    start_by_claim = _invert_claim_states(story_b_fresh.start_beliefs)
    all_zero = all(abs(value) <= EPSILON for value in story_b_fresh.start_location_memory.values())
    defaults_world = create_dinner_party_world()
    defaults = _capture_agent_beliefs(defaults_world)
    defaults_ok = defaults == story_b_fresh.start_beliefs

    lines = [
        "STORY B (seed 51) - Fresh Start",
        "-------------------------------",
        "Starting location memory: all 0.0" if all_zero else "Starting location memory: WARNING (not all zero)",
        "Starting beliefs: scenario defaults" if defaults_ok else "Starting beliefs: WARNING (not scenario defaults)",
    ]
    if not all_zero:
        lines.extend(_format_location_memory_with_context(story_b_fresh.start_location_memory, []))
    lines.extend(
        [
            "Starting beliefs by claim:",
            *_format_claim_state_map(start_by_claim),
            "",
            f"Events: {len(story_b_fresh.events)}",
            f"Catastrophes: {event_counts.get(EventType.CATASTROPHE.value, 0)}",
            f"Conflicts: {event_counts.get(EventType.CONFLICT.value, 0)}",
            "",
            "Locations with CATASTROPHE/CONFLICT events:",
            *_format_catastrophe_conflict_events(story_b_fresh.events),
            "",
            "Final location memory:",
            *_format_location_memory_with_context(story_b_fresh.final_location_memory, story_b_fresh.events),
            "",
            "Final belief states:",
            *_format_claim_state_map(story_b_fresh.final_claim_states),
            "",
            "Example claim distributions after Story B fresh:",
            *[_format_claim_distribution(claim_id, story_b_fresh.final_claim_states) for claim_id in EXAMPLE_CLAIMS],
            "",
        ]
    )
    return lines


def _build_story_b_canon_section(story_b_canon: StoryRun, story_a: StoryRun, story_b_fresh: StoryRun) -> list[str]:
    event_counts = _event_type_counts(story_b_canon.events)
    start_by_claim = _invert_claim_states(story_b_canon.start_beliefs)
    start_diffs = _starting_belief_diffs(story_b_fresh.start_beliefs, story_b_canon.start_beliefs)

    lines = [
        "STORY B (seed 51) - With Story A Canon",
        "---------------------------------------",
        "Starting location memory:",
        *_format_location_memory_with_context(
            story_b_canon.start_location_memory,
            [],
            compare_to=story_a.final_location_memory,
        ),
        "",
        "Starting beliefs (overridden by Story A final claim states):",
    ]

    if start_diffs:
        for agent_id, claim_id, fresh_state, canon_state in start_diffs:
            lines.append(
                f"  {agent_id}.{claim_id}: {canon_state} <- was {fresh_state} in fresh start"
            )
    else:
        lines.append("  (no overrides detected)")

    lines.extend(
        [
            "",
            "Starting beliefs by claim:",
            *_format_claim_state_map(start_by_claim),
            "",
            f"Events: {len(story_b_canon.events)}",
            f"Catastrophes: {event_counts.get(EventType.CATASTROPHE.value, 0)}",
            f"Conflicts: {event_counts.get(EventType.CONFLICT.value, 0)}",
            "",
            "Locations with CATASTROPHE/CONFLICT events:",
            *_format_catastrophe_conflict_events(story_b_canon.events),
            "",
            "Final location memory:",
            *_format_location_memory_with_context(story_b_canon.final_location_memory, story_b_canon.events),
            "",
            "Final belief states:",
            *_format_claim_state_map(story_b_canon.final_claim_states),
            "",
            "Example claim distributions after Story B with canon:",
            *[_format_claim_distribution(claim_id, story_b_canon.final_claim_states) for claim_id in EXAMPLE_CLAIMS],
            "",
        ]
    )
    return lines


def _build_comparison_section(story_b_fresh: StoryRun, story_b_canon: StoryRun) -> list[str]:
    belief_diffs = _starting_belief_diffs(story_b_fresh.start_beliefs, story_b_canon.start_beliefs)
    memory_diffs = _starting_memory_diffs(
        story_b_fresh.start_location_memory,
        story_b_canon.start_location_memory,
    )
    divergence = _first_divergence(story_b_fresh.events, story_b_canon.events)

    fresh_counts = _event_type_counts(story_b_fresh.events)
    canon_counts = _event_type_counts(story_b_canon.events)

    lines = [
        "WHAT CHANGED (Story B Fresh vs Story B With Canon)",
        "===================================================",
        "",
        "Initial belief differences:",
    ]
    if belief_diffs:
        for agent_id, claim_id, fresh_state, canon_state in belief_diffs:
            lines.append(
                f"  {agent_id}.{claim_id}: {fresh_state} -> {canon_state} (inherited from Story A)"
            )
    else:
        lines.append("  (none)")

    lines.extend(
        [
            "",
            "Location memory at start:",
        ]
    )
    if memory_diffs:
        for location_id, fresh_value, canon_value in memory_diffs:
            lines.append(f"  {location_id}: {fresh_value:.2f} -> {canon_value:.2f}")
    else:
        lines.append("  (none)")

    lines.extend(
        [
            "",
            "Simulation divergence:",
            f"  Event count: {len(story_b_fresh.events)} (fresh) vs {len(story_b_canon.events)} (canon)",
            "  Event count note: same count is acceptable if stream content diverges materially."
            if len(story_b_fresh.events) == len(story_b_canon.events)
            else "  Event count note: counts differ.",
            "  Catastrophe count: "
            f"{fresh_counts.get(EventType.CATASTROPHE.value, 0)} (fresh) vs "
            f"{canon_counts.get(EventType.CATASTROPHE.value, 0)} (canon)",
            "  Conflict count: "
            f"{fresh_counts.get(EventType.CONFLICT.value, 0)} (fresh) vs "
            f"{canon_counts.get(EventType.CONFLICT.value, 0)} (canon)",
        ]
    )

    if divergence is None:
        lines.append("  First divergence: none (event streams identical)")
    else:
        lines.append(f"  First divergence at event index: {divergence.index}")
        if divergence.fresh_event is not None:
            lines.append(
                "    fresh: "
                f"tick={divergence.fresh_event.tick_id} {divergence.fresh_event.id} "
                f"{divergence.fresh_event.type.value} @{divergence.fresh_event.location_id} "
                f"src={divergence.fresh_event.source_agent}"
            )
        if divergence.canon_event is not None:
            lines.append(
                "    canon: "
                f"tick={divergence.canon_event.tick_id} {divergence.canon_event.id} "
                f"{divergence.canon_event.type.value} @{divergence.canon_event.location_id} "
                f"src={divergence.canon_event.source_agent}"
            )

    has_non_zero_start_residue = any(new > 0.0 for _, _, new in memory_diffs)
    has_min_belief_diffs = len(belief_diffs) >= 3
    has_stream_divergence = (
        divergence is not None
        or len(story_b_fresh.events) != len(story_b_canon.events)
        or fresh_counts.get(EventType.CATASTROPHE.value, 0)
        != canon_counts.get(EventType.CATASTROPHE.value, 0)
        or fresh_counts.get(EventType.CONFLICT.value, 0) != canon_counts.get(EventType.CONFLICT.value, 0)
    )

    lines.extend(["", "Proof checks:"])
    lines.append(
        f"  Belief inheritance differences: {len(belief_diffs)} "
        f"{'(PASS)' if has_min_belief_diffs else '(WARNING: expected >= 3)'}"
    )
    lines.append(
        "  Canon-loaded non-zero starting tension residue: "
        f"{'PASS' if has_non_zero_start_residue else 'WARNING'}"
    )
    lines.append(
        "  Detectable simulation divergence: "
        f"{'PASS' if has_stream_divergence else 'ERROR'}"
    )

    if not has_stream_divergence:
        lines.append(
            "  ERROR: canon-loaded run matches fresh run exactly; check canon loading or claim-state overrides."
        )

    return lines


def _build_determinism_section(*runs: StoryRun) -> list[str]:
    lines = [
        "",
        "Determinism checks:",
    ]
    for run in runs:
        lines.append(
            f"  {run.label} (seed {run.seed}): "
            f"{'PASS' if run.determinism_ok else 'WARNING (repeat run mismatch)'}"
        )
    return lines


def _build_report(story_a: StoryRun, story_b_fresh: StoryRun, story_b_canon: StoryRun) -> str:
    lines: list[str] = [
        "=== CANON PERSISTENCE DEMO ===",
        "",
        *_build_story_a_section(story_a),
        *_build_story_b_fresh_section(story_b_fresh),
        *_build_story_b_canon_section(story_b_canon, story_a, story_b_fresh),
        *_build_comparison_section(story_b_fresh, story_b_canon),
        *_build_determinism_section(story_a, story_b_fresh, story_b_canon),
        "",
        "Visualization payloads saved. Load in the explorer:",
        "  cd src/visualization && npm run dev",
        "  Then use Control Panel -> Load file to compare runs.",
        "",
        "Output artifacts:",
        f"  {STORY_A_JSON}",
        f"  {STORY_B_FRESH_JSON}",
        f"  {STORY_B_CANON_JSON}",
        f"  {STORY_A_VIZ_JSON}",
        f"  {STORY_B_FRESH_VIZ_JSON}",
        f"  {STORY_B_CANON_VIZ_JSON}",
        f"  {SUMMARY_TXT}",
    ]
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    story_a = run_story("Story A", STORY_A_SEED, loaded_canon=None)
    _save_json(STORY_A_JSON, story_a.payload)

    story_b_fresh = run_story("Story B Fresh", STORY_B_SEED, loaded_canon=None)
    _save_json(STORY_B_FRESH_JSON, story_b_fresh.payload)

    story_a_raw = json.loads(STORY_A_JSON.read_text(encoding="utf-8"))
    loaded_canon = WorldCanon.from_dict(story_a_raw.get("world_canon"))
    story_b_canon = run_story("Story B With Canon", STORY_B_SEED, loaded_canon=loaded_canon)
    _save_json(STORY_B_CANON_JSON, story_b_canon.payload)

    _save_viz_payload(story_a.payload, STORY_A_VIZ_JSON)
    _save_viz_payload(story_b_fresh.payload, STORY_B_FRESH_VIZ_JSON)
    _save_viz_payload(story_b_canon.payload, STORY_B_CANON_VIZ_JSON)

    report = _build_report(story_a, story_b_fresh, story_b_canon)
    SUMMARY_TXT.write_text(report, encoding="utf-8")
    print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
