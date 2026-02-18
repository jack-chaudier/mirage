"""Perturbation experiment: test categorical claim-state sensitivity in Story C.

Run:
    cd src/engine && ./.venv/bin/python -m scripts.test_epiphany_bypass
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from random import Random
from typing import Any, Iterable

from narrativefield.extraction.rashomon import RashomonSet, extract_rashomon_set
from narrativefield.metrics.pipeline import parse_simulation_output, run_metrics_pipeline
from narrativefield.schema.agents import BeliefState
from narrativefield.schema.canon import WorldCanon
from narrativefield.schema.events import BeatType, Event
from narrativefield.simulation.scenarios import create_dinner_party_world
from narrativefield.simulation.tick_loop import SimulationConfig, init_canon_from_world, run_simulation

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
WOUND_ANALYSIS_PATH = OUTPUT_DIR / "wound_analysis_1_50.json"
DINNER_PARTY_AGENTS = ["thorne", "elena", "marcus", "lydia", "diana", "victor"]
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
class Mutation:
    claim_id: str
    agent_id: str
    old_value: str
    new_value: str
    reason: str


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


class TraceRecorder:
    """In-memory collector for belief read accesses."""

    def __init__(self, max_raw_lines: int = 5000) -> None:
        self.max_raw_lines = int(max_raw_lines)
        self.raw_lines: list[str] = []
        self.field_counts: Counter[tuple[str, str]] = Counter()
        self.via_counts: Counter[str] = Counter()

    def log(self, *, agent_id: str, claim_id: str, value: Any, via: str) -> None:
        self.field_counts[(str(agent_id), str(claim_id))] += 1
        self.via_counts[str(via)] += 1
        if len(self.raw_lines) < self.max_raw_lines:
            self.raw_lines.append(
                f"CANON_READ: agent={agent_id}, field=beliefs.{claim_id}, value={value}, via={via}"
            )


# TEMP: EPIPHANY EXPERIMENT
class TracedBeliefs(dict[str, BeliefState]):
    """Dict wrapper that records belief read accesses during simulation."""

    def __init__(self, agent_id: str, base: dict[str, BeliefState], recorder: TraceRecorder):
        super().__init__(base)
        self._agent_id = str(agent_id)
        self._recorder = recorder

    def __getitem__(self, key: str) -> BeliefState:
        value = super().__getitem__(key)
        self._recorder.log(agent_id=self._agent_id, claim_id=str(key), value=getattr(value, "value", value), via="getitem")
        return value

    def get(self, key: str, default: Any = None) -> Any:
        value = super().get(key, default)
        self._recorder.log(agent_id=self._agent_id, claim_id=str(key), value=getattr(value, "value", value), via="get")
        return value

    def items(self) -> list[tuple[str, BeliefState]]:
        pairs = list(super().items())
        for key, value in pairs:
            self._recorder.log(agent_id=self._agent_id, claim_id=str(key), value=getattr(value, "value", value), via="items")
        return pairs


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


def _simulate_story(
    *,
    label: str,
    seed: int,
    loaded_canon: WorldCanon | None,
    tick_limit: int,
    event_limit: int,
    trace_recorder: TraceRecorder | None = None,
) -> StoryRun:
    world = create_dinner_party_world()
    canon_copy = WorldCanon.from_dict(loaded_canon.to_dict()) if loaded_canon is not None else None
    world.canon = init_canon_from_world(world.definition, canon_copy)
    if canon_copy is not None:
        _apply_claim_state_overrides(world, world.canon)

    # TEMP: EPIPHANY EXPERIMENT
    if trace_recorder is not None:
        for agent_id, agent in world.agents.items():
            agent.beliefs = TracedBeliefs(agent_id=agent_id, base=dict(agent.beliefs), recorder=trace_recorder)

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

    return StoryRun(
        label=label,
        seed=seed,
        payload=payload,
        events=events,
        start_beliefs=start_beliefs,
        start_location_memory=start_location_memory,
        final_claim_states=final_claim_states,
        final_location_memory=final_location_memory,
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


def _next_belief_state(old_value: str) -> str | None:
    value = str(old_value).strip().lower()
    if value == BeliefState.UNKNOWN.value:
        return BeliefState.SUSPECTS.value
    if value == BeliefState.SUSPECTS.value:
        return BeliefState.BELIEVES_TRUE.value
    if value == BeliefState.BELIEVES_FALSE.value:
        return BeliefState.SUSPECTS.value
    return None


def _claims_with_asymmetry(claim_states: dict[str, dict[str, str]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for claim_id, states_by_agent in sorted(claim_states.items()):
        counts = Counter(str(state) for state in states_by_agent.values())
        has_known_true = counts.get(BeliefState.BELIEVES_TRUE.value, 0) > 0
        has_not_true = sum(v for k, v in counts.items() if k != BeliefState.BELIEVES_TRUE.value) > 0
        if has_known_true and has_not_true:
            out[claim_id] = {
                "claim_id": claim_id,
                "counts": dict(counts),
            }
    return out


def _mutate_canon(canon: WorldCanon, mutation: Mutation) -> None:
    canon.claim_states.setdefault(mutation.claim_id, {})[mutation.agent_id] = mutation.new_value
    # Keep confidence map coherent if present; simulation currently reads categorical state only.
    if mutation.new_value == BeliefState.UNKNOWN.value:
        if mutation.claim_id in canon.claim_confidence:
            canon.claim_confidence[mutation.claim_id].pop(mutation.agent_id, None)
            if not canon.claim_confidence[mutation.claim_id]:
                canon.claim_confidence.pop(mutation.claim_id, None)
    else:
        canon.claim_confidence.setdefault(mutation.claim_id, {})[mutation.agent_id] = 1.0


def _select_primary_mutation(claim_states: dict[str, dict[str, str]]) -> Mutation:
    asymmetry_claims = set(_claims_with_asymmetry(claim_states))

    candidates: list[tuple[tuple[Any, ...], Mutation]] = []
    for claim_id, states_by_agent in sorted(claim_states.items()):
        for agent_id, old_value_raw in sorted(states_by_agent.items()):
            old_value = str(old_value_raw)
            new_value = _next_belief_state(old_value)
            if new_value is None:
                continue

            claim_l = claim_id.lower()
            agent_l = agent_id.lower()
            is_thorne = agent_l == "thorne"
            is_affair_claim = "affair" in claim_l
            is_secret = claim_l.startswith("secret_")
            in_asymmetry = claim_id in asymmetry_claims

            priority = (
                0 if (claim_id == "secret_affair_01" and agent_id == "thorne") else 1,
                0 if is_thorne else 1,
                0 if is_affair_claim else 1,
                0 if in_asymmetry else 1,
                0 if is_secret else 1,
                0 if old_value == BeliefState.UNKNOWN.value else 1,
                claim_id,
                agent_id,
            )
            reason = "primary_target" if (claim_id == "secret_affair_01" and agent_id == "thorne") else "fallback_ranked"
            candidates.append(
                (
                    priority,
                    Mutation(
                        claim_id=claim_id,
                        agent_id=agent_id,
                        old_value=old_value,
                        new_value=new_value,
                        reason=reason,
                    ),
                )
            )

    if not candidates:
        raise ValueError("No valid primary mutation candidates found.")

    best = min(candidates, key=lambda item: item[0])[1]
    return best


def _select_control_mutation(claim_states: dict[str, dict[str, str]], primary: Mutation) -> Mutation:
    asymmetry = _claims_with_asymmetry(claim_states)

    non_unknown_by_claim: dict[str, int] = {}
    for claim_id, states in claim_states.items():
        non_unknown_by_claim[claim_id] = sum(1 for state in states.values() if str(state) != BeliefState.UNKNOWN.value)

    central_tokens = ("affair", "embezzle", "investigation")
    candidates: list[tuple[tuple[Any, ...], Mutation]] = []

    for claim_id, states_by_agent in sorted(claim_states.items()):
        if claim_id == primary.claim_id:
            continue
        claim_l = claim_id.lower()
        is_secret = claim_l.startswith("secret_")
        central_hit = any(token in claim_l for token in central_tokens)

        for agent_id, old_value_raw in sorted(states_by_agent.items()):
            old_value = str(old_value_raw)
            new_value = _next_belief_state(old_value)
            if new_value is None:
                continue
            if claim_id == primary.claim_id and agent_id == primary.agent_id:
                continue

            priority = (
                0 if not is_secret else 1,
                0 if not central_hit else 1,
                non_unknown_by_claim.get(claim_id, 0),
                0 if claim_id in asymmetry else 1,
                0 if old_value == BeliefState.UNKNOWN.value else 1,
                claim_id,
                agent_id,
            )
            candidates.append(
                (
                    priority,
                    Mutation(
                        claim_id=claim_id,
                        agent_id=agent_id,
                        old_value=old_value,
                        new_value=new_value,
                        reason="control_ranked",
                    ),
                )
            )

    if not candidates:
        raise ValueError("No valid control mutation candidates found.")

    return min(candidates, key=lambda item: item[0])[1]


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


def _invert_claim_states(claim_states: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = defaultdict(dict)
    for claim_id, states_by_agent in sorted(claim_states.items()):
        for agent_id, state in sorted(states_by_agent.items()):
            out[str(agent_id)][str(claim_id)] = str(state)
    return {agent_id: dict(claims) for agent_id, claims in sorted(out.items())}


def _claim_counts_per_agent(claim_states: dict[str, dict[str, str]]) -> dict[str, dict[str, int]]:
    totals: Counter[str] = Counter()
    non_unknown: Counter[str] = Counter()
    for states_by_agent in claim_states.values():
        for agent_id, state in states_by_agent.items():
            agent = str(agent_id)
            totals[agent] += 1
            if str(state) != BeliefState.UNKNOWN.value:
                non_unknown[agent] += 1

    return {
        agent_id: {
            "total": int(totals.get(agent_id, 0)),
            "non_unknown": int(non_unknown.get(agent_id, 0)),
        }
        for agent_id in sorted(set(totals) | set(non_unknown))
    }


def _save_trace_summary(path: Path, recorder: TraceRecorder) -> dict[str, Any]:
    top_fields = sorted(recorder.field_counts.items(), key=lambda item: (-item[1], item[0]))
    lines: list[str] = [
        "SIM READ PATH TRACE (belief reads)",
        "===============================",
        f"Total read entries captured: {sum(recorder.field_counts.values())}",
        f"Unique agent/claim fields: {len(recorder.field_counts)}",
        "",
        "Reads by access method:",
    ]
    for via, count in sorted(recorder.via_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"  {via}: {int(count)}")

    lines.extend(["", "Top read fields:"])
    for (agent_id, claim_id), count in top_fields[:30]:
        lines.append(f"  {agent_id}.beliefs.{claim_id}: {int(count)}")

    lines.extend(["", "Sample raw lines:"])
    for raw in recorder.raw_lines[:200]:
        lines.append(raw)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "total_reads": int(sum(recorder.field_counts.values())),
        "unique_fields": int(len(recorder.field_counts)),
        "via_counts": {k: int(v) for k, v in sorted(recorder.via_counts.items())},
        "top_fields": [
            {
                "agent": agent,
                "claim": claim,
                "reads": int(count),
            }
            for (agent, claim), count in top_fields[:30]
        ],
        "trace_path": str(path),
    }


def _fmt_score(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def _print_report(report: dict[str, Any]) -> None:
    canon = report["canon_state_before_mutation"]
    primary = report["mutations"]["primary"]
    control = report["mutations"]["control"]
    metrics = report["metrics"]

    print("EPIPHANY BYPASS EXPERIMENT RESULTS")
    print("===================================")
    print()
    print("CANON STATE BEFORE MUTATION:")
    print(f"- Total claim_states per agent: {canon['claim_counts_per_agent']}")
    print(f"- Thorne's beliefs: {canon['thorne_beliefs']}")
    print(f"- Key asymmetries: {[row['claim_id'] for row in canon['key_asymmetries']]}")
    print()

    print("MUTATION APPLIED:")
    print(f"- Agent: {primary['agent_id']}")
    print(f"- Claim: {primary['claim_id']}")
    print(f"- Old value: {primary['old_value']}")
    print(f"- New value: {primary['new_value']}")
    print()

    score = metrics["score_rebound"]
    print("METRIC 1: SCORE REBOUND")
    print(f"- Baseline Story C mean arc score: {score['baseline_mean']:.3f}")
    print(f"- Mutated Story C mean arc score:  {score['mutated_mean']:.3f}")
    print(f"- Delta: {score['delta']:+.3f}")
    print("- Per-agent scores comparison table:")
    for row in score["per_agent"]:
        print(
            f"  {row['agent']:<7} baseline={_fmt_score(row['baseline'])} "
            f"mutated={_fmt_score(row['mutated'])} delta={row['delta']:+.3f}"
        )
    print()

    wounds = metrics["wound_topology"]
    print("METRIC 2: WOUND TOPOLOGY SHIFT")
    print(f"- Baseline wounds present: {[row['pattern'] for row in wounds['baseline_present']]}")
    print(f"- Mutated wounds present:  {[row['pattern'] for row in wounds['mutated_present']]}")
    print(f"- Wounds that DISAPPEARED: {[row['pattern'] for row in wounds['disappeared']]}")
    print(f"- Wounds that APPEARED:    {[row['pattern'] for row in wounds['appeared']]}")
    print()

    overlap = metrics["overlap_matrix_change"]
    print("METRIC 3: OVERLAP MATRIX CHANGE")
    print(f"- Baseline top overlap pairs: {overlap['baseline_top5']}")
    print(f"- Mutated top overlap pairs:  {overlap['mutated_top5']}")
    print(f"- Largest shifts: {overlap['largest_shifts']}")
    print()

    validity = metrics["arc_validity"]
    print("METRIC 4: ARC VALIDITY")
    print(f"- Baseline valid arcs: {validity['baseline_valid']}/6")
    print(f"- Mutated valid arcs:  {validity['mutated_valid']}/6")
    print()

    divergence = metrics["event_divergence"]
    head = divergence["baseline_vs_mutated"]
    print("METRIC 5: EVENT DIVERGENCE")
    print(
        "- Baseline vs mutated first divergence: "
        f"{head['first_divergence_tick']} (index={head['first_divergence_index']})"
        if not head["identical"]
        else "- Baseline vs mutated sequences are identical"
    )
    print(
        f"- Total events in baseline vs mutated: {head['baseline_event_count']} vs {head['candidate_event_count']}"
    )
    print()

    print("CONTROL MUTATION (less central belief):")
    print(
        f"- What was mutated: {control['agent_id']}.{control['claim_id']} "
        f"{control['old_value']} -> {control['new_value']}"
    )
    print(f"- Score delta: {metrics['control']['delta']:+.3f}")
    print(
        f"- Wound shifts: appeared={len(metrics['control']['wounds_appeared'])}, "
        f"disappeared={len(metrics['control']['wounds_disappeared'])}"
    )
    print()

    print("CONCLUSION:")
    conclusion = report["conclusion"]
    print(f"- Is the simulation sensitive to claim_state mutations? {'YES' if conclusion['sensitive'] else 'NO'}")
    print(
        "- Is the sensitivity specific to structurally central beliefs? "
        f"{'YES' if conclusion['specific_to_central'] else 'NO'}"
    )
    print(
        "- Does a single categorical mutation rescue score degradation? "
        f"{'YES' if conclusion['rescues_score_degradation'] else 'NO'}"
    )
    print()
    print(f"Trace summary written to: {report['trace_summary']['trace_path']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the epiphany bypass perturbation experiment.")
    parser.add_argument("--seed-a", type=int, default=7)
    parser.add_argument("--seed-b", type=int, default=7)
    parser.add_argument("--seed-c", type=int, default=7)
    parser.add_argument("--event-limit", type=int, default=200)
    parser.add_argument("--tick-limit", type=int, default=300)
    parser.add_argument(
        "--output",
        type=str,
        default="scripts/output/epiphany_experiment.json",
        help="Path to JSON report output.",
    )
    parser.add_argument(
        "--canon-after-b",
        type=str,
        default="scripts/output/canon_after_B.json",
        help="Path to save baseline canon after Story B.",
    )
    parser.add_argument(
        "--canon-after-b-mutated",
        type=str,
        default="scripts/output/canon_after_B_mutated.json",
        help="Path to save primary-mutated canon after Story B.",
    )
    parser.add_argument(
        "--trace-output",
        type=str,
        default="scripts/output/sim_read_path_trace.txt",
        help="Path to save simulation read-path trace summary.",
    )
    args = parser.parse_args()

    output_path = _resolve_output_path(args.output)
    canon_b_path = _resolve_output_path(args.canon_after_b)
    canon_b_mutated_path = _resolve_output_path(args.canon_after_b_mutated)
    trace_output_path = _resolve_output_path(args.trace_output)

    wounds = _load_wound_patterns_or_fail(WOUND_ANALYSIS_PATH)
    wound_freq = {w.pattern: float(w.frequency) for w in wounds}

    story_a = _simulate_story(
        label="A",
        seed=int(args.seed_a),
        loaded_canon=None,
        tick_limit=int(args.tick_limit),
        event_limit=int(args.event_limit),
    )
    canon_after_a = WorldCanon.from_dict(story_a.payload.get("world_canon"))

    story_b = _simulate_story(
        label="B",
        seed=int(args.seed_b),
        loaded_canon=canon_after_a,
        tick_limit=int(args.tick_limit),
        event_limit=int(args.event_limit),
    )
    canon_after_b = WorldCanon.from_dict(story_b.payload.get("world_canon"))

    _save_json(
        canon_b_path,
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "seed_a": int(args.seed_a),
            "seed_b": int(args.seed_b),
            "world_canon": canon_after_b.to_dict(),
        },
    )

    baseline_c = _simulate_story(
        label="C_baseline",
        seed=int(args.seed_c),
        loaded_canon=canon_after_b,
        tick_limit=int(args.tick_limit),
        event_limit=int(args.event_limit),
    )
    baseline_metrics = _analyze_story(baseline_c.payload, int(args.seed_c), wounds)

    baseline_c_repeat = _simulate_story(
        label="C_baseline_repeat",
        seed=int(args.seed_c),
        loaded_canon=canon_after_b,
        tick_limit=int(args.tick_limit),
        event_limit=int(args.event_limit),
    )
    baseline_repeat_metrics = _analyze_story(baseline_c_repeat.payload, int(args.seed_c), wounds)
    determinism_divergence = _first_divergence(baseline_c.events, baseline_c_repeat.events)
    determinism_ok = (
        determinism_divergence is None
        and abs(baseline_metrics.mean_score - baseline_repeat_metrics.mean_score) <= EPSILON
        and baseline_metrics.valid_count == baseline_repeat_metrics.valid_count
    )

    claim_states_b = canon_after_b.claim_states
    claim_counts = _claim_counts_per_agent(claim_states_b)
    per_agent_claims = _invert_claim_states(claim_states_b)
    thorne_beliefs = per_agent_claims.get("thorne", {})
    asymmetry_map = _claims_with_asymmetry(claim_states_b)

    primary_mutation = _select_primary_mutation(claim_states_b)
    mutated_canon = WorldCanon.from_dict(canon_after_b.to_dict())
    _mutate_canon(mutated_canon, primary_mutation)

    _save_json(
        canon_b_mutated_path,
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "base_canon_path": str(canon_b_path),
            "mutation": {
                "claim_id": primary_mutation.claim_id,
                "agent_id": primary_mutation.agent_id,
                "old_value": primary_mutation.old_value,
                "new_value": primary_mutation.new_value,
                "reason": primary_mutation.reason,
            },
            "world_canon": mutated_canon.to_dict(),
        },
    )

    mutated_c = _simulate_story(
        label="C_mutated",
        seed=int(args.seed_c),
        loaded_canon=mutated_canon,
        tick_limit=int(args.tick_limit),
        event_limit=int(args.event_limit),
    )
    mutated_metrics = _analyze_story(mutated_c.payload, int(args.seed_c), wounds)

    control_mutation = _select_control_mutation(claim_states_b, primary_mutation)
    control_canon = WorldCanon.from_dict(canon_after_b.to_dict())
    _mutate_canon(control_canon, control_mutation)
    control_c = _simulate_story(
        label="C_control_mutation",
        seed=int(args.seed_c),
        loaded_canon=control_canon,
        tick_limit=int(args.tick_limit),
        event_limit=int(args.event_limit),
    )
    control_metrics = _analyze_story(control_c.payload, int(args.seed_c), wounds)

    primary_divergence = _first_divergence(baseline_c.events, mutated_c.events)
    control_divergence = _first_divergence(baseline_c.events, control_c.events)

    trace_recorder = TraceRecorder(max_raw_lines=5000)
    _ = _simulate_story(
        label="C_trace",
        seed=int(args.seed_c),
        loaded_canon=canon_after_b,
        tick_limit=int(args.tick_limit),
        event_limit=int(args.event_limit),
        trace_recorder=trace_recorder,
    )
    trace_summary = _save_trace_summary(trace_output_path, trace_recorder)

    baseline_present = _present_wounds_with_frequency(wounds, baseline_metrics.wound_present_set)
    mutated_present = _present_wounds_with_frequency(wounds, mutated_metrics.wound_present_set)
    control_present = _present_wounds_with_frequency(wounds, control_metrics.wound_present_set)

    wounds_appeared = _set_delta_rows(mutated_metrics.wound_present_set, baseline_metrics.wound_present_set, wound_freq)
    wounds_disappeared = _set_delta_rows(baseline_metrics.wound_present_set, mutated_metrics.wound_present_set, wound_freq)
    control_appeared = _set_delta_rows(control_metrics.wound_present_set, baseline_metrics.wound_present_set, wound_freq)
    control_disappeared = _set_delta_rows(baseline_metrics.wound_present_set, control_metrics.wound_present_set, wound_freq)

    per_agent_rows: list[dict[str, Any]] = []
    for agent in DINNER_PARTY_AGENTS:
        base_val = baseline_metrics.per_agent_scores.get(agent)
        mut_val = mutated_metrics.per_agent_scores.get(agent)
        delta = 0.0
        if base_val is not None and mut_val is not None:
            delta = float(mut_val - base_val)
        per_agent_rows.append(
            {
                "agent": agent,
                "baseline": base_val,
                "mutated": mut_val,
                "delta": float(delta),
            }
        )

    primary_score_delta = float(mutated_metrics.mean_score - baseline_metrics.mean_score)
    control_score_delta = float(control_metrics.mean_score - baseline_metrics.mean_score)

    sensitive = (
        primary_divergence is not None
        or abs(primary_score_delta) > EPSILON
        or bool(wounds_appeared)
        or bool(wounds_disappeared)
        or baseline_metrics.valid_count != mutated_metrics.valid_count
        or any(abs(row["delta"]) > EPSILON for row in _overlap_largest_shifts(baseline_metrics.overlap_matrix, mutated_metrics.overlap_matrix, limit=10))
    )

    primary_effect_size = abs(primary_score_delta) + 0.01 * (len(wounds_appeared) + len(wounds_disappeared))
    control_effect_size = abs(control_score_delta) + 0.01 * (len(control_appeared) + len(control_disappeared))
    specific_to_central = primary_effect_size > control_effect_size + EPSILON

    rescues_score_degradation = primary_score_delta > EPSILON

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "seed_a": int(args.seed_a),
            "seed_b": int(args.seed_b),
            "seed_c": int(args.seed_c),
            "event_limit": int(args.event_limit),
            "tick_limit": int(args.tick_limit),
            "wound_baseline_path": str(WOUND_ANALYSIS_PATH),
            "llm_calls": False,
        },
        "artifacts": {
            "canon_after_b_path": str(canon_b_path),
            "canon_after_b_mutated_path": str(canon_b_mutated_path),
            "trace_path": str(trace_output_path),
            "report_path": str(output_path),
        },
        "canon_state_before_mutation": {
            "claim_counts_per_agent": claim_counts,
            "thorne_beliefs": thorne_beliefs,
            "key_asymmetries": [
                {
                    "claim_id": claim_id,
                    "counts": row["counts"],
                }
                for claim_id, row in sorted(asymmetry_map.items())
            ],
        },
        "mutations": {
            "primary": {
                "claim_id": primary_mutation.claim_id,
                "agent_id": primary_mutation.agent_id,
                "old_value": primary_mutation.old_value,
                "new_value": primary_mutation.new_value,
                "reason": primary_mutation.reason,
            },
            "control": {
                "claim_id": control_mutation.claim_id,
                "agent_id": control_mutation.agent_id,
                "old_value": control_mutation.old_value,
                "new_value": control_mutation.new_value,
                "reason": control_mutation.reason,
            },
        },
        "metrics": {
            "score_rebound": {
                "baseline_mean": float(baseline_metrics.mean_score),
                "mutated_mean": float(mutated_metrics.mean_score),
                "delta": float(primary_score_delta),
                "per_agent": per_agent_rows,
            },
            "wound_topology": {
                "baseline_present": baseline_present,
                "mutated_present": mutated_present,
                "appeared": wounds_appeared,
                "disappeared": wounds_disappeared,
                "control_present": control_present,
                "control_appeared": control_appeared,
                "control_disappeared": control_disappeared,
            },
            "overlap_matrix_change": {
                "baseline_top5": _overlap_top(baseline_metrics.overlap_matrix),
                "mutated_top5": _overlap_top(mutated_metrics.overlap_matrix),
                "largest_shifts": _overlap_largest_shifts(baseline_metrics.overlap_matrix, mutated_metrics.overlap_matrix),
            },
            "arc_validity": {
                "baseline_valid": int(baseline_metrics.valid_count),
                "mutated_valid": int(mutated_metrics.valid_count),
                "control_valid": int(control_metrics.valid_count),
            },
            "event_divergence": {
                "baseline_vs_mutated": {
                    "identical": primary_divergence is None,
                    "first_divergence_index": None if primary_divergence is None else int(primary_divergence.index),
                    "first_divergence_tick": None if primary_divergence is None else primary_divergence.tick_id,
                    "baseline_event_id": None if primary_divergence is None else primary_divergence.baseline_event_id,
                    "candidate_event_id": None if primary_divergence is None else primary_divergence.candidate_event_id,
                    "baseline_event_count": int(len(baseline_c.events)),
                    "candidate_event_count": int(len(mutated_c.events)),
                },
                "baseline_vs_control": {
                    "identical": control_divergence is None,
                    "first_divergence_index": None if control_divergence is None else int(control_divergence.index),
                    "first_divergence_tick": None if control_divergence is None else control_divergence.tick_id,
                    "baseline_event_id": None if control_divergence is None else control_divergence.baseline_event_id,
                    "candidate_event_id": None if control_divergence is None else control_divergence.candidate_event_id,
                    "baseline_event_count": int(len(baseline_c.events)),
                    "candidate_event_count": int(len(control_c.events)),
                },
            },
            "control": {
                "baseline_mean": float(baseline_metrics.mean_score),
                "control_mean": float(control_metrics.mean_score),
                "delta": float(control_score_delta),
                "wounds_appeared": control_appeared,
                "wounds_disappeared": control_disappeared,
            },
        },
        "determinism_check": {
            "baseline_repeat_identical": bool(determinism_ok),
            "first_divergence_index": None if determinism_divergence is None else int(determinism_divergence.index),
            "first_divergence_tick": None if determinism_divergence is None else determinism_divergence.tick_id,
        },
        "trace_summary": trace_summary,
        "conclusion": {
            "sensitive": bool(sensitive),
            "specific_to_central": bool(specific_to_central),
            "rescues_score_degradation": bool(rescues_score_degradation),
            "primary_score_delta": float(primary_score_delta),
            "control_score_delta": float(control_score_delta),
            "notes": [
                "Wound topology is treated as binary present/absent per run; population frequencies are contextual labels.",
                "First-divergence tick is the primary causal diagnostic for simulation-level sensitivity.",
            ],
        },
    }

    _save_json(output_path, report)
    _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
