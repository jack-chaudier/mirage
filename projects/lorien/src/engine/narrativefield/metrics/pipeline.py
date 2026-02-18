from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from narrativefield.integration.bundler import BundleInputs, bundle_for_renderer
from narrativefield.integration.event_bundler import BundleResult, BundlerConfig, bundle_events
from narrativefield.metrics import irony, segmentation, tension, thematic
from narrativefield.metrics.significance import populate_significance
from narrativefield.schema.agents import AgentState
from narrativefield.schema.canon import WorldCanon
from narrativefield.schema.events import Event
from narrativefield.schema.index_tables import build_index_tables
from narrativefield.schema.scenes import Scene
from narrativefield.schema.world import ClaimDefinition, Location, SecretDefinition


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParsedSimulationOutput:
    format_version: str
    metadata: dict[str, Any]
    initial_state: dict[str, Any]
    snapshots: list[dict[str, Any]]
    events: list[Event]
    secrets: dict[str, SecretDefinition]
    claims: dict[str, ClaimDefinition]
    locations: dict[str, Location]
    initial_agents: dict[str, AgentState]
    world_canon: WorldCanon | None = None


def parse_simulation_output(raw: dict[str, Any]) -> ParsedSimulationOutput:
    fmt = str(raw.get("format_version", ""))
    metadata = raw.get("metadata") or {}
    initial_state = raw.get("initial_state") or {}
    snapshots = list(raw.get("snapshots") or [])

    secrets_list = raw.get("secrets") or []
    claims_list = raw.get("claims") or []
    locations_list = raw.get("locations") or []
    events_list = raw.get("events") or []
    world_canon_raw = raw.get("world_canon")

    secrets = {str(s["id"]): SecretDefinition.from_dict(s) for s in secrets_list}
    claims = {str(c["id"]): ClaimDefinition.from_dict(c) for c in claims_list}
    locations = {str(loc_raw["id"]): Location.from_dict(loc_raw) for loc_raw in locations_list}

    events = [Event.from_dict(e) for e in events_list]

    initial_agents_raw = (initial_state.get("agents") or {}) if isinstance(initial_state, dict) else {}
    initial_agents = {str(k): AgentState.from_dict(v) for k, v in initial_agents_raw.items()}
    world_canon = WorldCanon.from_dict(world_canon_raw) if isinstance(world_canon_raw, dict) else None

    return ParsedSimulationOutput(
        format_version=fmt,
        metadata=dict(metadata),
        initial_state=dict(initial_state),
        snapshots=snapshots,
        events=events,
        secrets=secrets,
        claims=claims,
        locations=locations,
        initial_agents=initial_agents,
        world_canon=world_canon,
    )


def validate_simulation_output(parsed: ParsedSimulationOutput) -> list[str]:
    """
    Minimal validation gate before metrics.

    Source: specs/integration/data-flow.md Section 8.1.
    """
    errors: list[str] = []
    if not parsed.events:
        errors.append("Event log is empty")
        return errors

    for i in range(1, len(parsed.events)):
        prev = parsed.events[i - 1]
        curr = parsed.events[i]
        if (curr.tick_id, curr.order_in_tick) <= (prev.tick_id, prev.order_in_tick):
            errors.append(f"Event ordering violation at {curr.id}")
            break

    return errors


@dataclass(frozen=True)
class MetricsPipelineOutput:
    events: list[Event]
    scenes: list[Scene]
    belief_snapshots: list[dict[str, Any]]
    bundle_result: BundleResult | None = None


def compute_significance(events: list[Event]) -> None:
    """
    Backwards-compatible wrapper for batch significance population.
    """
    populate_significance(events)


def _sanitize_events_for_pipeline(events: list[Event]) -> list[Event]:
    if not events:
        return events

    deduped: list[Event] = []
    seen_ids: set[str] = set()
    duplicate_ids: set[str] = set()

    for event in events:
        if event.id in seen_ids:
            duplicate_ids.add(event.id)
            continue
        seen_ids.add(event.id)
        deduped.append(event)

    if duplicate_ids:
        logger.warning(
            "Metrics pipeline dropped duplicate event ids: %s",
            sorted(duplicate_ids),
        )

    dangling_refs: dict[str, list[str]] = {}
    valid_ids = {e.id for e in deduped}
    for event in deduped:
        missing = [parent for parent in event.causal_links if parent not in valid_ids]
        if missing:
            dangling_refs[event.id] = missing

    if dangling_refs:
        preview = {eid: refs[:3] for eid, refs in list(dangling_refs.items())[:5]}
        logger.warning(
            "Metrics pipeline detected dangling causal links on %d event(s): %s",
            len(dangling_refs),
            preview,
        )

    return deduped


def run_metrics_pipeline(
    parsed: ParsedSimulationOutput,
    *,
    bundler_config: BundlerConfig | None = None,
) -> MetricsPipelineOutput:
    events = _sanitize_events_for_pipeline(parsed.events)

    parsed_for_validation = ParsedSimulationOutput(
        format_version=parsed.format_version,
        metadata=parsed.metadata,
        initial_state=parsed.initial_state,
        snapshots=parsed.snapshots,
        events=events,
        secrets=parsed.secrets,
        claims=parsed.claims,
        locations=parsed.locations,
        initial_agents=parsed.initial_agents,
        world_canon=parsed.world_canon,
    )
    errs = validate_simulation_output(parsed_for_validation)
    if errs:
        raise ValueError("Invalid simulation output:\\n" + "\\n".join(f"- {e}" for e in errs))

    # Step 1: irony
    all_claims = {secret_id: secret.to_claim() for secret_id, secret in parsed.secrets.items()}
    all_claims.update(parsed.claims)
    irony.run_irony_pipeline(events, parsed.initial_agents, parsed.secrets, claims=all_claims)

    # Step 2: thematic
    thematic.run_thematic_pipeline(events)

    # Step 3: tension
    index_tables = build_index_tables(events)
    # Tension pipeline runs against evolving state; use a clone of initial agent state.
    agents_state = {aid: AgentState.from_dict(a.to_dict()) for aid, a in parsed.initial_agents.items()}
    max_sim_time = float(parsed.metadata.get("total_sim_time") or 0.0) or max(
        (e.sim_time for e in parsed.events), default=0.0
    )
    tension.run_tension_pipeline(
        events,
        agents=agents_state,
        secrets=parsed.secrets,
        claims=all_claims,
        weights=None,
        index_tables=index_tables,
        max_sim_time=max_sim_time,
    )

    # Step 4: significance (used by bundling/filtering and extraction ranking).
    compute_significance(events)

    # Step 5: bundle (compress micro-events for viz/extraction).
    cfg = bundler_config if bundler_config is not None else BundlerConfig()
    br = bundle_events(events, config=cfg)
    bundled_events = br.events

    # Step 6: segmentation (on bundled events so scenes are fewer/cleaner).
    scenes = segmentation.segment_into_scenes(bundled_events, locations=parsed.locations)

    # Belief snapshots: include initial_state + periodic snapshots.
    snapshots = [parsed.initial_state] + parsed.snapshots
    return MetricsPipelineOutput(
        events=bundled_events,
        scenes=scenes,
        belief_snapshots=snapshots,
        bundle_result=br,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="narrativefield.metrics.pipeline")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    raw = json.loads(Path(args.input).read_text(encoding="utf-8"))
    parsed = parse_simulation_output(raw)

    out = run_metrics_pipeline(parsed)

    payload = bundle_for_renderer(
        BundleInputs(
            metadata=parsed.metadata,
            initial_agents=parsed.initial_agents,
            snapshots=out.belief_snapshots,
            events=out.events,
            scenes=out.scenes,
            secrets=parsed.secrets,
            locations=parsed.locations,
            world_canon=parsed.world_canon.to_dict() if parsed.world_canon is not None else None,
        )
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
