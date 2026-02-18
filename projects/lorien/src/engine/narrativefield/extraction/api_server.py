from __future__ import annotations

import json
import logging
import os
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import AliasChoices, BaseModel, Field

from narrativefield.extraction.arc_scorer import score_arc
from narrativefield.extraction.arc_search import search_arc, search_region_for_arcs
from narrativefield.extraction.arc_validator import validate_arc
from narrativefield.extraction.beat_classifier import classify_beats
from narrativefield.extraction.beat_sheet import build_beat_sheet
from narrativefield.extraction.prose_generator import generate_prose
from narrativefield.schema.agents import AgentState
from narrativefield.schema.canon import WorldCanon
from narrativefield.schema.events import Event
from narrativefield.schema.scenes import Scene
from narrativefield.schema.world import Location, SecretDefinition, WorldDefinition

logger = logging.getLogger(__name__)


def _default_payload_path() -> Path:
    # Default to the repo fake payload so "region select -> extract" works out of the box.
    # /src/engine/narrativefield/extraction/api_server.py -> repo root is parents[4]
    repo_root = Path(__file__).resolve().parents[4]
    return Path(os.getenv("NARRATIVEFIELD_PAYLOAD_PATH", str(repo_root / "data/fake-dinner-party.nf-viz.json")))


def _cors_origins_from_env() -> list[str]:
    raw = os.getenv("CORS_ORIGINS", "http://localhost:5173")
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    return origins or ["http://localhost:5173"]


class TensionWeightsModel(BaseModel):
    danger: float = Field(0.125, ge=0.0)
    time_pressure: float = Field(0.125, ge=0.0)
    goal_frustration: float = Field(0.125, ge=0.0)
    relationship_volatility: float = Field(0.125, ge=0.0)
    information_gap: float = Field(0.125, ge=0.0)
    resource_scarcity: float = Field(0.125, ge=0.0)
    moral_cost: float = Field(0.125, ge=0.0)
    irony_density: float = Field(0.125, ge=0.0)


class StoryContextModel(BaseModel):
    metadata: dict[str, Any] | None = None
    agents: list[dict[str, Any]] | None = None
    scenes: list[dict[str, Any]] | None = None
    secrets: list[dict[str, Any]] | None = None
    locations: list[dict[str, Any]] | None = None
    world_definition: dict[str, Any] | None = None
    world_canon: dict[str, Any] | None = None


class RegionSelectionModel(BaseModel):
    time_start: float | None = None
    time_end: float | None = None
    agent_ids: list[str] | None = None
    max_events: int | None = Field(default=None, ge=4, le=50)


class StoryExtractionRequestModel(BaseModel):
    selection_type: Literal["region", "arc", "query", "search"] = Field(
        validation_alias=AliasChoices("selection_type", "mode")
    )
    event_ids: list[str] = Field(default_factory=list)
    protagonist_agent_id: str | None = None
    tension_weights: TensionWeightsModel = Field(default_factory=TensionWeightsModel)
    genre_preset: str = "default"
    query_text: str | None = None

    # Search-mode parameters.
    region: RegionSelectionModel | None = None
    time_start: float | None = None
    time_end: float | None = None
    agent_ids: list[str] | None = None
    max_events: int = Field(default=20, ge=4, le=50)

    # Optional context payload to make the API work with arbitrary renderer-loaded datasets.
    selected_events: list[dict[str, Any]] | None = None
    context: StoryContextModel | None = None


class RegionSearchRequestModel(BaseModel):
    time_start: float
    time_end: float
    agent_filter: str | None = None
    max_candidates: int = Field(default=5, ge=1, le=20)


@dataclass
class LoadedPayload:
    metadata: dict[str, Any]
    agents_manifest: dict[str, dict[str, Any]]
    agent_states: dict[str, AgentState]
    events_by_id: dict[str, Event]
    scenes: list[Scene]
    secrets: dict[str, SecretDefinition]
    locations: dict[str, Location]
    world_canon: WorldCanon | None = None


def _build_world_data_from_scenario() -> dict[str, Any]:
    """Build world_data dict from the canonical dinner party scenario.

    MVP: only one scenario exists, so this is always correct.
    """
    from narrativefield.simulation.scenarios.dinner_party import create_dinner_party_world

    world_state = create_dinner_party_world()
    return {
        "world_definition": world_state.definition,
        "agents": list(world_state.agents.values()),
        "secrets": list(world_state.definition.secrets.values()),
    }


def _default_location_for_id(location_id: str) -> Location:
    label = location_id.replace("_", " ").strip() or "unknown location"
    return Location(
        id=location_id,
        name=label.title(),
        privacy=0.2,
        capacity=4,
        adjacent=[],
        overhear_from=[],
        overhear_probability=0.0,
        description=f"{label.title()} in the current story context.",
    )


def _locations_from_raw_or_events(
    raw_locations: list[dict[str, Any]] | None,
    events: list[Event],
) -> dict[str, Location]:
    out: dict[str, Location] = {}
    for raw in raw_locations or []:
        if not isinstance(raw, dict):
            continue
        try:
            loc = Location.from_dict(raw)
        except Exception:
            logger.debug("Skipping invalid location payload entry.", exc_info=True)
            continue
        out[loc.id] = loc

    for loc_id in sorted({e.location_id for e in events if e.location_id}):
        if loc_id not in out:
            out[loc_id] = _default_location_for_id(loc_id)
    return out


def _secrets_from_raw(raw_secrets: list[dict[str, Any]] | None) -> list[SecretDefinition]:
    out: list[SecretDefinition] = []
    for raw in raw_secrets or []:
        if not isinstance(raw, dict):
            continue
        try:
            out.append(SecretDefinition.from_dict(raw))
        except Exception:
            logger.debug("Skipping invalid secret payload entry.", exc_info=True)
    return out


def _agents_from_raw_or_events(
    raw_agents: list[dict[str, Any]] | None,
    events: list[Event],
) -> list[dict[str, Any]]:
    agents: list[dict[str, Any]] = [
        dict(a) for a in (raw_agents or []) if isinstance(a, dict) and a.get("id")
    ]
    if agents:
        return agents

    # Fallback when inline payload omits agents: synthesize lightweight manifests from events.
    first_location: dict[str, str] = {}
    for e in events:
        if e.source_agent and e.source_agent not in first_location:
            first_location[e.source_agent] = e.location_id
        for aid in e.target_agents:
            if aid and aid not in first_location:
                first_location[aid] = e.location_id

    manifests: list[dict[str, Any]] = []
    for aid in sorted(first_location.keys()):
        manifests.append(
            {
                "id": aid,
                "name": aid.replace("_", " ").title(),
                "initial_location": first_location.get(aid, ""),
                "goal_summary": "",
                "primary_flaw": "denial",
            }
        )
    return manifests


def _world_definition_from_parts(
    *,
    metadata: dict[str, Any],
    locations: dict[str, Location],
    secrets: dict[str, SecretDefinition],
    events: list[Event],
    default_world_id: str,
) -> WorldDefinition:
    merged_locations = dict(locations)
    for loc_id in sorted({e.location_id for e in events if e.location_id}):
        if loc_id not in merged_locations:
            merged_locations[loc_id] = _default_location_for_id(loc_id)

    total_sim_time = float(metadata.get("total_sim_time") or metadata.get("sim_duration_minutes") or 0.0)
    if total_sim_time <= 0.0 and events:
        total_sim_time = max(float(e.sim_time) for e in events)
    if total_sim_time <= 0.0:
        total_sim_time = 120.0

    ticks_per_minute = float(metadata.get("ticks_per_minute") or 0.0)
    if ticks_per_minute <= 0.0:
        total_ticks = float(metadata.get("total_ticks") or 0.0)
        if total_ticks > 0 and total_sim_time > 0:
            ticks_per_minute = total_ticks / total_sim_time
    if ticks_per_minute <= 0.0:
        ticks_per_minute = 1.0

    scenario = str(metadata.get("scenario") or "").strip()
    world_name = str(metadata.get("world_name") or scenario or "NarrativeField World")
    world_description = str(
        metadata.get("world_description")
        or metadata.get("description")
        or f"Auto-constructed world definition for {world_name}."
    )

    return WorldDefinition(
        id=str(metadata.get("world_id") or metadata.get("simulation_id") or default_world_id),
        name=world_name,
        description=world_description,
        sim_duration_minutes=total_sim_time,
        ticks_per_minute=ticks_per_minute,
        locations=merged_locations,
        secrets=secrets,
    )


def _build_world_data_from_context(
    context: StoryContextModel | None,
    events: list[Event],
) -> dict[str, Any]:
    ctx = context.model_dump() if context else {}
    metadata = dict(ctx.get("metadata") or {})

    agents = _agents_from_raw_or_events(ctx.get("agents"), events)
    secrets_list = _secrets_from_raw(ctx.get("secrets"))
    secret_map = {s.id: s for s in secrets_list}

    world_definition: WorldDefinition | None = None
    raw_world_definition = ctx.get("world_definition")
    if isinstance(raw_world_definition, dict):
        try:
            world_definition = WorldDefinition.from_dict(raw_world_definition)
        except Exception:
            logger.debug("Invalid inline context world_definition; rebuilding from parts.", exc_info=True)

    if world_definition is None:
        locations = _locations_from_raw_or_events(ctx.get("locations"), events)
        world_definition = _world_definition_from_parts(
            metadata=metadata,
            locations=locations,
            secrets=secret_map,
            events=events,
            default_world_id="inline_world",
        )

    if not secrets_list:
        secrets_list = list(world_definition.secrets.values())

    canon = None
    raw_canon = ctx.get("world_canon")
    if isinstance(raw_canon, dict):
        canon = WorldCanon.from_dict(raw_canon)

    return {
        "world_definition": world_definition,
        "agents": agents,
        "secrets": secrets_list,
        "world_canon": canon,
    }


def _build_world_data_from_loaded_payload(
    payload: LoadedPayload,
    events: list[Event],
) -> dict[str, Any]:
    secrets_list = list(payload.secrets.values())
    world_definition = _world_definition_from_parts(
        metadata=dict(payload.metadata),
        locations=dict(payload.locations),
        secrets={s.id: s for s in secrets_list},
        events=events,
        default_world_id="loaded_payload_world",
    )

    agents: list[AgentState] | list[dict[str, Any]]
    if payload.agent_states:
        agents = list(payload.agent_states.values())
    elif payload.agents_manifest:
        agents = list(payload.agents_manifest.values())
    else:
        agents = _agents_from_raw_or_events([], events)

    return {
        "world_definition": world_definition,
        "agents": agents,
        "secrets": secrets_list,
        "world_canon": payload.world_canon,
    }


def _load_payload(path: Path) -> LoadedPayload:
    raw = json.loads(path.read_text(encoding="utf-8"))
    metadata = raw.get("metadata") or {}
    agents = raw.get("agents") or []
    agents_manifest = {str(a.get("id")): dict(a) for a in agents if isinstance(a, dict) and a.get("id")}
    raw_initial_agents = ((raw.get("initial_state") or {}).get("agents") or {})
    agent_states: dict[str, AgentState] = {}
    for aid, data in raw_initial_agents.items():
        if not isinstance(data, dict):
            continue
        try:
            parsed = AgentState.from_dict(data)
        except Exception:
            logger.debug("Skipping invalid initial_state agent entry for %s", aid, exc_info=True)
            continue
        agent_states[str(aid)] = parsed
    events = [Event.from_dict(e) for e in (raw.get("events") or [])]
    scenes = [Scene.from_dict(s) for s in (raw.get("scenes") or [])]
    secrets = {str(s.get("id")): SecretDefinition.from_dict(s) for s in (raw.get("secrets") or []) if isinstance(s, dict)}
    locations = {
        str(raw_location.get("id")): Location.from_dict(raw_location)
        for raw_location in (raw.get("locations") or [])
        if isinstance(raw_location, dict)
    }
    world_canon = WorldCanon.from_dict(raw.get("world_canon")) if isinstance(raw.get("world_canon"), dict) else None
    return LoadedPayload(
        metadata=dict(metadata),
        agents_manifest=agents_manifest,
        agent_states=agent_states,
        events_by_id={e.id: e for e in events},
        scenes=scenes,
        secrets=secrets,
        locations=locations,
        world_canon=world_canon,
    )


_loaded_payload: LoadedPayload | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _loaded_payload
    path = _default_payload_path()
    if path.exists():
        _loaded_payload = _load_payload(path)
    yield


app = FastAPI(title="NarrativeField Story Extraction API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    # DEV ONLY: restrict origins for production deployment via CORS_ORIGINS.
    allow_origins=_cors_origins_from_env(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _pick_protagonist(events: list[Event]) -> str | None:
    counts: dict[str, int] = {}
    for e in events:
        for a in {e.source_agent, *e.target_agents}:
            counts[a] = counts.get(a, 0) + 1
    if not counts:
        return None
    return max(counts, key=counts.get)


def _events_from_request(req: StoryExtractionRequestModel) -> tuple[list[Event], dict[str, Any], dict[str, dict[str, Any]], list[Scene], dict[str, SecretDefinition]]:
    # Priority 1: selected_events inline (renderer can send only what's needed).
    if req.selected_events:
        events = [Event.from_dict(e) for e in req.selected_events]
        events.sort(key=lambda e: (e.sim_time, e.tick_id, e.order_in_tick))
        ctx = req.context.model_dump() if req.context else {}
        agents = {str(a.get("id")): dict(a) for a in (ctx.get("agents") or []) if isinstance(a, dict) and a.get("id")}
        scenes = [Scene.from_dict(s) for s in (ctx.get("scenes") or [])]
        secrets = {str(s.get("id")): SecretDefinition.from_dict(s) for s in (ctx.get("secrets") or []) if isinstance(s, dict)}
        metadata = dict(ctx.get("metadata") or {})
        return events, metadata, agents, scenes, secrets

    # Priority 2: IDs resolved against loaded payload.
    if _loaded_payload is None:
        raise HTTPException(status_code=400, detail="No payload loaded on server, and request did not include selected_events/context.")

    resolved: list[Event] = []
    missing: list[str] = []
    for eid in req.event_ids:
        e = _loaded_payload.events_by_id.get(eid)
        if e is None:
            missing.append(eid)
        else:
            resolved.append(e)
    if missing:
        raise HTTPException(status_code=400, detail=f"Unknown event_ids: {missing[:10]}")

    resolved.sort(key=lambda e: (e.sim_time, e.tick_id, e.order_in_tick))
    return resolved, _loaded_payload.metadata, _loaded_payload.agents_manifest, _loaded_payload.scenes, _loaded_payload.secrets


def _all_events_from_request(
    req: StoryExtractionRequestModel,
) -> tuple[
    list[Event],
    dict[str, Any],
    dict[str, dict[str, Any]],
    list[Scene],
    dict[str, SecretDefinition],
]:
    """Return ALL events from the payload (for search mode)."""
    if req.selected_events:
        events = [Event.from_dict(e) for e in req.selected_events]
        events.sort(key=lambda e: (e.sim_time, e.tick_id, e.order_in_tick))
        ctx = req.context.model_dump() if req.context else {}
        agents = {
            str(a.get("id")): dict(a)
            for a in (ctx.get("agents") or [])
            if isinstance(a, dict) and a.get("id")
        }
        scenes = [Scene.from_dict(s) for s in (ctx.get("scenes") or [])]
        secrets = {
            str(s.get("id")): SecretDefinition.from_dict(s)
            for s in (ctx.get("secrets") or [])
            if isinstance(s, dict)
        }
        metadata = dict(ctx.get("metadata") or {})
        return events, metadata, agents, scenes, secrets

    if _loaded_payload is None:
        raise HTTPException(
            status_code=400,
            detail="No payload loaded and no selected_events provided.",
        )

    all_events = sorted(
        _loaded_payload.events_by_id.values(),
        key=lambda e: (e.sim_time, e.tick_id, e.order_in_tick),
    )
    return (
        all_events,
        _loaded_payload.metadata,
        _loaded_payload.agents_manifest,
        _loaded_payload.scenes,
        _loaded_payload.secrets,
    )


@app.post("/extract")
def extract(req: StoryExtractionRequestModel) -> dict[str, Any]:
    # Auto-prefilter: if a region selection passes too many micro-events, switch to search mode
    # using the time window implied by the selected events.
    if req.selection_type != "search" and (len(req.event_ids) > 50 or (req.selected_events and len(req.selected_events) > 50)):
        events, _, _, _, _ = _events_from_request(req)
        if events:
            return _extract_search(req, inferred_time_start=events[0].sim_time, inferred_time_end=events[-1].sim_time)

    if req.selection_type == "search":
        return _extract_search(req)

    events, metadata, agents_manifest, scenes, secrets = _events_from_request(req)

    protagonist = req.protagonist_agent_id or _pick_protagonist(events)
    if not protagonist:
        raise HTTPException(
            status_code=400,
            detail="Could not determine protagonist (empty selection).",
        )

    beats = classify_beats(events)
    validation = validate_arc(
        events=events,
        beats=beats,
        total_sim_time=float(metadata.get("total_sim_time") or 0.0) or None,
    )

    response: dict[str, Any] = {"validation": validation.to_dict()}
    response["beats"] = [
        {"event_id": e.id, "beat_type": b.value} for e, b in zip(events, beats)
    ]

    if not validation.valid:
        response["suggestions"] = [
            "Try selecting a longer span that includes a turning point and aftermath."
        ]
        return response

    arc_score = score_arc(events, beats)
    response["score"] = arc_score.to_dict()

    beat_sheet = build_beat_sheet(
        events=events,
        beats=beats,
        protagonist=protagonist,
        genre_preset=req.genre_preset,
        arc_score=arc_score,
        agents_manifest=agents_manifest,
        secrets=secrets,
        scenes=scenes,
        setting_summary=None,
    )
    response["beat_sheet"] = beat_sheet.to_dict()

    prose = generate_prose(beat_sheet=beat_sheet)
    response["prose"] = prose.prose
    response["llm"] = prose.to_dict()

    return response


def _extract_search(
    req: StoryExtractionRequestModel,
    *,
    inferred_time_start: float | None = None,
    inferred_time_end: float | None = None,
) -> dict[str, Any]:
    all_events, metadata, agents_manifest, scenes, secrets = (
        _all_events_from_request(req)
    )

    # Determine effective search window.
    time_start = inferred_time_start
    time_end = inferred_time_end
    agent_ids = req.agent_ids
    max_events = req.max_events

    if req.region:
        if req.region.time_start is not None:
            time_start = req.region.time_start
        if req.region.time_end is not None:
            time_end = req.region.time_end
        if req.region.agent_ids is not None:
            agent_ids = req.region.agent_ids
        if req.region.max_events is not None:
            max_events = int(req.region.max_events)

    if time_start is None:
        time_start = req.time_start
    if time_end is None:
        time_end = req.time_end
    if agent_ids is None:
        agent_ids = req.agent_ids

    # If the client asked for search but only supplied event_ids, infer the window from those IDs.
    if (time_start is None or time_end is None) and req.event_ids:
        selected, _, _, _, _ = _events_from_request(req)
        if selected:
            time_start = selected[0].sim_time if time_start is None else time_start
            time_end = selected[-1].sim_time if time_end is None else time_end

    search_result = search_arc(
        all_events=all_events,
        time_start=time_start,
        time_end=time_end,
        agent_ids=agent_ids,
        protagonist=req.protagonist_agent_id,
        max_events=max_events,
        total_sim_time=float(metadata.get("total_sim_time") or 0.0) or None,
    )

    events = search_result.events
    protagonist = search_result.protagonist
    beats = search_result.beats
    validation = search_result.validation

    if not events or not protagonist:
        response: dict[str, Any] = {
            "validation": validation.to_dict(),
        }
        if search_result.diagnostics:
            response["diagnostics"] = search_result.diagnostics.to_dict()
        return response

    response = {"validation": validation.to_dict()}
    response["beats"] = [
        {"event_id": e.id, "beat_type": b.value} for e, b in zip(events, beats)
    ]
    if search_result.diagnostics:
        response["diagnostics"] = search_result.diagnostics.to_dict()

    if not validation.valid:
        response["suggestions"] = [
            "Search found events but the arc is not valid. "
            "Try a wider time window or specifying a protagonist."
        ]
        return response

    arc_score = search_result.arc_score or score_arc(events, beats)
    response["score"] = arc_score.to_dict()

    beat_sheet = build_beat_sheet(
        events=events,
        beats=beats,
        protagonist=protagonist,
        genre_preset=req.genre_preset,
        arc_score=arc_score,
        agents_manifest=agents_manifest,
        secrets=secrets,
        scenes=scenes,
        setting_summary=None,
    )
    response["beat_sheet"] = beat_sheet.to_dict()

    prose = generate_prose(beat_sheet=beat_sheet)
    response["prose"] = prose.prose
    response["llm"] = prose.to_dict()

    return response


@app.post("/api/search-region")
def search_region(req: RegionSearchRequestModel) -> dict[str, Any]:
    """Search a time region for valid arcs (region = search space, not answer)."""
    import time

    if _loaded_payload is None:
        raise HTTPException(
            status_code=400,
            detail="No payload loaded on server.",
        )

    all_events = sorted(
        _loaded_payload.events_by_id.values(),
        key=lambda e: (e.sim_time, e.tick_id, e.order_in_tick),
    )
    total_sim_time = float(
        _loaded_payload.metadata.get("total_sim_time", 0.0)
    ) or None

    events_in_region = [
        e for e in all_events
        if req.time_start <= e.sim_time <= req.time_end
    ]

    t0 = time.monotonic()
    candidates = search_region_for_arcs(
        events=all_events,
        time_start=req.time_start,
        time_end=req.time_end,
        agent_filter=req.agent_filter,
        max_candidates=req.max_candidates,
        total_sim_time=total_sim_time,
    )
    elapsed_ms = (time.monotonic() - t0) * 1000

    return {
        "candidates": [c.to_dict() for c in candidates],
        "search_metadata": {
            "region_size": len(events_in_region),
            "candidates_evaluated": len(candidates),
            "valid_candidates": sum(1 for c in candidates if c.validation.valid),
            "search_time_ms": round(elapsed_ms, 2),
        },
    }


# ---------------------------------------------------------------------------
# Prose generation endpoint (sequential narrator pipeline)
# ---------------------------------------------------------------------------


class ProseGenerationRequestModel(BaseModel):
    event_ids: list[str] = Field(default_factory=list)
    protagonist_agent_id: str | None = None
    resume_run_id: str | None = None
    config_overrides: dict[str, Any] | None = None
    # Optional inline data (same pattern as StoryExtractionRequestModel).
    selected_events: list[dict[str, Any]] | None = None
    context: StoryContextModel | None = None


_ConfigOverrideSpec = tuple[type[Any], tuple[int, int] | None]

_ALLOWED_CONFIG_OVERRIDES: dict[str, _ConfigOverrideSpec] = {
    "phase2_max_words_per_chunk": (int, (500, 4000)),
    "phase2_creative_max_tokens": (int, (512, 8192)),
    "phase2_creative_deep_max_tokens": (int, (512, 8192)),
    "phase2_use_extended_thinking_for_pivotal": (bool, None),
    "max_summary_words": (int, (100, 2000)),
}

_MAX_PROSE_RESULTS = 50
_prose_results: OrderedDict[str, dict[str, Any]] = OrderedDict()


def _validate_config_override(key: str, value: Any) -> tuple[bool, str | None]:
    spec = _ALLOWED_CONFIG_OVERRIDES.get(key)
    if spec is None:
        return False, "not allowlisted"

    expected_type, bounds = spec
    if expected_type is bool:
        if not isinstance(value, bool):
            return False, "expected bool"
    elif expected_type is int:
        if isinstance(value, bool) or not isinstance(value, int):
            return False, "expected int"
    elif not isinstance(value, expected_type):
        return False, f"expected {expected_type.__name__}"

    if bounds is not None and isinstance(value, (int, float)):
        min_allowed, max_allowed = bounds
        if not (min_allowed <= value <= max_allowed):
            return False, f"out of range [{min_allowed}, {max_allowed}]"

    return True, None


@app.post("/api/generate-prose")
async def generate_prose_endpoint(req: ProseGenerationRequestModel) -> dict[str, Any]:
    """Run the sequential narrator pipeline to generate a short story from arc events.

    This is a synchronous endpoint — it blocks until generation completes (typically 2-4 min
    with real LLM calls, instant with mocked calls in tests).
    """
    from narrativefield.llm.config import PipelineConfig
    from narrativefield.storyteller.narrator import SequentialNarrator

    # --- Resolve events ---
    events: list[Event]
    world_data: dict[str, Any] | None = None

    if req.selected_events:
        events = [Event.from_dict(e) for e in req.selected_events]
        events.sort(key=lambda e: (e.sim_time, e.tick_id, e.order_in_tick))
        world_data = _build_world_data_from_context(req.context, events)
    elif _loaded_payload is not None:
        resolved: list[Event] = []
        missing: list[str] = []
        for eid in req.event_ids:
            e = _loaded_payload.events_by_id.get(eid)
            if e is None:
                missing.append(eid)
            else:
                resolved.append(e)
        if missing:
            raise HTTPException(status_code=400, detail=f"Unknown event_ids: {missing[:10]}")
        events = sorted(resolved, key=lambda e: (e.sim_time, e.tick_id, e.order_in_tick))
        world_data = _build_world_data_from_loaded_payload(_loaded_payload, events)
    else:
        raise HTTPException(
            status_code=400,
            detail="No payload loaded and no selected_events provided.",
        )

    if not events:
        raise HTTPException(status_code=400, detail="No events to generate prose from.")

    # --- Build config ---
    config = PipelineConfig()
    if req.config_overrides:
        for key, val in req.config_overrides.items():
            if key not in _ALLOWED_CONFIG_OVERRIDES:
                logger.warning("Rejected config override: %s (not allowlisted)", key)
                continue
            if not hasattr(config, key):
                logger.warning("Rejected config override: %s (unknown config field)", key)
                continue
            is_valid, reason = _validate_config_override(key, val)
            if is_valid:
                object.__setattr__(config, key, val)
            else:
                logger.warning("Rejected config override: %s (%s)", key, reason)

    # --- Optionally set protagonist via beat classification + arc search ---
    protagonist = req.protagonist_agent_id or _pick_protagonist(events)

    # --- Build beat sheet if events have metrics ---
    beat_sheet = None
    try:
        beats = classify_beats(events)
        for ev, bt in zip(events, beats):
            ev.beat_type = bt
        validation = validate_arc(events=events, beats=beats)
        if validation.valid:
            arc_score = score_arc(events, beats)
            agents_manifest = {}
            secrets: dict[str, SecretDefinition] = {}
            if _loaded_payload:
                agents_manifest = _loaded_payload.agents_manifest
                secrets = _loaded_payload.secrets
            beat_sheet = build_beat_sheet(
                events=events,
                beats=beats,
                protagonist=protagonist or "",
                genre_preset="default",
                arc_score=arc_score,
                agents_manifest=agents_manifest,
                secrets=secrets,
                scenes=[],
            )
    except Exception:
        logger.debug("Beat sheet construction failed; narrator will handle classification.", exc_info=True)

    # --- Run narrator ---
    run_id = req.resume_run_id or f"prose_{uuid.uuid4().hex[:12]}"
    narrator = SequentialNarrator(config=config)
    narrator_canon = None
    if isinstance(world_data, dict):
        raw_canon = world_data.get("world_canon")
        if isinstance(raw_canon, WorldCanon):
            narrator_canon = raw_canon
        elif isinstance(raw_canon, dict):
            narrator_canon = WorldCanon.from_dict(raw_canon)

    try:
        result = await narrator.generate(
            events=events,
            beat_sheet=beat_sheet,
            world_data=world_data,
            run_id=run_id,
            resume=bool(req.resume_run_id),
            canon=narrator_canon,
        )
    except Exception as exc:
        logger.exception("Prose generation failed for run_id=%s", run_id)
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc

    response = {
        "prose": result.prose,
        "word_count": result.word_count,
        "scenes_generated": result.scenes_generated,
        "continuity_report": result.usage.get("continuity_report", []),
        "usage": {
            "total_input_tokens": result.usage.get("total_input_tokens", 0),
            "total_output_tokens": result.usage.get("total_output_tokens", 0),
            "estimated_cost_usd": result.usage.get("estimated_cost_usd", 0.0),
            "generation_time_seconds": result.generation_time_seconds,
            "llm_response_metadata": result.usage.get("llm_response_metadata", {}),
            "model_history": result.usage.get("model_history", []),
        },
        "run_id": run_id,
    }
    _prose_results[run_id] = response
    while len(_prose_results) > _MAX_PROSE_RESULTS:
        _prose_results.popitem(last=False)
    return response


@app.get("/api/prose-status/{run_id}")
async def prose_status(run_id: str) -> dict[str, Any]:
    """Check the status/result of a prose generation run."""
    if run_id in _prose_results:
        return {"status": "complete", "result": _prose_results[run_id]}
    return {"status": "not_found", "run_id": run_id}
