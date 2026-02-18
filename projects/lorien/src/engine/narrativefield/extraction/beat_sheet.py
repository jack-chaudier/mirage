from __future__ import annotations

from collections import Counter
from typing import Any

from narrativefield.extraction.types import Beat, BeatSheet, CharacterBrief
from narrativefield.schema.events import BeatType, Event
from narrativefield.schema.scenes import Scene
from narrativefield.schema.world import SecretDefinition


def _participants(event: Event) -> list[str]:
    return sorted({event.source_agent, *event.target_agents})


def _dominant_theme(events: list[Event]) -> str:
    totals: dict[str, float] = {}
    for e in events:
        for axis, delta in (e.metrics.thematic_shift or {}).items():
            totals[axis] = totals.get(axis, 0.0) + abs(float(delta))
    if not totals:
        return ""
    return max(totals, key=totals.get)


def _thematic_trajectory(dominant_theme: str) -> str:
    if not dominant_theme:
        return ""
    if dominant_theme == "truth_deception":
        return "deception \u2192 truth (forced)"
    if dominant_theme == "loyalty_betrayal":
        return "loyalty \u2192 fracture"
    if dominant_theme == "order_chaos":
        return "order \u2192 chaos"
    return dominant_theme


def _scene_id_by_event_id(scenes: list[Scene]) -> dict[str, str]:
    m: dict[str, str] = {}
    for s in scenes:
        for eid in s.event_ids:
            m[eid] = s.id
    return m


def _tone_for_beat(beat: BeatType, tension: float) -> str:
    if beat == BeatType.SETUP:
        return "subtle unease" if tension > 0.25 else "calm, social"
    if beat == BeatType.COMPLICATION:
        return "tense, intimate"
    if beat == BeatType.ESCALATION:
        return "heated, accelerating"
    if beat == BeatType.TURNING_POINT:
        return "explosive, irrevocable"
    return "aftershock, reflective"


def _pacing_for_beat(beat: BeatType) -> str:
    if beat == BeatType.SETUP:
        return "slow build"
    if beat == BeatType.COMPLICATION:
        return "tightening"
    if beat == BeatType.ESCALATION:
        return "accelerating"
    if beat == BeatType.TURNING_POINT:
        return "rapid, sharp"
    return "release and settling"


def _format_key_changes(event: Event) -> list[str]:
    out: list[str] = []
    for d in event.deltas:
        if d.kind.value == "relationship" and d.attribute == "trust" and isinstance(d.value, (int, float)):
            if float(d.value) < -0.15:
                out.append(f"Trust shifts down ({d.agent} \u2192 {d.agent_b}: {float(d.value):.2f})")
            elif float(d.value) > 0.15:
                out.append(f"Trust shifts up ({d.agent} \u2192 {d.agent_b}: {float(d.value):.2f})")
        if d.kind.value == "belief":
            out.append(f"Belief updates ({d.agent}: {d.attribute} \u2192 {d.value})")
        if d.kind.value == "agent_location":
            out.append(f"Move ({d.agent} \u2192 {d.value})")
    return out[:6]


def _role_in_arc(agent_id: str, protagonist: str, counts: Counter[str]) -> str:
    if agent_id == protagonist:
        return "protagonist"
    if counts.get(agent_id, 0) >= max(counts.values(), default=0) * 0.7:
        return "antagonist"
    return "catalyst"


def _pick_key_secret(agent_id: str, secrets: dict[str, SecretDefinition]) -> str | None:
    for s in secrets.values():
        if agent_id in s.holder:
            return f"Holds {s.id}"
        if s.about == agent_id:
            return f"About {s.id}"
    return None


def build_beat_sheet(
    *,
    events: list[Event],
    beats: list[BeatType],
    protagonist: str,
    genre_preset: str,
    arc_score: Any,  # ArcScore
    agents_manifest: dict[str, dict[str, Any]] | None = None,
    secrets: dict[str, SecretDefinition] | None = None,
    scenes: list[Scene] | None = None,
    setting_summary: str | None = None,
) -> BeatSheet:
    if secrets is None:
        secrets = {}
    if scenes is None:
        scenes = []
    scene_by_event = _scene_id_by_event_id(scenes)

    arc_id = f"arc_{protagonist}_{events[0].id}_{events[-1].id}" if events else "arc_empty"
    title = f"Arc: {protagonist}"
    setting = (
        setting_summary
        or "An evening dinner party. Six guests, multiple secrets, rising alcohol and falling composure."
    )
    time_span = f"{events[0].sim_time:.1f} - {events[-1].sim_time:.1f} sim minutes" if events else ""

    # Character briefs.
    counts: Counter[str] = Counter()
    for e in events:
        for a in {e.source_agent, *e.target_agents}:
            counts[a] += 1

    characters: list[CharacterBrief] = []
    for agent_id in sorted(counts.keys()):
        info = (agents_manifest or {}).get(agent_id) or {}
        characters.append(
            CharacterBrief(
                agent_id=agent_id,
                name=str(info.get("name") or agent_id),
                role_in_arc=_role_in_arc(agent_id, protagonist, counts),
                key_goal=str(info.get("goal_summary") or ""),
                key_flaw=str(info.get("primary_flaw") or ""),
                key_secret=_pick_key_secret(agent_id, secrets),
                emotional_start="",
                emotional_end="",
            )
        )

    dominant_theme = _dominant_theme(events)
    thematic_trajectory = _thematic_trajectory(dominant_theme)

    beat_rows: list[Beat] = []
    for event, beat in zip(events, beats):
        tone = _tone_for_beat(beat, float(event.metrics.tension))
        beat_rows.append(
            Beat(
                beat_type=beat,
                event_id=event.id,
                event_type=event.type,
                scene_id=scene_by_event.get(event.id),
                location=event.location_id,
                participants=_participants(event),
                description=event.description,
                tension=float(event.metrics.tension),
                irony_note=None,
                key_changes=_format_key_changes(event),
                emotional_states={},
                pov_suggestion=protagonist,
                tone_suggestion=tone,
                pacing_note=_pacing_for_beat(beat),
            )
        )

    return BeatSheet(
        arc_id=arc_id,
        protagonist=protagonist,
        title_suggestion=title,
        genre_preset=genre_preset,
        arc_score=arc_score,
        setting_summary=setting,
        time_span=time_span,
        characters=characters,
        beats=beat_rows,
        dominant_theme=dominant_theme,
        thematic_trajectory=thematic_trajectory,
    )
