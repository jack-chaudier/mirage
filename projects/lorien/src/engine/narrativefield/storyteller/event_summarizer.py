from __future__ import annotations

from collections import Counter

from narrativefield.schema.events import DeltaKind, Event, EventType


def summarize_event(event: Event) -> str:
    """Create a one-line narrative summary of an event.

    Local/deterministic: no LLM calls.

    Keeps the result under ~30 words by truncation (best-effort).
    """

    src = _display_agent(event.source_agent)
    targets = [_display_agent(t) for t in event.target_agents]
    loc = _display_location(event.location_id)
    tension = float(getattr(getattr(event, "metrics", None), "tension", 0.0) or 0.0)
    secret_id = (event.content_metadata or {}).get("secret_id")

    line = ""
    match event.type:
        case EventType.CHAT:
            if not targets:
                line = f"{src} chats at the {loc}"
            elif len(targets) == 1:
                line = f"{src} and {targets[0]} chat at the {loc}"
            else:
                line = f"{src} chats with the group at the {loc}"

        case EventType.REVEAL:
            secret = f"'{secret_id}'" if isinstance(secret_id, str) and secret_id else "a secret"
            if targets:
                line = f"{src} reveals {secret} to {targets[0]} at the {loc} (tension: {tension:.2f})"
            else:
                line = f"{src} reveals {secret} at the {loc} (tension: {tension:.2f})"

        case EventType.CONFIDE:
            secret = f"'{secret_id}'" if isinstance(secret_id, str) and secret_id else "a secret"
            if targets:
                line = f"{src} confides in {targets[0]} about {secret} at the {loc}"
            else:
                line = f"{src} confides about {secret} at the {loc}"

        case EventType.LIE:
            secret = f"'{secret_id}'" if isinstance(secret_id, str) and secret_id else "a secret"
            if targets:
                line = f"{src} lies to {targets[0]} about {secret} at the {loc}"
            else:
                line = f"{src} lies about {secret} at the {loc}"

        case EventType.CONFLICT:
            if targets:
                line = f"{src} confronts {targets[0]} at the {loc} (tension: {tension:.2f})"
            else:
                line = f"{src} sparks conflict at the {loc} (tension: {tension:.2f})"

        case EventType.CATASTROPHE:
            stress = _stress_from_pacing_delta(event)
            if stress is not None:
                line = f"{src} has a breakdown at the {loc} (stress: {stress:.2f})"
            else:
                line = f"{src} has a breakdown at the {loc} (tension: {tension:.2f})"

        case EventType.SOCIAL_MOVE:
            dest = (event.content_metadata or {}).get("destination")
            if isinstance(dest, str) and dest:
                line = f"{src} moves from {_display_location(event.location_id)} to {_display_location(dest)}"
            else:
                line = f"{src} changes location"

        case EventType.OBSERVE:
            # OBSERVE events often encode details in the description.
            if event.description:
                line = f"{src} observes: {event.description}"
            else:
                line = f"{src} observes quietly at the {loc}"

        case EventType.INTERNAL:
            if event.description:
                line = f"{src} reflects internally ({event.description})"
            else:
                line = f"{src} reflects internally"

        case EventType.PHYSICAL:
            if event.description:
                line = f"{src} {event.description.lower()} at the {loc}"
            else:
                line = f"{src} does something physical at the {loc}"

        case _:
            # Fallback: structured but readable.
            if event.description:
                line = f"{src} does {event.type.value}: {event.description}"
            else:
                line = f"{src} does {event.type.value} at the {loc}"

    return _truncate_words(line.strip(), 30)


def summarize_scene(events: list[Event]) -> str:
    """Create a paragraph summary of a scene (3-5 sentences).

    Local/deterministic: template-based MVP summary using event summaries + deltas.
    """

    if not events:
        return ""

    loc = _display_location(_mode_location(events))
    who = sorted({p for e in events for p in ([e.source_agent] + list(e.target_agents))})
    who_disp = ", ".join(_display_agent(x) for x in who) if who else "Someone"

    # Tension shape.
    tensions = [float(getattr(getattr(e, "metrics", None), "tension", 0.0) or 0.0) for e in events]
    t0, t1 = tensions[0], tensions[-1]
    t_peak = max(tensions) if tensions else 0.0
    if abs(t1 - t0) < 0.05:
        tension_sentence = f"Tension holds steady around {t_peak:.2f}."
    elif t1 > t0:
        tension_sentence = f"Tension rises from {t0:.2f} to {t1:.2f}, peaking at {t_peak:.2f}."
    else:
        tension_sentence = f"Tension eases from {t0:.2f} to {t1:.2f}, after peaking at {t_peak:.2f}."

    # Knowledge/relationship changes (very light).
    learned_secrets = _belief_deltas(events)
    rel_changes = _relationship_deltas(events)

    key = [summarize_event(e) for e in events if e.type in (EventType.CONFLICT, EventType.CATASTROPHE, EventType.REVEAL)]
    if not key:
        key = [summarize_event(e) for e in events[:3]]
    key = key[:3]

    sentences: list[str] = []
    sentences.append(f"At the {loc}, {who_disp} share a sequence of moments.")
    if key:
        sentences.append("Key beats: " + "; ".join(key) + ".")
    sentences.append(tension_sentence)
    if learned_secrets:
        sentences.append("Knowledge shifts: " + "; ".join(learned_secrets[:2]) + ".")
    if rel_changes:
        sentences.append("Relationships shift: " + "; ".join(rel_changes[:2]) + ".")

    return " ".join(sentences[:5]).strip()


def _display_agent(agent_id: str) -> str:
    s = str(agent_id or "").strip()
    if not s:
        return "Someone"
    # If it's already a multi-word name, keep it.
    if any(ch.isupper() for ch in s) and " " in s:
        return s
    return " ".join(part.capitalize() for part in s.replace("-", "_").split("_") if part) or s


def _display_location(location_id: str) -> str:
    s = str(location_id or "").strip()
    if not s:
        return "somewhere"
    return s.replace("_", " ")


def _truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).rstrip(",;:.") + "..."


def _mode_location(events: list[Event]) -> str:
    counts = Counter(e.location_id for e in events)
    if not counts:
        return ""
    max_count = max(counts.values())
    tied = {loc for loc, c in counts.items() if c == max_count}
    for e in events:
        if e.location_id in tied:
            return e.location_id
    return events[0].location_id


def _stress_from_pacing_delta(event: Event) -> float | None:
    # Some ticks attach pacing deltas to events; use them if present.
    for d in getattr(event, "deltas", []) or []:
        if d.kind != DeltaKind.PACING:
            continue
        if d.agent != event.source_agent:
            continue
        if d.attribute != "stress":
            continue
        try:
            return float(d.value)
        except (TypeError, ValueError):
            return None
    return None


def _belief_deltas(events: list[Event]) -> list[str]:
    out: list[str] = []
    for e in events:
        for d in getattr(e, "deltas", []) or []:
            if d.kind != DeltaKind.BELIEF:
                continue
            if not d.attribute:
                continue
            out.append(f"{_display_agent(d.agent)} learns '{d.attribute}'")
    return out


def _relationship_deltas(events: list[Event]) -> list[str]:
    out: list[str] = []
    for e in events:
        for d in getattr(e, "deltas", []) or []:
            if d.kind != DeltaKind.RELATIONSHIP:
                continue
            if not d.agent_b:
                continue
            if d.attribute != "trust":
                continue
            try:
                val = float(d.value)
            except (TypeError, ValueError):
                continue
            direction = "improves" if val > 0 else "worsens"
            out.append(f"Trust {direction} between {_display_agent(d.agent)} and {_display_agent(d.agent_b)}")
    return out
