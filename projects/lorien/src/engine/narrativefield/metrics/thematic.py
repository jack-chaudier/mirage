from __future__ import annotations

from narrativefield.schema.events import DeltaKind, Event, EventType
from narrativefield.schema.thematic_axes import THEMATIC_AXES_SET


THEMATIC_SHIFT_RULES: dict[EventType, dict[str, float]] = {
    EventType.CONFLICT: {"order_chaos": -0.15},
    EventType.REVEAL: {"truth_deception": 0.2},
    EventType.LIE: {"truth_deception": -0.2},
    EventType.CONFIDE: {"truth_deception": 0.1, "loyalty_betrayal": 0.1},
    EventType.CATASTROPHE: {"order_chaos": -0.3, "innocence_corruption": -0.15},
}


def compute_thematic_shift(event: Event) -> dict[str, float]:
    """
    Compute thematic shift for a single event.

    Source: specs/integration/data-flow.md Section 7.
    """
    shifts: dict[str, float] = {}

    # Base shift from event type.
    for axis, delta in THEMATIC_SHIFT_RULES.get(event.type, {}).items():
        shifts[axis] = shifts.get(axis, 0.0) + float(delta)

    # Delta-driven shifts.
    for d in event.deltas:
        if d.kind == DeltaKind.RELATIONSHIP and d.attribute == "trust":
            if isinstance(d.value, (int, float)):
                if float(d.value) < -0.2:
                    shifts["loyalty_betrayal"] = shifts.get("loyalty_betrayal", 0.0) - 0.1
                elif float(d.value) > 0.2:
                    shifts["loyalty_betrayal"] = shifts.get("loyalty_betrayal", 0.0) + 0.05

        if d.kind == DeltaKind.COMMITMENT:
            shifts["freedom_control"] = shifts.get("freedom_control", 0.0) - 0.1

    # Filter out near-zero shifts, keep canonical axes only, and round for JSON stability.
    return {
        axis: round(delta, 3)
        for axis, delta in shifts.items()
        if axis in THEMATIC_AXES_SET and abs(delta) > 0.01
    }


def run_thematic_pipeline(events: list[Event]) -> None:
    for e in events:
        e.metrics.thematic_shift = compute_thematic_shift(e)
