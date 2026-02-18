from __future__ import annotations

# Canonical thematic axes shared with visualization payload validation.
THEMATIC_AXES: list[str] = [
    "order_chaos",
    "truth_deception",
    "loyalty_betrayal",
    "innocence_corruption",
    "freedom_control",
]

THEMATIC_AXES_SET: set[str] = set(THEMATIC_AXES)

__all__ = ["THEMATIC_AXES", "THEMATIC_AXES_SET"]
