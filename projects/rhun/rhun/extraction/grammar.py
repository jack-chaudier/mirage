"""Parameterized sequential phase grammar for extraction validation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GrammarConfig:
    """
    Sequential phase constraints for extracted sequences.

    The grammar requires that extracted sequences follow a monotonic
    phase progression: SETUP -> DEVELOPMENT -> TURNING_POINT -> RESOLUTION.

    Key parameters:
        min_prefix_elements: Minimum elements before the turning point
            that must be in DEVELOPMENT phase. This is the critical
            regularization parameter (k in the theorem).
        max_phase_regressions: Number of allowed phase-order violations.
        max_turning_points: Maximum turning points allowed.
        min_length: Minimum total elements in the sequence.
        max_length: Maximum total elements.
        min_timespan_fraction: Sequence must span at least this fraction
            of the graph's total duration.
        max_temporal_gap: Maximum allowed timestamp gap between adjacent
            events in the extracted sequence (sorted by timestamp).
            `float("inf")` disables the constraint.
        focal_actor_coverage: Focal actor must appear in at least this
            fraction of sequence events.
    """

    min_prefix_elements: int = 1  # k in the theorem. THE critical parameter.
    max_phase_regressions: int = 0
    max_turning_points: int = 1
    min_length: int = 4
    max_length: int = 20
    min_timespan_fraction: float = 0.15
    max_temporal_gap: float = float("inf")
    focal_actor_coverage: float = 0.60

    # Convenience constructors for common configurations
    @classmethod
    def strict(cls) -> GrammarConfig:
        return cls()

    @classmethod
    def relaxed(cls) -> GrammarConfig:
        return cls(
            min_prefix_elements=0,
            max_phase_regressions=1,
            max_turning_points=2,
        )

    @classmethod
    def vacuous(cls) -> GrammarConfig:
        return cls(
            min_prefix_elements=0,
            max_phase_regressions=999,
            max_turning_points=999,
            min_length=1,
            max_length=999,
            min_timespan_fraction=0.0,
            max_temporal_gap=float("inf"),
            focal_actor_coverage=0.0,
        )

    @classmethod
    def parametric(cls, k: int, **kwargs) -> GrammarConfig:
        """Create grammar with specific prefix requirement k."""
        return cls(min_prefix_elements=k, **kwargs)
