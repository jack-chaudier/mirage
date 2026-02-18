"""Invariance sweeps across density, actor count, and weight distributions."""

from __future__ import annotations


def describe_invariance_axes() -> dict[str, tuple]:
    """Return default sweep axes for future invariance studies."""
    return {
        "causal_density": (0.02, 0.05, 0.08, 0.12),
        "n_actors": (3, 6, 12),
        "weight_distribution": ("uniform", "normal", "exponential"),
    }
