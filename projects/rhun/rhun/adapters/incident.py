"""Stub adapter for incident-style causal graph sources."""

from __future__ import annotations

from pathlib import Path

from rhun.schemas import CausalGraph


def load_incident_graph(path: Path) -> CausalGraph:
    """Future adapter entry point for incident response graph imports."""
    raise NotImplementedError(f"Incident adapter not implemented for {path}")
