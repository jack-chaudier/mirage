"""Optional adapter stubs for importing external narrative-style DAG data."""

from __future__ import annotations

from pathlib import Path

from rhun.schemas import CausalGraph


def load_graph_from_json(path: Path) -> CausalGraph:
    """
    Placeholder adapter for external project data.

    The core `rhun` package remains dependency-free and domain-agnostic.
    This adapter can be expanded to map external schemas onto `CausalGraph`.
    """
    raise NotImplementedError(f"Narrative adapter not implemented for {path}")
