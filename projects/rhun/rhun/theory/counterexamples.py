"""Boundary condition helpers where theorem preconditions do not hold."""

from __future__ import annotations

from rhun.extraction.grammar import GrammarConfig
from rhun.schemas import CausalGraph
from rhun.theory.theorem import check_precondition


def identify_counterexample(
    graph: CausalGraph,
    focal_actor: str,
    grammar: GrammarConfig,
) -> bool:
    """Return True when theorem precondition does not hold for this instance."""
    precondition = check_precondition(graph, focal_actor, grammar)
    return not bool(precondition["precondition_met"])
