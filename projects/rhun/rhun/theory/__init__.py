"""Theory and verification utilities."""

from rhun.theory.theorem import check_precondition, diagnose_absorption, verify_prediction
from rhun.theory.viability import compute_viability, partial_at_tp_assignment, viability_along_sequence

__all__ = [
    "check_precondition",
    "diagnose_absorption",
    "verify_prediction",
    "compute_viability",
    "partial_at_tp_assignment",
    "viability_along_sequence",
]
