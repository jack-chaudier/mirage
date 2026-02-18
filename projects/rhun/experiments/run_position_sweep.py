"""CLI entry point for epsilon position sweep experiment."""

from __future__ import annotations

from rhun.experiments.position_sweep import run_position_sweep


if __name__ == "__main__":
    result = run_position_sweep()
    print(result["results"]["table"])
