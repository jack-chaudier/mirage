"""Compute arc-validity covariance and conditional validity analyses.

Run:
    cd src/engine && ./.venv/bin/python -m scripts.validity_covariance
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

AGENTS = ["thorne", "elena", "marcus", "lydia", "diana", "victor"]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = SCRIPT_DIR / "output" / "k_sweep_experiment.json"
DEFAULT_OUTPUT_PATH = SCRIPT_DIR / "output" / "validity_covariance.json"


def _resolve_path(path_arg: str) -> Path:
    path = Path(path_arg)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _load_runs(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    runs = payload.get("runs")
    if not isinstance(runs, list):
        raise ValueError("Input JSON must contain a top-level 'runs' list.")
    return runs


def _validity_matrix_from_runs(runs: list[dict[str, Any]]) -> np.ndarray:
    rows: list[list[int]] = []
    for run in runs:
        invalid_agents = {str(agent) for agent in (run.get("invalid_agents") or [])}
        row = [0 if agent in invalid_agents else 1 for agent in AGENTS]
        rows.append(row)
    if not rows:
        return np.zeros((0, len(AGENTS)), dtype=np.int64)
    return np.array(rows, dtype=np.int64)


def _subset_runs(
    runs: list[dict[str, Any]],
    *,
    k: int | None = None,
    exclude_evolved: str | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for run in runs:
        run_k = int(run.get("k", -1))
        if k is not None and run_k != k:
            continue
        evolved = {str(agent) for agent in (run.get("evolved_agents") or [])}
        if exclude_evolved is not None and exclude_evolved in evolved:
            continue
        out.append(run)
    return out


def _corrcoef_dict(validity_matrix: np.ndarray) -> dict[str, dict[str, float | None]]:
    n_rows = int(validity_matrix.shape[0])
    if n_rows < 2:
        return {
            row_agent: {col_agent: None for col_agent in AGENTS}
            for row_agent in AGENTS
        }

    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.corrcoef(validity_matrix, rowvar=False)

    out: dict[str, dict[str, float | None]] = {}
    for row_index, row_agent in enumerate(AGENTS):
        out[row_agent] = {}
        for col_index, col_agent in enumerate(AGENTS):
            value = float(corr[row_index, col_index])
            out[row_agent][col_agent] = value if np.isfinite(value) else None
    return out


def _conditional_validity(validity_matrix: np.ndarray) -> dict[str, dict[str, dict[str, float | int | None]]]:
    out: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for i, agent_i in enumerate(AGENTS):
        out[agent_i] = {}
        mask_i_valid = validity_matrix[:, i] == 1
        mask_i_invalid = validity_matrix[:, i] == 0
        n_i_valid = int(mask_i_valid.sum())
        n_i_invalid = int(mask_i_invalid.sum())

        for j, agent_j in enumerate(AGENTS):
            p_j_given_i_valid: float | None
            p_j_given_i_invalid: float | None

            if n_i_valid > 0:
                p_j_given_i_valid = float(validity_matrix[mask_i_valid, j].mean())
            else:
                p_j_given_i_valid = None

            if n_i_invalid > 0:
                p_j_given_i_invalid = float(validity_matrix[mask_i_invalid, j].mean())
            else:
                p_j_given_i_invalid = None

            difference: float | None
            if p_j_given_i_valid is None or p_j_given_i_invalid is None:
                difference = None
            else:
                difference = p_j_given_i_valid - p_j_given_i_invalid

            out[agent_i][agent_j] = {
                "p_j_valid_given_i_valid": p_j_given_i_valid,
                "p_j_valid_given_i_invalid": p_j_given_i_invalid,
                "difference": difference,
                "n_i_valid": n_i_valid,
                "n_i_invalid": n_i_invalid,
            }
    return out


def _quality_by_valid_count(runs: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    grouped_scores: dict[int, list[float]] = defaultdict(list)
    for run in runs:
        valid_count = int(run.get("valid_arc_count", 0))
        mean_score = float(run.get("mean_score", 0.0))
        grouped_scores[valid_count].append(mean_score)

    out: dict[str, dict[str, float | int]] = {}
    for valid_count in sorted(grouped_scores.keys(), reverse=True):
        scores = grouped_scores[valid_count]
        out[str(valid_count)] = {
            "n": int(len(scores)),
            "mean_score": float(sum(scores) / len(scores)),
        }
    return out


def _format_matrix_table(matrix: dict[str, dict[str, float | None]]) -> str:
    cell_width = 9
    header = "agent".ljust(10) + "".join(agent[:8].rjust(cell_width) for agent in AGENTS)
    lines = [header]
    for row_agent in AGENTS:
        row = row_agent.ljust(10)
        for col_agent in AGENTS:
            value = matrix[row_agent][col_agent]
            text = "null" if value is None else f"{value:+.3f}"
            row += text.rjust(cell_width)
        lines.append(row)
    return "\n".join(lines)


def _find_negative_pairs(
    matrix: dict[str, dict[str, float | None]],
    *,
    threshold: float,
) -> list[tuple[str, str, float]]:
    hits: list[tuple[str, str, float]] = []
    for i, left in enumerate(AGENTS):
        for j in range(i + 1, len(AGENTS)):
            right = AGENTS[j]
            value = matrix[left][right]
            if value is not None and value < threshold:
                hits.append((left, right, value))
    hits.sort(key=lambda row: row[2])
    return hits


def _print_quality_table(title: str, quality: dict[str, dict[str, float | int]]) -> None:
    print(title)
    print("valid_count      n   mean_score")
    for valid_count in sorted(quality.keys(), key=lambda key: int(key), reverse=True):
        row = quality[valid_count]
        print(f"{int(valid_count):>10} {int(row['n']):>6} {float(row['mean_score']):>11.4f}")
    print()


def _interpret_zero_sum(
    neg_all: list[tuple[str, str, float]],
    neg_k6: list[tuple[str, str, float]],
    strong_k6: list[tuple[str, str, float]],
) -> str:
    if strong_k6:
        return "Interpretation: validity appears structurally zero-sum in k=6 (strong negative pair(s) present)."
    if neg_k6 or neg_all:
        return "Interpretation: validity shows mild zero-sum tendencies, but no strong structural competition."
    return "Interpretation: validity does not appear meaningfully zero-sum in these runs."


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze covariance of arc validity across agents.")
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT_PATH),
        help="Path to k_sweep_experiment.json (default: scripts/output/k_sweep_experiment.json).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path to output JSON (default: scripts/output/validity_covariance.json).",
    )
    args = parser.parse_args()

    input_path = _resolve_path(args.input)
    output_path = _resolve_path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    runs_all = _load_runs(input_path)
    runs_k0 = _subset_runs(runs_all, k=0)
    runs_k6 = _subset_runs(runs_all, k=6)
    runs_excl_thorne = _subset_runs(runs_all, k=5, exclude_evolved="thorne")
    runs_excl_diana = _subset_runs(runs_all, k=5, exclude_evolved="diana")

    matrix_all = _validity_matrix_from_runs(runs_all)
    matrix_k0 = _validity_matrix_from_runs(runs_k0)
    matrix_k6 = _validity_matrix_from_runs(runs_k6)
    matrix_excl_thorne = _validity_matrix_from_runs(runs_excl_thorne)
    matrix_excl_diana = _validity_matrix_from_runs(runs_excl_diana)

    corr_all = _corrcoef_dict(matrix_all)
    corr_k0 = _corrcoef_dict(matrix_k0)
    corr_k6 = _corrcoef_dict(matrix_k6)
    corr_excl_thorne = _corrcoef_dict(matrix_excl_thorne)
    corr_excl_diana = _corrcoef_dict(matrix_excl_diana)

    conditional_all = _conditional_validity(matrix_all)
    conditional_k6 = _conditional_validity(matrix_k6)

    quality_all = _quality_by_valid_count(runs_all)
    quality_k6 = _quality_by_valid_count(runs_k6)

    sample_sizes = {
        "all": int(len(runs_all)),
        "k0": int(len(runs_k0)),
        "k6": int(len(runs_k6)),
        "excl_thorne": int(len(runs_excl_thorne)),
        "excl_diana": int(len(runs_excl_diana)),
    }

    payload = {
        "correlation_matrix_all": corr_all,
        "correlation_matrix_k0": corr_k0,
        "correlation_matrix_k6": corr_k6,
        "correlation_matrix_excl_thorne": corr_excl_thorne,
        "correlation_matrix_excl_diana": corr_excl_diana,
        "conditional_validity_all": conditional_all,
        "conditional_validity_k6": conditional_k6,
        "quality_by_valid_count_all": quality_all,
        "quality_by_valid_count_k6": quality_k6,
        "sample_sizes": sample_sizes,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    neg_all = _find_negative_pairs(corr_all, threshold=-0.15)
    neg_k6 = _find_negative_pairs(corr_k6, threshold=-0.15)
    strong_k6 = _find_negative_pairs(corr_k6, threshold=-0.20)

    print()
    print("=== VALIDITY CORRELATION MATRIX (ALL RUNS) ===")
    print(_format_matrix_table(corr_all))
    print()
    print("=== VALIDITY CORRELATION MATRIX (k=6) ===")
    print(_format_matrix_table(corr_k6))
    print()
    print("=== NOTABLE NEGATIVE CORRELATIONS (< -0.15) ===")
    if not neg_all and not neg_k6:
        print("none")
    else:
        for left, right, value in neg_all:
            print(f"all-runs: {left} vs {right}: {value:+.3f}")
        for left, right, value in neg_k6:
            print(f"k=6: {left} vs {right}: {value:+.3f}")
    print()

    print("=== STRONG EVIDENCE CHECK (k=6 correlation < -0.20) ===")
    if strong_k6:
        for left, right, value in strong_k6:
            print(f"strong: {left} vs {right}: {value:+.3f}")
    else:
        print("none")
    print()

    _print_quality_table("=== QUALITY BY VALID ARC COUNT (ALL RUNS) ===", quality_all)
    _print_quality_table("=== QUALITY BY VALID ARC COUNT (k=6) ===", quality_k6)
    print(_interpret_zero_sum(neg_all, neg_k6, strong_k6))


if __name__ == "__main__":
    main()
