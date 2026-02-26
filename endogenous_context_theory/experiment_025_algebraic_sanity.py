#!/usr/bin/env python3
"""Experiment 2.5: algebraic sanity checks for endogenous context composition.

This script validates the exact algebraic properties required before running
Experiment 3 (LLM bracketing divergence):

1) identity
2) associativity on an Exp58 witness (or synthetic fallback)
3) associativity across all 3-block permutations
4) non-commutativity
5) absorbing-state behavior under committed vs endogenous semantics
6) random stress associativity checks

If all checks pass, any downstream bracketing divergence is attributable to
model compression dynamics rather than algebraic inconsistency.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


NEG_INF = float("-inf")


@dataclass(frozen=True)
class Context:
    """Original Paper-03 context tuple C = (w*, d_total, d_pre)."""

    w_star: float
    d_total: int
    d_pre: int


def compose(left: Context, right: Context) -> Context:
    """Endogenous composition with deterministic left tie-breaking."""

    if left.w_star >= right.w_star:
        return Context(
            w_star=left.w_star,
            d_total=left.d_total + right.d_total,
            d_pre=left.d_pre,
        )
    return Context(
        w_star=right.w_star,
        d_total=left.d_total + right.d_total,
        d_pre=left.d_total + right.d_pre,
    )


def compose_committed(left: Context, right: Context, kappa_left: int) -> Context:
    """Committed composition: if left is committed, suffix cannot shift pivot."""

    if int(kappa_left) == 1:
        return Context(
            w_star=left.w_star,
            d_total=left.d_total + right.d_total,
            d_pre=left.d_pre,
        )
    return compose(left, right)


def identity() -> Context:
    return Context(w_star=NEG_INF, d_total=0, d_pre=0)


def _is_context_dict(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    return all(key in value for key in ("w_star", "d_total", "d_pre"))


def _to_context(value: dict[str, Any]) -> Context:
    return Context(
        w_star=float(value["w_star"]),
        d_total=int(value["d_total"]),
        d_pre=int(value["d_pre"]),
    )


def _recursive_context_scan(value: Any) -> list[Context]:
    found: list[Context] = []
    if _is_context_dict(value):
        found.append(_to_context(value))
    if isinstance(value, dict):
        for nested in value.values():
            found.extend(_recursive_context_scan(nested))
    elif isinstance(value, list):
        for nested in value:
            found.extend(_recursive_context_scan(nested))
    return found


def _load_witness_blocks(repo_root: Path) -> tuple[dict[str, Context], str]:
    candidate_paths = [
        repo_root / "experiments" / "output" / "exp58_pivot_shift_witness_semantics.json",
        repo_root / "projects" / "rhun" / "experiments" / "output" / "exp58_pivot_shift_witness_semantics.json",
        repo_root / "endogenous_context_theory" / "experiments" / "output" / "exp58_pivot_shift_witness_semantics.json",
    ]

    for path in candidate_paths:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        if isinstance(payload, dict) and "blocks" in payload and isinstance(payload["blocks"], dict):
            block_map = payload["blocks"]
            labels = ["A", "B", "C"]
            if all(label in block_map and _is_context_dict(block_map[label]) for label in labels):
                return {label: _to_context(block_map[label]) for label in labels}, str(path)

        scanned = _recursive_context_scan(payload)
        if len(scanned) >= 3:
            return {"A": scanned[0], "B": scanned[1], "C": scanned[2]}, f"{path} (recursive extraction)"

    synthetic = {
        "A": Context(10.0, 5, 2),
        "B": Context(15.0, 8, 3),
        "C": Context(7.0, 6, 4),
    }
    return synthetic, "synthetic_fallback"


def _assoc_equal(a: Context, b: Context, c: Context) -> tuple[bool, Context, Context]:
    left = compose(compose(a, b), c)
    right = compose(a, compose(b, c))
    return (left == right), left, right


def _case_for_ordering(ordering: tuple[str, str, str], blocks: dict[str, Context]) -> int:
    contexts = [blocks[label] for label in ordering]
    winner_idx = 0
    winner_weight = contexts[0].w_star
    for idx in (1, 2):
        if contexts[idx].w_star > winner_weight:
            winner_idx = idx
            winner_weight = contexts[idx].w_star
    return winner_idx + 1


def run_experiment(repo_root: Path, output_path: Path, random_seed: int) -> int:
    blocks, witness_source = _load_witness_blocks(repo_root=repo_root)
    labels = ("A", "B", "C")
    e = identity()

    identity_details: list[dict[str, Any]] = []
    identity_pass = True
    for label in labels:
        ctx = blocks[label]
        left_ok = (compose(ctx, e) == ctx)
        right_ok = (compose(e, ctx) == ctx)
        identity_pass = identity_pass and left_ok and right_ok
        identity_details.append(
            {
                "label": label,
                "left_identity_ok": left_ok,
                "right_identity_ok": right_ok,
            }
        )
    identity_checks = len(labels) * 2

    witness_ok, witness_left, witness_right = _assoc_equal(blocks["A"], blocks["B"], blocks["C"])

    permutation_results: list[dict[str, Any]] = []
    perm_pass_count = 0
    for perm in itertools.permutations(labels):
        a, b, c = (blocks[perm[0]], blocks[perm[1]], blocks[perm[2]])
        ok, left, right = _assoc_equal(a, b, c)
        if ok:
            perm_pass_count += 1
        permutation_results.append(
            {
                "permutation": list(perm),
                "case": _case_for_ordering(perm, blocks),
                "pass": ok,
                "left": asdict(left),
                "right": asdict(right),
            }
        )
    perm_total = len(permutation_results)
    permutations_pass = (perm_pass_count == perm_total)

    pair_results: list[dict[str, Any]] = []
    non_comm_count = 0
    for left_label, right_label in itertools.combinations(labels, 2):
        lr = compose(blocks[left_label], blocks[right_label])
        rl = compose(blocks[right_label], blocks[left_label])
        non_comm = (lr != rl)
        if non_comm:
            non_comm_count += 1
        pair_results.append(
            {
                "pair": [left_label, right_label],
                "non_commutative": non_comm,
                "left_right": asdict(lr),
                "right_left": asdict(rl),
            }
        )
    non_comm_confirmed = (non_comm_count > 0)

    k = 3
    committed_prefix = Context(w_star=5.0, d_total=2, d_pre=2)
    escaping_suffix = Context(w_star=12.0, d_total=6, d_pre=4)
    committed_out = compose_committed(committed_prefix, escaping_suffix, kappa_left=1)
    endogenous_out = compose(committed_prefix, escaping_suffix)

    absorbing_committed_pass = (
        committed_out.d_pre == committed_prefix.d_pre
        and committed_out.d_pre < k
    )
    absorbing_endogenous_pass = (
        endogenous_out.d_pre > committed_prefix.d_pre
        and endogenous_out.d_pre >= k
    )

    rng = random.Random(int(random_seed))
    random_blocks: list[Context] = []
    for _ in range(20):
        d_total = rng.randint(1, 50)
        random_blocks.append(
            Context(
                w_star=float(rng.randint(1, 100)),
                d_total=d_total,
                d_pre=rng.randint(0, d_total),
            )
        )

    stress_total = 0
    stress_violations = 0
    for idx in range(len(random_blocks) - 2):
        stress_total += 1
        ok, _, _ = _assoc_equal(random_blocks[idx], random_blocks[idx + 1], random_blocks[idx + 2])
        if not ok:
            stress_violations += 1
    stress_pass = (stress_violations == 0)

    overall_pass = all(
        [
            identity_pass,
            witness_ok,
            permutations_pass,
            non_comm_confirmed,
            absorbing_committed_pass,
            absorbing_endogenous_pass,
            stress_pass,
        ]
    )

    print("=== EXPERIMENT 2.5: ALGEBRAIC SANITY CHECK ===")
    print(f"Witness source: {witness_source}")
    print(f"Identity checks: {'PASS' if identity_pass else 'FAIL'} ({identity_checks} checks)")
    print(f"Associativity (Exp58 witness): {'PASS' if witness_ok else 'FAIL'}")
    print(f"Associativity (all permutations): {'PASS' if permutations_pass else 'FAIL'} ({perm_pass_count}/{perm_total})")
    for row in permutation_results:
        perm_label = f"({','.join(row['permutation'])})"
        status = "PASS" if row["pass"] else "FAIL"
        print(f"  Permutation {perm_label}: Case {row['case']} - {status}")
    print(
        f"Non-commutativity: {'CONFIRMED' if non_comm_confirmed else 'FAILED'} "
        f"({non_comm_count} non-commutative pairs out of {len(pair_results)})"
    )
    print(f"Absorbing state (committed): {'PASS' if absorbing_committed_pass else 'FAIL'}")
    print(f"Absorbing state (endogenous escape): {'PASS' if absorbing_endogenous_pass else 'FAIL'}")
    print(
        f"Random stress test: {'PASS' if stress_pass else 'FAIL'} "
        f"({stress_total - stress_violations}/{stress_total} checks, {stress_violations} violations)"
    )
    if overall_pass:
        print("=== THEORY VALIDATION: monoid is exact, non-commutative, with absorbing ideal ===")
    else:
        print("=== THEORY VALIDATION: FAIL (inspect per-check diagnostics) ===")

    output_payload = {
        "witness_source": witness_source,
        "blocks": {label: asdict(blocks[label]) for label in labels},
        "identity": {
            "pass": identity_pass,
            "checks": identity_checks,
            "details": identity_details,
        },
        "associativity_witness": {
            "pass": witness_ok,
            "left": asdict(witness_left),
            "right": asdict(witness_right),
        },
        "associativity_permutations": {
            "pass": permutations_pass,
            "pass_count": perm_pass_count,
            "total": perm_total,
            "details": permutation_results,
        },
        "non_commutativity": {
            "pass": non_comm_confirmed,
            "non_commutative_pairs": non_comm_count,
            "total_pairs": len(pair_results),
            "details": pair_results,
        },
        "absorbing_state": {
            "k": k,
            "committed_prefix": asdict(committed_prefix),
            "suffix": asdict(escaping_suffix),
            "committed_result": asdict(committed_out),
            "endogenous_result": asdict(endogenous_out),
            "committed_pass": absorbing_committed_pass,
            "endogenous_escape_pass": absorbing_endogenous_pass,
        },
        "random_stress": {
            "seed": int(random_seed),
            "n_blocks": len(random_blocks),
            "checks": stress_total,
            "violations": stress_violations,
            "pass": stress_pass,
        },
        "overall_pass": overall_pass,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    print(f"Saved JSON: {output_path}")

    return 0 if overall_pass else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_json",
        type=Path,
        default=None,
        help="Where to save structured check results.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=58,
        help="Seed for the random triple stress test.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = (
        args.output_json
        if args.output_json is not None
        else (Path(__file__).resolve().parent / "results" / "experiment_025_algebraic_sanity.json")
    )
    exit_code = run_experiment(
        repo_root=Path(__file__).resolve().parent.parent,
        output_path=output_path,
        random_seed=args.random_seed,
    )
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
