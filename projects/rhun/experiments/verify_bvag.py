#!/usr/bin/env python3
"""BVAG verification diagnostic.

Verifies:
1) B_lb correctness
2) Multi-burst gap failure geometry (gap/G and B_lb)
3) Rejection reason comparison (gap-aware vs BVAG)
4) Valley bridge availability signal
5) Bursty recovered cases requiring 2+ bridges

Run:
  cd ~/rhun && .venv/bin/python experiments/verify_bvag.py
"""

from __future__ import annotations

import json
from collections import Counter
from statistics import mean, median

from rhun.extraction.pool_construction import injection_pool
from rhun.extraction.viability_greedy import _compute_bridge_budget
from rhun.generators.bursty import BurstyConfig, BurstyGenerator
from rhun.generators.multiburst import MultiBurstConfig, MultiBurstGenerator


BVAG_PATH = "experiments/output/bvag_evaluation.json"


def main() -> None:
    with open(BVAG_PATH, "r", encoding="utf-8") as handle:
        payload = json.load(handle)["results"]

    mb_rows = payload["per_case"]["multiburst_with_gap"]
    b_rows = payload["per_case"]["bursty_with_gap"]

    g_mb = float(payload["settings"]["multiburst_gap_threshold"])
    g_b = float(payload["settings"]["bursty_gap_threshold"])

    print("=" * 64)
    print("CHECK 1: _compute_bridge_budget sanity")
    print("=" * 64)
    cases = [
        ([0.0, 0.40], 0.40, 0),
        ([0.0, 0.401], 0.40, 1),
        ([0.0, 0.44], 0.40, 1),
        ([0.0, 0.80], 0.40, 1),
        ([0.0, 0.81], 0.40, 2),
        ([0.0, 0.28], 0.14, 1),
        ([0.0, 0.29], 0.14, 2),
    ]
    ok = True
    for ts, g, expected in cases:
        actual = _compute_bridge_budget(ts, g)
        passed = actual == expected
        ok = ok and passed
        print(f"  {'OK' if passed else 'FAIL'} B_lb({ts}, G={g})={actual} expect={expected}")
    print("  Result:", "PASS" if ok else "FAIL")

    print("\n" + "=" * 64)
    print("CHECK 2: Multi-burst exact-feasible failures (gap-aware VAG)")
    print("=" * 64)
    mg = MultiBurstGenerator()

    mb_fail = [
        row
        for row in mb_rows
        if row["results"]["exact_oracle"]["valid"] and not row["results"]["vag_gap_aware"]["valid"]
    ]
    print(f"Failures: {len(mb_fail)} (G={g_mb:.4f})")

    ratios: list[float] = []
    max_gaps: list[float] = []
    blb_counts: Counter[int] = Counter()

    for row in mb_fail:
        seed = int(row["seed"])
        graph = mg.generate(MultiBurstConfig(seed=seed, n_events=200, n_actors=6))
        by_id = {event.id: event for event in graph.events}

        ids = row["results"]["vag_gap_aware"]["event_ids"]
        ts = sorted(float(by_id[event_id].timestamp) for event_id in ids if event_id in by_id)
        if len(ts) < 2:
            continue
        gaps = [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]
        mx = max(gaps)
        max_gaps.append(mx)
        ratios.append(mx / g_mb)

        blb = _compute_bridge_budget(ts, g_mb)
        blb_counts[blb] += 1

    if max_gaps:
        print(
            "  max_gap stats: "
            f"mean={mean(max_gaps):.6f} median={median(max_gaps):.6f} "
            f"min={min(max_gaps):.6f} max={max(max_gaps):.6f}"
        )
        print(
            "  ratio max_gap/G stats: "
            f"mean={mean(ratios):.4f} median={median(ratios):.4f} "
            f"min={min(ratios):.4f} max={max(ratios):.4f}"
        )
    print(f"  B_lb distribution: {dict(sorted(blb_counts.items()))}")

    print("\n" + "=" * 64)
    print("CHECK 3: Rejection counts (multi-burst fail subset)")
    print("=" * 64)
    reasons_gap = Counter()
    reasons_bvag = Counter()
    for row in mb_fail:
        reasons_gap.update(row["results"]["vag_gap_aware"]["diagnostics"].get("viability_rejection_reason_counts", {}))
        reasons_bvag.update(row["results"]["vag_budget_aware"]["diagnostics"].get("viability_rejection_reason_counts", {}))

    print("  Gap-aware top reasons:", reasons_gap.most_common(6))
    print("  BVAG top reasons:", reasons_bvag.most_common(6))
    print("  bridge_budget_exceeded total (BVAG):", reasons_bvag.get("bridge_budget_exceeded", 0))

    print("\n" + "=" * 64)
    print("CHECK 4: Valley bridge signal in selected anchor pool")
    print("=" * 64)
    valley_focal_counts: list[int] = []
    max_gap_bridge_options: list[int] = []
    has_bridge = 0

    for row in mb_fail:
        seed = int(row["seed"])
        graph = mg.generate(MultiBurstConfig(seed=seed, n_events=200, n_actors=6))
        by_id = {event.id: event for event in graph.events}

        anchor_id = row["results"]["vag_gap_aware"]["selected_anchor_id"]
        pool_ids = injection_pool(
            graph=graph,
            anchor_id=anchor_id,
            focal_actor="actor_0",
            max_depth=3,
            injection_top_n=40,
        )
        pool = [by_id[event_id] for event_id in pool_ids if event_id in by_id]
        focal_pool = [event for event in pool if "actor_0" in event.actors]

        valley_focal = [
            event for event in focal_pool if 0.30 <= float(event.timestamp) <= 0.70
        ]
        valley_focal_counts.append(len(valley_focal))

        ids = row["results"]["vag_gap_aware"]["event_ids"]
        seq = sorted((by_id[event_id] for event_id in ids if event_id in by_id), key=lambda event: event.timestamp)
        if len(seq) < 2:
            continue
        gap_rows = [(seq[i], seq[i + 1], float(seq[i + 1].timestamp - seq[i].timestamp)) for i in range(len(seq) - 1)]
        left, right, _ = max(gap_rows, key=lambda triplet: triplet[2])
        bridge_candidates = [
            event
            for event in focal_pool
            if float(left.timestamp) < float(event.timestamp) < float(right.timestamp)
        ]
        max_gap_bridge_options.append(len(bridge_candidates))
        if bridge_candidates:
            has_bridge += 1

    if valley_focal_counts:
        print(
            "  focal valley events in pool: "
            f"mean={mean(valley_focal_counts):.3f} min={min(valley_focal_counts)} max={max(valley_focal_counts)}"
        )
    if max_gap_bridge_options:
        print(
            "  focal bridge options for max-gap interval: "
            f"mean={mean(max_gap_bridge_options):.3f} min={min(max_gap_bridge_options)} max={max(max_gap_bridge_options)}"
        )
        print(f"  max-gap interval has >=1 focal bridge in pool: {has_bridge}/{len(max_gap_bridge_options)}")

    print("\n" + "=" * 64)
    print("CHECK 5: Bursty recovered by BVAG")
    print("=" * 64)
    bg = BurstyGenerator()
    recovered = [
        row
        for row in b_rows
        if row["results"]["exact_oracle"]["valid"]
        and not row["results"]["vag_gap_aware"]["valid"]
        and row["results"]["vag_budget_aware"]["valid"]
    ]

    print(f"Recovered count: {len(recovered)} (G={g_b:.4f})")
    for row in recovered:
        seed = int(row["seed"])
        epsilon = float(row["epsilon"])
        graph = bg.generate(BurstyConfig(seed=seed, epsilon=epsilon, n_events=200, n_actors=6))
        by_id = {event.id: event for event in graph.events}

        gap_ids = row["results"]["vag_gap_aware"]["event_ids"]
        ts = sorted(float(by_id[event_id].timestamp) for event_id in gap_ids if event_id in by_id)
        gaps = [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]
        mx = max(gaps) if gaps else 0.0
        blb = _compute_bridge_budget(ts, g_b)

        chosen_anchor = row["results"]["vag_budget_aware"]["selected_anchor_id"]
        first_block = None
        for anchor_row in row["results"]["vag_budget_aware"]["diagnostics"].get("per_anchor", []):
            if anchor_row.get("anchor_id") == chosen_anchor:
                first_block = anchor_row.get("first_budget_block")
                break

        print(
            f"  seed={seed:2d} eps={epsilon:.2f} max_gap={mx:.6f} ratio={mx/g_b:.4f} "
            f"B_lb={blb} first_block={first_block}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
