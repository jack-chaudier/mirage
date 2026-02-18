#!/usr/bin/env python3
"""Create a balanced train JSONL file: all STRONG + equal-size DEGRADED sample.

Default labels use `evidence_degraded` if present; fallback parses assistant text.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--train-jsonl",
        default=str(ROOT / "data" / "processed" / "train.jsonl"),
        help="Input train JSONL",
    )
    p.add_argument(
        "--out-jsonl",
        default=str(ROOT / "data" / "processed" / "train_balanced.jsonl"),
        help="Output balanced JSONL",
    )
    p.add_argument("--seed", type=int, default=42, help="Sampling/shuffle seed")
    return p.parse_args()


def infer_degraded(row: Dict) -> bool:
    if "evidence_degraded" in row:
        return bool(row["evidence_degraded"])

    messages = row.get("messages", [])
    if len(messages) >= 2:
        assistant = str(messages[-1].get("content", ""))
        upper = assistant.upper()
        return "EVIDENCE ASSESSMENT: DEGRADED" in upper or "DEGRADED" in upper

    return False


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def describe(rows: List[Dict], label: str) -> None:
    label_counts = Counter("degraded" if infer_degraded(r) else "strong" for r in rows)
    diff_counts = Counter(r.get("difficulty", "unknown") for r in rows)
    cat_counts = Counter(r.get("category", "unknown") for r in rows)
    task_ids = [r.get("task_id") for r in rows if r.get("task_id") is not None]

    print(f"\n{label} summary:")
    print(f"  rows: {len(rows)}")
    print(f"  strong: {label_counts.get('strong', 0)}")
    print(f"  degraded: {label_counts.get('degraded', 0)}")
    print(f"  difficulty: {dict(diff_counts)}")
    print(f"  category: {dict(cat_counts)}")
    if task_ids:
        print(f"  unique task_id: {len(set(task_ids))}")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    train_path = Path(args.train_jsonl)
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(train_path)
    if not rows:
        raise ValueError(f"No rows found in {train_path}")

    strong_rows = [r for r in rows if not infer_degraded(r)]
    degraded_rows = [r for r in rows if infer_degraded(r)]

    if not strong_rows:
        raise ValueError("No STRONG rows found; cannot build balanced set.")
    if len(degraded_rows) < len(strong_rows):
        raise ValueError(
            f"Not enough DEGRADED rows to match STRONG count: "
            f"strong={len(strong_rows)}, degraded={len(degraded_rows)}"
        )

    sampled_degraded = rng.sample(degraded_rows, len(strong_rows))
    balanced_rows = strong_rows + sampled_degraded
    rng.shuffle(balanced_rows)

    write_jsonl(out_path, balanced_rows)

    describe(rows, "Input")
    describe(balanced_rows, "Balanced output")
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
