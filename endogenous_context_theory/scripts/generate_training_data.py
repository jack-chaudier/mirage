#!/usr/bin/env python3
"""Generate Mirage-aware training/eval JSONL data for MLX LoRA training.

This script reuses MirageBench task builders and produces:
  - 1 full-context example per task
  - 9 compressed-context examples per task (3 levels x 3 seeds)
for a total of 10 examples per task, with controlled prereq survival ratios
across compressed variants.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_miragebench_ollama as mirage_runtime  # noqa: E402


APPENDIX_MARKER = "Operational Appendix (intentionally low salience):"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate mirage-aware train/valid JSONL files for MLX LoRA."
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "data" / "processed"),
        help="Output directory for train.jsonl and valid.jsonl.",
    )
    parser.add_argument(
        "--tasks-per-difficulty",
        type=int,
        default=100,
        help="Tasks to generate per difficulty level.",
    )
    parser.add_argument(
        "--difficulties",
        default="easy,medium,hard,extreme",
        help="Comma-separated difficulty levels.",
    )
    parser.add_argument(
        "--categories",
        default="investment,incident,narrative",
        help=(
            "Comma-separated categories to generate "
            "(investment,incident,narrative)."
        ),
    )
    parser.add_argument(
        "--compression-levels",
        default="0.4,0.5,0.6",
        help="Comma-separated compression levels.",
    )
    parser.add_argument(
        "--seeds",
        default="101,202,303",
        help="Comma-separated seeds for compressed variants.",
    )
    parser.add_argument(
        "--train-task-count",
        type=int,
        default=None,
        help=(
            "Number of tasks allocated to train split (remaining go to valid). "
            "Defaults to ~5/6 of tasks if omitted."
        ),
    )
    parser.add_argument(
        "--max-context-words",
        type=int,
        default=1200,
        help="Maximum words kept for context before prompt assembly.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Seed for deterministic task split.",
    )
    parser.add_argument(
        "--balance-labels",
        dest="balance_labels",
        action="store_true",
        help=(
            "Balance labels by oversampling the minority class. "
            "No rows are dropped."
        ),
    )
    parser.add_argument(
        "--no-balance-labels",
        dest="balance_labels",
        action="store_false",
        help="Disable label balancing (keep all generated rows).",
    )
    parser.add_argument(
        "--balance-seed",
        type=int,
        default=42,
        help="Random seed used during label balancing.",
    )
    parser.add_argument(
        "--target-prereq-ratios",
        default="0.0,0.25,0.5,0.75",
        help=(
            "Comma-separated target prerequisite survival ratios for compressed "
            "variants. Applied in a cycle across (compression_level, seed) pairs."
        ),
    )
    parser.set_defaults(balance_labels=False)
    return parser.parse_args()


def _parse_csv_floats(raw: str) -> List[float]:
    return [float(piece.strip()) for piece in raw.split(",") if piece.strip()]


def _parse_csv_ints(raw: str) -> List[int]:
    return [int(piece.strip()) for piece in raw.split(",") if piece.strip()]


def _parse_csv_strs(raw: str) -> List[str]:
    return [piece.strip() for piece in raw.split(",") if piece.strip()]


def _word_count(text: str) -> int:
    return len((text or "").split())


def _line_contains_any(line: str, markers: Sequence[str]) -> bool:
    return any(marker in line for marker in markers)


def _trim_context(
    context: str,
    max_words: int,
    preserve_markers: Sequence[str],
) -> str:
    """Trim context while preserving pivot/prereq lines and rule line."""

    text = context.strip()
    if _word_count(text) <= max_words:
        return text

    # Drop low-salience appendix first.
    if APPENDIX_MARKER in text:
        text = text.split(APPENDIX_MARKER, 1)[0].rstrip()
        if _word_count(text) <= max_words:
            return text

    lines = text.splitlines()
    if not lines:
        return text

    keep_idx = set()

    # Keep preamble lines.
    for idx in range(min(3, len(lines))):
        keep_idx.add(idx)

    # Keep rule lines and lines with key markers.
    for idx, line in enumerate(lines):
        if line.startswith("Rule reminder:"):
            keep_idx.add(idx)
        elif _line_contains_any(line, preserve_markers):
            keep_idx.add(idx)

    # Build mandatory text first.
    mandatory_idx = sorted(keep_idx)
    mandatory_words = sum(_word_count(lines[i]) for i in mandatory_idx)

    # If mandatory set alone is too large, keep marker/rule lines and first preamble line.
    if mandatory_words > max_words:
        reduced = set()
        if lines:
            reduced.add(0)
        for idx, line in enumerate(lines):
            if line.startswith("Rule reminder:") or _line_contains_any(line, preserve_markers):
                reduced.add(idx)
        mandatory_idx = sorted(reduced)

    selected = set(mandatory_idx)
    current_words = sum(_word_count(lines[i]) for i in sorted(selected))

    # Fill remaining budget with earliest non-mandatory lines.
    for idx, line in enumerate(lines):
        if idx in selected:
            continue
        line_words = _word_count(line)
        if current_words + line_words > max_words:
            continue
        selected.add(idx)
        current_words += line_words
        if current_words >= max_words:
            break

    final_lines = [lines[i] for i in sorted(selected)]
    trimmed = "\n".join(final_lines).strip()
    if not trimmed:
        # Last-resort hard trim by words.
        words = text.split()
        return " ".join(words[:max_words])
    return trimmed


def _status_line(marker: str, exists: bool) -> str:
    state = "confirmed in context" if exists else "NOT FOUND in context"
    return f"- {marker}: {state}"


def _compute_oracle(task: Any, context: str) -> Dict[str, Any]:
    gt_pivot = str(task.pivot_ground_truth)
    decoy_pivot = str(task.decoy_pivot)
    true_setup_markers: List[str] = list(task.metadata.get("pivot_setup_markers", []))

    gt_present = gt_pivot in context
    oracle_pivot = gt_pivot if gt_present else decoy_pivot

    total_prereqs = len(true_setup_markers)
    prereqs_survived = sum(1 for marker in true_setup_markers if marker in context)
    prereq_ratio = prereqs_survived / total_prereqs if total_prereqs else 1.0

    evidence_degraded = (prereq_ratio < 1.0) or (not gt_present)
    return {
        "oracle_pivot": oracle_pivot,
        "gt_present": gt_present,
        "prereq_ratio": float(prereq_ratio),
        "prereqs_survived": int(prereqs_survived),
        "total_prereqs": int(total_prereqs),
        "evidence_degraded": bool(evidence_degraded),
        "true_setup_markers": true_setup_markers,
        "gt_pivot": gt_pivot,
        "decoy_pivot": decoy_pivot,
    }


def _build_marker_line_lookup(context: str, markers: Sequence[str]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    marker_set = [marker for marker in markers if marker]
    for line in context.splitlines():
        for marker in marker_set:
            if marker in line and marker not in lookup:
                lookup[marker] = line
    return lookup


def _insert_lines_before_rule(lines: List[str], new_lines: List[str]) -> List[str]:
    if not new_lines:
        return lines
    rule_idx = next(
        (idx for idx, line in enumerate(lines) if line.startswith("Rule reminder:")),
        len(lines),
    )
    return lines[:rule_idx] + new_lines + lines[rule_idx:]


def _stable_local_seed(
    task_uid: str,
    compression_level: float,
    compression_seed: int,
    split_seed: int,
) -> int:
    uid_acc = sum(ord(ch) for ch in task_uid)
    level_acc = int(round(float(compression_level) * 1000))
    return (split_seed * 1_000_003) + (compression_seed * 97_003) + (level_acc * 503) + uid_acc


def _enforce_prereq_ratio(
    context: str,
    full_context: str,
    true_setup_markers: Sequence[str],
    gt_pivot: str,
    target_ratio: float,
    rng: random.Random,
) -> str:
    """Force compressed variants to retain a controlled prereq survival ratio."""

    markers = [marker for marker in true_setup_markers if marker]
    text = context.strip()
    if not markers:
        return text

    total = len(markers)
    desired_keep = int(round(float(target_ratio) * total))
    # Compressed variants are intended to remain degraded (< 1.0).
    desired_keep = max(0, min(total - 1, desired_keep))

    present_markers = [marker for marker in markers if marker in text]
    if len(present_markers) >= desired_keep:
        keep_set = set(rng.sample(present_markers, desired_keep)) if desired_keep > 0 else set()
    else:
        keep_set = set(present_markers)
        missing_pool = [marker for marker in markers if marker not in keep_set]
        rng.shuffle(missing_pool)
        needed = desired_keep - len(keep_set)
        keep_set.update(missing_pool[:needed])

    remove_set = set(markers) - keep_set
    lines = text.splitlines()
    lines = [line for line in lines if not any(marker in line for marker in remove_set)]
    rebuilt = "\n".join(lines).strip()

    full_lookup = _build_marker_line_lookup(full_context, [*markers, gt_pivot])

    missing_keep_markers = [
        marker for marker in markers if marker in keep_set and marker not in rebuilt
    ]
    add_lines = [
        full_lookup[marker]
        for marker in missing_keep_markers
        if marker in full_lookup
    ]
    if gt_pivot and gt_pivot not in rebuilt and gt_pivot in full_lookup:
        add_lines.insert(0, full_lookup[gt_pivot])

    if add_lines:
        lines = rebuilt.splitlines() if rebuilt else []
        lines = _insert_lines_before_rule(lines, add_lines)
        rebuilt = "\n".join(lines).strip()

    return rebuilt


def _build_assistant_target(oracle: Dict[str, Any]) -> str:
    pivot_id = oracle["oracle_pivot"]
    total = oracle["total_prereqs"]
    survived = oracle["prereqs_survived"]
    missing = total - survived
    evidence_degraded = bool(oracle["evidence_degraded"])
    alt_pivot = oracle["decoy_pivot"] if pivot_id == oracle["gt_pivot"] else oracle["gt_pivot"]

    status_lines = [
        _status_line(marker, marker in set(oracle["present_markers"]))
        for marker in oracle["true_setup_markers"]
    ]
    prereq_block = "\n".join(status_lines) if status_lines else "- None: no prerequisites defined"

    if not evidence_degraded:
        return (
            f"PIVOT_ID={pivot_id}\n\n"
            "Evidence assessment: STRONG\n"
            f"- Primary pivot {pivot_id} present with {survived}/{total} prerequisite events confirmed\n"
            "- Confidence: HIGH\n\n"
            "Prerequisite events:\n"
            f"{prereq_block}\n\n"
            "Analysis: The selected pivot retains full prerequisite support in the provided timeline."
        )

    return (
        f"PIVOT_ID={pivot_id}\n\n"
        "Evidence assessment: DEGRADED\n"
        f"- Primary candidate {pivot_id} identified but {missing}/{total} prerequisite events not found in context\n"
        f"- Alternative candidate {alt_pivot} also viable\n"
        "- Confidence: LOW - evidence base appears incomplete\n\n"
        "Prerequisite events:\n"
        f"{prereq_block}\n\n"
        "Analysis: The context supports a salvage decision, but missing prerequisites reduce causal confidence."
    )


def _jsonl_write(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _balance_split_by_oversampling(
    rows: List[Dict[str, Any]],
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Balance labels by oversampling the minority class (never dropping rows)."""

    strong = [row for row in rows if not bool(row.get("evidence_degraded", False))]
    degraded = [row for row in rows if bool(row.get("evidence_degraded", False))]

    info: Dict[str, Any] = {
        "before_total": len(rows),
        "before_strong": len(strong),
        "before_degraded": len(degraded),
        "after_total": len(rows),
        "after_strong": len(strong),
        "after_degraded": len(degraded),
        "balanced": False,
        "strategy": "none",
        "added_rows": 0,
    }

    if not strong or not degraded:
        return rows, info
    if len(degraded) == len(strong):
        return rows, info

    rng = random.Random(seed)
    if len(strong) < len(degraded):
        minority = strong
        majority_size = len(degraded)
        strategy = "oversample_strong"
    else:
        minority = degraded
        majority_size = len(strong)
        strategy = "oversample_degraded"

    needed = majority_size - len(minority)
    duplicates = [dict(rng.choice(minority)) for _ in range(needed)]
    balanced = list(rows) + duplicates
    rng.shuffle(balanced)

    after_strong = sum(1 for row in balanced if not bool(row.get("evidence_degraded", False)))
    after_degraded = sum(1 for row in balanced if bool(row.get("evidence_degraded", False)))
    info.update(
        {
            "after_total": len(balanced),
            "after_strong": after_strong,
            "after_degraded": after_degraded,
            "balanced": True,
            "strategy": strategy,
            "added_rows": needed,
        }
    )
    return balanced, info


def _build_split_summary(
    rows: List[Dict[str, Any]],
    compression_levels: Sequence[float],
) -> Dict[str, Any]:
    by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task_id"])].append(row)

    counts = [len(task_rows) for task_rows in by_task.values()]
    min_count = min(counts) if counts else 0
    max_count = max(counts) if counts else 0
    mean_count = (sum(counts) / len(counts)) if counts else 0.0

    required_total_floor = max(4, 1 + len(compression_levels))
    tasks_below_floor = sorted(
        task_id for task_id, task_rows in by_task.items() if len(task_rows) < required_total_floor
    )

    expected_levels = {round(float(level), 6) for level in compression_levels}
    tasks_missing_levels: Dict[str, List[float]] = {}
    for task_id, task_rows in by_task.items():
        present_levels = {
            round(float(row["compression_level"]), 6)
            for row in task_rows
            if not bool(row["is_full_context"])
        }
        missing = sorted(expected_levels - present_levels)
        if missing:
            tasks_missing_levels[task_id] = missing

    prereq_dist = Counter(round(float(row.get("prereq_ratio", 0.0)), 2) for row in rows)
    label_split = Counter(
        "degraded" if bool(row.get("evidence_degraded", False)) else "strong"
        for row in rows
    )

    return {
        "num_tasks": len(by_task),
        "examples_per_task": {
            "min": min_count,
            "max": max_count,
            "mean": round(mean_count, 3),
        },
        "required_total_floor": required_total_floor,
        "tasks_below_floor_count": len(tasks_below_floor),
        "tasks_below_floor_examples": tasks_below_floor[:20],
        "tasks_missing_compression_levels_count": len(tasks_missing_levels),
        "tasks_missing_compression_levels_examples": {
            task_id: missing
            for task_id, missing in list(tasks_missing_levels.items())[:20]
        },
        "prereq_ratio_distribution": {
            f"{ratio:.2f}": count for ratio, count in sorted(prereq_dist.items())
        },
        "label_split": dict(label_split),
    }


def _preview_examples(rows: List[Dict[str, Any]], count: int = 3) -> None:
    print("\n=== Sample training pairs ===")
    for idx, row in enumerate(rows[:count], start=1):
        user_text = row["messages"][0]["content"]
        assistant_text = row["messages"][1]["content"]
        print(f"\n--- Example {idx} ---")
        print(
            f"task_id={row['task_id']} difficulty={row['difficulty']} "
            f"full={row['is_full_context']} level={row['compression_level']} seed={row['compression_seed']}"
        )
        print(f"oracle_pivot={row['oracle_pivot']} degraded={row['evidence_degraded']} prereq_ratio={row['prereq_ratio']:.3f}")
        print("USER (truncated):")
        print(user_text[:700] + ("..." if len(user_text) > 700 else ""))
        print("ASSISTANT:")
        print(assistant_text)


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    difficulties = _parse_csv_strs(args.difficulties)
    categories = [category.lower() for category in _parse_csv_strs(args.categories)]
    compression_levels = _parse_csv_floats(args.compression_levels)
    seeds = _parse_csv_ints(args.seeds)
    target_prereq_ratios = _parse_csv_floats(args.target_prereq_ratios)
    if not target_prereq_ratios:
        raise ValueError("--target-prereq-ratios must include at least one value.")

    compressed_variant_plan: List[Tuple[float, int, float]] = []
    ratio_cycle = list(target_prereq_ratios)
    compressed_pairs = [(float(level), int(seed)) for level in compression_levels for seed in seeds]
    for idx, (level, seed) in enumerate(compressed_pairs):
        # Keep compressed targets strictly degraded (< 1.0).
        ratio = min(0.999, max(0.0, float(ratio_cycle[idx % len(ratio_cycle)])))
        compressed_variant_plan.append((level, seed, ratio))
    target_ratio_counts = Counter(ratio for _, _, ratio in compressed_variant_plan)

    notebook_path = ROOT / "notebooks" / "miragebench_experiments_colab.ipynb"
    runtime = mirage_runtime._load_notebook_runtime(notebook_path)
    mirage_runtime._patch_runtime_with_methodology_fixes(runtime)

    render_compressed = runtime["render_compressed_variant"]
    make_prompt = runtime["make_prompt"]

    supported_categories = {"investment", "incident", "narrative"}
    if not categories:
        raise ValueError("--categories must include at least one category.")
    unsupported_categories = sorted(set(categories) - supported_categories)
    if unsupported_categories:
        raise ValueError(
            "Unsupported categories: "
            f"{', '.join(unsupported_categories)}. "
            "Supported values are investment,incident,narrative."
        )

    category_builders: Dict[str, Any] = {
        "investment": runtime.get("build_investment_task"),
        "incident": runtime.get("build_incident_task"),
        "narrative": runtime.get("build_narrative_task"),
    }
    missing_builders = sorted(
        category for category in categories if category_builders.get(category) is None
    )
    if missing_builders:
        raise RuntimeError(
            "Category builders missing from notebook runtime: "
            + ", ".join(missing_builders)
        )

    category_prefix = {
        "investment": "INV",
        "incident": "INC",
        "narrative": "NAR",
    }

    tasks: List[Dict[str, Any]] = []
    for category in categories:
        build_task = category_builders[category]
        prefix = category_prefix[category]

        for difficulty_idx, difficulty in enumerate(difficulties):
            for task_num in range(1, args.tasks_per_difficulty + 1):
                if category == "investment":
                    task = build_task(task_num, difficulty=difficulty)
                else:
                    # Incident/narrative builders do not take difficulty;
                    # synthesize unique task numbers across difficulty buckets.
                    category_task_num = (difficulty_idx * args.tasks_per_difficulty) + task_num
                    task = build_task(category_task_num)

                task_uid = f"{prefix}-{difficulty}-{task_num:03d}"
                tasks.append(
                    {
                        "task_uid": task_uid,
                        "category": category,
                        "difficulty": difficulty,
                        "task": task,
                    }
                )

    train_task_count = args.train_task_count
    if train_task_count is None:
        holdout = max(1, len(tasks) // 6)
        train_task_count = len(tasks) - holdout

    if train_task_count <= 0 or train_task_count >= len(tasks):
        raise ValueError(
            f"--train-task-count must be in [1, {len(tasks)-1}], got {train_task_count}"
        )

    rng = random.Random(args.split_seed)
    rng.shuffle(tasks)
    train_task_uids = {item["task_uid"] for item in tasks[:train_task_count]}

    train_rows: List[Dict[str, Any]] = []
    valid_rows: List[Dict[str, Any]] = []

    for item in tasks:
        task_uid = item["task_uid"]
        category = item["category"]
        difficulty = item["difficulty"]
        task = item["task"]
        metadata = task.metadata
        gt_pivot = str(task.pivot_ground_truth)
        true_setup_markers = list(metadata.get("pivot_setup_markers", []))
        preserve_markers = [
            gt_pivot,
            str(task.decoy_pivot),
            *true_setup_markers,
            *list(metadata.get("decoy_setup_markers", [])),
        ]

        variants: List[Tuple[str, bool, float, int, float]] = []
        variants.append((task.full_context, True, 0.0, 0, 1.0))
        for level, seed, target_ratio in compressed_variant_plan:
            raw_context = render_compressed(task, drop_fraction=level, seed=seed)
            local_seed = _stable_local_seed(
                task_uid=task_uid,
                compression_level=level,
                compression_seed=seed,
                split_seed=args.split_seed,
            )
            ratio_rng = random.Random(local_seed)
            adjusted_context = _enforce_prereq_ratio(
                context=raw_context,
                full_context=task.full_context,
                true_setup_markers=true_setup_markers,
                gt_pivot=gt_pivot,
                target_ratio=target_ratio,
                rng=ratio_rng,
            )
            variants.append((adjusted_context, False, level, seed, target_ratio))

        for context_text, is_full_context, compression_level, compression_seed, target_ratio in variants:
            trimmed_context = _trim_context(
                context=context_text,
                max_words=args.max_context_words,
                preserve_markers=preserve_markers,
            )

            oracle = _compute_oracle(task, trimmed_context)
            present_markers = set()
            for marker in oracle["true_setup_markers"]:
                if marker in trimmed_context:
                    present_markers.add(marker)
            oracle["present_markers"] = sorted(present_markers)

            user_prompt = make_prompt(trimmed_context, task.question)
            assistant_target = _build_assistant_target(oracle)

            row = {
                "messages": [
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_target},
                ],
                "oracle_pivot": oracle["oracle_pivot"],
                "evidence_degraded": oracle["evidence_degraded"],
                "prereq_ratio": round(float(oracle["prereq_ratio"]), 6),
                "category": category,
                "difficulty": difficulty,
                "task_id": task_uid,
                "is_full_context": bool(is_full_context),
                "compression_level": float(compression_level),
                "compression_seed": int(compression_seed),
                "target_prereq_ratio": round(float(target_ratio), 6),
                "gt_pivot": oracle["gt_pivot"],
                "decoy_pivot": oracle["decoy_pivot"],
            }

            if task_uid in train_task_uids:
                train_rows.append(row)
            else:
                valid_rows.append(row)

    pre_balance = {
        "train_total": len(train_rows),
        "train_strong": sum(1 for row in train_rows if not row["evidence_degraded"]),
        "train_degraded": sum(1 for row in train_rows if row["evidence_degraded"]),
        "valid_total": len(valid_rows),
        "valid_strong": sum(1 for row in valid_rows if not row["evidence_degraded"]),
        "valid_degraded": sum(1 for row in valid_rows if row["evidence_degraded"]),
    }

    balance_info = {"train": None, "valid": None}
    if args.balance_labels:
        train_rows, balance_info["train"] = _balance_split_by_oversampling(
            train_rows, seed=args.balance_seed
        )
        valid_rows, balance_info["valid"] = _balance_split_by_oversampling(
            valid_rows, seed=args.balance_seed + 1
        )

    train_path = output_dir / "train.jsonl"
    valid_path = output_dir / "valid.jsonl"
    _jsonl_write(train_path, train_rows)
    _jsonl_write(valid_path, valid_rows)

    # Validate parseability.
    for path in (train_path, valid_path):
        with path.open("r", encoding="utf-8") as handle:
            for line_num, line in enumerate(handle, start=1):
                try:
                    json.loads(line)
                except Exception as exc:  # pragma: no cover
                    raise RuntimeError(f"Failed to parse {path.name}:{line_num}: {exc}") from exc

    train_full = sum(1 for row in train_rows if row["is_full_context"])
    valid_full = sum(1 for row in valid_rows if row["is_full_context"])
    train_degraded = sum(1 for row in train_rows if row["evidence_degraded"])
    valid_degraded = sum(1 for row in valid_rows if row["evidence_degraded"])
    train_summary = _build_split_summary(train_rows, compression_levels=compression_levels)
    valid_summary = _build_split_summary(valid_rows, compression_levels=compression_levels)

    if train_summary["tasks_below_floor_count"] or valid_summary["tasks_below_floor_count"]:
        raise RuntimeError(
            "Coverage floor violation: some tasks have fewer than the required "
            f"{max(4, 1 + len(compression_levels))} examples."
        )
    if (
        train_summary["tasks_missing_compression_levels_count"]
        or valid_summary["tasks_missing_compression_levels_count"]
    ):
        raise RuntimeError(
            "Coverage violation: some tasks are missing at least one compression level."
        )

    stats = {
        "config": {
            "tasks_per_difficulty": args.tasks_per_difficulty,
            "categories": categories,
            "difficulties": difficulties,
            "compression_levels": compression_levels,
            "seeds": seeds,
            "target_prereq_ratios": target_prereq_ratios,
            "train_task_count": train_task_count,
            "valid_task_count": len(tasks) - train_task_count,
            "max_context_words": args.max_context_words,
            "split_seed": args.split_seed,
            "balance_labels": bool(args.balance_labels),
            "balance_seed": args.balance_seed,
        },
        "pre_balance_counts": pre_balance,
        "balance_info": balance_info,
        "counts": {
            "total_tasks": len(tasks),
            "total_examples": len(train_rows) + len(valid_rows),
            "train_examples": len(train_rows),
            "valid_examples": len(valid_rows),
            "train_full_context_examples": train_full,
            "valid_full_context_examples": valid_full,
            "train_compressed_examples": len(train_rows) - train_full,
            "valid_compressed_examples": len(valid_rows) - valid_full,
            "task_counts_by_category": dict(Counter(item["category"] for item in tasks)),
        },
        "label_balance": {
            "train_degraded_count": train_degraded,
            "valid_degraded_count": valid_degraded,
            "train_degraded_rate": round(train_degraded / max(1, len(train_rows)), 6),
            "valid_degraded_rate": round(valid_degraded / max(1, len(valid_rows)), 6),
        },
        "difficulty_counts": {
            "train": dict(Counter(row["difficulty"] for row in train_rows)),
            "valid": dict(Counter(row["difficulty"] for row in valid_rows)),
        },
        "category_counts": {
            "train": dict(Counter(row["category"] for row in train_rows)),
            "valid": dict(Counter(row["category"] for row in valid_rows)),
        },
        "coverage_summary": {
            "train": train_summary,
            "valid": valid_summary,
        },
    }
    stats_path = output_dir / "data_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("Saved:", train_path)
    print("Saved:", valid_path)
    print("Saved:", stats_path)
    print(
        "Counts:",
        f"train={len(train_rows)}",
        f"valid={len(valid_rows)}",
        f"total={len(train_rows) + len(valid_rows)}",
    )
    print(
        "Expected examples:",
        f"train={train_task_count * (1 + len(compression_levels) * len(seeds))}",
        f"valid={(len(tasks) - train_task_count) * (1 + len(compression_levels) * len(seeds))}",
    )
    print("Compressed variant prereq targets (level, seed, ratio):")
    for level, seed, ratio in compressed_variant_plan:
        print(f"  level={level:.2f} seed={seed} target_prereq_ratio={ratio:.2f}")
    print(
        "Compressed target ratio counts:",
        {f"{ratio:.2f}": count for ratio, count in sorted(target_ratio_counts.items())},
    )
    train_degraded_rate = stats["label_balance"]["train_degraded_rate"]
    valid_degraded_rate = stats["label_balance"]["valid_degraded_rate"]
    print(
        "Label balance:",
        f"train_degraded_rate={train_degraded_rate:.3f}",
        f"valid_degraded_rate={valid_degraded_rate:.3f}",
    )

    def _print_split_validation(name: str, summary: Dict[str, Any]) -> None:
        expt = summary["examples_per_task"]
        label_split = summary["label_split"]
        strong = int(label_split.get("strong", 0))
        degraded = int(label_split.get("degraded", 0))
        total = strong + degraded
        degraded_rate = (degraded / total) if total else 0.0
        prereq_dist = summary["prereq_ratio_distribution"]

        print(f"\n{name} split validation:")
        print(
            "  examples_per_task:",
            f"min={expt['min']}",
            f"max={expt['max']}",
            f"mean={expt['mean']}",
        )
        print(
            "  label_split:",
            f"strong={strong}",
            f"degraded={degraded}",
            f"degraded_rate={degraded_rate:.3f}",
        )
        print("  prereq_ratio_distribution:", prereq_dist)
        print(
            "  coverage_floor:",
            f"required={summary['required_total_floor']}",
            f"tasks_below_floor={summary['tasks_below_floor_count']}",
            f"tasks_missing_levels={summary['tasks_missing_compression_levels_count']}",
        )
        if summary["tasks_below_floor_count"] == 0:
            print("  floor_check: PASS (no task has fewer than 4 examples).")
        else:
            print("  floor_check: FAIL")

    _print_split_validation("Train", train_summary)
    _print_split_validation("Valid", valid_summary)

    if not (0.35 <= train_degraded_rate <= 0.65):
        print(
            "\nNote: train split is not close to 50/50 STRONG/DEGRADED. "
            "No rows were dropped; use oversampling or loss weighting if balance is required."
        )

    _preview_examples(train_rows, count=3)


if __name__ == "__main__":
    main()
