"""Run a five-story research chain (A→B→C→D→E) with Rashomon extraction.

This script composes existing simulation, metrics, Rashomon, and narration
pipelines to measure whether narrative structure enriches or flattens as canon
constraints accumulate across stories.

Run:
    cd src/engine
    python -m scripts.research_chain \
        --seeds 42,51,7,13,29 \
        --output scripts/output/research_chain.json

Optional:
    --skip-narration   # sim+metrics+rashomon only (no LLM calls)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from narrativefield.extraction.rashomon import RashomonArc, RashomonSet, extract_rashomon_set
from narrativefield.metrics.pipeline import parse_simulation_output, run_metrics_pipeline
from narrativefield.schema.canon import WorldCanon, decay_canon
from narrativefield.schema.events import BeatType
from scripts.demo_canon_persistence import StoryRun, run_story

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
WOUND_ANALYSIS_PATH = OUTPUT_DIR / "wound_analysis_1_50.json"
DINNER_PARTY_AGENTS = ["thorne", "elena", "marcus", "lydia", "diana", "victor"]
BEAT_ORDER = [
    BeatType.SETUP.value,
    BeatType.COMPLICATION.value,
    BeatType.ESCALATION.value,
    BeatType.TURNING_POINT.value,
    BeatType.CONSEQUENCE.value,
]


@dataclass(frozen=True)
class WoundPattern:
    pattern: str
    agent_pair: tuple[str, str]
    location_id: str
    frequency: float


def _resolve_output_path(output_arg: str) -> Path:
    output_path = Path(output_arg)
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    return output_path


def _required_api_keys_present() -> bool:
    required = ("ANTHROPIC_API_KEY", "XAI_API_KEY")
    missing = [key for key in required if not os.getenv(key)]
    if not missing:
        return True
    print("ERROR: Missing required API keys for narration stage.")
    print("Missing:")
    for key in missing:
        print(f"  - {key}")
    return False


def _parse_seeds(raw: str) -> list[int]:
    values = [item.strip() for item in raw.split(",")]
    if len(values) != 5:
        raise ValueError("--seeds must contain exactly 5 comma-separated integers for A→E.")
    seeds: list[int] = []
    for item in values:
        if item == "":
            raise ValueError("--seeds contains an empty value.")
        try:
            seeds.append(int(item))
        except ValueError as exc:
            raise ValueError(f"Invalid seed value '{item}'. Seeds must be integers.") from exc
    return seeds


def _pair_key(agent_a: str, agent_b: str) -> tuple[str, str]:
    left, right = sorted((str(agent_a), str(agent_b)))
    return (left, right)


def _parse_wound_pattern(raw_pattern: str) -> tuple[tuple[str, str], str]:
    if " @ " not in raw_pattern:
        raise ValueError(f"Invalid wound pattern '{raw_pattern}': expected '<a>-<b> @ <location>'.")
    pair_text, location_id = raw_pattern.split(" @ ", 1)
    pair_parts = pair_text.split("-", 1)
    if len(pair_parts) != 2:
        raise ValueError(f"Invalid wound pair in '{raw_pattern}'.")
    left, right = pair_parts[0].strip(), pair_parts[1].strip()
    if not left or not right:
        raise ValueError(f"Invalid wound pair in '{raw_pattern}'.")
    return _pair_key(left, right), location_id.strip()


def _load_wound_patterns_or_fail(path: Path) -> list[WoundPattern]:
    if not path.exists():
        raise FileNotFoundError(
            "Required wound baseline file missing: "
            f"{path}\nRun `python -m scripts.analyze_wounds --input scripts/output/rashomon_sweep_1_50.json "
            "--output scripts/output/wound_analysis_1_50.json` first."
        )

    raw = json.loads(path.read_text(encoding="utf-8"))
    candidates = list(raw.get("wound_candidates") or [])
    if not candidates:
        raise ValueError(f"Wound baseline file has no wound_candidates: {path}")

    out: list[WoundPattern] = []
    for row in candidates:
        pattern = str(row.get("pattern") or "").strip()
        if not pattern:
            continue
        pair, location = _parse_wound_pattern(pattern)
        out.append(
            WoundPattern(
                pattern=pattern,
                agent_pair=pair,
                location_id=location,
                frequency=float(row.get("frequency", 0.0) or 0.0),
            )
        )

    if not out:
        raise ValueError(f"No parseable wound patterns found in {path}")
    return out


def _beat_summary(beats: list[BeatType]) -> dict[str, int]:
    counts = Counter(beat.value for beat in beats)
    return {beat_name: int(counts.get(beat_name, 0)) for beat_name in BEAT_ORDER}


def _turning_point_fields(arc: RashomonArc) -> tuple[str | None, str | None, str | None]:
    for event, beat in zip(arc.events, arc.beats):
        if beat == BeatType.TURNING_POINT:
            return event.id, event.type.value, event.location_id
    return None, None, None


def _arc_payload(arc: RashomonArc) -> dict[str, Any]:
    tp_event_id, tp_event_type, tp_location = _turning_point_fields(arc)
    return {
        "protagonist": arc.protagonist,
        "valid": bool(arc.valid),
        "score": float(arc.arc_score.composite) if arc.arc_score else None,
        "event_count": int(len(arc.events)),
        "beat_summary": _beat_summary(arc.beats),
        "turning_point_event_id": tp_event_id,
        "turning_point_event_type": tp_event_type,
        "turning_point_location": tp_location,
        "violations": list(arc.violations),
    }


def _mean_overlap(overlap_matrix: dict[str, float]) -> float:
    if not overlap_matrix:
        return 0.0
    return sum(float(value) for value in overlap_matrix.values()) / float(len(overlap_matrix))


def _max_overlap_pair(overlap_matrix: dict[str, float]) -> str | None:
    if not overlap_matrix:
        return None
    return max(sorted(overlap_matrix.keys()), key=lambda key: float(overlap_matrix[key]))


def _count_inherited_beliefs(canon: WorldCanon | None) -> int:
    if canon is None:
        return 0
    total = 0
    for states_by_agent in canon.claim_states.values():
        for state in states_by_agent.values():
            if str(state) != "unknown":
                total += 1
    return total


def _location_residue(story_run: StoryRun) -> dict[str, float]:
    return {key: float(value) for key, value in sorted(story_run.start_location_memory.items())}


def _detect_wound_presence(rashomon_set: RashomonSet, wounds: list[WoundPattern]) -> dict[str, bool]:
    structural_hits: set[tuple[tuple[str, str], str]] = set()
    for arc in rashomon_set.arcs:
        if not arc.valid:
            continue
        for event, beat in zip(arc.events, arc.beats):
            if beat not in (BeatType.ESCALATION, BeatType.TURNING_POINT):
                continue
            target_head = event.target_agents[0] if event.target_agents else "(none)"
            pair = _pair_key(event.source_agent, target_head)
            structural_hits.add((pair, str(event.location_id)))

    return {
        wound.pattern: (wound.agent_pair, wound.location_id) in structural_hits
        for wound in wounds
    }


def _linear_slope(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    x_mean = (len(values) - 1) / 2.0
    y_mean = sum(values) / len(values)
    numerator = sum((i - x_mean) * (value - y_mean) for i, value in enumerate(values))
    denominator = sum((i - x_mean) ** 2 for i in range(len(values)))
    return numerator / denominator if denominator else 0.0


def _score_shape(values: list[float], eps: float = 1e-9) -> str:
    if len(values) < 2:
        return "mixed"
    deltas = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    has_pos = any(delta > eps for delta in deltas)
    has_neg = any(delta < -eps for delta in deltas)
    if not has_pos and not has_neg:
        return "flat"
    if has_pos and not has_neg:
        return "monotonic_up"

    first_last_delta = values[-1] - values[0]
    min_idx = min(range(len(values)), key=lambda idx: values[idx])
    if (
        has_pos
        and has_neg
        and first_last_delta > eps
        and 0 < min_idx < (len(values) - 1)
        and values[min_idx] < values[0]
        and values[min_idx] < values[-1]
    ):
        return "u_shaped"
    if has_pos and has_neg and first_last_delta > eps:
        return "noisy_up"
    return "mixed"


def _build_chain_summary(stories: list[dict[str, Any]], wall_time_seconds: float) -> dict[str, Any]:
    validity_progression = [int(story["rashomon"]["valid_count"]) for story in stories]
    mean_score_progression: list[float] = []
    overlap_progression: list[float] = []
    max_overlap_pairs: list[str | None] = []
    texture_before_progression: list[int] = []
    texture_after_progression: list[int] = []

    total_cost_usd = 0.0
    total_time_seconds = 0.0
    total_words = 0

    protagonist_score_trajectory: dict[str, list[float | None]] = {
        agent: [] for agent in DINNER_PARTY_AGENTS
    }

    for story in stories:
        arcs = list(story["rashomon"]["arcs"])
        valid_scores = [
            float(arc["score"])
            for arc in arcs
            if bool(arc["valid"]) and arc.get("score") is not None
        ]
        mean_score_progression.append(sum(valid_scores) / len(valid_scores) if valid_scores else 0.0)

        score_by_protagonist = {
            str(arc["protagonist"]): (float(arc["score"]) if arc.get("score") is not None else None)
            for arc in arcs
        }
        for agent in DINNER_PARTY_AGENTS:
            protagonist_score_trajectory[agent].append(score_by_protagonist.get(agent))

        overlap_matrix = {str(k): float(v) for k, v in story["rashomon"]["overlap_matrix"].items()}
        overlap_progression.append(_mean_overlap(overlap_matrix))
        max_overlap_pairs.append(_max_overlap_pair(overlap_matrix))

        texture_before_progression.append(int(story["canon_state"]["texture_facts_before"]))
        texture_after_progression.append(int(story["canon_state"]["texture_facts_after"]))

        narration = story.get("narration")
        if isinstance(narration, dict):
            total_cost_usd += float(narration.get("cost_usd", 0.0) or 0.0)
            total_time_seconds += float(narration.get("time_seconds", 0.0) or 0.0)
            total_words += int(narration.get("words", 0) or 0)

    score_deltas = [
        mean_score_progression[i + 1] - mean_score_progression[i]
        for i in range(len(mean_score_progression) - 1)
    ]
    overlap_deltas = [
        overlap_progression[i + 1] - overlap_progression[i]
        for i in range(len(overlap_progression) - 1)
    ]

    score_slope = _linear_slope(mean_score_progression)
    overlap_slope = _linear_slope(overlap_progression)
    score_first_last = mean_score_progression[-1] - mean_score_progression[0] if mean_score_progression else 0.0
    overlap_first_last = overlap_progression[-1] - overlap_progression[0] if overlap_progression else 0.0

    score_shape = _score_shape(mean_score_progression)
    scores_trending_up = bool(score_first_last > 0.0 and score_slope > 0.0)
    overlap_increasing = bool(overlap_first_last > 0.0 and overlap_slope > 0.0)
    valid_count_stable = bool(validity_progression and min(validity_progression) >= 5)

    texture_monotonic = all(
        texture_after_progression[i] <= texture_after_progression[i + 1]
        for i in range(len(texture_after_progression) - 1)
    )
    texture_continuity = all(
        texture_after_progression[i] == texture_before_progression[i + 1]
        for i in range(len(stories) - 1)
    )
    texture_compounding_clean = bool(texture_monotonic and texture_continuity)

    wound_keys = sorted(stories[0]["wound_presence"].keys()) if stories else []
    wound_persistence = {
        key: [bool(story["wound_presence"].get(key, False)) for story in stories]
        for key in wound_keys
    }
    new_wounds_emerging = any((not values[0]) and any(values[1:]) for values in wound_persistence.values())

    thesis_indicators = {
        "scores_trending_up": scores_trending_up,
        "overlap_increasing": overlap_increasing,
        "valid_count_stable": valid_count_stable,
        "texture_compounding_clean": texture_compounding_clean,
        "new_wounds_emerging": new_wounds_emerging,
        "score_deltas": score_deltas,
        "score_shape": score_shape,
        "score_first_last_delta": score_first_last,
        "score_slope": score_slope,
        "overlap_deltas": overlap_deltas,
        "overlap_first_last_delta": overlap_first_last,
        "overlap_slope": overlap_slope,
    }

    if scores_trending_up and valid_count_stable and texture_compounding_clean:
        verdict = "LANDSCAPE ENRICHING"
    elif (not scores_trending_up) and overlap_increasing and (not valid_count_stable):
        verdict = "LANDSCAPE FLATTENING"
    else:
        verdict = "LANDSCAPE MIXED"

    return {
        "total_cost_usd": total_cost_usd,
        "total_time_seconds": total_time_seconds,
        "total_words": total_words,
        "wall_time_seconds": wall_time_seconds,
        "validity_progression": validity_progression,
        "mean_score_progression": mean_score_progression,
        "protagonist_score_trajectory": protagonist_score_trajectory,
        "texture_accumulation": texture_before_progression,
        "texture_after_progression": texture_after_progression,
        "overlap_trend": {
            "mean_overlap": overlap_progression,
            "max_overlap_pair": max_overlap_pairs,
        },
        "wound_persistence": wound_persistence,
        "thesis_indicators": thesis_indicators,
        "verdict": verdict,
    }


def _format_score_sequence(values: list[float]) -> str:
    return " → ".join(f"{value:.2f}" for value in values)


def _format_int_sequence(values: list[int]) -> str:
    return ", ".join(str(value) for value in values)


def _print_story_header(index: int, story_label: str, seed: int, stories_done: int) -> None:
    if index == 0:
        context = "fresh"
    else:
        prior = "+".join(chr(ord("A") + i) for i in range(stories_done))
        context = f"canon from {prior}"
    print(f"\nStory {story_label} (seed {seed}, {context}):")


def _print_thesis(summary: dict[str, Any]) -> None:
    indicators = summary["thesis_indicators"]
    print("\n=== CHAIN COMPLETE ===")
    print("\nThesis Check:")
    print(
        "  Arc scores trending up?     "
        f"{'YES' if indicators['scores_trending_up'] else 'NO'} "
        f"({_format_score_sequence(summary['mean_score_progression'])})"
    )
    print(
        "  Score shape:                "
        f"{indicators['score_shape']} "
        f"(deltas: {', '.join(f'{delta:+.3f}' for delta in indicators['score_deltas'])})"
    )
    print(
        "  Valid count stable?         "
        f"{'YES' if indicators['valid_count_stable'] else 'NO'} "
        f"({_format_int_sequence(summary['validity_progression'])})"
    )
    overlap_trend = summary["overlap_trend"]["mean_overlap"]
    print(
        "  Overlap increasing?         "
        f"{'YES' if indicators['overlap_increasing'] else 'NO'} "
        f"({_format_score_sequence(overlap_trend)})"
    )
    texture_before = summary["texture_accumulation"]
    texture_after = summary["texture_after_progression"]
    print(
        "  Texture compounding clean?  "
        f"{'YES' if indicators['texture_compounding_clean'] else 'NO'} "
        f"({texture_before[0]} → {texture_after[0]} → {texture_after[1]} → "
        f"{texture_after[2]} → {texture_after[3]} → {texture_after[4]})"
    )
    print(
        "  New wounds emerging?        "
        f"{'YES' if indicators['new_wounds_emerging'] else 'NO'}"
    )
    print(f"\nVerdict: {summary['verdict']}")


def _run_chain(
    seeds: list[int],
    skip_narration: bool,
    output_path: Path,
    tension_decay: float,
    belief_decay: float,
) -> dict[str, Any]:
    wounds = _load_wound_patterns_or_fail(WOUND_ANALYSIS_PATH)

    print(f"=== RESEARCH CHAIN: 5 stories, seeds {seeds} ===")
    chain_start = time.monotonic()

    canon: WorldCanon | None = None
    stories: list[dict[str, Any]] = []

    for index, seed in enumerate(seeds):
        story_label = chr(ord("A") + index)
        _print_story_header(index=index, story_label=story_label, seed=seed, stories_done=index)

        loaded_canon = WorldCanon.from_dict(canon.to_dict()) if canon is not None else None
        inherited_beliefs = _count_inherited_beliefs(loaded_canon)
        texture_before = len(canon.texture) if canon is not None else 0

        story_run = run_story(label=f"Story {story_label}", seed=seed, loaded_canon=loaded_canon)
        location_count = len(story_run.start_location_memory)
        print(f"  Simulation: {len(story_run.events)} events, {location_count} locations")

        parsed = parse_simulation_output(story_run.payload)
        metrics_output = run_metrics_pipeline(parsed)
        total_sim_time = float(parsed.metadata.get("total_sim_time") or 0.0) or None
        rashomon = extract_rashomon_set(
            events=metrics_output.events,
            seed=seed,
            agents=list(DINNER_PARTY_AGENTS),
            total_sim_time=total_sim_time,
        )

        best_arc = next(
            (
                arc
                for arc in rashomon.arcs
                if arc.valid and arc.arc_score is not None
            ),
            rashomon.arcs[0],
        )
        best_label = (
            f"{best_arc.protagonist}({best_arc.arc_score.composite:.2f})"
            if best_arc.arc_score is not None
            else f"{best_arc.protagonist}(n/a)"
        )
        mean_overlap = _mean_overlap(rashomon.overlap_matrix)
        print(
            f"  Rashomon: {rashomon.valid_count}/{len(rashomon.arcs)} valid | "
            f"best={best_label} | mean_overlap={mean_overlap:.2f}"
        )

        if loaded_canon is not None:
            dining_tension = float(story_run.start_location_memory.get("dining_table", 0.0))
            print(
                f"  Canon state: {inherited_beliefs} inherited beliefs, "
                f"dining_table tension={dining_tension:.2f}"
            )

        # Preserve simulation canon updates while carrying forward prior texture.
        canon_next = WorldCanon.from_dict(story_run.payload.get("world_canon", WorldCanon().to_dict()))
        if canon is not None and canon.texture:
            canon_next.texture.update(dict(canon.texture))

        narration_payload: dict[str, Any] | None = None
        texture_committed = 0
        narr_error: str | None = None

        if skip_narration:
            print("  Narration: skipped (--skip-narration)")
        else:
            from scripts.demo_canon_stories import _narrate_story

            prose_path = OUTPUT_DIR / f"research_chain_story_{story_label.lower()}_prose.txt"
            meta_path = OUTPUT_DIR / f"research_chain_story_{story_label.lower()}_meta.json"
            narrator_texture_before = len(canon_next.texture)
            narr_result = _narrate_story(
                label=f"Story {story_label}",
                story_run=story_run,
                prose_path=prose_path,
                meta_path=meta_path,
                narrator_canon=canon_next,
            )
            narrator_texture_after = len(canon_next.texture)
            texture_committed = max(0, narrator_texture_after - narrator_texture_before)

            narr_error = narr_result.error
            narration_payload = {
                "status": "error" if narr_result.error else "ok",
                "protagonist": narr_result.protagonist,
                "words": int(narr_result.word_count),
                "scenes": int(narr_result.scene_count),
                "cost_usd": float(narr_result.cost_usd),
                "time_seconds": float(narr_result.generation_time_s),
                "arc_valid": bool(best_arc.valid),
                "error": narr_result.error,
                "prose_path": str(prose_path),
                "meta_path": str(meta_path),
            }
            if narr_result.error:
                print(f"  Narration: ERROR ({narr_result.error})")
            else:
                print(
                    f"  Narration: {narr_result.word_count:,} words, {narr_result.scene_count} scenes, "
                    f"${narr_result.cost_usd:.3f}, {narr_result.generation_time_s:.0f}s"
                )

        texture_after = len(canon_next.texture)
        if skip_narration:
            texture_committed = max(0, texture_after - texture_before)

        print(f"  Canon: {texture_before} → {texture_after} texture facts")

        story_record = {
            "story_label": story_label,
            "seed": int(seed),
            "chain_position": int(index),
            "simulation": {
                "event_count": int(len(story_run.events)),
                "location_count": int(location_count),
                "total_sim_time": float(parsed.metadata.get("total_sim_time", 0.0) or 0.0),
                "determinism_ok": bool(story_run.determinism_ok),
            },
            "rashomon": {
                "valid_count": int(rashomon.valid_count),
                "arcs": [_arc_payload(arc) for arc in rashomon.arcs],
                "overlap_matrix": {k: float(v) for k, v in rashomon.overlap_matrix.items()},
                "turning_point_overlap": rashomon.turning_point_overlap(),
                "mean_overlap": float(mean_overlap),
            },
            "canon_state": {
                "texture_facts_before": int(texture_before),
                "texture_facts_after": int(texture_after),
                "texture_committed_this_story": int(texture_committed),
                "inherited_beliefs": int(inherited_beliefs),
                "location_tension_residue": _location_residue(story_run),
            },
            "narration": narration_payload,
            "wound_presence": _detect_wound_presence(rashomon, wounds),
            "errors": {"narration": narr_error} if narr_error else {},
        }
        stories.append(story_record)

        if tension_decay < 1.0 or belief_decay < 1.0:
            decay_canon(canon_next, tension_decay=tension_decay, belief_decay=belief_decay)
        canon = canon_next

    chain_elapsed = time.monotonic() - chain_start
    chain_summary = _build_chain_summary(stories, wall_time_seconds=chain_elapsed)
    _print_thesis(chain_summary)

    output_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seeds": [int(seed) for seed in seeds],
        "config": {
            "skip_narration": bool(skip_narration),
            "tension_decay": float(tension_decay),
            "belief_decay": float(belief_decay),
            "wound_baseline_path": str(WOUND_ANALYSIS_PATH),
        },
        "stories": stories,
        "chain_summary": chain_summary,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"\nSaved research chain artifact: {output_path}")

    return output_payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a 5-story canon-accumulating Rashomon research chain."
    )
    parser.add_argument(
        "--seeds",
        required=True,
        type=str,
        help="Comma-separated list of 5 seeds (A through E).",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="Path to output JSON artifact.",
    )
    parser.add_argument(
        "--skip-narration",
        action="store_true",
        help="Run simulation+metrics+Rashomon only (no LLM narration or lore extraction).",
    )
    parser.add_argument(
        "--tension-decay",
        type=float,
        default=1.0,
        help="Tension residue decay rate per step (default: 1.0 = no decay).",
    )
    parser.add_argument(
        "--belief-decay",
        type=float,
        default=1.0,
        help="Belief confidence decay rate per step (default: 1.0 = no decay).",
    )
    args = parser.parse_args()

    try:
        seeds = _parse_seeds(args.seeds)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2

    if not args.skip_narration and not _required_api_keys_present():
        return 1

    if not (0.0 <= float(args.tension_decay) <= 1.0):
        print(f"ERROR: --tension-decay must be in [0.0, 1.0], got {args.tension_decay}")
        return 2
    if not (0.0 <= float(args.belief_decay) <= 1.0):
        print(f"ERROR: --belief-decay must be in [0.0, 1.0], got {args.belief_decay}")
        return 2

    output_path = _resolve_output_path(args.output)
    try:
        _run_chain(
            seeds=seeds,
            skip_narration=bool(args.skip_narration),
            output_path=output_path,
            tension_decay=float(args.tension_decay),
            belief_decay=float(args.belief_decay),
        )
    except Exception as exc:  # pylint: disable=broad-except
        print(f"ERROR: {type(exc).__name__}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
