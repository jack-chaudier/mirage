"""Demonstrate a three-story lore loop chain: A → B → C with texture compounding.

This script proves multi-story texture accumulation by:
1) Running Story A (seed 42, fresh) → narrate → collect texture → save canon.
2) Running Story B (seed 51, A's canon+texture) → narrate → collect texture → save canon.
3) Running Story C (seed 7, B's accumulated canon+texture) → narrate → observe compounding.
4) Writing a summary and per-story meta JSONs to output/ and examples/.

Run:
    cd src/engine && source .env && python -m scripts.demo_lore_loop
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from narrativefield.schema.canon import WorldCanon
from scripts.demo_canon_persistence import STORY_A_SEED, STORY_B_SEED, run_story
from scripts.demo_canon_stories import (
    ProseRunResult,
    _narrate_story,
    _required_api_keys_present,
    _save_text,
)

STORY_C_SEED = 7

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES_DIR = REPO_ROOT / "examples"

STORY_A_PROSE = OUTPUT_DIR / "lore_loop_story_a_prose.txt"
STORY_A_META = OUTPUT_DIR / "lore_loop_story_a_meta.json"
STORY_B_PROSE = OUTPUT_DIR / "lore_loop_story_b_prose.txt"
STORY_B_META = OUTPUT_DIR / "lore_loop_story_b_meta.json"
STORY_C_PROSE = OUTPUT_DIR / "lore_loop_story_c_prose.txt"
STORY_C_META = OUTPUT_DIR / "lore_loop_story_c_meta.json"
CHAIN_SUMMARY = OUTPUT_DIR / "lore_loop_chain_summary.txt"

EXAMPLE_CHAIN_SUMMARY = EXAMPLES_DIR / "lore_loop_chain_summary.txt"
EXAMPLE_STORY_C_PROSE = EXAMPLES_DIR / "lore_loop_story_c_seed7.txt"
EXAMPLE_STORY_C_META = EXAMPLES_DIR / "lore_loop_story_c_seed7_meta.json"


def _build_chain_summary(
    results: list[ProseRunResult],
    total_time_s: float,
) -> str:
    lines = [
        "=== LORE LOOP CHAIN: A → B → C ===",
        "",
        "Texture compounding across three stories:",
        "",
    ]

    total_cost = 0.0
    for r in results:
        total_cost += r.cost_usd
        lines.extend([
            f"{r.label} (seed {r.seed})",
            "-" * (len(f"{r.label} (seed {r.seed})")),
            f"  Protagonist: {r.protagonist}",
            f"  Words: {r.word_count}",
            f"  Scenes: {r.scene_count}",
            f"  Arc valid: {r.arc_validation_valid}",
            f"  Arc used fallback: {r.arc_used_fallback}",
            f"  Cost: ${r.cost_usd:.4f}",
            f"  Time: {r.generation_time_s:.1f}s",
            f"  Starting texture count: {r.starting_canon_texture_count}",
            f"  Ending texture count: {r.ending_canon_texture_count}",
            f"  Texture committed this story: {r.texture_committed_count}",
            f"  Lore: scenes={r.lore_scene_count}, canon_facts={r.lore_canon_fact_count}, "
            f"texture_facts={r.lore_texture_fact_count}",
            "",
        ])

    lines.extend([
        "Texture accumulation summary:",
    ])
    for r in results:
        lines.append(
            f"  {r.label}: {r.starting_canon_texture_count} → "
            f"{r.ending_canon_texture_count} texture facts"
        )

    monotonic = all(
        results[i].ending_canon_texture_count <= results[i + 1].starting_canon_texture_count
        for i in range(len(results) - 1)
    )
    compounding = (
        len(results) >= 3
        and results[-1].starting_canon_texture_count > results[0].ending_canon_texture_count
    )

    lines.extend([
        "",
        f"Monotonic texture growth: {'YES' if monotonic else 'NO'}",
        f"Cross-story compounding (C inherits > A produced): {'YES' if compounding else 'NO'}",
        "",
        f"Total chain cost: ${total_cost:.4f}",
        f"Total chain time: {total_time_s:.1f}s",
    ])

    return "\n".join(lines).rstrip() + "\n"


def _copy_artifacts() -> None:
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    pairs = [
        (CHAIN_SUMMARY, EXAMPLE_CHAIN_SUMMARY),
        (STORY_C_PROSE, EXAMPLE_STORY_C_PROSE),
        (STORY_C_META, EXAMPLE_STORY_C_META),
    ]
    for src, dst in pairs:
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def main() -> int:
    print("=== Lore Loop Chain Demo: A → B → C ===")

    if not _required_api_keys_present():
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    chain_start = time.monotonic()

    # --- Story A (seed 42, fresh) ---
    print("\n[1/6] Running Story A simulation (seed 42, fresh)...")
    story_a_run = run_story("Story A", STORY_A_SEED, loaded_canon=None)
    print(f"  Story A sim complete: {len(story_a_run.events)} events")

    print("\n[2/6] Narrating Story A...")
    canon_a = WorldCanon()
    story_a_result = _narrate_story(
        label="Story A",
        story_run=story_a_run,
        prose_path=STORY_A_PROSE,
        meta_path=STORY_A_META,
        narrator_canon=canon_a,
    )
    print(
        f"  Story A narrated: {story_a_result.word_count} words, "
        f"texture {story_a_result.starting_canon_texture_count} → "
        f"{story_a_result.ending_canon_texture_count}"
    )

    # --- Story B (seed 51, loaded with A's canon + texture) ---
    print("\n[3/6] Running Story B simulation (seed 51, with Story A canon)...")
    sim_canon_b = WorldCanon.from_dict(
        story_a_run.payload.get("world_canon", WorldCanon().to_dict())
    )
    story_b_run = run_story("Story B", STORY_B_SEED, loaded_canon=sim_canon_b)
    print(f"  Story B sim complete: {len(story_b_run.events)} events")

    print("\n[4/6] Narrating Story B...")
    canon_b = WorldCanon.from_dict(sim_canon_b.to_dict())
    canon_b.texture.update(dict(canon_a.texture))
    story_b_result = _narrate_story(
        label="Story B",
        story_run=story_b_run,
        prose_path=STORY_B_PROSE,
        meta_path=STORY_B_META,
        narrator_canon=canon_b,
    )
    print(
        f"  Story B narrated: {story_b_result.word_count} words, "
        f"texture {story_b_result.starting_canon_texture_count} → "
        f"{story_b_result.ending_canon_texture_count}"
    )

    # --- Story C (seed 7, loaded with B's accumulated canon + texture) ---
    print("\n[5/6] Running Story C simulation (seed 7, with Story B canon)...")
    sim_canon_c = WorldCanon.from_dict(
        story_b_run.payload.get("world_canon", WorldCanon().to_dict())
    )
    story_c_run = run_story("Story C", STORY_C_SEED, loaded_canon=sim_canon_c)
    print(f"  Story C sim complete: {len(story_c_run.events)} events")

    print("\n[6/6] Narrating Story C...")
    canon_c = WorldCanon.from_dict(sim_canon_c.to_dict())
    canon_c.texture.update(dict(canon_b.texture))
    story_c_result = _narrate_story(
        label="Story C",
        story_run=story_c_run,
        prose_path=STORY_C_PROSE,
        meta_path=STORY_C_META,
        narrator_canon=canon_c,
    )
    print(
        f"  Story C narrated: {story_c_result.word_count} words, "
        f"texture {story_c_result.starting_canon_texture_count} → "
        f"{story_c_result.ending_canon_texture_count}"
    )

    chain_elapsed = time.monotonic() - chain_start

    # --- Summary ---
    results = [story_a_result, story_b_result, story_c_result]
    summary = _build_chain_summary(results, chain_elapsed)
    _save_text(CHAIN_SUMMARY, summary)
    print(f"\n{summary}")

    _copy_artifacts()

    print("Artifacts written:")
    for path in [STORY_A_PROSE, STORY_A_META, STORY_B_PROSE, STORY_B_META,
                 STORY_C_PROSE, STORY_C_META, CHAIN_SUMMARY]:
        print(f"  {path}")
    print("Copied to examples/:")
    print(f"  {EXAMPLE_CHAIN_SUMMARY}")
    print(f"  {EXAMPLE_STORY_C_PROSE}")
    print(f"  {EXAMPLE_STORY_C_META}")

    total_cost = sum(r.cost_usd for r in results)
    print(f"\nTotal chain cost: ${total_cost:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
