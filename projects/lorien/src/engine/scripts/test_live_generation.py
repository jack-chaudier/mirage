"""Live test script: simulate → metrics → arc search → scenes → narrator → output.

Run with a real LLM key set (ANTHROPIC_API_KEY) for live generation,
or without one to verify the full pipeline wires up correctly (will error
at the LLM call stage, which is expected).

Usage:
    cd src/engine && source .venv/bin/activate
    python -m scripts.test_live_generation          # dry-run (no LLM)
    python -m scripts.test_live_generation --live    # real LLM calls
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from random import Random

from narrativefield.extraction.arc_search import search_arc
from narrativefield.llm.config import PipelineConfig
from narrativefield.metrics.pipeline import (
    MetricsPipelineOutput,
    ParsedSimulationOutput,
    run_metrics_pipeline,
)
from narrativefield.simulation.scenarios.dinner_party import create_dinner_party_world
from narrativefield.simulation.tick_loop import SimulationConfig, run_simulation
from narrativefield.storyteller.narrator import SequentialNarrator


def main(live: bool = False, seed: int = 42, strict: bool = False) -> None:
    print(f"=== NarrativeField Live Test (seed={seed}, live={live}) ===\n")

    # ------------------------------------------------------------------
    # Step 1: Simulate
    # ------------------------------------------------------------------
    print("[1/6] Running dinner party simulation...")
    t0 = time.monotonic()
    world = create_dinner_party_world()
    rng = Random(seed)
    cfg = SimulationConfig(
        tick_limit=300,
        event_limit=200,
        max_sim_time=150.0,
        snapshot_interval_events=20,
    )
    events, snapshots = run_simulation(world, rng, cfg)
    print(f"      {len(events)} events in {time.monotonic() - t0:.2f}s\n")

    # ------------------------------------------------------------------
    # Step 2: Metrics pipeline
    # ------------------------------------------------------------------
    print("[2/6] Running metrics pipeline...")
    t0 = time.monotonic()
    parsed = ParsedSimulationOutput(
        format_version="1.0",
        metadata={"total_sim_time": 150.0},
        initial_state={},
        snapshots=snapshots,
        events=events,
        secrets=world.definition.secrets,
        locations=world.definition.locations,
        initial_agents=world.agents,
    )
    metrics_out: MetricsPipelineOutput = run_metrics_pipeline(parsed)
    enriched_events = metrics_out.events
    scenes = metrics_out.scenes
    print(f"      {len(enriched_events)} enriched events, {len(scenes)} scenes in {time.monotonic() - t0:.2f}s\n")

    # ------------------------------------------------------------------
    # Step 3: Arc search
    # ------------------------------------------------------------------
    print("[3/6] Searching for best arc (protagonist=victor)...")
    t0 = time.monotonic()
    arc = search_arc(
        all_events=enriched_events,
        protagonist="victor",
        max_events=20,
        total_sim_time=150.0,
    )
    print(f"      Arc: {len(arc.events)} events, protagonist={arc.protagonist}")
    print(f"      Valid={arc.validation.valid}")
    if arc.arc_score:
        print(f"      Score={arc.arc_score.composite:.3f}")
    print(f"      Search took {time.monotonic() - t0:.2f}s\n")

    if not arc.events:
        print("ERROR: Arc search returned no events. Cannot proceed.")
        sys.exit(1)

    # Set beat types on arc events.
    for ev, bt in zip(arc.events, arc.beats):
        ev.beat_type = bt

    # ------------------------------------------------------------------
    # Step 4: Prepare world_data for lorebook
    # ------------------------------------------------------------------
    print("[4/6] Building world_data for Lorebook...")
    world_data = {
        "world_definition": world.definition,
        "agents": list(world.agents.values()),
        "secrets": list(world.definition.secrets.values()),
    }
    print(f"      {len(world_data['agents'])} agents, {len(world_data['secrets'])} secrets\n")

    # ------------------------------------------------------------------
    # Step 5: Generate prose
    # ------------------------------------------------------------------
    print("[5/6] Running narrator pipeline...")
    config = PipelineConfig(checkpoint_enabled=False)
    narrator = SequentialNarrator(config=config)

    if not live:
        # Dry-run: mock the gateway to avoid real API calls.
        from narrativefield.llm.gateway import ModelTier

        async def _mock_generate(tier, system_prompt, user_prompt, **kwargs):
            if tier is ModelTier.STRUCTURAL:
                # Used for both summary compression and continuity checking.
                if "continuity" in system_prompt.lower() or "continuity" in user_prompt.lower():
                    return '{"consistent": true, "violations": []}'
                return "Summary: Victor investigated Marcus at the dinner party."
            return (
                "<prose>\n"
                "Victor studied the room. The candlelight played across faces "
                "that revealed nothing and everything at once.\n"
                "</prose>\n"
                "<state_update>\n"
                "<summary>Victor observed the dinner guests.</summary>\n"
                "</state_update>"
            )

        narrator.gateway.generate = _mock_generate  # type: ignore[assignment]

    t0 = time.monotonic()
    result = asyncio.run(
        narrator.generate(
            events=arc.events,
            world_data=world_data,
            run_id=f"live_test_seed{seed}",
        )
    )
    elapsed = time.monotonic() - t0
    print(f"      {result.word_count} words, {result.scenes_generated} scenes in {elapsed:.2f}s\n")

    # ------------------------------------------------------------------
    # Step 6: Output
    # ------------------------------------------------------------------
    print("[6/6] Results:\n")

    scenes_total = len(result.scene_outcomes) if result.scene_outcomes else int(result.scenes_generated)
    scenes_ok = sum(1 for s in result.scene_outcomes if s.status == "ok")
    scenes_failed = sum(1 for s in result.scene_outcomes if s.status == "failed")
    first_failed = next((s for s in result.scene_outcomes if s.status == "failed"), None)

    status_label = str(result.status).upper()
    if result.status == "complete":
        status_detail = f"({scenes_ok}/{scenes_total} scenes)"
    elif result.status == "partial":
        extra = ""
        if first_failed is not None:
            extra = f" - scene {first_failed.scene_index} failed: {first_failed.error_type or 'unknown_error'}"
        status_detail = f"({scenes_ok}/{scenes_total} scenes{extra})"
    else:
        extra = ""
        if first_failed is not None:
            extra = f" - scene {first_failed.scene_index} failed: {first_failed.error_type or 'unknown_error'}"
        status_detail = f"({scenes_ok}/{scenes_total} scenes{extra})"

    print(f"  Status:           {status_label} {status_detail}")
    print(f"  Word count:       {result.word_count}")
    print(f"  Scenes generated: {result.scenes_generated}")
    print(f"  Generation time:  {result.generation_time_seconds:.2f}s")
    usage = result.usage
    print(f"  Input tokens:     {usage.get('total_input_tokens', 0)}")
    print(f"  Output tokens:    {usage.get('total_output_tokens', 0)}")
    print(f"  Est. cost:        ${usage.get('estimated_cost_usd', 0.0):.4f}")
    print()

    # Write run artifacts.
    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    meta_file = output_dir / f"seed_{seed}_{ts}_meta.json"
    meta = {
        "seed": int(seed),
        "status": str(result.status),
        "word_count": int(result.word_count),
        "scenes_total": int(scenes_total),
        "scenes_ok": int(scenes_ok),
        "scenes_failed": int(scenes_failed),
        "scene_outcomes": [
            {
                "scene_index": int(s.scene_index),
                "status": str(s.status),
                "word_count": int(s.word_count),
                "error_type": s.error_type,
                "retries": int(s.retries),
                "generation_time_s": float(s.generation_time_s),
            }
            for s in (result.scene_outcomes or [])
        ],
        "total_generation_time_s": float(result.generation_time_seconds),
        "input_tokens": int(usage.get("total_input_tokens", 0) or 0),
        "output_tokens": int(usage.get("total_output_tokens", 0) or 0),
        "est_cost": float(usage.get("estimated_cost_usd", 0.0) or 0.0),
        "repetitions_removed": int(usage.get("repetitions_removed", 0) or 0),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    meta_file.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if result.status == "failed":
        # Don't write empty story files; emit an ERROR report instead.
        error_file = output_dir / f"seed_{seed}_{ts}_ERROR.txt"
        lines: list[str] = []
        lines.append("GENERATION FAILED")
        lines.append(f"Seed: {seed}")
        lines.append(f"Status: {result.status}")
        lines.append(f"Word count: {result.word_count}")
        lines.append(f"Scenes: {scenes_ok}/{scenes_total} ok, {scenes_failed} failed")
        lines.append("")
        for s in result.scene_outcomes:
            if s.status != "ok":
                lines.append(
                    f"Scene {s.scene_index}: {s.status} ({s.word_count} words, retries={s.retries}, "
                    f"time_s={s.generation_time_s:.2f}, error_type={s.error_type or 'unknown_error'})"
                )
        lines.append("")
        lines.append(f"Meta: {meta_file}")
        error_file.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        print(f"  GENERATION FAILED: {error_file}")
        print(f"  Meta saved to:     {meta_file}")
    else:
        story_file = output_dir / f"seed_{seed}_{ts}.txt"
        story_text = result.prose or ""
        if not story_text.endswith("\n"):
            story_text += "\n"
        story_file.write_text(story_text, encoding="utf-8")
        print(f"  Story saved to:   {story_file}")
        print(f"  Meta saved to:    {meta_file}")

    # Print first 500 chars of prose.
    if result.prose:
        preview = result.prose[:500]
        if len(result.prose) > 500:
            preview += "..."
        print("\n--- Prose Preview ---")
        print(preview)
        print("--- End Preview ---\n")
    else:
        print("\n--- Prose Preview ---")
        print("(no prose)")
        print("--- End Preview ---\n")

    print("=== Test complete ===")
    if strict and result.status != "complete":
        sys.exit(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NarrativeField live test")
    parser.add_argument("--live", action="store_true", help="Use real LLM calls")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if generation is not COMPLETE",
    )
    args = parser.parse_args()
    main(live=args.live, seed=args.seed, strict=args.strict)
