"""Generate side-by-side prose for fresh vs canon-loaded Story B runs.

This demo proves storyworld memory carry-over by:
1) Running Story A (seed 42, no canon) to produce canon state.
2) Running Story B Fresh (seed 51, no canon).
3) Running Story B Canon (seed 51, loaded with Story A canon).
4) Generating prose for Story A (to seed lore texture canon), Story B Fresh, and Story B Canon via:
   parse_simulation_output -> run_metrics_pipeline -> search_arc -> SequentialNarrator.generate
5) Writing Story B prose/meta artifacts and a comparison summary, then copying outputs to examples/.

Run:
    cd src/engine && python -m scripts.demo_canon_stories
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from narrativefield.extraction.arc_search import fallback_event_sort_key, search_arc
from narrativefield.extraction.beat_classifier import classify_beats
from narrativefield.llm.config import PipelineConfig
from narrativefield.metrics.pipeline import ParsedSimulationOutput, parse_simulation_output, run_metrics_pipeline
from narrativefield.schema.canon import WorldCanon
from narrativefield.schema.events import BeatType, Event
from narrativefield.simulation.scenarios.dinner_party import create_dinner_party_world
from narrativefield.storyteller.narrator import SequentialNarrator
from scripts.demo_canon_persistence import STORY_A_SEED, STORY_B_SEED, StoryRun, run_story

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES_DIR = REPO_ROOT / "examples"

STORY_B_FRESH_PROSE = OUTPUT_DIR / "story_b_fresh_prose.txt"
STORY_B_CANON_PROSE = OUTPUT_DIR / "story_b_canon_prose.txt"
STORY_B_FRESH_META = OUTPUT_DIR / "story_b_fresh_meta.json"
STORY_B_CANON_META = OUTPUT_DIR / "story_b_canon_meta.json"
COMPARISON_SUMMARY = OUTPUT_DIR / "canon_story_comparison.txt"

EXAMPLE_STORY_B_FRESH_PROSE = EXAMPLES_DIR / "story_b_fresh_seed51.txt"
EXAMPLE_STORY_B_CANON_PROSE = EXAMPLES_DIR / "story_b_canon_seed51.txt"
EXAMPLE_STORY_B_FRESH_META = EXAMPLES_DIR / "story_b_fresh_seed51_meta.json"
EXAMPLE_STORY_B_CANON_META = EXAMPLES_DIR / "story_b_canon_seed51_meta.json"
EXAMPLE_COMPARISON_SUMMARY = EXAMPLES_DIR / "canon_story_comparison.txt"

FALLBACK_ARC_EVENT_LIMIT = 15
STORY_A_PROSE = OUTPUT_DIR / "story_a_prose.txt"
STORY_A_META = OUTPUT_DIR / "story_a_meta.json"


@dataclass(slots=True)
class ProseRunResult:
    label: str
    seed: int
    protagonist: str
    status: str
    word_count: int
    scene_count: int
    cost_usd: float
    generation_time_s: float
    model_used: str
    model_history: list[dict[str, Any]]
    prose: str
    opening_line: str
    arc_used_fallback: bool
    arc_validation_valid: bool
    arc_event_count: int
    arc_diagnostics: dict[str, Any] | None
    fallback_warning: str | None
    lore_scene_count: int
    lore_canon_fact_count: int
    lore_texture_fact_count: int
    texture_committed_count: int
    starting_canon_texture_count: int
    ending_canon_texture_count: int
    error: str | None
    prose_path: Path
    meta_path: Path


def _save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not text.endswith("\n"):
        text += "\n"
    path.write_text(text, encoding="utf-8")


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _required_api_keys_present() -> bool:
    required = ("ANTHROPIC_API_KEY", "XAI_API_KEY")
    missing = [key for key in required if not os.getenv(key)]
    if not missing:
        return True

    print("ERROR: Missing required API keys for live canon story generation.")
    print("Missing:")
    for key in missing:
        print(f"  - {key}")
    print()
    print("Set both keys, then run:")
    print("  cd src/engine")
    print("  ANTHROPIC_API_KEY=... XAI_API_KEY=... python -m scripts.demo_canon_stories")
    print("Or if keys are already exported:")
    print("  cd src/engine && python -m scripts.demo_canon_stories")
    return False


def _set_event_beats(events: list[Event], beats: list[BeatType]) -> None:
    for event, beat in zip(events, beats):
        event.beat_type = beat


def _first_agent(events: list[Event]) -> str:
    if not events:
        return "unknown"
    first = events[0]
    if first.source_agent:
        return first.source_agent
    if first.target_agents:
        return first.target_agents[0]
    return "unknown"


def _opening_line(prose: str, max_chars: int = 100) -> str:
    normalized = " ".join(prose.split())
    if not normalized:
        return "(none)"
    if len(normalized) <= max_chars:
        return normalized
    return normalized[:max_chars] + "..."


def _safe_total_sim_time(parsed: ParsedSimulationOutput, events: list[Event]) -> float:
    total = float(parsed.metadata.get("total_sim_time") or 0.0)
    if total > 0.0:
        return total
    return max((float(event.sim_time) for event in events), default=150.0)


def _model_used_from_usage(usage: dict[str, Any]) -> str:
    llm_meta = usage.get("llm_response_metadata")
    if isinstance(llm_meta, dict):
        model = llm_meta.get("model_used")
        if isinstance(model, str) and model.strip():
            return model

    model_history = usage.get("model_history")
    if isinstance(model_history, list):
        for item in reversed(model_history):
            if isinstance(item, dict):
                model = item.get("model")
                if isinstance(model, str) and model.strip():
                    return model
    return "unknown"


def _build_world_data(parsed: ParsedSimulationOutput) -> dict[str, Any]:
    # Keep the same lorebook wiring shape used by API server paths:
    # world_definition + agents + secrets.
    scenario_world = create_dinner_party_world()
    agents = list(parsed.initial_agents.values()) or list(scenario_world.agents.values())
    secrets = list(parsed.secrets.values()) or list(scenario_world.definition.secrets.values())

    return {
        "world_definition": scenario_world.definition,
        "agents": agents,
        "secrets": secrets,
    }


def _run_arc_search(
    label: str,
    events: list[Event],
    total_sim_time: float,
) -> tuple[str, list[Event], bool, bool, int, dict[str, Any] | None]:
    search_result = search_arc(all_events=events, total_sim_time=total_sim_time)
    is_valid = bool(search_result.validation.valid)
    diagnostics_payload = search_result.diagnostics.to_dict() if search_result.diagnostics else None

    if is_valid and search_result.events:
        _set_event_beats(search_result.events, search_result.beats)
        return (
            search_result.protagonist or _first_agent(search_result.events),
            search_result.events,
            False,
            True,
            len(search_result.events),
            diagnostics_payload,
        )

    print(
        f"WARNING: {label} arc search returned no valid arc "
        f"(using top {FALLBACK_ARC_EVENT_LIMIT} events by significance as fallback)."
    )

    ranked = sorted(events, key=lambda event: fallback_event_sort_key(event, search_result.protagonist or None))
    selected = ranked[:FALLBACK_ARC_EVENT_LIMIT]
    fallback_events = sorted(
        selected,
        key=lambda event: (
            float(event.sim_time),
            int(event.tick_id),
            int(event.order_in_tick),
            str(event.id),
        ),
    )

    if diagnostics_payload:
        candidates_evaluated = int(diagnostics_payload.get("candidates_evaluated", 0) or 0)
        best_violation_count = int(diagnostics_payload.get("best_candidate_violation_count", 0) or 0)
        primary_failure = str(diagnostics_payload.get("primary_failure", "") or "(none)")
        rule_counts_raw = diagnostics_payload.get("rule_failure_counts", {})
        rule_counts = rule_counts_raw if isinstance(rule_counts_raw, dict) else {}
        ranked_rules = sorted(
            ((str(rule), int(count)) for rule, count in rule_counts.items()),
            key=lambda item: (-item[1], item[0]),
        )
        rule_summary = ", ".join(f"{rule}={count}" for rule, count in ranked_rules) if ranked_rules else "(none)"
        print(f"⚠️  Arc search failed for {label.lower()}")
        print(f"    Candidates evaluated: {candidates_evaluated}")
        print(f"    Nearest miss: {best_violation_count} violation(s)")
        print(f"    Primary failure: {primary_failure}")
        print(f"    Rule frequency: {rule_summary}")

    print("        fallback events selected:")
    for event in fallback_events:
        significance = float(getattr(event.metrics, "significance", 0.0) or 0.0)
        print(
            f"          - {event.id}: type={event.type.value}, "
            f"sig={significance:.3f}, time={event.sim_time:.2f}, loc={event.location_id}"
        )

    fallback_beats = classify_beats(fallback_events)
    _set_event_beats(fallback_events, fallback_beats)

    return (
        search_result.protagonist or _first_agent(fallback_events),
        fallback_events,
        True,
        False,
        len(fallback_events),
        diagnostics_payload,
    )


def _narrate_story(
    label: str,
    story_run: StoryRun,
    prose_path: Path,
    meta_path: Path,
    narrator_canon: WorldCanon | None = None,
) -> ProseRunResult:
    print(f"\n--- {label} narration pipeline ---")

    try:
        print("  [1/4] Parsing simulation + metrics...")
        t0 = time.monotonic()
        parsed = parse_simulation_output(story_run.payload)
        metrics_output = run_metrics_pipeline(parsed)
        metrics_elapsed = time.monotonic() - t0
        print(
            f"        done: {len(metrics_output.events)} scored events, "
            f"{len(metrics_output.scenes)} scenes ({metrics_elapsed:.2f}s)"
        )

        total_sim_time = _safe_total_sim_time(parsed, metrics_output.events)

        print("  [2/4] Searching best arc...")
        t0 = time.monotonic()
        protagonist, events_for_narration, arc_used_fallback, arc_valid, arc_event_count, arc_diagnostics = _run_arc_search(
            label, metrics_output.events, total_sim_time
        )
        fallback_warning = None
        if arc_used_fallback:
            fallback_warning = (
                f"{label} used arc fallback events; lore-loop comparison may be less representative "
                "because this run narrates a ranked vignette instead of a validated arc."
            )
            print(f"        WARNING: {fallback_warning}")
        arc_elapsed = time.monotonic() - t0
        print(
            f"        done: protagonist={protagonist}, events={arc_event_count}, "
            f"fallback={arc_used_fallback} ({arc_elapsed:.2f}s)"
        )

        print("  [3/4] Building lorebook world_data...")
        world_data = _build_world_data(parsed)
        print(
            f"        done: {len(world_data['agents'])} agents, "
            f"{len(world_data['secrets'])} secrets"
        )

        print("  [4/4] Generating prose (live LLM calls; this can take a few minutes)...")
        config = PipelineConfig(checkpoint_enabled=False)
        narrator = SequentialNarrator(config=config)
        run_id = f"canon_story_{label.lower().replace(' ', '_')}_seed{story_run.seed}"
        starting_texture_count = len(narrator_canon.texture) if narrator_canon is not None else 0

        result = asyncio.run(
            narrator.generate(
                events=events_for_narration,
                world_data=world_data,
                run_id=run_id,
                canon=narrator_canon,
            )
        )

        prose = result.prose or ""
        if prose:
            _save_text(prose_path, prose)
        else:
            _save_text(
                prose_path,
                f"[NO_PROSE] Generation status={result.status} for {label}.",
            )

        usage = result.usage or {}
        model_history_raw = usage.get("model_history")
        model_history = model_history_raw if isinstance(model_history_raw, list) else []
        texture_committed = usage.get("texture_committed_keys")
        texture_committed_keys = [str(x) for x in texture_committed] if isinstance(texture_committed, list) else []
        story_lore = result.story_lore
        lore_scene_count = len(story_lore.scene_lore) if story_lore is not None else 0
        lore_canon_fact_count = len(story_lore.all_canon_facts) if story_lore is not None else 0
        lore_texture_fact_count = len(story_lore.all_texture_facts) if story_lore is not None else 0
        ending_texture_count = len(narrator_canon.texture) if narrator_canon is not None else 0

        meta = {
            "label": label,
            "seed": int(story_run.seed),
            "status": str(result.status),
            "protagonist": protagonist,
            "word_count": int(result.word_count),
            "scene_count": int(result.scenes_generated),
            "generation_time_seconds": float(result.generation_time_seconds),
            "estimated_cost_usd": float(usage.get("estimated_cost_usd", 0.0) or 0.0),
            "model_used": _model_used_from_usage(usage),
            "input_tokens": int(usage.get("total_input_tokens", 0) or 0),
            "output_tokens": int(usage.get("total_output_tokens", 0) or 0),
            "cache_read_tokens": int(usage.get("cache_read_tokens", 0) or 0),
            "cache_write_tokens": int(usage.get("cache_write_tokens", 0) or 0),
            "model_history": model_history,
            "arc_used_fallback": bool(arc_used_fallback),
            "arc_validation_valid": bool(arc_valid),
            "arc_event_count": int(arc_event_count),
            "arc_diagnostics": arc_diagnostics,
            "fallback_warning": fallback_warning,
            "lore_scene_count": int(lore_scene_count),
            "lore_canon_fact_count": int(lore_canon_fact_count),
            "lore_texture_fact_count": int(lore_texture_fact_count),
            "texture_committed_count": int(len(texture_committed_keys)),
            "texture_committed_keys": texture_committed_keys,
            "starting_canon_texture_count": int(starting_texture_count),
            "ending_canon_texture_count": int(ending_texture_count),
            "error": None,
        }
        _save_json(meta_path, meta)

        print(
            f"        done: {result.word_count} words, {result.scenes_generated} scenes, "
            f"cost=${meta['estimated_cost_usd']:.4f}, time={result.generation_time_seconds:.2f}s"
        )

        return ProseRunResult(
            label=label,
            seed=int(story_run.seed),
            protagonist=protagonist,
            status=str(result.status),
            word_count=int(result.word_count),
            scene_count=int(result.scenes_generated),
            cost_usd=float(meta["estimated_cost_usd"]),
            generation_time_s=float(result.generation_time_seconds),
            model_used=str(meta["model_used"]),
            model_history=model_history,
            prose=prose,
            opening_line=_opening_line(prose),
            arc_used_fallback=bool(arc_used_fallback),
            arc_validation_valid=bool(arc_valid),
            arc_event_count=int(arc_event_count),
            arc_diagnostics=arc_diagnostics,
            fallback_warning=fallback_warning,
            lore_scene_count=int(lore_scene_count),
            lore_canon_fact_count=int(lore_canon_fact_count),
            lore_texture_fact_count=int(lore_texture_fact_count),
            texture_committed_count=int(len(texture_committed_keys)),
            starting_canon_texture_count=int(starting_texture_count),
            ending_canon_texture_count=int(ending_texture_count),
            error=None,
            prose_path=prose_path,
            meta_path=meta_path,
        )

    except Exception as exc:  # pylint: disable=broad-except
        error = f"{type(exc).__name__}: {exc}"
        print(f"        FAILED: {error}")

        _save_text(
            prose_path,
            f"[GENERATION_FAILED] {label}\n{error}",
        )

        meta = {
            "label": label,
            "seed": int(story_run.seed),
            "status": "error",
            "protagonist": "unknown",
            "word_count": 0,
            "scene_count": 0,
            "generation_time_seconds": 0.0,
            "estimated_cost_usd": 0.0,
            "model_used": "unknown",
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "model_history": [],
            "arc_used_fallback": False,
            "arc_validation_valid": False,
            "arc_event_count": 0,
            "arc_diagnostics": None,
            "fallback_warning": None,
            "lore_scene_count": 0,
            "lore_canon_fact_count": 0,
            "lore_texture_fact_count": 0,
            "texture_committed_count": 0,
            "texture_committed_keys": [],
            "starting_canon_texture_count": 0,
            "ending_canon_texture_count": 0,
            "error": error,
        }
        _save_json(meta_path, meta)

        return ProseRunResult(
            label=label,
            seed=int(story_run.seed),
            protagonist="unknown",
            status="error",
            word_count=0,
            scene_count=0,
            cost_usd=0.0,
            generation_time_s=0.0,
            model_used="unknown",
            model_history=[],
            prose="",
            opening_line="(none)",
            arc_used_fallback=False,
            arc_validation_valid=False,
            arc_event_count=0,
            arc_diagnostics=None,
            fallback_warning=None,
            lore_scene_count=0,
            lore_canon_fact_count=0,
            lore_texture_fact_count=0,
            texture_committed_count=0,
            starting_canon_texture_count=0,
            ending_canon_texture_count=0,
            error=error,
            prose_path=prose_path,
            meta_path=meta_path,
        )


def _count_inherited_beliefs(
    fresh_start: dict[str, dict[str, str]],
    canon_start: dict[str, dict[str, str]],
) -> int:
    diffs = 0
    for agent_id in sorted(set(fresh_start) | set(canon_start)):
        fresh_claims = fresh_start.get(agent_id, {})
        canon_claims = canon_start.get(agent_id, {})
        for claim_id in sorted(set(fresh_claims) | set(canon_claims)):
            if fresh_claims.get(claim_id) != canon_claims.get(claim_id):
                diffs += 1
    return diffs


def _first_event_divergence(fresh_events: list[Event], canon_events: list[Event]) -> tuple[int, str, str] | None:
    limit = min(len(fresh_events), len(canon_events))

    for index in range(limit):
        fresh = fresh_events[index]
        canon = canon_events[index]
        same = (
            fresh.type == canon.type
            and fresh.source_agent == canon.source_agent
            and tuple(fresh.target_agents) == tuple(canon.target_agents)
            and fresh.location_id == canon.location_id
            and fresh.description == canon.description
        )
        if not same:
            return (index, fresh.type.value, canon.type.value)

    if len(fresh_events) != len(canon_events):
        fresh_type = fresh_events[limit].type.value if limit < len(fresh_events) else "(none)"
        canon_type = canon_events[limit].type.value if limit < len(canon_events) else "(none)"
        return (limit, fresh_type, canon_type)

    return None


def _build_comparison_summary(
    story_a: ProseRunResult,
    fresh: ProseRunResult,
    canon: ProseRunResult,
    story_b_fresh: StoryRun,
    story_b_canon: StoryRun,
) -> str:
    inherited_beliefs = _count_inherited_beliefs(story_b_fresh.start_beliefs, story_b_canon.start_beliefs)
    dining_tension = float(story_b_canon.start_location_memory.get("dining_table", 0.0))
    foyer_tension = float(story_b_canon.start_location_memory.get("foyer", 0.0))

    divergence = _first_event_divergence(story_b_fresh.events, story_b_canon.events)
    if divergence is None:
        divergence_line = "none"
    else:
        index, fresh_type, canon_type = divergence
        divergence_line = f"index {index} (fresh={fresh_type}, canon={canon_type})"

    total_cost = float(fresh.cost_usd + canon.cost_usd)

    lines = [
        "=== CANON STORY COMPARISON ===",
        "",
        "Story B Fresh (seed 51, no canon)",
        "---------------------------------",
        f"Protagonist: {fresh.protagonist}",
        f"Words: {fresh.word_count}",
        f"Scenes: {fresh.scene_count}",
        f"Cost: ${fresh.cost_usd:.4f}",
        f"Time: {fresh.generation_time_s:.2f}s",
        f"Opening line: \"{fresh.opening_line}\"",
        f"Lore extracted: scenes={fresh.lore_scene_count}, canon_facts={fresh.lore_canon_fact_count}, "
        f"texture_facts={fresh.lore_texture_fact_count}, committed={fresh.texture_committed_count}",
        "",
        "Story B Canon (seed 51, with Story A canon)",
        "---------------------------------------------",
        f"Protagonist: {canon.protagonist}",
        f"Words: {canon.word_count}",
        f"Scenes: {canon.scene_count}",
        f"Cost: ${canon.cost_usd:.4f}",
        f"Time: {canon.generation_time_s:.2f}s",
        f"Opening line: \"{canon.opening_line}\"",
        f"Lore extracted: scenes={canon.lore_scene_count}, canon_facts={canon.lore_canon_fact_count}, "
        f"texture_facts={canon.lore_texture_fact_count}, committed={canon.texture_committed_count}",
        "",
        "Key differences in simulation input:",
        f"  Inherited beliefs: {inherited_beliefs}",
        f"  Starting location tension: dining_table={dining_tension:.2f}, foyer={foyer_tension:.2f}",
        f"  First event divergence: {divergence_line}",
        "",
        "Lore-loop checks:",
        f"  Story A committed texture facts: {story_a.texture_committed_count}",
        f"  Story B Canon inherited starting texture facts: {canon.starting_canon_texture_count}",
        (
            "  Cross-story texture inheritance: YES"
            if canon.starting_canon_texture_count > 0
            else "  Cross-story texture inheritance: NO"
        ),
        "  Ghost-fact motivation: invented details (e.g., an 'Ashford portfolio' reference) are now capturable as texture facts.",
    ]

    if fresh.error is not None:
        lines.append(f"  Story B Fresh narration failure: {fresh.error}")
    if canon.error is not None:
        lines.append(f"  Story B Canon narration failure: {canon.error}")
    if fresh.fallback_warning:
        lines.append(f"  Story B Fresh fallback warning: {fresh.fallback_warning}")
    if canon.fallback_warning:
        lines.append(f"  Story B Canon fallback warning: {canon.fallback_warning}")

    lines.extend(
        [
            "",
            f"Total Story B generation cost: ${total_cost:.4f}",
        ]
    )

    return "\n".join(lines).rstrip() + "\n"


def _copy_artifacts_to_examples(
    fresh: ProseRunResult,
    canon: ProseRunResult,
    summary_path: Path,
) -> None:
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    copy_pairs = [
        (fresh.prose_path, EXAMPLE_STORY_B_FRESH_PROSE),
        (fresh.meta_path, EXAMPLE_STORY_B_FRESH_META),
        (canon.prose_path, EXAMPLE_STORY_B_CANON_PROSE),
        (canon.meta_path, EXAMPLE_STORY_B_CANON_META),
        (summary_path, EXAMPLE_COMPARISON_SUMMARY),
    ]

    for source, destination in copy_pairs:
        if source.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)


def main() -> int:
    print("=== Canon Story Prose Demo ===")

    if not _required_api_keys_present():
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/3] Running simulations...")
    t0 = time.monotonic()

    story_a = run_story("Story A", STORY_A_SEED, loaded_canon=None)
    print(f"  Story A complete: seed={STORY_A_SEED}, events={len(story_a.events)}")

    story_b_fresh = run_story("Story B Fresh", STORY_B_SEED, loaded_canon=None)
    print(f"  Story B Fresh complete: seed={STORY_B_SEED}, events={len(story_b_fresh.events)}")

    loaded_canon = WorldCanon.from_dict(story_a.payload.get("world_canon"))
    story_b_canon = run_story("Story B Canon", STORY_B_SEED, loaded_canon=loaded_canon)
    print(f"  Story B Canon complete: seed={STORY_B_SEED}, events={len(story_b_canon.events)}")

    print(f"  Simulation stage finished in {time.monotonic() - t0:.2f}s")

    print("\n[2/3] Generating prose for Story A, Story B Fresh, and Story B Canon...")
    story_a_texture_canon = WorldCanon()
    story_a_result = _narrate_story(
        label="Story A",
        story_run=story_a,
        prose_path=STORY_A_PROSE,
        meta_path=STORY_A_META,
        narrator_canon=story_a_texture_canon,
    )
    story_b_narration_canon = WorldCanon.from_dict(loaded_canon.to_dict())
    story_b_narration_canon.texture.update(dict(story_a_texture_canon.texture))
    fresh_result = _narrate_story(
        label="Story B Fresh",
        story_run=story_b_fresh,
        prose_path=STORY_B_FRESH_PROSE,
        meta_path=STORY_B_FRESH_META,
    )
    canon_result = _narrate_story(
        label="Story B Canon",
        story_run=story_b_canon,
        prose_path=STORY_B_CANON_PROSE,
        meta_path=STORY_B_CANON_META,
        narrator_canon=story_b_narration_canon,
    )

    print("\n[3/3] Writing comparison summary and copying example artifacts...")
    summary = _build_comparison_summary(
        story_a_result,
        fresh_result,
        canon_result,
        story_b_fresh,
        story_b_canon,
    )
    _save_text(COMPARISON_SUMMARY, summary)
    print(summary, end="")

    _copy_artifacts_to_examples(fresh_result, canon_result, COMPARISON_SUMMARY)

    print("\nArtifacts written:")
    print(f"  {STORY_A_PROSE}")
    print(f"  {STORY_A_META}")
    print(f"  {STORY_B_FRESH_PROSE}")
    print(f"  {STORY_B_FRESH_META}")
    print(f"  {STORY_B_CANON_PROSE}")
    print(f"  {STORY_B_CANON_META}")
    print(f"  {COMPARISON_SUMMARY}")
    print("Copied to examples/:")
    print(f"  {EXAMPLE_STORY_B_FRESH_PROSE}")
    print(f"  {EXAMPLE_STORY_B_FRESH_META}")
    print(f"  {EXAMPLE_STORY_B_CANON_PROSE}")
    print(f"  {EXAMPLE_STORY_B_CANON_META}")
    print(f"  {EXAMPLE_COMPARISON_SUMMARY}")
    print(
        "Total generation cost (Story A + Story B Fresh + Story B Canon): "
        f"${story_a_result.cost_usd + fresh_result.cost_usd + canon_result.cost_usd:.4f}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
