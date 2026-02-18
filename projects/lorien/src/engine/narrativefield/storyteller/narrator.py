from __future__ import annotations

import json
import logging
import re
import time
import uuid
from typing import Any

from narrativefield.schema.agents import (
    AgentState,
    CharacterFlaw,
    FlawType,
    GoalVector,
    PacingState,
    RelationshipState,
)
from narrativefield.schema.canon import CanonTexture, WorldCanon
from narrativefield.schema.lore import CanonFact, SceneLoreUpdates, StoryLore, TextureFact
from narrativefield.schema.world import SecretDefinition, WorldDefinition
from narrativefield.llm.gateway import LLMGateway, ModelTier
from narrativefield.llm.config import PipelineConfig
from narrativefield.storyteller.types import (
    CharacterState,
    GenerationResult,
    NarrativeStateObject,
    NarrativeThread,
    SceneChunk,
    SceneOutcome,
)
from narrativefield.storyteller.checkpoint import CheckpointManager
from narrativefield.storyteller.scene_splitter import split_into_scenes
from narrativefield.storyteller.event_summarizer import summarize_scene
from narrativefield.storyteller.lorebook import Lorebook
from narrativefield.storyteller.prompts import (
    build_system_prompt,
    build_scene_prompt,
    build_summary_compression_prompt,
)
from narrativefield.storyteller.postprocessor import PostProcessor


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Response parsing patterns
# ---------------------------------------------------------------------------

_PROSE_RE = re.compile(r"<prose>(.*?)</prose>", re.DOTALL)
_STATE_UPDATE_RE = re.compile(r"<state_update>(.*?)</state_update>", re.DOTALL)
_SUMMARY_RE = re.compile(r"<summary>(.*?)</summary>", re.DOTALL)
_CHARACTER_RE = re.compile(
    r'<character\s+[^>]*?id="([^"]*)"[^>]*?'
    r'emotional_state="([^"]*)"[^>]*?'
    r'current_goal="([^"]*)"[^/]*/?>',
    re.DOTALL,
)
# Alternate attribute order: emotional_state before id, etc.
_CHARACTER_ALT_RE = re.compile(
    r'<character\s+[^>]*?(?:'
    r'id="(?P<id>[^"]*)"'
    r'|emotional_state="(?P<emo>[^"]*)"'
    r'|current_goal="(?P<goal>[^"]*)"'
    r')[^>]*/?>',
    re.DOTALL,
)
_THREAD_RE = re.compile(
    r'<thread\s+[^>]*?description="([^"]*)"[^/]*/?>',
    re.DOTALL,
)
_RESOLVED_THREADS_RE = re.compile(
    r"<resolved_threads>(.*?)</resolved_threads>", re.DOTALL
)
_LORE_UPDATES_RE = re.compile(r"<lore_updates>(.*?)</lore_updates>", re.DOTALL)
_CANON_FACTS_RE = re.compile(r"<canon_facts>(.*?)</canon_facts>", re.DOTALL)
_TEXTURE_FACTS_RE = re.compile(r"<texture_facts>(.*?)</texture_facts>", re.DOTALL)
_FACT_RE = re.compile(
    r'<fact\s+event_ids="([^"]*)">(.*?)</fact>',
    re.DOTALL,
)
_DETAIL_RE = re.compile(
    r'<detail\s+type="([^"]*)"\s+entities="([^"]*)">(.*?)</detail>',
    re.DOTALL,
)

_LLM_ERROR_PREFIX = "[LLM_ERROR]"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_generation_response(response: str) -> tuple[str, dict | None]:
    """Extract prose and state update from the LLM response.

    Returns (prose_text, state_dict_or_None).
    """
    # Extract prose
    prose_match = _PROSE_RE.search(response)
    if prose_match:
        prose = prose_match.group(1).strip()
    else:
        # No <prose> tags: use full response minus any state_update block.
        prose = _STATE_UPDATE_RE.sub("", response).strip()

    # Extract state update
    state_match = _STATE_UPDATE_RE.search(response)
    if not state_match:
        return (prose, None)

    state_block = state_match.group(1)
    update: dict[str, Any] = {}

    # Summary
    summary_m = _SUMMARY_RE.search(state_block)
    if summary_m:
        update["summary"] = summary_m.group(1).strip()

    # Characters — try primary regex first, fall back to individual attribute extraction.
    characters: list[dict[str, str]] = []
    for m in _CHARACTER_RE.finditer(state_block):
        characters.append(
            {
                "id": m.group(1).strip(),
                "emotional_state": m.group(2).strip(),
                "current_goal": m.group(3).strip(),
            }
        )
    if not characters:
        # Fallback: scan for individual <character .../> with any attribute order.
        for raw_tag in re.findall(r"<character\s[^>]+/>", state_block, re.DOTALL):
            cid = _attr_value(raw_tag, "id")
            emo = _attr_value(raw_tag, "emotional_state")
            goal = _attr_value(raw_tag, "current_goal")
            if cid:
                characters.append(
                    {"id": cid, "emotional_state": emo, "current_goal": goal}
                )
    if characters:
        update["characters"] = characters

    # Threads
    threads: list[str] = [m.group(1).strip() for m in _THREAD_RE.finditer(state_block)]
    if threads:
        update["new_threads"] = threads

    # Resolved threads
    resolved_m = _RESOLVED_THREADS_RE.search(state_block)
    if resolved_m:
        raw = resolved_m.group(1).strip()
        indices: list[int] = []
        for part in raw.split(","):
            part = part.strip()
            if part.isdigit():
                indices.append(int(part))
        if indices:
            update["resolved_threads"] = indices

    return (prose, update if update else None)


def _extract_lore_updates(response: str, scene_index: int) -> SceneLoreUpdates:
    """Extract optional lore updates from an LLM response."""

    lore_match = _LORE_UPDATES_RE.search(response)
    if lore_match:
        lore_block = lore_match.group(1)
    elif ("<canon_facts>" in response) or ("<texture_facts>" in response):
        # Graceful fallback for malformed wrappers.
        lore_block = response
    else:
        return SceneLoreUpdates(scene_index=scene_index)

    canon_facts: list[CanonFact] = []
    cf_match = _CANON_FACTS_RE.search(lore_block)
    if cf_match:
        for idx, fact_match in enumerate(_FACT_RE.finditer(cf_match.group(1))):
            event_ids = [eid.strip() for eid in fact_match.group(1).split(",") if eid.strip()]
            statement = fact_match.group(2).strip()
            canon_facts.append(
                CanonFact(
                    id=f"cf_{scene_index}_{idx}",
                    statement=statement,
                    source_event_ids=event_ids,
                    entity_refs=list(event_ids),
                    scene_index=scene_index,
                )
            )

    texture_facts: list[TextureFact] = []
    tf_match = _TEXTURE_FACTS_RE.search(lore_block)
    if tf_match:
        for idx, detail_match in enumerate(_DETAIL_RE.finditer(tf_match.group(1))):
            detail_type = detail_match.group(1).strip()
            entities = [e.strip() for e in detail_match.group(2).split(",") if e.strip()]
            statement = detail_match.group(3).strip()
            texture_facts.append(
                TextureFact(
                    id=f"tf_{scene_index}_{idx}",
                    statement=statement,
                    entity_refs=entities,
                    detail_type=detail_type,
                    scene_index=scene_index,
                )
            )

    return SceneLoreUpdates(
        scene_index=scene_index,
        canon_facts=canon_facts,
        texture_facts=texture_facts,
    )


def _attr_value(tag: str, attr_name: str) -> str:
    """Extract attribute value from an XML-like tag string."""
    m = re.search(rf'{attr_name}="([^"]*)"', tag)
    return m.group(1).strip() if m else ""


def _apply_state_update(
    state: NarrativeStateObject,
    update: dict | None,
    scene: SceneChunk,
) -> NarrativeStateObject:
    """Create a new NarrativeStateObject by merging the update into state."""
    summary = state.summary_so_far
    characters = list(state.characters)
    threads = list(state.unresolved_threads)

    if update is not None:
        # Summary
        if "summary" in update:
            summary = str(update["summary"])

        # Character updates
        char_updates = update.get("characters", [])
        char_by_id = {c.agent_id: c for c in characters}
        for cu in char_updates:
            cid = cu.get("id", "")
            if cid in char_by_id:
                old = char_by_id[cid]
                char_by_id[cid] = CharacterState(
                    agent_id=old.agent_id,
                    name=old.name,
                    location=old.location,
                    emotional_state=cu.get("emotional_state") or old.emotional_state,
                    current_goal=cu.get("current_goal") or old.current_goal,
                    knowledge=list(old.knowledge),
                    secrets_revealed=list(old.secrets_revealed),
                    secrets_held=list(old.secrets_held),
                )
        characters = list(char_by_id.values())

        # New threads
        for desc in update.get("new_threads", []):
            threads.append(
                NarrativeThread(
                    description=desc,
                    involved_agents=list(scene.characters_present),
                    tension_level=0.5,
                    introduced_at_scene=scene.scene_index,
                )
            )

        # Resolved threads
        resolved_indices = set(update.get("resolved_threads", []))
        if resolved_indices:
            threads = [
                t for i, t in enumerate(threads) if i not in resolved_indices
            ]

    return NarrativeStateObject(
        summary_so_far=summary,
        last_paragraph=state.last_paragraph,
        current_scene_index=scene.scene_index,
        characters=characters,
        active_location=scene.location,
        unresolved_threads=threads,
        narrative_plan=list(state.narrative_plan),
        total_words_generated=state.total_words_generated,
        scenes_completed=state.scenes_completed,
    )


def _enforce_state_budget(
    state: NarrativeStateObject,
    max_tokens: int,
    max_summary_words: int,
    current_scene_characters: list[str] | None = None,
) -> NarrativeStateObject:
    """Truncation ladder to keep state under token budget."""
    if state.estimate_tokens() <= max_tokens:
        return state

    # Level 1: Trim narrative plan to next item only
    state.narrative_plan = state.narrative_plan[:1]
    if state.estimate_tokens() <= max_tokens:
        return state

    # Level 2: Trim per-character lists to last 5
    for char in state.characters:
        char.knowledge = char.knowledge[-5:]
        char.secrets_revealed = char.secrets_revealed[-5:]
        char.secrets_held = char.secrets_held[-5:]
    if state.estimate_tokens() <= max_tokens:
        return state

    # Level 3: Collapse offstage characters
    if current_scene_characters:
        for char in state.characters:
            if char.agent_id not in current_scene_characters:
                char.knowledge = []
                char.secrets_revealed = []
                char.secrets_held = []
                char.current_goal = ""
    if state.estimate_tokens() <= max_tokens:
        return state

    # Level 4: Keep only top 3 threads by tension
    state.unresolved_threads = sorted(
        state.unresolved_threads, key=lambda t: t.tension_level, reverse=True
    )[:3]
    if state.estimate_tokens() <= max_tokens:
        return state

    # Level 5: Hard truncate summary
    words = state.summary_so_far.split()
    state.summary_so_far = " ".join(words[:max_summary_words])
    return state


def _build_initial_characters(events: list) -> list[CharacterState]:
    """Build initial CharacterState entries from the events list."""
    seen: dict[str, str] = {}  # agent_id -> first location
    for e in events:
        agent_id = e.source_agent
        if agent_id and agent_id not in seen:
            seen[agent_id] = e.location_id
        for t in e.target_agents:
            if t and t not in seen:
                seen[t] = e.location_id

    characters: list[CharacterState] = []
    for agent_id in sorted(seen):
        characters.append(
            CharacterState(
                agent_id=agent_id,
                name=" ".join(
                    part.capitalize()
                    for part in agent_id.replace("-", "_").split("_")
                    if part
                ),
                location=seen[agent_id],
                emotional_state="",
                current_goal="",
                knowledge=[],
                secrets_revealed=[],
                secrets_held=[],
            )
        )
    return characters


def _infer_protagonist_id(events: list) -> str | None:
    counts: dict[str, int] = {}
    for event in events:
        if event.source_agent:
            counts[event.source_agent] = counts.get(event.source_agent, 0) + 1
        for target in event.target_agents:
            if target:
                counts[target] = counts.get(target, 0) + 1
    if not counts:
        return None
    return max(sorted(counts.keys()), key=lambda aid: counts[aid])


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class SequentialNarrator:
    """Generates prose from a narrative arc using sequential state-passing.

    Three-phase pipeline:
      1. Pre-processing  — classify beats, split scenes, build lorebook
      2. Sequential generation — iterate scene chunks, call LLM, update state
      3. Post-processing — continuity check, join prose
    """

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self.gateway = LLMGateway(self.config)
        self.checkpoint_mgr: CheckpointManager | None = None

    async def generate(
        self,
        events: list,
        beat_sheet: Any = None,
        world_data: dict | None = None,
        run_id: str | None = None,
        resume: bool = False,
        canon: WorldCanon | None = None,
    ) -> GenerationResult:
        """Run the full three-phase prose generation pipeline.

        Args:
            events: List of Event objects (typically ~20 per arc).
            beat_sheet: Optional BeatSheet for structural guidance.
            world_data: Optional dict with keys ``world_definition``,
                ``agents``, ``secrets`` for Lorebook construction.
            run_id: Identifier for checkpoint persistence. If *None* a UUID
                is generated.
            resume: If *True* and a checkpoint exists for *run_id*, resume
                from the last completed scene.
            canon: Optional pre-run world canon snapshot for lore context and
                texture canonicalization.

        Returns:
            A ``GenerationResult`` with the full prose and metadata.
        """
        t0 = time.monotonic()

        # ------------------------------------------------------------------
        # Phase 1: Pre-processing
        # ------------------------------------------------------------------
        lorebook = self._build_lorebook(world_data, canon=canon)
        story_lore = StoryLore()
        accumulated_texture: list[TextureFact] = []
        texture_validation_inputs: list[tuple[int, list[TextureFact], list[TextureFact]]] = []
        texture_commit_keys: list[str] = []
        starting_canon_for_validation = (
            WorldCanon.from_dict(canon.to_dict()) if canon is not None else None
        )
        designated_protagonist = (
            str(getattr(beat_sheet, "protagonist", "") or "").strip()
            if beat_sheet is not None
            else ""
        )
        if not designated_protagonist:
            designated_protagonist = _infer_protagonist_id(events) or ""

        # 1b. Classify beats if needed.
        if events and events[0].beat_type is None:
            from narrativefield.extraction.beat_classifier import classify_beats

            beats = classify_beats(events)
            for ev, bt in zip(events, beats):
                ev.beat_type = bt

        # 1c. Split into scene chunks.
        scene_chunks = split_into_scenes(
            events, target_chunk_size=self.config.phase2_events_per_chunk
        )

        # 1d. Fill scene summaries.
        for sc in scene_chunks:
            sc.summary = summarize_scene(sc.events)

        # 1e. Initial narrative state.
        state = NarrativeStateObject(
            summary_so_far="",
            last_paragraph="",
            current_scene_index=0,
            characters=_build_initial_characters(events),
            active_location=events[0].location_id if events else "",
            unresolved_threads=[],
            narrative_plan=[sc.summary for sc in scene_chunks],
            total_words_generated=0,
            scenes_completed=0,
        )

        # 1f. Checkpoint setup.
        prose_chunks: list[str] = []
        start_scene = 0
        character_states_history: list[list[CharacterState]] = []

        if run_id is None:
            run_id = uuid.uuid4().hex[:12]

        if self.config.checkpoint_enabled:
            self.checkpoint_mgr = CheckpointManager(
                self.config.checkpoint_dir, run_id
            )

            if resume and self.checkpoint_mgr.has_checkpoint():
                loaded = self.checkpoint_mgr.load_latest()
                if loaded is not None:
                    state, prose_chunks, last_done = loaded
                    start_scene = last_done + 1
                    logger.info(
                        "Resumed from checkpoint run_id=%s scene=%d",
                        run_id,
                        last_done,
                    )

        # Track per-scene outcomes for downstream reporting.
        # For resumed runs, we reconstruct prior scene outcomes from persisted prose (timing/retry info unavailable).
        scene_outcomes: list[SceneOutcome] = []
        if start_scene > 0:
            for j in range(min(start_scene, len(scene_chunks))):
                existing = prose_chunks[j] if j < len(prose_chunks) else ""
                wc = len(str(existing).split())
                status = "ok" if wc > 0 else "failed"
                scene_outcomes.append(
                    SceneOutcome(
                        scene_index=int(scene_chunks[j].scene_index),
                        status=status,
                        word_count=wc,
                        error_type=None if status == "ok" else "empty_prose",
                        retries=0,
                        generation_time_s=0.0,
                    )
                )

        # ------------------------------------------------------------------
        # Phase 2: Sequential Generation
        # ------------------------------------------------------------------
        system_prompt = self._build_system_prompt(lorebook)

        for i in range(start_scene, len(scene_chunks)):
            scene_t0 = time.monotonic()
            scene = scene_chunks[i]
            upcoming = scene_chunks[i + 1 : i + 3]

            if lorebook is not None:
                user_prompt = build_scene_prompt(
                    state,
                    scene,
                    lorebook,
                    upcoming,
                    self.config,
                    accumulated_texture=accumulated_texture,
                    protagonist_id=designated_protagonist or None,
                )
            else:
                user_prompt = self._build_fallback_scene_prompt(state, scene)

            # Choose tier and token budget.
            if scene.is_pivotal and self.config.phase2_use_extended_thinking_for_pivotal:
                tier = ModelTier.CREATIVE_DEEP
                max_tokens = int(self.config.phase2_creative_deep_max_tokens)
            else:
                tier = ModelTier.CREATIVE
                max_tokens = int(self.config.phase2_creative_max_tokens)

            # 2d. Generate.
            try:
                # Instrument retries by wrapping the gateway retry loop (without modifying gateway.py).
                attempts = 0
                had_instance_retry_attr = bool(
                    getattr(self.gateway, "__dict__", None)
                    and "_call_with_retry" in self.gateway.__dict__
                )
                orig_instance_retry_attr = (
                    self.gateway.__dict__.get("_call_with_retry")
                    if getattr(self.gateway, "__dict__", None)
                    else None
                )
                orig_call_with_retry = getattr(self.gateway, "_call_with_retry", None)
                if callable(orig_call_with_retry):

                    async def _wrapped_call_with_retry(*, provider: str, fn):  # type: ignore[no-untyped-def]
                        async def _wrapped_fn():  # type: ignore[no-untyped-def]
                            nonlocal attempts
                            attempts += 1
                            return await fn()

                        return await orig_call_with_retry(provider=provider, fn=_wrapped_fn)

                    self.gateway._call_with_retry = _wrapped_call_with_retry  # type: ignore[attr-defined, assignment]

                response = await self.gateway.generate(
                    tier,
                    system_prompt,
                    user_prompt,
                    cache_system_prompt=self.config.phase2_cache_system_prompt,
                    max_tokens=max_tokens,
                )
            except Exception:
                logger.exception(
                    "Scene %d generation failed; saving checkpoint and re-raising",
                    i,
                )
                if self.checkpoint_mgr is not None:
                    self.checkpoint_mgr.save(state, prose_chunks, max(0, i - 1))
                raise
            finally:
                if callable(orig_call_with_retry):
                    if had_instance_retry_attr:
                        # Restore any pre-existing instance override verbatim.
                        self.gateway._call_with_retry = orig_instance_retry_attr  # type: ignore[attr-defined, assignment]
                    else:
                        # Remove our instance override so the class method resolves normally.
                        try:
                            del self.gateway.__dict__["_call_with_retry"]
                        except Exception:
                            self.gateway._call_with_retry = orig_call_with_retry  # type: ignore[attr-defined, assignment]

            retries = max(0, attempts - 1) if "attempts" in locals() else 0

            # 2e. Parse response.
            scene_status = "ok"
            scene_error_type: str | None = None
            if response.startswith(_LLM_ERROR_PREFIX):
                logger.warning(
                    "LLM error for scene %d: %s", i, response[:200]
                )
                new_prose = ""
                parsed_update = None
                scene_status = "failed"
                scene_error_type = _llm_error_type_from_response(response)
                scene_lore = SceneLoreUpdates(scene_index=i)
            else:
                new_prose, parsed_update = _parse_generation_response(response)
                prior_texture = list(accumulated_texture)
                scene_lore = _extract_lore_updates(response, scene_index=i)
                if scene_lore.texture_facts:
                    texture_validation_inputs.append(
                        (
                            int(scene.scene_index),
                            list(scene_lore.texture_facts),
                            prior_texture,
                        )
                    )
                accumulated_texture.extend(scene_lore.texture_facts)
                if canon is not None:
                    for tf in scene_lore.texture_facts:
                        canon_key = f"{run_id}__{tf.id}"
                        if canon_key in canon.texture:
                            continue
                        canon.texture[canon_key] = CanonTexture(
                            id=canon_key,
                            statement=tf.statement,
                            entity_refs=list(tf.entity_refs),
                            detail_type=tf.detail_type,
                            source_story_id=run_id,
                            source_scene_index=int(tf.scene_index),
                            committed_at_canon_version=int(canon.canon_version),
                        )
                        texture_commit_keys.append(canon_key)
            story_lore.scene_lore.append(scene_lore)

            # 2f. Compress summary.
            state = _apply_state_update(state, parsed_update, scene)
            compressed_summary = await self._compress_summary(
                state.summary_so_far, new_prose
            )
            # Rebuild state with compressed summary and updated counters.
            word_count_delta = len(new_prose.split())
            if scene_status != "failed" and word_count_delta == 0:
                # Treat empty prose as a per-scene failure (even if the call technically succeeded).
                scene_status = "failed"
                scene_error_type = "empty_prose"
            last_para = _last_paragraph(new_prose)
            state = NarrativeStateObject(
                summary_so_far=compressed_summary,
                last_paragraph=last_para,
                current_scene_index=i,
                characters=list(state.characters),
                active_location=scene.location,
                unresolved_threads=list(state.unresolved_threads),
                narrative_plan=list(state.narrative_plan),
                total_words_generated=state.total_words_generated + word_count_delta,
                scenes_completed=state.scenes_completed + 1,
            )

            state = _enforce_state_budget(
                state,
                self.config.max_state_tokens,
                self.config.max_summary_words,
                current_scene_characters=scene.characters_present,
            )

            prose_chunks.append(new_prose)
            character_states_history.append(list(state.characters))

            # 2h. Checkpoint.
            if self.checkpoint_mgr is not None:
                self.checkpoint_mgr.save(state, prose_chunks, i)

            logger.info(
                "Scene %d/%d complete — %d words (total %d)",
                i + 1,
                len(scene_chunks),
                word_count_delta,
                state.total_words_generated,
            )

            scene_outcomes.append(
                SceneOutcome(
                    scene_index=int(scene.scene_index),
                    status=scene_status,
                    word_count=int(word_count_delta),
                    error_type=scene_error_type,
                    retries=int(retries),
                    generation_time_s=round(time.monotonic() - scene_t0, 2),
                )
            )

        # ------------------------------------------------------------------
        # Phase 3: Post-processing
        # ------------------------------------------------------------------
        post = PostProcessor(self.gateway, self.config)
        checks = await post.check_continuity(
            prose_chunks, scene_chunks, character_states_history
        )
        texture_validation_report: list[dict[str, Any]] = []
        for scene_idx, new_facts, prior_facts in texture_validation_inputs:
            validation = await post.validate_texture_facts(
                texture_facts=new_facts,
                canon=starting_canon_for_validation,
                existing_texture=prior_facts,
            )
            texture_validation_report.append(
                {
                    "scene": int(scene_idx),
                    "results": validation,
                }
            )
        if checks:
            logger.info(
                "Continuity check returned %d issue(s): %s",
                len(checks),
                [c.get("issue", "") for c in checks],
            )

        full_prose = post.join_prose(prose_chunks)

        # Run-level status classification.
        if scene_outcomes:
            if all(s.status == "ok" for s in scene_outcomes):
                status = "complete"
            elif any(s.status == "ok" for s in scene_outcomes):
                status = "partial"
            else:
                status = "failed"
        else:
            status = "failed" if not full_prose.strip() else "complete"

        # Clear checkpoint on success.
        if self.checkpoint_mgr is not None:
            self.checkpoint_mgr.clear()

        elapsed = time.monotonic() - t0
        usage = self.gateway.usage_total
        checkpoint_path: str | None = None
        if self.checkpoint_mgr is not None:
            checkpoint_path = str(self.checkpoint_mgr.run_dir)

        return GenerationResult(
            status=status,
            prose=full_prose,
            word_count=len(full_prose.split()),
            scenes_generated=len(scene_chunks),
            scene_outcomes=list(scene_outcomes),
            final_state=state,
            usage={
                "total_input_tokens": usage.input_tokens,
                "total_output_tokens": usage.output_tokens,
                "cache_read_tokens": usage.cache_read_tokens,
                "cache_write_tokens": usage.cache_write_tokens,
                "estimated_cost_usd": usage.estimated_cost_usd,
                "continuity_report": checks,
                "texture_validation_report": texture_validation_report,
                "texture_committed_keys": texture_commit_keys,
                "llm_response_metadata": dict(self.gateway.response_metadata),
                "model_history": list(self.gateway.model_history),
            },
            generation_time_seconds=round(elapsed, 2),
            checkpoint_path=checkpoint_path,
            story_lore=story_lore,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_lorebook(
        world_data: dict | None,
        canon: WorldCanon | None = None,
    ) -> Lorebook | None:
        if world_data is None:
            return None
        try:
            world_definition = _coerce_world_definition(world_data.get("world_definition"))
            agents = _coerce_agent_list(world_data.get("agents"))
            secrets = _coerce_secret_list(world_data.get("secrets"), world_definition)
            canon_state = canon
            if canon_state is None:
                raw_canon = world_data.get("world_canon")
                if isinstance(raw_canon, WorldCanon):
                    canon_state = raw_canon
                elif isinstance(raw_canon, dict):
                    canon_state = WorldCanon.from_dict(raw_canon)
            return Lorebook(world_definition, agents, secrets, canon=canon_state)
        except Exception:
            logger.exception("Failed to build Lorebook from world_data")
            return None

    def _build_system_prompt(self, lorebook: Lorebook | None) -> str:
        if lorebook is not None:
            return build_system_prompt(lorebook, self.config)
        # Fallback system prompt when no lorebook is available.
        return (
            "You are a literary fiction writer. Generate vivid prose for "
            "the scene described below. Write in close third-person, present "
            "tense. Wrap your prose in <prose>...</prose> tags and optionally "
            "provide a <state_update>...</state_update> block with character "
            "emotional states and unresolved threads."
        )

    @staticmethod
    def _build_fallback_scene_prompt(
        state: NarrativeStateObject, scene: SceneChunk
    ) -> str:
        """Minimal scene prompt when no lorebook is available."""
        from narrativefield.storyteller.event_summarizer import summarize_event

        lines = [state.to_prompt_xml(), ""]
        if state.summary_so_far.strip():
            lines.append("<previously_established>")
            lines.append(state.summary_so_far.strip())
            lines.append("</previously_established>")
            lines.append("")
        if state.last_paragraph:
            lines.append(f"<continuity>{state.last_paragraph}</continuity>")
            lines.append("")
        lines.append("<events>")
        for ev in scene.events:
            lines.append(f"  {summarize_event(ev)}")
        lines.append("</events>")
        lines.append("")
        lines.append("<instructions>")
        lines.append(f"Write scene {scene.scene_index} at {scene.location}.")
        lines.append(f"Target: {max(400, len(scene.events) * 100)} words.")
        if scene.is_pivotal:
            lines.append("This is a PIVOTAL scene. Slow down, add interiority.")
        lines.append("Return prose in <prose> tags and state update in <state_update> tags.")
        lines.append("</instructions>")
        return "\n".join(lines)

    async def _compress_summary(
        self, summary_so_far: str, new_prose: str
    ) -> str:
        """Compress the rolling summary using the structural tier."""
        if not new_prose:
            return summary_so_far

        compression_prompt = build_summary_compression_prompt(
            summary_so_far, new_prose, self.config.max_summary_words
        )
        try:
            compressed = await self.gateway.generate(
                ModelTier.STRUCTURAL,
                "You are a precise factual summarizer.",
                compression_prompt,
                max_tokens=800,
            )
            if compressed.startswith(_LLM_ERROR_PREFIX):
                logger.warning("Summary compression failed: %s", compressed[:200])
                # Degrade: keep existing summary plus a truncated new prose tail.
                fallback = summary_so_far + " " + _truncate_words(new_prose, 80)
                return fallback.strip()
            return compressed.strip()
        except Exception:
            logger.exception("Summary compression raised an exception")
            fallback = summary_so_far + " " + _truncate_words(new_prose, 80)
            return fallback.strip()


def _coerce_world_definition(raw: Any) -> WorldDefinition:
    if isinstance(raw, WorldDefinition):
        return raw
    if isinstance(raw, dict):
        return WorldDefinition.from_dict(raw)
    raise ValueError("world_data missing a usable world_definition")


def _coerce_secret_list(
    raw: Any,
    world_definition: WorldDefinition,
) -> list[SecretDefinition]:
    if raw is None:
        return list(world_definition.secrets.values())

    if isinstance(raw, dict):
        items = raw.values()
    elif isinstance(raw, list):
        items = raw
    else:
        items = []

    secrets: list[SecretDefinition] = []
    for item in items:
        if isinstance(item, SecretDefinition):
            secrets.append(item)
        elif isinstance(item, dict):
            try:
                secrets.append(SecretDefinition.from_dict(item))
            except Exception:
                logger.debug("Skipping invalid secret entry in world_data.", exc_info=True)

    if secrets:
        return secrets
    return list(world_definition.secrets.values())


def _coerce_agent_list(raw: Any) -> list[AgentState]:
    if isinstance(raw, dict):
        items = list(raw.values())
    elif isinstance(raw, list):
        items = raw
    else:
        items = []

    agents: list[AgentState] = []
    for item in items:
        if isinstance(item, AgentState):
            agents.append(item)
            continue
        if not isinstance(item, dict):
            continue
        agent = _coerce_agent_state(item)
        if agent is not None:
            agents.append(agent)

    _inject_default_relationships(agents)
    return agents


def _coerce_agent_state(raw: dict[str, Any]) -> AgentState | None:
    agent_id = str(raw.get("id") or "").strip()
    if not agent_id:
        return None

    normalized = dict(raw)
    if not normalized.get("location") and normalized.get("initial_location"):
        normalized["location"] = normalized.get("initial_location")

    # Rich agent payloads can be parsed directly.
    if any(
        key in normalized
        for key in ("goals", "flaws", "pacing", "relationships", "beliefs", "emotional_state")
    ):
        try:
            return AgentState.from_dict(normalized)
        except Exception:
            logger.debug("Failed to parse rich agent payload; falling back to manifest coercion.", exc_info=True)

    return _agent_state_from_manifest(normalized)


def _agent_state_from_manifest(raw: dict[str, Any]) -> AgentState | None:
    agent_id = str(raw.get("id") or "").strip()
    if not agent_id:
        return None

    name = str(raw.get("name") or agent_id).strip()
    location = str(raw.get("location") or raw.get("initial_location") or "").strip()
    goal_summary = str(raw.get("goal_summary") or "").strip()
    flaw_type = _coerce_flaw_type(raw.get("primary_flaw"))

    flaws = [
        CharacterFlaw(
            flaw_type=flaw_type,
            strength=0.6,
            trigger="social pressure",
            effect="deflection",
            description=goal_summary or "Under pressure, this flaw shapes their choices.",
        )
    ]

    return AgentState(
        id=agent_id,
        name=name,
        location=location,
        goals=_goal_vector_from_summary(goal_summary),
        flaws=flaws,
        pacing=PacingState(),
        emotional_state={},
        relationships={},
        beliefs={},
        alcohol_level=0.0,
        commitments=[],
    )


def _coerce_flaw_type(raw: Any) -> FlawType:
    val = str(raw or "").strip().lower()
    if not val:
        return FlawType.DENIAL

    for flaw in FlawType:
        if flaw.value == val or flaw.name.lower() == val:
            return flaw
    return FlawType.DENIAL


def _goal_vector_from_summary(summary: str) -> GoalVector:
    goals = GoalVector()
    text = summary.lower()

    if "status" in text or "order" in text:
        goals.status = 0.8
    if "safe" in text or "protect" in text:
        goals.safety = 0.8
    if "secret" in text or "hide" in text:
        goals.secrecy = 0.8
    if "truth" in text or "investigat" in text:
        goals.truth_seeking = 0.8
    if "autonom" in text or "independ" in text:
        goals.autonomy = 0.8
    if "loyal" in text:
        goals.loyalty = 0.8

    return goals


def _inject_default_relationships(agents: list[AgentState]) -> None:
    if len(agents) < 2:
        return

    ids = [a.id for a in agents if a.id]
    for agent in agents:
        if agent.relationships:
            continue
        defaults: dict[str, RelationshipState] = {}
        for other in ids:
            if other == agent.id:
                continue
            defaults[other] = RelationshipState()
            if len(defaults) >= 3:
                break
        agent.relationships = defaults


def _last_paragraph(text: str) -> str:
    """Return the last non-empty paragraph of *text*."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if paragraphs:
        return paragraphs[-1]
    stripped = text.strip()
    return stripped[-500:] if len(stripped) > 500 else stripped


def _truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).rstrip(",;:.") + "..."


def _llm_error_type_from_response(response: str) -> str | None:
    """Best-effort extraction of a structured LLM error type from a gateway response."""
    if not response.startswith(_LLM_ERROR_PREFIX):
        return None

    payload_text = response[len(_LLM_ERROR_PREFIX) :].strip()
    if not payload_text:
        return "llm_error"
    if not payload_text.startswith("{"):
        return "llm_error"
    try:
        payload = json.loads(payload_text)
        if isinstance(payload, dict):
            err = payload.get("error")
            if isinstance(err, dict):
                t = err.get("type")
                if isinstance(t, str) and t.strip():
                    return t.strip()
    except json.JSONDecodeError:
        return "llm_error"
    return "llm_error"
