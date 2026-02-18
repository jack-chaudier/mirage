from __future__ import annotations

import json
import logging
import re
from typing import Any

from narrativefield.llm.config import PipelineConfig
from narrativefield.llm.gateway import LLMGateway, ModelTier
from narrativefield.schema.canon import WorldCanon
from narrativefield.schema.lore import TextureFact
from narrativefield.storyteller.prompts import build_continuity_check_prompt
from narrativefield.storyteller.types import CharacterState, SceneChunk

logger = logging.getLogger(__name__)

# Pattern to strip leaked XML artifacts from prose output.
_XML_ARTIFACT_RE = re.compile(
    r"</?(?:prose|state_update|summary|character_updates|character|new_threads|thread|resolved_threads|"
    r"lore_updates|canon_facts|texture_facts|fact|detail)\b[^>]*>",
    re.IGNORECASE,
)

# Pattern to strip everything inside <state_update>...</state_update> blocks.
_STATE_UPDATE_BLOCK_RE = re.compile(
    r"<state_update>.*?</state_update>",
    re.IGNORECASE | re.DOTALL,
)
_LORE_UPDATES_BLOCK_RE = re.compile(
    r"<lore_updates>.*?</lore_updates>",
    re.IGNORECASE | re.DOTALL,
)

# Multiple consecutive blank lines collapsed to exactly one blank line.
_MULTI_BLANK_RE = re.compile(r"\n{3,}")

_ENDS_WITH_LETTER_RE = re.compile(r"[A-Za-z]$")
_SENTENCE_END_RE = re.compile(r"[.!?](?:[\"'])?")


def _trim_to_last_sentence_boundary(text: str, *, chunk_index: int) -> str:
    """Trim a truncated chunk back to the last sentence-ending punctuation.

    Heuristic: if a chunk ends with a letter (no trailing punctuation), assume the
    model may have hit max_tokens mid-word/sentence. In that case, keep only fully
    terminated sentences (., !, ? optionally followed by a quote mark).
    """

    if not text or not _ENDS_WITH_LETTER_RE.search(text):
        return text

    last: re.Match[str] | None = None
    for m in _SENTENCE_END_RE.finditer(text):
        last = m
    if last is None:
        return text

    trimmed = text[: last.end()].rstrip()
    if len(trimmed) < len(text):
        logger.warning(
            "Trimming truncated prose chunk %d to last sentence boundary (%d -> %d chars)",
            chunk_index,
            len(text),
            len(trimmed),
        )
    return trimmed


def _merge_split_word_boundary(prev: str, nxt: str) -> tuple[str, str, bool]:
    """Attempt to merge a word split across a scene boundary."""

    if not prev or not nxt:
        return prev, nxt, False
    if not _ENDS_WITH_LETTER_RE.search(prev):
        return prev, nxt, False
    if not re.match(r"[a-z]", nxt):
        return prev, nxt, False

    m = re.match(r"([a-z]+)", nxt)
    if not m:
        return prev, nxt, False

    merged_prev = prev + m.group(1)
    merged_next = nxt[m.end() :].lstrip()
    return merged_prev, merged_next, True


class PostProcessor:
    """Phase 3: continuity checking and prose assembly."""

    def __init__(self, gateway: LLMGateway, config: PipelineConfig):
        self.gateway = gateway
        self.config = config

    async def check_continuity(
        self,
        prose_chunks: list[str],
        scene_chunks: list[SceneChunk],
        character_states: list[list[CharacterState]],
    ) -> list[dict]:
        """Run parallel continuity checks on all prose chunks via Grok.

        Returns list of check results, one per chunk:
        [{"scene": 0, "consistent": True, "violations": []}, ...]
        """

        requests: list[dict] = []
        for i, (prose, scene, chars) in enumerate(
            zip(prose_chunks, scene_chunks, character_states)
        ):
            user_prompt = build_continuity_check_prompt(
                prose_chunk=prose,
                events=scene.events,
                character_states=chars,
            )
            requests.append({
                "system": "You are a continuity checker. Return only valid JSON.",
                "user": user_prompt,
            })

        responses = await self.gateway.generate_batch(
            ModelTier.STRUCTURAL,
            requests,
            max_concurrency=self.config.phase3_max_concurrency,
        )

        results: list[dict] = []
        for i, raw in enumerate(responses):
            result = self._parse_continuity_response(i, raw)
            results.append(result)

        return results

    async def validate_texture_facts(
        self,
        texture_facts: list[TextureFact],
        canon: WorldCanon | None,
        existing_texture: list[TextureFact],
    ) -> list[dict[str, Any]]:
        """Check whether newly proposed texture facts contradict established canon."""

        if not texture_facts:
            return []

        established: list[str] = []
        if canon is not None:
            for tex in sorted(canon.texture.values(), key=lambda item: item.id):
                established.append(f"- [{tex.detail_type}] {tex.statement}")
        for tex in existing_texture:
            established.append(f"- [{tex.detail_type}] {tex.statement}")

        new_items = [f"- [{tex.id}] [{tex.detail_type}] {tex.statement}" for tex in texture_facts]

        prompt = (
            "You are a continuity checker for a story generation pipeline.\n"
            "Determine whether each new detail contradicts previously established details.\n\n"
            "<established_details>\n"
            f"{chr(10).join(established[:20]) if established else '(none)'}\n"
            "</established_details>\n\n"
            "<new_details>\n"
            f"{chr(10).join(new_items)}\n"
            "</new_details>\n\n"
            'Return ONLY valid JSON as an array: [{"fact_id":"...", "valid": true|false, "reason":"..."}].\n'
            "If there is no contradiction, set valid=true and reason to an empty string."
        )

        raw = await self.gateway.generate(
            ModelTier.STRUCTURAL,
            system_prompt="You are a continuity checker. Return only valid JSON.",
            user_prompt=prompt,
            max_tokens=1000,
        )

        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned).strip()

        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                out: list[dict[str, Any]] = []
                for item in parsed:
                    if not isinstance(item, dict):
                        continue
                    out.append(
                        {
                            "fact_id": str(item.get("fact_id", "")),
                            "valid": bool(item.get("valid", True)),
                            "reason": str(item.get("reason", "")),
                        }
                    )
                if out:
                    return out
        except Exception:
            logger.warning("Failed to parse texture validation response.")

        # Graceful fallback: mark all new facts valid.
        return [{"fact_id": tf.id, "valid": True, "reason": ""} for tf in texture_facts]

    def join_prose(
        self,
        prose_chunks: list[str],
        scene_breaks: str = "\n\n* * *\n\n",
    ) -> str:
        """Join prose chunks with scene break markers.

        Strips XML artifacts, normalizes whitespace, ensures consistent
        paragraph breaks.
        """

        cleaned: list[str] = []
        for i, chunk in enumerate(prose_chunks):
            text = chunk

            # Remove <state_update>...</state_update> blocks first (greedy within block).
            text = _STATE_UPDATE_BLOCK_RE.sub("", text)
            text = _LORE_UPDATES_BLOCK_RE.sub("", text)

            # Remove individual leaked XML tags.
            text = _XML_ARTIFACT_RE.sub("", text)

            # Normalize multiple blank lines to exactly one blank line (two newlines).
            text = _MULTI_BLANK_RE.sub("\n\n", text)

            # Strip leading/trailing whitespace.
            text = text.strip()
            text = _trim_to_last_sentence_boundary(text, chunk_index=i)

            if text:
                cleaned.append(text)

        stitched: list[str] = []
        for text in cleaned:
            if stitched:
                merged_prev, merged_next, did_merge = _merge_split_word_boundary(stitched[-1], text)
                if did_merge:
                    stitched[-1] = merged_prev
                    text = merged_next
            if text:
                stitched.append(text)

        return scene_breaks.join(stitched)

    @staticmethod
    def _parse_continuity_response(scene_index: int, raw: str) -> dict:
        """Parse a continuity check JSON response, handling failures gracefully."""

        # Strip markdown code fences if present.
        text = raw.strip()
        if text.startswith("```"):
            # Remove opening fence (```json or ```)
            first_newline = text.find("\n")
            if first_newline != -1:
                text = text[first_newline + 1 :]
            # Remove closing fence
            if text.rstrip().endswith("```"):
                text = text.rstrip()[:-3]
            text = text.strip()

        try:
            parsed = json.loads(text)
            return {
                "scene": scene_index,
                "consistent": bool(parsed.get("consistent", True)),
                "violations": list(parsed.get("violations", [])),
            }
        except (json.JSONDecodeError, TypeError, AttributeError) as exc:
            logger.warning(
                "Failed to parse continuity check for scene %d: %s",
                scene_index,
                exc,
            )
            return {
                "scene": scene_index,
                "consistent": True,
                "violations": [f"Parse error: {exc}"],
            }
