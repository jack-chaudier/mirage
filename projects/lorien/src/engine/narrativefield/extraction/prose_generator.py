from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import anthropic

from narrativefield.extraction.types import BeatSheet


MODEL_DEFAULT = "claude-sonnet-4-5-20250929"
TEMPERATURE_DEFAULT = 0.8


def build_llm_prompt(beat_sheet: BeatSheet) -> str:
    """
    Build the LLM prompt from a beat sheet.

    Source: specs/metrics/story-extraction.md Section 6.
    """

    def tension_label(t: float) -> str:
        if t < 0.2:
            return "low (calm, casual)"
        if t < 0.4:
            return "moderate (undercurrents of tension)"
        if t < 0.6:
            return "elevated (palpable discomfort)"
        if t < 0.8:
            return "high (confrontation, distress)"
        return "extreme (breaking point)"

    def format_emotions(emotions: dict[str, str]) -> str:
        if not emotions:
            return "unspecified"
        return "; ".join(f"{k}: {v}" for k, v in emotions.items())

    prompt = (
        "You are writing a short story scene based on a simulation of a fictional dinner party. "
        "The story's structure has already been determined by the simulation; your job is to bring it "
        "to life with prose, dialogue, and interiority.\n\n"
        "## CONTEXT\n\n"
        f"Setting: {beat_sheet.setting_summary}\n"
        f"Time span: {beat_sheet.time_span}\n"
        f"Genre feel: {beat_sheet.genre_preset}\n"
        f"Dominant theme: {beat_sheet.dominant_theme}\n"
        f"Thematic trajectory: {beat_sheet.thematic_trajectory}\n\n"
        "## CHARACTERS\n\n"
    )

    for c in beat_sheet.characters:
        prompt += (
            f"{c.name} ({c.role_in_arc})\n"
            f"- Goal: {c.key_goal}\n"
            f"- Flaw: {c.key_flaw}\n"
            f"- Secret: {c.key_secret or 'None'}\n"
            f"- Starts: {c.emotional_start or 'unspecified'}\n"
            f"- Ends: {c.emotional_end or 'unspecified'}\n\n"
        )

    prompt += (
        "## BEAT SEQUENCE\n\n"
        "Write the story following these beats in order. Each beat MUST be included. "
        "Do not skip, reorder, or add beats.\n\n"
    )

    for i, b in enumerate(beat_sheet.beats, 1):
        prompt += (
            f"### Beat {i}: {b.beat_type.value.upper()}\n"
            f"- What happens: {b.description}\n"
            f"- Location: {b.location}\n"
            f"- Present: {', '.join(b.participants)}\n"
            f"- Tension level: {tension_label(float(b.tension))}\n"
            f"- Emotional states: {format_emotions(b.emotional_states)}\n"
        )
        if b.irony_note:
            prompt += f"- Dramatic irony: {b.irony_note}\n"
        if b.key_changes:
            prompt += f"- Key changes: {'; '.join(b.key_changes)}\n"
        prompt += f"- POV: {b.pov_suggestion}\n"
        prompt += f"- Tone: {b.tone_suggestion}\n"
        prompt += f"- Pacing: {b.pacing_note}\n\n"

    target_words = len(beat_sheet.beats) * 300
    prompt += (
        "## CONSTRAINTS\n\n"
        "1. DO NOT invent new plot events. Everything that happens must come from the beats above.\n"
        "2. DO NOT change the outcome of any beat.\n"
        "3. DO NOT give characters knowledge they don't have.\n"
        "4. You MAY add internal thoughts, dialogue, and sensory details, consistent with the beat guidance.\n"
        "5. Maintain the tension trajectory: build to the turning point and then release.\n\n"
        "## VOICE AND STYLE\n\n"
        f"- Write in close third person, primarily from the protagonist's POV ({beat_sheet.protagonist}).\n"
        "- Prose style: literary fiction with grounded detail.\n"
        "- Dialogue should sound natural.\n\n"
        "## OUTPUT FORMAT\n\n"
        f"Write the complete scene as continuous prose. Target length: 200-400 words per beat, ~{target_words} words total.\n"
        "Begin writing.\n"
    )
    return prompt


@dataclass(frozen=True)
class ProseResult:
    prose: str | None
    error: str | None
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "prose": self.prose,
            "error": self.error,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


def generate_prose(
    *,
    beat_sheet: BeatSheet,
    model: str = MODEL_DEFAULT,
    temperature: float = TEMPERATURE_DEFAULT,
    max_tokens: int | None = None,
) -> ProseResult:
    """
    Generate story prose from a beat sheet using the Anthropic SDK.

    If the API key is missing or the request fails, returns error and no prose.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return ProseResult(prose=None, error="Missing ANTHROPIC_API_KEY", model=None)

    prompt = build_llm_prompt(beat_sheet)
    if max_tokens is None:
        max_tokens = max(300, 300 * len(beat_sheet.beats))

    try:
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model=model,
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            messages=[{"role": "user", "content": prompt}],
        )
        text = ""
        # msg.content is a list of content blocks; extract text blocks.
        for block in getattr(msg, "content", []) or []:
            if getattr(block, "type", None) == "text":
                text += getattr(block, "text", "")
        usage = getattr(msg, "usage", None)
        return ProseResult(
            prose=text.strip() or None,
            error=None,
            model=model,
            input_tokens=getattr(usage, "input_tokens", None) if usage else None,
            output_tokens=getattr(usage, "output_tokens", None) if usage else None,
        )
    except Exception as e:  # pragma: no cover - exercised in integration, not unit tests
        return ProseResult(prose=None, error=str(e), model=model)

