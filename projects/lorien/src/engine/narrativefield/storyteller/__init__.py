"""Storyteller infrastructure for sequential prose generation.

This package contains the foundational, side-effect free building blocks used by higher-level
pipeline components (scene splitting, summarization, narration, post-processing).
"""

from __future__ import annotations

from narrativefield.storyteller.checkpoint import CheckpointManager
from narrativefield.storyteller.types import (
    CharacterState,
    GenerationResult,
    NarrativeStateObject,
    NarrativeThread,
    SceneChunk,
    SceneOutcome,
)

__all__ = [
    "CharacterState",
    "CheckpointManager",
    "GenerationResult",
    "NarrativeStateObject",
    "NarrativeThread",
    "SceneChunk",
    "SceneOutcome",
]
