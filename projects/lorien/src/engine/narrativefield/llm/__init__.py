"""LLM integration layer for NarrativeField.

This package is intentionally side-effect free on import. All network clients are created lazily.
"""

from __future__ import annotations

from narrativefield.llm.config import PipelineConfig
from narrativefield.llm.gateway import LLMGateway, ModelTier, UsageStats

__all__ = [
    "LLMGateway",
    "ModelTier",
    "PipelineConfig",
    "UsageStats",
]

