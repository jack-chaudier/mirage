from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PipelineConfig:
    # Model selection
    structural_model: str = "grok-4-1-fast"
    creative_model: str = "claude-haiku-4-5-20251001"

    # Phase 1 settings
    phase1_max_concurrency: int = 10
    phase1_events_summary_max_words: int = 30  # per-event summary length

    # Phase 2 settings
    phase2_events_per_chunk: int = 10  # target 8-12 events per scene chunk
    phase2_max_words_per_chunk: int = 1500
    # LLM output budgets for storyteller scene generation.
    # Raised to reduce mid-word truncation when creative calls hit max_tokens.
    phase2_creative_max_tokens: int = 4096
    phase2_creative_deep_max_tokens: int = 4096
    phase2_use_extended_thinking_for_pivotal: bool = True
    phase2_cache_system_prompt: bool = True

    # Phase 3 settings
    phase3_max_concurrency: int = 5

    # State object limits
    max_summary_words: int = 500
    max_state_tokens: int = 1500  # keep NarrativeStateObject under this

    # Checkpoint
    checkpoint_dir: str = ".storyteller_checkpoints"
    checkpoint_enabled: bool = True

    # LLM gateway retry (transient failures)
    retry_max_attempts: int = 5
    retry_base_delay: float = 2.0
    retry_max_delay: float = 30.0
