from __future__ import annotations

import asyncio
import json
import logging
import os
import random
from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable, Literal, TypeVar

import anthropic

from narrativefield.llm.config import PipelineConfig

try:  # Optional: the repo may not have openai pinned in pyproject yet.
    from openai import AsyncOpenAI  # type: ignore
except Exception:  # pragma: no cover - exercised when dependency is missing
    AsyncOpenAI = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)

_T = TypeVar("_T")


class ModelTier(Enum):
    STRUCTURAL = "structural"  # Grok 4.1 Fast: classification, summarization, validation
    CREATIVE = "creative"  # Claude Haiku 4.5: prose generation
    CREATIVE_DEEP = "creative_deep"  # Claude Haiku 4.5 + extended thinking: pivotal scenes


@dataclass(slots=True)
class UsageStats:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def estimated_cost_usd(self) -> float:
        """Rough cost estimate based on current pricing.

        This estimates Anthropic Claude Haiku 4.5 pricing, including prompt caching:
        - Input: $0.80 / MTok
        - Output: $4.00 / MTok
        - Cache reads: $0.08 / MTok (90% discount on cached prompt tokens)
        - Cache writes: $1.00 / MTok

        For Grok structural calls this will overestimate; treat as a conservative upper bound.
        """

        # Dollars per token.
        in_cost = 0.80 / 1_000_000.0
        out_cost = 4.0 / 1_000_000.0
        cache_read_cost = 0.08 / 1_000_000.0
        cache_write_cost = 1.0 / 1_000_000.0
        return (
            self.input_tokens * in_cost
            + self.output_tokens * out_cost
            + self.cache_read_tokens * cache_read_cost
            + self.cache_write_tokens * cache_write_cost
        )

    def estimate_cost(self, pricing: Literal["grok_fast", "claude_sonnet"]) -> float:
        """Estimate cost using a specific pricing profile."""
        if pricing == "grok_fast":
            in_cost = 0.20 / 1_000_000.0
            out_cost = 0.50 / 1_000_000.0
            # Grok does not support caching; treat cache tokens as normal input.
            return (self.input_tokens + self.cache_read_tokens + self.cache_write_tokens) * in_cost + (
                self.output_tokens * out_cost
            )
        return self.estimated_cost_usd

    def add(self, other: UsageStats) -> None:
        self.input_tokens += int(other.input_tokens)
        self.output_tokens += int(other.output_tokens)
        self.cache_read_tokens += int(other.cache_read_tokens)
        self.cache_write_tokens += int(other.cache_write_tokens)


class LLMGateway:
    """Unified interface for routing LLM calls to the appropriate provider."""

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self.usage_total = UsageStats()
        self.response_metadata: dict[str, Any] = {}
        self.model_history: list[dict[str, str]] = []
        self._usage_lock = asyncio.Lock()
        self._grok_client: Any | None = None
        self._claude_client: Any | None = None

    async def generate(
        self,
        tier: ModelTier,
        system_prompt: str,
        user_prompt: str,
        cache_system_prompt: bool = False,
        max_tokens: int = 2000,
    ) -> str:
        """Route to appropriate provider based on tier."""

        if tier is ModelTier.STRUCTURAL:
            return await self._generate_grok(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
            )
        if tier is ModelTier.CREATIVE:
            return await self._generate_claude_with_retry(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
                cache_system_prompt=cache_system_prompt,
                extended_thinking=False,
            )
        if tier is ModelTier.CREATIVE_DEEP:
            return await self._generate_claude_with_retry(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
                cache_system_prompt=cache_system_prompt,
                extended_thinking=True,
            )
        raise ValueError(f"Unknown model tier: {tier}")

    async def generate_batch(
        self,
        tier: ModelTier,
        requests: list[dict],  # list of {"system": str, "user": str}
        max_concurrency: int = 10,
    ) -> list[str]:
        """Run multiple requests concurrently with semaphore limiting."""

        semaphore = asyncio.Semaphore(int(max_concurrency))

        async def _run_one(req: dict) -> str:
            async with semaphore:
                system = str(req.get("system", ""))
                user = str(req.get("user", ""))
                return await self.generate(tier, system, user)

        tasks = [asyncio.create_task(_run_one(req)) for req in requests]
        # gather preserves task order.
        return await asyncio.gather(*tasks)

    def _get_grok_client(self) -> Any | None:
        if self._grok_client is not None:
            return self._grok_client
        if AsyncOpenAI is None:
            return None
        api_key = os.getenv("XAI_API_KEY")
        if not api_key:
            return None
        self._grok_client = AsyncOpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
        return self._grok_client

    def _get_claude_client(self) -> Any | None:
        if self._claude_client is not None:
            return self._claude_client
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return None
        self._claude_client = anthropic.AsyncAnthropic(api_key=api_key)
        return self._claude_client

    @staticmethod
    def _extract_status_code(exc: BaseException) -> int | None:
        status = getattr(exc, "status_code", None)
        if isinstance(status, int):
            return status

        resp = getattr(exc, "response", None)
        if resp is not None:
            resp_status = getattr(resp, "status_code", None)
            if isinstance(resp_status, int):
                return resp_status
        return None

    @staticmethod
    def _is_retryable_status_code(status_code: int | None) -> bool:
        if status_code is None:
            return False
        if status_code in {429, 529}:
            return True
        return 500 <= status_code <= 599

    @staticmethod
    def _is_retryable_exception(*, provider: str, exc: BaseException) -> bool:
        # Always allow caller-driven cancellations through.
        if isinstance(exc, asyncio.CancelledError):
            return False

        # Generic transport-level failures.
        if isinstance(exc, (TimeoutError, asyncio.TimeoutError, ConnectionError)):
            return True

        status_code = LLMGateway._extract_status_code(exc)
        if LLMGateway._is_retryable_status_code(status_code):
            return True

        if provider == "anthropic":
            retryable_types = (
                getattr(anthropic, "InternalServerError", None),
                getattr(anthropic, "RateLimitError", None),
                getattr(anthropic, "APITimeoutError", None),
                getattr(anthropic, "APIConnectionError", None),
            )
            for t in retryable_types:
                if isinstance(t, type) and isinstance(exc, t):
                    return True

        # OpenAI SDK is optional in this repo; fall back to status codes above when missing.
        if provider in {"xai", "openai"}:
            try:
                import openai  # type: ignore
            except Exception:
                openai = None  # type: ignore[assignment]
            if openai is not None:
                retryable_types = (
                    getattr(openai, "InternalServerError", None),
                    getattr(openai, "RateLimitError", None),
                    getattr(openai, "APITimeoutError", None),
                    getattr(openai, "APIConnectionError", None),
                )
                for t in retryable_types:
                    if isinstance(t, type) and isinstance(exc, t):
                        return True

        return False

    async def _call_with_retry(
        self,
        *,
        provider: str,
        fn: Callable[[], Awaitable[_T]],
    ) -> _T:
        max_attempts = max(1, int(self.config.retry_max_attempts))
        base_delay = max(0.0, float(self.config.retry_base_delay))
        max_delay = max(0.0, float(self.config.retry_max_delay))

        first_exc: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                return await fn()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                if first_exc is None:
                    first_exc = e

                retryable = self._is_retryable_exception(provider=provider, exc=e)
                if (not retryable) or attempt >= max_attempts:
                    # Per policy, raise the original (first) error we saw.
                    raise first_exc

                delay = min(max_delay, base_delay * (2 ** (attempt - 1))) * random.uniform(0.5, 1.5)
                logger.warning(
                    "Retry %d/%d for %s after %s, waiting %.1fs",
                    attempt,
                    max_attempts,
                    provider,
                    type(e).__name__,
                    delay,
                )
                await asyncio.sleep(delay)

        # Defensive: loop always returns or raises.
        assert first_exc is not None
        raise first_exc

    async def _generate_grok(self, *, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        client = self._get_grok_client()
        if client is None:
            missing = "openai" if AsyncOpenAI is None else "XAI_API_KEY"
            raise RuntimeError(f"Missing {missing}")

        model_primary = self.config.structural_model
        model_fallback = "grok-beta"

        # Try primary model first, then fall back if it appears unknown/unavailable.
        models = [model_primary]
        if model_primary != model_fallback and model_primary == "grok-4-1-fast":
            models.append(model_fallback)

        last_exc: Exception | None = None
        for model in models:
            try:
                text, usage = await self._call_with_retry(
                    provider="xai",
                    fn=lambda model=model: self._call_openai_chat_completions(
                        client=client,
                        model=model,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        max_tokens=max_tokens,
                    ),
                )
                await self._record_usage(
                    provider="xai",
                    model=model,
                    tier=ModelTier.STRUCTURAL,
                    usage=usage,
                )
                self.response_metadata = {
                    "provider": "xai",
                    "tier": ModelTier.STRUCTURAL.value,
                    "model_used": model,
                    "fallback_used": model != model_primary,
                }
                return text
            except Exception as e:
                last_exc = e
                if model == model_primary and model_primary == "grok-4-1-fast" and self._looks_like_unknown_model(e):
                    logger.warning(
                        "Grok primary model unavailable, falling back to %s",
                        model_fallback,
                    )
                    continue
                raise

        assert last_exc is not None
        raise last_exc

    async def _generate_claude_with_retry(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        cache_system_prompt: bool,
        extended_thinking: bool,
    ) -> str:
        provider = "anthropic"
        try:
            return await self._call_with_retry(
                provider=provider,
                fn=lambda: self._generate_claude(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_tokens=max_tokens,
                    cache_system_prompt=cache_system_prompt,
                    extended_thinking=extended_thinking,
                ),
            )
        except Exception as e:
            if extended_thinking and self._is_retryable_exception(provider=provider, exc=e):
                logger.warning(
                    "Downgrading CREATIVE_DEEP to CREATIVE for scene after %d failures",
                    int(self.config.retry_max_attempts),
                )
                try:
                    return await self._generate_claude(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        max_tokens=max_tokens,
                        cache_system_prompt=cache_system_prompt,
                        extended_thinking=False,
                    )
                except Exception:
                    raise e
            raise

    async def _generate_claude(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        cache_system_prompt: bool,
        extended_thinking: bool,
    ) -> str:
        client = self._get_claude_client()
        if client is None:
            raise RuntimeError("Missing ANTHROPIC_API_KEY")

        model = self.config.creative_model

        kwargs: dict[str, Any] = {}
        if extended_thinking:
            # Extended thinking constraints:
            # - max_tokens must be >= budget_tokens + expected output
            # - temperature must not be set (defaults to 1.0)
            # - system must be a plain string (cache_control conflicts with thinking)
            thinking_budget = 10_000
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
            effective_max_tokens = thinking_budget + max(max_tokens, 2000)
            system: str | list[dict[str, object]] = system_prompt
        else:
            effective_max_tokens = max_tokens
            if cache_system_prompt:
                system = [
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            else:
                system = system_prompt

        msg = await client.messages.create(
            model=model,
            max_tokens=int(effective_max_tokens),
            system=system,
            messages=[{"role": "user", "content": user_prompt}],
            **kwargs,
        )
        text = self._extract_anthropic_text(msg)
        usage = self._extract_anthropic_usage(msg)
        tier = ModelTier.CREATIVE_DEEP if extended_thinking else ModelTier.CREATIVE
        await self._record_usage(provider="anthropic", model=model, tier=tier, usage=usage)
        self.response_metadata = {
            "provider": "anthropic",
            "tier": tier.value,
            "model_used": model,
            "fallback_used": False,
        }
        return text

    async def _call_openai_chat_completions(
        self,
        *,
        client: Any,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
    ) -> tuple[str, UsageStats]:
        resp = await client.chat.completions.create(
            model=model,
            max_tokens=int(max_tokens),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = ""
        choices = getattr(resp, "choices", None) or []
        if choices:
            msg = getattr(choices[0], "message", None)
            text = getattr(msg, "content", "") if msg is not None else ""
        usage_obj = getattr(resp, "usage", None)
        usage = UsageStats(
            input_tokens=int(getattr(usage_obj, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage_obj, "completion_tokens", 0) or 0),
        )
        return (str(text).strip(), usage)

    @staticmethod
    def _extract_anthropic_text(msg: Any) -> str:
        text = ""
        for block in getattr(msg, "content", []) or []:
            if getattr(block, "type", None) == "text":
                text += getattr(block, "text", "")
        return text.strip()

    @staticmethod
    def _extract_anthropic_usage(msg: Any) -> UsageStats:
        usage_obj = getattr(msg, "usage", None)
        if usage_obj is None:
            return UsageStats()
        return UsageStats(
            input_tokens=int(getattr(usage_obj, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage_obj, "output_tokens", 0) or 0),
            cache_read_tokens=int(
                getattr(usage_obj, "cache_read_input_tokens", None)
                or getattr(usage_obj, "cache_read_tokens", 0)
                or 0
            ),
            cache_write_tokens=int(
                getattr(usage_obj, "cache_creation_input_tokens", None)
                or getattr(usage_obj, "cache_write_tokens", 0)
                or 0
            ),
        )

    async def _record_usage(self, *, provider: str, model: str, tier: ModelTier, usage: UsageStats) -> None:
        async with self._usage_lock:
            self.usage_total.add(usage)
            self.model_history.append(
                {
                    "provider": provider,
                    "tier": tier.value,
                    "model": model,
                }
            )

        cost_hint = (
            usage.estimate_cost("grok_fast")
            if tier is ModelTier.STRUCTURAL
            else usage.estimate_cost("claude_sonnet")
        )
        logger.info(
            "LLM usage provider=%s model=%s tier=%s input=%s output=%s cache_read=%s cache_write=%s cost=%.4f",
            provider,
            model,
            tier.value,
            usage.input_tokens,
            usage.output_tokens,
            usage.cache_read_tokens,
            usage.cache_write_tokens,
            cost_hint,
        )

    @staticmethod
    def _structured_error(*, provider: str, tier: str, error_type: str, message: str) -> str:
        payload = {
            "error": {
                "provider": provider,
                "tier": tier,
                "type": error_type,
                "message": message,
            }
        }
        return "[LLM_ERROR] " + json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _looks_like_unknown_model(exc: Exception) -> bool:
        msg = str(exc).lower()
        return "model" in msg and ("not found" in msg or "unknown" in msg or "does not exist" in msg)
