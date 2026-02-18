from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from narrativefield.llm.gateway import LLMGateway, ModelTier, UsageStats


class _FakeOpenAIChatCompletions:
    def __init__(self, recorder: dict):
        self._recorder = recorder

    async def create(self, **kwargs):  # type: ignore[no-untyped-def]
        self._recorder["openai_create_kwargs"] = dict(kwargs)
        # Mimic OpenAI response shape: choices[0].message.content + usage.prompt_tokens/completion_tokens
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="structural-ok"))],
            usage=SimpleNamespace(prompt_tokens=12, completion_tokens=34),
        )


class _FakeOpenAIChat:
    def __init__(self, recorder: dict):
        self.completions = _FakeOpenAIChatCompletions(recorder)


class _FakeAsyncOpenAI:
    def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
        self._kwargs = dict(kwargs)
        self._recorder = kwargs.get("_recorder")  # not used; tests patch module attr directly.
        self.chat = _FakeOpenAIChat(kwargs.get("_recorder", {}))


class _FakeAnthropicMessages:
    def __init__(self, recorder: dict):
        self._recorder = recorder

    async def create(self, **kwargs):  # type: ignore[no-untyped-def]
        self._recorder["anthropic_create_kwargs"] = dict(kwargs)
        # Mimic Anthropic response: content blocks + usage object.
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="creative-ok")],
            usage=SimpleNamespace(
                input_tokens=100,
                output_tokens=200,
                cache_read_input_tokens=10,
                cache_creation_input_tokens=20,
            ),
        )


class _FakeAsyncAnthropic:
    def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
        self._kwargs = dict(kwargs)
        self.messages = _FakeAnthropicMessages(kwargs.get("_recorder", {}))


def test_structural_uses_openai_client(monkeypatch: pytest.MonkeyPatch) -> None:
    from narrativefield.llm import gateway as gw_mod

    recorder: dict = {}

    def _fake_async_openai(**kwargs):  # type: ignore[no-untyped-def]
        recorder["openai_ctor_kwargs"] = dict(kwargs)
        # Put recorder on the instance so _FakeOpenAIChatCompletions can store create kwargs.
        kwargs["_recorder"] = recorder
        inst = _FakeAsyncOpenAI(**kwargs)
        inst.chat = _FakeOpenAIChat(recorder)
        return inst

    monkeypatch.setattr(gw_mod, "AsyncOpenAI", _fake_async_openai)
    monkeypatch.setenv("XAI_API_KEY", "xai-test")

    gw = LLMGateway()
    out = asyncio.run(
        gw.generate(
            ModelTier.STRUCTURAL,
            system_prompt="SYS",
            user_prompt="USER",
            max_tokens=123,
        )
    )
    assert out == "structural-ok"
    assert recorder["openai_ctor_kwargs"]["base_url"] == "https://api.x.ai/v1"
    assert recorder["openai_ctor_kwargs"]["api_key"] == "xai-test"
    assert recorder["openai_create_kwargs"]["model"] in {"grok-4-1-fast", "grok-beta"}


def test_creative_uses_anthropic_client(monkeypatch: pytest.MonkeyPatch) -> None:
    from narrativefield.llm import gateway as gw_mod

    recorder: dict = {}

    def _fake_async_anthropic(**kwargs):  # type: ignore[no-untyped-def]
        recorder["anthropic_ctor_kwargs"] = dict(kwargs)
        kwargs["_recorder"] = recorder
        return _FakeAsyncAnthropic(**kwargs)

    monkeypatch.setattr(gw_mod.anthropic, "AsyncAnthropic", _fake_async_anthropic)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anth-test")

    gw = LLMGateway()
    out = asyncio.run(
        gw.generate(
            ModelTier.CREATIVE,
            system_prompt="SYS",
            user_prompt="USER",
            cache_system_prompt=False,
            max_tokens=999,
        )
    )
    assert out == "creative-ok"
    assert recorder["anthropic_ctor_kwargs"]["api_key"] == "anth-test"
    assert recorder["anthropic_create_kwargs"]["model"] == gw.config.creative_model


def test_cache_system_prompt_flag_passed(monkeypatch: pytest.MonkeyPatch) -> None:
    from narrativefield.llm import gateway as gw_mod

    recorder: dict = {}

    def _fake_async_anthropic(**kwargs):  # type: ignore[no-untyped-def]
        kwargs["_recorder"] = recorder
        return _FakeAsyncAnthropic(**kwargs)

    monkeypatch.setattr(gw_mod.anthropic, "AsyncAnthropic", _fake_async_anthropic)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anth-test")

    gw = LLMGateway()
    _ = asyncio.run(
        gw.generate(
            ModelTier.CREATIVE,
            system_prompt="SYS-PROMPT",
            user_prompt="USER",
            cache_system_prompt=True,
        )
    )

    sys_param = recorder["anthropic_create_kwargs"]["system"]
    assert isinstance(sys_param, list)
    assert sys_param[0]["text"] == "SYS-PROMPT"
    assert sys_param[0]["cache_control"] == {"type": "ephemeral"}


def test_generate_batch_respects_concurrency_limit() -> None:
    gw = LLMGateway()

    active = 0
    max_active = 0
    lock = asyncio.Lock()

    async def _fake_generate(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal active, max_active
        async with lock:
            active += 1
            max_active = max(max_active, active)
        await asyncio.sleep(0.02)
        async with lock:
            active -= 1
        return "ok"

    gw.generate = _fake_generate  # type: ignore[assignment]

    reqs = [{"system": "s", "user": str(i)} for i in range(25)]
    out = asyncio.run(gw.generate_batch(ModelTier.STRUCTURAL, reqs, max_concurrency=3))
    assert out == ["ok"] * 25
    assert max_active <= 3


def test_usage_stats_cost_estimation() -> None:
    stats = UsageStats(input_tokens=1_000_000, output_tokens=2_000_000, cache_read_tokens=1_000_000, cache_write_tokens=0)
    # Haiku 4.5: $0.80 + $8.00 + $0.08 = $8.88
    assert abs(stats.estimated_cost_usd - 8.88) < 1e-6


def test_retry_on_transient_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gateway retries on 529/overloaded errors."""
    from narrativefield.llm import gateway as gw_mod

    class _OverloadedError(Exception):
        status_code = 529

    recorder: dict[str, object] = {"calls": 0, "sleep_delays": []}

    async def _fake_sleep(delay: float) -> None:
        recorder["sleep_delays"].append(float(delay))

    monkeypatch.setattr(gw_mod.asyncio, "sleep", _fake_sleep)
    monkeypatch.setattr(gw_mod.random, "uniform", lambda _a, _b: 1.0)

    class _FlakyMessages:
        async def create(self, **kwargs):  # type: ignore[no-untyped-def]
            recorder["calls"] = int(recorder["calls"]) + 1
            if int(recorder["calls"]) <= 2:
                raise _OverloadedError("overloaded")
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="creative-ok")],
                usage=SimpleNamespace(input_tokens=1, output_tokens=2),
            )

    class _FlakyAnthropic:
        def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
            self.messages = _FlakyMessages()

    monkeypatch.setattr(gw_mod.anthropic, "AsyncAnthropic", _FlakyAnthropic)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anth-test")

    gw = LLMGateway()
    out = asyncio.run(
        gw.generate(
            ModelTier.CREATIVE,
            system_prompt="SYS",
            user_prompt="USER",
        )
    )
    assert out == "creative-ok"
    assert int(recorder["calls"]) == 3


def test_retry_exhaustion_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """After max retries, the original error propagates."""
    from narrativefield.llm import gateway as gw_mod

    class _OverloadedError(Exception):
        status_code = 529

    recorder: dict[str, int] = {"calls": 0}

    async def _fake_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr(gw_mod.asyncio, "sleep", _fake_sleep)
    monkeypatch.setattr(gw_mod.random, "uniform", lambda _a, _b: 1.0)

    class _AlwaysFailMessages:
        async def create(self, **kwargs):  # type: ignore[no-untyped-def]
            recorder["calls"] += 1
            raise _OverloadedError("overloaded")

    class _AlwaysFailAnthropic:
        def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
            self.messages = _AlwaysFailMessages()

    monkeypatch.setattr(gw_mod.anthropic, "AsyncAnthropic", _AlwaysFailAnthropic)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anth-test")

    gw = LLMGateway()
    gw.config.retry_max_attempts = 5

    with pytest.raises(_OverloadedError):
        asyncio.run(
            gw.generate(
                ModelTier.CREATIVE,
                system_prompt="SYS",
                user_prompt="USER",
            )
        )

    assert recorder["calls"] == int(gw.config.retry_max_attempts)


def test_creative_deep_downgrades_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """CREATIVE_DEEP falls back to CREATIVE after retry exhaustion."""
    from narrativefield.llm import gateway as gw_mod

    class _OverloadedError(Exception):
        status_code = 529

    recorder: dict[str, object] = {"calls": 0, "kwargs": []}

    async def _fake_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr(gw_mod.asyncio, "sleep", _fake_sleep)
    monkeypatch.setattr(gw_mod.random, "uniform", lambda _a, _b: 1.0)

    class _ThinkingFailsMessages:
        async def create(self, **kwargs):  # type: ignore[no-untyped-def]
            recorder["calls"] = int(recorder["calls"]) + 1
            cast_kwargs = dict(kwargs)
            recorder["kwargs"].append(cast_kwargs)
            if "thinking" in cast_kwargs:
                raise _OverloadedError("overloaded")
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="creative-ok")],
                usage=SimpleNamespace(input_tokens=1, output_tokens=2),
            )

    class _ThinkingFailsAnthropic:
        def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
            self.messages = _ThinkingFailsMessages()

    monkeypatch.setattr(gw_mod.anthropic, "AsyncAnthropic", _ThinkingFailsAnthropic)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anth-test")

    gw = LLMGateway()
    gw.config.retry_max_attempts = 5

    out = asyncio.run(
        gw.generate(
            ModelTier.CREATIVE_DEEP,
            system_prompt="SYS",
            user_prompt="USER",
            cache_system_prompt=True,
        )
    )

    assert out == "creative-ok"
    assert int(recorder["calls"]) == int(gw.config.retry_max_attempts) + 1

    kwargs_list = recorder["kwargs"]
    assert isinstance(kwargs_list, list)
    assert len(kwargs_list) == int(gw.config.retry_max_attempts) + 1
    # All deep attempts include "thinking"; the final downgraded attempt should not.
    assert all("thinking" in kwargs_list[i] for i in range(int(gw.config.retry_max_attempts)))
    assert "thinking" not in kwargs_list[-1]
