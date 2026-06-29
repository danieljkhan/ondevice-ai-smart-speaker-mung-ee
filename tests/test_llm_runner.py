"""Focused tests for llama-cpp chat-generation helpers."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

import models.llm_runner as llm_runner
from models.llm_runner import _resolve_kv_type, run_chat_generation


def test_lowlevel_sampler_chain_api_available() -> None:
    """Assert llama_cpp C-binding sampler symbols exist."""
    llama_c = pytest.importorskip("llama_cpp.llama_cpp")
    required = [
        "llama_sampler_chain_default_params",
        "llama_sampler_chain_init",
        "llama_sampler_chain_add",
        "llama_sampler_init_penalties",
        "llama_sampler_init_top_k",
        "llama_sampler_init_top_p",
        "llama_sampler_init_min_p",
        "llama_sampler_init_temp",
        "llama_sampler_init_dist",
        "llama_sampler_sample",
        "llama_sampler_free",
    ]
    for name in required:
        assert hasattr(llama_c, name), f"Missing: {name}"


def test_run_chat_generation_lowlevel_empty_messages() -> None:
    """Return the skipped-generation tuple for empty low-level chat messages."""
    from models.llm_runner import run_chat_generation_lowlevel

    result = run_chat_generation_lowlevel(None, [])
    assert result == ("", 0, -1.0, 0.0, None, None)


def test_run_chat_generation_lowlevel_formats_evals_and_samples(monkeypatch: Any) -> None:
    """Exercise low-level chat generation without importing real llama_cpp."""
    from models.llm_runner import run_chat_generation_lowlevel

    sampled_tokens = [10, 11, 99]
    accepted_tokens: list[int] = []
    chain = object()
    llama_pkg = ModuleType("llama_cpp")
    llama_c = ModuleType("llama_cpp.llama_cpp")

    def sampler_chain_default_params() -> object:
        return object()

    def sampler_chain_init(_params: object) -> object:
        return chain

    def sampler_chain_add(_chain: object, _sampler: object) -> None:
        return None

    def sampler_factory(*_args: object) -> object:
        return object()

    def sampler_sample(_chain: object, _ctx: object, _index: int) -> int:
        return sampled_tokens.pop(0)

    def sampler_accept(_chain: object, token_id: int) -> None:
        accepted_tokens.append(token_id)

    def sampler_free(_chain: object) -> None:
        return None

    vars(llama_c).update(
        {
            "LLAMA_DEFAULT_SEED": 123,
            "llama_sampler_chain_default_params": sampler_chain_default_params,
            "llama_sampler_chain_init": sampler_chain_init,
            "llama_sampler_chain_add": sampler_chain_add,
            "llama_sampler_init_penalties": sampler_factory,
            "llama_sampler_init_top_k": sampler_factory,
            "llama_sampler_init_top_p": sampler_factory,
            "llama_sampler_init_min_p": sampler_factory,
            "llama_sampler_init_temp": sampler_factory,
            "llama_sampler_init_dist": sampler_factory,
            "llama_sampler_sample": sampler_sample,
            "llama_sampler_accept": sampler_accept,
            "llama_sampler_free": sampler_free,
        }
    )
    vars(llama_pkg)["llama_cpp"] = llama_c
    monkeypatch.setitem(sys.modules, "llama_cpp", llama_pkg)
    monkeypatch.setitem(sys.modules, "llama_cpp.llama_cpp", llama_c)

    class FakeModel:
        def apply_chat_template(self, *_args: object, **kwargs: object) -> str:
            assert kwargs["messages"] == [{"role": "user", "content": "hi"}]
            assert kwargs["enable_thinking"] is False
            return "formatted"

    class FakeLLM:
        def __init__(self) -> None:
            self._model = FakeModel()
            self._ctx = SimpleNamespace(ctx="ctx")
            self.evals: list[list[int]] = []
            self.reset_count = 0
            self.last_n_tokens_size = 8
            self._seed = 42

        def tokenize(self, prompt: bytes) -> list[int]:
            assert prompt == b"formatted"
            return [1, 2, 3]

        def reset(self) -> None:
            self.reset_count += 1

        def eval(self, tokens: list[int]) -> None:
            self.evals.append(tokens)

        def token_eos(self) -> int:
            return 99

        def detokenize(self, tokens: list[int]) -> bytes:
            return {10: b"Hi", 11: b"!"}[tokens[0]]

    fake_llm = FakeLLM()

    result = run_chat_generation_lowlevel(
        fake_llm,
        [{"role": "user", "content": "hi"}],
        max_tokens=5,
        stop=["<stop>"],
    )

    text, token_count, ttft, generation_time, cache_hit_tokens, cache_miss_tokens = result
    assert text == "Hi!"
    assert token_count == 2
    assert ttft >= 0.0
    assert generation_time >= ttft
    assert cache_hit_tokens is None
    assert cache_miss_tokens is None
    assert fake_llm.reset_count == 1
    assert fake_llm.evals == [[1, 2, 3], [10], [11]]
    assert accepted_tokens == [10, 11]


def test_run_chat_generation_passes_cache_prompt_flag() -> None:
    """Pass ``cache_prompt`` when the chat-completion API supports it."""

    class FakeLLM:
        def __init__(self) -> None:
            self.load_state_calls: list[Any] = []
            self.kwargs: dict[str, Any] = {}

        def load_state(self, state: Any) -> None:
            self.load_state_calls.append(state)

        def create_chat_completion(
            self,
            *,
            messages: list[dict[str, str]],
            cache_prompt: bool = False,
            **kwargs: Any,
        ) -> list[dict[str, Any]]:
            self.kwargs = {
                "messages": messages,
                "cache_prompt": cache_prompt,
                **kwargs,
            }
            return [{"choices": [{"delta": {"content": "안녕"}}]}]

    fake_llm = FakeLLM()

    result = run_chat_generation(
        fake_llm,
        [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
        cache_prompt=True,
    )

    assert len(result) == 6
    text, token_count, ttft, gen_time, cache_hit_tokens, cache_miss_tokens = result
    assert text == "안녕"
    assert token_count == 1
    assert ttft >= 0.0
    assert gen_time >= ttft
    assert cache_hit_tokens is not None
    assert cache_miss_tokens is not None
    assert fake_llm.kwargs["cache_prompt"] is True
    assert fake_llm.load_state_calls == []

    snapshot_state = object()
    run_chat_generation(
        fake_llm,
        [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
        cache_prompt=True,
        system_state=snapshot_state,
    )
    assert fake_llm.load_state_calls == [snapshot_state]


def test_prepare_system_state_snapshot_happy_path() -> None:
    """Capture a reusable state snapshot after evaluating the system prompt."""

    call_order: list[str] = []
    sentinel_state = object()

    class FakeLLM:
        def reset(self) -> None:
            call_order.append("reset")

        def tokenize(
            self,
            prompt: bytes,
            *,
            add_bos: bool = False,
            special: bool = False,
        ) -> list[int]:
            call_order.append("tokenize")
            assert prompt == b"system"
            assert add_bos is True
            assert special is True
            return [1, 2, 3]

        def eval(self, tokens: list[int]) -> None:
            call_order.append("eval")
            assert tokens == [1, 2, 3]

        def save_state(self) -> object:
            call_order.append("save_state")
            return sentinel_state

    assert llm_runner.prepare_system_state_snapshot(FakeLLM(), "system") is sentinel_state
    assert call_order == ["reset", "tokenize", "eval", "save_state", "reset"]


def test_prepare_system_state_snapshot_missing_api(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Disable snapshot preparation cleanly when the Llama API is incomplete."""

    class FakeLLM:
        def reset(self) -> None:
            return None

        def eval(self, _tokens: list[int]) -> None:
            return None

        def tokenize(self, _prompt: bytes, **_kwargs: Any) -> list[int]:
            return [1]

    caplog.set_level("WARNING", logger="mungi.models.llm_runner")
    fake_llm = FakeLLM()
    assert llm_runner.prepare_system_state_snapshot(fake_llm, "system") is None
    assert llm_runner.restore_system_state_snapshot(fake_llm, object()) is False
    assert "missing save_state/reset/eval/tokenize" in caplog.text


def test_resolve_kv_type_env_override(monkeypatch: Any) -> None:
    """Resolve KV cache quantization env overrides with safe fallback behavior."""
    monkeypatch.delenv("MUNGI_LLM_KV_TYPE", raising=False)
    assert _resolve_kv_type() is None

    monkeypatch.setenv("MUNGI_LLM_KV_TYPE", "q8_0")
    assert _resolve_kv_type() == 8

    monkeypatch.setenv("MUNGI_LLM_KV_TYPE", "Q4_0")
    assert _resolve_kv_type() == 2

    monkeypatch.setenv("MUNGI_LLM_KV_TYPE", "bogus")
    assert _resolve_kv_type() is None


def test_load_llm_model_applies_kv_type_only_for_opt_in(
    monkeypatch: Any,
    tmp_path: Any,
) -> None:
    """Keep default loader kwargs unchanged unless KV quantization is opted in."""

    captured_kwargs: list[dict[str, Any]] = []

    class FakeLlama:
        def __init__(self, **kwargs: Any) -> None:
            captured_kwargs.append(kwargs)

    monkeypatch.setitem(sys.modules, "llama_cpp", type("FakeModule", (), {"Llama": FakeLlama}))
    monkeypatch.setattr(llm_runner, "_patch_llama_cpp_kv_cache", lambda: None)
    model_path = tmp_path / "model.gguf"
    model_path.write_text("", encoding="utf-8")

    monkeypatch.delenv("MUNGI_LLM_KV_TYPE", raising=False)
    llm_runner.load_llm_model(str(model_path))
    assert "type_k" not in captured_kwargs[0]
    assert "type_v" not in captured_kwargs[0]

    monkeypatch.setenv("MUNGI_LLM_KV_TYPE", "q8_0")
    llm_runner.load_llm_model(str(model_path))
    assert captured_kwargs[1]["type_k"] == 8
    assert captured_kwargs[1]["type_v"] == 8
