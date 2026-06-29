"""Tests for llama-cpp-python KV cache compatibility patch."""

from __future__ import annotations

import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import models.llm_runner as llm_runner


def _make_fake_llama_modules() -> tuple[types.ModuleType, types.ModuleType]:
    """Build fake llama_cpp modules for patch testing."""

    llama_cpp_pkg = types.ModuleType("llama_cpp")
    llama_cpp_pkg.__path__ = []

    internals = types.ModuleType("llama_cpp._internals")

    class FakeContext:
        """Minimal stand-in for llama_cpp._internals.LlamaContext."""

        def __init__(self) -> None:
            self.memory = object()
            self.ctx = object()
            self.calls: list[tuple[int, int, int]] = []

        def kv_cache_seq_rm(self, seq_id: int, p0: int, p1: int) -> None:
            self.calls.append((seq_id, p0, p1))

    internals.LlamaContext = FakeContext  # type: ignore[attr-defined]
    return llama_cpp_pkg, internals


class TestKvCachePatch:
    """Verify the local 0.3.14 compatibility monkey-patch."""

    def teardown_method(self) -> None:
        llm_runner._LLAMA_CPP_PATCHED = False

    def test_negative_seq_id_is_clamped_to_zero(self) -> None:
        llama_cpp_pkg, internals = _make_fake_llama_modules()

        with patch.dict(
            "sys.modules",
            {
                "llama_cpp": llama_cpp_pkg,
                "llama_cpp._internals": internals,
            },
        ):
            llm_runner._LLAMA_CPP_PATCHED = False
            llm_runner._patch_llama_cpp_kv_cache()

            ctx = internals.LlamaContext()  # type: ignore[attr-defined]
            ctx.kv_cache_seq_rm(-1, 7, -1)

        assert ctx.calls == [(0, 7, -1)]

    def test_patch_is_idempotent(self) -> None:
        llama_cpp_pkg, internals = _make_fake_llama_modules()

        with patch.dict(
            "sys.modules",
            {
                "llama_cpp": llama_cpp_pkg,
                "llama_cpp._internals": internals,
            },
        ):
            llm_runner._LLAMA_CPP_PATCHED = False
            llm_runner._patch_llama_cpp_kv_cache()
            llm_runner._patch_llama_cpp_kv_cache()

            ctx = internals.LlamaContext()  # type: ignore[attr-defined]
            ctx.kv_cache_seq_rm(3, 11, -1)

        assert ctx.calls == [(3, 11, -1)]

    def test_load_llm_model_applies_patch_before_model_init(self, tmp_path: Path) -> None:
        fake_model = tmp_path / "test.gguf"
        fake_model.write_text("dummy", encoding="utf-8")

        fake_llama_cls = MagicMock(return_value="loaded-llm")
        llama_cpp_pkg = types.ModuleType("llama_cpp")
        llama_cpp_pkg.Llama = fake_llama_cls  # type: ignore[attr-defined]

        with (
            patch.dict("sys.modules", {"llama_cpp": llama_cpp_pkg}),
            patch("models.llm_runner._patch_llama_cpp_kv_cache") as mock_patch,
        ):
            result = llm_runner.load_llm_model(str(fake_model), n_gpu_layers=4, n_ctx=1024)

        assert result == "loaded-llm"
        mock_patch.assert_called_once_with()
        fake_llama_cls.assert_called_once_with(
            model_path=str(fake_model),
            n_gpu_layers=4,
            n_ctx=1024,
            flash_attn=True,
            verbose=False,
        )
