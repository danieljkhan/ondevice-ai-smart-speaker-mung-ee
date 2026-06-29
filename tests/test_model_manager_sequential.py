"""Tests for sequential GPU loading features in core.model_manager.

Covers ModelType enum, MemoryHealth enum, sequential GPU load protocol,
unload/verify logic, preloading, memory health checks, load_all
deprecation, TTS engine unload methods, and pipeline content filter
integration.

All tests run on Windows without actual AI models (pure mock tests).
"""

from __future__ import annotations

import gc
import threading
import warnings
import weakref
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from core.model_manager import (
    GemmaModelLoadResult,
    ManagerConfig,
    MemoryHealth,
    ModelManager,
    ModelState,
    ModelType,
)

# ===================================================================
# Helpers
# ===================================================================


def _make_manager() -> ModelManager:
    """Create a ModelManager with a fake model_dir."""
    return ModelManager(ManagerConfig(model_dir="/fake/models"))


class TestGemmaModelFallback:
    """Verify manager-owned Gemma primary/fallback loading."""

    def test_load_gemma_with_fallback_uses_primary_on_success(self) -> None:
        """Primary load success records the primary path and skips fallback recovery."""
        mm = _make_manager()
        primary = "/models/gemma-e4b.gguf"
        fallback = "/models/gemma-e2b.gguf"
        sentinel = object()
        attempts: list[str] = []

        def fake_load(path: str, *, n_gpu_layers: int, n_ctx: int) -> object:
            attempts.append(path)
            assert n_gpu_layers == 99
            assert n_ctx == 4096
            return sentinel

        with (
            patch("models.llm_runner.load_gemma4_text_llm", side_effect=fake_load),
            patch.object(mm, "_recover_cuda_memory_after_oom") as mock_recover,
        ):
            result = mm.load_gemma_with_fallback(
                primary,
                fallback,
                n_gpu_layers=99,
                n_ctx=4096,
            )

        assert attempts == [primary]
        assert result.model is sentinel
        assert result.model_path_actual == primary
        assert result.fallback_used is False
        assert result.fallback_reason is None
        assert mm.llm is sentinel
        assert mm.current_gpu_model == ModelType.LLM
        mock_recover.assert_not_called()

    def test_load_gemma_with_fallback_loads_fallback_after_primary_failure(self) -> None:
        """Primary load failure records a string reason and loads the fallback path."""
        mm = _make_manager()
        primary = "/models/missing-e4b.gguf"
        fallback = "/models/gemma-e2b.gguf"
        sentinel = object()
        events: list[str] = []

        def fake_load(path: str, *, n_gpu_layers: int, n_ctx: int) -> object:
            events.append(f"load:{path}")
            if path == primary:
                raise FileNotFoundError("primary missing")
            return sentinel

        def fake_recover() -> None:
            events.append("recover")

        with (
            patch("models.llm_runner.load_gemma4_text_llm", side_effect=fake_load),
            patch.object(mm, "_recover_cuda_memory_after_oom", side_effect=fake_recover),
        ):
            result = mm.load_gemma_with_fallback(
                primary,
                fallback,
                n_gpu_layers=99,
                n_ctx=4096,
            )

        assert events == [f"load:{primary}", "recover", f"load:{fallback}"]
        assert result.model is sentinel
        assert result.model_path_actual == fallback
        assert result.fallback_used is True
        assert result.fallback_reason == "primary missing"
        assert mm.llm is sentinel
        assert mm.current_gpu_model == ModelType.LLM

    def test_load_gemma_with_fallback_force_clears_resident_llm(self) -> None:
        """Resident LLM release hooks run before the new primary load lands."""
        mm = _make_manager()
        calls: list[str] = []

        class InnerModel:
            def unload_model(self) -> None:
                calls.append("inner_unload_model")

        class ResidentLlm:
            model = InnerModel()

            def unload(self) -> None:
                calls.append("outer_unload")

        sentinel = object()
        mm._models["llm"] = ResidentLlm()
        mm._current_gpu_model = ModelType.LLM

        with patch("models.llm_runner.load_gemma4_text_llm", return_value=sentinel):
            result = mm.load_gemma_with_fallback(
                "/models/gemma-e4b.gguf",
                "/models/gemma-e2b.gguf",
                n_gpu_layers=99,
                n_ctx=4096,
            )

        assert calls == ["inner_unload_model", "outer_unload"]
        assert result.model is sentinel
        assert mm.llm is sentinel
        assert mm.current_gpu_model == ModelType.LLM

    def test_load_gemma_with_fallback_skips_fallback_when_paths_match(self) -> None:
        """Identical resolved primary/fallback paths are attempted only once."""
        mm = _make_manager()
        path = "/models/gemma-single.gguf"
        attempts: list[str] = []

        def fake_load(model_path: str, *, n_gpu_layers: int, n_ctx: int) -> object:
            attempts.append(model_path)
            raise RuntimeError("load failed")

        with (
            patch("models.llm_runner.load_gemma4_text_llm", side_effect=fake_load),
            patch.object(mm, "_recover_cuda_memory_after_oom") as mock_recover,
            pytest.raises(RuntimeError, match="load failed"),
        ):
            mm.load_gemma_with_fallback(
                path,
                path,
                n_gpu_layers=99,
                n_ctx=4096,
            )

        assert attempts == [path]
        assert mm.current_gpu_model == ModelType.NONE
        mock_recover.assert_not_called()


# ===================================================================
# TestModelTypeEnum
# ===================================================================


class TestModelTypeEnum:
    """Verify ModelType enum values and membership."""

    def test_none_value(self) -> None:
        """ModelType.NONE has value 'none'."""
        assert ModelType.NONE.value == "none"

    def test_stt_value(self) -> None:
        """ModelType.STT has value 'stt'."""
        assert ModelType.STT.value == "stt"

    def test_llm_value(self) -> None:
        """ModelType.LLM has value 'llm'."""
        assert ModelType.LLM.value == "llm"

    def test_tts_value(self) -> None:
        """ModelType.TTS has value 'tts'."""
        assert ModelType.TTS.value == "tts"

    def test_vad_value(self) -> None:
        """ModelType.VAD has value 'vad'."""
        assert ModelType.VAD.value == "vad"

    def test_all_members_count(self) -> None:
        """ModelType has exactly 5 members."""
        assert len(ModelType) == 5

    def test_importable_from_module(self) -> None:
        """ModelType is importable from core.model_manager."""
        from core.model_manager import ModelType as MT  # noqa: F811

        assert MT is ModelType


# ===================================================================
# TestMemoryHealthEnum
# ===================================================================


class TestMemoryHealthEnum:
    """Verify MemoryHealth enum values."""

    def test_normal_value(self) -> None:
        """MemoryHealth.NORMAL has value 'normal'."""
        assert MemoryHealth.NORMAL.value == "normal"

    def test_warning_value(self) -> None:
        """MemoryHealth.WARNING has value 'warning'."""
        assert MemoryHealth.WARNING.value == "warning"

    def test_critical_value(self) -> None:
        """MemoryHealth.CRITICAL has value 'critical'."""
        assert MemoryHealth.CRITICAL.value == "critical"

    def test_all_members_count(self) -> None:
        """MemoryHealth has exactly 3 members."""
        assert len(MemoryHealth) == 3


# ===================================================================
# TestSequentialLoadInit
# ===================================================================


class TestSequentialLoadInit:
    """Verify sequential loading state after __init__."""

    def test_current_gpu_model_starts_none(self) -> None:
        """_current_gpu_model starts as ModelType.NONE."""
        mm = _make_manager()
        assert mm._current_gpu_model == ModelType.NONE

    def test_current_gpu_model_property(self) -> None:
        """current_gpu_model property exposes _current_gpu_model."""
        mm = _make_manager()
        assert mm.current_gpu_model == ModelType.NONE

    def test_preload_lock_exists(self) -> None:
        """_preload_lock is a threading.Lock."""
        mm = _make_manager()
        assert isinstance(mm._preload_lock, type(threading.Lock()))

    def test_preload_cancelled_exists(self) -> None:
        """_preload_cancelled is a threading.Event."""
        mm = _make_manager()
        assert isinstance(mm._preload_cancelled, threading.Event)

    def test_preload_cancelled_not_set(self) -> None:
        """_preload_cancelled is not set initially."""
        mm = _make_manager()
        assert not mm._preload_cancelled.is_set()

    def test_preload_thread_none_initially(self) -> None:
        """_preload_thread starts as None."""
        mm = _make_manager()
        assert mm._preload_thread is None


# ===================================================================
# TestInitialize
# ===================================================================


class TestInitialize:
    """Verify initialize() loads only VAD."""

    def test_initialize_calls_load_vad(self) -> None:
        """initialize() calls load_vad."""
        mm = _make_manager()
        with patch.object(mm, "load_vad") as mock_vad:
            mm.initialize()
            mock_vad.assert_called_once()

    def test_initialize_does_not_call_load_stt(self) -> None:
        """initialize() does not load STT."""
        mm = _make_manager()
        with (
            patch.object(mm, "load_vad"),
            patch.object(mm, "load_stt") as mock_stt,
        ):
            mm.initialize()
            mock_stt.assert_not_called()

    def test_initialize_does_not_call_load_llm(self) -> None:
        """initialize() does not load LLM."""
        mm = _make_manager()
        with (
            patch.object(mm, "load_vad"),
            patch.object(mm, "load_llm") as mock_llm,
        ):
            mm.initialize()
            mock_llm.assert_not_called()

    def test_initialize_does_not_call_load_tts(self) -> None:
        """initialize() does not load TTS."""
        mm = _make_manager()
        with (
            patch.object(mm, "load_vad"),
            patch.object(mm, "load_tts") as mock_tts,
        ):
            mm.initialize()
            mock_tts.assert_not_called()

    def test_after_initialize_vad_ready_others_not(self) -> None:
        """After initialize, VAD is ready but others are not."""
        mm = _make_manager()
        # Simulate load_vad setting state to READY
        with patch.object(mm, "load_vad") as mock_vad:

            def _set_vad_ready() -> None:
                mm._status["vad"].state = ModelState.READY
                mm._models["vad"] = MagicMock()

            mock_vad.side_effect = _set_vad_ready
            mm.initialize()
            assert mm.is_ready("vad")
            assert not mm.is_ready("stt")
            assert not mm.is_ready("llm")
            assert not mm.is_ready("tts")


# ===================================================================
# TestSequentialLoad
# ===================================================================


class TestSequentialLoad:
    """Verify load() sequential GPU loading protocol."""

    def test_load_none_is_noop(self) -> None:
        """load(ModelType.NONE) does nothing."""
        mm = _make_manager()
        with (
            patch.object(mm, "load_vad") as mock_vad,
            patch.object(mm, "load_stt") as mock_stt,
        ):
            mm.load(ModelType.NONE)
            mock_vad.assert_not_called()
            mock_stt.assert_not_called()

    def test_load_vad_calls_load_vad(self) -> None:
        """load(ModelType.VAD) calls load_vad (CPU direct)."""
        mm = _make_manager()
        with patch.object(mm, "load_vad") as mock_vad:
            mm.load(ModelType.VAD)
            mock_vad.assert_called_once()

    def test_load_tts_calls_load_tts(self) -> None:
        """load(ModelType.TTS) calls load_tts (CPU direct)."""
        mm = _make_manager()
        with patch.object(mm, "load_tts") as mock_tts:
            mm.load(ModelType.TTS)
            mock_tts.assert_called_once()

    def test_load_tts_unloads_resident_llm_when_projected_memory_exceeds_limit(
        self,
    ) -> None:
        """Pre-TTS gate unloads a resident LLM before loading TTS when over budget."""
        mm = ModelManager(
            ManagerConfig(
                model_dir="/fake/models",
                llm_resident=True,
                memory_limit_mb=6000,
            )
        )
        mm._models["llm"] = MagicMock()
        mm._status["llm"].state = ModelState.READY
        mm._current_gpu_model = ModelType.LLM
        call_order: list[str] = []

        with (
            patch.object(mm, "check_memory_mb", return_value=5600),
            patch.object(
                mm,
                "unload_llm",
                side_effect=lambda: call_order.append("unload_llm"),
            ) as mock_unload_llm,
            patch.object(
                mm,
                "load_tts",
                side_effect=lambda: call_order.append("load_tts"),
            ) as mock_load_tts,
        ):
            mm.load(ModelType.TTS)

        assert call_order == ["unload_llm", "load_tts"]
        mock_unload_llm.assert_called_once_with()
        mock_load_tts.assert_called_once_with()

    def test_load_tts_keeps_resident_llm_when_memory_headroom_is_sufficient(
        self,
    ) -> None:
        """Pre-TTS gate preserves resident LLM on the measured-headroom fast path."""
        mm = ModelManager(
            ManagerConfig(
                model_dir="/fake/models",
                llm_resident=True,
                memory_limit_mb=6000,
            )
        )
        fake_llm = MagicMock()
        mm._models["llm"] = fake_llm
        mm._status["llm"].state = ModelState.READY
        mm._current_gpu_model = ModelType.LLM

        with (
            patch.object(mm, "check_memory_mb", return_value=5400),
            patch.object(mm, "unload_llm") as mock_unload_llm,
            patch.object(mm, "load_tts") as mock_load_tts,
        ):
            mm.load(ModelType.TTS)

        mock_unload_llm.assert_not_called()
        mock_load_tts.assert_called_once_with()
        assert mm.llm is fake_llm

    def test_load_stt_calls_unload_then_load(self) -> None:
        """load(ModelType.STT) calls _unload_current_gpu then load_stt."""
        mm = _make_manager()
        call_order: list[str] = []
        with (
            patch.object(
                mm,
                "_unload_current_gpu",
                side_effect=lambda: call_order.append("unload"),
            ),
            patch.object(
                mm,
                "load_stt",
                side_effect=lambda: call_order.append("load_stt"),
            ),
        ):
            mm.load(ModelType.STT)
            assert call_order == ["unload", "load_stt"]

    def test_load_stt_sets_current_gpu_model(self) -> None:
        """load(ModelType.STT) sets _current_gpu_model to STT."""
        mm = _make_manager()
        with (
            patch.object(mm, "_unload_current_gpu"),
            patch.object(mm, "load_stt"),
        ):
            mm.load(ModelType.STT)
            assert mm.current_gpu_model == ModelType.STT

    def test_load_llm_calls_unload_then_load_full_gpu(self) -> None:
        """load(ModelType.LLM) calls _unload_current_gpu then _load_llm_full_gpu."""
        mm = _make_manager()
        call_order: list[str] = []
        with (
            patch.object(
                mm,
                "_unload_current_gpu",
                side_effect=lambda: call_order.append("unload"),
            ),
            patch.object(
                mm,
                "_load_llm_full_gpu",
                side_effect=lambda: call_order.append("load_llm"),
            ),
        ):
            mm.load(ModelType.LLM)
            assert call_order == ["unload", "load_llm"]

    def test_load_llm_sets_current_gpu_model(self) -> None:
        """load(ModelType.LLM) sets _current_gpu_model to LLM."""
        mm = _make_manager()
        with (
            patch.object(mm, "_unload_current_gpu"),
            patch.object(mm, "_load_llm_full_gpu"),
        ):
            mm.load(ModelType.LLM)
            assert mm.current_gpu_model == ModelType.LLM

    def test_load_stt_skips_when_already_current_and_ready(self) -> None:
        """Loading STT when STT is already the current GPU model and ready skips."""
        mm = _make_manager()
        mm._current_gpu_model = ModelType.STT
        mm._status["stt"].state = ModelState.READY
        # A prior real STT load records its hotword CSV; simulate it so the
        # no-hotword reload correctly skips (matches the model_manager load path).
        mm._active_stt_hotwords_csv = mm._resolve_stt_hotwords_csv(None)
        with (
            patch.object(mm, "_unload_current_gpu") as mock_unload,
            patch.object(mm, "load_stt") as mock_load,
        ):
            mm.load(ModelType.STT)
            mock_unload.assert_not_called()
            mock_load.assert_not_called()

    def test_load_llm_skips_when_already_current_and_ready(self) -> None:
        """Loading LLM when LLM is already current GPU model and ready skips."""
        mm = _make_manager()
        mm._current_gpu_model = ModelType.LLM
        mm._status["llm"].state = ModelState.READY
        with (
            patch.object(mm, "_unload_current_gpu") as mock_unload,
            patch.object(mm, "_load_llm_full_gpu") as mock_load,
        ):
            mm.load(ModelType.LLM)
            mock_unload.assert_not_called()
            mock_load.assert_not_called()

    def test_load_stt_does_not_skip_when_not_ready(self) -> None:
        """STT is current GPU model but not ready: proceeds to load."""
        mm = _make_manager()
        mm._current_gpu_model = ModelType.STT
        mm._status["stt"].state = ModelState.UNLOADED
        with (
            patch.object(mm, "_unload_current_gpu"),
            patch.object(mm, "load_stt") as mock_load,
        ):
            mm.load(ModelType.STT)
            mock_load.assert_called_once()

    def test_load_llm_after_stt_unloads_stt_first(self) -> None:
        """Loading LLM after STT unloads STT via _unload_current_gpu."""
        mm = _make_manager()
        mm._current_gpu_model = ModelType.STT
        unload_called_with: list[ModelType] = []

        def capture_unload() -> None:
            unload_called_with.append(mm._current_gpu_model)
            mm._current_gpu_model = ModelType.NONE

        with (
            patch.object(
                mm,
                "_unload_current_gpu",
                side_effect=capture_unload,
            ),
            patch.object(mm, "_load_llm_full_gpu"),
        ):
            mm.load(ModelType.LLM)
            assert unload_called_with == [ModelType.STT]
            assert mm.current_gpu_model == ModelType.LLM

    def test_load_stt_after_llm_unloads_llm_first(self) -> None:
        """Loading STT after LLM unloads LLM via _unload_current_gpu."""
        mm = _make_manager()
        mm._current_gpu_model = ModelType.LLM
        unload_called_with: list[ModelType] = []

        def capture_unload() -> None:
            unload_called_with.append(mm._current_gpu_model)
            mm._current_gpu_model = ModelType.NONE

        with (
            patch.object(
                mm,
                "_unload_current_gpu",
                side_effect=capture_unload,
            ),
            patch.object(mm, "load_stt"),
        ):
            mm.load(ModelType.STT)
            assert unload_called_with == [ModelType.LLM]
            assert mm.current_gpu_model == ModelType.STT

    def test_load_vad_does_not_affect_gpu_model(self) -> None:
        """load(VAD) does not change _current_gpu_model."""
        mm = _make_manager()
        mm._current_gpu_model = ModelType.STT
        with patch.object(mm, "load_vad"):
            mm.load(ModelType.VAD)
            assert mm.current_gpu_model == ModelType.STT

    def test_load_tts_does_not_affect_gpu_model(self) -> None:
        """load(TTS) does not change _current_gpu_model."""
        mm = _make_manager()
        mm._current_gpu_model = ModelType.LLM
        with patch.object(mm, "load_tts"):
            mm.load(ModelType.TTS)
            assert mm.current_gpu_model == ModelType.LLM

    def test_load_llm_full_gpu_uses_n_gpu_layers_minus_one(self) -> None:
        """_load_llm_full_gpu calls load_llm_model with n_gpu_layers=-1."""
        mm = _make_manager()
        mm._config.llm_model_path = "/fake/model.gguf"
        with (
            patch.object(mm, "_select_gpu_layers", return_value=-1),
            patch(
                "models.llm_runner.load_llm_model",
                return_value=MagicMock(),
            ) as mock_load,
        ):
            mm._load_llm_full_gpu()
            call_kwargs = mock_load.call_args
            assert call_kwargs[1]["n_gpu_layers"] == -1


# ===================================================================
# TestUnloadCurrentGpu
# ===================================================================


class TestUnloadCurrentGpu:
    """Verify _unload_current_gpu behavior."""

    def test_noop_when_none(self) -> None:
        """When _current_gpu_model is NONE, no action is taken."""
        mm = _make_manager()
        with (
            patch.object(mm, "_do_unload") as mock_do,
            patch.object(mm, "_gc_collect") as mock_gc,
        ):
            mm._unload_current_gpu()
            mock_do.assert_not_called()
            mock_gc.assert_not_called()

    def test_calls_model_unload_if_available(self) -> None:
        """Calls model.unload() if the model has an unload method."""
        mm = _make_manager()
        mm._current_gpu_model = ModelType.STT
        mock_model = MagicMock()
        mm._models["stt"] = mock_model
        with (
            patch.object(mm, "_do_unload"),
            patch.object(mm, "_gc_collect"),
            patch.object(mm, "_verify_vram_released", return_value=True),
        ):
            mm._unload_current_gpu()
            mock_model.unload.assert_called_once()

    def test_calls_do_unload_and_gc_collect(self) -> None:
        """Calls _do_unload and _gc_collect during GPU unload."""
        mm = _make_manager()
        mm.config.llm_resident = False
        mm._current_gpu_model = ModelType.LLM
        mm._models["llm"] = MagicMock(spec=[])  # no unload method
        with (
            patch.object(mm, "_do_unload") as mock_do,
            patch.object(mm, "_gc_collect") as mock_gc,
            patch.object(mm, "_verify_vram_released", return_value=True),
        ):
            mm._unload_current_gpu()
            mock_do.assert_called_once_with("llm")
            mock_gc.assert_called_once()

    def test_sets_current_gpu_model_to_none(self) -> None:
        """After unload, _current_gpu_model is NONE."""
        mm = _make_manager()
        mm._current_gpu_model = ModelType.STT
        mm._models["stt"] = MagicMock(spec=[])
        with (
            patch.object(mm, "_do_unload"),
            patch.object(mm, "_gc_collect"),
            patch.object(mm, "_verify_vram_released", return_value=True),
        ):
            mm._unload_current_gpu()
            assert mm.current_gpu_model == ModelType.NONE

    def test_handles_model_unload_exception_gracefully(self) -> None:
        """model.unload() raising an exception does not propagate."""
        mm = _make_manager()
        mm._current_gpu_model = ModelType.STT
        mock_model = MagicMock()
        mock_model.unload.side_effect = RuntimeError("unload failed")
        mm._models["stt"] = mock_model
        with (
            patch.object(mm, "_do_unload"),
            patch.object(mm, "_gc_collect"),
            patch.object(mm, "_verify_vram_released", return_value=True),
        ):
            # Should not raise
            mm._unload_current_gpu()
            assert mm.current_gpu_model == ModelType.NONE

    def test_calls_verify_vram_released(self) -> None:
        """_unload_current_gpu calls _verify_vram_released."""
        mm = _make_manager()
        mm._current_gpu_model = ModelType.STT
        mm._models["stt"] = MagicMock(spec=[])
        with (
            patch.object(mm, "_do_unload"),
            patch.object(mm, "_gc_collect"),
            patch.object(mm, "_verify_vram_released", return_value=True) as mock_verify,
        ):
            mm._unload_current_gpu()
            mock_verify.assert_called_once()

    def test_warns_when_vram_not_released(self) -> None:
        """Logs warning when _verify_vram_released returns False."""
        mm = _make_manager()
        mm.config.llm_resident = False
        mm._current_gpu_model = ModelType.LLM
        mm._models["llm"] = MagicMock(spec=[])
        with (
            patch.object(mm, "_do_unload"),
            patch.object(mm, "_gc_collect"),
            patch.object(mm, "_verify_vram_released", return_value=False),
            patch("core.model_manager.logger") as mock_logger,
        ):
            mm._unload_current_gpu()
            mock_logger.warning.assert_called()
            assert mm.current_gpu_model == ModelType.NONE

    def test_skips_model_unload_when_model_is_none(self) -> None:
        """When model slot is None, model.unload() is not called."""
        mm = _make_manager()
        mm._current_gpu_model = ModelType.STT
        mm._models["stt"] = None
        with (
            patch.object(mm, "_do_unload"),
            patch.object(mm, "_gc_collect"),
            patch.object(mm, "_verify_vram_released", return_value=True),
        ):
            mm._unload_current_gpu()
            assert mm.current_gpu_model == ModelType.NONE


class TestIndividualUnloadHooks:
    """Verify direct unload helpers release model resources."""

    def test_unload_stt_releases_resources_and_resets_slot(self) -> None:
        """unload_stt() releases model hooks and clears the active slot."""
        mm = _make_manager()
        model = MagicMock()
        mm._models["stt"] = model
        mm._current_gpu_model = ModelType.STT
        with (
            patch.object(mm, "_release_model_resources") as mock_release,
            patch.object(mm, "_do_unload") as mock_do_unload,
            patch.object(mm, "_gc_collect"),
            patch.object(mm, "_drop_page_cache"),
        ):
            mm.unload_stt()
            mock_release.assert_called_once_with("stt", model)
            mock_do_unload.assert_called_once_with("stt")
            assert mm.current_gpu_model == ModelType.NONE

    def test_unload_llm_resets_active_slot(self) -> None:
        """unload_llm() clears the active sequential slot."""
        mm = _make_manager()
        mm._models["llm"] = MagicMock(spec=[])
        mm._current_gpu_model = ModelType.LLM
        with (
            patch.object(mm, "_gc_collect"),
            patch.object(mm, "_drop_page_cache"),
        ):
            mm.unload_llm()
            assert mm.current_gpu_model == ModelType.NONE

    def test_unload_llm_clears_gemma_load_result_strong_ref(self) -> None:
        """unload_llm() releases the Gemma result object so GC can reclaim the model."""
        mm = _make_manager()

        class WeakrefableGemma:
            pass

        model = WeakrefableGemma()
        model_ref = weakref.ref(model)
        mm._models["llm"] = model
        mm._latest_gemma_model_load_result = GemmaModelLoadResult(
            model=model,
            model_path_actual="/models/gemma.gguf",
        )

        with patch.object(mm, "_drop_page_cache"):
            mm.unload_llm()
        del model
        gc.collect()

        assert mm._models["llm"] is None
        assert mm.latest_gemma_model_load_result() is None
        assert model_ref() is None

    def test_unload_tts_releases_engine_resources(self) -> None:
        """unload_tts() calls the engine cleanup hook before clearing."""
        mm = _make_manager()
        model = MagicMock()
        mm._models["tts"] = model
        with (
            patch.object(mm, "_release_model_resources") as mock_release,
            patch.object(mm, "_do_unload") as mock_do_unload,
            patch.object(mm, "_gc_collect"),
        ):
            mm.unload_tts()
            mock_release.assert_called_once_with("tts", model)
            mock_do_unload.assert_called_once_with("tts")


class TestTtsResidentMode:
    """Verify resident TTS lifecycle and memory-guard behavior."""

    def test_tts_resident_skips_unload_current_gpu(self) -> None:
        """Resident TTS should survive a subsequent sequential-model load."""
        mm = ModelManager(ManagerConfig(model_dir="fake/models", tts_resident=True))
        fake_tts = MagicMock()
        mm._models["tts"] = fake_tts
        mm._status["tts"].state = ModelState.READY
        mm._current_gpu_model = ModelType.TTS

        with (
            patch.object(mm, "_release_model_resources") as mock_release,
            patch.object(mm, "_do_unload") as mock_do_unload,
            patch.object(mm, "_gc_collect"),
            patch.object(mm, "_verify_vram_released", return_value=True),
            patch.object(mm, "load_stt"),
        ):
            mm.load(ModelType.STT)

        assert mm.tts is fake_tts
        mock_release.assert_not_called()
        mock_do_unload.assert_not_called()
        assert mm.current_gpu_model == ModelType.STT

    def test_tts_resident_false_unloads_on_turn(self) -> None:
        """Default config should keep unloading TTS during stage transitions."""
        mm = _make_manager()
        fake_tts = MagicMock()
        mm._models["tts"] = fake_tts
        mm._status["tts"].state = ModelState.READY
        mm._current_gpu_model = ModelType.TTS

        with (
            patch.object(mm, "_release_model_resources") as mock_release,
            patch.object(mm, "_gc_collect"),
            patch.object(mm, "_verify_vram_released", return_value=True),
            patch.object(mm, "load_stt"),
        ):
            mm.load(ModelType.STT)

        mock_release.assert_called_once_with("tts", fake_tts)
        assert mm.tts is None
        assert mm._status["tts"].state == ModelState.UNLOADED
        assert mm.current_gpu_model == ModelType.STT

    def test_unload_tts_force_true_overrides_resident(self) -> None:
        """force=True must unload TTS even when resident mode is enabled."""
        mm = ModelManager(ManagerConfig(model_dir="fake/models", tts_resident=True))
        fake_tts = MagicMock()
        mm._models["tts"] = fake_tts
        mm._status["tts"].state = ModelState.READY
        mm._current_gpu_model = ModelType.TTS

        with (
            patch.object(mm, "_release_model_resources") as mock_release,
            patch.object(mm, "_gc_collect") as mock_gc,
            patch.object(mm, "_drop_page_cache") as mock_drop_page_cache,
        ):
            mm.unload_tts(force=True)

        mock_release.assert_called_once_with("tts", fake_tts)
        mock_gc.assert_called_once()
        mock_drop_page_cache.assert_called_once()
        assert mm.tts is None
        assert mm._status["tts"].state == ModelState.UNLOADED
        assert mm.current_gpu_model == ModelType.NONE

    def test_guard_tts_resident_memory_unloads_when_low_memory(self) -> None:
        """Low MemAvailable should force resident TTS to unload."""
        mm = ModelManager(ManagerConfig(model_dir="fake/models", tts_resident=True))
        fake_tts = MagicMock()
        mm._models["tts"] = fake_tts
        mm._status["tts"].state = ModelState.READY
        mm._current_gpu_model = ModelType.TTS

        with (
            patch.object(mm, "_get_available_memory_mb", return_value=1024),
            patch.object(mm, "_release_model_resources"),
            patch.object(mm, "_gc_collect"),
            patch.object(mm, "_drop_page_cache"),
        ):
            unloaded = mm.guard_tts_resident_memory()

        assert unloaded is True
        assert mm.tts is None
        assert mm._status["tts"].state == ModelState.UNLOADED
        assert mm.current_gpu_model == ModelType.NONE

    def test_guard_tts_resident_memory_noop_when_not_resident(self) -> None:
        """Non-resident config should leave TTS untouched."""
        mm = _make_manager()
        fake_tts = MagicMock()
        mm._models["tts"] = fake_tts
        mm._status["tts"].state = ModelState.READY

        with patch.object(mm, "unload_tts") as mock_unload_tts:
            unloaded = mm.guard_tts_resident_memory()

        assert unloaded is False
        assert mm.tts is fake_tts
        mock_unload_tts.assert_not_called()

    def test_mungi_tts_resident_env_override_parsed(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """ManagerConfig should parse the MUNGI_TTS_RESIDENT env override."""
        monkeypatch.setenv("MUNGI_TTS_RESIDENT", "1")

        cfg = ManagerConfig(model_dir="fake/models")

        assert cfg.tts_resident is True


class TestSttResidentMode:
    """Verify resident STT retention uses layered eviction before unload."""

    def test_guard_stt_flushes_cache_before_unload(self) -> None:
        """Run LLM-adjacent reclaim before forcing STT unload under memory pressure."""
        mm = ModelManager(ManagerConfig(model_dir="fake/models", stt_resident=True))
        mm._models["stt"] = MagicMock()

        call_order: list[str] = []
        original_flush = mm.flush_llm_prompt_cache

        def _record_flush() -> bool:
            call_order.append("flush")
            return original_flush()

        def _record_drop_page_cache() -> None:
            call_order.append("drop")

        def _record_gc_collect() -> None:
            call_order.append("gc")

        def _record_unload(*args: Any, **kwargs: Any) -> None:
            del args, kwargs
            call_order.append("unload")

        with (
            patch.object(mm, "check_memory_health", return_value=MemoryHealth.NORMAL),
            patch.object(mm, "_get_available_memory_mb", side_effect=[900, 1400]),
            patch.object(
                mm,
                "_drop_page_cache",
                side_effect=_record_drop_page_cache,
            ) as mock_drop_page_cache,
            patch.object(
                mm,
                "_gc_collect",
                side_effect=_record_gc_collect,
            ) as mock_gc_collect,
            patch.object(
                mm,
                "flush_llm_prompt_cache",
                side_effect=_record_flush,
            ) as mock_flush_cache,
            patch.object(
                mm,
                "unload_stt",
                side_effect=_record_unload,
            ) as mock_unload_stt,
        ):
            retained = mm.guard_stt_resident_memory()

        assert retained is True
        mock_flush_cache.assert_called_once_with()
        mock_unload_stt.assert_not_called()
        mock_drop_page_cache.assert_called_once_with()
        mock_gc_collect.assert_called_once_with()
        assert call_order == ["flush", "drop", "gc"]

        call_order.clear()
        with (
            patch.object(mm, "check_memory_health", return_value=MemoryHealth.NORMAL),
            patch.object(mm, "_get_available_memory_mb", side_effect=[900, 900]),
            patch.object(
                mm,
                "_drop_page_cache",
                side_effect=_record_drop_page_cache,
            ) as mock_drop_page_cache,
            patch.object(
                mm,
                "_gc_collect",
                side_effect=_record_gc_collect,
            ) as mock_gc_collect,
            patch.object(
                mm,
                "flush_llm_prompt_cache",
                side_effect=_record_flush,
            ) as mock_flush_cache,
            patch.object(
                mm,
                "unload_stt",
                side_effect=_record_unload,
            ) as mock_unload_stt,
        ):
            retained = mm.guard_stt_resident_memory()

        assert retained is False
        mock_flush_cache.assert_called_once_with()
        mock_unload_stt.assert_called_once_with(force=True)
        mock_drop_page_cache.assert_called_once_with()
        mock_gc_collect.assert_called_once_with()
        assert call_order == ["flush", "drop", "gc", "unload"]


class TestSttProviderEnvOverride:
    """Verify MUNGI_STT_PROVIDER overrides ManagerConfig STT device selection."""

    def test_default_stt_device_is_cpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default STT provider should remain CPU when the env override is absent."""
        monkeypatch.delenv("MUNGI_STT_PROVIDER", raising=False)

        cfg = ManagerConfig()

        assert cfg.stt_device == "cpu"

    def test_mungi_stt_provider_cuda(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """ManagerConfig should accept cuda as an STT provider override."""
        monkeypatch.setenv("MUNGI_STT_PROVIDER", "cuda")

        cfg = ManagerConfig()

        assert cfg.stt_device == "cuda"

    def test_mungi_stt_provider_cpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ManagerConfig should accept cpu as an STT provider override."""
        monkeypatch.setenv("MUNGI_STT_PROVIDER", "cpu")

        cfg = ManagerConfig()

        assert cfg.stt_device == "cpu"

    def test_mungi_stt_provider_invalid_ignored(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Invalid STT provider overrides should be silently ignored."""
        monkeypatch.setenv("MUNGI_STT_PROVIDER", "tpu")

        cfg = ManagerConfig()

        assert cfg.stt_device == "cpu"

    def test_mungi_stt_provider_case_insensitive(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """ManagerConfig should normalize STT provider override casing."""
        monkeypatch.setenv("MUNGI_STT_PROVIDER", "CUDA")

        cfg = ManagerConfig()

        assert cfg.stt_device == "cuda"


class TestTtsActiveEngineRegistration:
    """Verify ModelManager syncs the active Supertonic engine slot."""

    def test_load_tts_registers_active_supertonic_engine(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """load_tts()/unload_tts(force=True) should publish and clear the active engine."""
        from models import tts_runner

        manager = ModelManager(
            ManagerConfig(
                model_dir="/fake/models",
                tts_engine="supertonic",
                tts_model_dir="/fake/tts",
            )
        )

        def fake_load(self: Any) -> None:
            """Mark the engine as loaded without touching the real model package."""
            self._model = MagicMock()
            self._voice_style = MagicMock()

        def fake_unload(self: Any) -> None:
            """Mirror unload state changes without clearing the active-engine slot."""
            self._model = None
            self._voice_style = None

        tts_runner._set_active_supertonic_engine(None)
        monkeypatch.setattr(tts_runner.SupertonicEngine, "load", fake_load)
        monkeypatch.setattr(tts_runner.SupertonicEngine, "unload", fake_unload)

        try:
            with (
                patch.object(manager, "_gc_collect"),
                patch.object(manager, "_drop_page_cache"),
            ):
                manager.load_tts()
                active_engine = tts_runner._ACTIVE_SUPERTONIC_ENGINE
                assert active_engine is not None
                assert active_engine._model_dir == "/fake/tts"
                manager.unload_tts(force=True)
                assert tts_runner._ACTIVE_SUPERTONIC_ENGINE is None
        finally:
            tts_runner._set_active_supertonic_engine(None)

    def test_load_tts_clears_active_engine_on_load_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """load_tts() should clear the active engine slot when Supertonic load fails.

        Verifies the failure-path cleanup contract so a partial load cannot leave
        `_ACTIVE_SUPERTONIC_ENGINE` pointing at a stale instance.
        """
        from models import tts_runner

        manager = ModelManager(
            ManagerConfig(
                model_dir="/fake/models",
                tts_engine="supertonic",
                tts_model_dir="/fake/tts",
            )
        )

        def fake_load(self: Any) -> None:
            """Raise a synthetic failure before the engine becomes usable."""
            raise RuntimeError("simulated load failure")

        tts_runner._set_active_supertonic_engine(None)
        monkeypatch.setattr(tts_runner.SupertonicEngine, "load", fake_load)

        try:
            with (
                patch.object(manager, "_gc_collect"),
                patch.object(manager, "_drop_page_cache"),
            ):
                with pytest.raises(RuntimeError, match="simulated load failure"):
                    manager.load_tts()

            assert tts_runner._ACTIVE_SUPERTONIC_ENGINE is None
        finally:
            tts_runner._set_active_supertonic_engine(None)

    def test_load_tts_raises_for_non_supertonic_engine(self) -> None:
        """ADR 0088: load_tts() raises ValueError for non-supertonic engines."""
        cfg = ManagerConfig(
            model_dir="/fake/models",
            tts_engine="piper",
            tts_model_dir="/fake/tts",
        )
        manager = ModelManager(cfg)

        with pytest.raises(ValueError, match="Unsupported TTS engine"):
            manager.load_tts()


# ===================================================================
# TestVerifyVramReleased
# ===================================================================


class TestVerifyVramReleased:
    """Verify _verify_vram_released behavior."""

    def test_returns_true_on_non_linux(self) -> None:
        """Returns True immediately on non-Linux (Windows dev env)."""
        mm = _make_manager()
        with patch("core.model_manager.sys") as mock_sys:
            mock_sys.platform = "win32"
            result = mm._verify_vram_released()
            assert result is True

    def test_returns_true_on_darwin(self) -> None:
        """Returns True immediately on macOS."""
        mm = _make_manager()
        with patch("core.model_manager.sys") as mock_sys:
            mock_sys.platform = "darwin"
            result = mm._verify_vram_released()
            assert result is True

    def test_returns_true_on_windows(self) -> None:
        """On actual Windows, returns True since platform != linux."""
        mm = _make_manager()
        # No patching — use real sys.platform (win32 on test host)
        result = mm._verify_vram_released()
        assert result is True


# ===================================================================
# TestPreloadStt
# ===================================================================


class TestPreloadStt:
    """Verify preload_stt() behavior."""

    def test_starts_a_thread(self) -> None:
        """preload_stt() creates and starts a daemon thread."""
        mm = _make_manager()
        with patch.object(mm, "load", return_value=None):
            mm.preload_stt()
            assert mm._preload_thread is not None
            assert mm._preload_thread.daemon is True
            mm._preload_thread.join(timeout=2.0)

    def test_thread_name_is_correct(self) -> None:
        """Thread name is 'mungi-stt-preload'."""
        mm = _make_manager()
        with patch.object(mm, "load", return_value=None):
            mm.preload_stt()
            assert mm._preload_thread is not None
            assert mm._preload_thread.name == "mungi-stt-preload"
            mm._preload_thread.join(timeout=2.0)

    def test_skips_if_cancelled(self) -> None:
        """preload_stt() returns immediately if _preload_cancelled is set."""
        mm = _make_manager()
        mm._preload_cancelled.set()
        with patch.object(mm, "load") as mock_load:
            mm.preload_stt()
            assert mm._preload_thread is None
            mock_load.assert_not_called()

    def test_skips_if_stt_already_ready_and_current(self) -> None:
        """Skips if STT is already ready and current GPU model."""
        mm = _make_manager()
        mm._current_gpu_model = ModelType.STT
        mm._status["stt"].state = ModelState.READY
        # Simulate the hotword CSV a prior real STT load would have recorded.
        mm._active_stt_hotwords_csv = mm._resolve_stt_hotwords_csv(None)
        with patch.object(mm, "load") as mock_load:
            mm.preload_stt()
            # Thread should not be started because early return
            assert mm._preload_thread is None
            mock_load.assert_not_called()

    def test_cancel_preload_sets_event(self) -> None:
        """cancel_preload() sets the _preload_cancelled event."""
        mm = _make_manager()
        assert not mm._preload_cancelled.is_set()
        mm.cancel_preload()
        assert mm._preload_cancelled.is_set()


# ===================================================================
# TestCheckMemoryHealth
# ===================================================================


class TestCheckMemoryHealth:
    """Verify check_memory_health() 3-level classification."""

    def test_returns_normal_when_zero(self) -> None:
        """Returns NORMAL when check_memory_mb returns 0 (non-Linux)."""
        mm = _make_manager()
        with patch.object(mm, "check_memory_mb", return_value=0):
            assert mm.check_memory_health() == MemoryHealth.NORMAL

    def test_returns_normal_when_below_4500(self) -> None:
        """Returns NORMAL when usage < 4500 MB."""
        mm = _make_manager()
        with patch.object(mm, "check_memory_mb", return_value=3000):
            assert mm.check_memory_health() == MemoryHealth.NORMAL

    def test_returns_warning_when_4500_to_6500(self) -> None:
        """Returns WARNING when usage is 4500-6500 MB."""
        mm = _make_manager()
        with (
            patch.object(mm, "check_memory_mb", return_value=5000),
            patch.object(mm, "_gc_collect") as mock_gc,
        ):
            result = mm.check_memory_health()
            assert result == MemoryHealth.WARNING
            mock_gc.assert_called_once()

    def test_warning_calls_gc_collect(self) -> None:
        """WARNING level forces a GC collect."""
        mm = _make_manager()
        with (
            patch.object(mm, "check_memory_mb", return_value=4501),
            patch.object(mm, "_gc_collect") as mock_gc,
        ):
            mm.check_memory_health()
            mock_gc.assert_called_once()

    def test_returns_critical_when_above_6500(self) -> None:
        """Returns CRITICAL when usage > 6500 MB."""
        mm = _make_manager()
        with (
            patch.object(mm, "check_memory_mb", return_value=7000),
            patch.object(mm, "unload_stt"),
            patch.object(mm, "unload_llm"),
            patch.object(mm, "unload_tts"),
            patch.object(mm, "_drop_page_cache"),
            patch.object(mm, "_gc_collect"),
        ):
            result = mm.check_memory_health()
            assert result == MemoryHealth.CRITICAL

    def test_critical_unloads_transient_models_and_resets_gpu_model(self) -> None:
        """CRITICAL level unloads transient models and resets _current_gpu_model."""
        mm = _make_manager()
        mm._current_gpu_model = ModelType.STT
        with (
            patch.object(mm, "check_memory_mb", return_value=7500),
            patch.object(mm, "unload_stt") as mock_unload_stt,
            patch.object(mm, "unload_llm") as mock_unload_llm,
            patch.object(mm, "unload_tts") as mock_unload_tts,
            patch.object(mm, "_drop_page_cache") as mock_drop_page_cache,
            patch.object(mm, "_gc_collect") as mock_gc_collect,
        ):
            mm.check_memory_health()
            mock_unload_stt.assert_called_once_with(force=True)
            mock_unload_llm.assert_called_once_with()
            mock_unload_tts.assert_called_once_with(force=True)
            mock_drop_page_cache.assert_called_once_with()
            mock_gc_collect.assert_called_once_with()
            assert mm.current_gpu_model == ModelType.NONE

    def test_critical_memory_preserves_vad(self) -> None:
        """CRITICAL recovery should retain VAD while unloading transient models."""
        mm = _make_manager()
        vad_model = MagicMock()
        mm._models["vad"] = vad_model
        mm._models["stt"] = MagicMock()
        mm._models["llm"] = MagicMock()
        mm._models["tts"] = MagicMock()

        with (
            patch.object(mm, "check_memory_mb", return_value=6700),
            patch.object(mm, "unload_stt") as mock_unload_stt,
            patch.object(mm, "unload_llm") as mock_unload_llm,
            patch.object(mm, "unload_tts") as mock_unload_tts,
            patch.object(mm, "_drop_page_cache") as mock_drop_page_cache,
            patch.object(mm, "_gc_collect") as mock_gc_collect,
        ):
            result = mm.check_memory_health()

        assert result == MemoryHealth.CRITICAL
        assert mm._models["vad"] is vad_model
        mock_unload_stt.assert_called_once_with(force=True)
        mock_unload_llm.assert_called_once_with()
        mock_unload_tts.assert_called_once_with(force=True)
        mock_drop_page_cache.assert_called_once_with()
        mock_gc_collect.assert_called_once()


# ===================================================================
# TestLoadAllDeprecated
# ===================================================================


class TestLoadAllDeprecated:
    """Verify load_all() emits DeprecationWarning."""

    def test_emits_deprecation_warning(self) -> None:
        """load_all() emits a DeprecationWarning."""
        mm = _make_manager()
        with (
            patch.object(mm, "load_vad"),
            patch.object(mm, "load_stt"),
            patch.object(mm, "load_llm"),
            patch.object(mm, "load_tts"),
        ):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                mm.load_all()
                assert len(caught) >= 1
                assert issubclass(caught[0].category, DeprecationWarning)

    def test_warning_message_mentions_initialize(self) -> None:
        """Deprecation message mentions initialize()."""
        mm = _make_manager()
        with (
            patch.object(mm, "load_vad"),
            patch.object(mm, "load_stt"),
            patch.object(mm, "load_llm"),
            patch.object(mm, "load_tts"),
        ):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                mm.load_all()
                assert "initialize()" in str(caught[0].message)

    def test_still_loads_all_models(self) -> None:
        """load_all() still calls all four individual loaders."""
        mm = _make_manager()
        with (
            patch.object(mm, "load_vad") as m_vad,
            patch.object(mm, "load_stt") as m_stt,
            patch.object(mm, "load_llm") as m_llm,
            patch.object(mm, "load_tts") as m_tts,
        ):
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                mm.load_all()
            m_vad.assert_called_once()
            m_stt.assert_called_once()
            m_llm.assert_called_once()
            m_tts.assert_called_once()


# ===================================================================
# TestTTSEngineUnload
# ===================================================================


class TestTTSEngineUnload:
    """Verify TTS engine unload methods."""

    def test_supertonic_unload_clears_model(self) -> None:
        """SupertonicEngine.unload() sets _model to None."""
        from models.tts_runner import SupertonicEngine

        engine = SupertonicEngine(model_dir="/fake/supertonic", voice_style="F1")
        engine._model = MagicMock()
        engine.unload()
        assert engine._model is None

    def test_supertonic_unload_clears_voice_style(self) -> None:
        """SupertonicEngine.unload() sets _voice_style to None."""
        from models.tts_runner import SupertonicEngine

        engine = SupertonicEngine(model_dir="/fake/supertonic", voice_style="F1")
        engine._voice_style = MagicMock()
        engine.unload()
        assert engine._voice_style is None

    def test_supertonic_unload_is_idempotent(self) -> None:
        """Calling unload() twice does not raise."""
        from models.tts_runner import SupertonicEngine

        engine = SupertonicEngine(model_dir="/fake/supertonic", voice_style="F1")
        engine.unload()
        engine.unload()  # Should not raise
        assert engine._model is None


# ===================================================================
# TestPipelineContentFilter
# ===================================================================


class TestPipelineContentFilter:
    """Verify pipeline content filter integration."""

    def test_enable_content_filter_defaults_true(self) -> None:
        """PipelineConfig.enable_content_filter defaults to True."""
        from core.pipeline import PipelineConfig

        cfg = PipelineConfig()
        assert cfg.enable_content_filter is True

    def test_turn_metrics_stt_load_time_defaults_zero(self) -> None:
        """TurnMetrics.stt_load_time_s defaults to 0.0."""
        from core.pipeline import TurnMetrics

        m = TurnMetrics()
        assert m.stt_load_time_s == 0.0

    def test_turn_metrics_llm_load_time_defaults_zero(self) -> None:
        """TurnMetrics.llm_load_time_s defaults to 0.0."""
        from core.pipeline import TurnMetrics

        m = TurnMetrics()
        assert m.llm_load_time_s == 0.0

    def test_turn_metrics_tts_load_time_defaults_zero(self) -> None:
        """TurnMetrics.tts_load_time_s defaults to 0.0."""
        from core.pipeline import TurnMetrics

        m = TurnMetrics()
        assert m.tts_load_time_s == 0.0

    def test_turn_metrics_playback_time_defaults_zero(self) -> None:
        """TurnMetrics.playback_time_s defaults to 0.0."""
        from core.pipeline import TurnMetrics

        m = TurnMetrics()
        assert m.playback_time_s == 0.0

    def test_turn_metrics_content_filter_blocked_defaults_false(
        self,
    ) -> None:
        """TurnMetrics.content_filter_blocked defaults to False."""
        from core.pipeline import TurnMetrics

        m = TurnMetrics()
        assert m.content_filter_blocked is False

    def test_turn_metrics_to_dict_includes_new_fields(self) -> None:
        """TurnMetrics.to_dict() includes load times, playback, and filter flag."""
        from core.pipeline import TurnMetrics

        m = TurnMetrics(
            stt_load_time_s=1.5,
            llm_load_time_s=2.3,
            tts_load_time_s=0.7,
            playback_time_s=1.1,
            content_filter_blocked=True,
        )
        d = m.to_dict()
        assert d["stt_load_time_s"] == 1.5
        assert d["llm_load_time_s"] == 2.3
        assert d["tts_load_time_s"] == 0.7
        assert d["playback_time_s"] == 1.1
        assert d["content_filter_blocked"] is True

    def test_pipeline_accepts_content_filter_param(self) -> None:
        """ConversationPipeline __init__ accepts content_filter kwarg."""
        from core.pipeline import ConversationPipeline, PipelineConfig

        mm = MagicMock()
        cf = MagicMock()
        p = ConversationPipeline(mm, PipelineConfig(), content_filter=cf)
        assert p._content_filter is cf

    def test_filter_text_returns_none_when_filter_is_none(self) -> None:
        """_filter_text returns None when content_filter is None."""
        from core.pipeline import ConversationPipeline, PipelineConfig

        mm = MagicMock()
        p = ConversationPipeline(mm, PipelineConfig(enable_content_filter=True))
        assert p._filter_text("hello") is None

    def test_filter_text_returns_none_when_disabled(self) -> None:
        """_filter_text returns None when enable_content_filter is False."""
        from core.pipeline import ConversationPipeline, PipelineConfig

        mm = MagicMock()
        cf = MagicMock()
        p = ConversationPipeline(
            mm,
            PipelineConfig(enable_content_filter=False),
            content_filter=cf,
        )
        result = p._filter_text("hello")
        assert result is None
        cf.filter.assert_not_called()


# ===================================================================
# TestPipelineSequentialTurn
# ===================================================================


class TestPipelineSequentialTurn:
    """Verify run_turn uses sequential GPU loading and content filter."""

    def _make_pipeline(
        self,
        content_filter: Any = None,
    ) -> Any:
        """Create a pipeline with mocked model_manager."""
        from core.llm_backend_config import LLMBackendConfig
        from core.pipeline import ConversationPipeline, PipelineConfig

        mm = MagicMock()
        cfg = PipelineConfig(enable_content_filter=True)
        legacy = LLMBackendConfig(
            backend="qwen3_legacy",
            model_path=None,
            n_ctx=2048,
            max_tokens=256,
            temperature=0.4,
            n_gpu_layers=99,
        )
        with patch("core.pipeline.LLMBackendConfig.load", return_value=legacy):
            return ConversationPipeline(mm, cfg, content_filter=content_filter)

    def _make_pipeline_gemma_resident(self, *, llm_resident: bool = True) -> Any:
        """Construct a default-Gemma pipeline with resident-mode flag for sequential tests."""
        from core.llm_backend_config import LLMBackendConfig
        from core.pipeline import ConversationPipeline, PipelineConfig

        mm = MagicMock()
        mm.llm = None
        mm.config = MagicMock()
        mm.config.llm_resident = llm_resident
        mm.config.tts_resident = False
        mm.config.stt_resident = False
        mm._config = mm.config
        mm.guard_tts_resident_memory.return_value = False
        mm.guard_stt_resident_memory.return_value = True

        gemma = LLMBackendConfig(
            backend="gemma4_text",
            model_path=None,
            n_ctx=4096,
            max_tokens=256,
            temperature=0.4,
            n_gpu_layers=99,
        )
        cfg = PipelineConfig(enable_content_filter=True)
        with patch("core.pipeline.LLMBackendConfig.load", return_value=gemma):
            return ConversationPipeline(mm, cfg)

    def test_run_turn_calls_mm_load_stt(self) -> None:
        """run_turn calls mm.load(ModelType.STT) before STT."""
        p = self._make_pipeline()
        fake_seg = MagicMock()
        fake_seg.start = 0.0
        fake_seg.end = 0.5

        with (
            patch.object(p, "_run_vad", return_value=[fake_seg]),
            patch.object(p, "_extract_speech", return_value=[0.1] * 8000),
            patch.object(p, "_run_stt", return_value="hello"),
            patch.object(p, "_run_llm", return_value=("hi!", 5, 0.1)),
            patch.object(p, "_run_tts", return_value=([0.0], 22050)),
        ):
            p.run_turn([0.0] * 16000)
            # Check mm.load was called with ModelType.STT
            load_calls = p._mm.load.call_args_list
            stt_calls = [c for c in load_calls if c.args[0] == ModelType.STT]
            assert len(stt_calls) >= 1

    def test_run_turn_calls_mm_load_llm(self) -> None:
        """run_turn calls mm.load(ModelType.LLM) before LLM generation."""
        p = self._make_pipeline()
        fake_seg = MagicMock()
        fake_seg.start = 0.0
        fake_seg.end = 0.5

        with (
            patch.object(p, "_run_vad", return_value=[fake_seg]),
            patch.object(p, "_extract_speech", return_value=[0.1] * 8000),
            patch.object(p, "_run_stt", return_value="hello"),
            patch.object(p, "_run_llm", return_value=("hi!", 5, 0.1)),
            patch.object(p, "_run_tts", return_value=([0.0], 22050)),
        ):
            p.run_turn([0.0] * 16000)
            load_calls = p._mm.load.call_args_list
            llm_calls = [c for c in load_calls if c.args[0] == ModelType.LLM]
            assert len(llm_calls) >= 1

    def test_gemma_run_turn_dispatches_llm_via_fallback_loader(self) -> None:
        """Gemma default: run_turn dispatches LLM via manager fallback loader."""
        p = self._make_pipeline_gemma_resident(llm_resident=False)
        sentinel_llm = MagicMock(name="sentinel_llm")
        load_result = SimpleNamespace(
            model=sentinel_llm,
            model_path_actual="/models/gemma.gguf",
            fallback_used=False,
            fallback_reason=None,
        )

        def fake_load_gemma_with_fallback(*_args: Any, **_kwargs: Any) -> Any:
            p._mm.llm = sentinel_llm
            return load_result

        p._mm.load_gemma_with_fallback.side_effect = fake_load_gemma_with_fallback

        fake_seg = MagicMock()
        fake_seg.start = 0.0
        fake_seg.end = 0.5

        with (
            patch.object(p, "_run_vad", return_value=[fake_seg]),
            patch.object(p, "_extract_speech", return_value=[0.1] * 8000),
            patch.object(p, "_run_stt", return_value="hello"),
            patch.object(p, "_run_llm", return_value=("hi!", 5, 0.1)),
            patch.object(p, "_run_tts", return_value=([0.0], 22050)),
        ):
            result = p.run_turn([0.0] * 16000)

        assert result.success is True
        assert p._mm.llm is sentinel_llm
        p._mm.load_gemma_with_fallback.assert_called_once()
        load_args = [c.args[0] for c in p._mm.load.call_args_list]
        assert ModelType.LLM not in load_args

    def test_gemma_run_turn_stt_load_before_llm_fallback_loader(self) -> None:
        """Gemma default: mm.load(STT) precedes Gemma fallback loader in time."""
        p = self._make_pipeline_gemma_resident(llm_resident=False)
        sentinel_llm = MagicMock(name="sentinel_llm")
        call_order: list[tuple[str, Any]] = []
        load_result = SimpleNamespace(
            model=sentinel_llm,
            model_path_actual="/models/gemma.gguf",
            fallback_used=False,
            fallback_reason=None,
        )

        def fake_load_gemma_with_fallback(*_args: Any, **_kwargs: Any) -> Any:
            p._mm.llm = sentinel_llm
            call_order.append(("load_gemma_with_fallback", "llm"))
            return load_result

        def fake_load(model_type: Any) -> None:
            call_order.append(("load", model_type))

        p._mm.load_gemma_with_fallback.side_effect = fake_load_gemma_with_fallback
        p._mm.load.side_effect = fake_load

        fake_seg = MagicMock()
        fake_seg.start = 0.0
        fake_seg.end = 0.5

        with (
            patch.object(p, "_run_vad", return_value=[fake_seg]),
            patch.object(p, "_extract_speech", return_value=[0.1] * 8000),
            patch.object(p, "_run_stt", return_value="hello"),
            patch.object(p, "_run_llm", return_value=("hi!", 5, 0.1)),
            patch.object(p, "_run_tts", return_value=([0.0], 22050)),
        ):
            result = p.run_turn([0.0] * 16000)

        assert result.success is True
        stt_idx = next(
            (
                i
                for i, (kind, key) in enumerate(call_order)
                if kind == "load" and key == ModelType.STT
            ),
            None,
        )
        llm_idx = next(
            (
                i
                for i, (kind, key) in enumerate(call_order)
                if kind == "load_gemma_with_fallback" and key == "llm"
            ),
            None,
        )
        assert stt_idx is not None, f"STT load missing from trace: {call_order!r}"
        assert llm_idx is not None, f"LLM dispatch missing from trace: {call_order!r}"
        assert stt_idx < llm_idx, f"STT must precede LLM: {call_order!r}"

    def test_run_turn_stt_before_llm(self) -> None:
        """mm.load(STT) is called before mm.load(LLM)."""
        p = self._make_pipeline()
        fake_seg = MagicMock()
        fake_seg.start = 0.0
        fake_seg.end = 0.5

        call_order: list[str] = []

        def track_load(model_type: ModelType) -> None:
            call_order.append(model_type.value)

        p._mm.load = track_load

        with (
            patch.object(p, "_run_vad", return_value=[fake_seg]),
            patch.object(p, "_extract_speech", return_value=[0.1] * 8000),
            patch.object(p, "_run_stt", return_value="hello"),
            patch.object(p, "_run_llm", return_value=("hi!", 5, 0.1)),
            patch.object(p, "_run_tts", return_value=([0.0], 22050)),
        ):
            p.run_turn([0.0] * 16000)
            stt_idx = call_order.index("stt")
            llm_idx = call_order.index("llm")
            assert stt_idx < llm_idx

    def test_run_turn_content_filter_blocks_input(self) -> None:
        """run_turn with content filter blocking input returns SAFE_FALLBACK."""
        from safety.content_filter import SAFE_FALLBACK_RESPONSE

        mock_filter = MagicMock()
        mock_result = MagicMock()
        mock_result.allowed = False
        mock_result.violations = ["blocked_word"]
        mock_filter.filter.return_value = mock_result

        p = self._make_pipeline(content_filter=mock_filter)
        fake_seg = MagicMock()
        fake_seg.start = 0.0
        fake_seg.end = 0.5

        with (
            patch.object(p, "_run_vad", return_value=[fake_seg]),
            patch.object(p, "_extract_speech", return_value=[0.1] * 8000),
            patch.object(p, "_run_stt", return_value="bad word"),
        ):
            result = p.run_turn([0.0] * 16000)
            assert result.response_text == SAFE_FALLBACK_RESPONSE
            assert result.metrics.content_filter_blocked is True

    def test_run_turn_content_filter_blocks_output(self) -> None:
        """run_turn with content filter blocking output uses filtered text."""
        mock_filter = MagicMock()

        # Input passes, output blocked
        input_result = MagicMock()
        input_result.allowed = True
        output_result = MagicMock()
        output_result.allowed = False
        output_result.violations = ["bad_output"]
        output_result.filtered = "safe response"
        mock_filter.filter.side_effect = [input_result, output_result]

        p = self._make_pipeline(content_filter=mock_filter)
        fake_seg = MagicMock()
        fake_seg.start = 0.0
        fake_seg.end = 0.5

        with (
            patch.object(p, "_run_vad", return_value=[fake_seg]),
            patch.object(p, "_extract_speech", return_value=[0.1] * 8000),
            patch.object(p, "_run_stt", return_value="hello"),
            patch.object(p, "_run_llm", return_value=("bad output", 5, 0.1)),
            patch.object(p, "_run_tts", return_value=([0.0], 22050)),
        ):
            result = p.run_turn([0.0] * 16000)
            assert result.response_text == "safe response"
            assert result.metrics.content_filter_blocked is True

    def test_run_turn_no_filter_passes_through(self) -> None:
        """run_turn without content filter does not block anything."""
        p = self._make_pipeline(content_filter=None)
        fake_seg = MagicMock()
        fake_seg.start = 0.0
        fake_seg.end = 0.5

        with (
            patch.object(p, "_run_vad", return_value=[fake_seg]),
            patch.object(p, "_extract_speech", return_value=[0.1] * 8000),
            patch.object(p, "_run_stt", return_value="hello"),
            patch.object(p, "_run_llm", return_value=("response!", 5, 0.1)),
            patch.object(p, "_run_tts", return_value=([0.0], 22050)),
        ):
            result = p.run_turn([0.0] * 16000)
            assert result.response_text == "response!"
            assert result.metrics.content_filter_blocked is False

    def test_run_turn_records_stt_load_time(self) -> None:
        """run_turn records stt_load_time_s in metrics."""
        p = self._make_pipeline()
        fake_seg = MagicMock()
        fake_seg.start = 0.0
        fake_seg.end = 0.5

        with (
            patch.object(p, "_run_vad", return_value=[fake_seg]),
            patch.object(p, "_extract_speech", return_value=[0.1] * 8000),
            patch.object(p, "_run_stt", return_value="hello"),
            patch.object(p, "_run_llm", return_value=("hi!", 5, 0.1)),
            patch.object(p, "_run_tts", return_value=([0.0], 22050)),
        ):
            result = p.run_turn([0.0] * 16000)
            # stt_load_time_s should be >= 0 (time around mm.load call)
            assert result.metrics.stt_load_time_s >= 0.0

    def test_run_turn_records_llm_load_time(self) -> None:
        """run_turn records llm_load_time_s in metrics."""
        p = self._make_pipeline()
        fake_seg = MagicMock()
        fake_seg.start = 0.0
        fake_seg.end = 0.5

        with (
            patch.object(p, "_run_vad", return_value=[fake_seg]),
            patch.object(p, "_extract_speech", return_value=[0.1] * 8000),
            patch.object(p, "_run_stt", return_value="hello"),
            patch.object(p, "_run_llm", return_value=("hi!", 5, 0.1)),
            patch.object(p, "_run_tts", return_value=([0.0], 22050)),
        ):
            result = p.run_turn([0.0] * 16000)
            assert result.metrics.llm_load_time_s >= 0.0


# ===================================================================
# TestCtranslate2UnloadModel (방안 1)
# ===================================================================


class TestCtranslate2UnloadModel:
    """Verify _unload_current_gpu calls ctranslate2 unload_model."""

    def test_calls_inner_unload_model_for_stt(self) -> None:
        """STT model's inner .model.unload_model() is called."""
        mm = _make_manager()
        mm._current_gpu_model = ModelType.STT
        inner = MagicMock()
        inner.unload_model = MagicMock()
        mock_model = MagicMock()
        mock_model.model = inner
        # Ensure hasattr(model, "unload") is False
        del mock_model.unload
        mm._models["stt"] = mock_model
        with (
            patch.object(mm, "_do_unload"),
            patch.object(mm, "_gc_collect"),
            patch.object(mm, "_verify_vram_released", return_value=True),
        ):
            mm._unload_current_gpu()
            inner.unload_model.assert_called_once()

    def test_skips_inner_unload_when_no_model_attr(self) -> None:
        """Models without .model attribute skip inner unload."""
        mm = _make_manager()
        mm._current_gpu_model = ModelType.LLM
        mock_model = MagicMock(spec=["some_method"])
        mm._models["llm"] = mock_model
        with (
            patch.object(mm, "_do_unload"),
            patch.object(mm, "_gc_collect"),
            patch.object(mm, "_verify_vram_released", return_value=True),
        ):
            mm._unload_current_gpu()  # Should not raise

    def test_inner_unload_exception_does_not_propagate(self) -> None:
        """Exception in inner unload_model() is caught gracefully."""
        mm = _make_manager()
        mm._current_gpu_model = ModelType.STT
        inner = MagicMock()
        inner.unload_model.side_effect = RuntimeError("ct2 error")
        mock_model = MagicMock()
        mock_model.model = inner
        del mock_model.unload
        mm._models["stt"] = mock_model
        with (
            patch.object(mm, "_do_unload"),
            patch.object(mm, "_gc_collect"),
            patch.object(mm, "_verify_vram_released", return_value=True),
        ):
            mm._unload_current_gpu()  # Should not raise
            assert mm.current_gpu_model == ModelType.NONE

    def test_calls_both_inner_and_outer_unload(self) -> None:
        """Model with both .model.unload_model() and .unload() calls both."""
        mm = _make_manager()
        mm._current_gpu_model = ModelType.TTS
        inner = MagicMock()
        inner.unload_model = MagicMock()
        mock_model = MagicMock()
        mock_model.model = inner
        mm._models["tts"] = mock_model
        with (
            patch.object(mm, "_do_unload"),
            patch.object(mm, "_gc_collect"),
            patch.object(mm, "_verify_vram_released", return_value=True),
        ):
            mm._unload_current_gpu()
            inner.unload_model.assert_called_once()
            mock_model.unload.assert_called_once()


# ===================================================================
# TestGcCollectMallocTrim (방안 1 — malloc_trim)
# ===================================================================


class TestGcCollectMallocTrim:
    """Verify _gc_collect includes malloc_trim on Linux."""

    def test_gc_collect_calls_gc(self) -> None:
        """_gc_collect calls gc.collect()."""
        with (
            patch("core.model_manager.gc.collect") as mock_gc,
            patch.dict("sys.modules", {"torch": MagicMock()}),
        ):
            ModelManager._gc_collect()
            mock_gc.assert_called_once()

    def test_gc_collect_calls_malloc_trim_on_linux(self) -> None:
        """_gc_collect calls malloc_trim(0) on Linux."""
        mock_libc = MagicMock()
        with (
            patch("core.model_manager.gc.collect"),
            patch("core.model_manager.sys") as mock_sys,
            patch("core.model_manager.ctypes.CDLL", return_value=mock_libc),
            patch.dict("sys.modules", {"torch": MagicMock()}),
        ):
            mock_sys.platform = "linux"
            ModelManager._gc_collect()
            mock_libc.malloc_trim.assert_called_once_with(0)

    def test_gc_collect_skips_malloc_trim_on_windows(self) -> None:
        """_gc_collect does not call malloc_trim on Windows."""
        with (
            patch("core.model_manager.gc.collect"),
            patch("core.model_manager.sys") as mock_sys,
            patch("core.model_manager.ctypes.CDLL") as mock_cdll,
            patch.dict("sys.modules", {"torch": MagicMock()}),
        ):
            mock_sys.platform = "win32"
            ModelManager._gc_collect()
            mock_cdll.assert_not_called()


# ===================================================================
# TestSelectGpuLayers (방안 3 — dynamic n_gpu_layers)
# ===================================================================


class TestSelectGpuLayers:
    """Verify _select_gpu_layers dynamic layer selection."""

    def test_full_offload_when_plenty_of_free_memory(self) -> None:
        """Returns -1 when MemFree >= 3000 MB."""
        mm = _make_manager()
        with (
            patch.object(mm, "_ensure_cuda_memory", return_value=3000),
        ):
            assert mm._select_gpu_layers() == -1

    def test_partial_offload_when_moderate_free_memory(self) -> None:
        """Returns 20 when MemFree 2500-2999 MB."""
        mm = _make_manager()
        with patch.object(mm, "_ensure_cuda_memory", return_value=2700):
            assert mm._select_gpu_layers() == 20

    def test_safe_fallback_when_low_free_memory(self) -> None:
        """Returns 10 (min fallback) when MemFree < 2500 MB."""
        mm = _make_manager()
        with patch.object(mm, "_ensure_cuda_memory", return_value=2000):
            assert mm._select_gpu_layers() == 10

    def test_full_offload_on_non_linux(self) -> None:
        """Returns -1 on non-Linux (dev machine)."""
        mm = _make_manager()
        with patch.object(mm, "_ensure_cuda_memory", return_value=0):
            assert mm._select_gpu_layers() == -1

    def test_boundary_3000(self) -> None:
        """Exactly 3000 MB returns -1."""
        mm = _make_manager()
        with patch.object(mm, "_ensure_cuda_memory", return_value=3000):
            assert mm._select_gpu_layers() == -1

    def test_boundary_2500(self) -> None:
        """Exactly 2500 MB returns 20."""
        mm = _make_manager()
        with patch.object(mm, "_ensure_cuda_memory", return_value=2500):
            assert mm._select_gpu_layers() == 20

    def test_boundary_2499(self) -> None:
        """2499 MB returns 10 (min fallback)."""
        mm = _make_manager()
        with patch.object(mm, "_ensure_cuda_memory", return_value=2499):
            assert mm._select_gpu_layers() == 10

    def test_uses_configured_memfree_thresholds(self) -> None:
        """Configured thresholds override the default 5000/4000 policy."""
        mm = ModelManager(
            ManagerConfig(
                model_dir="/fake/models",
                llm_full_offload_memfree_mb=4800,
                llm_partial_offload_memfree_mb=3200,
            )
        )
        with patch.object(mm, "_ensure_cuda_memory", return_value=4800):
            assert mm._select_gpu_layers() == -1
        with patch.object(mm, "_ensure_cuda_memory", return_value=3500):
            assert mm._select_gpu_layers() == 20


# ===================================================================
# TestLoadLlmFullGpuFallback (방안 3 — fallback on ENOMEM)
# ===================================================================


class TestLoadLlmFullGpuFallback:
    """Verify _load_llm_full_gpu retries with safe fallback."""

    def test_success_on_first_try(self) -> None:
        """Loads successfully with chosen layers."""
        mm = _make_manager()
        mm._config.llm_model_path = "/fake/model.gguf"
        with (
            patch.object(mm, "_select_gpu_layers", return_value=-1),
            patch(
                "models.llm_runner.load_llm_model",
                return_value=MagicMock(),
            ) as mock_load,
        ):
            mm._load_llm_full_gpu()
            assert mock_load.call_count == 1
            assert mock_load.call_args[1]["n_gpu_layers"] == -1
            assert mm.latest_llm_load_diagnostics() == {
                "selected_n_gpu_layers": -1,
                "loaded_n_gpu_layers": -1,
                "attempted_n_gpu_layers": [-1],
                "fallback_used": False,
            }

    def test_fallback_on_enomem(self) -> None:
        """Falls back from full offload to 20 layers on CUDA OOM."""
        mm = _make_manager()
        mm._config.llm_model_path = "/fake/model.gguf"
        attempted_layers: list[int] = []

        def _mock_load(*_args: Any, **kwargs: Any) -> MagicMock:
            attempted_layers.append(kwargs["n_gpu_layers"])
            if kwargs.get("n_gpu_layers") == -1:
                raise RuntimeError("ENOMEM")
            return MagicMock()

        with (
            patch.object(mm, "_select_gpu_layers", return_value=-1),
            patch.object(mm, "_recover_cuda_memory_after_oom"),
            patch.object(mm, "_log_llm_load_diagnostics"),
            patch("models.llm_runner.load_llm_model", side_effect=_mock_load),
        ):
            mm._load_llm_full_gpu()
            assert attempted_layers == [-1, 20]

    def test_retry_chain_descends_to_cpu_safe_mode(self) -> None:
        """OOM retries descend through 20, 10, then 0 layers."""
        mm = _make_manager()
        mm._config.llm_model_path = "/fake/model.gguf"
        attempted_layers: list[int] = []

        def _mock_load(*_args: Any, **kwargs: Any) -> MagicMock:
            attempted_layers.append(kwargs["n_gpu_layers"])
            if kwargs["n_gpu_layers"] in {-1, 20, 10}:
                raise RuntimeError("cudaMalloc failed: out of memory")
            return MagicMock()

        with (
            patch.object(mm, "_select_gpu_layers", return_value=-1),
            patch.object(mm, "_recover_cuda_memory_after_oom") as mock_recover,
            patch.object(mm, "_log_llm_load_diagnostics"),
            patch("models.llm_runner.load_llm_model", side_effect=_mock_load),
        ):
            mm._load_llm_full_gpu()
            assert attempted_layers == [-1, 20, 10, 0]
            assert mock_recover.call_count == 3

    def test_non_oom_load_error_does_not_retry(self) -> None:
        """Non-memory load failures propagate without fallback retries."""
        mm = _make_manager()
        mm._config.llm_model_path = "/fake/model.gguf"

        with (
            patch.object(mm, "_select_gpu_layers", return_value=-1),
            patch.object(mm, "_recover_cuda_memory_after_oom") as mock_recover,
            patch.object(mm, "_log_llm_load_diagnostics"),
            patch(
                "models.llm_runner.load_llm_model",
                side_effect=RuntimeError("unknown GGUF metadata"),
            ),
        ):
            try:
                mm._load_llm_full_gpu()
                raised = False
            except RuntimeError:
                raised = True

            assert raised
            mock_recover.assert_not_called()

    def test_generic_llama_load_failure_retries(self) -> None:
        """Generic llama-cpp file-load failures should still try lower offload."""
        mm = _make_manager()
        mm._config.llm_model_path = "/fake/model.gguf"
        attempted_layers: list[int] = []

        def _mock_load(*_args: Any, **kwargs: Any) -> MagicMock:
            attempted_layers.append(kwargs["n_gpu_layers"])
            if kwargs["n_gpu_layers"] == -1:
                raise ValueError("Failed to load model from file: /fake/model.gguf")
            return MagicMock()

        with (
            patch.object(mm, "_select_gpu_layers", return_value=-1),
            patch.object(mm, "_recover_cuda_memory_after_oom"),
            patch.object(mm, "_log_llm_load_diagnostics"),
            patch("models.llm_runner.load_llm_model", side_effect=_mock_load),
        ):
            mm._load_llm_full_gpu()
            assert attempted_layers == [-1, 20]
            assert mm.latest_llm_load_diagnostics()["fallback_used"] is True
            assert mm.latest_llm_load_diagnostics()["loaded_n_gpu_layers"] == 20

    def test_generic_llama_context_failure_retries(self) -> None:
        """Generic llama_context creation failures should still try lower offload."""
        mm = _make_manager()
        mm._config.llm_model_path = "/fake/model.gguf"
        attempted_layers: list[int] = []

        def _mock_load(*_args: Any, **kwargs: Any) -> MagicMock:
            attempted_layers.append(kwargs["n_gpu_layers"])
            if kwargs["n_gpu_layers"] in {-1, 20}:
                raise ValueError("Failed to create llama_context")
            return MagicMock()

        with (
            patch.object(mm, "_select_gpu_layers", return_value=-1),
            patch.object(mm, "_recover_cuda_memory_after_oom"),
            patch.object(mm, "_log_llm_load_diagnostics"),
            patch("models.llm_runner.load_llm_model", side_effect=_mock_load),
        ):
            mm._load_llm_full_gpu()
            assert attempted_layers == [-1, 20, 10]

    def test_auto_discover_gguf_model(self) -> None:
        """Uses find_gguf_model when llm_model_path is None."""
        mm = _make_manager()
        mm._config.llm_model_path = None
        with (
            patch.object(mm, "_select_gpu_layers", return_value=-1),
            patch(
                "models.llm_runner.find_gguf_model",
                return_value="/discovered/model.gguf",
            ),
            patch(
                "models.llm_runner.load_llm_model",
                return_value=MagicMock(),
            ) as mock_load,
        ):
            mm._load_llm_full_gpu()
            assert mock_load.call_args[0][0] == "/discovered/model.gguf"


class TestPasswordlessSudoRecovery:
    """Verify non-interactive sudo gating for cache drop and compaction."""

    def test_drop_page_cache_skips_without_passwordless_sudo(self) -> None:
        mm = _make_manager()
        with (
            patch("core.model_manager.sys.platform", "linux"),
            patch.object(mm, "_can_use_passwordless_sudo", return_value=False),
            patch("core.model_manager.logger.warning") as mock_warning,
            patch("core.model_manager.subprocess.run") as mock_run,
        ):
            mm._drop_page_cache()
            mm._drop_page_cache()

        mock_run.assert_not_called()
        assert mock_warning.call_count == 1

    def test_compact_memory_skips_without_passwordless_sudo(self) -> None:
        mm = _make_manager()
        with (
            patch("core.model_manager.sys.platform", "linux"),
            patch.object(mm, "_can_use_passwordless_sudo", return_value=False),
            patch("core.model_manager.logger.warning") as mock_warning,
            patch("core.model_manager.subprocess.run") as mock_run,
        ):
            mm._compact_memory()
            mm._compact_memory()

        mock_run.assert_not_called()
        assert mock_warning.call_count == 1


# ===================================================================
# TestVerifyVramReleasedUpdated (방안 2)
# ===================================================================


class TestVerifyVramReleasedUpdated:
    """Verify _verify_vram_released uses MemAvailable only."""

    def test_returns_true_on_non_linux(self) -> None:
        """Returns True on Windows (non-Linux)."""
        mm = _make_manager()
        with patch("core.model_manager.sys") as mock_sys:
            mock_sys.platform = "win32"
            assert mm._verify_vram_released() is True

    def test_returns_true_when_memory_available(self) -> None:
        """Returns True when MemAvailable >= 1024 MB."""
        mm = _make_manager()
        with (
            patch("core.model_manager.sys") as mock_sys,
            patch.object(mm, "_get_available_memory_mb", return_value=2000),
        ):
            mock_sys.platform = "linux"
            assert mm._verify_vram_released() is True

    def test_returns_true_when_get_available_returns_zero(self) -> None:
        """Returns True when _get_available_memory_mb returns 0 (read error)."""
        mm = _make_manager()
        with (
            patch("core.model_manager.sys") as mock_sys,
            patch.object(mm, "_get_available_memory_mb", return_value=0),
        ):
            mock_sys.platform = "linux"
            assert mm._verify_vram_released() is True


# ===================================================================
# TestEdgeCases — missing edge-case coverage
# ===================================================================


class TestEdgeCases:
    """Edge-case tests for memory management improvements."""

    def test_ensure_cuda_memory_unexpected_exception_propagates(
        self,
    ) -> None:
        """_select_gpu_layers propagates if _ensure_cuda_memory raises unexpectedly."""
        mm = _make_manager()
        with patch.object(
            mm,
            "_ensure_cuda_memory",
            side_effect=TypeError("unexpected"),
        ):
            try:
                mm._select_gpu_layers()
                raised = False
            except TypeError:
                raised = True
            assert raised

    def test_verify_vram_released_propagates_unexpected_exception(
        self,
    ) -> None:
        """_verify_vram_released propagates if _get_available_memory_mb raises."""
        mm = _make_manager()
        with (
            patch("core.model_manager.sys") as mock_sys,
            patch.object(
                mm,
                "_get_available_memory_mb",
                side_effect=TypeError("bad"),
            ),
        ):
            mock_sys.platform = "linux"
            try:
                mm._verify_vram_released(timeout_s=0.1)
                raised = False
            except TypeError:
                raised = True
            assert raised

    def test_gc_collect_malloc_trim_oserror_suppressed(self) -> None:
        """malloc_trim raising OSError is silently suppressed."""
        mock_libc = MagicMock()
        mock_libc.malloc_trim.side_effect = OSError("trim failed")
        with (
            patch("core.model_manager.gc.collect"),
            patch("core.model_manager.sys") as mock_sys,
            patch("core.model_manager.ctypes.CDLL", return_value=mock_libc),
            patch.dict("sys.modules", {"torch": MagicMock()}),
        ):
            mock_sys.platform = "linux"
            # Should not raise
            ModelManager._gc_collect()

    def test_gc_collect_cdll_oserror_suppressed(self) -> None:
        """ctypes.CDLL raising OSError (libc not found) is suppressed."""
        with (
            patch("core.model_manager.gc.collect"),
            patch("core.model_manager.sys") as mock_sys,
            patch(
                "core.model_manager.ctypes.CDLL",
                side_effect=OSError("libc not found"),
            ),
            patch.dict("sys.modules", {"torch": MagicMock()}),
        ):
            mock_sys.platform = "linux"
            # Should not raise
            ModelManager._gc_collect()

    def test_load_llm_full_gpu_both_attempts_fail(self) -> None:
        """When all staged OOM retries fail, exception propagates."""
        mm = _make_manager()
        mm._config.llm_model_path = "/fake/model.gguf"

        with (
            patch.object(mm, "_select_gpu_layers", return_value=-1),
            patch.object(mm, "_recover_cuda_memory_after_oom"),
            patch.object(mm, "_log_llm_load_diagnostics"),
            patch(
                "models.llm_runner.load_llm_model",
                side_effect=RuntimeError("cudaMalloc failed: out of memory"),
            ),
        ):
            try:
                mm._load_llm_full_gpu()
                raised = False
            except RuntimeError:
                raised = True
            assert raised

    def test_load_llm_full_gpu_find_gguf_returns_none(self) -> None:
        """_load_llm_full_gpu raises FileNotFoundError when no GGUF found."""
        mm = _make_manager()
        mm._config.llm_model_path = None

        with patch(
            "models.llm_runner.find_gguf_model",
            return_value=None,
        ):
            try:
                mm._load_llm_full_gpu()
                raised = False
            except FileNotFoundError:
                raised = True
            assert raised

    def test_llm_gpu_layer_candidates_descend_without_duplicates(self) -> None:
        """Candidate list should progressively reduce offload pressure."""
        mm = _make_manager()
        assert mm._llm_gpu_layer_candidates(-1) == [-1, 20, 10, 0]
        assert mm._llm_gpu_layer_candidates(20) == [20, 10, 0]
        assert mm._llm_gpu_layer_candidates(10) == [10, 0]
        assert mm._llm_gpu_layer_candidates(0) == [0]

    def test_verify_vram_returns_false_when_memory_stays_low(self) -> None:
        """_verify_vram_released returns False when MemAvailable stays below 1024."""
        mm = _make_manager()
        with (
            patch("core.model_manager.sys") as mock_sys,
            patch.object(mm, "_get_available_memory_mb", return_value=500),
            patch("core.model_manager.time.sleep"),
        ):
            mock_sys.platform = "linux"
            result = mm._verify_vram_released(timeout_s=0.01)
            assert result is False
