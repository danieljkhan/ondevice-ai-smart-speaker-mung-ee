"""Focused tests for ManagerConfig environment overrides."""

from __future__ import annotations

import logging
import threading
from unittest.mock import MagicMock, patch

import pytest

from core.model_manager import PRELOAD_JOIN_TIMEOUT, ManagerConfig, ModelManager, ModelType


@pytest.fixture(autouse=True)
def clear_stt_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep STT environment override tests isolated from the host environment."""
    monkeypatch.delenv("MUNGI_STT_MODEL_SIZE", raising=False)
    monkeypatch.delenv("MUNGI_STT_PROVIDER", raising=False)
    monkeypatch.delenv("MUNGI_LLM_RESIDENT", raising=False)


def test_stt_model_size_env_override_applies(monkeypatch: pytest.MonkeyPatch) -> None:
    """MUNGI_STT_MODEL_SIZE should override the default STT model selector."""
    monkeypatch.setenv("MUNGI_STT_MODEL_SIZE", "qwen3-asr")

    cfg = ManagerConfig(model_dir="/fake/models")

    assert cfg.stt_model_size == "qwen3-asr"


def test_stt_model_size_env_empty_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty MUNGI_STT_MODEL_SIZE value should preserve the default."""
    monkeypatch.setenv("MUNGI_STT_MODEL_SIZE", "")

    cfg = ManagerConfig(model_dir="/fake/models")

    assert cfg.stt_model_size == "small"


def test_stt_model_size_env_invalid_logs_warning(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Invalid MUNGI_STT_MODEL_SIZE values should warn and keep the default."""
    monkeypatch.setenv("MUNGI_STT_MODEL_SIZE", "bogus_model")
    caplog.set_level(logging.WARNING, logger="mungi.core.model_manager")

    cfg = ManagerConfig(model_dir="/fake/models")

    assert cfg.stt_model_size == "small"
    assert "Invalid MUNGI_STT_MODEL_SIZE" in caplog.text
    assert "bogus_model" in caplog.text


@pytest.mark.parametrize(
    "alias",
    [
        "qwen3",
        "qwen3-asr-0.6b",
        "qwen3-asr-0.6b-int8",
        "tiny",
    ],
)
def test_stt_model_size_env_various_aliases(
    monkeypatch: pytest.MonkeyPatch,
    alias: str,
) -> None:
    """Supported STT aliases should be accepted as env overrides."""
    monkeypatch.setenv("MUNGI_STT_MODEL_SIZE", alias)

    cfg = ManagerConfig(model_dir="/fake/models")

    assert cfg.stt_model_size == alias


def test_stt_provider_still_honored_with_new_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """MUNGI_STT_PROVIDER and MUNGI_STT_MODEL_SIZE should both apply."""
    monkeypatch.setenv("MUNGI_STT_PROVIDER", "cuda")
    monkeypatch.setenv("MUNGI_STT_MODEL_SIZE", "qwen3-asr")

    cfg = ManagerConfig(model_dir="/fake/models")

    assert cfg.stt_device == "cuda"
    assert cfg.stt_model_size == "qwen3-asr"


def test_default_llm_resident_is_true() -> None:
    """ManagerConfig should keep the LLM resident by default."""
    cfg = ManagerConfig(model_dir="/fake/models")

    assert cfg.llm_resident is True


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("0", False),
        ("1", True),
        ("true", True),
        ("yes", True),
        ("FALSE", False),
    ],
)
def test_mungi_llm_resident_env_override_values(
    monkeypatch: pytest.MonkeyPatch,
    raw_value: str,
    expected: bool,
) -> None:
    """MUNGI_LLM_RESIDENT should override the default with existing bool semantics."""
    monkeypatch.setenv("MUNGI_LLM_RESIDENT", raw_value)

    cfg = ManagerConfig(model_dir="/fake/models")

    assert cfg.llm_resident is expected


def test_mungi_llm_resident_env_invalid_returns_false_with_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid MUNGI_LLM_RESIDENT values should warn and disable resident mode."""
    monkeypatch.setenv("MUNGI_LLM_RESIDENT", "garbage")

    with pytest.warns(UserWarning, match="Ignoring invalid boolean value"):
        cfg = ManagerConfig(model_dir="/fake/models")

    assert cfg.llm_resident is False


def test_mungi_llm_resident_env_unset_preserves_default() -> None:
    """Unset MUNGI_LLM_RESIDENT should preserve the L1 default."""
    cfg = ManagerConfig(model_dir="/fake/models")

    assert cfg.llm_resident is True


def test_preload_stt_skips_duplicate_running_thread() -> None:
    """preload_stt is single-flight while a preload worker is alive."""
    manager = ModelManager(ManagerConfig(model_dir="/fake/models"))
    running_thread = MagicMock()
    running_thread.is_alive.return_value = True
    manager._preload_thread = running_thread

    with patch.object(manager, "load") as load:
        manager.preload_stt()

    load.assert_not_called()


def test_preload_stt_passes_hotwords_to_worker_load() -> None:
    """preload_stt forwards requested hotwords through the managed worker."""
    manager = ModelManager(ManagerConfig(model_dir="/fake/models"))
    load_calls: list[tuple[ModelType, str | None]] = []

    def record_load(model_type: ModelType, *, stt_hotwords_csv: str | None = None) -> None:
        load_calls.append((model_type, stt_hotwords_csv))

    with (
        patch.object(manager, "_resolve_stt_hotwords_csv", return_value="dog"),
        patch.object(manager, "load", side_effect=record_load),
    ):
        manager.preload_stt(stt_hotwords_csv="dog")
        assert manager._preload_thread is not None
        manager._preload_thread.join(timeout=2.0)

    assert load_calls == [(ModelType.STT, "dog")]


def test_reset_preload_state_joins_and_clears_after_thread_finishes() -> None:
    """reset_preload_state waits for an in-flight preload before clearing cancel state."""
    manager = ModelManager(ManagerConfig(model_dir="/fake/models"))
    thread = MagicMock()
    thread.is_alive.side_effect = [True, False, False]
    manager._preload_thread = thread
    manager._preload_cancelled.set()

    manager.reset_preload_state()

    thread.join.assert_called_once_with(timeout=PRELOAD_JOIN_TIMEOUT)
    assert manager._preload_thread is None
    assert manager._preload_cancelled.is_set() is False


def test_unload_stt_joins_running_preload_before_unload() -> None:
    """unload_stt waits briefly for a running preload worker before unloading."""
    manager = ModelManager(ManagerConfig(model_dir="/fake/models"))
    thread = MagicMock()
    thread.is_alive.side_effect = [True, False, False]
    manager._preload_thread = thread

    with (
        patch.object(manager, "_do_unload") as do_unload,
        patch.object(manager, "_gc_collect"),
        patch.object(manager, "_drop_page_cache"),
    ):
        manager.unload_stt()

    thread.join.assert_called_once_with(timeout=PRELOAD_JOIN_TIMEOUT)
    do_unload.assert_called_once_with("stt")
    assert manager._preload_thread is None


def test_load_stt_cancels_and_joins_preload_before_lifecycle_body() -> None:
    """Foreground STT load joins preload before entering the load critical section."""
    manager = ModelManager(ManagerConfig(model_dir="/fake/models"))
    call_order: list[str] = []
    thread = MagicMock()
    thread.is_alive.side_effect = [True, False, False, False]
    thread.join.side_effect = lambda *, timeout: call_order.append(f"join:{timeout}")
    manager._preload_thread = thread

    with (
        patch.object(
            manager,
            "_unload_current_gpu",
            side_effect=lambda: call_order.append("unload"),
        ),
        patch.object(
            manager,
            "load_stt",
            side_effect=lambda: call_order.append("load_stt"),
        ),
    ):
        manager.load(ModelType.STT)

    assert call_order == [f"join:{PRELOAD_JOIN_TIMEOUT}", "unload", "load_stt"]
    thread.join.assert_called_once_with(timeout=PRELOAD_JOIN_TIMEOUT)
    assert manager._preload_thread is None
    assert manager._preload_cancelled.is_set() is False


def test_preload_worker_load_stt_does_not_join_itself() -> None:
    """The preload worker acquires the lifecycle lock without self-joining."""
    manager = ModelManager(ManagerConfig(model_dir="/fake/models"))
    manager._preload_thread = threading.current_thread()

    with (
        patch.object(manager, "cancel_preload_and_join") as cancel_and_join,
        patch.object(manager, "_unload_current_gpu"),
        patch.object(manager, "load_stt") as load_stt,
    ):
        manager.load(ModelType.STT)

    cancel_and_join.assert_not_called()
    load_stt.assert_called_once_with()
