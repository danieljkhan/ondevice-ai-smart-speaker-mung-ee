"""ModelManager tests for Funny English STT hotword lifecycle."""

from __future__ import annotations

from unittest.mock import patch

from core.model_manager import ManagerConfig, ModelManager, ModelState, ModelType


def _ready_stt(manager: ModelManager, *, hotwords_csv: str, current: bool = True) -> None:
    manager._models["stt"] = object()
    manager._status["stt"].state = ModelState.READY
    manager._active_stt_hotwords_csv = hotwords_csv
    if current:
        manager._current_gpu_model = ModelType.STT


def test_stt_current_ready_same_hotwords_skips_reload() -> None:
    """A current-ready STT recognizer is reused when the CSV is unchanged."""
    manager = ModelManager(ManagerConfig(model_dir="/fake/models"))
    _ready_stt(manager, hotwords_csv="cat")

    with (
        patch.object(manager, "unload_stt") as unload_stt,
        patch.object(manager, "_unload_current_gpu") as unload_current,
        patch.object(manager, "load_stt") as load_stt,
    ):
        manager.load(ModelType.STT, stt_hotwords_csv="cat")

    unload_stt.assert_not_called()
    unload_current.assert_not_called()
    load_stt.assert_not_called()


def test_stt_current_ready_changed_hotwords_force_reloads() -> None:
    """A current-ready STT recognizer is force-reloaded when the CSV changes."""
    manager = ModelManager(ManagerConfig(model_dir="/fake/models"))
    _ready_stt(manager, hotwords_csv="cat")

    with (
        patch.object(manager, "unload_stt") as unload_stt,
        patch.object(manager, "_unload_current_gpu") as unload_current,
        patch.object(manager, "load_stt") as load_stt,
    ):
        manager.load(ModelType.STT, stt_hotwords_csv="dog")

    unload_stt.assert_called_once_with(force=True)
    unload_current.assert_called_once_with()
    load_stt.assert_called_once_with(stt_hotwords_csv="dog")
    assert manager._active_stt_hotwords_csv == "dog"


def test_stt_resident_same_hotwords_skips_reload() -> None:
    """A resident STT recognizer is reused only when hotwords match."""
    manager = ModelManager(ManagerConfig(model_dir="/fake/models", stt_resident=True))
    _ready_stt(manager, hotwords_csv="cat", current=False)

    with (
        patch.object(manager, "unload_stt") as unload_stt,
        patch.object(manager, "load_stt") as load_stt,
    ):
        manager.load(ModelType.STT, stt_hotwords_csv="cat")

    unload_stt.assert_not_called()
    load_stt.assert_not_called()
    assert manager.current_gpu_model is ModelType.STT


def test_stt_resident_changed_hotwords_force_reloads_before_skip() -> None:
    """Resident STT cannot preserve a recognizer with stale card hotwords."""
    manager = ModelManager(ManagerConfig(model_dir="/fake/models", stt_resident=True))
    _ready_stt(manager, hotwords_csv="cat", current=False)

    with (
        patch.object(manager, "unload_stt") as unload_stt,
        patch.object(manager, "_unload_current_gpu") as unload_current,
        patch.object(manager, "load_stt") as load_stt,
    ):
        manager.load(ModelType.STT, stt_hotwords_csv="dog")

    unload_stt.assert_called_once_with(force=True)
    unload_current.assert_called_once_with()
    load_stt.assert_called_once_with(stt_hotwords_csv="dog")


def test_load_stt_passes_exact_hotword_csv_to_runner() -> None:
    """The lower STT stack receives a CSV string, never a list."""
    manager = ModelManager(ManagerConfig(model_dir="/fake/models"))

    with patch("models.stt_runner.load_stt_model", return_value=object()) as load_stt_model:
        manager.load_stt(stt_hotwords_csv="cat,dog,뭉이")

    assert load_stt_model.call_args.kwargs["qwen3_asr_hotwords"] == "cat,dog,뭉이"


def test_force_unload_clears_active_hotword_tracking() -> None:
    """A real STT unload clears the active CSV marker."""
    manager = ModelManager(ManagerConfig(model_dir="/fake/models"))
    _ready_stt(manager, hotwords_csv="cat")

    with (
        patch.object(manager, "_release_model_resources"),
        patch.object(manager, "_gc_collect"),
        patch.object(manager, "_drop_page_cache"),
    ):
        manager.unload_stt(force=True)

    assert manager._active_stt_hotwords_csv is None
