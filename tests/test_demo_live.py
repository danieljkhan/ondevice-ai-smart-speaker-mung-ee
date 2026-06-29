"""Tests for the interactive live demo entrypoint."""

from __future__ import annotations

import importlib
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from pytest import MonkeyPatch

from core import system_probe
from core.pipeline import PipelineState, TurnMetrics, TurnResult
from scripts import demo_live


class TestDemoLiveDefaults:
    """Verify live-demo defaults stay aligned with the intended mode split."""

    def test_import_does_not_override_llm_threshold_env(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.delenv("MUNGI_LLM_FULL_OFFLOAD_MEMFREE_MB", raising=False)
        monkeypatch.delenv("MUNGI_LLM_PARTIAL_OFFLOAD_MEMFREE_MB", raising=False)

        importlib.reload(demo_live)

        assert os.getenv("MUNGI_LLM_FULL_OFFLOAD_MEMFREE_MB") is None
        assert os.getenv("MUNGI_LLM_PARTIAL_OFFLOAD_MEMFREE_MB") is None

    def test_pipeline_config_is_tuned_for_live_responsiveness(self) -> None:
        cfg = demo_live._build_pipeline_config()

        assert cfg.play_tts_audio is True
        assert cfg.tts_output_device == demo_live.OUTPUT_DEVICE_NAME
        assert cfg.llm_max_tokens == demo_live.DEMO_LLM_MAX_TOKENS == 80
        assert cfg.llm_temperature == demo_live.DEMO_LLM_TEMPERATURE
        assert cfg.llm_top_p == demo_live.DEMO_LLM_TOP_P
        assert cfg.llm_repeat_penalty == demo_live.DEMO_LLM_REPEAT_PENALTY
        assert cfg.max_history_turns == 1
        assert cfg.max_history_entries == 20
        assert cfg.enable_stt_preload is True

    def test_manager_config_keeps_stt_non_resident(self) -> None:
        """STT must stay non-resident on Jetson 8 GB.

        A 2026-06-01 E2E test showed resident STT pushes total memory over
        the Pre-TTS gate (projected 6078 MB > 6000 MB limit), evicting the
        resident LLM every turn (~9.3 s reload/turn, ~28.5 s turns). Only the LLM
        is resident; STT unloads per turn to free CUDA memory for LLM + TTS.
        """
        cfg = demo_live._build_manager_config()

        assert cfg.stt_resident is False

    def test_manager_config_preserves_cli_stt_model_non_resident(self) -> None:
        cfg = demo_live._build_manager_config("qwen3-asr")

        assert cfg.stt_model_size == "qwen3-asr"
        assert cfg.stt_resident is False

    def test_max_seconds_default_and_env_override(self, monkeypatch: MonkeyPatch) -> None:
        """Verify live-demo recording length defaults to 20s and supports env override."""
        monkeypatch.delenv("MUNGI_DEMO_MAX_RECORDING_SECONDS", raising=False)
        importlib.reload(demo_live)

        assert demo_live.MAX_SECONDS == 20

        monkeypatch.setenv("MUNGI_DEMO_MAX_RECORDING_SECONDS", "25")
        importlib.reload(demo_live)

        assert demo_live.MAX_SECONDS == 25

        monkeypatch.delenv("MUNGI_DEMO_MAX_RECORDING_SECONDS", raising=False)
        importlib.reload(demo_live)

        assert demo_live.MAX_SECONDS == 20

    def test_demo_live_probe_shim_warns_on_import(self) -> None:
        """The one-release system-probe shim emits a deprecation warning at import."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            importlib.reload(demo_live)

        assert demo_live._find_usb_input is system_probe.find_usb_input
        assert any("core.system_probe" in str(item.message) for item in caught)

    def test_main_constructs_session_manager_with_live_pipeline_factory(
        self,
        tmp_path: Path,
    ) -> None:
        """main() builds SessionManager and passes a factory using _build_pipeline_config."""
        cfg = demo_live._build_pipeline_config()
        mock_mm = MagicMock()
        mock_sound_bank = MagicMock()
        mock_event_log = MagicMock()
        mock_audio_capture = MagicMock()
        mock_touch = MagicMock()
        mock_session_manager = MagicMock()
        mock_pipeline = MagicMock()
        mock_content_filter = MagicMock()

        with (
            patch.object(demo_live, "ModelManager", return_value=mock_mm),
            patch.object(demo_live, "SoundBank", return_value=mock_sound_bank),
            patch.object(demo_live, "EventLog", return_value=mock_event_log),
            patch.object(demo_live, "_build_audio_capture", return_value=mock_audio_capture),
            patch.object(demo_live, "TouchInputListener", return_value=mock_touch),
            patch.object(
                demo_live,
                "SessionManager",
                return_value=mock_session_manager,
            ) as session_manager_cls,
            patch.object(demo_live, "ConversationPipeline", return_value=mock_pipeline),
            patch.object(
                demo_live.ContentFilter,
                "from_default",
                return_value=mock_content_filter,
            ) as build_filter,
            patch.object(demo_live, "_build_pipeline_config", return_value=cfg),
            patch.object(demo_live, "detect_runtime_paths") as detect_paths,
        ):
            detect_paths.return_value = SimpleNamespace(mutable_root=str(tmp_path))
            assert demo_live.main() == 0

            kwargs = session_manager_cls.call_args.kwargs
            factory = kwargs["pipeline_factory"]
            assert factory(mock_mm) is mock_pipeline
            build_filter.assert_called_once_with()
            demo_live.ConversationPipeline.assert_called_once_with(
                mock_mm,
                cfg,
                content_filter=mock_content_filter,
            )

        mock_mm.initialize.assert_called_once_with()
        mock_touch.start.assert_called_once_with()
        mock_session_manager.run.assert_called_once_with()
        mock_touch.stop.assert_called_once_with()
        mock_audio_capture.close.assert_called_once_with()
        mock_mm.unload_all.assert_called_once()

        assert kwargs["mm"] is mock_mm
        assert kwargs["touch"] is mock_touch
        assert kwargs["sound_bank"] is mock_sound_bank
        assert kwargs["audio_capture"] is mock_audio_capture
        assert kwargs["event_log"] is mock_event_log

    def test_build_audio_capture_retries_until_usb_input_appears(
        self,
        monkeypatch: MonkeyPatch,
    ) -> None:
        attempts = iter(
            [
                None,
                {"sample_rate": 16000, "channels": 1, "index": 7},
            ]
        )
        sleep_calls: list[float] = []
        now = 100.0
        capture = object()
        audio_capture_cls = MagicMock(return_value=capture)

        def fake_sleep(seconds: float) -> None:
            nonlocal now
            sleep_calls.append(seconds)
            now += seconds

        monkeypatch.setenv(demo_live.USB_AUDIO_WAIT_ENV, "3")
        monkeypatch.setattr(demo_live, "_find_usb_input", lambda **kwargs: next(attempts))
        refresh = MagicMock()
        monkeypatch.setattr(demo_live, "_refresh_sounddevice_devices", refresh)
        monkeypatch.setattr(demo_live.time, "monotonic", lambda: now)
        monkeypatch.setattr(demo_live.time, "sleep", fake_sleep)
        monkeypatch.setattr(demo_live, "AudioCapture", audio_capture_cls)

        assert demo_live._build_audio_capture() is capture
        assert sleep_calls == [0.5]
        refresh.assert_called_once_with()
        audio_capture_cls.assert_called_once_with(sample_rate=16000, channels=1, device=7)

    def test_build_audio_capture_uses_fallback_after_usb_wait_timeout(
        self,
        monkeypatch: MonkeyPatch,
    ) -> None:
        sleep_calls: list[float] = []
        now = 200.0
        capture = object()
        audio_capture_cls = MagicMock(return_value=capture)

        def fake_sleep(seconds: float) -> None:
            nonlocal now
            sleep_calls.append(seconds)
            now += seconds

        def fake_find_usb_input(*, allow_fallback: bool = True) -> dict[str, int] | None:
            if not allow_fallback:
                return None
            return {"sample_rate": 44100, "channels": 2, "index": 3}

        monkeypatch.setenv(demo_live.USB_AUDIO_WAIT_ENV, "2")
        monkeypatch.setattr(demo_live, "_find_usb_input", fake_find_usb_input)
        monkeypatch.setattr(demo_live.time, "monotonic", lambda: now)
        monkeypatch.setattr(demo_live.time, "sleep", fake_sleep)
        monkeypatch.setattr(demo_live, "AudioCapture", audio_capture_cls)

        assert demo_live._build_audio_capture() is capture

        assert sleep_calls == [0.5, 0.5, 0.5, 0.5]
        audio_capture_cls.assert_called_once_with(sample_rate=44100, channels=2, device=3)

    def test_usb_audio_wait_default_is_shortened_and_env_override_still_wins(
        self,
        monkeypatch: MonkeyPatch,
    ) -> None:
        monkeypatch.delenv(demo_live.USB_AUDIO_WAIT_ENV, raising=False)
        assert demo_live._usb_audio_wait_seconds() == 5.0

        monkeypatch.setenv(demo_live.USB_AUDIO_WAIT_ENV, "3")
        assert demo_live._usb_audio_wait_seconds() == 3.0

    def test_build_audio_capture_raises_when_no_input_device_exists(
        self,
        monkeypatch: MonkeyPatch,
    ) -> None:
        now = 300.0
        audio_capture_cls = MagicMock()

        monkeypatch.setenv(demo_live.USB_AUDIO_WAIT_ENV, "0")
        monkeypatch.setattr(demo_live, "_find_usb_input", lambda **kwargs: None)
        monkeypatch.setattr(demo_live.time, "monotonic", lambda: now)
        monkeypatch.setattr(demo_live, "AudioCapture", audio_capture_cls)

        expected_message = "No input-capable audio device found"
        with pytest.raises(RuntimeError, match=expected_message):
            demo_live._build_audio_capture()

        audio_capture_cls.assert_not_called()

    def test_refresh_sounddevice_devices_uses_private_reinitialize(
        self,
        monkeypatch: MonkeyPatch,
    ) -> None:
        """PortAudio refresh uses sounddevice reinitialize hooks when available."""
        calls: list[str] = []
        fake_sd = SimpleNamespace(
            _terminate=lambda: calls.append("terminate"),
            _initialize=lambda: calls.append("initialize"),
            query_devices=lambda: calls.append("query"),
        )
        monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)

        demo_live._refresh_sounddevice_devices()

        assert calls == ["terminate", "initialize"]

    def test_refresh_sounddevice_devices_falls_back_to_query_devices(
        self,
        monkeypatch: MonkeyPatch,
    ) -> None:
        """PortAudio refresh re-queries devices when private hooks are absent."""
        calls: list[str] = []
        fake_sd = SimpleNamespace(query_devices=lambda: calls.append("query"))
        monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)

        demo_live._refresh_sounddevice_devices()

        assert calls == ["query"]


class TestDemoLiveJsonlLogging:
    """Verify per-turn JSONL logging for live demo analysis."""

    def test_resolve_jsonl_path_uses_env_override(
        self, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> None:
        target = tmp_path / "custom.jsonl"
        monkeypatch.setenv(demo_live.JSONL_PATH_ENV, str(target))

        path = demo_live._resolve_jsonl_path()

        assert path == target

    def test_resolve_jsonl_path_defaults_to_conversations_dir(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        original_mutable_root = os.environ.get("MUNGI_MUTABLE_ROOT")
        monkeypatch.delenv("MUNGI_MUTABLE_ROOT", raising=False)
        importlib.reload(demo_live)
        try:
            monkeypatch.delenv(demo_live.JSONL_PATH_ENV, raising=False)
            monkeypatch.setattr(demo_live.sys, "platform", "linux")

            path = demo_live._resolve_jsonl_path(now=datetime(2026, 3, 27, 12, 34, 56))

            assert path.parent == Path("/var/lib/mungi/conversations")
            assert path.name == "demo_2026-03-27_12-34-56.jsonl"
            assert path.suffix == ".jsonl"
        finally:
            if original_mutable_root is None:
                monkeypatch.delenv("MUNGI_MUTABLE_ROOT", raising=False)
            else:
                monkeypatch.setenv("MUNGI_MUTABLE_ROOT", original_mutable_root)
            importlib.reload(demo_live)

    def test_resolve_jsonl_path_honors_mutable_root_env(
        self,
        monkeypatch: MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        original_mutable_root = os.environ.get("MUNGI_MUTABLE_ROOT")
        monkeypatch.setenv("MUNGI_MUTABLE_ROOT", str(tmp_path))
        importlib.reload(demo_live)
        try:
            monkeypatch.delenv(demo_live.JSONL_PATH_ENV, raising=False)
            monkeypatch.setattr(demo_live.sys, "platform", "linux")

            path = demo_live._resolve_jsonl_path(now=datetime(2026, 3, 27, 12, 34, 56))

            assert path.parent == tmp_path / "conversations"
            assert path.name == "demo_2026-03-27_12-34-56.jsonl"
        finally:
            if original_mutable_root is None:
                monkeypatch.delenv("MUNGI_MUTABLE_ROOT", raising=False)
            else:
                monkeypatch.setenv("MUNGI_MUTABLE_ROOT", original_mutable_root)
            importlib.reload(demo_live)

    def test_append_turn_log_writes_jsonl_record(self, tmp_path: Path) -> None:
        log_path = tmp_path / "demo.jsonl"
        result = TurnResult(
            user_text="안녕",
            response_text="안녕! 내 이름은 뭉이야!",
            audio_samples=None,
            sample_rate=24000,
            metrics=TurnMetrics(
                stt_load_time_s=1.2,
                llm_load_time_s=3.4,
                total_time_s=9.8,
                llm_tokens=24,
                llm_model_fallback_used=True,
                llm_model_path_actual="/models/gemma-e2b.gguf",
                llm_model_fallback_reason="primary missing",
            ),
            state=PipelineState.IDLE,
        )

        entry = demo_live._build_turn_log_entry(
            turn=3,
            recorded_duration_s=4.2,
            result=result,
            memory_before_mb={"memfree_mb": 5100},
            memory_after_mb={"memfree_mb": 4700},
            llm_diag={
                "selected_n_gpu_layers": -1,
                "loaded_n_gpu_layers": 20,
                "fallback_used": True,
            },
        )
        demo_live._append_turn_log(log_path, entry)

        payload = json.loads(log_path.read_text(encoding="utf-8").strip())
        assert payload["turn"] == 3
        assert payload["status"] == "ok"
        assert payload["recorded_duration_s"] == 4.2
        assert payload["n_gpu_layers"] == 20
        assert payload["fallback_used"] is True
        assert payload["llm_model_fallback_used"] is True
        assert payload["llm_model_path_actual"] == "/models/gemma-e2b.gguf"
        assert payload["llm_model_fallback_reason"] == "primary missing"
        assert payload["metrics"]["llm_load_time_s"] == 3.4
        assert payload["metrics"]["llm_model_fallback_used"] is True
        assert payload["memory_before_mb"]["memfree_mb"] == 5100
        assert payload["memory_after_mb"]["memfree_mb"] == 4700
        assert payload["response_len"] == len("안녕! 내 이름은 뭉이야!")

    def test_turn_log_marks_no_speech_and_error_states(self) -> None:
        empty_result = TurnResult(
            user_text="",
            response_text="",
            audio_samples=None,
            sample_rate=0,
            metrics=TurnMetrics(),
            state=PipelineState.IDLE,
        )
        error_result = TurnResult(
            user_text="",
            response_text="",
            audio_samples=None,
            sample_rate=0,
            metrics=TurnMetrics(),
            state=PipelineState.ERROR,
            error="boom",
        )

        empty_entry = demo_live._build_turn_log_entry(1, 5.0, empty_result, {}, {}, {})
        error_entry = demo_live._build_turn_log_entry(2, 5.0, error_result, {}, {}, {})

        assert empty_entry["status"] == "no_speech"
        assert error_entry["status"] == "error"

    def test_rotate_jsonl_path_moves_existing_file(self, tmp_path: Path) -> None:
        log_path = tmp_path / "demo.jsonl"
        log_path.write_text("old", encoding="utf-8")

        demo_live._rotate_jsonl_path(log_path)

        assert not log_path.exists()
        assert (tmp_path / "demo.jsonl.bak").read_text(encoding="utf-8") == "old"

    def test_friendly_error_message_for_model_failure(self) -> None:
        msg = demo_live._friendly_error_message("Failed to load model from file")

        assert "모델 로드에 실패" in msg
