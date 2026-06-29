"""Step-3 tests for the Utterance data contract only."""

from __future__ import annotations

import queue
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core.pipeline import (
    VAD_SAMPLE_RATE,
    ConversationPipeline,
    PipelineConfig,
    PipelineState,
    TurnMetrics,
    TurnResult,
    Utterance,
)
from core.pipeline import Utterance as PipelineUtterance
from models.vad_runner import Utterance as StreamingUtterance


def test_utterance_accepts_non_empty_mono_16khz_finite_audio() -> None:
    """Valid mono 16 kHz finite audio constructs an Utterance."""
    audio = np.zeros(160, dtype=np.float32)
    utterance = Utterance(audio=audio, sample_rate=VAD_SAMPLE_RATE)
    assert utterance.audio is audio
    assert utterance.sample_rate == 16_000


def test_utterance_rejects_zero_length_audio() -> None:
    """Zero-length audio cannot be sent to STT."""
    with pytest.raises(ValueError, match="non-empty"):
        Utterance(audio=np.array([], dtype=np.float32), sample_rate=VAD_SAMPLE_RATE)


def test_utterance_rejects_non_mono_audio() -> None:
    """Only mono audio is valid for the VAD-to-STT boundary."""
    with pytest.raises(ValueError, match="mono"):
        Utterance(audio=np.zeros((2, 2), dtype=np.float32), sample_rate=VAD_SAMPLE_RATE)


def test_utterance_rejects_wrong_sample_rate() -> None:
    """The streaming contract is fixed at 16 kHz."""
    with pytest.raises(ValueError, match="16000"):
        Utterance(audio=np.zeros(10, dtype=np.float32), sample_rate=48_000)


def test_utterance_rejects_nan_and_inf() -> None:
    """NaN and infinite samples are rejected before STT."""
    for bad_value in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match="finite"):
            Utterance(audio=np.array([bad_value], dtype=np.float32), sample_rate=VAD_SAMPLE_RATE)


def _make_pipeline(tmp_path: Path) -> ConversationPipeline:
    """Create a pipeline with a mocked model manager and isolated session root."""
    model_manager = MagicMock()
    model_manager.vad = MagicMock()
    model_manager.guard_tts_resident_memory.return_value = False
    model_manager.guard_stt_resident_memory.return_value = True
    model_manager.config = MagicMock()
    model_manager.config.llm_resident = False
    model_manager.config.tts_resident = False
    model_manager.config.stt_resident = False
    model_manager._config = model_manager.config
    pipeline = ConversationPipeline(model_manager, PipelineConfig(play_tts_audio=False))
    pipeline._conversation_dir = tmp_path
    return pipeline


def test_wait_for_utterance_returns_first_streaming_utterance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """wait_for_utterance delegates to streaming VAD and returns the first utterance."""
    pipeline = _make_pipeline(tmp_path)
    audio_queue: queue.Queue[np.ndarray] = queue.Queue()
    capture_stop_event = threading.Event()
    capture = SimpleNamespace(
        audio_queue=audio_queue,
        sample_rate=48_000,
        channels=2,
        stop_event=capture_stop_event,
    )
    streaming_audio = np.ones(512, dtype=np.float32)
    captured: dict[str, object] = {}

    def fake_iter_utterances(
        delegated_queue: queue.Queue[np.ndarray],
        timeout: float,
        *,
        stop_event: object,
    ) -> object:
        captured["queue"] = delegated_queue
        captured["timeout"] = timeout
        captured["stop_event"] = stop_event
        yield StreamingUtterance(audio=streaming_audio, sample_rate=VAD_SAMPLE_RATE)

    monkeypatch.setattr("models.vad_runner.iter_utterances", fake_iter_utterances)

    utterance = pipeline.wait_for_utterance(capture, timeout=3.5)  # type: ignore[arg-type]

    assert isinstance(utterance, PipelineUtterance)
    assert utterance is not None
    assert np.array_equal(utterance.audio, streaming_audio)
    assert utterance.sample_rate == VAD_SAMPLE_RATE
    assert captured["queue"] is audio_queue
    assert captured["timeout"] == 3.5
    assert captured["stop_event"] is capture_stop_event
    assert audio_queue.sample_rate == 48_000
    assert audio_queue.channels == 2
    assert audio_queue.vad_model is pipeline._mm.vad


def test_wait_for_utterance_returns_none_on_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """wait_for_utterance returns None when the streaming wrapper yields nothing."""
    pipeline = _make_pipeline(tmp_path)
    capture = SimpleNamespace(audio_queue=queue.Queue(), sample_rate=16_000, channels=1)

    def fake_iter_utterances(
        delegated_queue: queue.Queue[np.ndarray],
        timeout: float,
        *,
        stop_event: object,
    ) -> object:
        del delegated_queue, timeout, stop_event
        if False:
            yield

    monkeypatch.setattr("models.vad_runner.iter_utterances", fake_iter_utterances)

    assert pipeline.wait_for_utterance(capture, timeout=0.1) is None  # type: ignore[arg-type]


def test_wait_for_utterance_propagates_utterance_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Invalid streaming output is rejected by the core Utterance contract."""
    pipeline = _make_pipeline(tmp_path)
    capture = SimpleNamespace(audio_queue=queue.Queue(), sample_rate=16_000, channels=1)

    def fake_iter_utterances(
        delegated_queue: queue.Queue[np.ndarray],
        timeout: float,
        *,
        stop_event: object,
    ) -> object:
        del delegated_queue, timeout, stop_event
        yield SimpleNamespace(audio=np.zeros((2, 2), dtype=np.float32), sample_rate=VAD_SAMPLE_RATE)

    monkeypatch.setattr("models.vad_runner.iter_utterances", fake_iter_utterances)

    with pytest.raises(ValueError, match="mono"):
        pipeline.wait_for_utterance(capture, timeout=0.1)  # type: ignore[arg-type]


def test_run_turn_with_audio_bypasses_internal_vad(tmp_path: Path) -> None:
    """run_turn_with_audio transcribes the utterance audio without calling batch VAD."""
    pipeline = _make_pipeline(tmp_path)
    # 0.5s of audio (8000 samples @ 16kHz) — above the too-short-capture guard.
    utterance = Utterance(audio=np.ones(8000, dtype=np.float32), sample_rate=VAD_SAMPLE_RATE)
    turn_result = TurnResult(
        user_text="hello",
        response_text="hi",
        audio_samples=np.zeros(10, dtype=np.float32),
        sample_rate=24_000,
        metrics=TurnMetrics(),
        state=PipelineState.IDLE,
    )

    with (
        patch.object(pipeline, "_run_vad") as run_vad,
        patch.object(pipeline, "_extract_speech") as extract_speech,
        patch.object(
            pipeline,
            "_transcribe_speech_audio",
            return_value=("hello", "hello"),
        ) as transcribe,
        patch.object(pipeline, "_save_conversation_audio", return_value=None),
        patch.object(pipeline, "_respond_to_text", return_value=turn_result) as respond,
    ):
        pipeline.run_turn_with_audio(utterance)

    run_vad.assert_not_called()
    extract_speech.assert_not_called()
    transcribe.assert_called_once()
    respond.assert_called_once()


def test_run_turn_with_audio_reprompts_on_too_short_capture(tmp_path: Path) -> None:
    """A truncated (overflow-corrupted) capture is re-prompted, not transcribed.

    Severe audio-input-queue overflow can drop nearly all frames mid-capture,
    leaving a sub-second fragment the fine-tuned STT hallucinates a fixed phrase
    from. The duration guard must short-circuit to a re-prompt before STT runs.
    """
    pipeline = _make_pipeline(tmp_path)
    # 32ms (512 samples @ 16kHz) — below the too-short-capture guard threshold.
    utterance = Utterance(audio=np.ones(512, dtype=np.float32), sample_rate=VAD_SAMPLE_RATE)
    reprompt_result = TurnResult(
        user_text="",
        response_text="어? 잘 안 들렸어. 다시 한 번 말해 줄래?",
        audio_samples=np.zeros(10, dtype=np.float32),
        sample_rate=24_000,
        metrics=TurnMetrics(),
        state=PipelineState.IDLE,
    )

    with (
        patch.object(
            pipeline, "_transcribe_speech_audio", return_value=("hello", "hello")
        ) as transcribe,
        patch.object(
            pipeline, "_empty_stt_reprompt_result", return_value=reprompt_result
        ) as reprompt,
    ):
        pipeline.run_turn_with_audio(utterance)

    transcribe.assert_not_called()
    reprompt.assert_called_once()


def test_empty_stt_reprompt_follows_session_language(tmp_path: Path) -> None:
    """The empty/too-short re-prompt is English while the session is in English."""
    pipeline = _make_pipeline(tmp_path)
    pipeline._config.bilingual_mode = True
    pipeline.set_session_language("en")
    captured: dict[str, object] = {}

    def fake_fixed_response(**kwargs: object) -> TurnResult:
        captured.update(kwargs)
        return TurnResult(
            user_text="",
            response_text=str(kwargs.get("response_text", "")),
            audio_samples=np.zeros(10, dtype=np.float32),
            sample_rate=24_000,
            metrics=TurnMetrics(),
            state=PipelineState.IDLE,
        )

    with (
        patch.object(pipeline, "_save_conversation_audio", return_value=None),
        patch.object(pipeline, "_return_fixed_tts_response", side_effect=fake_fixed_response),
    ):
        pipeline._empty_stt_reprompt_result(
            audio_samples=np.zeros(160, dtype=np.float32),
            sample_rate=VAD_SAMPLE_RATE,
            raw_stt_text="",
            metrics=TurnMetrics(),
            turn_start=0.0,
        )

    assert captured["language"] == "en"
    assert captured["tts_language"] == "en"
    assert captured["response_text"] == "Hmm? Say that again!"


def test_run_turn_with_audio_raises_runtime_error_on_turn_error(tmp_path: Path) -> None:
    """run_turn_with_audio converts TurnResult.error into a command-style exception."""
    pipeline = _make_pipeline(tmp_path)
    utterance = Utterance(audio=np.ones(512, dtype=np.float32), sample_rate=VAD_SAMPLE_RATE)
    error_result = TurnResult(
        user_text="",
        response_text="",
        audio_samples=None,
        sample_rate=0,
        metrics=TurnMetrics(),
        state=PipelineState.ERROR,
        error="stt failed",
    )

    with patch.object(pipeline, "_run_turn_from_speech_audio", return_value=error_result):
        with pytest.raises(RuntimeError, match="stt failed"):
            pipeline.run_turn_with_audio(utterance)


def test_reset_session_reinitializes_session_id_dir_and_history(tmp_path: Path) -> None:
    """reset_session starts a fresh logical session and clears conversation state."""
    pipeline = _make_pipeline(tmp_path)
    old_session_id = pipeline.session_id
    pipeline._history.append({"role": "user", "content": "old"})
    pipeline._recent_assistant_responses.append("old response")
    pipeline._session_dir = tmp_path / "old"

    pipeline.reset_session()

    assert pipeline.session_id != old_session_id
    assert pipeline.conversation_history == []
    assert list(pipeline._recent_assistant_responses) == []
    assert pipeline.session_dir is not None
    assert pipeline.session_dir.exists()
    assert pipeline.session_dir.parent == tmp_path
