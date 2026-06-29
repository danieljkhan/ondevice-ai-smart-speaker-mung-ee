"""Silero VAD model loader and inference runner.

Provides functions to load the Silero VAD model and run voice activity
detection on audio samples. Extracted from ``scripts/test_vad.py`` to
establish correct dependency direction (``core/`` → ``models/``).
"""

from __future__ import annotations

import logging
import math
import os
import queue
import threading
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("mungi.models.vad_runner")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_RATE: int = 16000
WINDOW_SIZE_SAMPLES: int = 512  # 32ms at 16kHz, required by Silero VAD
DEFAULT_THRESHOLD: float = 0.5
MIN_SPEECH_DURATION_MS: int = 250
MIN_SILENCE_DURATION_MS: int = 100
DEFAULT_STREAMING_SOURCE_SAMPLE_RATE: int = 48_000
DEFAULT_STREAMING_SILENCE_FRAMES: int = 25
DEFAULT_ACTIVE_NO_FRAME_TIMEOUT_S: float = 5.0
STREAMING_QUEUE_POLL_S: float = 0.05

_streaming_lock = threading.Lock()
_streaming_active = False


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpeechSegment:
    """A detected speech segment with start/end timestamps in seconds."""

    start: float
    end: float

    def duration_ms(self) -> float:
        """Return segment duration in milliseconds."""
        return (self.end - self.start) * 1000.0

    def to_dict(self) -> dict[str, float]:
        """Serialize segment to a plain dict."""
        return asdict(self)


@dataclass(frozen=True)
class Utterance:
    """Streaming VAD utterance with mono 16 kHz float32 audio."""

    audio: np.ndarray
    sample_rate: int


# ---------------------------------------------------------------------------
# VAD model loading
# ---------------------------------------------------------------------------


def load_vad_model(model_path: str | None = None) -> Any:
    """Load the Silero VAD model from a local file.

    This device operates fully offline. Network-based loading
    (torch.hub) is intentionally not supported.

    Args:
        model_path: Path to a local Silero VAD model file (.jit or
            .onnx). When ``None``, falls back to the hub cache
            directory if it was previously downloaded.

    Returns:
        The loaded Silero VAD model.

    Raises:
        ImportError: If torch is not installed.
        FileNotFoundError: If no local model file is found.
    """
    try:
        import torch
    except ImportError:
        logger.error("torch is not installed. Install with: pip install torch")
        raise

    if model_path is not None:
        jit_path = Path(model_path)
        if not jit_path.exists():
            msg = f"Model file not found: {jit_path}"
            raise FileNotFoundError(msg)
        logger.info("Loading VAD model from local file: %s", jit_path)
        model = torch.jit.load(str(jit_path))
        model.eval()
        return model

    # Fallback: check torch.hub cache (offline only, no download)
    hub_cache = Path.home() / ".cache" / "torch" / "hub"
    hub_dir = hub_cache / "snakers4_silero-vad_master"
    if hub_dir.is_dir():
        logger.info("Loading VAD from hub cache (offline): %s", hub_dir)
        model, _utils = torch.hub.load(
            repo_or_dir=str(hub_dir),
            model="silero_vad",
            source="local",
            onnx=False,
        )
        return model

    msg = (
        "No local VAD model found. Provide model_path or "
        "ensure hub cache exists at "
        f"{hub_dir}. Network download is disabled (offline device)."
    )
    raise FileNotFoundError(msg)


# ---------------------------------------------------------------------------
# VAD inference
# ---------------------------------------------------------------------------


def run_vad(
    audio_samples: Any,
    model: Any,
    threshold: float = DEFAULT_THRESHOLD,
    min_speech_ms: int = MIN_SPEECH_DURATION_MS,
    min_silence_ms: int = MIN_SILENCE_DURATION_MS,
) -> list[SpeechSegment]:
    """Run Silero VAD on audio samples and return detected speech segments.

    Uses a simple threshold-based approach: iterate over windows, track
    speech/silence transitions, and merge segments.

    Args:
        audio_samples: Float audio samples in [-1.0, 1.0] at 16kHz.
        model: Loaded Silero VAD model.
        threshold: Speech probability threshold.
        min_speech_ms: Minimum speech segment duration in ms.
        min_silence_ms: Minimum silence duration to split segments in ms.

    Returns:
        List of SpeechSegment objects.
    """
    import torch

    samples = [float(sample) for sample in audio_samples]
    total_samples = len(samples)
    segments: list[SpeechSegment] = []

    speech_start: int | None = None
    silence_counter: int = 0
    min_silence_samples = int(min_silence_ms * SAMPLE_RATE / 1000)
    min_speech_samples = int(min_speech_ms * SAMPLE_RATE / 1000)

    model.reset_states()

    for offset in range(0, total_samples, WINDOW_SIZE_SAMPLES):
        end = min(offset + WINDOW_SIZE_SAMPLES, total_samples)
        chunk = samples[offset:end]

        if len(chunk) < WINDOW_SIZE_SAMPLES:
            chunk = chunk + [0.0] * (WINDOW_SIZE_SAMPLES - len(chunk))

        tensor = torch.FloatTensor(chunk)
        prob = model(tensor, SAMPLE_RATE).item()

        if prob >= threshold:
            silence_counter = 0
            if speech_start is None:
                speech_start = offset
        else:
            if speech_start is not None:
                silence_counter += WINDOW_SIZE_SAMPLES
                if silence_counter >= min_silence_samples:
                    speech_end = offset - silence_counter + WINDOW_SIZE_SAMPLES
                    duration = speech_end - speech_start
                    if duration >= min_speech_samples:
                        segments.append(
                            SpeechSegment(
                                start=speech_start / SAMPLE_RATE,
                                end=speech_end / SAMPLE_RATE,
                            )
                        )
                    speech_start = None
                    silence_counter = 0

    if speech_start is not None:
        speech_end = total_samples
        duration = speech_end - speech_start
        if duration >= min_speech_samples:
            segments.append(
                SpeechSegment(
                    start=speech_start / SAMPLE_RATE,
                    end=speech_end / SAMPLE_RATE,
                )
            )

    return segments


def iter_utterances(
    audio_queue: queue.Queue[Any],
    timeout: float,
    *,
    stop_event: threading.Event,
) -> Iterator[Utterance]:
    """Yield streaming VAD utterances from a live audio frame queue.

    The queue may expose optional ``sample_rate`` and ``vad_model`` attributes.
    ``ConversationPipeline.wait_for_utterance`` binds those attributes when it
    delegates from ``AudioCapture`` while preserving this wrapper signature.
    """
    _claim_streaming_wrapper()
    return _ClaimedUtteranceIterator(
        _iter_utterances_claimed(audio_queue, timeout, stop_event=stop_event)
    )


class _ClaimedUtteranceIterator:
    """Iterator wrapper that releases the single-owner claim on close."""

    def __init__(self, inner: Iterator[Utterance]) -> None:
        self._inner = inner
        self._closed = False

    def __iter__(self) -> _ClaimedUtteranceIterator:
        return self

    def __next__(self) -> Utterance:
        try:
            return next(self._inner)
        except StopIteration:
            self._closed = True
            raise

    def close(self) -> None:
        """Close the inner iterator and release any unstarted claim."""
        if self._closed:
            return
        close = getattr(self._inner, "close", None)
        if callable(close):
            close()
        self._closed = True
        _release_streaming_wrapper()

    def __del__(self) -> None:
        self.close()


def _iter_utterances_claimed(
    audio_queue: queue.Queue[Any],
    timeout: float,
    *,
    stop_event: threading.Event,
) -> Iterator[Utterance]:
    """Implementation for an already-claimed streaming wrapper slot."""
    try:
        model = getattr(audio_queue, "vad_model", None)
        source_sample_rate = int(
            getattr(audio_queue, "sample_rate", DEFAULT_STREAMING_SOURCE_SAMPLE_RATE)
        )
        threshold = _read_float_env("MUNGI_VAD_THRESHOLD", DEFAULT_THRESHOLD)
        silence_frames = _read_int_env(
            "MUNGI_VAD_SILENCE_FRAMES",
            DEFAULT_STREAMING_SILENCE_FRAMES,
        )
        if source_sample_rate <= 0:
            msg = f"Invalid streaming audio sample rate: {source_sample_rate}"
            raise ValueError(msg)
        if model is None:
            msg = "VAD streaming wrapper requires audio_queue.vad_model"
            raise RuntimeError(msg)

        raw_buffer = np.array([], dtype=np.float32)
        source_window_size = max(
            int(round(WINDOW_SIZE_SAMPLES * source_sample_rate / SAMPLE_RATE)),
            1,
        )
        utterance_windows: list[np.ndarray] = []
        speech_active = False
        silence_count = 0
        deadline = time.monotonic() + max(timeout, 0.0)
        active_no_frame_timeout_s = _read_float_env(
            "MUNGI_VAD_ACTIVE_NO_FRAME_TIMEOUT_S",
            DEFAULT_ACTIVE_NO_FRAME_TIMEOUT_S,
        )
        if active_no_frame_timeout_s <= 0.0:
            active_no_frame_timeout_s = DEFAULT_ACTIVE_NO_FRAME_TIMEOUT_S
        silence_timeout_s = silence_frames * WINDOW_SIZE_SAMPLES / SAMPLE_RATE
        active_no_frame_timeout_s = max(active_no_frame_timeout_s, silence_timeout_s)
        last_frame_at = time.monotonic()

        _reset_model_states(model)

        while not stop_event.is_set():
            if not speech_active and time.monotonic() >= deadline:
                return

            frame = _get_next_frame(audio_queue, deadline, speech_active)
            if frame is None:
                if speech_active and time.monotonic() - last_frame_at >= active_no_frame_timeout_s:
                    utterance = _finalize_streaming_utterance(
                        utterance_windows,
                        silence_count,
                    )
                    if utterance is not None:
                        yield utterance
                    return
                continue
            last_frame_at = time.monotonic()

            mono_frame = _coerce_frame_to_mono_float32(frame)
            if mono_frame.size == 0:
                continue
            raw_buffer = np.concatenate((raw_buffer, mono_frame))

            while raw_buffer.size >= source_window_size:
                source_chunk = raw_buffer[:source_window_size]
                raw_buffer = raw_buffer[source_window_size:]
                window = _resample_window(source_chunk, source_sample_rate)
                probability = _vad_probability(model, window)

                if probability >= threshold:
                    if not speech_active:
                        speech_active = True
                        utterance_windows = []
                    silence_count = 0
                    utterance_windows.append(window)
                    continue

                if not speech_active:
                    continue

                silence_count += 1
                utterance_windows.append(window)
                if silence_count < silence_frames:
                    continue

                utterance = _finalize_streaming_utterance(
                    utterance_windows,
                    silence_count,
                )
                if utterance is not None:
                    yield utterance
                utterance_windows = []
                speech_active = False
                silence_count = 0
                deadline = time.monotonic() + max(timeout, 0.0)
    finally:
        if stop_event.is_set():
            _drain_audio_queue(audio_queue)
        _release_streaming_wrapper()


def _finalize_streaming_utterance(
    utterance_windows: list[np.ndarray],
    silence_count: int,
) -> Utterance | None:
    """Build one utterance from buffered active-speech windows."""
    speech_windows = utterance_windows[: -silence_count or None]
    if not speech_windows:
        return None
    audio = np.concatenate(speech_windows).astype(np.float32, copy=False)
    return Utterance(audio=audio, sample_rate=SAMPLE_RATE)


def _claim_streaming_wrapper() -> None:
    """Mark the streaming wrapper as active or raise on re-entry."""
    global _streaming_active
    with _streaming_lock:
        if _streaming_active:
            msg = "VAD streaming wrapper already active"
            raise RuntimeError(msg)
        _streaming_active = True


def _release_streaming_wrapper() -> None:
    """Clear the streaming wrapper active flag."""
    global _streaming_active
    with _streaming_lock:
        _streaming_active = False


def _reset_model_states(model: Any) -> None:
    """Reset a Silero VAD model if it exposes the streaming-state hook."""
    reset_states = getattr(model, "reset_states", None)
    if callable(reset_states):
        reset_states()


def _get_next_frame(
    audio_queue: queue.Queue[Any],
    deadline: float,
    speech_active: bool,
) -> Any | None:
    """Return the next queued audio frame or ``None`` on timeout."""
    timeout_s = STREAMING_QUEUE_POLL_S
    if not speech_active:
        timeout_s = max(min(deadline - time.monotonic(), STREAMING_QUEUE_POLL_S), 0.0)
    try:
        return audio_queue.get(timeout=timeout_s)
    except queue.Empty:
        return None


def _coerce_frame_to_mono_float32(frame: Any) -> np.ndarray:
    """Convert a raw mono or multi-channel callback frame to finite mono float32."""
    samples = np.asarray(frame, dtype=np.float32)
    if samples.size == 0:
        return np.array([], dtype=np.float32)
    if samples.ndim == 0:
        samples = samples.reshape(1)
    elif samples.ndim > 1:
        samples = samples.reshape(samples.shape[0], -1).mean(axis=1)
    samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
    return samples.astype(np.float32, copy=False)


def _resample_window(source_chunk: np.ndarray, source_sample_rate: int) -> np.ndarray:
    """Resample one source-rate chunk into one exact Silero VAD window."""
    if source_sample_rate == SAMPLE_RATE:
        resampled = source_chunk.astype(np.float32, copy=False)
    else:
        from scipy.signal import resample_poly  # type: ignore[import-not-found, import-untyped]

        divisor = math.gcd(source_sample_rate, SAMPLE_RATE)
        resampled = np.asarray(
            resample_poly(
                source_chunk,
                SAMPLE_RATE // divisor,
                source_sample_rate // divisor,
            ),
            dtype=np.float32,
        )

    if resampled.size < WINDOW_SIZE_SAMPLES:
        return np.pad(
            resampled,
            (0, WINDOW_SIZE_SAMPLES - resampled.size),
            mode="constant",
        ).astype(np.float32, copy=False)
    if resampled.size > WINDOW_SIZE_SAMPLES:
        return resampled[:WINDOW_SIZE_SAMPLES].astype(np.float32, copy=False)
    return resampled.astype(np.float32, copy=False)


def _vad_probability(model: Any, window: np.ndarray) -> float:
    """Run one 16 kHz VAD window and return a speech probability."""
    import torch  # type: ignore[import-not-found, import-untyped]

    tensor = torch.FloatTensor(window.tolist())
    raw_probability = model(tensor, SAMPLE_RATE)
    item = getattr(raw_probability, "item", None)
    if callable(item):
        return float(item())
    return float(raw_probability)


def _drain_audio_queue(audio_queue: queue.Queue[Any]) -> None:
    """Consume all currently queued audio frames."""
    while True:
        try:
            audio_queue.get_nowait()
        except queue.Empty:
            return


def _read_int_env(name: str, default: int) -> int:
    """Read a positive integer environment override."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Ignoring invalid integer %s=%r", name, raw)
        return default
    return value if value > 0 else default


def _read_float_env(name: str, default: float) -> float:
    """Read a float environment override."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Ignoring invalid float %s=%r", name, raw)
        return default
