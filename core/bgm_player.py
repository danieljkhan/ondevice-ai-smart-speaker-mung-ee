"""Looping background-music playback channel for Funny English."""

from __future__ import annotations

import logging
import math
import os
import threading
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_sd: Any = None
_DEFAULT_VOLUME = 0.16
_DEFAULT_DUCK_VOLUME = 0.0
_USB_DEVICE_HINTS: tuple[str, ...] = (
    "usb pnp audio device",
    "usb audio",
    "jmtek",
    "waveshare",
    "solid state system",
    "c-media",
)


class BgmPlayer:
    """Play a low-volume WAV loop on a dedicated non-blocking output stream."""

    def __init__(self, *, device: str | int | None = None) -> None:
        """Create a BGM channel for the configured output device."""
        self._device = _normalize_device(device)
        self._stream: Any | None = None
        self._samples = np.zeros(1, dtype=np.float32)
        self._sample_rate = 0
        self._position = 0
        self._volume = _DEFAULT_VOLUME
        self._unducked_volume = _DEFAULT_VOLUME
        self._lock = threading.RLock()

    @property
    def volume(self) -> float:
        """Return the current playback multiplier."""
        with self._lock:
            return self._volume

    def start_loop(
        self,
        audio_samples: Any,
        sample_rate: int,
        *,
        volume: float = _DEFAULT_VOLUME,
    ) -> None:
        """Start or replace the current looping BGM stream."""
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        samples = _to_playback_float32(audio_samples)
        if samples.size == 0:
            logger.info("Skipping BGM start for empty audio buffer")
            return
        if samples.ndim > 1:
            samples = samples.reshape(-1)

        sd = _get_sd()
        output_device = _resolve_output_device(sd, self._device)
        device_info = sd.query_devices()[output_device]
        target_sample_rate = _default_samplerate(device_info) or sample_rate
        if target_sample_rate != sample_rate:
            samples = _resample_audio(samples, sample_rate, target_sample_rate)
            sample_rate = target_sample_rate

        with self._lock:
            self._stop_locked()
            self._samples = samples
            self._sample_rate = sample_rate
            self._position = 0
            self._volume = _clamp_volume(volume)
            self._unducked_volume = self._volume
            self._stream = sd.OutputStream(
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
                device=output_device,
                callback=self._callback,
            )
            self._stream.start()

    def duck(self, *, volume: float = _DEFAULT_DUCK_VOLUME) -> None:
        """Lower the BGM level immediately before listen capture starts."""
        with self._lock:
            self._volume = _clamp_volume(volume)

    def unduck(self) -> None:
        """Restore the previous non-ducked loop volume."""
        with self._lock:
            self._volume = self._unducked_volume

    def stop(self) -> None:
        """Stop the active BGM stream if one exists."""
        with self._lock:
            self._stop_locked()

    def close(self) -> None:
        """Release the BGM stream."""
        self.stop()

    def _callback(
        self,
        outdata: Any,
        frames: int,
        _time_info: Any,
        _status: Any,
    ) -> None:
        """Fill one sounddevice callback buffer from the loop."""
        with self._lock:
            samples = self._samples
            if samples.size == 0:
                outdata.fill(0)
                return
            volume = self._volume
            position = self._position

            output = np.empty(frames, dtype=np.float32)
            filled = 0
            while filled < frames:
                remaining = frames - filled
                chunk_size = min(remaining, samples.size - position)
                output[filled : filled + chunk_size] = samples[position : position + chunk_size]
                filled += chunk_size
                position = (position + chunk_size) % samples.size
            self._position = position
            outdata[:, 0] = output * volume

    def _stop_locked(self) -> None:
        stream = self._stream
        self._stream = None
        if stream is None:
            return
        try:
            stream.stop()
        finally:
            stream.close()


def _get_sd() -> Any:
    """Import and cache sounddevice lazily."""
    global _sd
    if _sd is None:
        import sounddevice as sd  # type: ignore[import-not-found, import-untyped]

        _sd = sd
    return _sd


def _normalize_device(device: str | int | None) -> str | int | None:
    if device is None:
        env_device = os.getenv("MUNGI_AUDIO_OUTPUT_DEVICE", "").strip()
        device = env_device or None
    if isinstance(device, str):
        normalized = device.strip()
        if not normalized:
            return None
        if normalized.isdigit():
            return int(normalized)
        return normalized
    return device


def _resolve_output_device(sd: Any, device: str | int | None = None) -> int | None:
    devices = sd.query_devices()
    output_devices = [
        (idx, info) for idx, info in enumerate(devices) if _max_output_channels(info) > 0
    ]
    if not output_devices:
        raise RuntimeError("No output audio devices available.")
    if isinstance(device, int):
        if device < 0 or device >= len(devices):
            raise RuntimeError(f"Configured output device index out of range: {device}")
        if _max_output_channels(devices[device]) <= 0:
            raise RuntimeError(f"Configured output device is not playback-capable: {device}")
        return device
    if isinstance(device, str):
        needle = device.casefold()
        for idx, info in output_devices:
            if _device_name(info).casefold() == needle:
                return idx
        for idx, info in output_devices:
            if needle in _device_name(info).casefold():
                return idx
        raise RuntimeError(f"Configured output device not found: {device}")
    for idx, info in output_devices:
        name = _device_name(info).casefold()
        if any(hint in name for hint in _USB_DEVICE_HINTS):
            return idx
    return output_devices[0][0]


def _device_name(info: Any) -> str:
    if isinstance(info, dict):
        return str(info.get("name", ""))
    return str(getattr(info, "name", ""))


def _max_output_channels(info: Any) -> int:
    if isinstance(info, dict):
        return int(info.get("max_output_channels", 0))
    return int(getattr(info, "max_output_channels", 0))


def _default_samplerate(info: Any) -> int:
    value = info.get("default_samplerate", 0) if isinstance(info, dict) else 0
    if not isinstance(info, dict):
        value = getattr(info, "default_samplerate", 0)
    return int(round(float(value))) if value else 0


def _to_playback_float32(audio: Any) -> np.ndarray:
    raw_samples = np.asarray(audio)
    if np.issubdtype(raw_samples.dtype, np.integer):
        float_samples = raw_samples.astype(np.float32)
        if np.issubdtype(raw_samples.dtype, np.unsignedinteger):
            scale = (float(np.iinfo(raw_samples.dtype).max) + 1.0) / 2.0
            return (float_samples - scale) / scale
        dtype_info = np.iinfo(raw_samples.dtype)
        scale = float(max(abs(dtype_info.min), dtype_info.max))
        return float_samples / scale
    return raw_samples.astype(np.float32)


def _resample_audio(
    audio: np.ndarray,
    source_sample_rate: int,
    target_sample_rate: int,
) -> np.ndarray:
    if source_sample_rate <= 0 or target_sample_rate <= 0:
        return audio
    if source_sample_rate == target_sample_rate or audio.size <= 1:
        return audio
    duration_s = audio.size / float(source_sample_rate)
    target_size = max(int(round(duration_s * target_sample_rate)), 1)
    source_positions = np.linspace(0.0, audio.size - 1, num=audio.size, dtype=np.float32)
    target_positions = np.linspace(
        0.0,
        audio.size - 1,
        num=target_size,
        dtype=np.float32,
    )
    result: np.ndarray = np.interp(target_positions, source_positions, audio).astype(np.float32)
    return result


def _clamp_volume(volume: float) -> float:
    if not math.isfinite(volume):
        return 0.0
    return min(1.0, max(0.0, volume))


__all__ = ["BgmPlayer"]
