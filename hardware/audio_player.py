"""Audio playback via sounddevice for Jetson speaker output.

Provides blocking audio playback functions for synthesized TTS output.
Prefers USB audio devices by default so Jetson HDMI output does not
silently steal playback from the external speaker path.
"""

from __future__ import annotations

import logging
import math
import os
import threading
from typing import Any

import numpy as np

logger = logging.getLogger("mungi.hardware.audio_player")

_sd: Any = None
_USB_DEVICE_HINTS: tuple[str, ...] = (
    "usb pnp audio device",
    "usb audio",
    "jmtek",
    "waveshare",
    "solid state system",
    "c-media",
)
_DEFAULT_OUTPUT_LATENCY_S = 0.4
_DEFAULT_PLAYBACK_LEAD_SILENCE_S = 0.30
_DEFAULT_PLAYBACK_TRAIL_SILENCE_S = 0.40
_DEFAULT_PLAYBACK_CHUNK_S = 0.10
_AUDIO_LATENCY_PRESETS = {"low", "high"}
_AudioLatency = float | str
_playback_abort = threading.Event()


def _resolve_audio_latency(raw_value: str | None = None) -> _AudioLatency:
    """Resolve ``MUNGI_AUDIO_LATENCY`` as seconds or a PortAudio preset.

    Accepted values are ``"low"``, ``"high"``, or a positive finite float
    string in seconds. ``None``, an empty string, unknown presets, invalid
    floats, non-positive values, ``NaN``, and infinity fall back to
    ``_DEFAULT_OUTPUT_LATENCY_S`` (0.4 seconds).
    """
    if raw_value is None:
        return _DEFAULT_OUTPUT_LATENCY_S

    latency = raw_value.strip()
    if not latency:
        return _DEFAULT_OUTPUT_LATENCY_S

    preset = latency.casefold()
    if preset in _AUDIO_LATENCY_PRESETS:
        return preset

    try:
        parsed_latency = float(latency)
    except ValueError:
        return _DEFAULT_OUTPUT_LATENCY_S

    if not math.isfinite(parsed_latency) or parsed_latency <= 0.0:
        return _DEFAULT_OUTPUT_LATENCY_S

    return parsed_latency


_DEFAULT_AUDIO_LATENCY = _resolve_audio_latency(os.getenv("MUNGI_AUDIO_LATENCY"))


def _resolve_non_negative_seconds(raw_value: str | None, default_value: float) -> float:
    """Resolve an optional environment value as non-negative finite seconds."""
    if raw_value is None:
        return default_value

    value = raw_value.strip()
    if not value:
        return default_value

    try:
        parsed_value = float(value)
    except ValueError:
        return default_value

    if not math.isfinite(parsed_value) or parsed_value < 0.0:
        return default_value

    return parsed_value


def _resolve_playback_lead_silence(raw_value: str | None = None) -> float:
    """Resolve ``MUNGI_AUDIO_LEAD_SILENCE`` as non-negative finite seconds."""
    return _resolve_non_negative_seconds(raw_value, _DEFAULT_PLAYBACK_LEAD_SILENCE_S)


def _resolve_playback_trail_silence(raw_value: str | None = None) -> float:
    """Resolve ``MUNGI_AUDIO_TRAIL_SILENCE`` as non-negative finite seconds."""
    return _resolve_non_negative_seconds(raw_value, _DEFAULT_PLAYBACK_TRAIL_SILENCE_S)


def _resolve_playback_chunk_s(raw_value: str | None = None) -> float:
    """Resolve ``MUNGI_AUDIO_PLAYBACK_CHUNK_S`` as positive finite seconds."""
    resolved_value = _resolve_non_negative_seconds(raw_value, _DEFAULT_PLAYBACK_CHUNK_S)
    if resolved_value <= 0.0:
        return _DEFAULT_PLAYBACK_CHUNK_S
    return resolved_value


_PLAYBACK_LEAD_SILENCE_S = _resolve_playback_lead_silence(os.getenv("MUNGI_AUDIO_LEAD_SILENCE"))
_PLAYBACK_TRAIL_SILENCE_S = _resolve_playback_trail_silence(os.getenv("MUNGI_AUDIO_TRAIL_SILENCE"))
_PLAYBACK_CHUNK_S = _resolve_playback_chunk_s(os.getenv("MUNGI_AUDIO_PLAYBACK_CHUNK_S"))


def _get_sd() -> Any:
    """Import and cache the sounddevice module lazily."""
    global _sd
    if _sd is None:
        import sounddevice as sd  # type: ignore[import-not-found, import-untyped]

        sd.default.latency = _DEFAULT_AUDIO_LATENCY
        _sd = sd
        logger.info(
            "sounddevice %s initialized (latency=%s)",
            sd.__version__,
            _DEFAULT_AUDIO_LATENCY,
        )
    return _sd


def _device_name(info: Any) -> str:
    """Return a device name from a sounddevice info object."""
    if isinstance(info, dict):
        return str(info.get("name", ""))
    return str(getattr(info, "name", ""))


def _max_output_channels(info: Any) -> int:
    """Return the number of output channels advertised by a device."""
    if isinstance(info, dict):
        return int(info.get("max_output_channels", 0))
    return int(getattr(info, "max_output_channels", 0))


def _default_samplerate(info: Any) -> int:
    """Return the preferred playback sample rate for a device."""
    if isinstance(info, dict):
        value = info.get("default_samplerate", 0)
    else:
        value = getattr(info, "default_samplerate", 0)
    return int(round(float(value))) if value else 0


def _is_hdmi_or_hda_device_name(name: str) -> bool:
    """Return whether a device name looks like an HDMI/HDA output sink."""
    normalized = name.casefold()
    return "hdmi" in normalized or "hda" in normalized


def _normalize_device(device: str | int | None) -> str | int | None:
    """Normalize a configured output device identifier."""
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


def list_output_devices() -> list[dict[str, Any]]:
    """Return output-capable audio devices for logging or CLI display."""
    sd = _get_sd()
    devices = sd.query_devices()
    results: list[dict[str, Any]] = []
    for idx, info in enumerate(devices):
        if _max_output_channels(info) <= 0:
            continue
        results.append(
            {
                "index": idx,
                "name": _device_name(info),
                "max_output_channels": _max_output_channels(info),
            }
        )
    return results


def _resolve_output_device(device: str | int | None = None) -> int | None:
    """Resolve the best output device index for playback."""
    sd = _get_sd()
    devices = sd.query_devices()
    output_devices = [
        (idx, info) for idx, info in enumerate(devices) if _max_output_channels(info) > 0
    ]
    if not output_devices:
        msg = "No output audio devices available."
        raise RuntimeError(msg)

    preferred = _normalize_device(device)
    if isinstance(preferred, int):
        if preferred < 0 or preferred >= len(devices):
            msg = f"Configured output device index out of range: {preferred}"
            raise RuntimeError(msg)
        if _max_output_channels(devices[preferred]) <= 0:
            msg = f"Configured output device is not playback-capable: {preferred}"
            raise RuntimeError(msg)
        return preferred

    if isinstance(preferred, str):
        needle = preferred.casefold()
        for idx, info in output_devices:
            if _device_name(info).casefold() == needle:
                return idx
        for idx, info in output_devices:
            if needle in _device_name(info).casefold():
                return idx
        msg = f"Configured output device not found: {preferred}"
        raise RuntimeError(msg)

    for idx, info in output_devices:
        name = _device_name(info).casefold()
        if any(hint in name for hint in _USB_DEVICE_HINTS):
            return idx

    default_device: Any = getattr(sd.default, "device", None)
    try:
        output_idx = default_device[1]
    except (TypeError, IndexError, KeyError):
        output_idx = None
    if (
        isinstance(output_idx, int)
        and 0 <= output_idx < len(devices)
        and _max_output_channels(devices[output_idx]) > 0
    ):
        return output_idx

    for sink_name in ("pulse", "default"):
        for idx, info in output_devices:
            name = _device_name(info)
            if sink_name in name.casefold():
                logger.info(
                    "Selected PulseAudio-compatible output fallback: index=%s name=%s",
                    idx,
                    name,
                )
                return idx

    for idx, info in output_devices:
        name = _device_name(info)
        if not _is_hdmi_or_hda_device_name(name):
            logger.warning(
                "Selected non-HDMI/HDA output fallback: index=%s name=%s",
                idx,
                name,
            )
            return idx

    return output_devices[0][0]


def _resample_audio(
    audio: np.ndarray,
    source_sample_rate: int,
    target_sample_rate: int,
) -> np.ndarray:
    """Resample mono float audio with linear interpolation."""
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


def _to_playback_float32(audio: Any) -> np.ndarray:
    """Return float32 playback samples with integer PCM scaled to [-1.0, 1.0]."""
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


def _write_stream_chunks(
    sd: Any,
    samples: np.ndarray,
    sample_rate: int,
    output_device: int | None,
) -> None:
    """Write mono samples to an owner-thread stream, honoring playback aborts."""
    blocksize = max(1, int(sample_rate * _PLAYBACK_CHUNK_S))
    with sd.OutputStream(
        samplerate=sample_rate,
        device=output_device,
        channels=1,
        dtype="float32",
        latency=_DEFAULT_AUDIO_LATENCY,
    ) as stream:
        for start_idx in range(0, samples.size, blocksize):
            if _playback_abort.is_set():
                stream.abort()
                break
            stream.write(samples[start_idx : start_idx + blocksize])


def _play_audio_nonblocking(
    sd: Any,
    samples: np.ndarray,
    sample_rate: int,
    output_device: int | None,
) -> None:
    """Start fire-and-forget playback on a daemon owner thread."""

    def run_playback() -> None:
        try:
            _write_stream_chunks(sd, samples, sample_rate, output_device)
        except Exception:
            logger.exception("Non-blocking audio playback failed on device %s", output_device)

    thread = threading.Thread(
        target=run_playback,
        name="MungiAudioPlayback",
        daemon=True,
    )
    thread.start()


def play_audio(
    audio: Any,
    sample_rate: int,
    device: str | int | None = None,
    *,
    blocking: bool = True,
) -> None:
    """Play audio through the resolved output device."""
    samples = _to_playback_float32(audio)
    if samples.size == 0:
        logger.info("Skipping playback for empty audio buffer")
        return

    if samples.ndim > 1:
        samples = samples.reshape(-1)

    sd = _get_sd()
    output_device = _resolve_output_device(device)
    device_info = sd.query_devices()[output_device]
    target_sample_rate = _default_samplerate(device_info)
    logger.info(
        "Playing %d samples at %d Hz on device %s",
        samples.size,
        sample_rate,
        output_device,
    )
    if target_sample_rate and target_sample_rate != sample_rate:
        logger.info(
            "Resampling audio for device %s: %d Hz -> %d Hz",
            output_device,
            sample_rate,
            target_sample_rate,
        )
        samples = _resample_audio(samples, sample_rate, target_sample_rate)
        sample_rate = target_sample_rate
    lead_samples = int(round(sample_rate * _PLAYBACK_LEAD_SILENCE_S))
    if lead_samples > 0:
        # Fresh USB output streams can drop initial frames; spend that warmup on silence.
        samples = np.concatenate(
            (np.zeros(lead_samples, dtype=np.float32), samples),
        )
    trail_samples = int(round(sample_rate * _PLAYBACK_TRAIL_SILENCE_S))
    if trail_samples > 0:
        # USB DAC teardown can drop final frames; keep spoken tails inside the stream.
        samples = np.concatenate(
            (samples, np.zeros(trail_samples, dtype=np.float32)),
        )
    _playback_abort.clear()
    try:
        if blocking:
            _write_stream_chunks(sd, samples, sample_rate, output_device)
        else:
            _play_audio_nonblocking(sd, samples.copy(), sample_rate, output_device)
    except Exception as exc:
        msg = f"Audio playback failed on device {output_device}: {exc}"
        raise RuntimeError(msg) from exc


def stop_playback() -> None:
    """Request interruption of any active audio playback."""
    _playback_abort.set()
