"""Tests for hardware.audio_player."""

from __future__ import annotations

import importlib
import sys
import threading
from collections.abc import Callable
from types import SimpleNamespace, TracebackType
from typing import Any

import numpy as np
from pytest import MonkeyPatch


class FakeOutputStream:
    """Minimal sounddevice OutputStream stub for playback tests."""

    def __init__(
        self,
        owner: FakeSoundDevice,
        samplerate: int,
        device: int | None,
        channels: int,
        dtype: str,
        latency: Any | None,
    ) -> None:
        self._owner = owner
        self.samplerate = samplerate
        self.device = device
        self.channels = channels
        self.dtype = dtype
        self.latency = latency
        self.write_calls: list[np.ndarray] = []
        self.abort_calls = 0

    def __enter__(self) -> FakeOutputStream:
        self._owner.streams.append(self)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        if self._owner.on_stream_exit is not None:
            self._owner.on_stream_exit(self)
        return False

    def write(self, audio: np.ndarray) -> None:
        self.write_calls.append(np.array(audio, copy=True))
        if self._owner.on_write is not None:
            self._owner.on_write(self)

    def abort(self) -> None:
        self.abort_calls += 1


class FakeSoundDevice:
    """Minimal sounddevice stub for audio player tests."""

    def __init__(self, devices: list[dict[str, object]], default_output: int = 0) -> None:
        self._devices = devices
        self.default = SimpleNamespace(device=(-1, default_output), latency=None)
        self.play_calls: list[tuple[np.ndarray, int, int | None]] = []
        self.latency_calls: list[Any | None] = []
        self.output_stream_calls: list[dict[str, Any]] = []
        self.streams: list[FakeOutputStream] = []
        self.on_write: Callable[[FakeOutputStream], None] | None = None
        self.on_stream_exit: Callable[[FakeOutputStream], None] | None = None
        self.wait_calls = 0
        self.stop_calls = 0
        self.__version__ = "fake"

    def query_devices(self) -> list[dict[str, object]]:
        return self._devices

    def play(
        self,
        audio: np.ndarray,
        sample_rate: int,
        device: int | None = None,
        latency: Any | None = None,
    ) -> None:
        self.play_calls.append((audio, sample_rate, device))
        self.latency_calls.append(latency)

    def wait(self) -> None:
        self.wait_calls += 1

    def stop(self) -> None:
        self.stop_calls += 1

    def OutputStream(
        self,
        *,
        samplerate: int,
        device: int | None = None,
        channels: int,
        dtype: str,
        latency: Any | None = None,
    ) -> FakeOutputStream:
        self.output_stream_calls.append(
            {
                "samplerate": samplerate,
                "device": device,
                "channels": channels,
                "dtype": dtype,
                "latency": latency,
            }
        )
        return FakeOutputStream(self, samplerate, device, channels, dtype, latency)

    def played_audio(self) -> np.ndarray:
        chunks = [chunk for stream in self.streams for chunk in stream.write_calls]
        if not chunks:
            return np.array([], dtype=np.float32)
        return np.concatenate(chunks)

    def stream_abort_calls(self) -> int:
        return sum(stream.abort_calls for stream in self.streams)


class FakeDefaultDevice:
    """Non-list subscriptable default-device pair."""

    def __init__(self, input_device: int, output_device: int) -> None:
        self._devices = (input_device, output_device)

    def __getitem__(self, index: int) -> int:
        return self._devices[index]


def test_resolve_output_device_prefers_usb(monkeypatch: MonkeyPatch) -> None:
    from hardware import audio_player

    fake_sd = FakeSoundDevice(
        [
            {"name": "HDA HDMI 0", "max_output_channels": 2},
            {"name": "USB PnP Audio Device", "max_output_channels": 2},
        ],
        default_output=0,
    )
    monkeypatch.setattr(audio_player, "_sd", fake_sd)

    assert audio_player._resolve_output_device() == 1


def test_resolve_output_device_prefers_pulse_over_hdmi_fallback(
    monkeypatch: MonkeyPatch,
) -> None:
    from hardware import audio_player

    fake_sd = FakeSoundDevice(
        [
            {
                "name": "USB PnP Audio Device: Audio (hw:0,0)",
                "max_input_channels": 2,
                "max_output_channels": 0,
            },
            {
                "name": "NVIDIA Jetson Orin Nano HDA: HDMI 0 (hw:1,3)",
                "max_output_channels": 2,
            },
            {"name": "pulse", "max_output_channels": 32},
            {"name": "default", "max_output_channels": 32},
        ],
        default_output=-1,
    )
    monkeypatch.setattr(audio_player, "_sd", fake_sd)

    resolved_device = audio_player._resolve_output_device(None)

    assert resolved_device == 2
    assert resolved_device != 1


def test_resolve_output_device_still_prefers_output_capable_usb(
    monkeypatch: MonkeyPatch,
) -> None:
    from hardware import audio_player

    fake_sd = FakeSoundDevice(
        [
            {
                "name": "NVIDIA Jetson Orin Nano HDA: HDMI 0 (hw:1,3)",
                "max_output_channels": 2,
            },
            {
                "name": "USB PnP Audio Device: Audio (hw:0,0)",
                "max_input_channels": 2,
                "max_output_channels": 2,
            },
            {"name": "pulse", "max_output_channels": 32},
            {"name": "default", "max_output_channels": 32},
        ],
        default_output=0,
    )
    monkeypatch.setattr(audio_player, "_sd", fake_sd)

    assert audio_player._resolve_output_device(None) == 1


def test_resolve_output_device_honors_subscriptable_os_default(
    monkeypatch: MonkeyPatch,
) -> None:
    from hardware import audio_player

    fake_sd = FakeSoundDevice(
        [
            {
                "name": "NVIDIA Jetson Orin Nano HDA: HDMI 0 (hw:1,3)",
                "max_output_channels": 2,
            },
            {"name": "pulse", "max_output_channels": 32},
            {"name": "Built-in Speaker", "max_output_channels": 2},
        ],
        default_output=-1,
    )
    fake_sd.default.device = FakeDefaultDevice(-1, 2)
    monkeypatch.setattr(audio_player, "_sd", fake_sd)

    assert audio_player._resolve_output_device(None) == 2


def test_get_sd_uses_default_output_latency_when_env_unset(
    monkeypatch: MonkeyPatch,
) -> None:
    from hardware import audio_player

    monkeypatch.delenv("MUNGI_AUDIO_LATENCY", raising=False)
    reloaded_audio_player = importlib.reload(audio_player)
    fake_sd = FakeSoundDevice(
        [{"name": "USB PnP Audio Device", "max_output_channels": 2}],
        default_output=0,
    )
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)

    assert reloaded_audio_player._DEFAULT_OUTPUT_LATENCY_S == 0.4
    assert reloaded_audio_player._DEFAULT_AUDIO_LATENCY == 0.4
    assert reloaded_audio_player._get_sd() is fake_sd
    assert fake_sd.default.latency == 0.4


def test_get_sd_honors_latency_env(monkeypatch: MonkeyPatch) -> None:
    from hardware import audio_player

    monkeypatch.setenv("MUNGI_AUDIO_LATENCY", "low")
    reloaded_audio_player = importlib.reload(audio_player)
    fake_sd = FakeSoundDevice(
        [{"name": "USB PnP Audio Device", "max_output_channels": 2}],
        default_output=0,
    )
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)

    assert reloaded_audio_player._DEFAULT_AUDIO_LATENCY == "low"
    assert reloaded_audio_player._get_sd() is fake_sd
    assert fake_sd.default.latency == "low"

    monkeypatch.delenv("MUNGI_AUDIO_LATENCY", raising=False)
    importlib.reload(audio_player)


def test_to_playback_float32_passes_through_float() -> None:
    from hardware import audio_player

    samples = np.array([0.25, -0.5], dtype=np.float32)

    converted = audio_player._to_playback_float32(samples)

    assert converted.dtype == np.float32
    np.testing.assert_array_equal(converted, samples)


def test_to_playback_float32_normalizes_int16() -> None:
    from hardware import audio_player

    samples = np.array([-32768, 0, 32767], dtype=np.int16)

    converted = audio_player._to_playback_float32(samples)

    assert converted.dtype == np.float32
    assert float(np.max(np.abs(converted))) <= 1.0
    np.testing.assert_allclose(converted, np.array([-1.0, 0.0, 32767 / 32768]))


def test_to_playback_float32_normalizes_uint8() -> None:
    from hardware import audio_player

    samples = np.array([0, 128, 255], dtype=np.uint8)

    converted = audio_player._to_playback_float32(samples)

    assert converted.dtype == np.float32
    assert float(np.max(np.abs(converted))) <= 1.0
    np.testing.assert_allclose(converted, np.array([-1.0, 0.0, 127 / 128]))


def test_resolve_audio_latency_returns_default_when_unset() -> None:
    from hardware import audio_player

    assert audio_player._resolve_audio_latency(None) == 0.4


def test_resolve_audio_latency_accepts_low_high_presets() -> None:
    from hardware import audio_player

    assert audio_player._resolve_audio_latency("low") == "low"
    assert audio_player._resolve_audio_latency("high") == "high"
    assert audio_player._resolve_audio_latency("LOW") == "low"


def test_resolve_audio_latency_parses_positive_float() -> None:
    from hardware import audio_player

    assert audio_player._resolve_audio_latency("0.25") == 0.25


def test_resolve_audio_latency_falls_back_on_invalid() -> None:
    from hardware import audio_player

    for raw_latency in ("abc", "-0.1", "0", "inf"):
        assert audio_player._resolve_audio_latency(raw_latency) == 0.4


def test_resolve_playback_lead_silence_returns_default_when_unset_or_invalid() -> None:
    from hardware import audio_player

    for raw_lead_silence in (None, "", "abc", "-0.1", "inf", "nan"):
        assert (
            audio_player._resolve_playback_lead_silence(raw_lead_silence)
            == audio_player._DEFAULT_PLAYBACK_LEAD_SILENCE_S
        )


def test_resolve_playback_lead_silence_accepts_non_negative_float() -> None:
    from hardware import audio_player

    assert audio_player._resolve_playback_lead_silence("0") == 0.0
    assert audio_player._resolve_playback_lead_silence("0.25") == 0.25


def test_resolve_playback_trail_silence_returns_default_when_unset_or_invalid() -> None:
    from hardware import audio_player

    for raw_trail_silence in (None, "", "abc", "-0.1", "inf", "nan"):
        assert (
            audio_player._resolve_playback_trail_silence(raw_trail_silence)
            == audio_player._DEFAULT_PLAYBACK_TRAIL_SILENCE_S
        )


def test_resolve_playback_trail_silence_accepts_non_negative_float() -> None:
    from hardware import audio_player

    assert audio_player._resolve_playback_trail_silence("0") == 0.0
    assert audio_player._resolve_playback_trail_silence("0.25") == 0.25


def test_resolve_playback_chunk_s_returns_default_when_unset_or_invalid() -> None:
    from hardware import audio_player

    for raw_chunk_s in (None, "", "abc", "-0.1", "0", "inf", "nan"):
        assert (
            audio_player._resolve_playback_chunk_s(raw_chunk_s)
            == audio_player._DEFAULT_PLAYBACK_CHUNK_S
        )


def test_resolve_playback_chunk_s_accepts_positive_float() -> None:
    from hardware import audio_player

    assert audio_player._resolve_playback_chunk_s("0.01") == 0.01
    assert audio_player._resolve_playback_chunk_s("0.25") == 0.25


def test_resolve_output_device_matches_requested_name(monkeypatch: MonkeyPatch) -> None:
    from hardware import audio_player

    fake_sd = FakeSoundDevice(
        [
            {"name": "HDA HDMI 0", "max_output_channels": 2},
            {"name": "USB PnP Audio Device", "max_output_channels": 2},
        ],
        default_output=0,
    )
    monkeypatch.setattr(audio_player, "_sd", fake_sd)

    assert audio_player._resolve_output_device("USB PnP") == 1


def test_play_audio_flattens_and_writes_stream(monkeypatch: MonkeyPatch) -> None:
    from hardware import audio_player

    fake_sd = FakeSoundDevice(
        [
            {
                "name": "USB PnP Audio Device",
                "max_output_channels": 2,
                "default_samplerate": 22050,
            }
        ],
        default_output=0,
    )
    monkeypatch.setattr(audio_player, "_sd", fake_sd)
    monkeypatch.setattr(audio_player, "_PLAYBACK_LEAD_SILENCE_S", 0.0)
    monkeypatch.setattr(audio_player, "_PLAYBACK_TRAIL_SILENCE_S", 0.0)

    audio_player.play_audio(np.array([[0.1, -0.1]], dtype=np.float32), 22050)

    played_audio = fake_sd.played_audio()
    assert played_audio.ndim == 1
    np.testing.assert_array_equal(played_audio, np.array([0.1, -0.1], dtype=np.float32))
    assert fake_sd.output_stream_calls[0]["samplerate"] == 22050
    assert fake_sd.output_stream_calls[0]["device"] == 0
    assert fake_sd.play_calls == []
    assert fake_sd.wait_calls == 0
    assert fake_sd.stop_calls == 0


def test_play_audio_writes_chunks_without_global_play_wait_stop(
    monkeypatch: MonkeyPatch,
) -> None:
    from hardware import audio_player

    fake_sd = FakeSoundDevice(
        [
            {
                "name": "USB PnP Audio Device",
                "max_output_channels": 2,
                "default_samplerate": 10,
            }
        ],
        default_output=0,
    )
    monkeypatch.setattr(audio_player, "_sd", fake_sd)
    monkeypatch.setattr(audio_player, "_PLAYBACK_LEAD_SILENCE_S", 0.0)
    monkeypatch.setattr(audio_player, "_PLAYBACK_TRAIL_SILENCE_S", 0.0)
    monkeypatch.setattr(audio_player, "_PLAYBACK_CHUNK_S", 0.2)

    audio_player.play_audio(np.arange(5, dtype=np.float32), 10)

    stream = fake_sd.streams[0]
    assert [chunk.size for chunk in stream.write_calls] == [2, 2, 1]
    np.testing.assert_array_equal(fake_sd.played_audio(), np.arange(5, dtype=np.float32))
    assert fake_sd.play_calls == []
    assert fake_sd.wait_calls == 0
    assert fake_sd.stop_calls == 0


def test_play_audio_prepends_lead_silence_after_resample(
    monkeypatch: MonkeyPatch,
) -> None:
    from hardware import audio_player

    fake_sd = FakeSoundDevice(
        [
            {
                "name": "USB PnP Audio Device",
                "max_output_channels": 2,
                "default_samplerate": 48000,
            }
        ],
        default_output=0,
    )
    monkeypatch.setattr(audio_player, "_sd", fake_sd)
    monkeypatch.setattr(audio_player, "_PLAYBACK_LEAD_SILENCE_S", 0.01)
    monkeypatch.setattr(audio_player, "_PLAYBACK_TRAIL_SILENCE_S", 0.0)

    audio_player.play_audio(np.linspace(-0.5, 0.5, 441, dtype=np.float32), 44100)

    played_audio = fake_sd.played_audio()
    lead_samples = 480
    assert played_audio.dtype == np.float32
    assert fake_sd.output_stream_calls[0]["samplerate"] == 48000
    assert played_audio.size == lead_samples + 480
    np.testing.assert_array_equal(
        played_audio[:lead_samples],
        np.zeros(lead_samples, dtype=np.float32),
    )
    assert np.any(played_audio[lead_samples:] != 0.0)


def test_play_audio_appends_trail_silence_after_lead(
    monkeypatch: MonkeyPatch,
) -> None:
    from hardware import audio_player

    fake_sd = FakeSoundDevice(
        [
            {
                "name": "USB PnP Audio Device",
                "max_output_channels": 2,
                "default_samplerate": 48000,
            }
        ],
        default_output=0,
    )
    monkeypatch.setattr(audio_player, "_sd", fake_sd)
    monkeypatch.setattr(audio_player, "_PLAYBACK_LEAD_SILENCE_S", 0.01)
    monkeypatch.setattr(audio_player, "_PLAYBACK_TRAIL_SILENCE_S", 0.02)

    audio_player.play_audio(np.linspace(-0.5, 0.5, 441, dtype=np.float32), 44100)

    played_audio = fake_sd.played_audio()
    lead_samples = 480
    resampled_samples = 480
    trail_samples = 960
    assert played_audio.dtype == np.float32
    assert fake_sd.output_stream_calls[0]["samplerate"] == 48000
    assert played_audio.size == lead_samples + resampled_samples + trail_samples
    np.testing.assert_array_equal(
        played_audio[:lead_samples],
        np.zeros(lead_samples, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        played_audio[-trail_samples:],
        np.zeros(trail_samples, dtype=np.float32),
    )
    assert np.any(played_audio[lead_samples:-trail_samples] != 0.0)


def test_play_audio_lead_silence_env_zero_disables_prepend(
    monkeypatch: MonkeyPatch,
) -> None:
    from hardware import audio_player

    monkeypatch.setenv("MUNGI_AUDIO_LEAD_SILENCE", "0")
    monkeypatch.setenv("MUNGI_AUDIO_TRAIL_SILENCE", "0")
    reloaded_audio_player = importlib.reload(audio_player)
    fake_sd = FakeSoundDevice(
        [
            {
                "name": "USB PnP Audio Device",
                "max_output_channels": 2,
                "default_samplerate": 48000,
            }
        ],
        default_output=0,
    )
    monkeypatch.setattr(reloaded_audio_player, "_sd", fake_sd)

    reloaded_audio_player.play_audio(
        np.linspace(-0.5, 0.5, 441, dtype=np.float32),
        44100,
    )

    played_audio = fake_sd.played_audio()
    assert reloaded_audio_player._PLAYBACK_LEAD_SILENCE_S == 0.0
    assert fake_sd.output_stream_calls[0]["samplerate"] == 48000
    assert played_audio.size == 480

    monkeypatch.delenv("MUNGI_AUDIO_LEAD_SILENCE", raising=False)
    monkeypatch.delenv("MUNGI_AUDIO_TRAIL_SILENCE", raising=False)
    importlib.reload(audio_player)


def test_play_audio_trail_silence_env_zero_disables_append(
    monkeypatch: MonkeyPatch,
) -> None:
    from hardware import audio_player

    monkeypatch.setenv("MUNGI_AUDIO_LEAD_SILENCE", "0")
    monkeypatch.setenv("MUNGI_AUDIO_TRAIL_SILENCE", "0")
    reloaded_audio_player = importlib.reload(audio_player)
    fake_sd = FakeSoundDevice(
        [
            {
                "name": "USB PnP Audio Device",
                "max_output_channels": 2,
                "default_samplerate": 48000,
            }
        ],
        default_output=0,
    )
    monkeypatch.setattr(reloaded_audio_player, "_sd", fake_sd)

    reloaded_audio_player.play_audio(
        np.linspace(-0.5, 0.5, 441, dtype=np.float32),
        44100,
    )

    played_audio = fake_sd.played_audio()
    assert reloaded_audio_player._PLAYBACK_TRAIL_SILENCE_S == 0.0
    assert fake_sd.output_stream_calls[0]["samplerate"] == 48000
    assert played_audio.size == 480

    monkeypatch.delenv("MUNGI_AUDIO_LEAD_SILENCE", raising=False)
    monkeypatch.delenv("MUNGI_AUDIO_TRAIL_SILENCE", raising=False)
    importlib.reload(audio_player)


def test_play_audio_resamples_to_device_default(monkeypatch: MonkeyPatch) -> None:
    from hardware import audio_player

    fake_sd = FakeSoundDevice(
        [
            {
                "name": "USB PnP Audio Device",
                "max_output_channels": 2,
                "default_samplerate": 48000,
            }
        ],
        default_output=0,
    )
    monkeypatch.setattr(audio_player, "_sd", fake_sd)
    monkeypatch.setattr(audio_player, "_PLAYBACK_LEAD_SILENCE_S", 0.0)
    monkeypatch.setattr(audio_player, "_PLAYBACK_TRAIL_SILENCE_S", 0.0)

    audio_player.play_audio(np.linspace(-0.5, 0.5, 441, dtype=np.float32), 44100)

    played_audio = fake_sd.played_audio()
    assert fake_sd.output_stream_calls[0]["samplerate"] == 48000
    assert played_audio.size == 480


def test_play_audio_passes_resolved_latency_to_output_stream(
    monkeypatch: MonkeyPatch,
) -> None:
    from hardware import audio_player

    fake_sd = FakeSoundDevice(
        [{"name": "USB PnP Audio Device", "max_output_channels": 2}],
        default_output=0,
    )
    monkeypatch.setattr(audio_player, "_sd", fake_sd)
    monkeypatch.setattr(
        audio_player,
        "_DEFAULT_AUDIO_LATENCY",
        audio_player._resolve_audio_latency("0.25"),
    )

    audio_player.play_audio(np.array([0.1, -0.1], dtype=np.float32), 22050)

    assert fake_sd.output_stream_calls[0]["latency"] == 0.25


def test_play_audio_aborts_stream_when_stop_requested_mid_playback(
    monkeypatch: MonkeyPatch,
) -> None:
    from hardware import audio_player

    fake_sd = FakeSoundDevice(
        [
            {
                "name": "USB PnP Audio Device",
                "max_output_channels": 2,
                "default_samplerate": 10,
            }
        ],
        default_output=0,
    )
    monkeypatch.setattr(audio_player, "_sd", fake_sd)
    monkeypatch.setattr(audio_player, "_PLAYBACK_LEAD_SILENCE_S", 0.0)
    monkeypatch.setattr(audio_player, "_PLAYBACK_TRAIL_SILENCE_S", 0.0)
    monkeypatch.setattr(audio_player, "_PLAYBACK_CHUNK_S", 0.2)

    def request_stop_after_first_write(_stream: FakeOutputStream) -> None:
        if fake_sd.played_audio().size == 2:
            audio_player.stop_playback()

    fake_sd.on_write = request_stop_after_first_write

    audio_player.play_audio(np.arange(6, dtype=np.float32), 10)

    np.testing.assert_array_equal(fake_sd.played_audio(), np.array([0.0, 1.0], dtype=np.float32))
    assert fake_sd.stream_abort_calls() == 1
    assert fake_sd.stop_calls == 0


def test_play_audio_nonblocking_runs_daemon_chunk_writer(
    monkeypatch: MonkeyPatch,
) -> None:
    from hardware import audio_player

    fake_sd = FakeSoundDevice(
        [
            {
                "name": "USB PnP Audio Device",
                "max_output_channels": 2,
                "default_samplerate": 10,
            }
        ],
        default_output=0,
    )
    playback_done = threading.Event()
    fake_sd.on_stream_exit = lambda _stream: playback_done.set()
    monkeypatch.setattr(audio_player, "_sd", fake_sd)
    monkeypatch.setattr(audio_player, "_PLAYBACK_LEAD_SILENCE_S", 0.0)
    monkeypatch.setattr(audio_player, "_PLAYBACK_TRAIL_SILENCE_S", 0.0)
    monkeypatch.setattr(audio_player, "_PLAYBACK_CHUNK_S", 0.2)

    audio_player.play_audio(np.arange(5, dtype=np.float32), 10, blocking=False)

    assert playback_done.wait(timeout=1.0)
    np.testing.assert_array_equal(fake_sd.played_audio(), np.arange(5, dtype=np.float32))
    assert fake_sd.play_calls == []
    assert fake_sd.wait_calls == 0
    assert fake_sd.stop_calls == 0


def test_stop_playback_sets_abort_event_without_sounddevice_stop(
    monkeypatch: MonkeyPatch,
) -> None:
    from hardware import audio_player

    fake_sd = FakeSoundDevice(
        [{"name": "USB PnP Audio Device", "max_output_channels": 2}],
        default_output=0,
    )
    monkeypatch.setattr(audio_player, "_sd", fake_sd)
    audio_player._playback_abort.clear()

    audio_player.stop_playback()

    assert audio_player._playback_abort.is_set()
    assert fake_sd.stop_calls == 0
