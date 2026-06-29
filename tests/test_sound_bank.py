"""Tests for immutable RAM-loaded sound bank behavior."""

from __future__ import annotations

from collections import Counter
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from core import sound_bank
from core.sound_bank import WAKE_DAY_HOUR_OFFSET, SoundBank


def create_sound_layout(root: Path) -> None:
    """Create the directory and WAV fixture files SoundBank expects."""
    for relative in (
        "wake/welcome_morning/01.wav",
        "wake/welcome_afternoon/01.wav",
        "wake/welcome_evening/01.wav",
        "wake/welcome_night/01.wav",
        "wake/wake_ack/01.wav",
        "error/stt_load_fail/01.wav",
        "sleep/01.wav",
        "feedback/long_press_chime.wav",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake")


def install_fake_wav_reader(monkeypatch: Any) -> None:
    """Patch scipy wav loading with deterministic arrays."""

    def read(path: Path) -> tuple[int, np.ndarray]:
        token = sum(path.as_posix().encode("utf-8")) % 100
        return 44_100, np.array([token, token + 1], dtype=np.int16)

    monkeypatch.setattr(sound_bank, "_load_wavfile_module", lambda: SimpleNamespace(read=read))


def test_loads_categories_and_returns_immutable_arrays(monkeypatch: Any, tmp_path: Path) -> None:
    """SoundBank loads all categories into tuple/read-only containers."""
    create_sound_layout(tmp_path)
    install_fake_wav_reader(monkeypatch)

    bank = SoundBank(tmp_path)
    audio, sample_rate = bank.pick_sleep()

    assert sample_rate == 44_100
    assert audio.flags.writeable is False
    with pytest.raises(ValueError):
        audio[0] = 1
    assert isinstance(bank._sleep, tuple)
    with pytest.raises(AttributeError):
        bank._sleep.append((audio, sample_rate))  # type: ignore[attr-defined]


def test_time_bucket_boundaries(monkeypatch: Any, tmp_path: Path) -> None:
    """Wake greeting buckets switch at 05, 11, 17, and 21 local hours."""
    create_sound_layout(tmp_path)
    install_fake_wav_reader(monkeypatch)
    bank = SoundBank(tmp_path)

    assert WAKE_DAY_HOUR_OFFSET == 5
    expected = {
        4: "night",
        5: "morning",
        10: "morning",
        11: "afternoon",
        16: "afternoon",
        17: "evening",
        20: "evening",
        21: "night",
        23: "night",
        0: "night",
    }
    for hour, bucket in expected.items():
        assert bank._time_bucket(hour) == bucket


def test_pick_wake_uses_0500_wake_day(monkeypatch: Any, tmp_path: Path) -> None:
    """Before 05:00 belongs to the previous wake day; 05:00 starts a new one."""
    create_sound_layout(tmp_path)
    install_fake_wav_reader(monkeypatch)
    bank = SoundBank(tmp_path)

    before_boundary = datetime(2026, 5, 26, 4, 30)
    after_boundary = datetime(2026, 5, 26, 5, 0)

    previous_wake_day = date(2026, 5, 25)
    _, before_sr = bank.pick_wake(before_boundary, previous_wake_day)
    _, after_sr = bank.pick_wake(after_boundary, previous_wake_day)

    assert before_sr == 44_100
    assert after_sr == 44_100
    assert (before_boundary - sound_bank.timedelta(hours=5)).date() == date(2026, 5, 25)
    assert (after_boundary - sound_bank.timedelta(hours=5)).date() == date(2026, 5, 26)


def _add_wake_ack_clips(root: Path, last_index: int) -> None:
    """Add wake_ack fixture WAVs 02..last_index alongside the base layout."""
    for index in range(2, last_index + 1):
        path = root / "wake" / "wake_ack" / f"{index:02d}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake")


def test_pick_wake_rotates_through_all_wake_ack_clips(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Same-day acks rotate round-robin: every clip plays once before any repeat."""
    create_sound_layout(tmp_path)
    _add_wake_ack_clips(tmp_path, last_index=8)
    install_fake_wav_reader(monkeypatch)
    bank = SoundBank(tmp_path)
    bank._wake_ack_cursor = 0  # deterministic start for assertion

    now = datetime(2026, 5, 26, 12, 0)
    wake_day = date(2026, 5, 26)  # equals (now - 5h).date() -> same-day ack branch

    first_cycle = [int(bank.pick_wake(now, wake_day)[0][0]) for _ in range(8)]
    second_cycle = [int(bank.pick_wake(now, wake_day)[0][0]) for _ in range(8)]

    assert len(set(first_cycle)) == 8  # no repeat within one full rotation
    assert first_cycle == second_cycle  # rotation order is stable across cycles
    assert all(count == 2 for count in Counter(first_cycle + second_cycle).values())


def test_pick_wake_ack_cursor_starts_within_pool(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """The random initial cursor stays inside the wake_ack pool bounds."""
    create_sound_layout(tmp_path)
    _add_wake_ack_clips(tmp_path, last_index=8)
    install_fake_wav_reader(monkeypatch)
    bank = SoundBank(tmp_path)

    assert 0 <= bank._wake_ack_cursor < len(bank._wake["wake_ack"])


def test_pick_error_and_chime(monkeypatch: Any, tmp_path: Path) -> None:
    """Error and chime categories return loaded clips."""
    create_sound_layout(tmp_path)
    install_fake_wav_reader(monkeypatch)
    bank = SoundBank(tmp_path)

    assert bank.pick_error("stt_load_fail")[1] == 44_100
    assert bank.chime()[1] == 44_100


def test_pick_ack_returns_none_when_optional_asset_is_missing(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Missing ack assets are optional and return None."""
    create_sound_layout(tmp_path)
    install_fake_wav_reader(monkeypatch)
    bank = SoundBank(tmp_path)

    assert bank.pick_ack() is None
    assert bank.pick_language_switch() is None


def test_pick_ack_returns_loaded_flat_ack_wav(monkeypatch: Any, tmp_path: Path) -> None:
    """Ack assets load from feedback/ack.wav when the flat optional file exists."""
    create_sound_layout(tmp_path)
    (tmp_path / "feedback" / "ack.wav").write_bytes(b"fake")
    install_fake_wav_reader(monkeypatch)
    bank = SoundBank(tmp_path)

    audio, sample_rate = bank.pick_ack() or (None, 0)

    assert sample_rate == 44_100
    assert audio is not None
    assert audio.flags.writeable is False


def test_pick_ack_returns_loaded_optional_clip(monkeypatch: Any, tmp_path: Path) -> None:
    """Ack assets load from the optional feedback/ack pool."""
    create_sound_layout(tmp_path)
    (tmp_path / "feedback" / "ack" / "01.wav").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "feedback" / "ack" / "01.wav").write_bytes(b"fake")
    install_fake_wav_reader(monkeypatch)
    bank = SoundBank(tmp_path)

    audio, sample_rate = bank.pick_ack() or (None, 0)

    assert sample_rate == 44_100
    assert audio is not None
    assert audio.flags.writeable is False


def test_pick_language_switch_returns_loaded_optional_clip(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Language-switch cue assets load from the optional feedback pool."""
    create_sound_layout(tmp_path)
    (tmp_path / "feedback" / "language_switch" / "01.wav").parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    (tmp_path / "feedback" / "language_switch" / "01.wav").write_bytes(b"fake")
    install_fake_wav_reader(monkeypatch)
    bank = SoundBank(tmp_path)

    audio, sample_rate = bank.pick_language_switch() or (None, 0)

    assert sample_rate == 44_100
    assert audio is not None
    assert audio.flags.writeable is False
