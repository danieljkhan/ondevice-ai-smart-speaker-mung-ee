"""Quality gates for KO/EN language-switch UI assets."""

from __future__ import annotations

import wave
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image  # type: ignore[import-not-found, import-untyped]

from core import sound_bank
from core.sound_bank import SoundBank

BADGE_DIR = Path("assets") / "character" / "indicator"
SOUNDS_DIR = Path("assets") / "sounds"
LANGUAGE_SWITCH_DIR = SOUNDS_DIR / "feedback" / "language_switch"


@pytest.mark.parametrize("filename", ["flag_ko.png", "flag_en.png"])
def test_language_badges_are_rgba_and_anti_aliased(filename: str) -> None:
    """Language badges must stay 48x48 RGBA and above placeholder-grade art quality."""
    path = BADGE_DIR / filename
    with Image.open(path) as image:
        assert image.mode == "RGBA"
        assert image.size == (48, 48)
        rgba = image.convert("RGBA")

    pixels = np.asarray(rgba)
    alpha = pixels[:, :, 3]
    intermediate_alpha = alpha[(alpha > 0) & (alpha < 255)]
    distinct_intermediate_alpha = np.unique(intermediate_alpha)
    unique_colors = {tuple(pixel) for row in pixels.tolist() for pixel in row}

    assert intermediate_alpha.size >= 80
    assert distinct_intermediate_alpha.size >= 8
    assert len(unique_colors) >= 50


def _read_wav_as_int16(path: Path) -> tuple[int, np.ndarray]:
    """Read one PCM WAV file into an int16 array for SoundBank tests."""
    with wave.open(str(path), "rb") as wav_file:
        sample_rate = int(wav_file.getframerate())
        channel_count = int(wav_file.getnchannels())
        sample_width = int(wav_file.getsampwidth())
        frames = wav_file.readframes(wav_file.getnframes())
    if sample_width != 2:
        msg = f"expected 16-bit PCM WAV, got sample width {sample_width}: {path}"
        raise ValueError(msg)
    audio = np.frombuffer(frames, dtype="<i2")
    if channel_count > 1:
        audio = audio.reshape(-1, channel_count)
    return sample_rate, audio.copy()


def test_language_switch_cue_asset_is_loadable_and_pickable(monkeypatch: Any) -> None:
    """Language-switch cue pool should contain a loadable WAV picked by SoundBank."""
    cue_paths = sorted(LANGUAGE_SWITCH_DIR.glob("*.wav"))
    assert cue_paths
    for path in cue_paths:
        sample_rate, audio = _read_wav_as_int16(path)
        assert sample_rate > 0
        assert audio.size > 0
        assert 0.35 <= audio.shape[0] / sample_rate <= 0.50

    monkeypatch.setattr(
        sound_bank,
        "_load_wavfile_module",
        lambda: SimpleNamespace(read=_read_wav_as_int16),
    )
    bank = SoundBank(SOUNDS_DIR)
    cue = bank.pick_language_switch()

    assert cue is not None
    audio, sample_rate = cue
    assert sample_rate > 0
    assert audio.flags.writeable is False
