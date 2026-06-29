"""Immutable RAM-loaded sound bank for touchscreen session cues."""

from __future__ import annotations

import random
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

WAKE_DAY_HOUR_OFFSET = 5

AudioClip = tuple[np.ndarray, int]


class SoundBank:
    """All sound categories loaded into RAM at init."""

    def __init__(self, sounds_dir: Path) -> None:
        """Load wake, error, sleep, and feedback sounds from ``sounds_dir``."""
        self._wake = self._load_dir_immutable(sounds_dir / "wake")
        self._error = self._load_dir_immutable(sounds_dir / "error")
        self._sleep = self._load_flat_immutable(sounds_dir / "sleep")
        self._chime = self._load_one_immutable(sounds_dir / "feedback" / "long_press_chime.wav")
        self._ack = self._load_optional_ack(sounds_dir)
        self._language_switch = self._load_optional_feedback_pool(
            sounds_dir,
            "language_switch",
        )
        self._funny_english_bgm = self._load_optional_repo_wav(
            Path("assets") / "funny_english" / "music" / "bgm_loop.wav"
        )
        # Round-robin cursor for same-day wake acknowledgements. A random start
        # index keeps the first ack after a process restart varied, while the
        # rotation guarantees every clip plays once before any repeat.
        ack_pool = self._wake.get("wake_ack", ())
        self._wake_ack_cursor: int = random.randrange(len(ack_pool)) if ack_pool else 0

    def pick_wake(self, now: datetime, last_wake_date: date | None) -> AudioClip:
        """Pick a first-of-day welcome or same-day wake acknowledgement.

        The first-of-day welcome is chosen at random from the time-bucket pool.
        Same-day acknowledgements rotate round-robin through the ``wake_ack``
        pool so every clip plays once before any repeats.
        """
        wake_day = (now - timedelta(hours=WAKE_DAY_HOUR_OFFSET)).date()
        is_first_of_day = last_wake_date != wake_day
        if is_first_of_day:
            bucket = self._time_bucket(now.hour)
            return random.choice(self._wake[f"welcome_{bucket}"])
        ack_pool = self._wake["wake_ack"]
        clip = ack_pool[self._wake_ack_cursor % len(ack_pool)]
        self._wake_ack_cursor += 1
        return clip

    def pick_error(self, kind: str = "stt_load_fail") -> AudioClip:
        """Pick an error cue by kind."""
        return random.choice(self._error[kind])

    def pick_sleep(self) -> AudioClip:
        """Pick a sleep cue."""
        return random.choice(self._sleep)

    def pick_ack(self) -> AudioClip | None:
        """Pick a short tap acknowledgement cue, or ``None`` when no asset exists."""
        if not self._ack:
            return None
        return random.choice(self._ack)

    def pick_language_switch(self) -> AudioClip | None:
        """Pick a language-switch cue, or ``None`` when no asset exists."""
        if not self._language_switch:
            return None
        return random.choice(self._language_switch)

    def funny_english_bgm(self) -> AudioClip | None:
        """Return the optional Funny English BGM loop."""
        return self._funny_english_bgm

    def chime(self) -> AudioClip:
        """Return the long-press chime clip."""
        return self._chime

    @staticmethod
    def _time_bucket(hour: int) -> str:
        """Return the wake greeting bucket for an hour in local time."""
        if 5 <= hour <= 10:
            return "morning"
        if 11 <= hour <= 16:
            return "afternoon"
        if 17 <= hour <= 20:
            return "evening"
        return "night"

    def _load_dir_immutable(self, root: Path) -> dict[str, tuple[AudioClip, ...]]:
        """Load every immediate child directory into immutable clip tuples."""
        if not root.exists():
            msg = f"Sound directory not found: {root}"
            raise FileNotFoundError(msg)
        loaded: dict[str, tuple[AudioClip, ...]] = {}
        for child in sorted(path for path in root.iterdir() if path.is_dir()):
            loaded[child.name] = self._load_flat_immutable(child)
        return loaded

    def _load_flat_immutable(self, root: Path) -> tuple[AudioClip, ...]:
        """Load all WAV files in one directory into an immutable tuple."""
        if not root.exists():
            msg = f"Sound directory not found: {root}"
            raise FileNotFoundError(msg)
        clips = tuple(self._load_one_immutable(path) for path in sorted(root.glob("*.wav")))
        if not clips:
            msg = f"No WAV files found in {root}"
            raise FileNotFoundError(msg)
        return clips

    def _load_optional_ack(self, sounds_dir: Path) -> tuple[AudioClip, ...]:
        """Load optional acknowledgement clips without requiring assets to exist."""
        try:
            return self._load_flat_immutable(sounds_dir / "feedback" / "ack")
        except FileNotFoundError:
            pass
        ack_file = sounds_dir / "feedback" / "ack.wav"
        if not ack_file.exists():
            return ()
        try:
            return (self._load_one_immutable(ack_file),)
        except FileNotFoundError:
            return ()

    def _load_optional_feedback_pool(
        self,
        sounds_dir: Path,
        pool_name: str,
    ) -> tuple[AudioClip, ...]:
        """Load an optional feedback cue directory without requiring assets to exist."""
        try:
            return self._load_flat_immutable(sounds_dir / "feedback" / pool_name)
        except FileNotFoundError:
            return ()

    def _load_optional_repo_wav(self, path: Path) -> AudioClip | None:
        """Load one optional repository WAV without requiring it to exist."""
        if not path.exists():
            return None
        try:
            return self._load_one_immutable(path)
        except FileNotFoundError:
            return None

    def _load_one_immutable(self, path: Path) -> AudioClip:
        """Load one WAV file and mark its samples read-only."""
        wavfile = _load_wavfile_module()
        sample_rate, audio = wavfile.read(path)
        samples = np.asarray(audio)
        samples.setflags(write=False)
        return samples, int(sample_rate)


def _load_wavfile_module() -> Any:
    """Import scipy wavfile lazily for ARM64-friendly optional loading."""
    import scipy.io.wavfile as wavfile  # type: ignore[import-not-found, import-untyped]

    return wavfile
