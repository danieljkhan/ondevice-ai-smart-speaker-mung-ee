"""Tests for the touchscreen wake-audio generator."""

from __future__ import annotations

import wave
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from scripts import generate_wake_audio

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_JSON = REPO_ROOT / "assets" / "sounds" / "scripts.json"


class FakeSupertonicEngine:
    """Minimal SupertonicEngine stand-in for wake script tests."""

    instances: list[FakeSupertonicEngine] = []

    def __init__(self, model_dir: str, voice_style: str = "F1") -> None:
        self.model_dir = model_dir
        self.voice_style = voice_style
        self.loaded = False
        self.calls: list[tuple[str | None, str, int]] = []
        FakeSupertonicEngine.instances.append(self)

    def load(self) -> None:
        """Record that the fake model was loaded."""
        self.loaded = True

    def synthesize(
        self,
        text: str | None,
        language: str = "ko",
        total_steps: int = 7,
    ) -> tuple[np.ndarray, int]:
        """Return deterministic 44.1 kHz audio for one script entry."""
        self.calls.append((text, language, total_steps))
        return np.linspace(-0.25, 0.25, 441, dtype=np.float32), 44_100


def _read_wav_metadata(path: Path) -> tuple[int, int, int]:
    """Return sample rate, channels, and sample width for a generated WAV."""
    with wave.open(str(path), "rb") as wav_file:
        return wav_file.getframerate(), wav_file.getnchannels(), wav_file.getsampwidth()


def test_scripts_json_schema_and_phase1_inventory() -> None:
    """The committed manifest should match the Phase 1 inventory contract."""
    manifest = generate_wake_audio.load_scripts_manifest(SCRIPTS_JSON)
    counts = Counter(entry.category for entry in manifest.files)

    assert manifest.version == "1.0"
    assert manifest.voice_style == "mung-ee"
    assert manifest.sample_rate == 44_100
    assert manifest.speed == 1.0
    assert manifest.language_default == "ko"
    assert len(manifest.files) == 27
    assert counts == generate_wake_audio.EXPECTED_CATEGORY_COUNTS
    assert len({entry.path for entry in manifest.files}) == 27
    assert len({entry.text for entry in manifest.files}) == 27
    assert all(entry.language == "ko" for entry in manifest.files)


def test_dry_run_prints_27_entry_plan_without_runtime_assets(
    tmp_path: Path,
    capsys: Any,
) -> None:
    """Dry-run mode should not require local Supertonic model or voice files."""
    output_root = tmp_path / "sounds"

    result = generate_wake_audio.main(
        [
            "--scripts-json",
            str(SCRIPTS_JSON),
            "--output-root",
            str(output_root),
            "--model-dir",
            str(tmp_path / "missing-model"),
            "--voice-json",
            str(tmp_path / "missing-tobi.json"),
            "--total-steps",
            "24",
            "--dry-run",
        ],
    )

    captured = capsys.readouterr()
    assert result == 0
    assert "Synthesis plan: 27 voice file(s)" in captured.out
    assert "27/27" in captured.out
    normalized_output = captured.out.replace("\\", "/")
    assert "wake/welcome_morning/01.wav" in normalized_output
    assert "sleep/03.wav" in normalized_output


def test_generate_filtered_category_with_fake_supertonic(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Filtered generation should synthesize only the selected category."""
    output_root = tmp_path / "sounds"
    model_dir = tmp_path / "supertonic-2"
    voice_json = tmp_path / "tobi.json"
    model_dir.mkdir()
    voice_json.write_text("{}", encoding="utf-8")
    FakeSupertonicEngine.instances.clear()
    monkeypatch.setattr(generate_wake_audio, "SupertonicEngine", FakeSupertonicEngine)

    result = generate_wake_audio.main(
        [
            "--scripts-json",
            str(SCRIPTS_JSON),
            "--output-root",
            str(output_root),
            "--model-dir",
            str(model_dir),
            "--voice-json",
            str(voice_json),
            "--filter",
            "sleep",
            "--total-steps",
            "24",
        ],
    )

    assert result == 0
    assert len(FakeSupertonicEngine.instances) == 1
    assert FakeSupertonicEngine.instances[0].loaded is True
    assert len(FakeSupertonicEngine.instances[0].calls) == 3
    assert {call[2] for call in FakeSupertonicEngine.instances[0].calls} == {24}
    for index in range(1, 4):
        wav_path = output_root / "sleep" / f"{index:02d}.wav"
        assert wav_path.is_file()
        assert _read_wav_metadata(wav_path) == (44_100, 1, 2)


def test_supertonic_synthesize_total_steps_default_and_override() -> None:
    """Supertonic synthesis should default to 10 steps and honor overrides."""
    from unittest.mock import MagicMock, call

    from models.tts_runner import SupertonicEngine

    engine = SupertonicEngine("unused")
    engine._model = MagicMock()
    engine._model.synthesize.return_value = np.array([0.1], dtype=np.float32)

    engine.synthesize("default")
    engine.synthesize("offline cue", total_steps=24)

    engine._model.synthesize.assert_has_calls(
        [
            call("default", lang="ko", speed=0.95, total_steps=10),
            call("offline cue", lang="ko", speed=0.95, total_steps=24),
        ],
    )


def test_missing_runtime_paths_return_error(tmp_path: Path) -> None:
    """Non-dry-run generation should fail cleanly when runtime assets are absent."""
    result = generate_wake_audio.main(
        [
            "--scripts-json",
            str(SCRIPTS_JSON),
            "--output-root",
            str(tmp_path / "sounds"),
            "--model-dir",
            str(tmp_path / "missing-model"),
            "--voice-json",
            str(tmp_path / "missing-tobi.json"),
            "--filter",
            "sleep",
        ],
    )

    assert result == 1


def test_resolve_output_path_rejects_traversal(tmp_path: Path) -> None:
    """Manifest paths may not escape the selected output root."""
    import pytest

    with pytest.raises(ValueError, match="relative"):
        generate_wake_audio.resolve_output_path(tmp_path / "sounds", "../escape.wav")


def test_filter_selects_expected_entry_count() -> None:
    """Category filtering should preserve the manifest order for matching entries."""
    manifest = generate_wake_audio.load_scripts_manifest(SCRIPTS_JSON)

    evening_entries = generate_wake_audio.select_entries(manifest, "welcome_evening")

    assert [entry.path for entry in evening_entries] == [
        "wake/welcome_evening/01.wav",
        "wake/welcome_evening/02.wav",
        "wake/welcome_evening/03.wav",
        "wake/welcome_evening/04.wav",
    ]
