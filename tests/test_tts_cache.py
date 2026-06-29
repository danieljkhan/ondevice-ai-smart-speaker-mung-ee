"""Tests for the validated pre-rendered TTS cache loader."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import wave
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from core.pipeline import ConversationPipeline, PipelineConfig, TurnMetrics
from models import tts_cache, tts_runner
from safety.approved_template_router import fixed_response_cache_texts


@pytest.fixture(autouse=True)
def _reset_tts_cache() -> None:
    """Keep module-level cache state isolated between tests."""
    tts_runner._set_active_supertonic_engine(None)
    tts_cache._reset_for_tests()
    yield
    tts_runner._set_active_supertonic_engine(None)
    tts_cache._reset_for_tests()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model_dir_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(path.rglob("*"), key=lambda item: item.relative_to(path).as_posix()):
        if not child.is_file():
            continue
        stat = child.stat()
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _write_wav(
    path: Path,
    *,
    sample_rate: int = 16_000,
    channels: int = 1,
    sample_width: int = 2,
    frame_count: int = 4,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if sample_width == 2:
        frames = b"\x00\x00" * frame_count * channels
    else:
        frames = b"\x00" * frame_count * channels * sample_width
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(sample_width)
        handle.setframerate(sample_rate)
        handle.writeframes(frames)


def _write_runtime_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    cache_dir = tmp_path / "cache"
    model_dir = tmp_path / "model"
    voice_dir = tmp_path / "voices"
    model_dir.mkdir()
    voice_dir.mkdir()
    (model_dir / "model.bin").write_bytes(b"supertonic model")
    ko_voice = voice_dir / "ko.json"
    en_voice = voice_dir / "en.json"
    ko_voice.write_bytes(b'{"voice":"ko"}')
    en_voice.write_bytes(b'{"voice":"en"}')
    monkeypatch.setenv("MUNGI_TTS_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("MUNGI_TTS_MODEL_DIR", str(model_dir))
    monkeypatch.setenv("MUNGI_TTS_VOICE_STYLE_KO", str(ko_voice))
    monkeypatch.setenv("MUNGI_TTS_VOICE_STYLE_EN", str(en_voice))
    return {
        "cache_dir": cache_dir,
        "model_dir": model_dir,
        "ko_voice": ko_voice,
        "en_voice": en_voice,
    }


def _write_meta(cache_dir: Path, identity: dict[str, Any]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "cache_meta.json").write_text(json.dumps(identity), encoding="utf-8")


def _write_manifest(cache_dir: Path, payload: dict[str, Any]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_cache_entry(
    cache_dir: Path,
    *,
    text: str,
    lang: str,
    sample_rate: int = 16_000,
    channels: int = 1,
    sample_width: int = 2,
    frame_count: int = 4,
    manifest_sha256: str | None = None,
    manifest_bytes: int | None = None,
    write_file: bool = True,
) -> Path:
    key = tts_cache.compute_key(text, lang)
    wav_path = cache_dir / f"{key}.wav"
    if write_file:
        _write_wav(
            wav_path,
            sample_rate=sample_rate,
            channels=channels,
            sample_width=sample_width,
            frame_count=frame_count,
        )
        checksum = manifest_sha256 if manifest_sha256 is not None else _sha256_file(wav_path)
        byte_count = manifest_bytes if manifest_bytes is not None else wav_path.stat().st_size
    else:
        checksum = manifest_sha256 or ("0" * 64)
        byte_count = manifest_bytes or 128
    _write_manifest(
        cache_dir,
        {
            key: {
                "wav": wav_path.name,
                "sha256": checksum,
                "bytes": byte_count,
                "lang": lang,
            }
        },
    )
    return wav_path


def test_compute_key_is_deterministic_and_uses_normalized_text() -> None:
    """Equivalent normalized text should produce one cache key."""
    assert tts_cache.compute_key("hello   world", "en") == tts_cache.compute_key(
        "hello world", "en"
    )
    assert tts_cache.compute_key("hello world", "ko") != tts_cache.compute_key("hello world", "en")


def test_compute_key_uses_cjk_normalized_text() -> None:
    """CJK-normalized texts should share a stable cache key."""
    cjk_text = "\uace0(\u53e4)"
    normalized_text = "\uace0"
    key = tts_cache.compute_key(cjk_text, "ko")
    raw_key = hashlib.sha256(f"ko\x1f{cjk_text}".encode()).hexdigest()

    assert len(key) == 64
    assert key == tts_cache.compute_key(cjk_text, "ko")
    assert key == tts_cache.compute_key(normalized_text, "ko")
    assert key == tts_cache.compute_key("\uace0(\u9ad8)", "ko")
    assert key != raw_key


def test_compute_key_uses_unsupported_character_normalized_text() -> None:
    """Unsupported-character-normalized texts should share stable cache keys."""
    normalized_text = "\ubc31\uc81c \uac00\uc57c"
    key = tts_cache.compute_key("\ubc31\uc81c\u318d\uac00\uc57c", "ko")

    assert len(key) == 64
    assert key == tts_cache.compute_key("\ubc31\uc81c\u318d\uac00\uc57c", "ko")
    assert key == tts_cache.compute_key(normalized_text, "ko")
    assert key == tts_cache.compute_key("\ubc31\uc81c\u00b7\uac00\uc57c", "ko")
    assert tts_cache.compute_key("900\u2103", "ko") == tts_cache.compute_key("900\ub3c4", "ko")


def test_runtime_identity_helper_hashes_model_and_voice_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runtime identity helper is testable without a cache directory."""
    paths = _write_runtime_files(tmp_path, monkeypatch)

    identity = tts_cache._build_runtime_identity(
        model_dir=paths["model_dir"],
        voice_style_ko=str(paths["ko_voice"]),
        voice_style_en=str(paths["en_voice"]),
    )

    assert identity["schema_version"] == 1
    assert identity["engine_id"] == "supertonic"
    assert identity["model_id"] == _model_dir_fingerprint(paths["model_dir"])
    assert identity["voice_id_ko"] == _sha256_file(paths["ko_voice"])
    assert identity["voice_id_en"] == _sha256_file(paths["en_voice"])
    assert identity["speed"] == 0.95
    assert identity["total_steps"] == 30
    assert identity["sample_rate"] == 16_000
    assert identity["audio_format"] == "pcm_s16le"


def test_runtime_identity_model_id_uses_fast_metadata_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Model identity detects metadata changes without reading file content."""
    paths = _write_runtime_files(tmp_path, monkeypatch)
    model_file = paths["model_dir"] / "model.bin"
    original_stat = model_file.stat()

    first = tts_cache._build_runtime_identity(
        model_dir=paths["model_dir"],
        voice_style_ko=str(paths["ko_voice"]),
        voice_style_en=str(paths["en_voice"]),
    )["model_id"]
    model_file.write_bytes(b"different model!")
    os.utime(model_file, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    same_metadata = tts_cache._build_runtime_identity(
        model_dir=paths["model_dir"],
        voice_style_ko=str(paths["ko_voice"]),
        voice_style_en=str(paths["en_voice"]),
    )["model_id"]
    model_file.write_bytes(b"different model with new size")
    changed_size = tts_cache._build_runtime_identity(
        model_dir=paths["model_dir"],
        voice_style_ko=str(paths["ko_voice"]),
        voice_style_en=str(paths["en_voice"]),
    )["model_id"]

    assert same_metadata == first
    assert changed_size != first


def test_lookup_returns_none_when_cache_meta_is_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing cache identity disables cache reads."""
    _write_runtime_files(tmp_path, monkeypatch)

    assert tts_cache.lookup("hello", "en") is None


def test_lookup_valid_hit_returns_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A manifest hit returns only after identity and WAV validation pass."""
    paths = _write_runtime_files(tmp_path, monkeypatch)
    identity = tts_cache._build_runtime_identity()
    cache_dir = paths["cache_dir"]
    _write_meta(cache_dir, identity)
    wav_path = _write_cache_entry(cache_dir, text="hello", lang="en")
    tts_cache._reset_for_tests()

    assert tts_cache.lookup("hello", "en") == wav_path


def test_lookup_identity_mismatch_disables_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Any cache identity mismatch makes otherwise valid hits miss."""
    paths = _write_runtime_files(tmp_path, monkeypatch)
    identity = tts_cache._build_runtime_identity()
    identity["model_id"] = "other-model"
    cache_dir = paths["cache_dir"]
    _write_meta(cache_dir, identity)
    _write_cache_entry(cache_dir, text="hello", lang="en")
    tts_cache._reset_for_tests()

    assert tts_cache.lookup("hello", "en") is None


def test_lookup_manifest_miss_returns_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A valid identity with no manifest entry is a miss."""
    paths = _write_runtime_files(tmp_path, monkeypatch)
    cache_dir = paths["cache_dir"]
    _write_meta(cache_dir, tts_cache._build_runtime_identity())
    _write_manifest(cache_dir, {})
    tts_cache._reset_for_tests()

    assert tts_cache.lookup("missing text", "en") is None


@pytest.mark.parametrize(
    "entry_kwargs",
    [
        {"write_file": False},
        {"manifest_bytes": 999_999},
        {"manifest_sha256": "0" * 64},
        {"sample_rate": 22_050},
        {"channels": 2},
        {"sample_width": 1},
        {"frame_count": 0},
    ],
)
def test_lookup_rejects_bad_hits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entry_kwargs: dict[str, Any],
) -> None:
    """Bad manifest or WAV hits are rejected as misses."""
    paths = _write_runtime_files(tmp_path, monkeypatch)
    cache_dir = paths["cache_dir"]
    _write_meta(cache_dir, tts_cache._build_runtime_identity())
    _write_cache_entry(cache_dir, text="hello", lang="en", **entry_kwargs)
    tts_cache._reset_for_tests()

    assert tts_cache.lookup("hello", "en") is None


def test_live_conversation_run_tts_does_not_reference_tts_cache() -> None:
    """The live conversation TTS path remains direct synthesis only."""
    source = inspect.getsource(ConversationPipeline._run_tts)

    assert "tts_cache" not in source
    assert "models.tts_cache" not in source


def test_fixed_response_uses_validated_cache_before_live_tts(tmp_path: Path) -> None:
    """Fixed block responses should play validated cache WAVs before live synthesis."""
    wav_path = tmp_path / "cached.wav"
    _write_wav(wav_path)
    pipeline = ConversationPipeline(MagicMock(), PipelineConfig(play_tts_audio=False))
    metrics = TurnMetrics()
    response_text = sorted(fixed_response_cache_texts())[0]

    with (
        patch("core.pipeline.tts_cache.lookup", return_value=wav_path) as mock_lookup,
        patch.object(pipeline, "_load_tts_and_sync_system_state") as mock_load_tts,
        patch.object(pipeline, "_run_tts") as mock_run_tts,
        patch.object(pipeline, "_maybe_unload_tts_after_success"),
        patch.object(pipeline, "_post_turn_maintenance"),
        patch.object(pipeline, "_maybe_drop_page_cache_after_success"),
    ):
        result = pipeline._return_fixed_tts_response(
            user_text="너는 누구야?",
            response_text=response_text,
            language="ko",
            detected_language="ko",
            metrics=metrics,
            turn_start=0.0,
            result_raw_stt_text="너는 누구야?",
            turn_num=None,
            input_wav=None,
        )

    mock_lookup.assert_called_once_with(response_text, "ko")
    mock_load_tts.assert_not_called()
    mock_run_tts.assert_not_called()
    assert result.metrics.tts_load_time_s == 0.0
    assert result.sample_rate == 16_000
    assert result.audio_samples.tolist() == [0, 0, 0, 0]


def test_fixed_response_cache_lookup_precedes_tts_load_on_allowlisted_miss() -> None:
    """Allowlisted cache misses should check the cache before loading live TTS."""
    pipeline = ConversationPipeline(MagicMock(), PipelineConfig(play_tts_audio=False))
    metrics = TurnMetrics()
    response_text = sorted(fixed_response_cache_texts())[0]
    events: list[str] = []

    def record_lookup(text: str, lang: str) -> None:
        del text, lang
        events.append("lookup")
        return None

    with (
        patch("core.pipeline.tts_cache.lookup", side_effect=record_lookup),
        patch.object(
            pipeline,
            "_load_tts_and_sync_system_state",
            side_effect=lambda: events.append("load_tts"),
        ),
        patch.object(pipeline, "_run_tts", return_value=([0.1] * 10, 22050)),
        patch.object(pipeline, "_maybe_unload_tts_after_success"),
        patch.object(pipeline, "_post_turn_maintenance"),
        patch.object(pipeline, "_maybe_drop_page_cache_after_success"),
    ):
        pipeline._return_fixed_tts_response(
            user_text="너는 누구야?",
            response_text=response_text,
            language="ko",
            detected_language="ko",
            metrics=metrics,
            turn_start=0.0,
            result_raw_stt_text="너는 누구야?",
            turn_num=None,
            input_wav=None,
        )

    assert events == ["lookup", "load_tts"]


def test_fixed_response_cache_lookup_is_allowlisted_only() -> None:
    """Non-allowlisted fixed responses should not query the TTS cache."""
    pipeline = ConversationPipeline(MagicMock(), PipelineConfig(play_tts_audio=False))
    metrics = TurnMetrics()

    with (
        patch("core.pipeline.tts_cache.lookup") as mock_lookup,
        patch.object(pipeline, "_load_tts_and_sync_system_state"),
        patch.object(pipeline, "_run_tts", return_value=([0.1] * 10, 22050)) as mock_run_tts,
        patch.object(pipeline, "_maybe_unload_tts_after_success"),
        patch.object(pipeline, "_post_turn_maintenance"),
        patch.object(pipeline, "_maybe_drop_page_cache_after_success"),
    ):
        pipeline._return_fixed_tts_response(
            user_text="",
            response_text="not an approved prebaked response",
            language="ko",
            detected_language="ko",
            metrics=metrics,
            turn_start=0.0,
            result_raw_stt_text="",
            turn_num=None,
            input_wav=None,
        )

    mock_lookup.assert_not_called()
    mock_run_tts.assert_called_once_with("not an approved prebaked response", language="ko")
