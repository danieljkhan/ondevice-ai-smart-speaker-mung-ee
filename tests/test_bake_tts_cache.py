"""Tests for the offline TTS cache bake pipeline."""

from __future__ import annotations

import hashlib
import json
import logging
import wave
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from core.funny_english_mode import FE_KOREAN_SPEECH_LINES
from core.history_mode import CONSENT_PROMPT, CONSENT_REPROMPT, history_narration_segments
from models import tts_cache
from scripts import bake_tts_cache


@pytest.fixture(autouse=True)
def _reset_tts_cache() -> None:
    """Keep cache state isolated between tests."""
    tts_cache._reset_for_tests()
    yield
    tts_cache._reset_for_tests()


class FakeEngine:
    """Fake synth engine that records calls and emits short 16 kHz audio."""

    def __init__(self) -> None:
        self.calls: list[tuple[str | None, str, int]] = []

    def synthesize(
        self,
        text: str | None,
        language: str = "ko",
        total_steps: int = 10,
    ) -> tuple[np.ndarray, int]:
        """Record synth args and return deterministic audio."""
        self.calls.append((text, language, total_steps))
        return np.array([0.0, 0.25, -0.25, 0.0], dtype=np.float32), 16_000


class FailingEngine:
    """Fake synth engine that fails for one configured text."""

    def __init__(self, fail_text: str) -> None:
        self.fail_text = fail_text
        self.calls: list[tuple[str | None, str, int]] = []

    def synthesize(
        self,
        text: str | None,
        language: str = "ko",
        total_steps: int = 10,
    ) -> tuple[np.ndarray, int]:
        """Record synth args, raising for the configured text."""
        self.calls.append((text, language, total_steps))
        if text == self.fail_text:
            raise ValueError("Found 2 unsupported character(s)")
        return np.array([0.0, 0.25, -0.25, 0.0], dtype=np.float32), 16_000


def test_collect_inventory_counts_and_languages() -> None:
    """The production inventory includes every in-scope runtime text class."""
    result = bake_tts_cache.collect_bake_texts()

    assert result.inventory.history_segments > 0
    assert result.inventory.history_lead_ins == 240
    assert result.inventory.history_consent == 2
    assert result.inventory.fe_ko == 8
    # 24 starter EN cards + 15 Aesop story reader cards (stages 6-7) = 39.
    assert result.inventory.fe_en == 39
    assert result.inventory.approved_template_fixed == 2
    assert {item.lang for item in result.items} == {"ko", "en"}


def test_funny_english_ko_collection_uses_shared_inventory() -> None:
    """Funny English Korean bake texts come from the shared runtime tuple."""
    items = bake_tts_cache._collect_funny_english_ko_items()

    assert [item.text for item in items] == list(FE_KOREAN_SPEECH_LINES)
    assert [item.lang for item in items] == ["ko"] * len(FE_KOREAN_SPEECH_LINES)
    assert [item.source for item in items] == [
        f"funny_english:ko:{index}" for index in range(len(FE_KOREAN_SPEECH_LINES))
    ]


def test_approved_template_fixed_collection_uses_verbatim_json() -> None:
    """Approved-template bake texts should be exact KO block responses."""
    items = bake_tts_cache._collect_approved_template_fixed_items(Path("."))
    templates = json.loads(
        Path("assets/filters/approved_templates.json").read_text(encoding="utf-8")
    )

    assert [item.source for item in items] == [
        f"approved_template:fixed:{topic_id}"
        for topic_id in bake_tts_cache.APPROVED_TEMPLATE_FIXED_TOPIC_IDS
    ]
    assert [item.lang for item in items] == ["ko", "ko"]
    assert [item.text for item in items] == [
        templates[topic_id]["response_ko"]
        for topic_id in bake_tts_cache.APPROVED_TEMPLATE_FIXED_TOPIC_IDS
    ]
    assert all("\n\n" in item.text for item in items)
    assert all(item.key == tts_cache.compute_key(item.text, "ko") for item in items)


def test_history_collection_matches_runtime_segmenter(tmp_path: Path) -> None:
    """History collection bakes the same non-pause units runtime playback uses."""
    docs_dir = tmp_path / "assets" / "history" / "docs"
    docs_dir.mkdir(parents=True)
    narration = "앞말입니다. 작은 제목 뒤말입니다."
    (docs_dir / "doc.json").write_text(
        json.dumps(
            {
                "title": "작은 이야기",
                "scenes": [
                    {
                        "seq": 1,
                        "section_title": "작은 제목",
                        "narration": narration,
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    items = bake_tts_cache._collect_history_items(tmp_path)
    scene_texts = [item.text for item in items if item.source.startswith("history:scene:")]

    assert scene_texts == [
        segment
        for segment in history_narration_segments(narration, "작은 제목")
        if segment is not None
    ]
    assert [item.text for item in items if item.source.startswith("history:consent:")] == [
        CONSENT_PROMPT,
        CONSENT_REPROMPT,
    ]


def test_collection_excludes_mode_entry_confirmations() -> None:
    """Mode-entry confirmations stay on the live conversation TTS path."""
    result = bake_tts_cache.collect_bake_texts()
    texts = {item.text for item in result.items}

    assert "좋아! 재미있는 우리역사를 시작할게!" not in texts
    assert "좋아! 퍼니 잉글리시를 시작할게!" not in texts


def test_resume_skips_valid_and_regenerates_missing_or_invalid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Valid done entries are skipped; missing and invalid entries are regenerated."""
    paths = _write_runtime_files(tmp_path, monkeypatch)
    items = (
        _item("already baked", "ko", "test:skip"),
        _item("missing wav", "ko", "test:missing"),
        _item("bad wav", "en", "test:bad"),
    )
    monkeypatch.setattr(bake_tts_cache, "collect_bake_texts", lambda repo_root: _collection(items))
    _write_valid_manifest_entry(paths["cache_dir"], items[0])
    _write_missing_manifest_entry(paths["cache_dir"], items[1])
    _write_corrupt_manifest_entry(paths["cache_dir"], items[2])
    engine = FakeEngine()

    summary = bake_tts_cache.bake_cache(
        out_dir=paths["cache_dir"],
        engine=engine,
        checkpoint_every=1,
    )

    assert summary.skipped == 1
    assert summary.rendered == 2
    assert engine.calls == [("missing wav", "ko", 30), ("bad wav", "en", 30)]
    assert not list(paths["cache_dir"].glob("*.tmp"))


def test_bake_skips_synthesis_error_and_retries_on_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One synthesis error should be reported without aborting the bake run."""
    paths = _write_runtime_files(tmp_path, monkeypatch)
    items = (
        _item("ok text", "ko", "test:ok"),
        _item("bad text", "ko", "test:bad"),
        _item("after text", "en", "test:after"),
    )
    monkeypatch.setattr(bake_tts_cache, "collect_bake_texts", lambda repo_root: _collection(items))
    caplog.set_level(logging.INFO, logger=bake_tts_cache.__name__)
    engine = FailingEngine("bad text")

    summary = bake_tts_cache.bake_cache(
        out_dir=paths["cache_dir"],
        engine=engine,
        checkpoint_every=1,
    )

    assert summary.rendered == 2
    assert summary.skipped == 0
    assert summary.skipped_existing == 0
    assert summary.skipped_error == 1
    assert summary.error_sources == ("test:bad",)
    assert summary.skipped_errors == (
        bake_tts_cache.BakeSkippedError(
            source="test:bad",
            error="Found 2 unsupported character(s)",
        ),
    )
    assert engine.calls == [
        ("ok text", "ko", 30),
        ("bad text", "ko", 30),
        ("after text", "en", 30),
    ]

    manifest = _read_manifest(paths["cache_dir"])
    assert items[0].key in manifest
    assert items[1].key not in manifest
    assert items[2].key in manifest

    events = [
        json.loads(record.message)
        for record in caplog.records
        if record.name == bake_tts_cache.__name__
    ]
    skip_event = next(event for event in events if event["event"] == "cache_skip_error")
    assert skip_event["source"] == "test:bad"
    assert skip_event["error"] == "Found 2 unsupported character(s)"
    done_event = next(event for event in reversed(events) if event["event"] == "bake_done")
    assert done_event["rendered"] == 2
    assert done_event["skipped_existing"] == 0
    assert done_event["skipped_error"] == 1
    assert done_event["error_sources"] == ["test:bad"]

    caplog.clear()
    retry_engine = FakeEngine()
    retry_summary = bake_tts_cache.bake_cache(
        out_dir=paths["cache_dir"],
        engine=retry_engine,
        checkpoint_every=1,
    )

    assert retry_summary.rendered == 1
    assert retry_summary.skipped_existing == 2
    assert retry_summary.skipped_error == 0
    assert retry_engine.calls == [("bad text", "ko", 30)]
    retry_manifest = _read_manifest(paths["cache_dir"])
    assert {item.key for item in items}.issubset(retry_manifest)


def test_bake_writes_loader_readable_manifest_and_meta(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A freshly baked item is a valid hit through ``tts_cache.lookup``."""
    paths = _write_runtime_files(tmp_path, monkeypatch)
    item = _item("fresh baked", "en", "test:fresh")
    monkeypatch.setattr(
        bake_tts_cache, "collect_bake_texts", lambda repo_root: _collection((item,))
    )

    bake_tts_cache.bake_cache(
        out_dir=paths["cache_dir"],
        engine=FakeEngine(),
        checkpoint_every=1,
    )
    tts_cache._reset_for_tests()

    assert tts_cache.lookup("fresh baked", "en") == paths["cache_dir"] / f"{item.key}.wav"
    meta = json.loads((paths["cache_dir"] / tts_cache.META_FILENAME).read_text(encoding="utf-8"))
    assert meta == tts_cache._build_runtime_identity()
    manifest = json.loads(
        (paths["cache_dir"] / tts_cache.MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    entry = manifest[item.key]
    assert entry["wav"] == f"{item.key}.wav"
    assert entry["text"] == "fresh baked"
    assert entry["lang"] == "en"
    assert entry["status"] == "done"
    assert entry["sr"] == 16_000


def test_match_filters_to_substring_items(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--match`` bakes only items whose text contains a provided substring."""
    paths = _write_runtime_files(tmp_path, monkeypatch)
    items = (
        _item("주먹도끼 이야기", "ko", "test:match"),
        _item("other text", "ko", "test:other"),
        _item("another 도끼 line", "ko", "test:match2"),
    )
    monkeypatch.setattr(bake_tts_cache, "collect_bake_texts", lambda repo_root: _collection(items))
    engine = FakeEngine()

    summary = bake_tts_cache.bake_cache(
        out_dir=paths["cache_dir"],
        engine=engine,
        checkpoint_every=1,
        match=("도끼",),
    )

    assert summary.considered == 2
    assert summary.rendered == 2
    assert engine.calls == [
        ("주먹도끼 이야기", "ko", 30),
        ("another 도끼 line", "ko", 30),
    ]
    manifest = _read_manifest(paths["cache_dir"])
    assert items[0].key in manifest
    assert items[1].key not in manifest
    assert items[2].key in manifest


def test_match_accepts_multiple_substrings_with_or_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multiple ``--match`` values match an item if ANY substring is present."""
    paths = _write_runtime_files(tmp_path, monkeypatch)
    items = (
        _item("alpha line", "ko", "test:alpha"),
        _item("beta line", "ko", "test:beta"),
        _item("gamma line", "ko", "test:gamma"),
    )
    monkeypatch.setattr(bake_tts_cache, "collect_bake_texts", lambda repo_root: _collection(items))
    engine = FakeEngine()

    summary = bake_tts_cache.bake_cache(
        out_dir=paths["cache_dir"],
        engine=engine,
        checkpoint_every=1,
        match=("alpha", "gamma"),
    )

    assert summary.considered == 2
    assert summary.rendered == 2
    assert engine.calls == [
        ("alpha line", "ko", 30),
        ("gamma line", "ko", 30),
    ]


def test_force_rebakes_valid_hit_while_default_skips(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``--force`` re-synthesizes a valid hit; the default run skips it."""
    paths = _write_runtime_files(tmp_path, monkeypatch)
    item = _item("already baked", "ko", "test:hit")
    monkeypatch.setattr(
        bake_tts_cache, "collect_bake_texts", lambda repo_root: _collection((item,))
    )
    _write_valid_manifest_entry(paths["cache_dir"], item)

    default_engine = FakeEngine()
    default_summary = bake_tts_cache.bake_cache(
        out_dir=paths["cache_dir"],
        engine=default_engine,
        checkpoint_every=1,
    )

    assert default_summary.rendered == 0
    assert default_summary.skipped_existing == 1
    assert default_engine.calls == []

    caplog.set_level(logging.INFO, logger=bake_tts_cache.__name__)
    force_engine = FakeEngine()
    force_summary = bake_tts_cache.bake_cache(
        out_dir=paths["cache_dir"],
        engine=force_engine,
        checkpoint_every=1,
        force=True,
    )

    assert force_summary.rendered == 1
    assert force_summary.skipped_existing == 0
    assert force_engine.calls == [("already baked", "ko", 30)]

    events = [
        json.loads(record.message)
        for record in caplog.records
        if record.name == bake_tts_cache.__name__
    ]
    rebake_event = next(event for event in events if event["event"] == "cache_force_rebake")
    assert rebake_event["key"] == item.key
    assert rebake_event["source"] == "test:hit"

    manifest = _read_manifest(paths["cache_dir"])
    assert manifest[item.key]["text"] == "already baked"
    assert manifest[item.key]["status"] == "done"


def test_default_run_without_new_flags_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Omitting ``--match`` / ``--force`` preserves the skip-if-valid behavior."""
    paths = _write_runtime_files(tmp_path, monkeypatch)
    items = (
        _item("already baked", "ko", "test:skip"),
        _item("missing wav", "ko", "test:missing"),
    )
    monkeypatch.setattr(bake_tts_cache, "collect_bake_texts", lambda repo_root: _collection(items))
    _write_valid_manifest_entry(paths["cache_dir"], items[0])
    engine = FakeEngine()

    summary = bake_tts_cache.bake_cache(
        out_dir=paths["cache_dir"],
        engine=engine,
        checkpoint_every=1,
    )

    assert summary.considered == 2
    assert summary.rendered == 1
    assert summary.skipped_existing == 1
    assert engine.calls == [("missing wav", "ko", 30)]


def test_runner_script_contains_condition_interlock_and_restore_sequence() -> None:
    """The shell runner carries the zero-impact systemd interlock sequence."""
    script = Path("scripts/bake_tts_cache_run.sh").read_text(encoding="utf-8")

    assert "flock -n 9" in script
    assert "ConditionPathExists=!/run/mungi-tts-bake.lock" in script
    assert 'sudo -n touch "${INTERLOCK_LOCK}"' in script
    assert 'sudo -n rm -f "${INTERLOCK_LOCK}"' in script
    assert 'sudo -n rm -f "${DROPIN_PATH}"' in script
    assert 'sudo -n systemctl start "${SERVICE}"' in script
    assert "systemctl mask" not in script
    assert "systemctl enable" not in script
    assert "systemctl disable" not in script


def _item(text: str, lang: str, source: str) -> bake_tts_cache.BakeText:
    return bake_tts_cache.BakeText(
        text=text,
        lang=lang,
        key=tts_cache.compute_key(text, lang),
        source=source,
    )


def _collection(items: tuple[bake_tts_cache.BakeText, ...]) -> bake_tts_cache.CollectionResult:
    return bake_tts_cache.CollectionResult(
        items=items,
        inventory=bake_tts_cache.CollectionInventory(
            history_segments=0,
            history_lead_ins=0,
            history_consent=0,
            fe_ko=0,
            fe_en=0,
            approved_template_fixed=0,
            total_raw=len(items),
            total_unique=len(items),
        ),
    )


def _write_runtime_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Path]:
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


def _write_valid_manifest_entry(cache_dir: Path, item: bake_tts_cache.BakeText) -> None:
    wav_path = cache_dir / f"{item.key}.wav"
    _write_wav(wav_path)
    manifest = _read_manifest(cache_dir)
    manifest[item.key] = {
        "wav": wav_path.name,
        "text": item.text,
        "lang": item.lang,
        "steps": 30,
        "sr": 16_000,
        "speed": 0.95,
        "bytes": wav_path.stat().st_size,
        "sha256": _sha256_file(wav_path),
        "status": "done",
        "source": item.source,
    }
    _write_manifest(cache_dir, manifest)


def _write_missing_manifest_entry(cache_dir: Path, item: bake_tts_cache.BakeText) -> None:
    manifest = _read_manifest(cache_dir)
    manifest[item.key] = {
        "wav": f"{item.key}.wav",
        "text": item.text,
        "lang": item.lang,
        "bytes": 100,
        "sha256": "0" * 64,
        "status": "done",
    }
    _write_manifest(cache_dir, manifest)


def _write_corrupt_manifest_entry(cache_dir: Path, item: bake_tts_cache.BakeText) -> None:
    wav_path = cache_dir / f"{item.key}.wav"
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    wav_path.write_bytes(b"not a wav")
    manifest = _read_manifest(cache_dir)
    manifest[item.key] = {
        "wav": wav_path.name,
        "text": item.text,
        "lang": item.lang,
        "bytes": wav_path.stat().st_size,
        "sha256": _sha256_file(wav_path),
        "status": "done",
    }
    _write_manifest(cache_dir, manifest)


def _read_manifest(cache_dir: Path) -> dict[str, Any]:
    manifest_path = cache_dir / tts_cache.MANIFEST_FILENAME
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _write_manifest(cache_dir: Path, manifest: dict[str, Any]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / tts_cache.MANIFEST_FILENAME).write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )


def _write_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(b"\x00\x00" * 4)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
