"""Tests for the raw Wikipedia ingest utility."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts import ingest_raw_wikipedia as mod


def make_source_tree(tmp_path: Path, relative_dir: Path, shard_names: list[str]) -> list[Path]:
    """Create a fake raw-source directory tree with placeholder parquet shards."""

    data_dir = tmp_path / relative_dir / "data"
    data_dir.mkdir(parents=True)
    paths: list[Path] = []
    for shard_name in shard_names:
        shard_path = data_dir / shard_name
        shard_path.write_bytes(b"placeholder")
        paths.append(shard_path)
    return paths


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dictionaries."""

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def long_paragraph(seed: str, repeat: int = 8) -> str:
    """Build a paragraph comfortably above the minimum text threshold."""

    return " ".join([seed] * repeat)


def test_ingest_en_reconstructs_articles_and_applies_filters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir = tmp_path / "raw"
    output_dir = tmp_path / "cache"
    parquet_paths = make_source_tree(
        source_dir,
        mod.SOURCE_CONFIGS["en"].relative_dir,
        ["train-00000.parquet"],
    )

    first_body = long_paragraph(
        "April is the fourth month of the year and children often learn its weather and calendar facts."
    )
    second_body = long_paragraph(
        "It comes after March and before May so the article still needs another paragraph for the threshold."
    )
    overflow_paragraph = (
        "This paragraph intentionally has no final punctuation but it is much longer than eighty characters "
        "so the title detector must keep it inside the April article body"
    )
    too_short_body = "Tiny bee article."
    too_long_body = "x" * (mod.MAX_TEXT_LENGTH + 1)

    fake_rows: dict[Path, list[dict[str, Any]]] = {
        parquet_paths[0]: [
            {"text": "April"},
            {"text": first_body},
            {"text": overflow_paragraph},
            {"text": second_body},
            {"text": "Bee"},
            {"text": too_short_body},
            {"text": "Giant article"},
            {"text": too_long_body},
        ]
    }

    def fake_iter_parquet_rows(parquet_path: Path, columns: tuple[str, ...]) -> Any:
        assert columns == ("text",)
        return iter(fake_rows[parquet_path])

    monkeypatch.setattr(mod, "iter_parquet_rows", fake_iter_parquet_rows)

    stats = mod.ingest_source(
        source_dir=source_dir,
        output_dir=output_dir,
        config=mod.SOURCE_CONFIGS["en"],
        limit=None,
        dry_run=False,
        force=False,
    )

    cache_path = output_dir / "en_ingested.jsonl"
    metadata_path = output_dir / "en_ingested.meta.json"
    rows = read_jsonl(cache_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert stats.emitted == 1
    assert stats.skipped_too_short == 1
    assert stats.skipped_too_long == 1
    assert rows == [
        {
            "id": "en_sw_0",
            "title": "April",
            "text": "\n\n".join([first_body, overflow_paragraph, second_body]),
            "source": "rahular_simple_wikipedia",
            "language": "en",
        }
    ]
    assert metadata["ingest_version"] == mod.INGEST_VERSION
    assert metadata["source_hash"] == mod.compute_source_hash(parquet_paths)
    assert metadata["record_count"] == 1


def test_ingest_ko_direct_ingest_preserves_section_titles_and_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir = tmp_path / "raw"
    output_dir = tmp_path / "cache"
    parquet_paths = make_source_tree(
        source_dir,
        mod.SOURCE_CONFIGS["ko"].relative_dir,
        ["train-00000.parquet", "train-00001.parquet"],
    )

    first_text = long_paragraph(
        "수학은 숫자와 모양과 변화를 다루는 학문이며 어린이도 기본 개념을 차근차근 배울 수 있다.",
        repeat=10,
    )
    second_text = long_paragraph(
        "과학은 관찰과 질문을 통해 세상을 이해하려는 과정이며 다양한 실험으로 배움을 넓힌다.",
        repeat=10,
    )
    fake_rows: dict[Path, list[dict[str, Any]]] = {
        parquet_paths[0]: [
            {
                "title": "수학",
                "text": first_text,
                "section_titles": ["Introduction", "역사", "같이 보기"],
            }
        ],
        parquet_paths[1]: [
            {
                "title": "과학",
                "text": second_text,
                "section_titles": ["Introduction"],
            }
        ],
    }

    def fake_iter_parquet_rows(parquet_path: Path, columns: tuple[str, ...]) -> Any:
        assert columns == ("title", "text", "section_titles")
        return iter(fake_rows[parquet_path])

    monkeypatch.setattr(mod, "iter_parquet_rows", fake_iter_parquet_rows)

    stats = mod.ingest_source(
        source_dir=source_dir,
        output_dir=output_dir,
        config=mod.SOURCE_CONFIGS["ko"],
        limit=1,
        dry_run=False,
        force=False,
    )

    rows = read_jsonl(output_dir / "ko_ingested.jsonl")
    assert stats.emitted == 1
    assert rows == [
        {
            "id": "ko_w_0",
            "title": "수학",
            "text": first_text,
            "source": "lcw99_wikipedia_korean_20240501",
            "language": "ko",
            "section_titles": ["Introduction", "역사", "같이 보기"],
        }
    ]


def test_ingest_source_uses_fresh_cache_until_force(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir = tmp_path / "raw"
    output_dir = tmp_path / "cache"
    output_dir.mkdir(parents=True)
    parquet_paths = make_source_tree(
        source_dir,
        mod.SOURCE_CONFIGS["en"].relative_dir,
        ["train-00000.parquet"],
    )
    source_hash = mod.compute_source_hash(parquet_paths)
    cache_path = output_dir / "en_ingested.jsonl"
    metadata_path = output_dir / "en_ingested.meta.json"
    cache_path.write_text(
        json.dumps(
            {
                "id": "en_sw_0",
                "title": "Cached",
                "text": "cached text " * 30,
                "source": "rahular_simple_wikipedia",
                "language": "en",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    mod.write_cache_metadata(
        metadata_path,
        source_hash=source_hash,
        record_count=1,
        limit=None,
    )

    def fail_iter_parquet_rows(parquet_path: Path, columns: tuple[str, ...]) -> Any:
        raise AssertionError("fresh cache should bypass parquet ingestion")

    monkeypatch.setattr(mod, "iter_parquet_rows", fail_iter_parquet_rows)
    cached_stats = mod.ingest_source(
        source_dir=source_dir,
        output_dir=output_dir,
        config=mod.SOURCE_CONFIGS["en"],
        limit=None,
        dry_run=False,
        force=False,
    )

    assert cached_stats.used_cache is True
    assert cached_stats.emitted == 1

    forced_rows = {
        parquet_paths[0]: [
            {"text": "Planet"},
            {
                "text": long_paragraph(
                    "A planet moves around a star and children can compare different worlds in space.",
                    repeat=10,
                )
            },
            {
                "text": long_paragraph(
                    "This second paragraph keeps the reconstructed article above the minimum threshold.",
                    repeat=10,
                )
            },
        ]
    }

    def forced_iter_parquet_rows(parquet_path: Path, columns: tuple[str, ...]) -> Any:
        assert columns == ("text",)
        return iter(forced_rows[parquet_path])

    monkeypatch.setattr(mod, "iter_parquet_rows", forced_iter_parquet_rows)
    forced_stats = mod.ingest_source(
        source_dir=source_dir,
        output_dir=output_dir,
        config=mod.SOURCE_CONFIGS["en"],
        limit=None,
        dry_run=False,
        force=True,
    )

    rows = read_jsonl(cache_path)
    assert forced_stats.used_cache is False
    assert forced_stats.emitted == 1
    assert rows[0]["title"] == "Planet"
