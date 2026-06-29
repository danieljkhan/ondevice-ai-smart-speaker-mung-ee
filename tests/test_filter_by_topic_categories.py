"""Tests for the topic-category JSONL filter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts import filter_by_topic_categories as mod

CONFIG_PATH = Path("scripts") / "_phaseA_category_patterns.json"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to a JSONL file for tests."""

    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dictionaries."""

    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def build_row(row_id: str, title: str, text: str, language: str) -> dict[str, str]:
    """Build a minimal ingested JSONL row for tests."""

    return {
        "id": row_id,
        "title": title,
        "text": text,
        "language": language,
    }


def test_load_category_config_has_expected_17_categories() -> None:
    config = mod.load_category_config(CONFIG_PATH)

    assert config.category_order == mod.EXPECTED_CATEGORIES
    assert set(config.topic_rules) == {"en", "ko"}
    assert len(config.topic_rules["en"]) > 17
    assert len(config.topic_rules["ko"]) > 17
    assert config.violence_saturation_max_mentions == 3


def test_cli_filters_rows_and_writes_cache_outputs(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    write_jsonl(
        input_dir / "en_ingested.jsonl",
        [
            build_row("en-1", "Dog", "Dogs are common pets for children.", "en"),
        ],
    )
    write_jsonl(
        input_dir / "ko_ingested.jsonl",
        [
            build_row("ko-1", "지도와 대륙", "지도는 여러 대륙과 바다를 보여 준다.", "ko"),
            build_row(
                "ko-2",
                "조선",
                "조선은 1392년 건국 이후 학교와 과학 활동으로 이어졌다. 전쟁은 한 번만 언급된다.",
                "ko",
            ),
        ],
    )

    exit_code = mod.main(
        [
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--categories-config",
            str(CONFIG_PATH),
        ]
    )

    cache_dir = output_dir / "_cache"
    en_rows = read_jsonl(cache_dir / "en_filtered.jsonl")
    ko_rows = read_jsonl(cache_dir / "ko_filtered.jsonl")
    summary = json.loads((cache_dir / "filter_summary.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert en_rows == [
        {
            "id": "en-1",
            "title": "Dog",
            "text": "Dogs are common pets for children.",
            "category": "animal",
            "matched_topic": "dog",
            "language": "en",
        }
    ]
    assert ko_rows == [
        {
            "id": "ko-1",
            "title": "지도와 대륙",
            "text": "지도는 여러 대륙과 바다를 보여 준다.",
            "category": "world_geography",
            "matched_topic": "대륙",
            "language": "ko",
        },
        {
            "id": "ko-2",
            "title": "조선",
            "text": "조선은 1392년 건국 이후 학교와 과학 활동으로 이어졌다. 전쟁은 한 번만 언급된다.",
            "category": "world_history_light",
            "matched_topic": "조선",
            "language": "ko",
        },
    ]
    assert summary["kept_rows"] == 3
    assert summary["per_category_counts"]["animal"] == 1
    assert summary["per_category_counts"]["world_geography"] == 1
    assert summary["per_category_counts"]["world_history_light"] == 1


def test_filter_rejects_exclude_families_with_cross_language_terms(tmp_path: Path) -> None:
    input_dir = tmp_path / "_cache"
    output_dir = tmp_path / "_cache"
    input_dir.mkdir()
    config = mod.load_category_config(CONFIG_PATH)

    write_jsonl(
        input_dir / "en_ingested.jsonl",
        [
            build_row(
                "en-1",
                "Robot",
                "The article spends time on psychiatric trauma instead of child-friendly basics.",
                "en",
            ),
        ],
    )
    write_jsonl(
        input_dir / "ko_ingested.jsonl",
        [
            build_row(
                "ko-1",
                "강아지",
                "이 글은 어린이 설명이 아니라 담배와 광고를 다룬다.",
                "ko",
            ),
        ],
    )

    summary = mod.filter_topic_categories(input_dir=input_dir, output_dir=output_dir, config=config)

    assert read_jsonl(output_dir / "en_filtered.jsonl") == []
    assert read_jsonl(output_dir / "ko_filtered.jsonl") == []
    assert summary["per_exclude_family_rejections"]["medical_sensitive"] == 1
    assert summary["per_exclude_family_rejections"]["crime"] == 1


def test_world_history_light_requires_whitelist(tmp_path: Path) -> None:
    input_dir = tmp_path / "_cache"
    output_dir = tmp_path / "_cache"
    input_dir.mkdir()
    config = mod.load_category_config(CONFIG_PATH)

    write_jsonl(
        input_dir / "en_ingested.jsonl",
        [
            build_row(
                "en-1",
                "Joseon",
                "Joseon became known for culture and scholarship, but this row has no dated event cue.",
                "en",
            ),
        ],
    )
    write_jsonl(input_dir / "ko_ingested.jsonl", [])

    summary = mod.filter_topic_categories(input_dir=input_dir, output_dir=output_dir, config=config)

    assert read_jsonl(output_dir / "en_filtered.jsonl") == []
    assert summary["other_rejections"]["world_history_whitelist"] == 1
    assert summary["kept_rows"] == 0


def test_world_history_light_rejects_violence_saturation(tmp_path: Path) -> None:
    input_dir = tmp_path / "_cache"
    output_dir = tmp_path / "_cache"
    input_dir.mkdir()
    config = mod.load_category_config(CONFIG_PATH)

    write_jsonl(
        input_dir / "en_ingested.jsonl",
        [
            build_row(
                "en-1",
                "Imjin War",
                (
                    "In 1592 the event started and a teacher later described the war, battle, "
                    "battle, and war in detail."
                ),
                "en",
            ),
        ],
    )
    write_jsonl(input_dir / "ko_ingested.jsonl", [])

    summary = mod.filter_topic_categories(input_dir=input_dir, output_dir=output_dir, config=config)

    assert read_jsonl(output_dir / "en_filtered.jsonl") == []
    assert summary["per_exclude_family_rejections"]["violence"] == 1
    assert summary["kept_rows"] == 0
