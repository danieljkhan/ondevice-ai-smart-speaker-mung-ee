"""Tests for heuristic age-band tagging."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import tag_age_bands as mod


def test_en_thresholds_classify_preschool() -> None:
    text = "Cat and dog play in the sun. " * 5

    assert mod.infer_age_band(text, "en") == "preschool"


def test_en_thresholds_classify_middle_school_on_rare_words() -> None:
    text = "Photosynthesis mitochondria chromosome biodiversity ecosystem. " * 40

    assert mod.infer_age_band(text, "en") == "middle_school"


def test_ko_thresholds_classify_preschool() -> None:
    text = "달이 떠요. 해가 떠요. 비가 와요. " * 10

    assert mod.infer_age_band(text, "ko") == "preschool"


def test_ko_thresholds_classify_middle_school_on_hanja_density() -> None:
    text = ("漢字 역사 설명 문장입니다. " * 120).strip()

    assert mod.infer_age_band(text, "ko") == "middle_school"


def test_overrides_take_precedence() -> None:
    record = {
        "id": "row-1",
        "title": "Title",
        "text": "simple text",
        "language": "en",
        "category": "general",
        "matched_topic": "topic",
    }

    tagged = mod.tag_record(record, {"row-1": "middle_school"})

    assert tagged["age_band_hint"] == "middle_school"


def test_process_source_writes_tagged_jsonl(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    (input_dir / "en_filtered.jsonl").write_text(
        json.dumps(
            {
                "id": "en-1",
                "title": "Moon",
                "text": "Moon is bright and round. " * 20,
                "category": "science",
                "matched_topic": "moon",
                "language": "en",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    output_path = mod.process_source(
        input_dir=input_dir,
        output_dir=output_dir,
        source="en",
        overrides={},
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["age_band_hint"] == "preschool"


def test_validate_age_band_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="Unsupported age band"):
        mod.validate_age_band("adult")
