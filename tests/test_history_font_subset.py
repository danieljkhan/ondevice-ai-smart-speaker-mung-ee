"""Coverage tests for the committed Korean-history font subset."""

from __future__ import annotations

import json
from pathlib import Path

from fontTools.ttLib import TTFont  # type: ignore[import-untyped]

from scripts.generate_history_font_subset import (
    DEFAULT_OUTPUT_FONT,
    collect_history_font_text,
    displayable_history_text,
)


def test_displayable_history_text_drops_non_renderable_characters() -> None:
    """The display filter removes Hanja, controls, and empty bracket pairs."""
    assert displayable_history_text("삼국유사(國家遺産廳)") == "삼국유사"
    assert displayable_history_text("왕의 이름[王石]\n다음") == "왕의 이름 다음"
    assert displayable_history_text("따옴표 “보존” · 점") == "따옴표 “보존” · 점"
    assert displayable_history_text("점수 ★★★★☆") == "점수 ★★★★☆"


def test_collect_history_font_text_includes_display_text_and_excludes_narration(
    tmp_path: Path,
) -> None:
    """The collector follows docs and gathers only rendered history text."""
    manifest_path = tmp_path / "assets" / "history" / "manifest.json"
    doc_path = tmp_path / "assets" / "history" / "docs" / "doc1.json"
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "title": "임시 문서(臨時)",
                "scenes": [
                    {
                        "seq": 1,
                        "section_title": "보이는 장 제목(王石)\n",
                        "narration": "수집되면 안 되는 숨은 내레이션(王石)",
                        "image_captions": ["장면 그림 설명(國家遺産廳)\t"],
                        "images": [
                            {"path": "unused.jpg", "caption": "첫 이미지 설명(王)"},
                            {"path": "unused2.jpg", "caption": "둘째 이미지 설명[石]"},
                        ],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "title": "재미있는 우리역사",
                "era_order": ["고조선"],
                "docs": [
                    {
                        "title": "목록 문서 제목(王)",
                        "doc_path": "assets/history/docs/doc1.json",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    collected = collect_history_font_text(manifest_path)

    assert "다음 이야기 들려줄까? 화면을 톡 누르면 들려줄게." in collected
    assert "지금부터 '' 이야기를 들려줄게." in collected
    assert "나가기" in collected
    assert "목록 문서 제목" in collected
    assert "보이는 장 제목" in collected
    assert "장면 그림 설명" in collected
    assert "첫 이미지 설명" in collected
    assert "둘째 이미지 설명" in collected
    assert "수집되면 안 되는 숨은 내레이션" not in collected
    assert "\n" not in collected
    assert "\t" not in collected
    assert "王" not in collected
    assert "石" not in collected
    assert "國" not in collected
    assert "()" not in collected
    assert "[]" not in collected


def test_history_font_subset_covers_all_rendered_history_text() -> None:
    """The committed subset must cover all rendered history glyphs."""
    required_codepoints = {ord(char) for char in collect_history_font_text()}
    font_path = Path(DEFAULT_OUTPUT_FONT)
    font = TTFont(str(font_path))
    covered_codepoints: set[int] = set()
    for table in font["cmap"].tables:
        covered_codepoints.update(table.cmap)

    missing = sorted(required_codepoints - covered_codepoints)

    assert not missing, "missing glyphs: " + ", ".join(
        f"U+{codepoint:04X} {chr(codepoint)!r}" for codepoint in missing
    )


def test_history_font_subset_covers_full_modern_hangul_syllables() -> None:
    """The committed subset must cover every modern Hangul syllable."""
    font_path = Path(DEFAULT_OUTPUT_FONT)
    font = TTFont(str(font_path))
    covered_codepoints: set[int] = set()
    for table in font["cmap"].tables:
        covered_codepoints.update(table.cmap)

    required_codepoints = set(range(0xAC00, 0xD7A4))
    missing = sorted(required_codepoints - covered_codepoints)

    assert not missing, f"missing {len(missing)} modern Hangul syllables"


def test_history_font_subset_covers_feedback_stars() -> None:
    """The committed subset must cover filled and empty feedback stars."""
    font_path = Path(DEFAULT_OUTPUT_FONT)
    font = TTFont(str(font_path))
    covered_codepoints: set[int] = set()
    for table in font["cmap"].tables:
        covered_codepoints.update(table.cmap)

    assert 0x2605 in covered_codepoints
    assert 0x2606 in covered_codepoints
