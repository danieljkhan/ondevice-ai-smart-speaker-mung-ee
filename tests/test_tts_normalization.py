"""Tests for TTS text normalization."""

from __future__ import annotations


def test_normalize_tts_text_decodes_unicode_escape() -> None:
    from models.tts_runner import normalize_tts_text

    assert normalize_tts_text(r"\uc548\ub155\ud558\uc138\uc694") == "안녕하세요"


def test_normalize_tts_text_strips_think_tags() -> None:
    from models.tts_runner import normalize_tts_text

    assert normalize_tts_text("<think>reasoning</think>안녕하세요") == "안녕하세요"


def test_normalize_tts_text_repairs_utf8_mojibake() -> None:
    from models.tts_runner import normalize_tts_text

    assert normalize_tts_text("ì•ˆë…•í•˜ì„¸ìš”") == "안녕하세요"


def test_normalize_tts_text_expands_korean_adjacent_integers() -> None:
    from models.tts_runner import normalize_tts_text

    assert normalize_tts_text("1978년에 1,234개를 보았어.") == (
        "천구백칠십팔년에 천이백삼십사개를 보았어."
    )
    assert normalize_tts_text("제1장") == "제일장"


def test_normalize_tts_text_preserves_non_korean_numeric_contexts() -> None:
    from models.tts_runner import normalize_tts_text

    assert normalize_tts_text("version 1.2") == "version 1.2"
    assert normalize_tts_text("v1.2") == "v1.2"
    assert normalize_tts_text("AB123") == "AB123"
    assert normalize_tts_text("I have 2 cats") == "I have 2 cats"
    assert normalize_tts_text("잘못된 1,23개") == "잘못된 1,23개"
