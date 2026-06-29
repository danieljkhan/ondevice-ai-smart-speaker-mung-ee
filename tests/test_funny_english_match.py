"""Tests for Funny English coarse STT matching."""

from __future__ import annotations

from core.funny_english_match import (
    match_funny_english_attempt,
    normalize_funny_english_tokens,
    normalize_hotword_csv,
)


def test_match_accepts_exact_and_close_attempts() -> None:
    """Exact and near-token reads are accepted as participation passes."""
    exact = match_funny_english_attempt("I see a cat.", ["i", "see", "a", "cat"])
    typo = match_funny_english_attempt("kat", ["cat"])

    assert exact.band == "pass"
    assert typo.band == "pass"


def test_match_returns_close_low_and_silent_bands() -> None:
    """Non-pass attempts map to scaffold bands instead of a failure state."""
    close = match_funny_english_attempt(
        "I see",
        ["i", "see", "a", "cat"],
        pass_pct=0.6,
        pass_similarity=0.7,
    )
    low = match_funny_english_attempt("i x x", ["i", "see", "a", "cat"])
    silent = match_funny_english_attempt("", ["cat"])
    hangul = match_funny_english_attempt("고양이", ["cat"])

    assert close.band == "close"
    assert low.band == "low"
    assert silent.band == "silent_junk"
    assert hangul.band == "silent_junk"


def test_match_default_thresholds_are_baked_for_device_env() -> None:
    """Default FE scoring accepts the device-tuned 0.3 pct / 0.4 similarity bands."""
    similarity_pass = match_funny_english_attempt("see", ["i", "see", "a", "cat"])
    pct_pass = match_funny_english_attempt(
        "red blue sun",
        [
            "red",
            "blue",
            "sun",
            "elephant",
            "giraffe",
            "kangaroo",
            "umbrella",
            "spaceship",
            "mountain",
            "rainbow",
        ],
    )
    low = match_funny_english_attempt("i x x", ["i", "see", "a", "cat"])

    assert similarity_pass.matched_pct < 0.3
    assert similarity_pass.similarity >= 0.4
    assert similarity_pass.band == "pass"
    assert pct_pass.matched_pct == 0.3
    assert pct_pass.similarity < 0.4
    assert pct_pass.band == "pass"
    assert low.matched_pct < 0.3
    assert low.similarity < 0.4
    assert low.band == "low"


def test_normalization_keeps_letter_a_as_target_token() -> None:
    """The letter/article ``a`` remains available for alphabet cards."""
    assert normalize_funny_english_tokens("A!") == ("a",)


def test_match_accepts_letter_name_aliases_without_changing_regular_scoring() -> None:
    """Single-letter cards accept spoken letter-name aliases, including Hangul ASR."""
    hangul = match_funny_english_attempt("비", ("b",), accept_aliases=("b", "bee", "비"))
    english = match_funny_english_attempt("bee", ("b",), accept_aliases=("b", "bee", "비"))
    unrelated = match_funny_english_attempt("moon", ("b",), accept_aliases=("b", "bee", "비"))
    unrelated_hangul = match_funny_english_attempt(
        "가",
        ("b",),
        accept_aliases=("b", "bee", "비"),
    )
    wrong_letter = match_funny_english_attempt("c", ("b",), accept_aliases=("b", "bee", "비"))
    regular_low = match_funny_english_attempt("moon", ("cat",))

    assert hangul.band == "pass"
    assert hangul.matched_pct == 1.0
    assert english.band == "pass"
    assert unrelated.band != "pass"
    assert unrelated_hangul.band != "pass"
    assert wrong_letter.band != "pass"
    assert regular_low.band == "low"


def test_hotword_csv_appends_mungi_baseline_once() -> None:
    """Hotword normalization returns the exact CSV shape expected by Qwen3-ASR."""
    assert normalize_hotword_csv(("Cat", "cat", "뭉이")) == "cat,뭉이,뭉이야"
