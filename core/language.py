"""Language detection helpers for bilingual input telemetry."""

from __future__ import annotations


def detect_language(text: str) -> str:
    """Detect input language based on Korean character presence.

    Korean Unicode ranges:
    - Hangul Syllables: U+AC00 - U+D7AF

    Returns ``"ko"`` if any Korean characters are found, ``"en"`` otherwise.
    """
    for char in text:
        if "\uac00" <= char <= "\ud7af":
            return "ko"
    return "en"
