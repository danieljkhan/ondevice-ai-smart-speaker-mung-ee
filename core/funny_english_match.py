"""Coarse word matching for the Funny English read-along mode."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Literal

FunnyEnglishBand = Literal["pass", "close", "low", "silent_junk"]

_TOKEN_RE = re.compile(r"[a-zA-Z']+")
_HANGUL_RE = re.compile(r"[\uac00-\ud7a3]")
_ALIAS_TOKEN_RE = re.compile(r"[a-zA-Z']+|[\uac00-\ud7a3]+")
DEFAULT_FUNNY_ENGLISH_PASS_PCT = 0.3
DEFAULT_FUNNY_ENGLISH_PASS_SIMILARITY = 0.4
_FILLER_WORDS = frozenset(
    {
        "ah",
        "eh",
        "er",
        "hm",
        "hmm",
        "mmm",
        "oh",
        "okay",
        "ok",
        "um",
        "uh",
    }
)


@dataclass(frozen=True)
class FunnyEnglishMatchResult:
    """Result of comparing one child utterance to one known target card."""

    transcript: str
    normalized_transcript_tokens: tuple[str, ...]
    target_tokens: tuple[str, ...]
    matched_tokens: tuple[str, ...]
    missed_tokens: tuple[str, ...]
    matched_pct: float
    similarity: float
    band: FunnyEnglishBand


def normalize_funny_english_tokens(text: str) -> tuple[str, ...]:
    """Return lower-cased English tokens with filler words removed."""
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(text.casefold()):
        token = match.group(0).strip("'")
        if not token or token in _FILLER_WORDS:
            continue
        tokens.append(token)
    return tuple(tokens)


def normalize_hotword_csv(tokens: list[str] | tuple[str, ...]) -> str:
    """Return the exact CSV string passed to Qwen3-ASR hotword loading."""
    ordered: list[str] = []
    seen: set[str] = set()
    for token in (*tokens, "뭉이", "뭉이야"):
        clean = token.strip().casefold()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        ordered.append(clean)
    return ",".join(ordered)


def match_funny_english_attempt(
    transcript: str,
    target_tokens: list[str] | tuple[str, ...],
    *,
    accept_aliases: list[str] | tuple[str, ...] = (),
    pass_pct: float = DEFAULT_FUNNY_ENGLISH_PASS_PCT,
    pass_similarity: float = DEFAULT_FUNNY_ENGLISH_PASS_SIMILARITY,
    close_pct: float = 0.3,
) -> FunnyEnglishMatchResult:
    """Compare an STT transcript to a card target using lenient word matching."""
    normalized_target = tuple(
        token
        for token in normalize_funny_english_tokens(" ".join(target_tokens))
        if token not in _FILLER_WORDS
    )
    normalized_transcript = normalize_funny_english_tokens(transcript)
    if not normalized_target:
        msg = "target_tokens must contain at least one English token"
        raise ValueError(msg)
    alias_match = _find_matching_alias(transcript, accept_aliases)
    if alias_match is not None:
        return FunnyEnglishMatchResult(
            transcript=transcript,
            normalized_transcript_tokens=normalized_transcript,
            target_tokens=normalized_target,
            matched_tokens=normalized_target,
            missed_tokens=(),
            matched_pct=1.0,
            similarity=1.0,
            band="pass",
        )
    if _is_silent_or_junk(transcript, normalized_transcript):
        return FunnyEnglishMatchResult(
            transcript=transcript,
            normalized_transcript_tokens=normalized_transcript,
            target_tokens=normalized_target,
            matched_tokens=(),
            missed_tokens=normalized_target,
            matched_pct=0.0,
            similarity=0.0,
            band="silent_junk",
        )

    unmatched_transcript = list(normalized_transcript)
    matched: list[str] = []
    missed: list[str] = []
    for target in normalized_target:
        index = _find_matching_token(target, unmatched_transcript)
        if index is None:
            missed.append(target)
            continue
        matched.append(target)
        unmatched_transcript.pop(index)

    matched_pct = len(matched) / len(normalized_target)
    similarity = SequenceMatcher(
        None,
        " ".join(normalized_transcript),
        " ".join(normalized_target),
    ).ratio()
    if matched_pct >= pass_pct or similarity >= pass_similarity:
        band: FunnyEnglishBand = "pass"
    elif matched_pct >= close_pct:
        band = "close"
    else:
        band = "low"
    return FunnyEnglishMatchResult(
        transcript=transcript,
        normalized_transcript_tokens=normalized_transcript,
        target_tokens=normalized_target,
        matched_tokens=tuple(matched),
        missed_tokens=tuple(missed),
        matched_pct=matched_pct,
        similarity=similarity,
        band=band,
    )


def _find_matching_alias(
    transcript: str,
    accept_aliases: list[str] | tuple[str, ...],
) -> str | None:
    """Return the accepted alias that matches a lightly normalized transcript."""
    if not accept_aliases:
        return None
    candidates = list(_normalize_alias_tokens(transcript))
    if not candidates:
        return None
    for alias in accept_aliases:
        alias_tokens = _normalize_alias_tokens(alias)
        if not alias_tokens:
            continue
        if _alias_tokens_match(alias_tokens, candidates):
            return " ".join(alias_tokens)
    return None


def _normalize_alias_tokens(text: str) -> tuple[str, ...]:
    """Return lower-cased English/Hangul tokens for letter-name alias matching."""
    return tuple(match.group(0).strip("'") for match in _ALIAS_TOKEN_RE.finditer(text.casefold()))


def _alias_tokens_match(alias_tokens: tuple[str, ...], candidates: list[str]) -> bool:
    """Return whether all alias tokens appear in order in transcript candidates."""
    start = 0
    for alias_token in alias_tokens:
        index = _find_matching_alias_token(alias_token, candidates[start:])
        if index is None:
            return False
        start += index + 1
    return True


def _find_matching_alias_token(target: str, candidates: list[str]) -> int | None:
    """Return a matching alias candidate index, using fuzzy match for English aliases."""
    if target.isascii():
        for index, candidate in enumerate(candidates):
            if not candidate.isascii():
                continue
            if candidate == target:
                return index
            if len(target) > 1 and _is_close_token(candidate, target):
                return index
        return None
    for index, candidate in enumerate(candidates):
        if candidate == target:
            return index
    return None


def _find_matching_token(target: str, candidates: list[str]) -> int | None:
    """Return the first candidate index close enough to the target token."""
    for index, candidate in enumerate(candidates):
        if candidate == target:
            return index
        if _is_close_token(candidate, target):
            return index
    return None


def _is_close_token(candidate: str, target: str) -> bool:
    """Return whether two tokens are within the existing fuzzy-match threshold.

    Very short targets (<=2 chars, e.g. single alphabet letters) require an exact
    match: a 1-edit threshold would make every single letter interchangeable
    (``c`` would match ``b``). Letter-name pronunciations are handled separately
    via the accept-aliases path.
    """
    if len(target) <= 2:
        return False
    return _levenshtein_distance(candidate, target) <= max(1, math.ceil(len(target) / 4))


def _is_silent_or_junk(text: str, tokens: tuple[str, ...]) -> bool:
    stripped = text.strip()
    if not stripped or not tokens:
        return True
    hangul_chars = len(_HANGUL_RE.findall(stripped))
    if hangul_chars == 0:
        return False
    alpha_chars = sum(1 for char in stripped if char.isalpha())
    return alpha_chars > 0 and hangul_chars / alpha_chars >= 0.5


def _levenshtein_distance(left: str, right: str) -> int:
    """Return the Levenshtein edit distance between two short tokens."""
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            substitution_cost = 0 if left_char == right_char else 1
            current.append(
                min(
                    current[right_index - 1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1]


__all__ = [
    "FunnyEnglishBand",
    "FunnyEnglishMatchResult",
    "match_funny_english_attempt",
    "normalize_funny_english_tokens",
    "normalize_hotword_csv",
]
