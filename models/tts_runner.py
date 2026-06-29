"""TTS engine abstractions and implementations.

Provides the abstract TTSEngine base class and SupertonicEngine.
Extracted from ``scripts/test_tts.py`` to establish correct dependency
direction (``core/`` -> ``models/``).
"""

from __future__ import annotations

import abc
import codecs
import concurrent.futures
import logging
import os
import queue
import re
import struct
import tempfile
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger("mungi.models.tts_runner")

# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------

DEFAULT_SAMPLE_RATE: int = 22050
_WHITESPACE_RE: re.Pattern[str] = re.compile(r"\s+")
_UNICODE_ESCAPE_RE: re.Pattern[str] = re.compile(
    r"(?:\\u[0-9a-fA-F]{4}|\\U[0-9a-fA-F]{8}|\\x[0-9a-fA-F]{2})"
)
_MOJIBAKE_HINT_RE: re.Pattern[str] = re.compile(r"[À-ÿ]")
_HANGUL_RE: re.Pattern[str] = re.compile(r"[\uac00-\ud7a3]")
# Marker identifying the Supertonic "unsupported character(s)" pipeline failure.
_UNSUPPORTED_CHAR_ERROR_MARKER = "unsupported character"
# Captures each single-quoted offending character inside the Supertonic error,
# e.g. "...unsupported character(s): ['\u53e4', '\u00b7']".
_UNSUPPORTED_CHAR_ERROR_RE: re.Pattern[str] = re.compile(r"'(.)'")
_NUMBER_TOKEN_RE: re.Pattern[str] = re.compile(r"\d[\d,]*(?:[년월일개번살])?")
_VALID_COMMA_INTEGER_RE: re.Pattern[str] = re.compile(r"\d{1,3}(?:,\d{3})+")
_CJK_EXT_A_RANGE = (0x3400, 0x4DBF)
_CJK_UNIFIED_RANGE = (0x4E00, 0x9FFF)
_CJK_COMPATIBILITY_RANGE = (0xF900, 0xFAFF)
_CJK_IDEOGRAPH_RANGES = (
    _CJK_EXT_A_RANGE,
    _CJK_UNIFIED_RANGE,
    _CJK_COMPATIBILITY_RANGE,
)
_HIRAGANA_RANGE = (0x3040, 0x309F)
_KATAKANA_RANGE = (0x30A0, 0x30FF)
_KATAKANA_PHONETIC_EXTENSIONS_RANGE = (0x31F0, 0x31FF)
_HALFWIDTH_KATAKANA_RANGE = (0xFF65, 0xFF9F)
_PRIVATE_USE_AREA_RANGE = (0xE000, 0xF8FF)
_SUPPLEMENTARY_PRIVATE_USE_AREA_A_RANGE = (0xF0000, 0xFFFFD)
_SUPPLEMENTARY_PRIVATE_USE_AREA_B_RANGE = (0x100000, 0x10FFFD)
_UNSUPPORTED_TTS_STRIP_RANGES = (
    _HIRAGANA_RANGE,
    _KATAKANA_RANGE,
    _KATAKANA_PHONETIC_EXTENSIONS_RANGE,
    _HALFWIDTH_KATAKANA_RANGE,
    _PRIVATE_USE_AREA_RANGE,
    _SUPPLEMENTARY_PRIVATE_USE_AREA_A_RANGE,
    _SUPPLEMENTARY_PRIVATE_USE_AREA_B_RANGE,
)
_UNSUPPORTED_TTS_SEPARATOR_CHARS = frozenset(
    ("\u00b7", "\u318d", "\u2022", "\u30fb", "\u223c", "\u25b2")
)
_UNSUPPORTED_TTS_REPLACEMENTS = {
    "\u2103": "\ub3c4",  # \u2103 -> \ub3c4 (Korean degree reading)
    "\uff0c": ",",  # fullwidth comma -> ASCII comma (Supertonic-safe)
    "\uff0e": ".",  # fullwidth full stop -> ASCII period
    "\u3002": ".",  # ideographic full stop -> ASCII period
    "\uff1b": ";",  # fullwidth semicolon -> ASCII semicolon
    "\uff1a": ":",  # fullwidth colon -> ASCII colon
    "\uff01": "!",  # fullwidth exclamation -> ASCII exclamation
    "\uff1f": "?",  # fullwidth question mark -> ASCII question mark
}
_UNSUPPORTED_TTS_STRIP_CHARS = frozenset(("\u25cb", "\u25a1"))
_CJK_BRACKET_PAIRS: tuple[tuple[str, str], ...] = (
    ("(", ")"),
    ("\uff08", "\uff09"),
    ("\u3014", "\u3015"),
    ("\u3008", "\u3009"),
    ("\u300a", "\u300b"),
    ("\u300c", "\u300d"),
    ("\u300e", "\u300f"),
    ("\uff3b", "\uff3d"),
    ("[", "]"),
)
_CJK_BRACKET_CLOSE_BY_OPEN = dict(_CJK_BRACKET_PAIRS)
_ASCII_IDENTIFIER_NEIGHBORS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-/"
)
_KO_COUNTERS = frozenset("년월일개번살")
_EMERGENCY_NUMBER_READING: dict[str, str] = {"112": "일일이", "119": "일일구"}
_EMERGENCY_NUMBER_TRAILING_PUNCTUATION = frozenset(".!?。！？")
_SINO_DIGITS = ("", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구")
_SINO_SMALL_UNITS = ("", "십", "백", "천")
_SINO_LARGE_UNITS = ("", "만", "억", "조", "경")
_NATIVE_COUNTERS = frozenset("개번살")
_NATIVE_COUNTER_UNITS = {
    1: "한",
    2: "두",
    3: "세",
    4: "네",
    5: "다섯",
    6: "여섯",
    7: "일곱",
    8: "여덟",
    9: "아홉",
}
_NATIVE_COUNTER_TENS = {
    10: "열",
    20: "스물",
    30: "서른",
    40: "마흔",
    50: "쉰",
    60: "예순",
    70: "일흔",
    80: "여든",
    90: "아흔",
}
_KO_SENTENCE_PUNCTUATION = frozenset(".!?\u3002\uff01\uff1f\u2026")
_EN_SENTENCE_PUNCTUATION = frozenset(".!?")
_SENTENCE_PUNCTUATION = _KO_SENTENCE_PUNCTUATION | _EN_SENTENCE_PUNCTUATION
_SENTENCE_CLOSER_CHARS = "\u201d\u2019\"')\uff09\u300d\u300f]\u203a\u00bb"
_SENTENCE_CLOSERS = frozenset(_SENTENCE_CLOSER_CHARS)
_SENTENCE_OPENERS = frozenset('\u201c\u2018"')
_COMMON_ABBREVIATIONS: tuple[str, ...] = (
    "mr.",
    "mrs.",
    "dr.",
    "ms.",
    "vs.",
    "e.g.",
    "i.e.",
    "etc.",
    "jr.",
    "sr.",
    "st.",
)
_MIN_SENTENCE_LENGTH = 8
_MIN_FIRST_CHUNK_MS = 200.0
_SHORT_FIRST_CHUNK_PADDING_MS = 100.0
_JETSON_DEFAULT_SUPERTONIC_MODEL_DIR = "/opt/mungi/ai_models/supertonic-2"


def _maybe_decode_unicode_escape(text: str) -> str:
    """Decode shell-safe unicode escapes when they are present."""
    if not _UNICODE_ESCAPE_RE.search(text):
        return text
    try:
        return codecs.decode(text, "unicode_escape")
    except UnicodeDecodeError:
        return text


def _maybe_fix_utf8_mojibake(text: str) -> str:
    """Repair common UTF-8-as-Latin1 mojibake used in terminal hops."""
    if not _MOJIBAKE_HINT_RE.search(text):
        return text

    for source_encoding in ("latin-1", "cp1252"):
        try:
            repaired = text.encode(source_encoding).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
        if any("\uac00" <= char <= "\ud7a3" for char in repaired):
            return repaired
    return text


def _is_hangul_char(char: str | None) -> bool:
    return bool(char and "\uac00" <= char <= "\ud7a3")


def _is_valid_integer_token(number: str) -> bool:
    if "," in number:
        if _VALID_COMMA_INTEGER_RE.fullmatch(number) is None:
            return False
        first_group = number.split(",", 1)[0]
        return len(first_group) == 1 or not first_group.startswith("0")
    if not number.isdigit():
        return False
    return len(number) == 1 or not number.startswith("0")


def _four_digit_to_sino_korean(value: int) -> str:
    parts: list[str] = []
    divisor = 1000
    for unit_index in range(3, -1, -1):
        digit = value // divisor
        value %= divisor
        divisor //= 10
        if digit == 0:
            continue
        unit = _SINO_SMALL_UNITS[unit_index]
        if digit == 1 and unit:
            parts.append(unit)
        else:
            parts.append(f"{_SINO_DIGITS[digit]}{unit}")
    return "".join(parts)


def _integer_to_sino_korean(number: str) -> str:
    value = int(number.replace(",", ""))
    if value == 0:
        return "영"
    groups: list[int] = []
    while value > 0:
        groups.append(value % 10000)
        value //= 10000
    parts: list[str] = []
    for index in range(len(groups) - 1, -1, -1):
        group = groups[index]
        if group == 0:
            continue
        text = _four_digit_to_sino_korean(group)
        unit = _SINO_LARGE_UNITS[index] if index < len(_SINO_LARGE_UNITS) else ""
        if unit and text == "일":
            text = ""
        parts.append(f"{text}{unit}")
    return "".join(parts)


def _integer_to_native_counter_korean(number: str, counter: str) -> str | None:
    if counter not in _NATIVE_COUNTERS or "," in number:
        return None

    value = int(number)
    if value <= 0 or value >= 100:
        return None

    if value <= 9:
        reading = _NATIVE_COUNTER_UNITS[value]
    elif value == 20:
        reading = "스무"
    else:
        tens_value = (value // 10) * 10
        unit_value = value % 10
        reading = _NATIVE_COUNTER_TENS[tens_value]
        if unit_value:
            reading = f"{reading}{_NATIVE_COUNTER_UNITS[unit_value]}"
    return f"{reading} {counter}"


def _should_expand_ko_number(
    text: str,
    start: int,
    end: int,
    counter: str,
    *,
    allow_bare: bool = False,
) -> bool:
    left = text[start - 1] if start > 0 else None
    right = text[end] if end < len(text) else None
    right_is_bare_sentence_punctuation = (
        allow_bare
        and right in _EMERGENCY_NUMBER_TRAILING_PUNCTUATION
        and _has_only_emergency_number_tail(text[end:])
    )
    if left in _ASCII_IDENTIFIER_NEIGHBORS or (
        right in _ASCII_IDENTIFIER_NEIGHBORS and not right_is_bare_sentence_punctuation
    ):
        return False
    return bool(
        counter
        or _is_hangul_char(left)
        or _is_hangul_char(right)
        or (allow_bare and not text[:start].strip() and _has_only_emergency_number_tail(text[end:]))
    )


def _has_only_emergency_number_tail(text: str) -> bool:
    stripped = text.strip()
    return not stripped or all(char in _EMERGENCY_NUMBER_TRAILING_PUNCTUATION for char in stripped)


def _expand_ko_number_tokens(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        counter = token[-1] if token[-1] in _KO_COUNTERS else ""
        number = token[:-1] if counter else token
        if not _is_valid_integer_token(number):
            return token
        if (
            number in _EMERGENCY_NUMBER_READING
            and counter == "번"
            and match.end() < len(text)
            and _is_hangul_char(text[match.end()])
        ):
            return token
        emergency_reading = _EMERGENCY_NUMBER_READING.get(number) if not counter else None
        if not _should_expand_ko_number(
            text,
            match.start(),
            match.end(),
            counter,
            allow_bare=emergency_reading is not None,
        ):
            return token
        if emergency_reading is not None:
            return emergency_reading
        native_counter_reading = _integer_to_native_counter_korean(number, counter)
        if native_counter_reading is not None:
            return native_counter_reading
        return f"{_integer_to_sino_korean(number)}{counter}"

    return _NUMBER_TOKEN_RE.sub(replace, text)


def _is_cjk_ideograph(char: str) -> bool:
    codepoint = ord(char)
    return any(start <= codepoint <= end for start, end in _CJK_IDEOGRAPH_RANGES)


def _contains_cjk_ideograph(text: str) -> bool:
    return any(_is_cjk_ideograph(char) for char in text)


def _strip_cjk_ideographs(text: str) -> str:
    result: list[str] = []
    in_cjk_run = False
    for char in text:
        if _is_cjk_ideograph(char):
            if not in_cjk_run:
                result.append(" ")
                in_cjk_run = True
            continue
        result.append(char)
        in_cjk_run = False
    return "".join(result)


def _is_unsupported_tts_strip_char(char: str) -> bool:
    codepoint = ord(char)
    return char in _UNSUPPORTED_TTS_STRIP_CHARS or any(
        start <= codepoint <= end for start, end in _UNSUPPORTED_TTS_STRIP_RANGES
    )


def _replace_unsupported_tts_chars(text: str) -> str:
    result: list[str] = []
    in_strip_run = False
    for char in text:
        if char in _UNSUPPORTED_TTS_SEPARATOR_CHARS:
            result.append(" ")
            in_strip_run = False
            continue
        replacement = _UNSUPPORTED_TTS_REPLACEMENTS.get(char)
        if replacement is not None:
            result.append(replacement)
            in_strip_run = False
            continue
        if _is_unsupported_tts_strip_char(char):
            if not in_strip_run:
                result.append(" ")
                in_strip_run = True
            continue
        result.append(char)
        in_strip_run = False
    return "".join(result)


def _is_unsupported_char_error(exc: BaseException) -> bool:
    """Return True when an exception is a Supertonic unsupported-character failure."""
    return _UNSUPPORTED_CHAR_ERROR_MARKER in str(exc).lower()


def _extract_unsupported_chars(message: str) -> tuple[str, ...]:
    """Parse the offending characters out of a Supertonic unsupported-char message.

    The Supertonic pipeline raises messages such as
    ``Found 1 unsupported character(s): ['古']``. This extracts the quoted
    characters so the caller can strip exactly those and retry.
    """
    matches = _UNSUPPORTED_CHAR_ERROR_RE.findall(message)
    seen: set[str] = set()
    ordered: list[str] = []
    for char in matches:
        if char and char not in seen:
            seen.add(char)
            ordered.append(char)
    return tuple(ordered)


def _strip_chars(text: str, chars: tuple[str, ...]) -> str:
    """Replace each character in ``chars`` with a space and collapse whitespace."""
    if not chars:
        return text
    strip_set = set(chars)
    result = "".join(" " if char in strip_set else char for char in text)
    return _WHITESPACE_RE.sub(" ", result).strip()


def _remove_bracketed_cjk_groups(text: str) -> str:
    result: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        close_char = _CJK_BRACKET_CLOSE_BY_OPEN.get(char)
        if close_char is None:
            result.append(char)
            index += 1
            continue

        close_index = text.find(close_char, index + 1)
        if close_index < 0:
            result.append(char)
            index += 1
            continue

        group = text[index : close_index + 1]
        inner = text[index + 1 : close_index]
        if _contains_cjk_ideograph(inner):
            index = close_index + 1
            continue

        result.append(group)
        index = close_index + 1
    return "".join(result)


def _remove_empty_bracket_groups(text: str) -> str:
    result: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        close_char = _CJK_BRACKET_CLOSE_BY_OPEN.get(char)
        if close_char is None:
            result.append(char)
            index += 1
            continue

        close_index = text.find(close_char, index + 1)
        if close_index < 0:
            result.append(char)
            index += 1
            continue

        inner = text[index + 1 : close_index]
        if not inner.strip():
            index = close_index + 1
            continue

        result.append(text[index : close_index + 1])
        index = close_index + 1
    return "".join(result)


def _normalize_cjk(text: str) -> str:
    without_bracketed_groups = _remove_bracketed_cjk_groups(text)
    return _strip_cjk_ideographs(without_bracketed_groups)


def normalize_tts_text(text: str | None) -> str:
    """Normalize text before passing it to a TTS engine."""
    if text is None:
        return ""

    from models.llm_runner import strip_think_tags

    cleaned = _maybe_decode_unicode_escape(text)
    cleaned = _maybe_fix_utf8_mojibake(cleaned)
    cleaned = strip_think_tags(cleaned)
    cleaned = _normalize_cjk(cleaned)
    cleaned = _replace_unsupported_tts_chars(cleaned)
    cleaned = _remove_empty_bracket_groups(cleaned)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    cleaned = _expand_ko_number_tokens(cleaned)
    # Phonetic substitution: English TTS pronounces "Mung-i" as "mung-eye".
    # "Moong-ee" produces the closest pronunciation to Korean 뭉이.
    cleaned = re.sub(r"(?i)mung[\-\s]?i\b", "Moong-ee", cleaned)
    return cleaned


_ACTIVE_SUPERTONIC_ENGINE: SupertonicEngine | None = None
_ACTIVE_SUPERTONIC_ENGINE_LOCK = threading.Lock()
_STREAMING_ENGINE_CACHE: dict[tuple[str, str], SupertonicEngine] = {}


@dataclass(frozen=True)
class SentenceSynthesisResult:
    """Metrics and optional artifact path for sentence-level TTS playback."""

    total_duration_ms: float
    first_chunk_ms: float
    sentence_count: int
    full_wav_path: str | None = None


def _split_text_into_sentences(text: str | None) -> list[str]:
    """Split normalized text into ordered sentence chunks for TTS."""
    normalized = normalize_tts_text(text)
    if not normalized:
        return []

    punctuation = (
        _KO_SENTENCE_PUNCTUATION if _HANGUL_RE.search(normalized) else _EN_SENTENCE_PUNCTUATION
    )
    sentences: list[str] = []
    start = 0
    index = 0
    while index < len(normalized):
        if normalized[index] not in punctuation:
            index += 1
            continue

        end = index
        while end + 1 < len(normalized) and normalized[end + 1] in punctuation:
            end += 1

        immediate_next = normalized[end + 1] if end + 1 < len(normalized) else ""
        if normalized[index] == "." and end == index and immediate_next.isdigit():
            index = end + 1
            continue

        closer_end = end
        while closer_end + 1 < len(normalized) and normalized[closer_end + 1] in _SENTENCE_CLOSERS:
            closer_end += 1

        next_char = normalized[closer_end + 1] if closer_end + 1 < len(normalized) else ""
        boundary_end = closer_end
        if next_char and not next_char.isspace():
            if immediate_next not in _SENTENCE_OPENERS:
                index = end + 1
                continue
            boundary_end = end

        candidate = normalized[start : boundary_end + 1].strip()
        abbreviation_candidate = candidate.rstrip(_SENTENCE_CLOSER_CHARS)
        lowered = abbreviation_candidate.casefold()
        if candidate and not any(lowered.endswith(abbrev) for abbrev in _COMMON_ABBREVIATIONS):
            sentences.append(candidate)
            start = boundary_end + 1
        index = end + 1
        if boundary_end > end:
            index = boundary_end + 1

    tail = normalized[start:].strip()
    if tail:
        sentences.append(tail)
    return _merge_short_sentences(sentences)


def _merge_short_sentences(sentences: list[str]) -> list[str]:
    """Merge undersized sentence chunks into adjacent chunks."""
    merged: list[str] = []
    pending_prefix = ""

    for sentence in sentences:
        current = sentence.strip()
        if not current:
            continue

        if pending_prefix:
            current = f"{pending_prefix} {current}".strip()
            pending_prefix = ""

        if len(current) < _MIN_SENTENCE_LENGTH:
            if merged:
                previous_core = merged[-1].rstrip().rstrip(_SENTENCE_CLOSER_CHARS)
                current_core = current.rstrip().rstrip(_SENTENCE_CLOSER_CHARS)
                if (
                    previous_core
                    and current_core
                    and previous_core[-1] in _SENTENCE_PUNCTUATION
                    and current_core[-1] in _SENTENCE_PUNCTUATION
                ):
                    merged.append(current)
                    continue
                merged[-1] = f"{merged[-1]} {current}".strip()
            else:
                pending_prefix = current
            continue

        merged.append(current)

    if pending_prefix:
        if merged:
            merged[-1] = f"{merged[-1]} {pending_prefix}".strip()
        else:
            merged.append(pending_prefix)
    return [sentence for sentence in merged if sentence]


def _flatten_audio_samples(audio: Any) -> np.ndarray:
    """Return audio as a flattened float32 array."""
    import numpy as np

    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim > 1:
        samples = samples.reshape(-1)
    return samples


def _audio_duration_ms(audio: Any, sample_rate: int) -> float:
    """Compute audio duration in milliseconds for one chunk."""
    samples = _flatten_audio_samples(audio)
    if sample_rate <= 0 or samples.size == 0:
        return 0.0
    return (samples.size / float(sample_rate)) * 1000.0


def _prepend_silence(audio: Any, sample_rate: int, duration_ms: float) -> np.ndarray:
    """Prepend silence padding to a chunk to reduce short-turn clicks."""
    import numpy as np

    samples = _flatten_audio_samples(audio)
    if sample_rate <= 0 or duration_ms <= 0.0:
        return samples
    silence_samples = max(int(round(sample_rate * (duration_ms / 1000.0))), 1)
    silence = np.zeros(silence_samples, dtype=np.float32)
    return np.concatenate((silence, samples))


def _write_temp_wav(audio: Any, sample_rate: int) -> str | None:
    """Persist synthesized audio to a temporary WAV file."""
    samples = _flatten_audio_samples(audio)
    if sample_rate <= 0 or samples.size == 0:
        return None

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        temp_path = Path(handle.name)

    pcm_16 = (_flatten_audio_samples(samples).clip(-1.0, 1.0) * 32767).astype("int16")
    raw_data = struct.pack(f"<{len(pcm_16)}h", *pcm_16)
    with wave.open(str(temp_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(raw_data)
    return str(temp_path)


def _play_sentence_chunk(audio: Any, sample_rate: int, output_device: str | None) -> None:
    """Play one synthesized sentence chunk via the sounddevice-backed player."""
    from hardware.audio_player import play_audio

    play_audio(audio, sample_rate, device=output_device)


def _streaming_env_overrides_configured() -> bool:
    """Return True when streaming model resolution has an explicit env override."""
    return bool(
        os.getenv("MUNGI_TTS_MODEL_DIR", "").strip() or os.getenv("MUNGI_MODEL_DIR", "").strip()
    )


def _resolve_streaming_model_dir() -> str:
    """Resolve the streaming model directory from env overrides or Jetson defaults."""
    explicit_model_dir = os.getenv("MUNGI_TTS_MODEL_DIR", "").strip()
    if explicit_model_dir:
        return explicit_model_dir

    model_root = os.getenv("MUNGI_MODEL_DIR", "").strip()
    if model_root:
        return str(Path(model_root) / "supertonic-2")

    logger.warning(
        "MUNGI_MODEL_DIR not set; using Jetson runtime default %s",
        _JETSON_DEFAULT_SUPERTONIC_MODEL_DIR,
    )
    return _JETSON_DEFAULT_SUPERTONIC_MODEL_DIR


def _set_active_supertonic_engine(engine: SupertonicEngine | None) -> None:
    """Track the most recently loaded Supertonic engine for streaming reuse."""
    global _ACTIVE_SUPERTONIC_ENGINE
    with _ACTIVE_SUPERTONIC_ENGINE_LOCK:
        _ACTIVE_SUPERTONIC_ENGINE = engine


def _resolve_sentence_engine(
    voice_style: str,
    *,
    model_dir: str | None = None,
) -> SupertonicEngine:
    """Return a loaded Supertonic engine suitable for sentence streaming."""
    with _ACTIVE_SUPERTONIC_ENGINE_LOCK:
        active_engine = _ACTIVE_SUPERTONIC_ENGINE
    if (
        model_dir is None
        and active_engine is not None
        and active_engine._model is not None
        and active_engine._voice_style_name == voice_style
    ):
        return active_engine

    if model_dir is None:
        if active_engine is None and not _streaming_env_overrides_configured():
            logger.warning(
                "No active Supertonic engine registered for voice style %s; "
                "constructing a new streaming engine at the fallback model path.",
                voice_style,
            )
        resolved_model_dir = _resolve_streaming_model_dir()
    else:
        resolved_model_dir = model_dir

    cache_key = (resolved_model_dir, voice_style)
    with _ACTIVE_SUPERTONIC_ENGINE_LOCK:
        cached_engine = _STREAMING_ENGINE_CACHE.get(cache_key)
    if cached_engine is None or cached_engine._model is None:
        cached_engine = SupertonicEngine(
            model_dir=resolved_model_dir,
            voice_style=voice_style,
        )
        cached_engine.load()
        with _ACTIVE_SUPERTONIC_ENGINE_LOCK:
            _STREAMING_ENGINE_CACHE[cache_key] = cached_engine
    return cached_engine


def synthesize_to_speaker_by_sentence(
    text: str,
    voice_style: str,
    *,
    model_dir: str | None = None,
    output_device: str | None = None,
) -> SentenceSynthesisResult:
    """Synthesize sentence chunks, start playback early, and return timing metrics."""
    import numpy as np

    sentences = _split_text_into_sentences(text)
    if not sentences:
        return SentenceSynthesisResult(
            total_duration_ms=0.0,
            first_chunk_ms=0.0,
            sentence_count=0,
            full_wav_path=None,
        )

    engine = _resolve_sentence_engine(voice_style, model_dir=model_dir)
    start_time = time.monotonic()

    if len(sentences) == 1:
        audio, sample_rate = engine.synthesize(text)
        normalized_audio = _flatten_audio_samples(audio)
        if normalized_audio.size > 0:
            _play_sentence_chunk(normalized_audio, sample_rate, output_device)
        total_duration_ms = (time.monotonic() - start_time) * 1000.0
        return SentenceSynthesisResult(
            total_duration_ms=total_duration_ms,
            first_chunk_ms=total_duration_ms,
            sentence_count=1,
            full_wav_path=_write_temp_wav(normalized_audio, sample_rate),
        )

    first_audio, sample_rate = engine.synthesize(sentences[0])
    normalized_first_audio = _flatten_audio_samples(first_audio)
    first_chunk_ms = (time.monotonic() - start_time) * 1000.0
    first_chunk_duration_ms = _audio_duration_ms(normalized_first_audio, sample_rate)

    playback_queue: queue.Queue[tuple[np.ndarray, int] | None] = queue.Queue()
    playback_errors: list[BaseException] = []

    def _consume_playback_queue() -> None:
        try:
            while True:
                item = playback_queue.get()
                if item is None:
                    return
                chunk_audio, chunk_sample_rate = item
                _play_sentence_chunk(chunk_audio, chunk_sample_rate, output_device)
        except Exception as exc:  # pragma: no cover
            playback_errors.append(exc)

    playback_thread = threading.Thread(
        target=_consume_playback_queue,
        name="mungi-tts-sentence-playback",
        daemon=True,
    )
    playback_thread.start()
    playback_queue.put((normalized_first_audio, sample_rate))

    synthesized_chunks: list[np.ndarray] = [normalized_first_audio]
    try:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="mungi-tts-sentence",
        ) as executor:
            futures = [executor.submit(engine.synthesize, sentence) for sentence in sentences[1:]]
            for sentence_index, future in enumerate(futures, start=1):
                audio_chunk, chunk_sample_rate = future.result()
                normalized_chunk = _flatten_audio_samples(audio_chunk)
                if chunk_sample_rate != sample_rate:
                    msg = (
                        "Sentence chunk sample rate mismatch: "
                        f"expected {sample_rate}, got {chunk_sample_rate}"
                    )
                    raise RuntimeError(msg)
                if sentence_index == 1 and first_chunk_duration_ms < _MIN_FIRST_CHUNK_MS:
                    normalized_chunk = _prepend_silence(
                        normalized_chunk,
                        sample_rate,
                        _SHORT_FIRST_CHUNK_PADDING_MS,
                    )
                synthesized_chunks.append(normalized_chunk)
                playback_queue.put((normalized_chunk, sample_rate))

        total_duration_ms = (time.monotonic() - start_time) * 1000.0
        full_audio = (
            np.concatenate(synthesized_chunks)
            if synthesized_chunks
            else np.zeros(0, dtype=np.float32)
        )
        full_wav_path = _write_temp_wav(full_audio, sample_rate)
    finally:
        playback_queue.put(None)
        playback_thread.join()
    if playback_errors:
        first_error = playback_errors[0]
        if isinstance(first_error, RuntimeError):
            raise first_error
        msg = f"Sentence playback failed: {first_error}"
        raise RuntimeError(msg) from first_error

    return SentenceSynthesisResult(
        total_duration_ms=total_duration_ms,
        first_chunk_ms=first_chunk_ms,
        sentence_count=len(sentences),
        full_wav_path=full_wav_path,
    )


# -------------------------------------------------------------------
# Abstract TTS engine
# -------------------------------------------------------------------


class TTSEngine(abc.ABC):
    """Abstract base class for TTS engines."""

    @abc.abstractmethod
    def load(self) -> None:
        """Load the TTS model into memory.

        Raises:
            ImportError: If required package is not installed.
            FileNotFoundError: If model files are missing.
            RuntimeError: If model loading fails.
        """

    @abc.abstractmethod
    def synthesize(
        self,
        text: str | None,
        language: str = "ko",
    ) -> tuple[np.ndarray, int]:
        """Synthesize speech from text.

        Args:
            text: Input text to synthesize, or ``None``.
            language: Target speech language. ``"ko"`` (default) for Korean,
                ``"en"`` for English. Engines that only support a single
                language must still accept this parameter for interface
                consistency and log a warning when the requested language
                cannot be honored.

        Returns:
            Tuple of (audio_samples as float32 ndarray, sample_rate).
            Returns an empty array if text is None or blank.

        Raises:
            RuntimeError: If synthesis fails.
        """

    @abc.abstractmethod
    def unload(self) -> None:
        """Release model resources from memory.

        After calling this method the engine must not be used for
        synthesis until :meth:`load` is called again.
        """

    @abc.abstractmethod
    def engine_name(self) -> str:
        """Return the engine identifier string."""


# -------------------------------------------------------------------
# Supertonic 2 engine
# -------------------------------------------------------------------


class SupertonicEngine(TTSEngine):
    """Supertonic 2 Korean TTS engine wrapper.

    Wraps the supertonic Python package with defensive error
    handling for import and runtime failures.
    """

    def __init__(
        self,
        model_dir: str,
        voice_style: str = "F1",
        *,
        voice_style_ko: str | None = None,
        voice_style_en: str | None = None,
    ) -> None:
        """Initialize SupertonicEngine.

        Args:
            model_dir: Path to Supertonic 2 model directory.
            voice_style: Voice style name (e.g. "F1", "M1").
            voice_style_ko: Korean voice style name or custom JSON path.
            voice_style_en: English voice style name or custom JSON path.

        Raises:
            ValueError: If only one per-language voice style is supplied.
        """
        if (voice_style_ko is None) ^ (voice_style_en is None):
            msg = "voice_style_ko and voice_style_en must be supplied together"
            raise ValueError(msg)

        self._model_dir = model_dir
        if voice_style_ko is not None and voice_style_en is not None:
            self._voice_style_name = f"<bilingual:{voice_style_ko}|{voice_style_en}>"
            self._voice_style_ko_name: str | None = voice_style_ko
            self._voice_style_en_name: str | None = voice_style_en
        else:
            self._voice_style_name = voice_style
            self._voice_style_ko_name = None
            self._voice_style_en_name = None
        self._model: Any = None
        self._voice_style: Any = None
        self._voice_style_ko: Any = None
        self._voice_style_en: Any = None
        self._sample_rate: int = DEFAULT_SAMPLE_RATE

    def engine_name(self) -> str:
        """Return the engine identifier string."""
        return "supertonic"

    def unload(self) -> None:
        """Release the Supertonic model from memory."""
        self._model = None
        self._voice_style = None
        self._voice_style_ko = None
        self._voice_style_en = None
        _set_active_supertonic_engine(None)
        logger.info("Supertonic engine unloaded")

    @staticmethod
    def _is_voice_style_path(voice_style_name: str) -> bool:
        """Return True when a Supertonic voice selector is a file path."""
        return (
            voice_style_name.endswith(".json")
            or "/" in voice_style_name
            or "\\" in voice_style_name
        )

    def _resolve_voice_style(self, voice_style_name: str) -> Any:
        """Resolve a Supertonic preset or custom voice JSON path."""
        if self._model is None:
            msg = "Model not loaded. Call load() first."
            raise RuntimeError(msg)

        if self._is_voice_style_path(voice_style_name) and hasattr(
            self._model,
            "get_voice_style_from_path",
        ):
            voice_style = self._model.get_voice_style_from_path(voice_style_name)
            logger.info("Voice style loaded from path: %s", voice_style_name)
            return voice_style
        if hasattr(self._model, "get_voice_style"):
            voice_style = self._model.get_voice_style(voice_style_name)
            logger.info("Voice style preset '%s' loaded", voice_style_name)
            return voice_style

        logger.warning("get_voice_style* not available, synthesis may use defaults")
        return None

    def _supported_synthesis_chars(self) -> frozenset[str] | None:
        """Return Supertonic supported characters when the processor exposes them."""
        if self._model is None:
            return None

        text_processor = getattr(self._model, "text_processor", None)
        if text_processor is None:
            return None
        supported_chars_raw = getattr(text_processor, "supported_chars", None)
        if isinstance(supported_chars_raw, str):
            return frozenset(supported_chars_raw)
        if not isinstance(supported_chars_raw, (set, frozenset, list, tuple)):
            return None

        supported_chars = {
            char for char in supported_chars_raw if isinstance(char, str) and len(char) == 1
        }
        return frozenset(supported_chars) if supported_chars else None

    @staticmethod
    def _filter_text_to_supported_chars(
        text: str,
        supported_chars: frozenset[str],
    ) -> tuple[str, tuple[str, ...]]:
        """Replace characters outside Supertonic support with spaces."""
        result: list[str] = []
        stripped_chars: list[str] = []
        stripped_seen: set[str] = set()
        in_unsupported_run = False

        for char in text:
            if char in supported_chars or char.isspace():
                result.append(char)
                in_unsupported_run = False
                continue
            if char not in stripped_seen:
                stripped_chars.append(char)
                stripped_seen.add(char)
            if not in_unsupported_run:
                result.append(" ")
                in_unsupported_run = True

        cleaned = _WHITESPACE_RE.sub(" ", "".join(result)).strip()
        return cleaned, tuple(stripped_chars)

    def _strip_unsupported_synthesis_chars(self, text: str) -> str:
        """Defensively remove characters unsupported by the loaded Supertonic processor."""
        supported_chars = self._supported_synthesis_chars()
        if supported_chars is None:
            return text

        cleaned, stripped_chars = self._filter_text_to_supported_chars(text, supported_chars)
        if stripped_chars:
            logger.warning(
                "Stripped unsupported Supertonic text characters before synthesis: %s",
                ", ".join(f"{char} (U+{ord(char):04X})" for char in stripped_chars),
                extra={
                    "event": "tts_unsupported_chars_stripped",
                    "unsupported_chars": stripped_chars,
                    "unsupported_codepoints": tuple(
                        f"U+{ord(char):04X}" for char in stripped_chars
                    ),
                },
            )
        return cleaned

    def load(self) -> None:
        """Load the Supertonic 2 model.

        Tries multiple known API patterns for the supertonic package.

        Raises:
            ImportError: If supertonic package is not installed.
            FileNotFoundError: If model directory does not exist.
            RuntimeError: If model loading fails.
        """
        model_path = Path(self._model_dir)
        if not model_path.exists():
            msg = f"Supertonic model dir not found: {model_path}"
            raise FileNotFoundError(msg)

        try:
            import supertonic  # type: ignore[import-not-found, import-untyped]
        except ImportError:
            logger.error("supertonic package not installed. Install with: pip install supertonic")
            raise

        logger.info("Loading Supertonic 2 from: %s", self._model_dir)

        self._model = supertonic.TTS(
            model_dir=self._model_dir,
        )
        if hasattr(self._model, "sample_rate"):
            self._sample_rate = self._model.sample_rate

        if self._voice_style_ko_name is not None and self._voice_style_en_name is not None:
            self._voice_style_ko = self._resolve_voice_style(self._voice_style_ko_name)
            self._voice_style_en = self._resolve_voice_style(self._voice_style_en_name)
            if self._voice_style_ko is None or self._voice_style_en is None:
                msg = "bilingual mode requires both styles resolved"
                raise RuntimeError(msg)
        else:
            # Load voice style. Branch on whether voice_style is a path to a
            # custom voice JSON file (e.g. Supertone Voice Builder output at
            # /var/lib/mungi/voices/<name>.json) or a preset name shipped with
            # the model (e.g. "F2", "M1").
            self._voice_style = self._resolve_voice_style(self._voice_style_name)

        logger.info(
            "Supertonic loaded (sample_rate=%d, style=%s)",
            self._sample_rate,
            self._voice_style_name,
        )
        _set_active_supertonic_engine(self)

    def synthesize(
        self,
        text: str | None,
        language: str = "ko",
        total_steps: int = 10,
    ) -> tuple[np.ndarray, int]:
        """Synthesize speech using Supertonic 2.

        Args:
            text: Text to synthesize.
            language: Target speech language. ``"ko"`` for Korean or ``"en"``
                for English.
            total_steps: Supertonic synthesis step count. Defaults to ``10``
                (restored 2026-06-15 from the latency-driven ``7`` of #149,
                which under-converged the diffusion sampler and caused mushy /
                dropped syllables on the mung-ee voice; ``10`` is the user's
                2026-06-01 A/B/C clarity selection). Offline generators may pass
                a higher value for cleaner pre-rendered assets.

        Returns:
            Tuple of (audio_samples as float32 ndarray, sample_rate).
            Returns an empty array if text is None or blank.

        Raises:
            RuntimeError: If synthesis fails or model not loaded.
        """
        text = normalize_tts_text(text)
        if not text:
            import numpy as np

            logger.warning("Empty text received, skipping TTS synthesis")
            return np.array([], dtype=np.float32), self._sample_rate

        if self._model is None:
            msg = "Model not loaded. Call load() first."
            raise RuntimeError(msg)

        text = self._strip_unsupported_synthesis_chars(text)
        if not text:
            import numpy as np

            logger.warning(
                "Text empty after unsupported character filtering, skipping TTS synthesis"
            )
            return np.array([], dtype=np.float32), self._sample_rate

        kwargs: dict[str, Any] = {}
        language_norm = (language or "").strip().lower()
        if language_norm not in ("ko", "en"):
            logger.warning(
                "Unknown TTS language %r; defaulting to 'en'. Caller should pass 'ko' or 'en'.",
                language,
            )
            language_norm = "en"

        if self._voice_style_ko is not None:
            active_voice_style = (
                self._voice_style_ko if language_norm == "ko" else self._voice_style_en
            )
        elif self._voice_style is not None:
            active_voice_style = self._voice_style
        else:
            active_voice_style = None
        if active_voice_style is not None:
            kwargs["voice_style"] = active_voice_style
        kwargs["lang"] = language_norm
        kwargs["speed"] = 0.95
        kwargs["total_steps"] = total_steps

        result = self._synthesize_with_unsupported_char_recovery(text, kwargs)
        if result is None:
            import numpy as np

            return np.array([], dtype=np.float32), self._sample_rate

        audio = self._extract_audio(result)
        # Supertonic may return 2D arrays (e.g. shape (1, N)); flatten to 1D
        if audio.ndim > 1:
            audio = audio.reshape(-1)
        return audio, self._sample_rate

    def _invoke_model_synthesize(self, text: str, kwargs: dict[str, Any]) -> Any:
        """Call the loaded Supertonic model, falling back to the no-kwargs API.

        Raises:
            RuntimeError: If synthesis fails for a reason other than the older
                Supertonic ``synthesize`` signature lacking keyword arguments.
        """
        try:
            return self._model.synthesize(text, **kwargs)
        except TypeError:
            # Fallback for older Supertonic API without voice_style kwarg.
            try:
                return self._model.synthesize(text)
            except RuntimeError:
                raise
            except Exception as exc:
                msg = f"Supertonic synthesis failed: {exc}"
                raise RuntimeError(msg) from exc
        except RuntimeError as exc:
            # Re-wrap so the unsupported-character marker survives even when the
            # model raises a bare RuntimeError without the standard prefix.
            if _is_unsupported_char_error(exc):
                msg = f"Supertonic synthesis failed: {exc}"
                raise RuntimeError(msg) from exc
            raise
        except Exception as exc:
            msg = f"Supertonic synthesis failed: {exc}"
            raise RuntimeError(msg) from exc

    def _synthesize_with_unsupported_char_recovery(
        self, text: str, kwargs: dict[str, Any]
    ) -> Any | None:
        """Invoke synthesis, recovering from Supertonic unsupported-character failures.

        Defense-in-depth net so a single unforeseen character never crashes the
        caller. On an unsupported-character failure the offending characters are
        stripped and synthesis is retried once; if it still fails (or no text
        remains), a structured warning is logged and ``None`` is returned so the
        caller degrades to "no audio for this segment" instead of raising.

        Returns:
            The raw synthesis result, or ``None`` when the segment was skipped.
        """
        try:
            return self._invoke_model_synthesize(text, kwargs)
        except RuntimeError as exc:
            if not _is_unsupported_char_error(exc):
                raise
            offending = _extract_unsupported_chars(str(exc))
            retry_text = _strip_chars(text, offending) if offending else ""
            if retry_text and retry_text != text:
                logger.warning(
                    "Supertonic rejected unsupported character(s) %s; "
                    "stripping and retrying synthesis once",
                    offending,
                    extra={
                        "event": "tts_unsupported_chars_retry",
                        "unsupported_chars": offending,
                        "unsupported_codepoints": tuple(f"U+{ord(char):04X}" for char in offending),
                    },
                )
                try:
                    return self._invoke_model_synthesize(retry_text, kwargs)
                except RuntimeError as retry_exc:
                    if not _is_unsupported_char_error(retry_exc):
                        raise
                    exc = retry_exc
            logger.warning(
                "Supertonic synthesis skipped after unsupported-character failure: %s",
                exc,
                extra={
                    "event": "tts_unsupported_chars_skipped",
                    "unsupported_chars": _extract_unsupported_chars(str(exc)),
                },
            )
            return None

    def _extract_audio(self, result: Any) -> np.ndarray:
        """Extract numpy audio array from synthesis result.

        Args:
            result: Raw result from the TTS engine.

        Returns:
            Float32 numpy array of audio samples.

        Raises:
            RuntimeError: If audio cannot be extracted.
        """
        import numpy as np

        if isinstance(result, np.ndarray):
            return result.astype(np.float32)

        if isinstance(result, tuple) and len(result) >= 1:
            audio = result[0]
            if len(result) >= 2 and isinstance(result[1], int):
                self._sample_rate = result[1]
            if isinstance(audio, np.ndarray):
                return audio.astype(np.float32)

        if isinstance(result, dict) and "audio" in result:
            audio = result["audio"]
            if "sample_rate" in result:
                self._sample_rate = int(result["sample_rate"])
            if isinstance(audio, np.ndarray):
                return audio.astype(np.float32)

        if isinstance(result, bytes | bytearray):
            count = len(result) // 2
            samples = struct.unpack(f"<{count}h", result)
            return np.array(samples, dtype=np.float32) / 32768.0

        if isinstance(result, list):
            return np.array(result, dtype=np.float32)

        msg = f"Cannot extract audio from result type: {type(result).__name__}"
        raise RuntimeError(msg)
