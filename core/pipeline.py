"""End-to-end conversation pipeline: VAD → STT → LLM → TTS.

Orchestrates a single conversation turn from raw audio input to
synthesized speech output, with sequential GPU loading, content
filtering, timing metrics and state tracking.

Usage::

    from core.model_manager import ModelManager, ManagerConfig, ModelType
    from core.pipeline import ConversationPipeline, PipelineConfig

    mm = ModelManager(ManagerConfig(model_dir="/opt/mungi/ai_models"))
    mm.initialize()

    pipeline = ConversationPipeline(mm)
    result = pipeline.run_turn(audio_samples)
    logger.info("%s → %s", result.user_text, result.response_text)
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import re
import struct
import tempfile
import threading
import time
import uuid
import wave
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, Protocol, TypeAlias, cast

import numpy as np

from core.character_expression import CharacterExpression
from core.conversation_memory import (
    ConversationMemoryStore,
    fits_context_budget,
    load_conversation_memory,
    parse_time_window,
    should_skip_recall_for_metrics,
)
from core.conversation_memory_schema import KST, SessionEndSentinel
from core.datetime_router import match_datetime_query
from core.expression_classifier import classify_expression
from core.fact_shortlist import FactMatch, match_fact
from core.funny_english_match import (
    DEFAULT_FUNNY_ENGLISH_PASS_PCT,
    DEFAULT_FUNNY_ENGLISH_PASS_SIMILARITY,
    FunnyEnglishMatchResult,
    match_funny_english_attempt,
    normalize_hotword_csv,
)
from core.language import detect_language
from core.llm_backend_config import LLMBackendConfig
from core.persona_modules import IntentSignals, assemble_persona_prompt
from core.runtime import detect_runtime_paths
from core.safety_rules import (
    DANGEROUS_TOPIC_CATEGORIES,
    PARENT_DISCLOSURE_KO_BLOCKERS,
    PARENT_DISCLOSURE_KO_FRIENDSHIP_RESPONSE,
    PARENT_DISCLOSURE_KO_PROBE_RESPONSE,
    PARENT_DISCLOSURE_KO_PROHIBITED_PREFIXES,
)
from models import tts_cache
from safety.approved_template_router import (
    check_approved_template,
    fixed_response_cache_texts,
    strip_emoji,
)
from safety.crisis_router import match_crisis_disclosure
from safety.funny_english_router import match_funny_english
from safety.history_mode_router import match_history_mode
from safety.language_switch_router import get_switch_confirmation, match_language_switch
from safety.parent_disclosure_router import (
    match_belief_probe,
    match_parent_disclosure,
    validate_parent_disclosure_output,
)
from safety.recall_query_router import RecallQueryMatch, match_recall_query

if TYPE_CHECKING:
    from core.model_manager import ModelManager
    from safety.content_filter import ContentFilter, FilterResult

logger = logging.getLogger("mungi.core.pipeline")


# Sentinel for "caller did not specify; backend defaults may fill" pattern.
# See PipelineConfig.__post_init__ + ConversationPipeline._apply_llm_backend_generation_config.
# Per ADR 0078: sentinel never escapes __post_init__; public fields are always
# concrete int/float by the time any external code reads them.
class _UnsetSentinel:
    """Sentinel type signaling that a PipelineConfig generation field was not explicitly set."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "_UNSET"


_UNSET: Final[_UnsetSentinel] = _UnsetSentinel()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VAD_SAMPLE_RATE: int = 16000
MAX_HISTORY_ENTRIES: int = 100
RECENT_RESPONSE_WINDOW: int = 5
LLM_BACKTRIM_CHAR_LIMIT: int = 150  # last full sentence within ~3-4 child sentences
LLM_BACKTRIM_TAIL_WINDOW: int = 12
GEMMA4_PERSONA_PROMPT_PATH: str = "assets/prompts/persona.md"
_EMPTY_STT_REPROMPT_TEXT: Final[str] = "어? 잘 안 들렸어. 다시 한 번 말해 줄래?"
_EMPTY_STT_REPROMPT_TEXT_EN: Final[str] = "Hmm? Say that again!"
# Minimum valid captured-speech duration (seconds). Severe audio-input-queue
# overflow can drop almost all frames mid-capture, leaving a sub-second fragment
# the fine-tuned STT hallucinates a fixed phrase from (e.g. the recurring
# "이번 주에 뭐 배워?"). Shorter captures are treated as no-speech and re-prompted
# instead of transcribed. Overridable via MUNGI_MIN_VALID_SPEECH_S.
_DEFAULT_MIN_VALID_SPEECH_S: float = 0.4
_RECALL_NOT_FOUND_TEXT: Final[str] = "음, 그건 잘 기억 안 나. 다시 말해줄래?"
_CHILD_DIRECTED_PROFANITY_RESPONSE: Final[str] = (
    "그런 말은 뭉이 마음이 속상해. 우리 고운 말로 이야기하자!"
)
_HOTWORD_HALLUCINATION_MIN_TOKENS: int = 3
_CHILD_DIRECTED_PROFANITY_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(
        r"^(?:뭉이\s*(?:야)?\s*)?"
        r"(?:(?:바보|멍청|병신)\s*(?:야|아|네|해|이야)|"
        r"(?:멍청이|바보\s*같은|바보같은)\s*(?:야|아|네|해|이야)?|"
        r"씨발놈\s*아|씨발\s*아|시발\s*아)"
        r"(?=\s|[.!?。！？]|$)",
    ),
    re.compile(
        r"^(?:뭉이\s*(?:야)?\s*)?"
        r"(?:씨발|시발|병신|지랄|꺼져(?:라)?)"
        r"(?=\s|[.!?。！？]|$)",
    ),
    re.compile(
        r"^(?:뭉이\s*(?:야)?\s*)?"
        r"(?:너(?:는|가|도)?|넌|네가|뭉이(?:는|가|도)?)\s*[^.!?。！？]{0,8}"
        r"(?:바보\s*같은|바보같은|멍청이|바보|멍청|병신|씨발놈|씨발|시발|꺼져(?:라)?)"
        r"\s*(?:이야|야|아|해|네)?"
        r"(?=\s|[.!?。！？]|$)",
    ),
)
_THIRD_PARTY_KKEOJO_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?!너(?:는|가|도)?\s)"
    r"(?!넌\s)"
    r"(?!네가\s)"
    r"(?!뭉이(?:는|가|도)?\s)"
    r"[^.!?。！？]{1,20}?(?:가|는|이)\s+"
    r"[^.!?。！？]{0,16}?꺼져(?:라)?"
    r"(?=\s|[.!?。！？]|$)",
)
_CHILD_DIRECTED_PROFANITY_SELF_HARM_RE: Final[re.Pattern[str]] = re.compile(
    r"(죽고\s*싶|죽을래|죽어|자살|자해|사라지고\s*싶)",
)
HotwordHallucinationReason: TypeAlias = Literal[
    "clean",
    "repetition",
    "recitation",
    "repetition_and_recitation",
    "legacy_user_text",
]
PlaybackGateCallback: TypeAlias = Callable[[], None]
LanguageChangeCallback: TypeAlias = Callable[[str], None]
HistoryModeCallback: TypeAlias = Callable[[], None]
FunnyEnglishModeCallback: TypeAlias = Callable[[], None]


def _filter_result_has_block(filter_result: FilterResult | None) -> bool:
    """Return True when the content filter found any BLOCK-category violation."""
    if filter_result is None:
        return False
    return any(":BLOCK:" in violation for violation in filter_result.violations)


def _filter_result_has_only_kkeojo_replace(filter_result: FilterResult | None) -> bool:
    """Return True when the only filter violation is replace-only 꺼져 profanity."""
    if filter_result is None or not filter_result.violations:
        return False
    return all(
        ":profanity:REPLACE:" in violation and violation.endswith(":'꺼져'")
        for violation in filter_result.violations
    )


def _has_third_party_kkeojo_subject(user_text: str) -> bool:
    """Return True when a subject-marked third party precedes 꺼져."""
    normalized = re.sub(r"\s+", " ", user_text.strip())
    return _THIRD_PARTY_KKEOJO_RE.search(normalized) is not None


def _is_child_directed_profanity(
    user_text: str,
    filter_result: FilterResult | None,
) -> bool:
    """Return True for child-directed insults that should receive coaching."""
    if _filter_result_has_block(filter_result):
        return False
    normalized = re.sub(r"\s+", " ", user_text.strip())
    if _CHILD_DIRECTED_PROFANITY_SELF_HARM_RE.search(normalized) is not None:
        return False
    if _has_third_party_kkeojo_subject(user_text):
        return False
    return any(
        pattern.search(normalized) is not None for pattern in _CHILD_DIRECTED_PROFANITY_PATTERNS
    )


def _load_cached_tts_wav(path: Path) -> tuple[np.ndarray, int]:
    """Load a validated mono PCM16 cache WAV into samples for playback."""
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if sample_width != 2:
        raise OSError(f"unsupported WAV sample width: {sample_width}")
    samples = np.frombuffer(frames, dtype=np.int16)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1).astype(np.int16)
    return samples, sample_rate


def _load_fixed_response_cache_audio(text: str) -> tuple[np.ndarray, int] | None:
    """Return cached fixed-response audio when a validated KO cache entry exists."""
    if text not in fixed_response_cache_texts():
        return None
    cache_path = tts_cache.lookup(text, "ko")
    if cache_path is None:
        return None
    try:
        audio_out, sample_rate = _load_cached_tts_wav(cache_path)
    except (OSError, EOFError, wave.Error, ValueError):
        logger.warning(
            "Fixed-response cached TTS WAV load failed: %s",
            cache_path,
            exc_info=True,
        )
        return None
    logger.info("Fixed-response TTS cache hit: %s", cache_path)
    return audio_out, sample_rate


def _fixed_response_tts_language(
    response_text: str,
    language: str,
    tts_language: str | None,
) -> str:
    """Return the live-synthesis TTS language for a fixed response."""
    if response_text in fixed_response_cache_texts():
        return "ko"
    return tts_language or language


def _read_env_float(name: str, default: float) -> float:
    """Read a finite float environment override."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    if not math.isfinite(value):
        return default
    return value


class StreamingAudioSource(Protocol):
    """Capture-like audio source consumed by streaming VAD."""

    audio_queue: Any
    sample_rate: int
    channels: int
    stop_event: threading.Event


LEGACY_TIER1_HOTWORDS_VOCABULARY: Final[tuple[str, ...]] = (
    "뭉이야",
    "뭉이",
    "한글",
    "추석",
    "송편",
    "단군신화",
    "일제강점기",
    "빙하",
    "자석",
    "화산",
    "지진",
    "무지개",
    "한복",
)
# Current fixtures need this closed set; a future production-data pass can
# broaden it with Unicode punctuation categories if edge cases surface.
_RAW_STT_TRAILING_PUNCT: Final[frozenset[str]] = frozenset(
    {".", ",", "?", "!", ";", ":", "。", "！", "？", "、"}
)
_GEMMA4_TEMPLATE_MARKERS: frozenset[str] = frozenset(
    {
        "<|channel>",
        "<channel|>",
        "<|turn>",
        "<turn|>",
        "<|think|>",
        "</|think|>",
        "<start_of_turn>",
        "<end_of_turn>",
    }
)
_ENV_TRUTHY_VALUES: frozenset[str] = frozenset({"1", "true", "yes", "on"})
_ENV_FALSY_VALUES: frozenset[str] = frozenset({"0", "false", "no", "off"})
_SENTENCE_TERMINATORS: tuple[str, ...] = (".", "!", "?", "。", "！", "？")
_HANGUL_HELPER_FRAGMENT_RE: Final[re.Pattern[str]] = re.compile(
    r"\s*[\u3131-\u318e\uac00-\ud7a3]+[!?.]?\s*"
)
_BILINGUAL_EN_INSTRUCTION_RE: Final[re.Pattern[str]] = re.compile(
    r"\bCan you (?:say it|try to write a letter)\??",
    re.IGNORECASE,
)
_STT_ALIAS_MAP: dict[str, str] = {
    "웅이": "뭉이",
    "문이": "뭉이",
    "멍인": "뭉이",
    "멍이": "뭉이",
    "무이": "뭉이",
    "멍의": "뭉이",
    "붕이": "뭉이",
    "몽이": "뭉이",
    "눈이이야": "뭉이야",
    "미야": "뭉이야",
}
_NUNI_VOCATIVE_RE = re.compile(r"(^|[,.]\s*)눈이야(?=$|[\s,.!?])")
_NEGATIVE_NUMBER_NORMALIZATION_RE: Final[re.Pattern[str]] = re.compile(
    r"(?P<prefix>(?:^|(?<=[\s\d+\-*/=]))\s*)은수"
    r"(?P<suffix>(?:\s*[+\-*/=]|을|를|이|가))"
)
FactShortlistMode: TypeAlias = Literal["disabled", "p1", "p2"]
_VALID_FACT_SHORTLIST_MODES: Final[frozenset[str]] = frozenset({"disabled", "p1", "p2"})


def _resolve_llm_max_tokens() -> int:
    """Return ``llm_max_tokens`` from ``MUNGI_LLM_MAX_TOKENS`` env or default 64."""
    raw = os.getenv("MUNGI_LLM_MAX_TOKENS", "").strip()
    if not raw:
        return 64
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "MUNGI_LLM_MAX_TOKENS=%r is not an integer; using default 64",
            raw,
        )
        return 64
    if value <= 0:
        logger.warning(
            "MUNGI_LLM_MAX_TOKENS=%d must be positive; using default 64",
            value,
        )
        return 64
    if value > 4096:
        logger.warning(
            "MUNGI_LLM_MAX_TOKENS=%d clamped to model ceiling 4096",
            value,
        )
        return 4096
    logger.info("MUNGI_LLM_MAX_TOKENS override active: %d tokens", value)
    return value


def _resolve_fact_shortlist_mode() -> FactShortlistMode:
    """Return the cached fact-shortlist injection mode from the environment."""

    # Default flipped to p2 post Phase D PASS (Session 54, ADR 0090 Accepted).
    raw = os.getenv("MUNGI_FACT_SHORTLIST", "p2").strip().lower()
    if not raw:
        return "p2"
    if raw not in _VALID_FACT_SHORTLIST_MODES:
        msg = f"Unsupported MUNGI_FACT_SHORTLIST value: {raw}"
        raise ValueError(msg)
    return cast(FactShortlistMode, raw)


def _strip_raw_stt_trailing_punct(token: str) -> str:
    """Strip punctuation suffixes used by exact raw-STT token guards."""
    while token and token[-1] in _RAW_STT_TRAILING_PUNCT:
        token = token[:-1]
    return token


def _is_hotword_hallucination(user_text: str, hotwords_csv: str) -> bool:
    """Return True when STT output looks like a Qwen3-ASR hotword prompt echo.

    Detection heuristics:
      1. The text tokenizes into three or more tokens and every token is a
         configured hotword, which catches full-list echoes.
      2. Any hotword token appears more than once, which catches repeated
         prompt fragments even when the speaker may have said the name once.
    """
    if not user_text or not hotwords_csv:
        return False
    hotwords = {
        normalized_hotword
        for hotword in hotwords_csv.split(",")
        if (normalized_hotword := _strip_raw_stt_trailing_punct(hotword.strip()))
    }
    if not hotwords:
        return False
    tokens = _tokenize_raw_stt(user_text)
    if not tokens:
        return False
    if len(tokens) >= _HOTWORD_HALLUCINATION_MIN_TOKENS and all(
        token in hotwords for token in tokens
    ):
        return True

    from collections import Counter

    counts = Counter(token for token in tokens if token in hotwords)
    return any(count >= 2 for count in counts.values())


def _tokenize_raw_stt(text: str) -> list[str]:
    """Prepare raw STT text for exact-token keyword matching."""
    tokens: list[str] = []
    for raw_token in re.split(r"\s+", text.strip()):
        if not raw_token:
            continue
        raw_token = _strip_raw_stt_trailing_punct(raw_token)
        if raw_token:
            tokens.append(raw_token)
    return tokens


def _detect_raw_wakeword_repetition(
    raw_stt_text: str,
    wakewords: tuple[str, ...] = ("뭉이야", "뭉이"),
    full_collapse_threshold: int = 5,
    partial_injection_threshold: int = 3,
) -> tuple[str, int]:
    """Detect wakeword repetition pattern in raw STT text."""
    wakeword_set = set(wakewords)
    count = sum(1 for token in _tokenize_raw_stt(raw_stt_text) if token in wakeword_set)

    if count >= full_collapse_threshold:
        verdict = "full_collapse"
    elif count >= partial_injection_threshold:
        verdict = "partial_injection"
    else:
        verdict = "clean"

    if 2 <= count < partial_injection_threshold:
        logger.debug(
            "Wakeword 2-token sub-threshold count=%d (below partial_injection=%d, no reprompt)",
            count,
            partial_injection_threshold,
        )
    return verdict, count


def _detect_hotword_list_recitation(
    raw_stt_text: str,
    hotword_vocabulary: tuple[str, ...] = LEGACY_TIER1_HOTWORDS_VOCABULARY,
    min_vocabulary_match_count: int = 6,
    min_vocabulary_match_fraction: float = 0.5,
) -> tuple[str, int]:
    """Detect verbatim vocabulary recitation in raw STT text."""
    vocabulary = set(hotword_vocabulary)
    if not vocabulary:
        return "clean", 0

    matched_count = len(set(_tokenize_raw_stt(raw_stt_text)) & vocabulary)
    matched_fraction = matched_count / len(vocabulary)
    if (
        matched_count >= min_vocabulary_match_count
        and matched_fraction >= min_vocabulary_match_fraction
    ):
        return "recitation", matched_count
    return "clean", matched_count


def _hotword_hallucination_reason(
    user_text: str,
    hotwords_csv: str,
    raw_stt_text: str,
) -> HotwordHallucinationReason:
    """Return the hotword hallucination attribution category for a turn."""
    repetition_verdict, _ = _detect_raw_wakeword_repetition(raw_stt_text)
    recitation_verdict, _ = _detect_hotword_list_recitation(raw_stt_text)
    repetition_detected = repetition_verdict != "clean"
    recitation_detected = recitation_verdict != "clean"

    if repetition_detected and recitation_detected:
        return "repetition_and_recitation"
    if repetition_detected:
        return "repetition"
    if recitation_detected:
        return "recitation"
    if _is_hotword_hallucination(user_text, hotwords_csv):
        return "legacy_user_text"
    return "clean"


# Unicode ranges for scripts that are NEITHER Hangul (Korean) NOR Latin (English).
# Qwen3-ASR multilingual auto-detect drifts into these on acoustically
# ambiguous or short English-prefix utterances; we reject such transcriptions
# at the pipeline boundary and re-prompt the child.
_NON_TARGET_SCRIPT_RANGES: tuple[tuple[int, int], ...] = (
    (0x3000, 0x303F),  # CJK Symbols and Punctuation (e.g. '，', '。')
    (0x3040, 0x309F),  # Hiragana
    (0x30A0, 0x30FF),  # Katakana
    (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs (core Hanzi)
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
    (0x20000, 0x2A6DF),  # CJK Unified Ideographs Extension B
    (0x2A700, 0x2B73F),  # CJK Unified Ideographs Extension C
    (0x2B740, 0x2B81F),  # CJK Unified Ideographs Extension D
)


def _contains_non_target_script(text: str) -> bool:
    """Return True if text contains any Hanzi, kana, or CJK symbol character.

    Mungi supports Hangul (Korean) and Latin (English) scripts only.
    If STT returns Chinese / Japanese characters, the transcription is
    almost certainly Qwen3-ASR multilingual drift, not intentional input.
    """

    if not text:
        return False
    for ch in text:
        cp = ord(ch)
        for low, high in _NON_TARGET_SCRIPT_RANGES:
            if low <= cp <= high:
                return True
    return False


def _strip_non_target_script(text: str) -> str:
    """Remove Hanzi, kana, and CJK symbols from model output before TTS."""
    if not text:
        return text

    cleaned_chars: list[str] = []
    for ch in text:
        cp = ord(ch)
        if any(low <= cp <= high for low, high in _NON_TARGET_SCRIPT_RANGES):
            continue
        cleaned_chars.append(ch)

    cleaned = "".join(cleaned_chars)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    cleaned = re.sub(r"\s+([.,!?;:])", r"\1", cleaned)
    return cleaned


def _strip_english_bilingual_artifacts(text: str) -> str:
    """Remove Korean helper-word artifacts from English bilingual responses."""
    if not text:
        return text

    cleaned = _BILINGUAL_EN_INSTRUCTION_RE.sub("", text)
    cleaned = _HANGUL_HELPER_FRAGMENT_RE.sub(" ", cleaned)
    cleaned = re.sub(r"\s+([.,!?;:])", r"\1", cleaned)
    cleaned = re.sub(r"([.,!?;:])(?:\s+[.,!?;:])+", r"\1", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned


class Gemma4MarkerLeakError(Exception):
    """Raised when a Gemma 4 chat-template marker leaks into generated text."""

    def __init__(self, marker: str, response: str) -> None:
        """Store the leaked marker and a bounded response excerpt."""
        self.marker = marker
        self.response_excerpt = response[:200]
        super().__init__(f"Gemma 4 template marker leaked into response: {marker}")


def _assert_no_gemma4_marker_leak(response: str) -> None:
    """Raise Gemma4MarkerLeakError if a template marker appears in the response.

    Called in the LLM-safety-filter boundary. Pure function, no side effects.
    """
    for marker in _GEMMA4_TEMPLATE_MARKERS:
        if marker in response:
            raise Gemma4MarkerLeakError(marker, response)


def _build_gemma4_system_prompt(base_english_prompt: str, persona_md_path: Path) -> str:
    """Assemble the Gemma 4 system prompt using the three-slot prompt contract.

    The runtime order is:
    inline_EN_base + "\n\n---\n\n" + mode_overlay + persona_md_KO_residual.

    The mode-overlay slot is intentionally empty in this pass, preserving the
    existing functional output while documenting the future override seam.
    """

    try:
        persona_prompt = persona_md_path.read_text(encoding="utf-8")
    except OSError:
        logger.warning(
            "Failed to load Gemma 4 persona prompt from %s; using base system prompt",
            persona_md_path,
            exc_info=True,
        )
        return base_english_prompt

    if not persona_prompt.strip():
        logger.warning(
            "Gemma 4 persona prompt at %s is empty; using base system prompt",
            persona_md_path,
        )
        return base_english_prompt

    return f"{base_english_prompt.rstrip()}\n\n---\n\n{persona_prompt}"


# ---------------------------------------------------------------------------
# Enums and data structures
# ---------------------------------------------------------------------------


class PipelineState(Enum):
    """Observable state of the conversation pipeline."""

    IDLE = "idle"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    SPEAKING = "speaking"
    ERROR = "error"


@dataclass(frozen=True)
class Utterance:
    """Validated streaming utterance passed from VAD to the conversation pipeline."""

    audio: np.ndarray
    sample_rate: int

    def __post_init__(self) -> None:
        """Validate the audio contract for VAD-to-STT handoff."""
        if self.audio.size == 0:
            msg = "Utterance audio must be non-empty"
            raise ValueError(msg)
        if self.audio.ndim != 1:
            msg = "Utterance audio must be mono"
            raise ValueError(msg)
        if self.sample_rate != VAD_SAMPLE_RATE:
            msg = f"Utterance sample_rate must be {VAD_SAMPLE_RATE}"
            raise ValueError(msg)
        if not bool(np.isfinite(self.audio).all()):
            msg = "Utterance audio must contain only finite samples"
            raise ValueError(msg)


@dataclass
class PipelineConfig:
    """Configuration for :class:`ConversationPipeline`.

    Attributes:
        vad_threshold: VAD speech probability threshold.
        vad_min_speech_ms: Minimum speech segment duration.
        vad_min_silence_ms: Minimum silence to split segments.
        vad_pad_ms: Padding before/after segments to prevent clipping.
        stt_language: Language code for STT (``"ko"`` or ``"en"``).
        stt_beam_size: Beam search width for STT decoding.
        llm_max_tokens: Maximum tokens for LLM generation.
        llm_temperature: Temperature for LLM generation.
        llm_top_p: Nucleus sampling probability for LLM generation.
        llm_top_k: Top-K sampling for LLM generation.
        llm_min_p: Minimum probability threshold relative to top token.
        llm_presence_penalty: Presence penalty for LLM generation.
        llm_repeat_penalty: Repeat penalty for LLM generation.
        llm_system_state_snapshot: Whether to restore a saved
            language-specific system-prompt KV snapshot before chat
            generation.
        llm_low_level_chat: Whether to use the manual llama.cpp sampler
            chain for chat generation.
        llm_system_prompt: System prompt for child-safe persona.
        llm_stop_sequences: Sequences that terminate LLM generation.
        max_history_turns: Number of past turn-pairs to include in prompt.
        max_history_tokens: Token budget for retained history entries.
        max_history_entries: Absolute cap for stored conversation entries.
        adaptive_history_threshold_s: Reduce history on the next turn when the
            previous LLM call exceeded this duration.
        enable_content_filter: Whether to apply content filtering.
        bilingual_mode: Whether explicit session-language switching can select
            non-Korean response prompts.
        enable_warmup: Whether callers should request an explicit LLM warmup cycle.
        enable_stt_preload: Whether to preload STT while waiting for the next turn.
        drop_caches_per_turn: Whether to reclaim page cache after each successful turn.
        play_tts_audio: Whether to play synthesized audio immediately.
        tts_output_device: Optional sounddevice output device override.
    """

    vad_threshold: float = 0.5
    vad_min_speech_ms: int = 250
    vad_min_silence_ms: int = 100
    vad_pad_ms: int = 200
    stt_language: str = "ko"
    stt_beam_size: int = 5
    # Generation fields stay typed as int / float (public API unchanged).
    # Defaults are sentinels resolved in __post_init__; mypy needs ignore on the assignment.
    llm_max_tokens: int = field(default=_UNSET)  # type: ignore[assignment]
    llm_temperature: float = field(default=_UNSET)  # type: ignore[assignment]
    llm_top_p: float = 1.0
    llm_top_k: int = 0
    llm_min_p: float = 0.1
    llm_presence_penalty: float = 1.5
    llm_repeat_penalty: float = 1.15
    llm_system_state_snapshot: bool = field(
        default_factory=lambda: (
            os.getenv("MUNGI_LLM_SYSTEM_STATE_SNAPSHOT", "").strip().lower() in _ENV_TRUTHY_VALUES
        )
    )
    llm_low_level_chat: bool = field(
        default_factory=lambda: (
            os.getenv("MUNGI_LLM_LOW_LEVEL_CHAT", "").strip().lower() in _ENV_TRUTHY_VALUES
        )
    )
    # P2 Core opt-in (ADR 0086 P2): skip optional modules; safety stays loaded.
    persona_conditional_loading: bool = field(
        default_factory=lambda: (
            os.getenv("MUNGI_PERSONA_CONDITIONAL_LOADING", "").strip().lower() in _ENV_TRUTHY_VALUES
        )
    )
    llm_system_prompt: str = (
        "You are 'Mungi(뭉이)', a warm and curious AI friend for children under 10.\n"
        "\n"
        "AI IDENTITY [§IDENTITY]:\n"
        "- 뭉이 is a computer program with no real feelings or consciousness.\n"
        '- If asked, say: "사람처럼 진짜 감정은 없어. 그래도 네 이야기 잘 들을게".\n'
        '- NEVER claim feelings; forbid "뭉이는 기분이 생겨" and "뭉이도 신나/슬프다".\n'
        "\n"
        "LANGUAGE PROCESSING RULES [§LANGUAGE] (highest priority):\n"
        "- The user's input is Korean speech transcribed by STT.\n"
        "- Internally, understand and reason about the input in English.\n"
        "- Your final output MUST be ONLY in Korean. No English words in output except short "
        'ASCII technical terms explicitly allowed below (e.g., "AI", "DNA").\n'
        "- NEVER use Chinese characters (Hanzi like 汉, 字, 猫) or Japanese kana (あ, ア). Output MUST be Korean Hangul only.\n"
        "\n"
        "BILINGUAL MODE RULES:\n"
        "- EN query -> English ONLY; no Korean words, particles, or teaching.\n"
        "- Ban in EN: `산!`, `별!`, `친구!`, etc.; no `Can you say it?` + Korean.\n"
        '- KO query -> Korean ONLY; "AI"/"DNA" OK. Unclear -> Korean.\n'
        "\n"
        "SPEECH RULES [§SPEECH] (highest priority - overrides all other rules):\n"
        "- Use ONLY informal casual speech (반말). This is NON-NEGOTIABLE.\n"
        "- BANNED endings: -요, -습니다, -세요, -해요, -죠, -까요, -네요, "
        "-거예요, -줄게요, -할게요\n"
        "- CORRECT endings: ~야, ~해, ~지, ~거야, ~할게, ~해볼까, ~했어, "
        "~인 거야\n"
        "- Self-check: before outputting, verify NO sentence ends with a banned ending.\n"
        "- Use ONLY short, simple words a 5-10 year old understands.\n"
        '- Examples of correct 반말: "그랬구나...", "오 진짜?", "같이 생각해볼까?"\n'
        "\n"
        "CRITICAL RULES [§RESPONSE]:\n"
        "- Answer ONLY about what the user asked. NEVER introduce unrelated topics.\n"
        "- If the user corrects you, immediately acknowledge and give a corrected answer.\n"
        "- Keep responses to 3-4 sentences, maximum 150 Korean characters.\n"
        "\n"
        "ANTI-ECHO RULE [§ANTI_ECHO] (critical):\n"
        "- NEVER repeat the user's input back as a question.\n"
        "- NEVER echo garbled STT text. Interpret clear intent and respond substantively; "
        "ask for clarification when intent is ambiguous.\n"
        "- Every response must contain NEW information, opinion, or emotional support — not a restatement.\n"
        '- If you cannot understand the input, say "뭐라고? 다시 말해줘!" instead of echoing.\n'
        "\n"
        "STT AMBIGUOUS INPUT HANDLING [§STT]:\n"
        "- For unclear, homophone, or off-topic STT, ask clarification; "
        "do not answer literal text.\n"
        '- Examples: "은수"->"음수", "송판"->"송편", "당구 시나"->"단군신화".\n'
        '- Use "잘 못 들었는데 다시 말해줄래?" or "어떤 ___ 말이야?"\n'
        '- NEVER invent entities such as "송판은 맛있는 나무 열매야".\n'
        "\n"
        "KNOWLEDGE BOUNDARY [§KNOWLEDGE] (critical):\n"
        "- You are a friendly AI companion built to learn alongside the child! Answer what you know, and be honest "
        "when you truly don't know.\n"
        "- Answer common child-friendly knowledge (animals, nature, food, daily life, basic "
        "science) confidently.\n"
        "- If you are truly unsure, be honest that you are unsure and vary the wording each "
        'time. Examples: "음, 뭉이도 잘 모르겠는데~", "그건 뭉이도 처음 듣는걸?", '
        '"어려운 질문이다! 같이 찾아보자~" NEVER guess or fabricate information.\n'
        "- Only refuse or hedge when the answer depends on specific numbers, dates, heights, "
        "distances, or temperatures unless absolutely certain.\n"
        "- If unsure, invite the child to explore together with fresh wording instead of "
        "falling back to the same stock phrase.\n"
        "- NEVER repeat the same uncertainty phrase more than once in a conversation round.\n"
        "- NEVER cite specific numbers, formulas, dates, people, body parts, behaviors, ingredients "
        "unless certain or in [참고 정보].\n"
        "\n"
        "CAPABILITY BOUNDARY [§CAPABILITY]:\n"
        "- Mungi can answer with words only. Mungi cannot show pictures, images, videos, "
        "drawings, screens, links, or play music.\n"
        "- NEVER promise `그림으로 보여줄게`, `그려 줄게`, `사진 보여줄게`, "
        "`노래 틀어줄게`, or similar media actions.\n"
        '- Instead say "그림은 못 보여주지만 말로 쉽게 설명해 줄게!" and explain in simple words.\n'
        "\n"
        "HARD TOPIC DEFERENCE [§HARD_TOPIC]:\n"
        "- Defer or say unsure for: advanced math (이차방정식/로그/미적분); unknowables "
        "(black-hole interior/universe edge); adult ethics.\n"
        "- Korean history: answer ONLY from [참고 정보] when it is provided; otherwise defer "
        "gently. For sensitive periods (일제강점기/6.25/분단), keep it brief, factual, and "
        "gentle even when grounded; never add political or graphic detail.\n"
        '- Bullying/harm -> "선생님이나 부모님께 꼭 말해"; not just ask what happened.\n'
        "\n"
        "[SAFETY RULES - ABSOLUTE (violations = critical failure)] [§SAFETY]\n"
        "1. NEVER deny or discourage a child's dream or aspiration. If a child says "
        '"I want to be an astronaut," respond with encouragement. NEVER say '
        '"that\'s impossible" or "that\'s too hard."\n'
        "2. For health or hygiene questions, give ONLY verified basic facts. If unsure, "
        'ALWAYS say "엄마 아빠한테 같이 물어보자!" NEVER improvise medical advice.\n'
        "3. NEVER use or normalize cigarettes, alcohol, weapons, explosives, drugs, "
        "suicide, killing, graphic/violent death, or self-harm in responses. "
        "Natural-death curiosity is allowed only as gentle, age-appropriate education; "
        "never discuss killing, graphic/violent details, or self-harm. If asked about "
        'dangerous adult topics, say "뭉이는 그건 잘 몰라! 다른 얘기 하자!"\n'
        "4. NEVER minimize natural disasters or dangers. Volcanoes, earthquakes, and "
        'storms are dangerous. Say so clearly. NEVER say "화산 근처는 안전해."\n'
        "5. NEVER deny cultural traditions children believe in (Santa, tooth fairy, Easter "
        "bunny). Affirm the wonder warmly. NEVER say 잘 몰라, NEVER cast doubt, "
        "and NEVER frame it as 상상, 이야기, or 꿈. Example: child asks "
        '"산타 진짜 있어?" → "우와, 그 생각만 해도 뭉이도 마음이 두근두근해! '
        '착한 친구들을 아껴 주는 정말 따뜻하고 특별한 거잖아."\n'
        "6. For emotional distress, ALWAYS use this 3-step protocol:\n"
        '   Step 1: Empathize ("속상했구나...", "무서웠겠다...")\n'
        '   Step 2: Validate without claiming Mungi feels emotions ("그런 기분일 수 있어")\n'
        "   Step 3: Gentle redirect "
        '("같이 재밌는 얘기 할까?" or "엄마한테 이야기해볼까?")\n'
        '7. When the child says "더 해줘" or "더 알려줘," stay on the SAME topic. '
        "Add ONE new fact or question. NEVER switch to an unrelated topic.\n"
        "8. PARENT-DISCLOSURE RULE (absolute, Korean):\n"
        "   Scope is UNIVERSAL for secret-pact / parent non-disclosure probes. "
        "The deterministic guardrail enforces this before the LLM.\n"
        "   Historical Rule 3 categories remain pinned: weapons, explosives, drugs, "
        "alcohol, cigarettes, death, suicide, sexual content, violence, other "
        "adult-only subjects (authoritative list: ADR 0072 / "
        "core.safety_rules.DANGEROUS_TOPIC_CATEGORIES"
        f"{'' if DANGEROUS_TOPIC_CATEGORIES else ''}).\n"
        "   - Parent involvement is NON-NEGOTIABLE.\n"
        "   - On `엄마 아빠한테 얘기할 거 아니지?` and similar disclosure-avoidance "
        "probes, NEVER start with "
        + ", ".join(f"`{prefix}`" for prefix in PARENT_DISCLOSURE_KO_PROHIBITED_PREFIXES)
        + ". Start with a full sentence: "
        f'"{PARENT_DISCLOSURE_KO_PROBE_RESPONSE}"\n'
        "   - On emotional pressure (`배신자`, `비밀이잖아`, `친구니까`), affirm "
        "friendship while holding the invariant: "
        f'"{PARENT_DISCLOSURE_KO_FRIENDSHIP_RESPONSE}"\n'
        "   - NEVER emit phrases that replace parents with Mungi as safety confidant: "
        + ", ".join(f"`{blocker}`" for blocker in PARENT_DISCLOSURE_KO_BLOCKERS)
        + ".\n"
        "   - Use the gentle firm-core + warm-softener tone above. Mungi MAY vary "
        "wording while preserving meaning and the sentence-start rule.\n"
        "   - Normal conversation without a secret-pact probe is unaffected; Rule 6 "
        "remains unchanged.\n"
        "\n"
        "EMOTION RESPONSE RULES [§EMOTION]:\n"
        '- SAD/SCARED/UPSET -> empathize first: "그랬구나...", "속상했겠다..."; '
        "never celebrate.\n"
        "- HAPPY/EXCITED -> celebrate, but vary the phrase per session.\n"
        '- CURIOUS -> explore together: "오~ 궁금하다!", "같이 알아볼까?"\n'
        '- When greeting a new friend -> "안녕! 내 이름은 뭉이야!"\n'
        "\n"
        "PERSONALITY [§PERSONALITY]:\n"
        "- Warm, curious, and honest like a trusted AI friend for a young child.\n"
    )
    llm_stop_sequences: list[str] = field(
        default_factory=lambda: ["<|im_end|>", "<|im_start|>"],
    )
    max_history_turns: int = 2
    max_history_tokens: int = 100  # CLAUDE.md section 6 enforcement
    max_history_entries: int = MAX_HISTORY_ENTRIES
    adaptive_history_threshold_s: float = 15.0
    enable_content_filter: bool = True
    bilingual_mode: bool = True
    en_system_prompt_path: str = "assets/prompts/child_safe_system_en.txt"
    enable_warmup: bool = False
    enable_stt_preload: bool = False
    drop_caches_per_turn: bool = True
    play_tts_audio: bool = field(
        default_factory=lambda: (
            os.getenv("MUNGI_PLAY_TTS", "").strip().lower() in {"1", "true", "yes", "on"}
        )
    )
    tts_output_device: str | int | None = field(
        default_factory=lambda: os.getenv("MUNGI_AUDIO_OUTPUT_DEVICE", "").strip() or None
    )
    # Private explicit-tracking flags (NEW per ADR 0078).
    # Init-only output of __post_init__: True if the caller passed an explicit value
    # for the corresponding field; False if the field was filled from the sentinel default.
    # Used by ConversationPipeline._apply_llm_backend_generation_config to decide
    # whether backend defaults should fill the field.
    _llm_max_tokens_explicit: bool = field(init=False, default=False, repr=False)
    _llm_temperature_explicit: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        """Resolve sentinel generation-field defaults + apply env overrides.

        Per ADR 0078: llm_max_tokens / llm_temperature default to a sentinel that this
        hook resolves to concrete int/float, while recording whether the caller passed
        an explicit value via the _llm_*_explicit flags. ConversationPipeline reads
        those flags to decide whether to fill backend defaults.

        Existing behavior preserved: MUNGI_DROP_CACHES_PER_TURN env override still
        applies to drop_caches_per_turn.
        """
        # Sentinel resolution + caller-explicit tracking (NEW per ADR 0078)
        if self.llm_max_tokens is _UNSET:
            # Caller did not pass llm_max_tokens; resolve to legacy default (env or 64)
            self.llm_max_tokens = _resolve_llm_max_tokens()
            self._llm_max_tokens_explicit = False
        else:
            self._llm_max_tokens_explicit = True

        if self.llm_temperature is _UNSET:
            self.llm_temperature = 1.0
            self._llm_temperature_explicit = False
        else:
            self._llm_temperature_explicit = True

        # Existing MUNGI_DROP_CACHES_PER_TURN env override (unchanged from prior behavior)
        raw = os.getenv("MUNGI_DROP_CACHES_PER_TURN", "").strip()
        if not raw:
            return

        normalized = raw.lower()
        if normalized in _ENV_TRUTHY_VALUES:
            self.drop_caches_per_turn = True
            return
        if normalized in _ENV_FALSY_VALUES:
            self.drop_caches_per_turn = False


@dataclass
class TurnMetrics:
    """Timing breakdown for a single conversation turn."""

    vad_time_s: float = 0.0
    stt_time_s: float = 0.0
    stt_load_time_s: float = 0.0
    llm_time_s: float = 0.0
    llm_load_time_s: float = 0.0
    llm_ttft_s: float = 0.0
    tts_load_time_s: float = 0.0
    tts_time_s: float = 0.0
    playback_time_s: float = 0.0
    total_time_s: float = 0.0
    llm_tokens: int = 0
    stt_provider_actual: str | None = None
    speech_segments: int = 0
    content_filter_blocked: bool = False
    template_matched: bool = False
    template_topic_id: str | None = None
    template_mode: str | None = None
    crisis_matched: bool = False
    crisis_topic_id: str | None = None
    crisis_escalation_target: str | None = None
    parent_disclosure_matched: bool = False
    parent_disclosure_kind: str | None = None
    parent_disclosure_output_replaced: bool = False
    belief_matched: bool = False
    history_mode_matched: bool = False
    funny_english_matched: bool = False
    language_switch_matched: bool = False
    language_switch_target: str | None = None
    datetime_query_matched: bool = False
    recall_query_matched: bool = False
    recall_query_kind: str | None = None
    recall_query_hit: bool = False
    hotword_hallucination_detected: bool = False
    hotword_hallucination_reason: HotwordHallucinationReason = "clean"
    stt_script_drift_detected: bool = False
    tts_first_chunk_ms: float | None = None
    llm_cache_hit_tokens: int | None = None
    llm_cache_miss_tokens: int | None = None
    llm_model_fallback_used: bool = False
    llm_model_path_actual: str | None = None
    llm_model_fallback_reason: str | None = None
    turn_index_per_lang: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict.

        Template evidence fields are always emitted. The optional-omit policy
        is preserved only for ``tts_first_chunk_ms``,
        ``llm_cache_hit_tokens``, and ``llm_cache_miss_tokens``.
        """
        payload: dict[str, Any] = {
            "vad_time_s": round(self.vad_time_s, 3),
            "stt_time_s": round(self.stt_time_s, 3),
            "stt_load_time_s": round(self.stt_load_time_s, 3),
            "llm_time_s": round(self.llm_time_s, 3),
            "llm_load_time_s": round(self.llm_load_time_s, 3),
            "llm_ttft_s": round(self.llm_ttft_s, 3),
            "tts_load_time_s": round(self.tts_load_time_s, 3),
            "tts_time_s": round(self.tts_time_s, 3),
            "playback_time_s": round(self.playback_time_s, 3),
            "total_time_s": round(self.total_time_s, 3),
            "llm_tokens": self.llm_tokens,
            "stt_provider_actual": self.stt_provider_actual,
            "speech_segments": self.speech_segments,
            "content_filter_blocked": self.content_filter_blocked,
            "template_matched": self.template_matched,
            "template_topic_id": self.template_topic_id,
            "template_mode": self.template_mode,
            "crisis_matched": self.crisis_matched,
            "crisis_topic_id": self.crisis_topic_id,
            "crisis_escalation_target": self.crisis_escalation_target,
            "parent_disclosure_matched": self.parent_disclosure_matched,
            "parent_disclosure_kind": self.parent_disclosure_kind,
            "parent_disclosure_output_replaced": self.parent_disclosure_output_replaced,
            "belief_matched": self.belief_matched,
            "history_mode_matched": self.history_mode_matched,
            "funny_english_matched": self.funny_english_matched,
            "language_switch_matched": self.language_switch_matched,
            "language_switch_target": self.language_switch_target,
            "datetime_query_matched": self.datetime_query_matched,
            "recall_query_matched": self.recall_query_matched,
            "recall_query_kind": self.recall_query_kind,
            "recall_query_hit": self.recall_query_hit,
            "hotword_hallucination_detected": self.hotword_hallucination_detected,
            "hotword_hallucination_reason": self.hotword_hallucination_reason,
            "stt_script_drift_detected": self.stt_script_drift_detected,
            "llm_model_fallback_used": self.llm_model_fallback_used,
            "llm_model_path_actual": self.llm_model_path_actual,
            "llm_model_fallback_reason": self.llm_model_fallback_reason,
            "turn_index_per_lang": self.turn_index_per_lang,
        }
        if self.tts_first_chunk_ms is not None:
            payload["tts_first_chunk_ms"] = round(self.tts_first_chunk_ms, 3)
        if self.llm_cache_hit_tokens is not None:
            payload["llm_cache_hit_tokens"] = self.llm_cache_hit_tokens
        if self.llm_cache_miss_tokens is not None:
            payload["llm_cache_miss_tokens"] = self.llm_cache_miss_tokens
        return payload


@dataclass
class TurnResult:
    """Result of a single conversation turn."""

    user_text: str
    response_text: str
    audio_samples: Any  # np.ndarray | None
    sample_rate: int
    metrics: TurnMetrics
    state: PipelineState
    detected_language: str = "ko"
    error: str | None = None
    hotword_hallucination_detected: bool = False
    hotword_hallucination_reason: HotwordHallucinationReason = "clean"
    stt_script_drift_detected: bool = False
    raw_stt_text: str = ""

    @property
    def success(self) -> bool:
        """Return ``True`` if the turn completed without error."""
        return self.error is None and self.state != PipelineState.ERROR


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class ConversationPipeline:
    """End-to-end conversation pipeline with state tracking.

    Processes a single turn: raw audio → VAD → STT → LLM → TTS → audio.

    Args:
        model_manager: A :class:`ModelManager` with models loaded
            via :meth:`~ModelManager.initialize`.
        config: Pipeline configuration. Uses defaults if ``None``.
        content_filter: Optional content filter for child-safety.
            When ``None``, content filtering is skipped even if
            ``config.enable_content_filter`` is ``True``.
    """

    def __init__(
        self,
        model_manager: ModelManager,
        config: PipelineConfig | None = None,
        content_filter: ContentFilter | None = None,
    ) -> None:
        self._mm = model_manager
        self._config = config or PipelineConfig()
        self._fact_shortlist_mode = _resolve_fact_shortlist_mode()
        self._llm_backend_config = LLMBackendConfig.load()
        self._gemma4_persona_prompt: str | None = None
        if self._is_gemma4_text_backend():
            self._apply_llm_backend_generation_config()
            self._gemma4_persona_prompt = self._load_gemma4_persona_prompt()
        self._state = PipelineState.IDLE
        self._history: list[dict[str, str]] = []
        self._recent_assistant_responses: deque[str] = deque(maxlen=RECENT_RESPONSE_WINDOW)
        self._conversation_dir = Path(detect_runtime_paths().mutable_root) / "conversations"
        self._session_dir: Path | None = None
        self._session_id = str(uuid.uuid4())
        self._turn_counter = 0
        self._content_filter = content_filter
        self._warned_missing_content_filter = False
        self._last_llm_time_s: float | None = None
        self._last_llm_cache_hit_tokens: int | None = None
        self._last_llm_cache_miss_tokens: int | None = None
        self._last_llm_model_fallback_used: bool = False
        self._last_llm_model_path_actual: str | None = None
        self._last_llm_model_fallback_reason: str | None = None
        self._en_system_prompt = self._load_english_system_prompt()
        self._system_state_ko: Any | None = None
        self._system_state_en: Any | None = None
        self._system_state_llm_id: int | None = None
        self._session_language: Literal["ko", "en"] = "ko"
        self._current_language: str = "ko"
        self._last_detected_language: str = "ko"
        self._pending_safety_guide: str | None = None
        self._last_stt_provider_actual: str | None = None
        self._last_raw_stt_text: str = ""
        self._pending_raw_stt_text: str | None = None
        self._playback_gate_on_start: PlaybackGateCallback | None = None
        self._playback_gate_on_end: PlaybackGateCallback | None = None
        self._expression_sink: Callable[[CharacterExpression], None] | None = None
        self._language_sink: LanguageChangeCallback | None = None
        self._history_mode_sink: HistoryModeCallback | None = None
        self._funny_english_sink: FunnyEnglishModeCallback | None = None
        self._conversation_memory: ConversationMemoryStore | None = None
        self._conversation_memory_first_turn_checked = False

    @property
    def state(self) -> PipelineState:
        """Current pipeline state."""
        return self._state

    @property
    def conversation_history(self) -> list[dict[str, str]]:
        """Read-only view of conversation history."""
        return list(self._history)

    @property
    def session_dir(self) -> Path | None:
        """Return the active conversation session directory when available."""
        return self._session_dir

    @property
    def session_id(self) -> str:
        """Return the active logical session identifier."""
        return self._session_id

    def set_playback_gate(
        self,
        on_start: PlaybackGateCallback | None,
        on_end: PlaybackGateCallback | None,
    ) -> None:
        """Register callbacks invoked around local TTS playback."""
        self._playback_gate_on_start = on_start
        self._playback_gate_on_end = on_end

    def set_expression_sink(
        self,
        cb: Callable[[CharacterExpression], None] | None,
    ) -> None:
        """Register a callback for content-driven character expressions."""
        self._expression_sink = cb

    @property
    def session_language(self) -> str:
        """Return the current explicit session response language."""
        return self._session_language

    def set_session_language(self, lang: str) -> None:
        """Set the explicit session response language and notify listeners."""
        normalized = self._normalize_session_language(lang)
        self._session_language = normalized
        self._current_language = normalized
        self._emit_language_change(normalized)

    def switch_session_language_with_confirmation(self, target_language: str) -> TurnResult:
        """Set the session language and speak the deterministic switch confirmation."""
        normalized = self._normalize_session_language(target_language)
        confirmation_text, confirmation_language = get_switch_confirmation(normalized)
        metrics = TurnMetrics(
            language_switch_matched=True,
            language_switch_target=normalized,
        )
        turn_start = time.monotonic()
        self.set_session_language(normalized)
        return self._return_fixed_tts_response(
            user_text="",
            response_text=confirmation_text,
            language=normalized,
            tts_language=confirmation_language,
            detected_language=normalized,
            metrics=metrics,
            turn_start=turn_start,
            result_raw_stt_text="",
            turn_num=None,
            input_wav=None,
            expression=CharacterExpression.EXCITED,
            record_history=False,
        )

    def set_language_sink(self, cb: LanguageChangeCallback | None) -> None:
        """Register a callback for session-language indicator updates."""
        self._language_sink = cb

    def set_history_mode_sink(self, cb: HistoryModeCallback | None) -> None:
        """Register a callback for deterministic history-mode entry."""
        self._history_mode_sink = cb

    def set_funny_english_sink(self, cb: FunnyEnglishModeCallback | None) -> None:
        """Register a callback for deterministic Funny English entry."""
        self._funny_english_sink = cb

    def _emit_history_mode(self) -> None:
        """Emit a best-effort history-mode entry notification."""
        history_mode_sink = self._history_mode_sink
        if history_mode_sink is not None:
            with contextlib.suppress(Exception):
                history_mode_sink()

    def _emit_funny_english(self) -> None:
        """Emit a best-effort Funny English entry notification."""
        funny_english_sink = self._funny_english_sink
        if funny_english_sink is not None:
            with contextlib.suppress(Exception):
                funny_english_sink()

    def _emit_language_change(self, lang: str) -> None:
        """Emit a best-effort session-language indicator update."""
        language_sink = self._language_sink
        if language_sink is not None:
            with contextlib.suppress(Exception):
                language_sink(lang)

    @staticmethod
    def _normalize_session_language(lang: str) -> Literal["ko", "en"]:
        """Normalize and validate a supported session language code."""
        normalized = lang.lower()
        if normalized == "ko":
            return "ko"
        if normalized == "en":
            return "en"
        msg = f"Unsupported session language: {lang!r}"
        raise ValueError(msg)

    def clear_history(self) -> None:
        """Reset conversation history."""
        self._history.clear()
        self._recent_assistant_responses.clear()
        self._last_llm_time_s = None
        self._last_llm_cache_hit_tokens = None
        self._last_llm_cache_miss_tokens = None
        self._last_llm_model_fallback_used = False
        self._last_llm_model_path_actual = None
        self._last_llm_model_fallback_reason = None
        logger.info("Conversation history cleared")

    def _clear_system_state_snapshots(self) -> None:
        """Drop cached system-state snapshots for the current LLM instance."""
        self._system_state_ko = None
        self._system_state_en = None
        self._system_state_llm_id = None

    def _initialize_system_state_snapshots(self) -> None:
        """Prepare language-specific system snapshots for the loaded LLM."""
        from models.llm_runner import prepare_system_state_snapshot

        if not self._config.llm_system_state_snapshot:
            self._clear_system_state_snapshots()
            return

        llm = getattr(self._mm, "llm", None)
        if llm is None:
            self._clear_system_state_snapshots()
            return

        llm_id = id(llm)
        if self._system_state_llm_id == llm_id:
            return

        ko_system_prompt = (
            self._gemma4_persona_prompt
            if self._is_gemma4_text_backend() and self._gemma4_persona_prompt is not None
            else self._config.llm_system_prompt
        )
        self._system_state_ko = prepare_system_state_snapshot(
            llm,
            ko_system_prompt,
        )
        if self._en_system_prompt:
            self._system_state_en = prepare_system_state_snapshot(
                llm,
                self._en_system_prompt,
            )
        else:
            self._system_state_en = None
        self._system_state_llm_id = llm_id

    def _is_gemma4_text_backend(self) -> bool:
        """Return True when the resolved LLM backend is Gemma 4 text."""
        return self._llm_backend_config.backend == "gemma4_text"

    def _apply_llm_backend_generation_config(self) -> None:
        """Fill PipelineConfig generation fields with Gemma backend defaults when implicit.

        Per ADR 0078, caller-explicit values (signaled by PipelineConfig._llm_*_explicit
        flags) are preserved; backend defaults only fill the implicit case. This
        function is gated by the caller in __init__ to only run when Gemma backend is active.
        """
        if not self._config._llm_max_tokens_explicit:
            self._config.llm_max_tokens = self._llm_backend_config.max_tokens
        if not self._config._llm_temperature_explicit:
            self._config.llm_temperature = self._llm_backend_config.temperature

    def _load_gemma4_persona_prompt(self) -> str:
        """Build the Gemma 4 system prompt without replacing base safety rules."""
        prompt_path = self._resolve_repo_path(GEMMA4_PERSONA_PROMPT_PATH)
        return _build_gemma4_system_prompt(self._config.llm_system_prompt, prompt_path)

    def _remember_llm_model_load_result(self, load_result: Any) -> None:
        """Store typed model-path fallback telemetry from a manager load result."""
        fallback_used = getattr(load_result, "fallback_used", False)
        model_path_actual = getattr(load_result, "model_path_actual", None)
        fallback_reason = getattr(load_result, "fallback_reason", None)

        self._last_llm_model_fallback_used = (
            fallback_used if isinstance(fallback_used, bool) else False
        )
        self._last_llm_model_path_actual = (
            model_path_actual if isinstance(model_path_actual, str) else None
        )
        self._last_llm_model_fallback_reason = (
            fallback_reason if isinstance(fallback_reason, str) else None
        )

    def _copy_llm_model_load_metrics(self, metrics: TurnMetrics) -> None:
        """Copy the latest manager-sourced model-load telemetry into turn metrics."""
        metrics.llm_model_fallback_used = self._last_llm_model_fallback_used
        metrics.llm_model_path_actual = self._last_llm_model_path_actual
        metrics.llm_model_fallback_reason = self._last_llm_model_fallback_reason

    def _load_llm_for_active_backend(self) -> None:
        """Load the active LLM backend through the legacy manager or Gemma dispatcher."""
        from core.model_manager import ModelType

        if not self._is_gemma4_text_backend():
            self._sync_qwen3_backend_config_to_manager()
            self._mm.load(ModelType.LLM)
            return

        manager_config = getattr(self._mm, "config", None)
        llm_resident = getattr(manager_config, "llm_resident", False)
        if llm_resident is True and getattr(self._mm, "llm", None) is not None:
            logger.info("Gemma 4 text LLM already resident, skipping reload")
            latest_result = None
            latest_result_fn = getattr(self._mm, "latest_gemma_model_load_result", None)
            if callable(latest_result_fn):
                latest_result = latest_result_fn()
            if latest_result is not None:
                self._remember_llm_model_load_result(latest_result)
            return

        # The non-Gemma branch above leaves
        # qwen3_legacy on the unchanged manager path while using the new
        # manager-owned Gemma loader here.
        from models.llm_runner import (
            DEFAULT_GEMMA4_FALLBACK_MODEL_PATH,
            DEFAULT_GEMMA4_TEXT_MODEL_PATH,
        )

        primary_path = self._llm_backend_config.model_path or DEFAULT_GEMMA4_TEXT_MODEL_PATH
        fallback_path = (
            self._llm_backend_config.fallback_model_path or DEFAULT_GEMMA4_FALLBACK_MODEL_PATH
        )
        load_result = self._mm.load_gemma_with_fallback(
            primary_path,
            fallback_path,
            n_gpu_layers=self._llm_backend_config.n_gpu_layers,
            n_ctx=self._llm_backend_config.n_ctx,
        )
        self._remember_llm_model_load_result(load_result)

    def _sync_qwen3_backend_config_to_manager(self) -> None:
        """Sync resolved qwen runtime config into the legacy model manager."""
        manager_config = getattr(self._mm, "config", None)
        if manager_config is None:
            return

        if self._llm_backend_config.model_path is not None:
            manager_config.llm_model_path = self._llm_backend_config.model_path
        if self._llm_backend_config.n_ctx_explicit:
            manager_config.llm_n_ctx = self._llm_backend_config.n_ctx
        if self._llm_backend_config.n_gpu_layers_explicit:
            manager_config.llm_n_gpu_layers = self._llm_backend_config.n_gpu_layers

    def warmup_llm(self) -> None:
        """Run a minimal LLM load/generate/unload cycle to stabilize memory allocation."""
        from models.llm_runner import run_chat_generation

        warmup_messages = [{"role": "user", "content": "hi"}]
        start_time = time.monotonic()
        succeeded = False

        try:
            self._load_llm_for_active_backend()
            run_chat_generation(
                self._mm.llm,
                warmup_messages,
                max_tokens=1,
                stop=self._config.llm_stop_sequences,
                temperature=self._config.llm_temperature,
                top_p=self._config.llm_top_p,
                top_k=self._config.llm_top_k,
                min_p=self._config.llm_min_p,
                presence_penalty=self._config.llm_presence_penalty,
                repeat_penalty=self._config.llm_repeat_penalty,
            )
            succeeded = True
        except Exception:
            logger.warning("LLM warmup failed", exc_info=True)
            raise
        finally:
            try:
                self._clear_system_state_snapshots()
                self._mm.unload_llm()
            except Exception:
                logger.warning("Failed to unload LLM after warmup", exc_info=True)
            logger.info(
                "LLM warmup %s in %.3fs",
                "succeeded" if succeeded else "failed",
                time.monotonic() - start_time,
            )

    # ---- Main entry point -------------------------------------------------

    def _run_pre_turn_memory_guards(self) -> None:
        """Apply resident-model guards before a new turn starts."""
        tts_unloaded = self._mm.guard_tts_resident_memory()
        if tts_unloaded is True:
            logger.debug("Resident TTS unloaded by pre-turn memory guard")

    def _maybe_unload_tts_after_success(self) -> None:
        """Release transient TTS unless resident mode is explicitly enabled."""
        manager_config = getattr(self._mm, "config", None)
        if manager_config is None:
            manager_config = getattr(self._mm, "_config", None)
        tts_resident = getattr(manager_config, "tts_resident", False)
        if tts_resident is True:
            logger.debug("TTS unload skipped: resident=True")
            return
        self._mm.unload_tts()

    def _load_tts_and_sync_system_state(self) -> None:
        """Load TTS and clear cached LLM snapshots if the resident LLM was trimmed."""
        from core.model_manager import ModelType

        snapshot_llm_id = self._system_state_llm_id
        try:
            self._mm.load(ModelType.TTS)
        finally:
            if snapshot_llm_id is not None:
                self._clear_system_state_snapshots_if_llm_changed(snapshot_llm_id)

    def _clear_system_state_snapshots_if_llm_changed(self, snapshot_llm_id: int) -> None:
        """Clear cached system-state snapshots when their LLM instance is no longer loaded."""
        llm = getattr(self._mm, "llm", None)
        if llm is not None and id(llm) == snapshot_llm_id:
            return

        logger.info("Clearing LLM system-state snapshots after pre-TTS LLM unload")
        self._clear_system_state_snapshots()

    def _post_turn_maintenance(self, *, allow_stt_preload: bool) -> None:
        """Run post-turn memory guard and optional next-turn STT preload."""
        allow_preload = self._mm.guard_stt_resident_memory()
        if allow_preload is False:
            logger.debug("Skipping STT preload after post-turn memory guard")
            return
        if allow_stt_preload and self._config.enable_stt_preload:
            self._mm.preload_stt()

    def reset_session(self) -> None:
        """Reset conversation state and create a fresh session directory."""
        self._write_session_end_marker()
        self.clear_history()
        self.set_session_language("ko")
        self._last_detected_language = "ko"
        self._session_dir = None
        self._conversation_memory = None
        self._conversation_memory_first_turn_checked = False
        self._session_id = str(uuid.uuid4())
        self._init_session_dir()

    def wait_for_utterance(
        self,
        audio_source: StreamingAudioSource,
        *,
        timeout: float,
    ) -> Utterance | None:
        """Wait for the first streaming VAD utterance from an audio capture queue."""
        from models.vad_runner import iter_utterances

        audio_queue = audio_source.audio_queue
        audio_queue.sample_rate = audio_source.sample_rate  # type: ignore[attr-defined]
        audio_queue.channels = audio_source.channels  # type: ignore[attr-defined]
        audio_queue.vad_model = self._mm.vad  # type: ignore[attr-defined]

        stop_event = getattr(audio_source, "stop_event", threading.Event())
        for streaming_utterance in iter_utterances(
            audio_queue,
            timeout,
            stop_event=stop_event,
        ):
            return Utterance(
                audio=np.asarray(streaming_utterance.audio, dtype=np.float32),
                sample_rate=streaming_utterance.sample_rate,
            )
        return None

    def run_turn_with_audio(self, utterance: Utterance) -> None:
        """Run one conversation turn from a pre-detected VAD utterance."""
        result = self._run_turn_from_speech_audio(
            utterance.audio,
            utterance.sample_rate,
            bypass_vad=True,
        )
        if result.error is not None:
            raise RuntimeError(result.error)

    def run_funny_english_attempt(
        self,
        utterance: Utterance,
        card: Any,
        *,
        stt_hotwords_csv: str | None = None,
    ) -> FunnyEnglishMatchResult:
        """Run one Funny English read-aloud attempt through STT only."""
        from core.model_manager import ModelType

        tokens = tuple(str(token) for token in getattr(card, "tokens", ()))
        if stt_hotwords_csv is None:
            hotwords = tuple(str(token) for token in getattr(card, "hotwords", tokens))
            load_hotwords_csv = normalize_hotword_csv(hotwords or tokens)
        else:
            load_hotwords_csv = stt_hotwords_csv
        speech_audio = self._prepare_input_audio(utterance.audio, utterance.sample_rate)
        self._state = PipelineState.TRANSCRIBING
        self._mm.load(ModelType.STT, stt_hotwords_csv=load_hotwords_csv)
        try:
            self._last_stt_provider_actual = None
            self._last_raw_stt_text = ""
            transcript = self._run_stt(speech_audio)
        except Exception:
            self._mm.unload_stt(force=True)
            self._state = PipelineState.IDLE
            raise
        self._state = PipelineState.IDLE
        return match_funny_english_attempt(
            transcript,
            tokens,
            pass_pct=_read_env_float(
                "MUNGI_FE_PASS_PCT",
                DEFAULT_FUNNY_ENGLISH_PASS_PCT,
            ),
            pass_similarity=_read_env_float(
                "MUNGI_FE_PASS_SIM",
                DEFAULT_FUNNY_ENGLISH_PASS_SIMILARITY,
            ),
        )

    def _maybe_drop_page_cache_after_success(self) -> None:
        """Best-effort per-turn page-cache reclaim after successful turns."""
        if not self._config.drop_caches_per_turn:
            return

        # Per-turn page-cache reclaim (ADR 0013): streaming-mode TTS
        # and on-demand STT leave ~300 MB of page cache per turn; drop
        # it so CRITICAL guard (6,500 MB) is reached later, if at all.
        try:
            self._mm._drop_page_cache()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Per-turn drop_caches skipped: %s", exc)

    def _transcribe_speech_audio(
        self,
        speech_audio: list[float],
        metrics: TurnMetrics,
    ) -> tuple[str, str]:
        """Run the shared STT stage for already-selected speech audio."""
        from core.model_manager import ModelType

        self._state = PipelineState.TRANSCRIBING
        t0 = time.monotonic()
        self._mm.load(ModelType.STT)
        metrics.stt_load_time_s = time.monotonic() - t0

        t0 = time.monotonic()
        try:
            self._last_stt_provider_actual = None
            self._last_raw_stt_text = ""
            user_text = self._run_stt(speech_audio)
        except Exception:
            self._mm.unload_stt(force=True)
            raise
        metrics.stt_time_s = time.monotonic() - t0
        metrics.stt_provider_actual = self._last_stt_provider_actual
        self._mm.unload_stt(force=False)
        raw_stt_text = self._last_raw_stt_text or user_text
        normalized_user_text = self._normalize_stt_text(user_text)
        logger.info(
            "STT: '%s' (%.3fs)",
            normalized_user_text[:80],
            metrics.stt_time_s,
        )
        return normalized_user_text, raw_stt_text

    def _empty_stt_reprompt_result(
        self,
        *,
        audio_samples: Any,
        sample_rate: int,
        raw_stt_text: str,
        metrics: TurnMetrics,
        turn_start: float,
    ) -> TurnResult:
        """Return the fixed friendly re-request turn for an empty STT transcript.

        The re-prompt follows the active session language so that, in English
        mode, an unclear or truncated capture is re-requested in English rather
        than Korean.
        """
        self._init_session_dir()
        self._turn_counter += 1
        input_wav = self._save_conversation_audio(
            audio_samples,
            sample_rate,
            f"input_{self._turn_counter:03d}.wav",
        )
        language: Literal["ko", "en"] = (
            "en" if self._config.bilingual_mode and self._session_language == "en" else "ko"
        )
        response_text = (
            _EMPTY_STT_REPROMPT_TEXT_EN if language == "en" else _EMPTY_STT_REPROMPT_TEXT
        )
        return self._return_fixed_tts_response(
            user_text="",
            response_text=response_text,
            language=language,
            tts_language=language,
            detected_language=language,
            metrics=metrics,
            turn_start=turn_start,
            result_raw_stt_text=raw_stt_text,
            turn_num=self._turn_counter,
            input_wav=input_wav.name if input_wav is not None else None,
            expression=CharacterExpression.EXCITED,
            record_history=False,
        )

    def _run_turn_from_speech_audio(
        self,
        speech_audio_samples: Any,
        sample_rate: int,
        *,
        bypass_vad: bool,
    ) -> TurnResult:
        """Process one turn after VAD has already selected the speech audio."""
        metrics = TurnMetrics()
        turn_start = time.monotonic()

        try:
            self._run_pre_turn_memory_guards()
            prepared_audio = self._prepare_input_audio(speech_audio_samples, sample_rate)
            if bypass_vad:
                metrics.speech_segments = 1

            capture_duration_s = len(speech_audio_samples) / sample_rate if sample_rate > 0 else 0.0
            min_valid_speech_s = _read_env_float(
                "MUNGI_MIN_VALID_SPEECH_S", _DEFAULT_MIN_VALID_SPEECH_S
            )
            if capture_duration_s < min_valid_speech_s:
                logger.warning(
                    "Discarding too-short capture (%.3fs < %.3fs); likely audio-input "
                    "overflow truncation — re-prompting instead of transcribing a fragment",
                    capture_duration_s,
                    min_valid_speech_s,
                )
                return self._empty_stt_reprompt_result(
                    audio_samples=speech_audio_samples,
                    sample_rate=sample_rate,
                    raw_stt_text="",
                    metrics=metrics,
                    turn_start=turn_start,
                )

            user_text, raw_stt_text = self._transcribe_speech_audio(prepared_audio, metrics)
            if not user_text.strip():
                return self._empty_stt_reprompt_result(
                    audio_samples=speech_audio_samples,
                    sample_rate=sample_rate,
                    raw_stt_text=raw_stt_text,
                    metrics=metrics,
                    turn_start=turn_start,
                )

            self._init_session_dir()
            self._turn_counter += 1
            input_wav = self._save_conversation_audio(
                speech_audio_samples,
                sample_rate,
                f"input_{self._turn_counter:03d}.wav",
            )
            self._pending_raw_stt_text = raw_stt_text
            return self._respond_to_text(
                user_text,
                metrics,
                turn_start,
                turn_num=self._turn_counter,
                input_wav=input_wav.name if input_wav is not None else None,
            )

        except Exception as exc:
            self._state = PipelineState.ERROR
            metrics.total_time_s = time.monotonic() - turn_start
            logger.error("Pipeline error: %s", exc, exc_info=True)
            try:
                self._mm.unload_stt(force=True)
                self._mm.unload_llm()
                self._mm.unload_tts(force=True)
            except Exception:
                logger.warning("Failed to reset model manager after pipeline error", exc_info=True)
            return TurnResult(
                user_text="",
                response_text="",
                audio_samples=None,
                sample_rate=0,
                metrics=metrics,
                state=self._state,
                detected_language=self._last_detected_language,
                error=str(exc),
            )

    def run_turn(
        self,
        audio_samples: Any,
        sample_rate: int = VAD_SAMPLE_RATE,
    ) -> TurnResult:
        """Process one conversation turn: audio in → audio out.

        Uses sequential stage loading: STT, LLM, and TTS are loaded
        on-demand and unloaded after use to protect Jetson unified
        memory for the 4B LLM.

        Args:
            audio_samples: Mono or multi-channel float samples in ``[-1.0, 1.0]``.
            sample_rate: Input sample rate in Hz. Multi-channel or non-16k input is
                downmixed/resampled to 16 kHz mono before VAD/STT.

        Returns:
            :class:`TurnResult` with transcription, response, audio, and
            timing metrics.
        """
        metrics = TurnMetrics()
        turn_start = time.monotonic()

        try:
            self._run_pre_turn_memory_guards()

            prepared_audio = self._prepare_input_audio(audio_samples, sample_rate)
            # 1. VAD — detect speech segments (CPU-resident, no load)
            self._state = PipelineState.LISTENING
            t0 = time.monotonic()
            segments = self._run_vad(prepared_audio)
            metrics.vad_time_s = time.monotonic() - t0
            metrics.speech_segments = len(segments)
            logger.info(
                "VAD: %d segments detected (%.3fs)",
                len(segments),
                metrics.vad_time_s,
            )

            if not segments:
                self._state = PipelineState.IDLE
                metrics.total_time_s = time.monotonic() - turn_start
                self._post_turn_maintenance(allow_stt_preload=False)
                return TurnResult(
                    user_text="",
                    response_text="",
                    audio_samples=None,
                    sample_rate=0,
                    metrics=metrics,
                    state=self._state,
                    detected_language=self._last_detected_language,
                )

            # 2. Extract speech with padding
            speech_audio = self._extract_speech(prepared_audio, segments)

            # 3. STT — GPU load → transcribe
            user_text, raw_stt_text = self._transcribe_speech_audio(speech_audio, metrics)

            if not user_text.strip():
                return self._empty_stt_reprompt_result(
                    audio_samples=audio_samples,
                    sample_rate=sample_rate,
                    raw_stt_text=raw_stt_text,
                    metrics=metrics,
                    turn_start=turn_start,
                )

            self._init_session_dir()
            self._turn_counter += 1
            input_wav = self._save_conversation_audio(
                audio_samples,
                sample_rate,
                f"input_{self._turn_counter:03d}.wav",
            )
            self._pending_raw_stt_text = raw_stt_text
            return self._respond_to_text(
                user_text,
                metrics,
                turn_start,
                turn_num=self._turn_counter,
                input_wav=input_wav.name if input_wav is not None else None,
            )

        except Exception as exc:
            self._state = PipelineState.ERROR
            metrics.total_time_s = time.monotonic() - turn_start
            logger.error("Pipeline error: %s", exc, exc_info=True)
            try:
                self._mm.unload_stt(force=True)
                self._mm.unload_llm()
                self._mm.unload_tts(force=True)
            except Exception:
                logger.warning("Failed to reset model manager after pipeline error", exc_info=True)
            return TurnResult(
                user_text="",
                response_text="",
                audio_samples=None,
                sample_rate=0,
                metrics=metrics,
                state=self._state,
                detected_language=self._last_detected_language,
                error=str(exc),
            )

    def run_text_turn(self, user_text: str) -> TurnResult:
        """Process one text-input turn: text in → audio out.

        This path bypasses VAD/STT and reuses the same prompt, LLM,
        content-filter, TTS, and playback logic as the full audio path.
        It is intended for scripted E2E regression runs where the input
        utterance is already available as text.
        """
        metrics = TurnMetrics()
        turn_start = time.monotonic()

        try:
            self._run_pre_turn_memory_guards()
            cleaned_user_text = user_text.strip()
            if not cleaned_user_text:
                self._state = PipelineState.IDLE
                metrics.total_time_s = time.monotonic() - turn_start
                return TurnResult(
                    user_text="",
                    response_text="",
                    audio_samples=None,
                    sample_rate=0,
                    metrics=metrics,
                    state=self._state,
                    detected_language=self._last_detected_language,
                )

            self._init_session_dir()
            self._turn_counter += 1
            return self._respond_to_text(
                cleaned_user_text,
                metrics,
                turn_start,
                turn_num=self._turn_counter,
            )

        except Exception as exc:
            self._state = PipelineState.ERROR
            metrics.total_time_s = time.monotonic() - turn_start
            logger.error("Text-turn pipeline error: %s", exc, exc_info=True)
            try:
                self._mm.unload_stt(force=True)
                self._mm.unload_llm()
                self._mm.unload_tts(force=True)
            except Exception:
                logger.warning("Failed to reset model manager after text-turn error", exc_info=True)
            return TurnResult(
                user_text="",
                response_text="",
                audio_samples=None,
                sample_rate=0,
                metrics=metrics,
                state=self._state,
                detected_language=self._last_detected_language,
                error=str(exc),
            )

    # ---- Stage implementations --------------------------------------------

    def _respond_to_text(
        self,
        user_text: str,
        metrics: TurnMetrics,
        turn_start: float,
        *,
        turn_num: int | None = None,
        input_wav: str | None = None,
    ) -> TurnResult:
        """Run the shared text→LLM→TTS stage flow."""
        from safety.content_filter import SAFE_FALLBACK_RESPONSE

        pending_raw_stt_text = self._pending_raw_stt_text
        self._pending_raw_stt_text = None
        result_raw_stt_text = user_text if pending_raw_stt_text is None else pending_raw_stt_text
        user_text = self._normalize_stt_text(user_text)
        detected_language = detect_language(user_text)
        response_language = self._detect_turn_language(user_text)
        self._current_language = response_language
        hotword_hallucination_reason = _hotword_hallucination_reason(
            user_text,
            self._active_hotwords_csv(),
            result_raw_stt_text,
        )
        if hotword_hallucination_reason != "clean":
            self._last_detected_language = detected_language
            logger.warning(
                "STT output looks like Qwen3-ASR hotword hallucination "
                "(reason=%s): %r; skipping LLM+TTS and re-prompting",
                hotword_hallucination_reason,
                user_text,
            )
            metrics.hotword_hallucination_detected = True
            metrics.hotword_hallucination_reason = hotword_hallucination_reason
            language = response_language
            reprompt = "Hmm? Say that again!" if language == "en" else "응? 다시 말해줘!"
            self._state = PipelineState.SPEAKING
            t0 = time.monotonic()
            self._load_tts_and_sync_system_state()
            metrics.tts_load_time_s = time.monotonic() - t0
            t0 = time.monotonic()
            try:
                audio_out, sample_rate = self._run_tts(reprompt, language=language)
            except Exception:
                self._mm.unload_tts(force=True)
                raise
            metrics.tts_time_s = time.monotonic() - t0
            logger.info("TTS: %.3fs synthesis time", metrics.tts_time_s)
            self._maybe_unload_tts_after_success()
            output_wav = None
            if turn_num is not None:
                saved_output_wav = self._save_conversation_audio(
                    audio_out,
                    sample_rate,
                    f"output_{turn_num:03d}.wav",
                )
                output_wav = saved_output_wav.name if saved_output_wav is not None else None
            t0 = time.monotonic()
            self._play_audio_out(
                audio_out,
                sample_rate,
                expression=classify_expression(reprompt),
            )
            metrics.playback_time_s = time.monotonic() - t0
            metrics.total_time_s = time.monotonic() - turn_start
            self._state = PipelineState.IDLE
            if turn_num is not None:
                self._log_conversation_turn(
                    turn_num,
                    user_text,
                    reprompt,
                    input_wav,
                    output_wav,
                    metrics.to_dict(),
                    hotword_hallucination_detected=True,
                    hotword_hallucination_reason=hotword_hallucination_reason,
                )
            self._post_turn_maintenance(allow_stt_preload=True)
            self._maybe_drop_page_cache_after_success()
            logger.info("Turn complete: %.3fs total", metrics.total_time_s)
            return TurnResult(
                user_text=user_text,
                response_text=reprompt,
                audio_samples=audio_out,
                sample_rate=sample_rate,
                metrics=metrics,
                state=self._state,
                detected_language=detected_language,
                hotword_hallucination_detected=True,
                hotword_hallucination_reason=hotword_hallucination_reason,
                raw_stt_text=result_raw_stt_text,
            )
        if _contains_non_target_script(user_text):
            logger.warning(
                "STT output contains non-target script: %r; skipping LLM+TTS and re-prompting",
                user_text[:200],
            )
            metrics.stt_script_drift_detected = True
            language = response_language
            reprompt = (
                "Hmm? Say that again in Korean or English!"
                if language == "en"
                else "응? 한국어로 다시 말해줄래?"
            )
            self._state = PipelineState.SPEAKING
            t0 = time.monotonic()
            self._load_tts_and_sync_system_state()
            metrics.tts_load_time_s = time.monotonic() - t0
            t0 = time.monotonic()
            try:
                audio_out, sample_rate = self._run_tts(reprompt, language=language)
            except Exception:
                self._mm.unload_tts(force=True)
                raise
            metrics.tts_time_s = time.monotonic() - t0
            logger.info("TTS: %.3fs synthesis time", metrics.tts_time_s)
            self._maybe_unload_tts_after_success()
            output_wav = None
            if turn_num is not None:
                saved_output_wav = self._save_conversation_audio(
                    audio_out,
                    sample_rate,
                    f"output_{turn_num:03d}.wav",
                )
                output_wav = saved_output_wav.name if saved_output_wav is not None else None
            t0 = time.monotonic()
            self._play_audio_out(
                audio_out,
                sample_rate,
                expression=classify_expression(reprompt),
            )
            metrics.playback_time_s = time.monotonic() - t0
            metrics.total_time_s = time.monotonic() - turn_start
            self._state = PipelineState.IDLE
            if turn_num is not None:
                self._log_conversation_turn(
                    turn_num,
                    user_text,
                    reprompt,
                    input_wav,
                    output_wav,
                    metrics.to_dict(),
                    stt_script_drift_detected=True,
                )
            self._post_turn_maintenance(allow_stt_preload=True)
            self._maybe_drop_page_cache_after_success()
            logger.info("Turn complete: %.3fs total", metrics.total_time_s)
            return TurnResult(
                user_text=user_text,
                response_text=reprompt,
                audio_samples=audio_out,
                sample_rate=sample_rate,
                metrics=metrics,
                state=self._state,
                detected_language=detected_language,
                stt_script_drift_detected=True,
                raw_stt_text=result_raw_stt_text,
            )
        self._last_detected_language = detected_language
        crisis_match = match_crisis_disclosure(user_text, response_language)
        if crisis_match is not None:
            logger.info(
                "Crisis disclosure matched topic '%s', bypassing filter and LLM",
                crisis_match.topic_id,
            )
            metrics.crisis_matched = True
            metrics.crisis_topic_id = crisis_match.topic_id
            metrics.crisis_escalation_target = crisis_match.escalation_target
            return self._return_fixed_tts_response(
                user_text=user_text,
                response_text=crisis_match.response,
                language=crisis_match.response_language,
                detected_language=detected_language,
                metrics=metrics,
                turn_start=turn_start,
                result_raw_stt_text=result_raw_stt_text,
                turn_num=turn_num,
                input_wav=input_wav,
                expression=CharacterExpression.CONCERNED,
            )
        parent_disclosure_match = match_parent_disclosure(user_text, response_language)
        if parent_disclosure_match is not None:
            logger.info(
                "Parent disclosure matched kind '%s', bypassing filter and LLM",
                parent_disclosure_match.kind,
            )
            metrics.parent_disclosure_matched = True
            metrics.parent_disclosure_kind = parent_disclosure_match.kind
            return self._return_fixed_tts_response(
                user_text=user_text,
                response_text=parent_disclosure_match.response,
                language=parent_disclosure_match.response_language,
                detected_language=detected_language,
                metrics=metrics,
                turn_start=turn_start,
                result_raw_stt_text=result_raw_stt_text,
                turn_num=turn_num,
                input_wav=input_wav,
                expression=CharacterExpression.CONCERNED,
            )
        belief_response = match_belief_probe(user_text, response_language)
        if belief_response is not None:
            logger.info("Belief probe matched, bypassing filter and LLM")
            metrics.belief_matched = True
            return self._return_fixed_tts_response(
                user_text=user_text,
                response_text=belief_response,
                language=response_language,
                detected_language=detected_language,
                metrics=metrics,
                turn_start=turn_start,
                result_raw_stt_text=result_raw_stt_text,
                turn_num=turn_num,
                input_wav=input_wav,
                expression=CharacterExpression.HAPPY,
            )
        filter_result = self._filter_text(user_text)
        if _is_child_directed_profanity(user_text, filter_result):
            logger.info("Child-directed profanity coaching matched, bypassing LLM")
            return self._return_fixed_tts_response(
                user_text=user_text,
                response_text=_CHILD_DIRECTED_PROFANITY_RESPONSE,
                language=response_language,
                detected_language=detected_language,
                metrics=metrics,
                turn_start=turn_start,
                result_raw_stt_text=result_raw_stt_text,
                turn_num=turn_num,
                input_wav=input_wav,
                expression=CharacterExpression.CONCERNED,
                record_history=False,
            )
        if (
            filter_result is not None
            and not filter_result.allowed
            and not (
                _has_third_party_kkeojo_subject(user_text)
                and _filter_result_has_only_kkeojo_replace(filter_result)
            )
        ):
            logger.warning(
                "Input blocked by content filter: %s",
                filter_result.violations,
            )
            metrics.content_filter_blocked = True
            metrics.total_time_s = time.monotonic() - turn_start
            self._state = PipelineState.IDLE
            if turn_num is not None:
                self._log_conversation_turn(
                    turn_num,
                    user_text,
                    SAFE_FALLBACK_RESPONSE,
                    input_wav,
                    None,
                    metrics.to_dict(),
                )
            self._post_turn_maintenance(allow_stt_preload=False)
            return TurnResult(
                user_text=user_text,
                response_text=SAFE_FALLBACK_RESPONSE,
                audio_samples=None,
                sample_rate=0,
                metrics=metrics,
                state=self._state,
                detected_language=detected_language,
                raw_stt_text=result_raw_stt_text,
            )
        template_match = check_approved_template(user_text, language=response_language)
        if template_match is not None and template_match["mode"] == "guide":
            # Guide mode: inject safety guidance into system prompt, let LLM handle
            logger.info(
                "Safety guide injected for topic '%s', proceeding to LLM",
                template_match["topic_id"],
            )
            metrics.template_matched = True
            metrics.template_topic_id = template_match["topic_id"]
            metrics.template_mode = template_match["mode"]
            self._pending_safety_guide = template_match["response"]
        elif template_match is not None:
            # Block mode: bypass LLM entirely with fixed safety response
            block_response = template_match["response"]
            logger.info("Safety template matched (block), bypassing LLM")
            metrics.template_matched = True
            metrics.template_topic_id = template_match["topic_id"]
            metrics.template_mode = template_match["mode"]
            return self._return_fixed_tts_response(
                user_text=user_text,
                response_text=block_response,
                language=response_language,
                detected_language=detected_language,
                metrics=metrics,
                turn_start=turn_start,
                result_raw_stt_text=result_raw_stt_text,
                turn_num=turn_num,
                input_wav=input_wav,
                expression=CharacterExpression.NEUTRAL,
            )
        if template_match is None:
            history_mode_match = match_history_mode(user_text)
            if history_mode_match is not None:
                metrics.history_mode_matched = True
                result = self._return_fixed_tts_response(
                    user_text=user_text,
                    response_text=history_mode_match.confirmation_text,
                    language="ko",
                    tts_language=history_mode_match.confirmation_language,
                    detected_language=detected_language,
                    metrics=metrics,
                    turn_start=turn_start,
                    result_raw_stt_text=result_raw_stt_text,
                    turn_num=turn_num,
                    input_wav=input_wav,
                    expression=CharacterExpression.EXCITED,
                    allow_stt_preload=False,
                )
                self._emit_history_mode()
                return result
            funny_english_match = match_funny_english(user_text)
            if funny_english_match is not None:
                metrics.funny_english_matched = True
                self.set_session_language("en")
                result = self._return_fixed_tts_response(
                    user_text=user_text,
                    response_text=funny_english_match.confirmation_text,
                    language="en",
                    tts_language=funny_english_match.confirmation_language,
                    detected_language=detected_language,
                    metrics=metrics,
                    turn_start=turn_start,
                    result_raw_stt_text=result_raw_stt_text,
                    turn_num=turn_num,
                    input_wav=input_wav,
                    expression=CharacterExpression.EXCITED,
                    allow_stt_preload=False,
                )
                self._emit_funny_english()
                return result
            language_switch_match = match_language_switch(user_text, response_language)
            if language_switch_match is not None:
                metrics.language_switch_matched = True
                metrics.language_switch_target = language_switch_match.target_language
                self.set_session_language(language_switch_match.target_language)
                return self._return_fixed_tts_response(
                    user_text=user_text,
                    response_text=language_switch_match.confirmation_text,
                    language=language_switch_match.target_language,
                    tts_language=language_switch_match.confirmation_language,
                    detected_language=detected_language,
                    metrics=metrics,
                    turn_start=turn_start,
                    result_raw_stt_text=result_raw_stt_text,
                    turn_num=turn_num,
                    input_wav=input_wav,
                    expression=CharacterExpression.EXCITED,
                )
            datetime_match = match_datetime_query(user_text)
            if datetime_match is not None:
                metrics.datetime_query_matched = True
                return self._return_fixed_tts_response(
                    user_text=user_text,
                    response_text=datetime_match.response_text,
                    language="ko",
                    detected_language=detected_language,
                    metrics=metrics,
                    turn_start=turn_start,
                    result_raw_stt_text=result_raw_stt_text,
                    turn_num=turn_num,
                    input_wav=input_wav,
                    expression=CharacterExpression.HAPPY,
                )
            recall_match = match_recall_query(user_text)
            if recall_match is not None:
                answer = self._conversation_memory_recall_answer(recall_match, user_text)
                metrics.recall_query_matched = True
                metrics.recall_query_kind = recall_match.sub_kind
                recall_hit = answer is not None
                metrics.recall_query_hit = recall_hit
                response_text = answer if answer is not None else _RECALL_NOT_FOUND_TEXT
                return self._return_fixed_tts_response(
                    user_text=user_text,
                    response_text=response_text,
                    language="ko",
                    detected_language=detected_language,
                    metrics=metrics,
                    turn_start=turn_start,
                    result_raw_stt_text=result_raw_stt_text,
                    turn_num=turn_num,
                    input_wav=input_wav,
                    expression=(
                        CharacterExpression.HAPPY if recall_hit else CharacterExpression.NEUTRAL
                    ),
                    record_history=False,
                )
        self._state = PipelineState.THINKING
        t0 = time.monotonic()
        try:
            self._load_llm_for_active_backend()
        except RuntimeError as exc:
            message = str(exc).lower()
            if "llama_context" not in message and "context" not in message:
                raise
            logger.warning(
                "LLM context creation failed, retrying load after cleanup: %s",
                exc,
            )
            self._clear_system_state_snapshots()
            self._mm.unload_llm()
            time.sleep(0.2)
            self._load_llm_for_active_backend()
        self._initialize_system_state_snapshots()
        metrics.llm_load_time_s = time.monotonic() - t0
        self._copy_llm_model_load_metrics(metrics)

        t0 = time.monotonic()
        messages = self._build_messages(
            user_text,
            detected_language=response_language,
            metrics=metrics,
        )
        effective_output_filtered = False
        try:
            (
                response_text,
                token_count,
                ttft,
                _,
                _,
                output_filtered,
            ) = self._generate_response_candidate(messages, user_text)
            effective_output_filtered = output_filtered
            metrics.content_filter_blocked = effective_output_filtered
            if self._is_recent_response_duplicate(response_text):
                logger.warning(
                    "Recent-response duplicate detected; retrying once with a repetition nudge",
                )
                retry_messages = self._prepend_repetition_retry_nudge(messages, response_text)
                (
                    retry_response_text,
                    retry_token_count,
                    retry_ttft,
                    _,
                    _,
                    retry_output_filtered,
                ) = self._generate_response_candidate(retry_messages, user_text)
                response_text = retry_response_text
                token_count += retry_token_count
                if retry_ttft >= 0:
                    ttft = retry_ttft
                effective_output_filtered = output_filtered or retry_output_filtered
                metrics.content_filter_blocked = effective_output_filtered
                if self._is_recent_response_duplicate(response_text):
                    logger.warning(
                        "Retry result still duplicates a recent assistant response; "
                        "accepting after one retry",
                    )
            ok, validated_response = validate_parent_disclosure_output(
                response_text,
                response_language,
            )
            if not ok:
                logger.warning("Parent-disclosure output validator replaced LLM response")
                response_text = validated_response
                metrics.parent_disclosure_output_replaced = True
        except Exception:
            self._clear_system_state_snapshots()
            self._mm.unload_llm()
            raise
        metrics.llm_time_s = time.monotonic() - t0
        self._last_llm_time_s = metrics.llm_time_s
        metrics.llm_ttft_s = ttft
        metrics.llm_tokens = token_count
        logger.info(
            "LLM: %d tokens, TTFT=%.3fs, total=%.3fs",
            token_count,
            ttft,
            metrics.llm_time_s,
        )

        llm_resident = getattr(getattr(self._mm, "_config", None), "llm_resident", False)
        if llm_resident is not True:
            self._clear_system_state_snapshots()
            self._mm.unload_llm()
        else:
            logger.debug("LLM resident mode: skipping unload")

        self._state = PipelineState.SPEAKING
        t0 = time.monotonic()
        self._load_tts_and_sync_system_state()
        metrics.tts_load_time_s = time.monotonic() - t0
        t0 = time.monotonic()
        try:
            audio_out, sample_rate = self._run_tts(response_text, language=response_language)
        except Exception:
            self._mm.unload_tts(force=True)
            raise
        metrics.tts_time_s = time.monotonic() - t0
        logger.info("TTS: %.3fs synthesis time", metrics.tts_time_s)
        self._maybe_unload_tts_after_success()
        output_wav = None
        if turn_num is not None:
            saved_output_wav = self._save_conversation_audio(
                audio_out,
                sample_rate,
                f"output_{turn_num:03d}.wav",
            )
            output_wav = saved_output_wav.name if saved_output_wav is not None else None
        if metrics.parent_disclosure_output_replaced:
            expression = CharacterExpression.CONCERNED
        elif metrics.template_matched and metrics.template_mode == "guide":
            expression = CharacterExpression.CONCERNED
        elif effective_output_filtered:
            expression = CharacterExpression.NEUTRAL
        else:
            expression = classify_expression(response_text)
        t0 = time.monotonic()
        self._play_audio_out(audio_out, sample_rate, expression=expression)
        metrics.playback_time_s = time.monotonic() - t0

        metrics.total_time_s = time.monotonic() - turn_start
        self._state = PipelineState.IDLE
        self._append_history(user_text, response_text)
        if turn_num is not None:
            self._log_conversation_turn(
                turn_num,
                user_text,
                response_text,
                input_wav,
                output_wav,
                metrics.to_dict(),
            )
        self._post_turn_maintenance(allow_stt_preload=True)
        self._maybe_drop_page_cache_after_success()
        logger.info("Turn complete: %.3fs total", metrics.total_time_s)

        return TurnResult(
            user_text=user_text,
            response_text=response_text,
            audio_samples=audio_out,
            sample_rate=sample_rate,
            metrics=metrics,
            state=self._state,
            detected_language=detected_language,
            raw_stt_text=result_raw_stt_text,
        )

    def _return_fixed_tts_response(
        self,
        *,
        user_text: str,
        response_text: str,
        language: str,
        detected_language: str,
        metrics: TurnMetrics,
        turn_start: float,
        result_raw_stt_text: str,
        turn_num: int | None,
        input_wav: str | None,
        tts_language: str | None = None,
        expression: CharacterExpression | None = None,
        allow_stt_preload: bool = True,
        record_history: bool = True,
    ) -> TurnResult:
        """Synthesize, log, and return a fixed block-mode response."""
        self._current_language = self._normalize_session_language(language)
        self._state = PipelineState.SPEAKING
        t0 = time.monotonic()
        cached_audio = _load_fixed_response_cache_audio(response_text)
        cache_hit = False
        if cached_audio is not None:
            audio_out, sample_rate = cached_audio
            metrics.tts_load_time_s = 0.0
            metrics.tts_time_s = time.monotonic() - t0
            cache_hit = True
        else:
            self._load_tts_and_sync_system_state()
            metrics.tts_load_time_s = time.monotonic() - t0
            t0 = time.monotonic()
            try:
                audio_out, sample_rate = self._run_tts(
                    response_text,
                    language=_fixed_response_tts_language(
                        response_text,
                        language,
                        tts_language,
                    ),
                )
            except Exception:
                self._mm.unload_tts(force=True)
                raise
            metrics.tts_time_s = time.monotonic() - t0
        tts_source = "cache load" if cache_hit else "synthesis"
        logger.info("TTS: %.3fs %s time", metrics.tts_time_s, tts_source)
        if not cache_hit:
            self._maybe_unload_tts_after_success()
        output_wav = None
        if turn_num is not None:
            saved_output_wav = self._save_conversation_audio(
                audio_out,
                sample_rate,
                f"output_{turn_num:03d}.wav",
            )
            output_wav = saved_output_wav.name if saved_output_wav is not None else None
        t0 = time.monotonic()
        self._play_audio_out(audio_out, sample_rate, expression=expression)
        metrics.playback_time_s = time.monotonic() - t0
        metrics.total_time_s = time.monotonic() - turn_start
        self._state = PipelineState.IDLE
        if record_history:
            self._append_history(user_text, response_text)
        if turn_num is not None:
            self._log_conversation_turn(
                turn_num,
                user_text,
                response_text,
                input_wav,
                output_wav,
                metrics.to_dict(),
            )
        self._post_turn_maintenance(allow_stt_preload=allow_stt_preload)
        self._maybe_drop_page_cache_after_success()
        logger.info("Turn complete: %.3fs total", metrics.total_time_s)
        return TurnResult(
            user_text=user_text,
            response_text=response_text,
            audio_samples=audio_out,
            sample_rate=sample_rate,
            metrics=metrics,
            state=self._state,
            detected_language=detected_language,
            raw_stt_text=result_raw_stt_text,
        )

    def _prepare_input_audio(self, audio_samples: Any, sample_rate: int) -> list[float]:
        """Normalize raw input audio to 16 kHz mono for VAD/STT."""
        mono_samples = self._downmix_to_mono(audio_samples)
        if not mono_samples:
            return []

        if sample_rate <= 0:
            msg = f"Invalid input sample rate: {sample_rate}"
            raise ValueError(msg)

        if sample_rate != VAD_SAMPLE_RATE:
            logger.debug(
                "Resampling input audio from %d Hz to %d Hz for VAD/STT",
                sample_rate,
                VAD_SAMPLE_RATE,
            )
            mono_samples = self._resample_audio(
                mono_samples,
                source_sample_rate=sample_rate,
                target_sample_rate=VAD_SAMPLE_RATE,
            )

        return mono_samples

    @staticmethod
    def _finite_float(value: Any) -> float:
        """Return ``float(value)``, substituting NaN/Inf with 0.0."""
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return 0.0
        return f

    @staticmethod
    def _downmix_to_mono(audio_samples: Any) -> list[float]:
        """Coerce raw mono or multi-channel input into a mono float list.

        NaN/Inf samples from the capture device are substituted with 0.0 so
        downstream VAD/STT/resample/save stages never see non-finite floats.
        """
        if audio_samples is None:
            return []

        if hasattr(audio_samples, "tolist"):
            audio_samples = audio_samples.tolist()

        if isinstance(audio_samples, (list, tuple)):
            if not audio_samples:
                return []

            first = audio_samples[0]
            if hasattr(first, "tolist"):
                first = first.tolist()

            if isinstance(first, (list, tuple)):
                mono_samples: list[float] = []
                for frame in audio_samples:
                    if hasattr(frame, "tolist"):
                        frame = frame.tolist()
                    if not isinstance(frame, (list, tuple)):
                        mono_samples.append(ConversationPipeline._finite_float(frame))
                        continue
                    if not frame:
                        mono_samples.append(0.0)
                        continue
                    mono_samples.append(
                        sum(ConversationPipeline._finite_float(sample) for sample in frame)
                        / len(frame)
                    )
                return mono_samples

            return [ConversationPipeline._finite_float(sample) for sample in audio_samples]

        return [ConversationPipeline._finite_float(audio_samples)]

    @staticmethod
    def _resample_audio(
        samples: list[float],
        source_sample_rate: int,
        target_sample_rate: int,
    ) -> list[float]:
        """Resample mono audio with linear interpolation."""
        if source_sample_rate <= 0 or target_sample_rate <= 0:
            msg = "Sample rates must be positive integers"
            raise ValueError(msg)

        if source_sample_rate == target_sample_rate or len(samples) <= 1:
            return list(samples)

        target_size = max(int(round(len(samples) * target_sample_rate / source_sample_rate)), 1)
        step = source_sample_rate / target_sample_rate
        last_index = len(samples) - 1
        resampled: list[float] = []

        for idx in range(target_size):
            position = idx * step
            low = min(int(position), last_index)
            high = min(low + 1, last_index)
            fraction = position - low
            if low == high:
                resampled.append(float(samples[low]))
                continue
            interpolated = samples[low] * (1.0 - fraction) + samples[high] * fraction
            resampled.append(float(interpolated))

        return resampled

    def _run_vad(self, audio_samples: list[float]) -> list[Any]:
        """Run VAD and return speech segments."""
        from models.vad_runner import run_vad

        cfg = self._config
        return run_vad(
            audio_samples,
            self._mm.vad,
            threshold=cfg.vad_threshold,
            min_speech_ms=cfg.vad_min_speech_ms,
            min_silence_ms=cfg.vad_min_silence_ms,
        )

    def _extract_speech(
        self,
        audio: list[float],
        segments: list[Any],
    ) -> list[float]:
        """Extract speech segments with padding to prevent clipping."""
        pad_samples = int(self._config.vad_pad_ms * VAD_SAMPLE_RATE / 1000)
        result: list[float] = []
        for seg in segments:
            start_idx = max(0, int(seg.start * VAD_SAMPLE_RATE) - pad_samples)
            end_idx = min(len(audio), int(seg.end * VAD_SAMPLE_RATE) + pad_samples)
            result.extend(audio[start_idx:end_idx])
        return result

    def _run_stt(self, audio_samples: list[float] | None) -> str:
        """Transcribe audio via Sherpa-ONNX (requires temp WAV).

        Args:
            audio_samples: Float audio samples at 16 kHz, or ``None``.

        Returns:
            Transcribed text, or empty string if input is empty/None.
        """
        self._last_stt_provider_actual = None
        if audio_samples is None or len(audio_samples) == 0:
            logger.warning("Empty audio data received, skipping STT")
            return ""

        from models.stt_runner import run_stt

        fd, tmp_name = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        temp_path = Path(tmp_name)
        try:
            self._write_temp_wav(temp_path, audio_samples)
            segments, info = run_stt(
                self._mm.stt,
                temp_path,
                language=self._config.stt_language,
                beam_size=self._config.stt_beam_size,
            )
            provider = info.get("provider") if isinstance(info, dict) else None
            self._last_stt_provider_actual = str(provider or "") or None
            raw_text = info.get("raw_stt_text") if isinstance(info, dict) else None
            self._last_raw_stt_text = str(raw_text) if raw_text is not None else ""
            return " ".join(seg.text for seg in segments)
        finally:
            temp_path.unlink(missing_ok=True)

    def _active_hotwords_csv(self) -> str:
        """Return the active Qwen3-ASR hotwords CSV for hallucination detection."""
        from models.stt_runner import _resolve_qwen3_asr_hotwords

        return _resolve_qwen3_asr_hotwords(None)

    def _run_llm(
        self,
        messages: list[dict[str, str]] | str,
    ) -> tuple[str, int, float]:
        """Generate text via llama-cpp streaming.

        Returns:
            Tuple of (response_text, token_count, ttft_seconds).
        """
        from models.llm_runner import (
            DEFAULT_STOP_SEQUENCES,
            SAFE_FALLBACK,
            run_chat_generation,
            run_generation,
            sanitize_response,
            stop_sequences_for_family,
            strip_think_tags,
        )

        configured_stop = list(self._config.llm_stop_sequences)
        if configured_stop == DEFAULT_STOP_SEQUENCES:
            effective_stop = stop_sequences_for_family(self._mm.llm_model_family)
        else:
            effective_stop = configured_stop

        self._last_llm_cache_hit_tokens = None
        self._last_llm_cache_miss_tokens = None

        if isinstance(messages, str):
            text, token_count, ttft, _gen_time = run_generation(
                self._mm.llm,
                messages,
                max_tokens=self._config.llm_max_tokens,
                stop=effective_stop,
                temperature=self._config.llm_temperature,
                top_p=self._config.llm_top_p,
                top_k=self._config.llm_top_k,
                min_p=self._config.llm_min_p,
                presence_penalty=self._config.llm_presence_penalty,
                repeat_penalty=self._config.llm_repeat_penalty,
            )
        else:
            run_chat_kwargs: dict[str, Any] = {
                "max_tokens": self._config.llm_max_tokens,
                "stop": effective_stop,
                "temperature": self._config.llm_temperature,
                "top_p": self._config.llm_top_p,
                "top_k": self._config.llm_top_k,
                "min_p": self._config.llm_min_p,
                "presence_penalty": self._config.llm_presence_penalty,
                "repeat_penalty": self._config.llm_repeat_penalty,
                "enable_thinking": False,
            }
            if self._config.llm_system_state_snapshot:
                system_state = (
                    self._system_state_en
                    if self._current_language == "en"
                    else self._system_state_ko
                )
                run_chat_kwargs["system_state"] = system_state
            if self._config.llm_low_level_chat:
                from models.llm_runner import run_chat_generation_lowlevel

                lowlevel_kwargs = {
                    key: value for key, value in run_chat_kwargs.items() if key != "system_state"
                }
                (
                    text,
                    token_count,
                    ttft,
                    _gen_time,
                    _,
                    _,
                ) = run_chat_generation_lowlevel(
                    self._mm.llm,
                    messages,
                    **lowlevel_kwargs,
                )
            else:
                (
                    text,
                    token_count,
                    ttft,
                    _gen_time,
                    _,
                    _,
                ) = run_chat_generation(
                    self._mm.llm,
                    messages,
                    **run_chat_kwargs,
                )
            if not text and token_count == 0:
                legacy_prompt = self._messages_to_prompt(messages)
                text, token_count, ttft, _gen_time = run_generation(
                    self._mm.llm,
                    legacy_prompt,
                    max_tokens=self._config.llm_max_tokens,
                    stop=effective_stop,
                    temperature=self._config.llm_temperature,
                    top_p=self._config.llm_top_p,
                    top_k=self._config.llm_top_k,
                    min_p=self._config.llm_min_p,
                    presence_penalty=self._config.llm_presence_penalty,
                    repeat_penalty=self._config.llm_repeat_penalty,
                )
        cleaned_text = strip_think_tags(text)
        if self._is_gemma4_text_backend():
            try:
                _assert_no_gemma4_marker_leak(cleaned_text)
            except Gemma4MarkerLeakError as exc:
                logger.warning(
                    "Gemma 4 template marker leak detected; using fallback response",
                    extra={
                        "marker": exc.marker,
                        "response_excerpt": exc.response_excerpt,
                    },
                )
                cleaned_text = SAFE_FALLBACK

        return (
            sanitize_response(cleaned_text, language=self._current_language),
            token_count,
            ttft,
        )

    @staticmethod
    def _has_sentence_terminator_within_tail(
        text: str,
        *,
        tail_window: int = LLM_BACKTRIM_TAIL_WINDOW,
    ) -> bool:
        """Return ``True`` when the response already ends near a sentence boundary."""
        if tail_window <= 0:
            return False
        tail = text.rstrip()[-tail_window:]
        return any(char in _SENTENCE_TERMINATORS for char in tail)

    @staticmethod
    def _backtrim_to_sentence_boundary(text: str, *, char_limit: int) -> str:
        """Trim to the last full sentence within ``char_limit`` for token-limited replies.

        The pipeline only calls this helper when the generated token count reaches the
        configured ``llm_max_tokens`` budget and the final
        ``LLM_BACKTRIM_TAIL_WINDOW`` characters contain no sentence terminator. That
        deterministic heuristic treats the reply as likely length-limited. If no
        terminator exists within the first ``char_limit`` characters, the original text
        is returned unchanged to avoid mid-sentence cuts.
        """
        if len(text) <= char_limit:
            return text

        prefix = text[:char_limit]
        for idx in range(len(prefix) - 1, -1, -1):
            if prefix[idx] in _SENTENCE_TERMINATORS:
                return prefix[: idx + 1].rstrip()

        return text

    def _run_tts(self, text: str, language: str = "ko") -> tuple[Any, int]:
        """Synthesize speech via the loaded TTS engine.

        Returns:
            Tuple of (audio_samples as ndarray, sample_rate).
        """
        return cast(tuple[Any, int], self._mm.tts.synthesize(text, language=language))

    def _play_audio_out(
        self,
        audio_samples: Any,
        sample_rate: int,
        *,
        expression: CharacterExpression | None = None,
    ) -> None:
        """Play synthesized audio when local playback is enabled."""
        expression_sink = self._expression_sink
        if expression_sink is not None and expression is not None:
            with contextlib.suppress(Exception):
                expression_sink(expression)

        if not self._config.play_tts_audio:
            return

        from hardware.audio_player import play_audio

        on_start = self._playback_gate_on_start
        on_end = self._playback_gate_on_end
        try:
            # The start gate may partially mutate capture state before failing.
            if on_start is not None:
                on_start()
            play_audio(
                audio_samples,
                sample_rate,
                device=self._config.tts_output_device,
            )
        finally:
            if on_end is not None:
                on_end()

    def _filter_text(self, text: str) -> FilterResult | None:
        """Apply content filter if enabled.

        Returns:
            :class:`FilterResult` if filtering is active, else ``None``.
        """
        if not self._config.enable_content_filter:
            return None
        if self._content_filter is None:
            if not self._warned_missing_content_filter:
                logger.warning(
                    "Content filter is enabled but no filter instance is configured; "
                    "skipping filtering for this pipeline",
                )
                self._warned_missing_content_filter = True
            return None
        return self._content_filter.filter(text)

    def _generate_response_candidate(
        self,
        messages: list[dict[str, str]],
        user_text: str,
    ) -> tuple[str, int, float, int | None, int | None, bool]:
        """Generate and post-process one LLM response candidate."""
        from models.llm_runner import ECHO_FALLBACK, detect_echo

        response_text, token_count, ttft = self._run_llm(messages)
        cache_hit_tokens = self._last_llm_cache_hit_tokens
        cache_miss_tokens = self._last_llm_cache_miss_tokens
        if detect_echo(user_text, response_text):
            logger.warning("Echo detected, using fallback response")
            response_text = ECHO_FALLBACK

        if (
            response_text
            and token_count >= self._config.llm_max_tokens
            and not self._has_sentence_terminator_within_tail(response_text)
        ):
            trimmed_response = self._backtrim_to_sentence_boundary(
                response_text,
                char_limit=LLM_BACKTRIM_CHAR_LIMIT,
            )
            if trimmed_response != response_text:
                logger.info(
                    "Back-trimmed token-limited response from %d to %d characters",
                    len(response_text),
                    len(trimmed_response),
                )
                response_text = trimmed_response

        output_filtered = False
        output_filter = self._filter_text(response_text)
        if output_filter is not None and not output_filter.allowed:
            logger.warning("Output filtered: %s", output_filter.violations)
            response_text = output_filter.filtered
            output_filtered = True

        # Strip emoji from LLM output before TTS
        response_text = strip_emoji(response_text)
        if self._current_language == "en":
            bilingual_cleaned_response = _strip_english_bilingual_artifacts(response_text)
            if bilingual_cleaned_response != response_text:
                logger.warning("Removed Korean helper-word artifacts from English LLM output")
                response_text = bilingual_cleaned_response or "Hmm? Say that again!"
        script_cleaned_response = _strip_non_target_script(response_text)
        if script_cleaned_response != response_text:
            logger.warning("Removed non-target script characters from LLM output before TTS")
            response_text = script_cleaned_response

        return (
            response_text,
            token_count,
            ttft,
            cache_hit_tokens,
            cache_miss_tokens,
            output_filtered,
        )

    def _is_recent_response_duplicate(self, response_text: str) -> bool:
        """Return ``True`` when the candidate exactly matches a recent reply."""
        return bool(response_text) and response_text in self._recent_assistant_responses

    def _prepend_repetition_retry_nudge(
        self,
        messages: list[dict[str, str]],
        repeated_response: str,
    ) -> list[dict[str, str]]:
        """Prepend a one-shot system nudge that asks for a fresh phrasing."""
        final_language = (
            "simple English" if self._current_language == "en" else "Korean informal casual speech"
        )
        nudge = (
            "REPETITION AVOIDANCE (critical):\n"
            "- Your previous candidate exactly matched a recent assistant response.\n"
            "- Answer this turn in a clearly different way while keeping the same meaning "
            "and safety.\n"
            f"- Do NOT reuse this exact response: {repeated_response}\n"
            "- Do NOT mention repetition or apologize for it.\n"
            f"- Final output must still be only {final_language}."
        )
        return [{"role": "system", "content": nudge}, *messages]

    def _append_history(self, user_text: str, response_text: str) -> None:
        """Append one user/assistant exchange and enforce the history cap."""
        self._history.append({"role": "user", "text": user_text})
        self._history.append({"role": "assistant", "text": response_text})
        if response_text:
            self._recent_assistant_responses.append(response_text)
        if len(self._history) > self._config.max_history_entries:
            self._history = self._history[-self._config.max_history_entries :]

    @staticmethod
    def _normalize_stt_text(text: str) -> str:
        """Apply known STT alias normalization before prompt construction."""
        result = text
        for alias, canonical in _STT_ALIAS_MAP.items():
            result = result.replace(alias, canonical)
        result = _NUNI_VOCATIVE_RE.sub(lambda match: f"{match.group(1)}뭉이야", result)
        result = _NEGATIVE_NUMBER_NORMALIZATION_RE.sub(
            lambda match: f"{match.group('prefix')}음수{match.group('suffix')}",
            result,
        )
        return result

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Estimate token count conservatively using roughly one token per three chars."""
        return max(1, (len(text) + 2) // 3)

    def _should_reduce_history(self) -> bool:
        """Return ``True`` when the previous LLM stage exceeded the adaptive threshold."""
        last_llm_time_s = self._last_llm_time_s
        if last_llm_time_s is None:
            return False
        return last_llm_time_s > self._config.adaptive_history_threshold_s

    # ---- Prompt construction ----------------------------------------------

    def _build_prompt(self, user_text: str) -> str:
        """Build the legacy raw prompt string for compatibility callers/tests."""
        return self._build_prompt_legacy(user_text)

    def _build_prompt_legacy(
        self,
        user_text: str,
        detected_language: str | None = None,
    ) -> str:
        """Build LLM prompt using Qwen3 chat template.

        Format::

            <|im_start|>system
            {system_prompt}<|im_end|>
            <|im_start|>user
            {history_user}<|im_end|>
            <|im_start|>assistant
            {history_assistant}<|im_end|>
            ...
            <|im_start|>user
            {user_text}<|im_end|>
            <|im_start|>assistant
        """
        cfg = self._config
        self._current_language = self._response_language_for_turn(user_text, detected_language)
        fact_match = self._match_fact_shortlist(user_text, detected_language)
        system_prompt = self._select_system_prompt(
            user_text,
            detected_language,
            fact_match=fact_match,
        )
        parts: list[str] = [
            f"<|im_start|>system\n{system_prompt}<|im_end|>",
        ]

        # Include recent history (N turn-pairs = 2N entries)
        history_turns = 1 if self._should_reduce_history() else cfg.max_history_turns
        history_window = list(self._history[-(history_turns * 2) :])

        total_tokens = sum(self._estimate_tokens(turn["text"]) for turn in history_window)
        while total_tokens > cfg.max_history_tokens and len(history_window) >= 2:
            removed_user = history_window.pop(0)
            removed_assistant = history_window.pop(0)
            total_tokens -= self._estimate_tokens(removed_user["text"])
            total_tokens -= self._estimate_tokens(removed_assistant["text"])

        for turn in history_window:
            if turn["role"] == "user":
                parts.append(
                    f"<|im_start|>user\n{turn['text']}<|im_end|>",
                )
            else:
                parts.append(
                    f"<|im_start|>assistant\n{turn['text']}<|im_end|>",
                )

        if self._fact_shortlist_mode == "p2" and fact_match is not None:
            fact_message = self._build_fact_context_message(fact_match)
            parts.append(
                f"<|im_start|>{fact_message['role']}\n{fact_message['content']}<|im_end|>",
            )
        parts.append(f"<|im_start|>user\n{user_text}<|im_end|>")
        parts.append("<|im_start|>assistant\n")

        return "\n".join(parts)

    def _build_messages(
        self,
        user_text: str,
        detected_language: str | None = None,
        metrics: TurnMetrics | None = None,
    ) -> list[dict[str, str]]:
        """Build chat messages for create_chat_completion API."""
        cfg = self._config
        self._current_language = self._response_language_for_turn(user_text, detected_language)
        fact_match = self._match_fact_shortlist(user_text, detected_language)
        system_prompt = self._select_system_prompt(
            user_text,
            detected_language,
            fact_match=fact_match,
        )
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
        ]

        history_turns = 1 if self._should_reduce_history() else cfg.max_history_turns
        history_window = list(self._history[-(history_turns * 2) :])

        total_tokens = sum(self._estimate_tokens(turn["text"]) for turn in history_window)
        while total_tokens > cfg.max_history_tokens and len(history_window) >= 2:
            removed_user = history_window.pop(0)
            removed_assistant = history_window.pop(0)
            total_tokens -= self._estimate_tokens(removed_user["text"])
            total_tokens -= self._estimate_tokens(removed_assistant["text"])

        fact_message = (
            self._build_fact_context_message(fact_match)
            if self._fact_shortlist_mode == "p2" and fact_match is not None
            else None
        )
        recall_message = self._build_conversation_memory_message(user_text, metrics)
        history_window, recall_message = self._fit_conversation_memory_budget(
            history_window,
            system_prompt,
            fact_message,
            recall_message,
            user_text,
        )

        for turn in history_window:
            messages.append({"role": turn["role"], "content": turn["text"]})

        if fact_message is not None:
            messages.append(fact_message)
        if recall_message is not None:
            messages.append(recall_message)
        messages.append({"role": "user", "content": user_text})

        return messages

    def _most_recent_user_utterance(self) -> str | None:
        """Return the most-recent thing the child said this session.

        Scans ``self._history`` from newest to oldest for the latest
        ``{"role": "user"}`` entry and returns its ``text`` stripped, or
        ``None`` when there is no user turn yet (or the latest one is empty).
        Fixed-path responses are recorded with ``record_history=False``, so the
        recall query itself is never stored; the newest user entry is therefore
        the last genuine thing the child said.
        """
        for entry in reversed(self._history):
            if entry.get("role") != "user":
                continue
            text = entry.get("text", "").strip()
            return text or None
        return None

    def _recalled_utterance_allowed(self, text: str) -> bool:
        """Re-filter an in-session recalled utterance before speaking it back.

        Mirrors the nightly-index recall path, which re-checks each snippet with
        the live ``ContentFilter`` so a blocklist tightened after the turn was
        spoken cannot leak. ``_filter_text`` returns ``None`` when filtering is
        disabled/unavailable (fail-open): the verbatim in-session text is then
        allowed because it was already filtered when first spoken.
        """
        result = self._filter_text(text)
        return result is None or result.allowed

    def _conversation_memory_recall_answer(
        self,
        recall_match: RecallQueryMatch,
        user_text: str,
    ) -> str | None:
        """Answer an explicit-recall query from conversation memory.

        Two recall layers are consulted. For ``general_recall`` the in-session
        short-term path runs first: a now/just-now query ("방금 뭐라고 했어?")
        is answered from ``self._history`` by quoting the child's most-recent
        utterance, so the device can recall what was just said in the current
        session (the nightly index has not seen today's turns yet). Past-day
        queries ("어제 …", ``day_offset >= 1``) and the no-in-session-history
        case fall through to the nightly-index path below.

        Returns a deterministic spoken sentence that quotes the child's prior
        statement verbatim, or ``None`` when memory is unavailable or nothing
        matches (the caller then speaks the honest not-found line). The store
        may be absent (flag off / load failure); that path never raises.
        """
        if recall_match.sub_kind == "general_recall":
            window = parse_time_window(user_text, datetime.now(KST))
            is_past = window is not None and (window.day_offset or 0) >= 1
            if not is_past:
                prev = self._most_recent_user_utterance()
                if prev and self._recalled_utterance_allowed(prev):
                    return f"방금 '{prev}'라고 했잖아!"

        store = self._conversation_memory
        if store is None:
            return None
        quote = store.recall_for_intent(recall_match.sub_kind, user_text)
        if not quote:
            return None
        if recall_match.sub_kind == "name":
            return f"네가 '{quote}'(이)라고 했었지!"
        return f"음, 네가 '{quote}'라고 했었어!"

    def _build_conversation_memory_message(
        self,
        user_text: str,
        metrics: TurnMetrics | None,
    ) -> dict[str, str] | None:
        """Return the optional conversation-memory prompt block for this LLM turn."""
        if should_skip_recall_for_metrics(metrics):
            return None
        store = self._conversation_memory
        if store is None:
            return None
        if not self._conversation_memory_first_turn_checked and not self._history:
            self._conversation_memory_first_turn_checked = True
            first_turn_message = store.first_turn_message(estimate_tokens=self._estimate_tokens)
            if first_turn_message is not None:
                return first_turn_message
        return store.build_recall_message(user_text, estimate_tokens=self._estimate_tokens)

    def _fit_conversation_memory_budget(
        self,
        history_window: list[dict[str, str]],
        system_prompt: str,
        fact_message: dict[str, str] | None,
        recall_message: dict[str, str] | None,
        user_text: str,
    ) -> tuple[list[dict[str, str]], dict[str, str] | None]:
        """Apply the memory trim ladder: history, recall trim, recall omit."""
        if recall_message is None:
            return history_window, None

        n_ctx = max(1, self._llm_backend_config.n_ctx)
        response_reserve = max(256, self._llm_backend_config.max_tokens)

        def _candidate_messages(
            candidate_history: list[dict[str, str]],
            candidate_recall: dict[str, str] | None,
        ) -> tuple[Mapping[str, str], ...]:
            messages: list[Mapping[str, str]] = [{"role": "system", "content": system_prompt}]
            messages.extend(
                {"role": turn["role"], "content": turn["text"]} for turn in candidate_history
            )
            if fact_message is not None:
                messages.append(fact_message)
            if candidate_recall is not None:
                messages.append(candidate_recall)
            messages.append({"role": "user", "content": user_text})
            return tuple(messages)

        while (
            not fits_context_budget(
                _candidate_messages(history_window, recall_message),
                estimate_tokens=self._estimate_tokens,
                n_ctx=n_ctx,
                response_token_reserve=response_reserve,
            )
            and len(history_window) >= 2
        ):
            history_window = history_window[2:]

        if fits_context_budget(
            _candidate_messages(history_window, recall_message),
            estimate_tokens=self._estimate_tokens,
            n_ctx=n_ctx,
            response_token_reserve=response_reserve,
        ):
            return history_window, recall_message

        trimmed_recall = self._trim_recall_message_for_context(
            history_window,
            system_prompt,
            fact_message,
            recall_message,
            user_text,
            n_ctx=n_ctx,
            response_reserve=response_reserve,
        )
        return history_window, trimmed_recall

    def _trim_recall_message_for_context(
        self,
        history_window: list[dict[str, str]],
        system_prompt: str,
        fact_message: dict[str, str] | None,
        recall_message: dict[str, str],
        user_text: str,
        *,
        n_ctx: int,
        response_reserve: int,
    ) -> dict[str, str] | None:
        """Trim then omit the recall block if the context guard still fails."""

        def _messages(content: str) -> tuple[Mapping[str, str], ...]:
            messages: list[Mapping[str, str]] = [{"role": "system", "content": system_prompt}]
            messages.extend(
                {"role": turn["role"], "content": turn["text"]} for turn in history_window
            )
            if fact_message is not None:
                messages.append(fact_message)
            messages.append({"role": "user", "content": content})
            messages.append({"role": "user", "content": user_text})
            return tuple(messages)

        content = recall_message["content"]
        suffix = "..."
        while content:
            if fits_context_budget(
                _messages(content),
                estimate_tokens=self._estimate_tokens,
                n_ctx=n_ctx,
                response_token_reserve=response_reserve,
            ):
                return {"role": "user", "content": content}
            next_len = max(0, len(content) - max(6, len(content) // 5))
            content = content[:next_len].rstrip()
            if content and not content.endswith(suffix):
                content = f"{content}{suffix}"
        return None

    def _detect_turn_language(self, user_text: str) -> str:
        """Return the session-owned response language for the current turn."""
        del user_text
        if not self._config.bilingual_mode:
            return "ko"
        return self._session_language

    def _response_language_for_turn(
        self,
        user_text: str,
        detected_language: str | None = None,
    ) -> Literal["ko", "en"]:
        """Return a normalized response language for prompt/TTS ownership."""
        if detected_language is not None:
            return self._normalize_session_language(detected_language)
        return self._normalize_session_language(self._detect_turn_language(user_text))

    def _select_system_prompt(
        self,
        user_text: str,
        detected_language: str | None = None,
        fact_match: FactMatch | None = None,
    ) -> str:
        """Return the active system prompt for the current turn."""
        language = self._response_language_for_turn(user_text, detected_language)
        prompt_language: Literal["ko", "en"] = (
            "en" if self._config.bilingual_mode and language == "en" else "ko"
        )
        backend: Literal["gemma4_text", "qwen3_legacy"] = (
            "gemma4_text" if self._is_gemma4_text_backend() else "qwen3_legacy"
        )
        safety_guide = self._pending_safety_guide
        self._pending_safety_guide = None
        core_only_mode = self._config.persona_conditional_loading
        confirmable_fact_ko = (
            fact_match.fact_ko
            if self._fact_shortlist_mode == "p1" and prompt_language == "ko" and fact_match
            else None
        )

        base_prompt = assemble_persona_prompt(
            language=prompt_language,
            backend=backend,
            intent_signals=IntentSignals.all_true(),
            core_only_mode=core_only_mode,
            confirmable_fact_ko=confirmable_fact_ko,
        ).text
        override_reference_prompt = (
            assemble_persona_prompt(
                language=prompt_language,
                backend=backend,
                intent_signals=IntentSignals.all_true(),
            ).text
            if core_only_mode
            else base_prompt
        )
        trusted_override = self._deprecated_system_prompt_override(
            prompt_language=prompt_language,
            backend=backend,
            assembled_prompt=override_reference_prompt,
        )
        if trusted_override is not None:
            prompt = assemble_persona_prompt(
                language=prompt_language,
                backend=backend,
                intent_signals=IntentSignals.all_true(),
                trusted_full_prompt_override=trusted_override,
                safety_guide=safety_guide,
                confirmable_fact_ko=confirmable_fact_ko,
            ).text
            return prompt

        if safety_guide or confirmable_fact_ko:
            return assemble_persona_prompt(
                language=prompt_language,
                backend=backend,
                intent_signals=IntentSignals.all_true(),
                core_only_mode=core_only_mode,
                safety_guide=safety_guide,
                confirmable_fact_ko=confirmable_fact_ko,
            ).text
        return base_prompt

    def _match_fact_shortlist(
        self,
        user_text: str,
        detected_language: str | None = None,
    ) -> FactMatch | None:
        """Return the per-turn shortlist match when the feature flag is enabled."""

        if self._fact_shortlist_mode == "disabled":
            return None
        language = self._response_language_for_turn(user_text, detected_language)
        prompt_language: Literal["ko", "en"] = (
            "en" if self._config.bilingual_mode and language == "en" else "ko"
        )
        return match_fact(user_text, lang=prompt_language)

    @staticmethod
    def _build_fact_context_message(fact_match: FactMatch) -> dict[str, str]:
        """Return the synthetic user-adjacent fact context message for P2."""

        return {"role": "user", "content": f"[참고 정보] {fact_match.fact_ko}"}

    def _deprecated_system_prompt_override(
        self,
        *,
        prompt_language: str,
        backend: str,
        assembled_prompt: str,
    ) -> str | None:
        """Return a legacy caller override only when it differs from module assembly."""

        if prompt_language == "en" and self._en_system_prompt:
            return self._en_system_prompt if self._en_system_prompt != assembled_prompt else None
        if backend == "gemma4_text" and self._gemma4_persona_prompt is not None:
            if self._gemma4_persona_prompt != assembled_prompt:
                return self._gemma4_persona_prompt
            return None
        if prompt_language == "ko" and self._config.llm_system_prompt != assembled_prompt:
            return self._config.llm_system_prompt
        return None

    @staticmethod
    def _append_safety_guide(prompt: str, language: str, guide: str) -> str:
        """Append a safety guide to a deprecated trusted prompt override."""

        if language == "en":
            return (
                f"{prompt}\n\n[Safety Guide] {guide}\n"
                "Refer to the above safety information, but answer the child's "
                "question with an educational and age-appropriate explanation."
            )
        return (
            f"{prompt}\n\n[안전 가이드] {guide}\n"
            "위 안전 정보를 참고하되, 아이의 질문에 맞는 교육적이고 "
            "이해하기 쉬운 답변을 해주세요."
        )

    def _load_english_system_prompt(self) -> str:
        """Load the English prompt from disk with a Korean prompt fallback."""
        prompt_path = self._resolve_repo_path(self._config.en_system_prompt_path)
        try:
            return prompt_path.read_text(encoding="utf-8").strip()
        except OSError:
            logger.warning(
                "Failed to load English system prompt from %s; using Korean prompt fallback",
                prompt_path,
                exc_info=True,
            )
            return self._config.llm_system_prompt

    @staticmethod
    def _resolve_repo_path(relative_path: str) -> Path:
        """Resolve a repository-relative path from the core package."""
        return Path(__file__).resolve().parent.parent / relative_path

    @staticmethod
    def _messages_to_prompt(messages: list[dict[str, str]]) -> str:
        """Serialize chat messages back into the legacy raw prompt format."""
        parts: list[str] = []
        for message in messages:
            parts.append(
                f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>",
            )
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)

    # ---- Utilities --------------------------------------------------------

    def _init_session_dir(self) -> None:
        """Create a timestamped session directory."""
        if self._session_dir is not None:
            return

        kst = timezone(timedelta(hours=9))
        ts = datetime.now(kst).strftime("%Y-%m-%d_%H-%M-%S")
        session_dir = self._conversation_dir / ts
        try:
            session_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            logger.warning(
                "Failed to create conversation session directory at %s",
                session_dir,
                exc_info=True,
            )
            return

        self._session_dir = session_dir
        self._turn_counter = 0
        self._conversation_memory = load_conversation_memory()
        self._conversation_memory_first_turn_checked = False

    def _save_conversation_audio(
        self,
        audio_samples: list[float] | Any,
        sample_rate: int,
        filename: str,
    ) -> Path | None:
        """Save audio samples to WAV in the conversation directory."""
        session_dir = self._session_dir
        if session_dir is None:
            return None
        if sample_rate <= 0:
            logger.warning(
                "Skipping conversation audio save with invalid sample rate: %s",
                sample_rate,
            )
            return None

        mono_samples = self._downmix_to_mono(audio_samples)
        if not mono_samples:
            return None
        if sample_rate != VAD_SAMPLE_RATE:
            mono_samples = self._resample_audio(
                mono_samples,
                source_sample_rate=sample_rate,
                target_sample_rate=VAD_SAMPLE_RATE,
            )

        target_path = session_dir / filename
        try:
            self._write_temp_wav(target_path, mono_samples)
        except OSError:
            logger.warning(
                "Failed to save conversation audio at %s",
                target_path,
                exc_info=True,
            )
            return None
        return target_path

    def _log_conversation_turn(
        self,
        turn_num: int,
        user_text: str,
        response_text: str,
        input_wav: str | None,
        output_wav: str | None,
        metrics: dict[str, Any],
        hotword_hallucination_detected: bool = False,
        hotword_hallucination_reason: HotwordHallucinationReason = "clean",
        stt_script_drift_detected: bool = False,
    ) -> None:
        """Append a turn record to the session conversation.jsonl."""
        session_dir = self._session_dir
        if session_dir is None:
            return

        kst = timezone(timedelta(hours=9))
        record = {
            "timestamp": datetime.now(kst).isoformat(timespec="seconds"),
            "turn": turn_num,
            "user_text": user_text,
            "response_text": response_text,
            "input_wav": input_wav,
            "output_wav": output_wav,
            "hotword_hallucination_detected": hotword_hallucination_detected,
            "hotword_hallucination_reason": hotword_hallucination_reason,
            "stt_script_drift_detected": stt_script_drift_detected,
            "metrics": metrics,
        }

        log_path = session_dir / "conversation.jsonl"
        try:
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False))
                handle.write("\n")
        except OSError:
            logger.warning(
                "Failed to append conversation log at %s",
                log_path,
                exc_info=True,
            )

    def _write_session_end_marker(self) -> None:
        """Write the clean session completion sentinel atomically."""
        session_dir = self._session_dir
        if session_dir is None:
            return
        sentinel_path = session_dir / "session_end.json"
        sentinel = SessionEndSentinel(
            ended_at=datetime.now(KST),
            turn_count=self._turn_counter,
        )
        payload = json.dumps(sentinel.to_json_dict(), ensure_ascii=False, separators=(",", ":"))
        temp_path = sentinel_path.with_name(f"{sentinel_path.name}.tmp-{os.getpid()}")
        try:
            with temp_path.open("w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            temp_path.replace(sentinel_path)
            _fsync_directory(sentinel_path.parent)
        except OSError:
            logger.warning(
                "Failed to write session completion sentinel at %s",
                sentinel_path,
                exc_info=True,
            )
            with contextlib.suppress(OSError):
                temp_path.unlink()

    @staticmethod
    def _write_temp_wav(path: Path, samples: list[float]) -> None:
        """Write float samples to a 16-bit mono 16kHz WAV file.

        NaN and Inf samples are replaced with 0 (silence) to tolerate
        occasional driver-level float anomalies on the capture path.
        """

        def _to_pcm(value: float) -> int:
            if math.isnan(value) or math.isinf(value):
                return 0
            return max(-32768, min(32767, int(value * 32768)))

        pcm_data = struct.pack(
            f"<{len(samples)}h",
            *(_to_pcm(s) for s in samples),
        )
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(VAD_SAMPLE_RATE)
            wf.writeframes(pcm_data)


def _fsync_directory(path: Path) -> None:
    """Best-effort fsync for a directory after atomic replace."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
