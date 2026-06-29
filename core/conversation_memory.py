"""Encoder-free daytime conversation-memory recall.

This module owns the v0 day-path lookup for conversation memory. It deliberately
loads only JSON artifacts produced by the nightly job; no vector, numpy, ONNX, or
LLM resources are used in the conversation process.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Final, TypeAlias, cast

from core.conversation_memory_schema import (
    CONVERSATION_MEMORY_ENV_FLAG,
    CONVERSATION_MEMORY_RUNTIME_SUBPATH,
    GENERATION_POINTER_FILENAME,
    KST,
    PREFIX_TOL,
    RECALL_INJECTION_HARD_CAP_TOKENS,
    IndexReference,
    SchemaError,
    TurnSnippet,
    parse_generation_pointer,
)
from core.runtime import detect_runtime_paths

if TYPE_CHECKING:
    from safety.content_filter import ContentFilter

logger = logging.getLogger(__name__)

EstimateTokens: TypeAlias = Callable[[str], int]

_NORM_RE: Final[re.Pattern[str]] = re.compile(r"[^0-9a-z가-힣\s]")
_NDAYS_RE: Final[re.Pattern[str]] = re.compile(r"(\d+)\s*일\s*전")
_INTENT_RE: Final[re.Pattern[str]] = re.compile(
    r"기억나|기억 나|말했잖아|말 했잖아|얘기했|이야기했|했었지|했잖아"
)
# Korean wh-question terminals used to flag a turn as interrogative when the
# trailing ``?`` was dropped (e.g. by ASR). Restricted to ``뭐``/``뭘`` + an
# interrogative suffix, which is never a declarative sentence ending, so it
# cannot misclassify statements like "제일 좋아" or "이름은 종경".
_INTERROGATIVE_ENDINGS: Final[tuple[str, ...]] = ("뭐야", "뭐니", "뭘까", "뭐냐")
_PARTICLE_SUFFIXES: Final[tuple[str, ...]] = (
    "가",
    "이",
    "은",
    "는",
    "을",
    "를",
    "도",
    "만",
    "에",
    "야",
    "의",
    "랑",
)
_DAY_WORDS: Final[dict[str, int]] = {"오늘": 0, "어제": 1, "그저께": 2, "그제": 2}
_DAYPARTS: Final[dict[str, tuple[int, int]]] = {
    "아침": (5, 11),
    "점심": (11, 15),
    "저녁": (17, 21),
    "밤": (20, 24),
}
_LASTWEEK: Final[str] = "지난주"
_TIME_ONLY_MAX_DAYS: Final[int] = 3
_RECENCY_DAYS: Final[int] = 7
_RARE_DF_MAX: Final[int] = 2
_TEMPORAL_PREFIX_TOL: Final[int] = 2
_RESPONSE_TOKEN_RESERVE: Final[int] = 256
_LOAD_FAILURE_LOGGED = False

STOPWORDS: Final[frozenset[str]] = frozenset(
    {
        "뭉이",
        "뭉이야",
        "근데",
        "그리고",
        "그래서",
        "나도",
        "내가",
        "너는",
        "네가",
        "우리",
        "진짜",
        "정말",
        "너무",
        "엄청",
        "좀",
        "다시",
        "했어",
        "있어",
        "없어",
        "좋아",
        "싫어",
        "뭐",
        "뭐야",
        "왜",
        "어떻게",
        "어디",
        "누구",
        "언제",
        "그게",
        "이게",
        "저게",
        "거야",
        "같아",
        "한테",
        "에서",
        "으로",
        "이랑",
        "랑",
        "하고",
        "지금",
        "그냥",
        "아니",
        "응",
        "싶어",
        "싶다",
        "했지",
        "할래",
        "줄래",
        "해줘",
        "보여줘",
        "말해줘",
        "알려줘",
    }
)
_TEMPORAL_SURFACE: Final[frozenset[str]] = frozenset(
    {*_DAY_WORDS.keys(), *_DAYPARTS.keys(), _LASTWEEK, "일전"}
)
_FIXED_PATH_METRIC_FIELDS: Final[tuple[str, ...]] = (
    "crisis_matched",
    "parent_disclosure_matched",
    "template_matched",
    "belief_matched",
    "content_filter_blocked",
    "history_mode_matched",
    "funny_english_matched",
    "language_switch_matched",
    "recall_query_matched",
)

# Per-sub_kind candidate-keyword seeds for the explicit-recall relaxation. The
# seeds nudge the keyword gate toward the snippet the child is asking about
# (their name) when the spoken query itself carries no content tokens
# (e.g. "내 이름 기억나?").
_RECALL_INTENT_SEEDS: Final[dict[str, tuple[str, ...]]] = {
    "name": ("이름", "이름이"),
    "general_recall": (),
}


@dataclass(frozen=True)
class TimeWindow:
    """A parsed KST time window used only as an AND filter."""

    start: datetime
    end: datetime
    day_offset: int | None
    source: str


@dataclass(frozen=True)
class RecallMatch:
    """One accepted recall result ready for prompt injection."""

    snippet: TurnSnippet
    matched_query_tokens: tuple[str, ...]
    matched_turn_tokens: tuple[str, ...]
    via: str
    score: float


@dataclass(frozen=True)
class RecallDecision:
    """Diagnostics for one recall lookup."""

    query_tokens: tuple[str, ...]
    time_window: TimeWindow | None
    recall_intent: bool
    match: RecallMatch | None


def conversation_memory_enabled() -> bool:
    """Return whether the shared conversation-memory rollout flag is enabled."""

    return os.getenv(CONVERSATION_MEMORY_ENV_FLAG, "").strip() == "1"


def conversation_memory_root(mutable_root: Path | None = None) -> Path:
    """Return the mutable root for conversation-memory artifacts."""

    root = mutable_root if mutable_root is not None else Path(detect_runtime_paths().mutable_root)
    return root / CONVERSATION_MEMORY_RUNTIME_SUBPATH


def normalize(text: str) -> str:
    """Normalize text for v0 Hangul-aware keyword matching."""

    return re.sub(r"\s+", " ", _NORM_RE.sub(" ", text.lower())).strip()


def tokens_of(text: str) -> tuple[str, ...]:
    """Return normalized whitespace tokens with length >= 2."""

    return tuple(token for token in normalize(text).split() if len(token) >= 2)


def strip_particle(token: str) -> str:
    """Strip one trailing single-character Korean particle when safe."""

    if len(token) >= 3 and token[-1] in _PARTICLE_SUFFIXES:
        return token[:-1]
    return token


def token_hit(query_token: str, turn_token: str) -> bool:
    """Return whether two surface tokens match under the G0 particle/prefix rule."""

    query_stem = strip_particle(query_token)
    turn_stem = strip_particle(turn_token)
    if query_stem == turn_stem:
        return True
    shorter, longer = sorted((query_stem, turn_stem), key=len)
    return (
        len(shorter) >= 2
        and longer.startswith(shorter)
        and len(longer) - len(shorter) <= PREFIX_TOL
    )


def content_tokens(text: str) -> tuple[str, ...]:
    """Return content-bearing query/index tokens.

    Temporal words, stopwords, and recall-intent expressions are not content
    hits. Temporal exclusion is particle tolerant; stopwords are exact-match.
    """

    return tuple(
        token
        for token in tokens_of(text)
        if not _is_stopword(token) and not _is_temporal(token) and not _INTENT_RE.search(token)
    )


def parse_time_window(query: str, now: datetime) -> TimeWindow | None:
    """Parse the deterministic Korean time window from a query."""

    norm = normalize(query)
    day_offset: int | None = None
    source = ""
    for word, offset in _DAY_WORDS.items():
        if word in norm:
            day_offset = offset
            source = word
            break

    ndays = _NDAYS_RE.search(norm)
    if day_offset is None and ndays is not None:
        day_offset = int(ndays.group(1))
        source = "n_days_ago"

    if day_offset is None and _LASTWEEK in norm:
        start_day = (now - timedelta(days=now.weekday() + 7)).replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )
        return TimeWindow(
            start=start_day, end=start_day + timedelta(days=7), day_offset=None, source=_LASTWEEK
        )

    if day_offset is None:
        return None

    day_start = (now - timedelta(days=day_offset)).replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )
    for part, (start_hour, end_hour) in _DAYPARTS.items():
        if part in norm:
            return TimeWindow(
                start=day_start + timedelta(hours=start_hour),
                end=day_start + timedelta(hours=end_hour),
                day_offset=day_offset,
                source=f"{source}:{part}",
            )
    return TimeWindow(
        start=day_start,
        end=day_start + timedelta(days=1),
        day_offset=day_offset,
        source=source,
    )


def should_skip_recall_for_metrics(metrics: object | Mapping[str, object] | None) -> bool:
    """Return whether a fixed/router path has already claimed the turn."""

    if metrics is None:
        return False
    return any(_metric_value(metrics, field) is True for field in _FIXED_PATH_METRIC_FIELDS)


def load_conversation_memory(mutable_root: Path | None = None) -> ConversationMemoryStore | None:
    """Load the current memory generation when the rollout flag is enabled.

    Any load failure returns ``None`` and logs at most once per process so the
    live conversation path never blocks on memory artifacts.
    """

    if not conversation_memory_enabled():
        return None
    root = conversation_memory_root(mutable_root)
    try:
        return ConversationMemoryStore.load(root)
    except (OSError, ValueError, json.JSONDecodeError, SchemaError) as exc:
        _log_load_failure_once(root, exc)
        return None


class ConversationMemoryStore:
    """Loaded v0 conversation-memory generation for pure-Python recall."""

    def __init__(
        self,
        *,
        generation_id: str,
        snippets: Mapping[str, TurnSnippet],
        index: Mapping[str, tuple[IndexReference, ...]],
        quarantined_days: frozenset[date],
        content_filter: ContentFilter | None = None,
    ) -> None:
        self.generation_id = generation_id
        self._snippets = dict(snippets)
        self._index = dict(index)
        self._quarantined_days = quarantined_days
        self._content_filter = content_filter
        self._content_filter_resolved = content_filter is not None
        self._snippet_tokens = {
            snippet_id: content_tokens(snippet.text)
            for snippet_id, snippet in self._snippets.items()
        }
        self._df = self._build_df()
        self._lookup = self._build_lookup()

    @classmethod
    def load(cls, memory_root: Path) -> ConversationMemoryStore:
        """Load the generation pointed to by ``current`` under ``memory_root``."""

        generation_id = parse_generation_pointer(
            (memory_root / GENERATION_POINTER_FILENAME).read_text(encoding="utf-8")
        )
        generation_dir = memory_root / "generations" / generation_id
        index = _load_index(generation_dir / "index.json")
        snippets = _load_turns(generation_dir / "turns.jsonl")
        quarantined_days = _load_quarantined_days(generation_dir / "quarantined_days.json")
        return cls(
            generation_id=generation_id,
            snippets=snippets,
            index=index,
            quarantined_days=quarantined_days,
        )

    def recall(
        self,
        query: str,
        *,
        now: datetime | None = None,
        recall_intent: bool | None = None,
        extra_query_tokens: tuple[str, ...] = (),
        snippet_predicate: Callable[[TurnSnippet], bool] | None = None,
    ) -> RecallDecision:
        """Return the top recall match for one query, if it passes the gate.

        Args:
            query: The child's spoken query.
            now: Optional clock override (KST is used when omitted) for testing.
            recall_intent: Force the recall-intent relaxation on/off; when
                ``None`` the intent is inferred from the query text.
            extra_query_tokens: Additional content tokens biasing candidate
                selection (used by the explicit-recall sub_kind seeds).
            snippet_predicate: Optional filter applied to each candidate's
                snippet before final selection. Candidates whose snippet fails
                the predicate are dropped, so a lower-scored snippet that passes
                still wins over a higher-scored one that fails.
        """

        anchor = _normalize_kst(now or datetime.now(KST))
        query_tokens = tuple(dict.fromkeys((*content_tokens(query), *extra_query_tokens)))
        window = parse_time_window(query, anchor)
        effective_intent = (
            recall_intent
            if recall_intent is not None
            else bool(_INTENT_RE.search(normalize(query)))
        )
        candidates = self._candidate_ids(query_tokens, window, effective_intent, anchor)

        accepted: list[RecallMatch] = []
        for snippet_id in candidates:
            snippet = self._snippets.get(snippet_id)
            if snippet is None:
                continue
            if snippet_predicate is not None and not snippet_predicate(snippet):
                continue
            in_window = window is None or window.start <= snippet.timestamp < window.end
            if not in_window:
                continue
            accepted_match = self._evaluate_candidate(
                snippet,
                query_tokens,
                window,
                effective_intent,
                anchor,
            )
            if accepted_match is not None:
                accepted.append(accepted_match)

        accepted.sort(key=lambda item: (item.score, item.snippet.timestamp), reverse=True)
        return RecallDecision(
            query_tokens=query_tokens,
            time_window=window,
            recall_intent=effective_intent,
            match=accepted[0] if accepted else None,
        )

    def recall_for_intent(
        self,
        sub_kind: str,
        query: str,
        *,
        now: datetime | None = None,
    ) -> str | None:
        """Return the verbatim top snippet text for an explicit-recall intent.

        The explicit-recall path forces the recall-intent relaxation and biases
        candidate selection with ``sub_kind`` keyword seeds (name) so a short
        spoken query that carries no content tokens still finds the snippet the
        child is asking about. Interrogative snippets are excluded so recall
        surfaces declarative statements only (never echoes a question back).
        The recalled text is returned verbatim (never paraphrased), or ``None``
        when nothing passes the gate or the snippet is rejected by the
        defensive content re-filter.
        """

        anchor = _normalize_kst(now or datetime.now(KST))
        seeds = _RECALL_INTENT_SEEDS.get(sub_kind, ())
        decision = self.recall(
            query,
            now=anchor,
            recall_intent=True,
            extra_query_tokens=seeds,
            snippet_predicate=lambda snippet: not _is_interrogative(snippet.text),
        )
        if decision.match is None:
            return None
        text = decision.match.snippet.text.strip()
        if not self._snippet_text_allowed(text):
            return None
        return text or None

    def build_recall_message(
        self,
        query: str,
        *,
        estimate_tokens: EstimateTokens,
        now: datetime | None = None,
    ) -> dict[str, str] | None:
        """Build the injectable ``[기억]`` message for one query."""

        anchor = _normalize_kst(now or datetime.now(KST))
        decision = self.recall(query, now=anchor)
        if decision.match is None:
            return None
        if not self._snippet_text_allowed(decision.match.snippet.text):
            return None
        content = _format_recall_content(decision.match.snippet, anchor)
        content = trim_recall_content(content, estimate_tokens)
        if not content:
            return None
        return {"role": "user", "content": content}

    def _snippet_text_allowed(self, text: str) -> bool:
        """Return whether a recalled snippet still passes the content filter.

        Guards against a blocklist tightened after the nightly index was built:
        a snippet that was clean at index time may now contain blocked content,
        so it is re-filtered with the live ``ContentFilter`` before it can be
        recalled. A missing/unavailable filter is fail-open (no re-filter), and
        a filter error never blocks the live path.
        """

        content_filter = self._get_content_filter()
        if content_filter is None:
            return True
        try:
            return content_filter.filter(text).allowed
        except Exception:
            logger.warning("conversation_memory_recall_refilter_failed", exc_info=True)
            return True

    def _get_content_filter(self) -> ContentFilter | None:
        """Lazily resolve the live content filter the pipeline would build."""

        if not self._content_filter_resolved:
            self._content_filter_resolved = True
            try:
                from safety.content_filter import ContentFilter as _ContentFilter

                self._content_filter = _ContentFilter.from_default()
            except Exception:
                logger.warning("conversation_memory_content_filter_unavailable", exc_info=True)
                self._content_filter = None
        return self._content_filter

    def should_suppress_first_turn(self, *, now: datetime | None = None) -> bool:
        """Return whether the v1 first-turn summary seam is suppressed."""

        anchor = _normalize_kst(now or datetime.now(KST))
        return (anchor.date() - timedelta(days=1)) in self._quarantined_days

    def first_turn_message(
        self,
        *,
        estimate_tokens: EstimateTokens,
        now: datetime | None = None,
    ) -> dict[str, str] | None:
        """Return the v1 first-turn summary message.

        v0 has no summary layer, so this hook is intentionally inert while still
        enforcing quarantine-day suppression for the future summary seam.
        """

        del estimate_tokens
        if self.should_suppress_first_turn(now=now):
            return None
        return None

    def _candidate_ids(
        self,
        query_tokens: tuple[str, ...],
        window: TimeWindow | None,
        recall_intent: bool,
        now: datetime,
    ) -> tuple[str, ...]:
        if not query_tokens:
            if _is_time_only_query(query_tokens, window, recall_intent, now):
                return tuple(self._snippets)
            return ()

        ids: dict[str, None] = {}
        for token in query_tokens:
            for lookup_key in _lookup_keys_for_query_token(token):
                for reference in self._lookup.get(lookup_key, ()):
                    if reference.layer == "turns":
                        ids[reference.id] = None
        return tuple(ids)

    def _evaluate_candidate(
        self,
        snippet: TurnSnippet,
        query_tokens: tuple[str, ...],
        window: TimeWindow | None,
        recall_intent: bool,
        now: datetime,
    ) -> RecallMatch | None:
        snippet_timestamp = _normalize_kst(snippet.timestamp)
        age_days = max(0, (now - snippet_timestamp).days)
        time_only_query = _is_time_only_query(query_tokens, window, recall_intent, now)
        if time_only_query:
            if age_days > _TIME_ONLY_MAX_DAYS:
                return None
            return RecallMatch(
                snippet=snippet,
                matched_query_tokens=(),
                matched_turn_tokens=(),
                via="time_only",
                score=1.0 / (age_days + 1),
            )

        snippet_tokens = self._snippet_tokens.get(snippet.id, ())
        matched_query: list[str] = []
        matched_turn: list[str] = []
        for query_token in query_tokens:
            turn_matches = [token for token in snippet_tokens if token_hit(query_token, token)]
            if turn_matches:
                matched_query.append(query_token)
                matched_turn.extend(turn_matches)

        rare_hit = any(self._df.get(token, 0) <= _RARE_DF_MAX for token in matched_turn)
        keyword_pass = len(matched_query) >= 2 or (
            bool(matched_query) and rare_hit and age_days <= _RECENCY_DAYS
        )
        if not keyword_pass:
            return None
        recency_score = 1.0 / (age_days + 1)
        return RecallMatch(
            snippet=snippet,
            matched_query_tokens=tuple(dict.fromkeys(matched_query)),
            matched_turn_tokens=tuple(dict.fromkeys(matched_turn)),
            via="keyword",
            score=(len(set(matched_query)) * 10.0) + recency_score,
        )

    def _build_df(self) -> dict[str, int]:
        df: dict[str, int] = {}
        for tokens in self._snippet_tokens.values():
            for token in set(tokens):
                df[token] = df.get(token, 0) + 1
        return df

    def _build_lookup(self) -> dict[str, tuple[IndexReference, ...]]:
        lookup: dict[str, dict[IndexReference, None]] = {}
        for keyword, references in self._index.items():
            for variant in _index_token_variants(keyword):
                for lookup_key in _lookup_keys_for_index_token(variant):
                    bucket = lookup.setdefault(lookup_key, {})
                    for reference in references:
                        bucket[reference] = None
        return {key: tuple(values) for key, values in lookup.items()}


def trim_recall_content(content: str, estimate_tokens: EstimateTokens) -> str:
    """Trim a recall block to the hard token budget using the pipeline estimator."""

    if estimate_tokens(content) <= RECALL_INJECTION_HARD_CAP_TOKENS:
        return content
    marker = "[기억] "
    suffix = "..."
    if not content.startswith(marker):
        marker = ""
    available = max(0, RECALL_INJECTION_HARD_CAP_TOKENS - estimate_tokens(marker + suffix))
    if available <= 0:
        return ""
    body = content[len(marker) :]
    trimmed = body
    while (
        trimmed and estimate_tokens(f"{marker}{trimmed}{suffix}") > RECALL_INJECTION_HARD_CAP_TOKENS
    ):
        trimmed = trimmed[:-3].rstrip()
    return f"{marker}{trimmed}{suffix}" if trimmed else ""


def fits_context_budget(
    messages: tuple[Mapping[str, str], ...],
    *,
    estimate_tokens: EstimateTokens,
    n_ctx: int,
    response_token_reserve: int = _RESPONSE_TOKEN_RESERVE,
) -> bool:
    """Return whether messages fit the G0 measured prompt-budget guard."""

    estimated = sum(estimate_tokens(message["content"]) for message in messages)
    return math.ceil(estimated * 1.15) + response_token_reserve <= n_ctx


def _load_index(path: Path) -> dict[str, tuple[IndexReference, ...]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise SchemaError("index.json must be an object")
    index: dict[str, tuple[IndexReference, ...]] = {}
    for keyword, references in raw.items():
        if not isinstance(keyword, str) or not keyword.strip():
            raise SchemaError("index keyword must be non-empty text")
        if not isinstance(references, list):
            raise SchemaError("index references must be an array")
        index[keyword] = tuple(
            IndexReference.from_json_dict(_require_mapping(item, "index reference"))
            for item in references
        )
    return index


def _load_turns(path: Path) -> dict[str, TurnSnippet]:
    snippets: dict[str, TurnSnippet] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            snippet = TurnSnippet.from_json_dict(_require_mapping(json.loads(stripped), "turn"))
            snippets[snippet.id] = snippet
    return snippets


def _load_quarantined_days(path: Path) -> frozenset[date]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return frozenset()
    if not isinstance(raw, Mapping):
        raise SchemaError("quarantined_days marker must be an object")
    days = raw.get("quarantined_days", [])
    if not isinstance(days, list) or not all(isinstance(item, str) for item in days):
        raise SchemaError("quarantined_days must be a string array")
    return frozenset(date.fromisoformat(item) for item in days)


def _require_mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise SchemaError(f"{field} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise SchemaError(f"{field} keys must be strings")
    return cast(Mapping[str, object], value)


def _is_interrogative(text: str) -> bool:
    """Return whether a turn reads as a question, for explicit-recall exclusion.

    Explicit recall must surface the child's declarative statements, never echo
    a question back ("you said 'what's my name?'"). Detection is deliberately
    conservative and precision-favoring: a trailing ``?`` is the primary signal,
    plus a small set of unambiguous Korean wh-question endings that cannot be a
    declarative terminal. Endings that risk matching declaratives are excluded.
    """

    stripped = text.rstrip()
    if not stripped:
        return False
    if stripped.endswith(("?", "？")):
        return True
    # Drop a trailing sentence terminal (other than ``?``) before suffix checks.
    core = stripped.rstrip(".!。！…").rstrip()
    return core.endswith(_INTERROGATIVE_ENDINGS)


def _is_stopword(token: str) -> bool:
    return token in STOPWORDS


def _is_temporal(token: str) -> bool:
    return (
        any(
            token.startswith(word) and len(token) <= len(word) + _TEMPORAL_PREFIX_TOL
            for word in _TEMPORAL_SURFACE
        )
        or bool(_NDAYS_RE.fullmatch(normalize(token)))
        or token.isdigit()
    )


def _is_time_only_query(
    query_tokens: tuple[str, ...],
    window: TimeWindow | None,
    recall_intent: bool,
    now: datetime,
) -> bool:
    if query_tokens or window is None or not recall_intent:
        return False
    if window.source == _LASTWEEK:
        return False
    if window.day_offset is None:
        age_days = max(0, (now.date() - window.start.date()).days)
    else:
        age_days = window.day_offset
    return age_days <= _TIME_ONLY_MAX_DAYS


def _index_token_variants(token: str) -> tuple[str, ...]:
    normalized = normalize(token)
    if not normalized:
        return ()
    variants = {normalized, strip_particle(normalized)}
    return tuple(item for item in variants if len(item) >= 2)


def _lookup_keys_for_index_token(token: str) -> tuple[str, ...]:
    keys = {token}
    min_len = max(2, len(token) - PREFIX_TOL)
    for end in range(min_len, len(token) + 1):
        keys.add(token[:end])
    return tuple(keys)


def _lookup_keys_for_query_token(token: str) -> tuple[str, ...]:
    variants = {token, strip_particle(token)}
    keys: set[str] = set()
    for variant in variants:
        if len(variant) < 2:
            continue
        keys.add(variant)
        min_len = max(2, len(variant) - PREFIX_TOL)
        for end in range(min_len, len(variant) + 1):
            keys.add(variant[:end])
    return tuple(keys)


def _format_recall_content(snippet: TurnSnippet, now: datetime) -> str:
    phrase = _friendly_time_phrase(snippet.timestamp, now)
    text = snippet.text.strip()
    if text.endswith((".", "!", "?", "。", "！", "？")):
        text = text[:-1].rstrip()
    return f"[기억] {phrase} {text}라고 했어."


def _friendly_time_phrase(timestamp: datetime, now: datetime) -> str:
    day_delta = max(0, (now.date() - timestamp.date()).days)
    if day_delta == 0:
        day = "오늘"
    elif day_delta == 1:
        day = "어제"
    elif day_delta == 2:
        day = "그저께"
    else:
        day = f"{day_delta}일 전에"
    part = _friendly_daypart(timestamp.hour)
    return f"{day} {part}에" if part else f"{day}에"


def _friendly_daypart(hour: int) -> str:
    if 5 <= hour < 11:
        return "아침"
    if 11 <= hour < 15:
        return "점심"
    if 17 <= hour < 21:
        return "저녁"
    if 21 <= hour < 24 or 0 <= hour < 5:
        return "밤"
    return ""


def _metric_value(metrics: object | Mapping[str, object], field: str) -> object:
    if isinstance(metrics, Mapping):
        return metrics.get(field)
    return getattr(metrics, field, False)


def _normalize_kst(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=KST)
    return value.astimezone(KST)


def _log_load_failure_once(root: Path, exc: Exception) -> None:
    global _LOAD_FAILURE_LOGGED
    if _LOAD_FAILURE_LOGGED:
        return
    _LOAD_FAILURE_LOGGED = True
    logger.warning(
        "conversation_memory_load_failed root=%s error=%s",
        root,
        exc,
    )


__all__ = [
    "ConversationMemoryStore",
    "RecallDecision",
    "RecallMatch",
    "STOPWORDS",
    "TimeWindow",
    "content_tokens",
    "conversation_memory_enabled",
    "conversation_memory_root",
    "fits_context_budget",
    "load_conversation_memory",
    "normalize",
    "parse_time_window",
    "should_skip_recall_for_metrics",
    "strip_particle",
    "token_hit",
    "tokens_of",
    "trim_recall_content",
]
