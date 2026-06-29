"""Tests for daytime conversation-memory recall and pipeline wiring."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from core.conversation_memory import (
    _FIXED_PATH_METRIC_FIELDS,
    ConversationMemoryStore,
    content_tokens,
    load_conversation_memory,
    parse_time_window,
    should_skip_recall_for_metrics,
    token_hit,
    trim_recall_content,
)
from core.conversation_memory_schema import (
    GENERATION_POINTER_FILENAME,
    KST,
    IndexReference,
    TurnSnippet,
    format_generation_pointer,
)
from core.llm_backend_config import LLMBackendConfig
from core.pipeline import ConversationPipeline, PipelineConfig, TurnMetrics

HASH_A = "a" * 64
HASH_B = "b" * 64


def _kst(day: int, hour: int) -> datetime:
    return datetime(2026, 6, day, hour, 0, 0, tzinfo=KST)


def _snippet(
    snippet_id: str, text: str, timestamp: datetime, *, source_hash: str = HASH_A
) -> TurnSnippet:
    return TurnSnippet(
        id=snippet_id,
        session_dir=f"session-{snippet_id}",
        turn=1,
        text=text,
        timestamp=timestamp,
        source_hash=source_hash,
    )


def _store(
    snippets: list[TurnSnippet],
    *,
    quarantined_days: set[str] | None = None,
    content_filter: Any = None,
) -> ConversationMemoryStore:
    index: dict[str, dict[IndexReference, None]] = {}
    for snippet in snippets:
        ref = IndexReference(layer="turns", id=snippet.id)
        for token in content_tokens(snippet.text):
            index.setdefault(token, {})[ref] = None
            stripped = (
                token[:-1]
                if token[-1:] in "가이은는을를도만에야의랑" and len(token) >= 3
                else token
            )
            index.setdefault(stripped, {})[ref] = None
    parsed_days = {datetime.fromisoformat(day).date() for day in (quarantined_days or set())}
    return ConversationMemoryStore(
        generation_id="testgen",
        snippets={snippet.id: snippet for snippet in snippets},
        index={key: tuple(value) for key, value in index.items()},
        quarantined_days=frozenset(parsed_days),
        content_filter=content_filter,
    )


def _write_generation(
    root: Path, snippets: list[TurnSnippet], quarantined_days: list[str] | None = None
) -> Path:
    memory_root = root / "conversation_memory"
    generation_dir = memory_root / "generations" / "gen1"
    generation_dir.mkdir(parents=True)
    index: dict[str, list[dict[str, str]]] = {}
    for snippet in snippets:
        for token in content_tokens(snippet.text):
            ref = {"layer": "turns", "id": snippet.id}
            index.setdefault(token, []).append(ref)
            stripped = (
                token[:-1]
                if token[-1:] in "가이은는을를도만에야의랑" and len(token) >= 3
                else token
            )
            index.setdefault(stripped, []).append(ref)
    (generation_dir / "turns.jsonl").write_text(
        "".join(
            json.dumps(snippet.to_json_dict(), ensure_ascii=False) + "\n" for snippet in snippets
        ),
        encoding="utf-8",
    )
    (generation_dir / "index.json").write_text(
        json.dumps(index, ensure_ascii=False),
        encoding="utf-8",
    )
    (generation_dir / "quarantined_days.json").write_text(
        json.dumps({"quarantined_days": quarantined_days or []}),
        encoding="utf-8",
    )
    (memory_root / GENERATION_POINTER_FILENAME).write_text(
        format_generation_pointer("gen1"),
        encoding="utf-8",
    )
    return memory_root


def test_particle_prefix_matching_uses_g0_cases() -> None:
    """The validated Korean particle/prefix cases keep matching."""
    assert token_hit("티라노사우루스가", "티라노사우루스")
    assert token_hit("세종대왕이", "세종대왕은")
    assert token_hit("운동", "운동했다고")
    assert not token_hit("친구", "친환경")


def test_content_tokens_exclude_temporal_and_intent_words() -> None:
    """Temporal and recall-intent tokens never count as content hits."""
    assert content_tokens("어제 아침에 기억나?") == ()
    assert content_tokens("어제 티라노사우루스가 기억나?") == ("티라노사우루스가",)


def test_content_tokens_keep_long_temporal_prefix_words() -> None:
    """Temporal prefix exclusion is bounded to particle-sized suffixes."""
    assert content_tokens("오늘은 기억나?") == ()
    assert content_tokens("오늘하루종일 기억나?") == ("오늘하루종일",)


def test_content_tokens_use_exact_stopword_matching() -> None:
    """Stopwords are exact-match so longer content words remain searchable."""
    assert content_tokens("누구인지 기억나?") == ("누구인지",)


def test_time_parser_vectors_and_false_positive_guard() -> None:
    """The deterministic time parser avoids duration false positives."""
    now = _kst(12, 20)

    assert parse_time_window("1박 2일 여행", now) is None
    yesterday = parse_time_window("어제 저녁에", now)

    assert yesterday is not None
    assert yesterday.start == _kst(11, 17)
    assert yesterday.end == _kst(11, 21)


def test_matcher_accepts_two_hits_and_rare_recent_single_hit() -> None:
    """Gate accepts either two hits or one rare recent token."""
    store = _store(
        [
            _snippet("a", "어제 티라노사우루스 그림을 그렸어", _kst(11, 19)),
            _snippet("b", "오늘 세종대왕은 한글 이야기를 했어", _kst(12, 8), source_hash=HASH_B),
        ]
    )
    now = _kst(12, 20)

    two_hit = store.recall("세종대왕이 한글 기억나?", now=now)
    rare_hit = store.recall("티라노사우루스가 기억나?", now=now)

    assert two_hit.match is not None
    assert two_hit.match.snippet.id == "b"
    assert rare_hit.match is not None
    assert rare_hit.match.snippet.id == "a"


def test_matcher_uses_surface_df_for_particle_variant_rarity() -> None:
    """Rare-token df is based on surface tokens, not stripped particle stems."""
    store = _store(
        [
            _snippet("target", "오늘 세종대왕은 한글 이야기를 했어", _kst(12, 9)),
            _snippet("same-surface", "오늘 세종대왕은 책에 나왔어", _kst(12, 8)),
            _snippet("particle-variant", "오늘 세종대왕이 훈민정음을 만들었어", _kst(12, 7)),
            _snippet("honorific-variant", "오늘 세종대왕님은 위인이야", _kst(12, 6)),
        ]
    )

    match = store.recall("세종대왕이 기억나?", now=_kst(12, 20)).match

    assert match is not None
    assert match.snippet.id == "target"


def test_matcher_uses_surface_df_when_stem_is_common() -> None:
    """Single-hit recall passes when the surface token is rare but the stem is common."""
    store = _store(
        [
            _snippet("target", "오늘 이름이 별이라고 했어", _kst(12, 9)),
            _snippet("stem-common-a", "오늘 이름은 아직 비밀이야", _kst(12, 8)),
            _snippet("stem-common-b", "오늘 이름을 종이에 썼어", _kst(12, 7)),
        ]
    )

    match = store.recall("이름이 기억나?", now=_kst(12, 20)).match

    assert match is not None
    assert match.snippet.id == "target"


def test_matcher_recency_uses_full_datetime_days() -> None:
    """The seven-day rare-hit gate uses timedelta days, not calendar-date subtraction."""
    now = datetime(2026, 6, 12, 0, 30, 0, tzinfo=KST)
    store = _store(
        [
            _snippet(
                "target",
                "반짝단어를 만들었어",
                datetime(2026, 6, 4, 23, 45, 0, tzinfo=KST),
            )
        ]
    )

    match = store.recall("반짝단어를 기억나?", now=now).match

    assert match is not None
    assert match.snippet.id == "target"


def test_matcher_rejects_common_single_hit_and_time_and_filters() -> None:
    """One common token is below threshold and time windows are AND filters."""
    snippets = [
        _snippet("old", "오늘 친구랑 운동했다고 말했어", _kst(12, 9)),
        _snippet("target", "어제 친구랑 운동했다고 말했어", _kst(11, 18), source_hash=HASH_B),
        _snippet("other", "그저께 친구랑 그림 그렸어", _kst(10, 18)),
    ]
    store = _store(snippets)
    now = _kst(12, 20)

    common = store.recall("친구 기억나?", now=now)
    yesterday = store.recall("어제 운동 기억나?", now=now)

    assert common.match is None
    assert yesterday.match is not None
    assert yesterday.match.snippet.id == "target"


def test_time_only_recall_requires_intent_and_three_day_bound() -> None:
    """Time-only recall is query-level, intent-gated, and capped at three days."""
    store = _store(
        [
            _snippet("yesterday", "블록 놀이를 했어", _kst(11, 9)),
            _snippet("old", "퍼즐 놀이를 했어", _kst(7, 9), source_hash=HASH_B),
        ]
    )
    now = _kst(12, 20)

    match = store.recall("어제 기억나?", now=now).match

    assert match is not None
    assert match.snippet.id == "yesterday"
    assert store.recall("어제 뭐 했어?", now=now).match is None
    assert store.recall("5일 전 기억나?", now=now).match is None


@pytest.mark.parametrize(
    "field",
    [
        "crisis_matched",
        "parent_disclosure_matched",
        "template_matched",
        "belief_matched",
        "content_filter_blocked",
        "history_mode_matched",
        "funny_english_matched",
        "language_switch_matched",
        "recall_query_matched",
    ],
)
def test_router_metric_context_exclusions(field: str) -> None:
    """Every fixed/router metric suppresses recall."""
    metrics = TurnMetrics()
    setattr(metrics, field, True)

    assert should_skip_recall_for_metrics(metrics)


def test_recall_query_matched_in_fixed_path_metric_fields() -> None:
    """Explicit-recall turns must be excluded from contextual injection."""
    assert "recall_query_matched" in _FIXED_PATH_METRIC_FIELDS


def test_recall_for_intent_returns_verbatim_name_snippet() -> None:
    """Name recall seeds the gate and returns the prior statement verbatim."""
    store = _store(
        [
            _snippet("a", "오늘 이름이 별이라고 했어", _kst(12, 9)),
            _snippet("b", "오늘 이름은 아직 비밀이야", _kst(12, 8)),
            _snippet("c", "오늘 이름을 종이에 썼어", _kst(12, 7)),
        ]
    )

    answer = store.recall_for_intent("name", "내 이름 뭐라고 했어?", now=_kst(12, 20))

    assert answer == "오늘 이름이 별이라고 했어"


@pytest.mark.parametrize(
    "text",
    [
        "그럼 내가 제일 좋아하는 과일은 뭐야?",
        "내 이름이 뭐야?",
        "편지 쓰면 좋아할까?",
        "너는 어느 부위가 제일 좋아?",
    ],
)
def test_is_interrogative_true_for_questions(text: str) -> None:
    """Questions (trailing ``?`` or wh-ending) are flagged interrogative."""
    from core.conversation_memory import _is_interrogative

    assert _is_interrogative(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "나의 이름은 종경.",
        "나는 딸기맛 사탕이 제일 좋아.",
        "그래도 동생이 귀여울 때도 있어",
        "달콤한 사탕 먹으면 기분이 진짜 좋아지지.",
    ],
)
def test_is_interrogative_false_for_declaratives(text: str) -> None:
    """Declarative statements are not flagged interrogative."""
    from core.conversation_memory import _is_interrogative

    assert _is_interrogative(text) is False


def test_recall_for_intent_prefers_declarative_over_interrogative() -> None:
    """A lower-scored declarative wins over a higher-scored question."""
    # The interrogative snippet is the most recent and keyword-richer, so it
    # would win on score; the predicate must exclude it and surface the
    # declarative statement instead.
    store = _store(
        [
            _snippet("q", "내가 제일 좋아하는 과일은 뭐야?", _kst(12, 10)),
            _snippet("d", "나는 제일 좋아하는 과일이 딸기야", _kst(11, 9)),
        ]
    )

    answer = store.recall_for_intent("name", "내가 제일 좋아하는 과일 뭐야?", now=_kst(12, 20))

    assert answer == "나는 제일 좋아하는 과일이 딸기야"


def test_recall_for_intent_returns_none_when_all_candidates_interrogative() -> None:
    """Recall returns None when every matching candidate is a question."""
    store = _store(
        [
            _snippet("q1", "내가 제일 좋아하는 과일은 뭐야?", _kst(12, 10)),
            _snippet("q2", "내가 제일 좋아하는 과일이 뭐니?", _kst(11, 9)),
        ]
    )

    assert (
        store.recall_for_intent("name", "내가 제일 좋아하는 과일 뭐야?", now=_kst(12, 20)) is None
    )


def test_recall_for_intent_returns_none_when_nothing_matches() -> None:
    """Recall returns None (honest not-found) when the gate fails."""
    store = _store([_snippet("x", "블록 놀이를 했어", _kst(11, 9))])

    assert store.recall_for_intent("name", "내 이름 뭐라고 했어?", now=_kst(12, 20)) is None


def test_recall_for_intent_drops_now_blocked_snippet() -> None:
    """A snippet blocked by a tightened blocklist is re-filtered out."""

    class _BlockAllFilter:
        def filter(self, text: str) -> Any:
            from safety.content_filter import FilterResult

            return FilterResult(allowed=False, original=text, filtered="", violations=["block"])

    store = _store(
        [_snippet("a", "오늘 이름이 별이라고 했어", _kst(12, 9))],
        content_filter=_BlockAllFilter(),
    )

    assert store.recall_for_intent("name", "내 이름 뭐라고 했어?", now=_kst(12, 20)) is None


def test_build_recall_message_drops_now_blocked_snippet() -> None:
    """The passive recall block is also defended by the content re-filter."""

    class _BlockAllFilter:
        def filter(self, text: str) -> Any:
            from safety.content_filter import FilterResult

            return FilterResult(allowed=False, original=text, filtered="", violations=["block"])

    store = _store(
        [_snippet("a", "어제 티라노사우루스 그림을 그렸어", _kst(11, 19))],
        content_filter=_BlockAllFilter(),
    )

    message = store.build_recall_message(
        "티라노사우루스가 기억나?",
        estimate_tokens=lambda text: 1,
        now=_kst(12, 20),
    )

    assert message is None


def test_first_turn_suppression_marker_is_loaded() -> None:
    """The v1 first-turn seam is suppressed after a quarantined previous day."""
    store = _store([], quarantined_days={"2026-06-11"})

    assert store.should_suppress_first_turn(now=_kst(12, 8))
    assert store.first_turn_message(estimate_tokens=lambda text: 1, now=_kst(12, 8)) is None


def test_flag_off_and_load_failure_disable_day_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Flag-off is a zero-behavior path and load failure returns None."""
    monkeypatch.delenv("MUNGI_CONV_MEMORY", raising=False)
    assert load_conversation_memory(tmp_path) is None

    monkeypatch.setenv("MUNGI_CONV_MEMORY", "1")
    assert load_conversation_memory(tmp_path) is None


def test_day_path_load_does_not_open_vector_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Daytime load touches JSON artifacts only, never vector or numpy files."""
    _write_generation(
        tmp_path,
        [_snippet("a", "어제 블록 놀이를 했어", _kst(11, 9))],
    )
    monkeypatch.setenv("MUNGI_CONV_MEMORY", "1")
    opened: list[str] = []
    original_open = Path.open

    def _recording_open(self: Path, *args: Any, **kwargs: Any) -> Any:
        opened.append(self.name)
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", _recording_open)

    assert load_conversation_memory(tmp_path) is not None
    assert "vectors.npy" not in opened
    assert not any(name.endswith(".npy") for name in opened)


def test_pipeline_injects_memory_before_current_user(monkeypatch: pytest.MonkeyPatch) -> None:
    """Recall block is inserted immediately before the current user message."""
    monkeypatch.delenv("MUNGI_CONV_MEMORY", raising=False)

    # Pin the recall clock so the snippet stays "recent" regardless of the real
    # date (the pipeline calls build_recall_message without an explicit ``now``,
    # which falls back to ``datetime.now(KST)``; an unpinned clock made this test
    # date-dependent and it broke after the 2026-06-12 -> 06-13 rollover).
    class _FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz: object | None = None) -> datetime:  # type: ignore[override]
            return _kst(12, 20)

    monkeypatch.setattr("core.conversation_memory.datetime", _FrozenDatetime)
    backend = LLMBackendConfig(
        backend="gemma4_text",
        model_path="/models/gemma.gguf",
        n_ctx=6144,
        max_tokens=256,
        temperature=0.4,
        n_gpu_layers=99,
    )
    with patch("core.pipeline.LLMBackendConfig.load", return_value=backend):
        pipeline = ConversationPipeline(MagicMock(), PipelineConfig())
    pipeline._conversation_memory = _store(
        [_snippet("a", "어제 저녁에 티라노사우루스 그림을 그렸어", _kst(11, 19))]
    )

    messages = pipeline._build_messages("티라노사우루스가 기억나?", metrics=TurnMetrics())

    assert messages[-2]["role"] == "user"
    assert messages[-2]["content"].startswith("[기억] 어제 저녁에")
    assert messages[-1] == {"role": "user", "content": "티라노사우루스가 기억나?"}


def test_pipeline_skips_memory_on_fixed_metric(monkeypatch: pytest.MonkeyPatch) -> None:
    """LLM guide/template turns do not receive recall blocks."""
    monkeypatch.delenv("MUNGI_CONV_MEMORY", raising=False)
    pipeline = ConversationPipeline(MagicMock(), PipelineConfig())
    pipeline._conversation_memory = _store(
        [_snippet("a", "어제 저녁에 티라노사우루스 그림을 그렸어", _kst(11, 19))]
    )
    metrics = TurnMetrics(template_matched=True)

    messages = pipeline._build_messages("티라노사우루스가 기억나?", metrics=metrics)

    assert all(not message["content"].startswith("[기억]") for message in messages)


def test_pipeline_trim_ladder_drops_history_before_recall(monkeypatch: pytest.MonkeyPatch) -> None:
    """The prompt budget ladder removes history before trimming/omitting recall."""
    monkeypatch.delenv("MUNGI_CONV_MEMORY", raising=False)
    backend = LLMBackendConfig(
        backend="qwen3_legacy",
        model_path="/models/gemma.gguf",
        n_ctx=330,
        max_tokens=64,
        temperature=0.4,
        n_gpu_layers=99,
    )
    with patch("core.pipeline.LLMBackendConfig.load", return_value=backend):
        pipeline = ConversationPipeline(
            MagicMock(),
            PipelineConfig(max_history_turns=2, llm_system_prompt="system"),
        )
    pipeline._history = [
        {"role": "user", "text": "older user " * 12},
        {"role": "assistant", "text": "older assistant " * 12},
        {"role": "user", "text": "recent user"},
        {"role": "assistant", "text": "recent assistant"},
    ]
    pipeline._conversation_memory = _store(
        [_snippet("a", "어제 저녁에 티라노사우루스 그림을 그렸어", _kst(11, 19))]
    )

    class _FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz: object | None = None) -> datetime:  # type: ignore[override]
            return _kst(12, 20)

    monkeypatch.setattr("core.conversation_memory.datetime", _FrozenDatetime)

    messages = pipeline._build_messages("티라노사우루스가 기억나?", metrics=TurnMetrics())
    contents = [message["content"] for message in messages]

    assert "older user " * 12 not in contents
    assert "recent user" in contents
    assert any(content.startswith("[기억]") for content in contents)


def test_trim_recall_content_hard_cap() -> None:
    """Recall messages are capped by the provided pipeline estimator."""
    long_content = "[기억] " + ("아주긴문장" * 80)

    trimmed = trim_recall_content(long_content, lambda text: len(text) // 2)

    assert trimmed.startswith("[기억] ")
    assert len(trimmed) < len(long_content)
