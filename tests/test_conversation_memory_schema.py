"""Tests for shared conversation-memory schema contracts."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import pytest

from core.conversation_memory_schema import (
    CONVERSATION_MEMORY_ENV_FLAG,
    CONVERSATION_MEMORY_RUNTIME_SUBPATH,
    DAY_SUMMARY_RETENTION_YEARS,
    GENERATION_POINTER_FILENAME,
    KST,
    METRIC_QUARANTINE_FLAGS,
    PREFIX_TOL,
    RAW_TURN_RETENTION_DAYS,
    RECALL_INJECTION_HARD_CAP_TOKENS,
    SCHEMA_VERSION,
    SESSION_QUARANTINE_FLAG,
    TOP_LEVEL_QUARANTINE_FLAGS,
    DaySummary,
    DaySummaryProvenance,
    IndexEntry,
    IndexReference,
    ManifestEntry,
    SchemaError,
    SessionEndSentinel,
    TurnSnippet,
    format_generation_pointer,
    is_crisis_turn,
    is_quarantined_turn,
    parse_generation_pointer,
    parse_turn_json_line,
    parse_turn_record,
)

HASH_A = "a" * 64
HASH_B = "b" * 64
TURN_TIMESTAMP = "2026-06-12T14:03:22+09:00"
EXPECTED_TOP_LEVEL_QUARANTINE_FLAGS = (
    "hotword_hallucination_detected",
    "stt_script_drift_detected",
)
EXPECTED_METRIC_QUARANTINE_FLAGS = (
    "crisis_matched",
    "parent_disclosure_matched",
    "template_matched",
    "belief_matched",
    "content_filter_blocked",
    "history_mode_matched",
    "funny_english_matched",
    "language_switch_matched",
    "datetime_query_matched",
    "recall_query_matched",
    "hotword_hallucination_detected",
    "stt_script_drift_detected",
)


def _is_targeted_schema_test_run(args: list[str]) -> bool:
    target = Path("tests/test_conversation_memory_schema.py")
    return len(args) == 1 and Path(args[0]).as_posix() == target.as_posix()


@pytest.fixture(scope="session", autouse=True)
def _allow_targeted_schema_test_coverage(pytestconfig: pytest.Config) -> None:
    """Keep the single-file schema run from reporting unrelated packages."""
    if os.getenv("MUNGI_SCHEMA_STRICT_COVERAGE") == "1":
        return
    cov_plugin = pytestconfig.pluginmanager.get_plugin("_cov")
    if cov_plugin is None or not _is_targeted_schema_test_run(pytestconfig.args):
        return
    cov_plugin.options.cov_fail_under = 0


def _record(
    *,
    include_top_flags: bool = True,
    include_metrics: bool = True,
    metrics: dict[str, object] | None = None,
    **overrides: object,
) -> dict[str, object]:
    record: dict[str, object] = {
        "timestamp": TURN_TIMESTAMP,
        "turn": 3,
        "user_text": "어제 유치원에서 민지랑 블록 놀이했어",
        "response_text": "민지랑 블록 놀이를 했구나. 재미있었겠다.",
        "input_wav": "input_003.wav",
        "output_wav": "output_003.wav",
    }
    if include_top_flags:
        record.update(
            {
                "hotword_hallucination_detected": False,
                "hotword_hallucination_reason": "clean",
                "stt_script_drift_detected": False,
            }
        )
    if include_metrics:
        record["metrics"] = dict(metrics or {})
    record.update(overrides)
    return record


def _kst_dt(hour: int, minute: int = 0, second: int = 0) -> datetime:
    return datetime(2026, 6, 12, hour, minute, second, tzinfo=KST)


def test_parse_turn_record_accepts_all_four_real_schema_variants() -> None:
    """Historical rows default absent flag keys to false."""
    variants = [
        # G0.3 oldest variant: no top-level flags and no denylist metric keys.
        _record(include_top_flags=False, metrics={"llm_tokens": 12}),
        # G0.3 second variant: top-level hotword/drift flags only.
        _record(metrics={"llm_tokens": 13}),
        # G0.3 third variant: crisis/parent/belief metrics added.
        _record(
            metrics={
                "crisis_matched": False,
                "parent_disclosure_matched": False,
                "belief_matched": False,
            }
        ),
        # G0.3 current variant: history/FE/language-switch metrics added.
        _record(
            metrics={
                "crisis_matched": False,
                "parent_disclosure_matched": False,
                "belief_matched": False,
                "history_mode_matched": False,
                "funny_english_matched": False,
                "language_switch_matched": False,
            }
        ),
    ]

    parsed = [parse_turn_record(variant) for variant in variants]

    assert [item.hotword_hallucination_detected for item in parsed] == [False] * 4
    assert [item.stt_script_drift_detected for item in parsed] == [False] * 4
    assert parsed[0].hotword_hallucination_reason == "clean"
    assert parsed[0].timestamp.tzinfo == KST
    assert parsed[0].user_text.startswith("어제")
    assert not is_quarantined_turn(parsed[0])
    assert parsed[3].metrics["history_mode_matched"] is False


def test_quarantine_constants_match_g0_final_denylist() -> None:
    """Tests must fail if the final G0 denylist is weakened."""
    assert TOP_LEVEL_QUARANTINE_FLAGS == EXPECTED_TOP_LEVEL_QUARANTINE_FLAGS
    assert METRIC_QUARANTINE_FLAGS == EXPECTED_METRIC_QUARANTINE_FLAGS
    assert SESSION_QUARANTINE_FLAG == "crisis_matched"


@pytest.mark.parametrize("flag", ["datetime_query_matched", "recall_query_matched"])
def test_deterministic_intercept_flags_quarantine_turn(flag: str) -> None:
    """Datetime/recall intercept turns must be excluded from memory artifacts."""
    assert flag in METRIC_QUARANTINE_FLAGS
    assert is_quarantined_turn(parse_turn_record(_record(metrics={flag: True})))


def test_parse_turn_record_defaults_absent_metrics_to_empty_dict() -> None:
    """Older rows without metrics remain parseable."""
    parsed = parse_turn_record(_record(include_top_flags=False, include_metrics=False))

    assert parsed.metrics == {}
    assert not is_quarantined_turn(parsed)


@pytest.mark.parametrize("flag", TOP_LEVEL_QUARANTINE_FLAGS)
def test_top_level_quarantine_flags_drop_turn(flag: str) -> None:
    """Each top-level denylist flag quarantines a turn by itself."""
    record = _record()
    record[flag] = True
    parsed = parse_turn_record(record)

    assert is_quarantined_turn(parsed)


@pytest.mark.parametrize("flag", METRIC_QUARANTINE_FLAGS)
def test_metric_quarantine_flags_drop_turn(flag: str) -> None:
    """Each metric denylist flag quarantines a turn by itself."""
    parsed = parse_turn_record(_record(metrics={flag: True}))

    assert is_quarantined_turn(parsed)


def test_guide_template_turn_is_quarantined() -> None:
    """Guide-mode template turns still carry the template flag and are excluded."""
    parsed = parse_turn_record(
        _record(metrics={"template_matched": True, "template_mode": "guide"})
    )

    assert is_quarantined_turn(parsed)


def test_crisis_predicate_uses_session_quarantine_flag() -> None:
    """Only the crisis flag triggers the session-level predicate."""
    crisis = parse_turn_record(_record(metrics={SESSION_QUARANTINE_FLAG: True}))
    parent_disclosure = parse_turn_record(_record(metrics={"parent_disclosure_matched": True}))

    assert is_crisis_turn(crisis)
    assert not is_crisis_turn(parent_disclosure)


def test_parse_turn_record_rejects_malformed_records() -> None:
    """Malformed JSON shape and malformed timestamps raise SchemaError."""
    with pytest.raises(SchemaError, match="object"):
        parse_turn_record(cast(Any, []))
    with pytest.raises(SchemaError, match="timestamp"):
        parse_turn_record(_record(timestamp="not-a-timestamp"))
    with pytest.raises(SchemaError, match="malformed JSON"):
        parse_turn_json_line("{not json")
    with pytest.raises(SchemaError, match="boolean"):
        parse_turn_record(_record(hotword_hallucination_detected="false"))


def test_turn_snippet_round_trip_serializes_second_precision_kst() -> None:
    """Raw snippets round-trip through the JSON shape used by turns.jsonl."""
    utc_timestamp = datetime(2026, 6, 12, 5, 3, 22, 900000, tzinfo=timezone.utc)
    snippet = TurnSnippet(
        id="turn-20260612-0003",
        session_dir="sessions/2026-06-12/child-a",
        turn=3,
        text="민지랑 블록 놀이한 이야기",
        timestamp=utc_timestamp,
        source_hash=HASH_A,
    )

    payload = snippet.to_json_dict()
    parsed = TurnSnippet.from_json_dict(payload)

    assert payload["timestamp"] == "2026-06-12T14:03:22+09:00"
    assert parsed == TurnSnippet(
        id=snippet.id,
        session_dir=snippet.session_dir,
        turn=snippet.turn,
        text=snippet.text,
        timestamp=_kst_dt(14, 3, 22),
        source_hash=snippet.source_hash,
    )


def test_day_summary_and_provenance_round_trip() -> None:
    """Summary snippets require source provenance and preserve timestamp ranges."""
    provenance = DaySummaryProvenance(
        session_dir="sessions/2026-06-12/child-a",
        turn_refs=("turn-20260612-0003",),
        timestamp_range=(_kst_dt(14, 3, 22), _kst_dt(14, 4, 10)),
        source_hashes=(HASH_A,),
    )
    summary = DaySummary(
        id="summary-20260612-0001",
        text="민지와 블록 놀이를 한 일을 기억한다.",
        timestamp_range=(_kst_dt(14, 3, 22), _kst_dt(14, 4, 10)),
        provenance=provenance,
    )

    payload = summary.to_json_dict()
    parsed = DaySummary.from_json_dict(payload)

    assert parsed == summary
    assert parsed.provenance.source_hashes == (HASH_A,)


def test_day_summary_without_provenance_is_rejected() -> None:
    """Summary provenance is required and non-empty."""
    payload = DaySummary(
        id="summary-20260612-0001",
        text="아빠와 공원에 간 일을 기억한다.",
        timestamp_range=(_kst_dt(8), _kst_dt(8, 5)),
        provenance=DaySummaryProvenance(
            session_dir="sessions/2026-06-12/child-a",
            turn_refs=("turn-1",),
            timestamp_range=(_kst_dt(8), _kst_dt(8, 5)),
            source_hashes=(HASH_A,),
        ),
    ).to_json_dict()
    payload.pop("provenance")

    with pytest.raises(SchemaError, match="provenance"):
        DaySummary.from_json_dict(payload)
    with pytest.raises(SchemaError, match="turn_refs"):
        DaySummaryProvenance.from_json_dict(
            {
                "session_dir": "sessions/2026-06-12/child-a",
                "turn_refs": [],
                "timestamp_range": {
                    "start": "2026-06-12T08:00:00+09:00",
                    "end": "2026-06-12T08:05:00+09:00",
                },
                "source_hashes": [HASH_A],
            }
        )
    with pytest.raises(SchemaError, match="session_dir"):
        DaySummaryProvenance.from_json_dict(
            {
                "session_dir": "",
                "turn_refs": ["turn-1"],
                "timestamp_range": {
                    "start": "2026-06-12T08:00:00+09:00",
                    "end": "2026-06-12T08:05:00+09:00",
                },
                "source_hashes": [HASH_A],
            }
        )
    with pytest.raises(SchemaError, match="source_hashes"):
        DaySummaryProvenance.from_json_dict(
            {
                "session_dir": "sessions/2026-06-12/child-a",
                "turn_refs": ["turn-1"],
                "timestamp_range": {
                    "start": "2026-06-12T08:00:00+09:00",
                    "end": "2026-06-12T08:05:00+09:00",
                },
                "source_hashes": [],
            }
        )
    with pytest.raises(SchemaError, match="timestamp range"):
        DaySummaryProvenance.from_json_dict(
            {
                "session_dir": "sessions/2026-06-12/child-a",
                "turn_refs": ["turn-1"],
                "source_hashes": [HASH_A],
            }
        )
    with pytest.raises(SchemaError, match="turn_refs"):
        DaySummaryProvenance(
            session_dir="sessions/2026-06-12/child-a",
            turn_refs=(),
            timestamp_range=(_kst_dt(8), _kst_dt(8, 5)),
            source_hashes=(HASH_A,),
        )
    with pytest.raises(SchemaError, match="source_hashes"):
        DaySummaryProvenance(
            session_dir="sessions/2026-06-12/child-a",
            turn_refs=("turn-1",),
            timestamp_range=(_kst_dt(8), _kst_dt(8, 5)),
            source_hashes=(),
        )


def test_index_entry_round_trip_and_layer_validation() -> None:
    """Index entries use only the raw or summary artifact layers."""
    entry = IndexEntry(
        keyword="유치원",
        references=(
            IndexReference(layer="turns", id="turn-20260612-0003"),
            IndexReference(layer="summaries", id="summary-20260612-0001"),
        ),
    )

    assert IndexEntry.from_json_dict(entry.to_json_dict()) == entry
    expanded_payload = {
        "keyword": "블록",
        "references": [{"layer": "turns", "id": "turn-20260612-0003"}],
    }
    assert IndexEntry.from_json_dict(expanded_payload).keyword == "블록"
    with pytest.raises(SchemaError, match="layer"):
        IndexEntry.from_json_dict({"유치원": [{"layer": "vectors", "id": "vec-1"}]})
    with pytest.raises(SchemaError, match="layer"):
        IndexReference(layer=cast(Any, "vectors"), id="vec-1")


def test_manifest_sentinel_and_pointer_contracts_round_trip() -> None:
    """Manifest, sentinel, and generation pointer contracts stay parseable."""
    manifest = ManifestEntry(
        session_dir="sessions/2026-06-12/child-a",
        sha256=HASH_B,
        processed_at=_kst_dt(3),
    )
    sentinel = SessionEndSentinel(ended_at=_kst_dt(21, 10), turn_count=12)

    manifest_payload = manifest.to_json_dict()

    assert list(manifest_payload) == [manifest.session_dir]
    assert ManifestEntry.from_json_dict(manifest_payload) == manifest
    assert (
        ManifestEntry.from_json_dict(
            {
                "session_dir": manifest.session_dir,
                "sha256": manifest.sha256,
                "processed_at": "2026-06-12T03:00:00+09:00",
            }
        )
        == manifest
    )
    assert SessionEndSentinel.from_json_dict(sentinel.to_json_dict()) == sentinel
    assert GENERATION_POINTER_FILENAME == "current"
    assert format_generation_pointer("20260612T030000KST") == "20260612T030000KST\n"
    assert parse_generation_pointer("20260612T030000KST\n") == "20260612T030000KST"
    with pytest.raises(SchemaError, match="generation"):
        parse_generation_pointer("bad/generation\n")
    with pytest.raises(SchemaError, match="SHA-256"):
        ManifestEntry.from_json_dict(
            {
                "session_dir": "sessions/child-a",
                "sha256": "not-a-hash",
                "processed_at": TURN_TIMESTAMP,
            }
        )


def test_shared_constants_match_plan_values() -> None:
    """Shared constants encode the plan's cross-path values."""
    assert SCHEMA_VERSION == 1
    assert DAY_SUMMARY_RETENTION_YEARS == 5
    assert RAW_TURN_RETENTION_DAYS == 90
    assert RECALL_INJECTION_HARD_CAP_TOKENS == 100
    assert PREFIX_TOL == 3
    assert CONVERSATION_MEMORY_ENV_FLAG == "MUNGI_CONV_MEMORY"
    assert CONVERSATION_MEMORY_RUNTIME_SUBPATH == "conversation_memory"
    assert not CONVERSATION_MEMORY_RUNTIME_SUBPATH.startswith(("/", "\\"))


def test_timestamp_range_rejects_reversed_or_naive_values() -> None:
    """Artifact timestamps must stay aware and ordered."""
    with pytest.raises(SchemaError, match="end"):
        DaySummaryProvenance.from_json_dict(
            {
                "session_dir": "sessions/2026-06-12/child-a",
                "turn_refs": ["turn-1"],
                "timestamp_range": {
                    "start": "2026-06-12T09:00:00+09:00",
                    "end": "2026-06-12T08:00:00+09:00",
                },
                "source_hashes": [HASH_A],
            }
        )
    with pytest.raises(SchemaError, match="timezone-aware"):
        TurnSnippet(
            id="turn-1",
            session_dir="sessions/child-a",
            turn=1,
            text="오늘은 그림을 그렸어",
            timestamp=datetime(2026, 6, 12, 9, 0, 0),
            source_hash=HASH_A,
        ).to_json_dict()
