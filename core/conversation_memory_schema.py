"""Shared schemas for conversation-memory night and day paths.

The live ``conversation.jsonl`` rows predate this module and do not carry an
embedded schema version. ``SCHEMA_VERSION`` therefore versions the artifact
contracts introduced here, while the turn reader treats historical row shapes
as presence-based variants and defaults absent optional flags to safe ``False``
values.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal, TypeAlias, cast

JsonDict: TypeAlias = dict[str, object]
IndexLayer: TypeAlias = Literal["turns", "summaries"]

KST = timezone(timedelta(hours=9))
SCHEMA_VERSION = 1

TOP_LEVEL_QUARANTINE_FLAGS = (
    "hotword_hallucination_detected",
    "stt_script_drift_detected",
)
METRIC_QUARANTINE_FLAGS = (
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
SESSION_QUARANTINE_FLAG = "crisis_matched"

DAY_SUMMARY_RETENTION_YEARS = 5
RAW_TURN_RETENTION_DAYS = 90
RECALL_INJECTION_HARD_CAP_TOKENS = 100
PREFIX_TOL = 3
CONVERSATION_MEMORY_ENV_FLAG = "MUNGI_CONV_MEMORY"
CONVERSATION_MEMORY_RUNTIME_SUBPATH = "conversation_memory"
GENERATION_POINTER_FILENAME = "current"

_VALID_INDEX_LAYERS = ("turns", "summaries")


class SchemaError(ValueError):
    """Raised when a conversation-memory record or artifact schema is invalid."""


@dataclass(frozen=True)
class ConversationTurnRecord:
    """One parsed turn from a session ``conversation.jsonl`` file."""

    timestamp: datetime
    turn: int
    user_text: str
    response_text: str
    input_wav: str | None
    output_wav: str | None
    hotword_hallucination_detected: bool
    hotword_hallucination_reason: str
    stt_script_drift_detected: bool
    metrics: dict[str, object]


@dataclass(frozen=True)
class TurnSnippet:
    """Raw-layer memory snippet written to ``turns.jsonl``."""

    id: str
    session_dir: str
    turn: int
    text: str
    timestamp: datetime
    source_hash: str

    def __post_init__(self) -> None:
        """Validate direct writer construction before serialization."""
        _validate_non_empty_text(self.id, "turn snippet id")
        _validate_non_empty_text(self.session_dir, "turn snippet session_dir")
        if self.turn < 0:
            raise SchemaError("turn snippet turn must be >= 0")
        _validate_non_empty_text(self.text, "turn snippet text")
        _normalize_kst_datetime(self.timestamp, "timestamp")
        _validate_non_empty_text(self.source_hash, "turn snippet source_hash")

    def to_json_dict(self) -> JsonDict:
        """Serialize the snippet with a second-precision KST timestamp."""
        return {
            "id": self.id,
            "session_dir": self.session_dir,
            "turn": self.turn,
            "text": self.text,
            "timestamp": _format_kst_datetime(self.timestamp, "timestamp"),
            "source_hash": self.source_hash,
        }

    @classmethod
    def from_json_dict(cls, obj: Mapping[str, object]) -> TurnSnippet:
        """Parse a raw-layer snippet from a JSON object."""
        payload = _require_mapping(obj, "turn snippet")
        return cls(
            id=_require_non_empty_str(payload, "id"),
            session_dir=_require_non_empty_str(payload, "session_dir"),
            turn=_require_int(payload, "turn", minimum=0),
            text=_require_non_empty_str(payload, "text"),
            timestamp=_parse_kst_datetime(payload.get("timestamp"), "timestamp"),
            source_hash=_require_non_empty_str(payload, "source_hash"),
        )


@dataclass(frozen=True)
class DaySummaryProvenance:
    """Source mapping for one summary-layer snippet."""

    session_dir: str
    turn_refs: tuple[str, ...]
    timestamp_range: tuple[datetime, datetime]
    source_hashes: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate required non-empty source mapping on direct construction."""
        _validate_non_empty_text(self.session_dir, "summary provenance session_dir")
        _validate_non_empty_text_tuple(self.turn_refs, "summary provenance turn_refs")
        _format_timestamp_range(self.timestamp_range)
        _validate_non_empty_text_tuple(self.source_hashes, "summary provenance source_hashes")

    def to_json_dict(self) -> JsonDict:
        """Serialize provenance as the required JSON mapping."""
        return {
            "session_dir": self.session_dir,
            "turn_refs": list(self.turn_refs),
            "timestamp_range": _format_timestamp_range(self.timestamp_range),
            "source_hashes": list(self.source_hashes),
        }

    @classmethod
    def from_json_dict(cls, obj: Mapping[str, object]) -> DaySummaryProvenance:
        """Parse and validate summary provenance."""
        payload = _require_mapping(obj, "summary provenance")
        return cls(
            session_dir=_require_non_empty_str(payload, "session_dir"),
            turn_refs=_require_non_empty_str_tuple(payload, "turn_refs"),
            timestamp_range=_parse_timestamp_range(payload.get("timestamp_range")),
            source_hashes=_require_non_empty_str_tuple(payload, "source_hashes"),
        )


@dataclass(frozen=True)
class DaySummary:
    """Summary-layer memory snippet written to ``summaries.jsonl``."""

    id: str
    text: str
    timestamp_range: tuple[datetime, datetime]
    provenance: DaySummaryProvenance

    def __post_init__(self) -> None:
        """Validate direct writer construction before serialization."""
        _validate_non_empty_text(self.id, "day summary id")
        _validate_non_empty_text(self.text, "day summary text")
        _format_timestamp_range(self.timestamp_range)
        if not isinstance(self.provenance, DaySummaryProvenance):
            raise SchemaError("day summary provenance must be DaySummaryProvenance")

    def to_json_dict(self) -> JsonDict:
        """Serialize the day summary with required provenance."""
        return {
            "id": self.id,
            "text": self.text,
            "timestamp_range": _format_timestamp_range(self.timestamp_range),
            "provenance": self.provenance.to_json_dict(),
        }

    @classmethod
    def from_json_dict(cls, obj: Mapping[str, object]) -> DaySummary:
        """Parse a summary-layer snippet from a JSON object."""
        payload = _require_mapping(obj, "day summary")
        provenance = payload.get("provenance")
        if provenance is None:
            raise SchemaError("day summary field 'provenance' is required")
        return cls(
            id=_require_non_empty_str(payload, "id"),
            text=_require_non_empty_str(payload, "text"),
            timestamp_range=_parse_timestamp_range(payload.get("timestamp_range")),
            provenance=DaySummaryProvenance.from_json_dict(
                _require_mapping(provenance, "day summary provenance")
            ),
        )


@dataclass(frozen=True)
class IndexReference:
    """One ``index.json`` reference to a raw or summary artifact row."""

    layer: IndexLayer
    id: str

    def __post_init__(self) -> None:
        """Validate direct writer construction before serialization."""
        if self.layer not in _VALID_INDEX_LAYERS:
            raise SchemaError("index reference field 'layer' must be 'turns' or 'summaries'")
        _validate_non_empty_text(self.id, "index reference id")

    def to_json_dict(self) -> JsonDict:
        """Serialize the index reference."""
        return {"layer": self.layer, "id": self.id}

    @classmethod
    def from_json_dict(cls, obj: Mapping[str, object]) -> IndexReference:
        """Parse an index reference and reject unknown layers."""
        payload = _require_mapping(obj, "index reference")
        raw_layer = _require_non_empty_str(payload, "layer")
        if raw_layer not in _VALID_INDEX_LAYERS:
            raise SchemaError("index reference field 'layer' must be 'turns' or 'summaries'")
        return cls(
            layer=cast(IndexLayer, raw_layer),
            id=_require_non_empty_str(payload, "id"),
        )


@dataclass(frozen=True)
class IndexEntry:
    """One keyword entry in ``index.json``."""

    keyword: str
    references: tuple[IndexReference, ...]

    def __post_init__(self) -> None:
        """Validate direct writer construction before serialization."""
        _validate_non_empty_text(self.keyword, "index entry keyword")
        if not self.references:
            raise SchemaError("index entry references must not be empty")
        if not all(isinstance(reference, IndexReference) for reference in self.references):
            raise SchemaError("index entry references must be IndexReference values")

    def to_json_dict(self) -> JsonDict:
        """Serialize as ``{keyword: [{layer, id}, ...]}``."""
        return {self.keyword: [reference.to_json_dict() for reference in self.references]}

    @classmethod
    def from_json_dict(cls, obj: Mapping[str, object]) -> IndexEntry:
        """Parse either the compact index mapping or an expanded keyword object."""
        payload = _require_mapping(obj, "index entry")
        if "keyword" in payload:
            keyword = _require_non_empty_str(payload, "keyword")
            raw_references = payload.get("references", payload.get("refs"))
        else:
            if len(payload) != 1:
                raise SchemaError("index entry must contain exactly one keyword")
            keyword, raw_references = next(iter(payload.items()))
            if not isinstance(keyword, str) or not keyword.strip():
                raise SchemaError("index entry keyword must be non-empty text")
        references = tuple(
            IndexReference.from_json_dict(_require_mapping(item, "index reference"))
            for item in _require_sequence(raw_references, "index entry references")
        )
        if not references:
            raise SchemaError("index entry references must not be empty")
        return cls(keyword=keyword, references=references)


@dataclass(frozen=True)
class ManifestEntry:
    """Processed-session manifest row keyed by session directory."""

    session_dir: str
    sha256: str
    processed_at: datetime

    def __post_init__(self) -> None:
        """Validate direct writer construction before serialization."""
        _validate_non_empty_text(self.session_dir, "manifest entry session_dir")
        _validate_sha256_text(self.sha256, "manifest entry sha256")
        _normalize_kst_datetime(self.processed_at, "processed_at")

    def to_json_dict(self) -> JsonDict:
        """Serialize as a session-dir-keyed manifest entry."""
        return {
            self.session_dir: {
                "sha256": self.sha256,
                "processed_at": _format_kst_datetime(self.processed_at, "processed_at"),
            }
        }

    @classmethod
    def from_json_dict(cls, obj: Mapping[str, object]) -> ManifestEntry:
        """Parse a session-dir-keyed processed-session manifest entry."""
        payload = _require_mapping(obj, "manifest entry")
        if "session_dir" in payload:
            return cls(
                session_dir=_require_non_empty_str(payload, "session_dir"),
                sha256=_require_sha256(payload, "sha256"),
                processed_at=_parse_kst_datetime(payload.get("processed_at"), "processed_at"),
            )
        if len(payload) != 1:
            raise SchemaError("manifest entry must contain exactly one session_dir key")
        session_dir, raw_entry = next(iter(payload.items()))
        if not isinstance(session_dir, str) or not session_dir.strip():
            raise SchemaError("manifest entry session_dir key must be non-empty text")
        entry = _require_mapping(raw_entry, "manifest entry value")
        return cls(
            session_dir=session_dir,
            sha256=_require_sha256(entry, "sha256"),
            processed_at=_parse_kst_datetime(entry.get("processed_at"), "processed_at"),
        )


@dataclass(frozen=True)
class SessionEndSentinel:
    """Clean session completion marker stored as ``session_end.json``."""

    ended_at: datetime
    turn_count: int

    def __post_init__(self) -> None:
        """Validate direct writer construction before serialization."""
        _normalize_kst_datetime(self.ended_at, "ended_at")
        if self.turn_count < 0:
            raise SchemaError("session end sentinel turn_count must be >= 0")

    def to_json_dict(self) -> JsonDict:
        """Serialize the session completion sentinel."""
        return {
            "ended_at": _format_kst_datetime(self.ended_at, "ended_at"),
            "turn_count": self.turn_count,
        }

    @classmethod
    def from_json_dict(cls, obj: Mapping[str, object]) -> SessionEndSentinel:
        """Parse a session completion sentinel."""
        payload = _require_mapping(obj, "session end sentinel")
        return cls(
            ended_at=_parse_kst_datetime(payload.get("ended_at"), "ended_at"),
            turn_count=_require_int(payload, "turn_count", minimum=0),
        )


def parse_turn_json_line(line: str) -> ConversationTurnRecord:
    """Parse one JSONL row and raise ``SchemaError`` for malformed JSON or shape."""
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise SchemaError("conversation JSONL row is malformed JSON") from exc
    if not isinstance(payload, Mapping):
        raise SchemaError("conversation JSONL row must be an object")
    return parse_turn_record(cast(Mapping[str, object], payload))


def parse_turn_record(obj: Mapping[str, object]) -> ConversationTurnRecord:
    """Parse one version-tolerant conversation turn record.

    Historical ``conversation.jsonl`` rows do not include a ``SCHEMA_VERSION`` field. The reader
    therefore recognizes all observed row variants by optional-key presence and treats absent
    boolean flag keys as ``False`` so older benign turns do not become quarantine false positives.
    Malformed rows raise ``SchemaError`` for caller skip-and-count accounting.
    """
    payload = _require_mapping(obj, "conversation turn record")
    metrics_raw = payload.get("metrics")
    if metrics_raw is None:
        metrics: dict[str, object] = {}
    else:
        metrics = dict(_require_mapping(metrics_raw, "conversation turn metrics"))

    return ConversationTurnRecord(
        timestamp=_parse_kst_datetime(payload.get("timestamp"), "timestamp"),
        turn=_require_int(payload, "turn", minimum=0),
        user_text=_require_str(payload, "user_text", allow_empty=True),
        response_text=_require_str(payload, "response_text", allow_empty=True),
        input_wav=_optional_str(payload.get("input_wav"), "input_wav"),
        output_wav=_optional_str(payload.get("output_wav"), "output_wav"),
        hotword_hallucination_detected=_optional_bool(
            payload,
            "hotword_hallucination_detected",
        ),
        hotword_hallucination_reason=_optional_reason(payload),
        stt_script_drift_detected=_optional_bool(payload, "stt_script_drift_detected"),
        metrics=metrics,
    )


def is_quarantined_turn(record: ConversationTurnRecord) -> bool:
    """Return whether the turn must be excluded from memory artifacts."""
    for flag in TOP_LEVEL_QUARANTINE_FLAGS:
        if getattr(record, flag) is True:
            return True
    return any(record.metrics.get(flag) is True for flag in METRIC_QUARANTINE_FLAGS)


def is_crisis_turn(record: ConversationTurnRecord) -> bool:
    """Return whether this turn triggers session-level quarantine."""
    return record.metrics.get(SESSION_QUARANTINE_FLAG) is True


def parse_generation_pointer(content: str) -> str:
    """Parse the generation id stored in the ``current`` pointer file."""
    lines = content.strip().splitlines()
    if len(lines) != 1:
        raise SchemaError("generation pointer must contain exactly one generation id")
    generation_id = lines[0].strip()
    if not generation_id:
        raise SchemaError("generation pointer id must not be empty")
    if generation_id in (".", "..") or "/" in generation_id or "\\" in generation_id:
        raise SchemaError("generation pointer id must be a plain generation directory name")
    return generation_id


def format_generation_pointer(generation_id: str) -> str:
    """Format a generation id as pointer-file content."""
    return f"{parse_generation_pointer(generation_id)}\n"


def _require_mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise SchemaError(f"{field} must be a JSON object")
    if not all(isinstance(key, str) for key in value):
        raise SchemaError(f"{field} keys must be text")
    return cast(Mapping[str, object], value)


def _require_sequence(value: object, field: str) -> Sequence[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise SchemaError(f"{field} must be a JSON array")
    return cast(Sequence[object], value)


def _require_str(
    payload: Mapping[str, object],
    key: str,
    *,
    allow_empty: bool = False,
) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise SchemaError(f"field {key!r} must be text")
    if not allow_empty and not value.strip():
        raise SchemaError(f"field {key!r} must be non-empty text")
    return value


def _require_non_empty_str(payload: Mapping[str, object], key: str) -> str:
    return _require_str(payload, key, allow_empty=False)


def _require_non_empty_str_value(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{field} must be non-empty text")
    return value


def _validate_non_empty_text(value: object, field: str) -> None:
    _require_non_empty_str_value(value, field)


def _validate_non_empty_text_tuple(values: tuple[str, ...], field: str) -> None:
    if not values:
        raise SchemaError(f"{field} must not be empty")
    for value in values:
        _validate_non_empty_text(value, f"{field} item")


def _require_non_empty_str_tuple(payload: Mapping[str, object], key: str) -> tuple[str, ...]:
    raw_values = _require_sequence(payload.get(key), f"field {key!r}")
    values = tuple(
        _require_non_empty_str_value(value, f"field {key!r} item") for value in raw_values
    )
    if not values:
        raise SchemaError(f"field {key!r} must not be empty")
    return values


def _require_int(payload: Mapping[str, object], key: str, *, minimum: int | None = None) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaError(f"field {key!r} must be an integer")
    if minimum is not None and value < minimum:
        raise SchemaError(f"field {key!r} must be >= {minimum}")
    return value


def _require_sha256(payload: Mapping[str, object], key: str) -> str:
    value = _require_non_empty_str(payload, key)
    _validate_sha256_text(value, f"field {key!r}")
    return value


def _validate_sha256_text(value: str, field: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdefABCDEF" for char in value):
        raise SchemaError(f"{field} must be a SHA-256 hex digest")


def _optional_str(value: object, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise SchemaError(f"field {field!r} must be text or null")
    return value


def _optional_bool(payload: Mapping[str, object], key: str) -> bool:
    if key not in payload:
        return False
    value = payload[key]
    if not isinstance(value, bool):
        raise SchemaError(f"field {key!r} must be boolean")
    return value


def _optional_reason(payload: Mapping[str, object]) -> str:
    value = payload.get("hotword_hallucination_reason", "clean")
    if not isinstance(value, str) or not value.strip():
        raise SchemaError("field 'hotword_hallucination_reason' must be non-empty text")
    return value


def _parse_kst_datetime(value: object, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"field {field!r} must be an aware ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise SchemaError(f"field {field!r} must be an aware ISO timestamp") from exc
    return _normalize_kst_datetime(parsed, field)


def _normalize_kst_datetime(value: datetime, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise SchemaError(f"field {field!r} must be timezone-aware")
    return value.astimezone(KST)


def _format_kst_datetime(value: datetime, field: str) -> str:
    kst_value = _normalize_kst_datetime(value, field)
    return kst_value.replace(microsecond=0).isoformat(timespec="seconds")


def _parse_timestamp_range(value: object) -> tuple[datetime, datetime]:
    if isinstance(value, Mapping):
        payload = _require_mapping(value, "timestamp range")
        start = _parse_kst_datetime(payload.get("start"), "timestamp_range.start")
        end = _parse_kst_datetime(payload.get("end"), "timestamp_range.end")
    else:
        raw_range = _require_sequence(value, "timestamp range")
        if len(raw_range) != 2:
            raise SchemaError("timestamp range must contain start and end")
        start = _parse_kst_datetime(raw_range[0], "timestamp_range.start")
        end = _parse_kst_datetime(raw_range[1], "timestamp_range.end")
    if end < start:
        raise SchemaError("timestamp range end must be >= start")
    return (start, end)


def _format_timestamp_range(timestamp_range: tuple[datetime, datetime]) -> JsonDict:
    if len(timestamp_range) != 2:
        raise SchemaError("timestamp range must contain start and end")
    start, end = timestamp_range
    start_kst = _normalize_kst_datetime(start, "timestamp_range.start")
    end_kst = _normalize_kst_datetime(end, "timestamp_range.end")
    if end_kst < start_kst:
        raise SchemaError("timestamp range end must be >= start")
    return {
        "start": _format_kst_datetime(start_kst, "timestamp_range.start"),
        "end": _format_kst_datetime(end_kst, "timestamp_range.end"),
    }


__all__ = [
    "CONVERSATION_MEMORY_ENV_FLAG",
    "CONVERSATION_MEMORY_RUNTIME_SUBPATH",
    "DAY_SUMMARY_RETENTION_YEARS",
    "GENERATION_POINTER_FILENAME",
    "KST",
    "METRIC_QUARANTINE_FLAGS",
    "PREFIX_TOL",
    "RAW_TURN_RETENTION_DAYS",
    "RECALL_INJECTION_HARD_CAP_TOKENS",
    "SCHEMA_VERSION",
    "SESSION_QUARANTINE_FLAG",
    "TOP_LEVEL_QUARANTINE_FLAGS",
    "ConversationTurnRecord",
    "DaySummary",
    "DaySummaryProvenance",
    "IndexEntry",
    "IndexLayer",
    "IndexReference",
    "ManifestEntry",
    "SchemaError",
    "SessionEndSentinel",
    "TurnSnippet",
    "format_generation_pointer",
    "is_crisis_turn",
    "is_quarantined_turn",
    "parse_generation_pointer",
    "parse_turn_json_line",
    "parse_turn_record",
]
