"""Run the Stage 1 real-voice STT spot-check gate.

The helper exercises only the live audio preparation, VAD, and STT portions of
``ConversationPipeline`` against a deterministic balanced KO/EN fixture sample.
It intentionally exits before LLM/TTS stages.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if TYPE_CHECKING:
    from core.model_manager import ModelManager
    from core.pipeline import ConversationPipeline

logger = logging.getLogger("mungi.scripts.voice_stt_spot_check")

TARGET_SAMPLE_RATE: int = 16000
DEFAULT_SAMPLE_COUNT: int = 10
DEFAULT_RANDOM_SEED: int = 42
DEFAULT_CER_THRESHOLD: float = 0.30
DEFAULT_AGGREGATE_PASS_RATIO: float = 0.90


@dataclass(frozen=True)
class QueryEntry:
    """One PR5-100 query-pool entry needed by the STT spot-check."""

    query_id: int
    lang: str
    query: str


@dataclass(frozen=True)
class SpotCheckRecord:
    """Serializable result for one selected fixture."""

    query_id: int
    fixture_path: Path
    expected_query: str
    user_text: str
    cer: float
    speech_segments: int
    hotword_hallucination: bool
    script_drift: bool
    pass_bool: bool

    def to_json(self) -> dict[str, object]:
        """Return this record using the D4 JSON field names."""
        return {
            "id": self.query_id,
            "fixture_path": str(self.fixture_path),
            "expected_query": self.expected_query,
            "user_text": self.user_text,
            "cer": self.cer,
            "speech_segments": self.speech_segments,
            "hotword_hallucination": self.hotword_hallucination,
            "script_drift": self.script_drift,
            "pass_bool": self.pass_bool,
        }


@dataclass(frozen=True)
class SpotCheckRuntime:
    """Runtime objects shared across fixture checks."""

    manager: ModelManager
    pipeline: ConversationPipeline


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the STT spot-check helper."""
    parser = argparse.ArgumentParser(
        description=(
            "Run a deterministic balanced KO/EN STT-only spot-check over generated voice fixtures."
        ),
    )
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        required=True,
        help="Directory containing generated query_<NNN>_<lang>.wav fixtures.",
    )
    parser.add_argument(
        "--pool",
        type=Path,
        required=True,
        help="Path to pr5_100_query_pool.json.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=DEFAULT_SAMPLE_COUNT,
        help="Total balanced KO/EN fixtures to sample (default: 10).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Deterministic sample-selection seed (default: 42).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON path for spot-check records and summary.",
    )
    parser.add_argument(
        "--cer-threshold",
        type=float,
        default=DEFAULT_CER_THRESHOLD,
        help="Maximum normalized CER for one fixture to pass (default: 0.30).",
    )
    return parser


def load_query_pool(path: Path) -> list[QueryEntry]:
    """Load and validate the PR5-100 query pool."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        msg = f"Failed to read query pool: {path} ({exc})"
        raise ValueError(msg) from exc
    except json.JSONDecodeError as exc:
        msg = f"Query pool JSON is invalid: {path} ({exc})"
        raise ValueError(msg) from exc

    if not isinstance(payload, list):
        msg = f"Query pool must be a JSON list: {path}"
        raise ValueError(msg)

    entries: list[QueryEntry] = []
    for index, raw_entry in enumerate(payload, start=1):
        if not isinstance(raw_entry, dict):
            msg = f"Query pool entry {index} must be an object."
            raise ValueError(msg)

        raw_id = raw_entry.get("id")
        raw_lang = raw_entry.get("lang")
        raw_query = raw_entry.get("query")
        if not isinstance(raw_id, int):
            msg = f"Query pool entry {index} has non-integer id: {raw_id!r}"
            raise ValueError(msg)
        if not isinstance(raw_lang, str):
            msg = f"Query pool entry {index} has non-string lang: {raw_lang!r}"
            raise ValueError(msg)
        if not isinstance(raw_query, str) or not raw_query.strip():
            msg = f"Query pool entry {index} has invalid query text."
            raise ValueError(msg)

        lang = raw_lang.strip().lower()
        if lang not in {"ko", "en"}:
            msg = f"Query pool entry {raw_id} has unsupported lang: {raw_lang!r}"
            raise ValueError(msg)

        entries.append(QueryEntry(query_id=raw_id, lang=lang, query=raw_query))

    if not entries:
        msg = f"Query pool is empty: {path}"
        raise ValueError(msg)
    return entries


def select_balanced_entries(
    entries: Sequence[QueryEntry],
    sample_count: int,
    random_seed: int,
) -> list[QueryEntry]:
    """Select a deterministic balanced KO/EN sample from the pool."""
    if sample_count <= 0:
        msg = "--sample-count must be greater than 0."
        raise ValueError(msg)

    ko_target = (sample_count + 1) // 2
    en_target = sample_count // 2
    by_lang = {
        "ko": [entry for entry in entries if entry.lang == "ko"],
        "en": [entry for entry in entries if entry.lang == "en"],
    }

    if len(by_lang["ko"]) < ko_target or len(by_lang["en"]) < en_target:
        msg = (
            "Query pool does not contain enough KO/EN entries for balanced "
            f"sample_count={sample_count}: ko={len(by_lang['ko'])}, en={len(by_lang['en'])}"
        )
        raise ValueError(msg)

    rng = random.Random(random_seed)
    selected = [
        *rng.sample(by_lang["ko"], ko_target),
        *rng.sample(by_lang["en"], en_target),
    ]
    return sorted(selected, key=lambda entry: entry.query_id)


def fixture_path_for_entry(fixture_dir: Path, entry: QueryEntry) -> Path:
    """Return the expected fixture WAV path for a query-pool entry."""
    return fixture_dir / f"query_{entry.query_id:03d}_{entry.lang}.wav"


def validate_selected_fixture_paths(fixture_dir: Path, entries: Sequence[QueryEntry]) -> None:
    """Fail before model initialization when selected fixtures are missing."""
    missing_paths = [
        fixture_path_for_entry(fixture_dir, entry)
        for entry in entries
        if not fixture_path_for_entry(fixture_dir, entry).is_file()
    ]
    if missing_paths:
        preview = ", ".join(str(path) for path in missing_paths[:5])
        suffix = "" if len(missing_paths) <= 5 else f", ... (+{len(missing_paths) - 5} more)"
        msg = f"Missing selected fixture WAV(s): {preview}{suffix}"
        raise FileNotFoundError(msg)


def create_runtime() -> SpotCheckRuntime:
    """Create the model manager and pipeline with Stage 1 playback disabled."""
    from core.model_manager import ManagerConfig, ModelManager
    from core.pipeline import ConversationPipeline, PipelineConfig

    manager = ModelManager(ManagerConfig())
    manager.initialize()
    config = PipelineConfig(
        enable_content_filter=False,
        enable_stt_preload=False,
        play_tts_audio=False,
        tts_output_device=None,
    )
    pipeline = ConversationPipeline(manager, config)
    if (
        pipeline._config.play_tts_audio is not False
        or pipeline._config.tts_output_device is not None
    ):
        msg = "Stage 1 playback invariants were not applied."
        raise RuntimeError(msg)
    return SpotCheckRuntime(manager=manager, pipeline=pipeline)


def _read_wav_float32(path: Path) -> tuple[Any, int]:
    """Read a WAV as float32 samples and validate its sample rate."""
    try:
        import soundfile as sf  # type: ignore[import-not-found, import-untyped]
    except ImportError as exc:
        msg = "soundfile package is required for STT spot-check WAV loading."
        raise RuntimeError(msg) from exc

    try:
        samples, sample_rate = sf.read(path, dtype="float32")
    except OSError as exc:
        msg = f"Failed to read WAV fixture: {path} ({exc})"
        raise ValueError(msg) from exc

    sample_rate_int = int(sample_rate)
    if sample_rate_int != TARGET_SAMPLE_RATE:
        msg = f"Fixture must be {TARGET_SAMPLE_RATE} Hz: {path} is {sample_rate_int} Hz"
        raise ValueError(msg)
    return samples, sample_rate_int


def _is_hangul_char(char: str) -> bool:
    """Return whether a character is in a Hangul block used by Korean text."""
    codepoint = ord(char)
    return (
        0xAC00 <= codepoint <= 0xD7A3
        or 0x1100 <= codepoint <= 0x11FF
        or 0x3130 <= codepoint <= 0x318F
        or 0xA960 <= codepoint <= 0xA97F
        or 0xD7B0 <= codepoint <= 0xD7FF
    )


def _is_cer_char(char: str) -> bool:
    """Return whether a normalized CER character should be preserved."""
    return (char.isascii() and char.isalnum()) or _is_hangul_char(char)


def normalize_for_cer(text: str) -> str:
    """Normalize text for D4 CER: lowercase and keep only Hangul or ASCII alnum."""
    normalized = text.lower()
    return "".join(char for char in normalized if _is_cer_char(char))


def levenshtein_distance(source: Sequence[str], target: Sequence[str]) -> int:
    """Compute Levenshtein edit distance for two generic sequences."""
    if not source:
        return len(target)
    if not target:
        return len(source)

    previous = list(range(len(target) + 1))
    for row_index, source_item in enumerate(source, start=1):
        current = [row_index]
        for column_index, target_item in enumerate(target, start=1):
            cost = 0 if source_item == target_item else 1
            current.append(
                min(
                    previous[column_index] + 1,
                    current[column_index - 1] + 1,
                    previous[column_index - 1] + cost,
                ),
            )
        previous = current
    return previous[-1]


def compute_cer(reference: str, hypothesis: str) -> float:
    """Compute normalized character error rate against the expected query."""
    normalized_reference = normalize_for_cer(reference)
    normalized_hypothesis = normalize_for_cer(hypothesis)
    edits = levenshtein_distance(list(normalized_reference), list(normalized_hypothesis))
    return edits / max(len(normalized_reference), 1)


def compute_aggregate_threshold(sample_count: int) -> int:
    """Return the adjusted aggregate-pass threshold for the selected sample count."""
    if sample_count <= 0:
        msg = "sample_count must be greater than 0."
        raise ValueError(msg)
    return math.ceil(sample_count * DEFAULT_AGGREGATE_PASS_RATIO)


def _failed_record(entry: QueryEntry, fixture_path: Path, cer: float) -> SpotCheckRecord:
    """Create a failed record for a fixture that could not complete VAD/STT."""
    return SpotCheckRecord(
        query_id=entry.query_id,
        fixture_path=fixture_path,
        expected_query=entry.query,
        user_text="",
        cer=cer,
        speech_segments=0,
        hotword_hallucination=False,
        script_drift=False,
        pass_bool=False,
    )


def run_fixture_check(
    runtime: SpotCheckRuntime,
    entry: QueryEntry,
    fixture_path: Path,
    cer_threshold: float,
) -> SpotCheckRecord:
    """Run the exact D4 VAD/STT spot-check sequence for one fixture."""
    from core.model_manager import ModelType
    from core.pipeline import _contains_non_target_script, _is_hotword_hallucination

    samples, sample_rate = _read_wav_float32(fixture_path)
    prepared = runtime.pipeline._prepare_input_audio(samples, sample_rate)
    segments = runtime.pipeline._run_vad(prepared)
    speech_segments = len(segments)

    if speech_segments == 0:
        cer = compute_cer(entry.query, "")
        return _failed_record(entry, fixture_path, cer)

    speech_audio = runtime.pipeline._extract_speech(prepared, segments)
    runtime.pipeline._mm.load(ModelType.STT)
    user_text = runtime.pipeline._run_stt(speech_audio)
    user_text = runtime.pipeline._normalize_stt_text(user_text)
    hotword_hallucination = _is_hotword_hallucination(
        user_text,
        runtime.pipeline._active_hotwords_csv(),
    )
    script_drift = _contains_non_target_script(user_text)
    cer = compute_cer(entry.query, user_text)
    pass_bool = (
        bool(user_text.strip())
        and not script_drift
        and not hotword_hallucination
        and cer <= cer_threshold
        and speech_segments >= 1
    )

    return SpotCheckRecord(
        query_id=entry.query_id,
        fixture_path=fixture_path,
        expected_query=entry.query,
        user_text=user_text,
        cer=cer,
        speech_segments=speech_segments,
        hotword_hallucination=hotword_hallucination,
        script_drift=script_drift,
        pass_bool=pass_bool,
    )


def run_spot_check(
    fixture_dir: Path,
    pool_path: Path,
    sample_count: int,
    random_seed: int,
    cer_threshold: float,
) -> tuple[list[SpotCheckRecord], bool]:
    """Run all selected fixtures and return records plus aggregate status."""
    if cer_threshold < 0:
        msg = "--cer-threshold must be >= 0."
        raise ValueError(msg)
    if not fixture_dir.exists():
        msg = f"Fixture directory does not exist: {fixture_dir}"
        raise FileNotFoundError(msg)
    if not fixture_dir.is_dir():
        msg = f"Fixture path is not a directory: {fixture_dir}"
        raise NotADirectoryError(msg)

    entries = load_query_pool(pool_path)
    selected_entries = select_balanced_entries(entries, sample_count, random_seed)
    validate_selected_fixture_paths(fixture_dir, selected_entries)
    logger.info(
        "spot_check_selection sample_count=%d random_seed=%d selected_ids=%s",
        sample_count,
        random_seed,
        ",".join(str(entry.query_id) for entry in selected_entries),
    )

    runtime = create_runtime()
    records: list[SpotCheckRecord] = []
    try:
        for entry in selected_entries:
            fixture_path = fixture_path_for_entry(fixture_dir, entry)
            try:
                record = run_fixture_check(runtime, entry, fixture_path, cer_threshold)
            except Exception as exc:
                cer = compute_cer(entry.query, "")
                record = _failed_record(entry, fixture_path, cer)
                logger.error(
                    "fixture_result id=%d lang=%s pass=false error=%s",
                    entry.query_id,
                    entry.lang,
                    exc,
                    exc_info=True,
                )
            else:
                logger.info(
                    "fixture_result id=%d lang=%s pass=%s cer=%.4f speech_segments=%d "
                    "hotword_hallucination=%s script_drift=%s",
                    entry.query_id,
                    entry.lang,
                    str(record.pass_bool).lower(),
                    record.cer,
                    record.speech_segments,
                    str(record.hotword_hallucination).lower(),
                    str(record.script_drift).lower(),
                )
            records.append(record)
    finally:
        runtime.manager.unload_all(force=True)

    pass_count = sum(1 for record in records if record.pass_bool)
    aggregate_threshold = compute_aggregate_threshold(len(records))
    aggregate_pass = pass_count >= aggregate_threshold
    logger.info(
        "spot_check_summary total=%d pass_count=%d threshold=%d aggregate_pass=%s",
        len(records),
        pass_count,
        aggregate_threshold,
        str(aggregate_pass).lower(),
    )
    return records, aggregate_pass


def write_output(path: Path, records: Sequence[SpotCheckRecord], aggregate_pass: bool) -> None:
    """Write per-fixture records plus the trailing D4 summary object."""
    pass_count = sum(1 for record in records if record.pass_bool)
    payload: list[dict[str, object]] = [record.to_json() for record in records]
    payload.append(
        {
            "summary": {
                "total": len(records),
                "pass_count": pass_count,
                "aggregate_pass": aggregate_pass,
            },
        },
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        records, aggregate_pass = run_spot_check(
            fixture_dir=args.fixture_dir,
            pool_path=args.pool,
            sample_count=args.sample_count,
            random_seed=args.random_seed,
            cer_threshold=args.cer_threshold,
        )
        write_output(args.output, records, aggregate_pass)
        logger.info("Wrote STT spot-check result to %s", args.output)
        return 0 if aggregate_pass else 1
    except (FileNotFoundError, NotADirectoryError, RuntimeError, ValueError) as exc:
        logger.error("STT spot-check failed: %s", exc)
        return 1
    except Exception as exc:
        logger.error("STT spot-check failed unexpectedly: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
