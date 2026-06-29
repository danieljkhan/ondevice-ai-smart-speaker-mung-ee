"""Run the accepted PR 5 100-query pool as independent voice-input turns."""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import logging
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("mungi.artifacts.pr5_100_voice")

STAGE1_PLAYBACK_DISABLED_REASON = "Stage 1 boundary (algorithmic latency only, no playback)"
EXPECTED_FIXTURE_SAMPLE_RATE = 16000
SUMMARY_LATENCY_FIELDS: tuple[str, ...] = (
    "vad_time_s",
    "stt_load_time_s",
    "stt_time_s",
    "llm_load_time_s",
    "llm_ttft_s",
    "llm_time_s",
    "tts_load_time_s",
    "tts_time_s",
    "playback_time_s",
    "total_time_s",
)


@dataclass(frozen=True)
class QueryEntry:
    """One accepted PR 5 query-pool entry."""

    id: int
    category: str
    lang: str
    age: str
    query: str
    recommended_answer: str
    test_criteria: str


@dataclass(frozen=True)
class FixtureEntry:
    """One voice fixture manifest entry."""

    query_id: int
    wav_path: Path
    wav_path_manifest: str
    lang: str
    sha256: str


def _repo_root() -> Path:
    """Return the repository root used for imports when this artifact is copied to /tmp."""
    env_root = os.getenv("MUNGI_REPO_ROOT", "").strip()
    if env_root:
        return Path(env_root).resolve()
    return Path.cwd().resolve()


def _prepare_imports() -> None:
    """Make repository modules importable before loading runtime dependencies."""
    repo_root = _repo_root()
    sys.path.insert(0, str(repo_root))


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_pool(path: Path) -> list[QueryEntry]:
    """Load and validate the accepted 100-query pool JSON."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        msg = f"Query pool must be a JSON list: {path}"
        raise ValueError(msg)

    entries: list[QueryEntry] = []
    seen_ids: set[int] = set()
    for item in data:
        if not isinstance(item, dict):
            msg = f"Query pool item must be an object: {item!r}"
            raise ValueError(msg)
        entry = QueryEntry(
            id=int(item["id"]),
            category=str(item["category"]),
            lang=str(item["lang"]).upper(),
            age=str(item["age"]),
            query=str(item["query"]),
            recommended_answer=str(item["recommended_answer"]),
            test_criteria=str(item["test_criteria"]),
        )
        if entry.id in seen_ids:
            msg = f"Duplicate query id: {entry.id}"
            raise ValueError(msg)
        seen_ids.add(entry.id)
        entries.append(entry)

    expected_ids = set(range(1, 101))
    actual_ids = {entry.id for entry in entries}
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        msg = f"Expected ids 1..100. missing={missing}, extra={extra}"
        raise ValueError(msg)
    return sorted(entries, key=lambda entry: entry.id)


def _resolve_manifest_wav_path(
    wav_path_value: str,
    *,
    fixture_dir: Path,
    manifest_path: Path,
) -> Path:
    """Resolve a manifest WAV path across the supported transfer layouts."""
    raw_path = Path(wav_path_value)
    if raw_path.is_absolute():
        return raw_path

    candidates = [
        fixture_dir / raw_path,
        manifest_path.parent / raw_path,
        fixture_dir.parent / raw_path,
        fixture_dir / raw_path.name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    # Return the first candidate so the caller's error message points at the
    # most direct fixture-dir-relative interpretation.
    return candidates[0].resolve()


def _load_manifest(path: Path, *, fixture_dir: Path) -> dict[int, FixtureEntry]:
    """Load the voice fixture manifest and index entries by query id."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        msg = f"Fixture manifest must be a JSON list: {path}"
        raise ValueError(msg)

    fixtures: dict[int, FixtureEntry] = {}
    for item in data:
        if not isinstance(item, dict):
            msg = f"Fixture manifest item must be an object: {item!r}"
            raise ValueError(msg)

        raw_query_id = item.get("query_id") if "query_id" in item else item.get("id")
        if raw_query_id is None:
            msg = f"Fixture manifest item missing query id: {item!r}"
            raise ValueError(msg)
        query_id = int(raw_query_id)
        if query_id in fixtures:
            msg = f"Duplicate fixture query id: {query_id}"
            raise ValueError(msg)

        wav_path_value = str(item["wav_path"])
        fixtures[query_id] = FixtureEntry(
            query_id=query_id,
            wav_path=_resolve_manifest_wav_path(
                wav_path_value,
                fixture_dir=fixture_dir,
                manifest_path=path,
            ),
            wav_path_manifest=wav_path_value,
            lang=str(item["lang"]).lower(),
            sha256=str(item["sha256"]).lower(),
        )

    return fixtures


def _select_entries(entries: list[QueryEntry], *, start_id: int, limit: int) -> list[QueryEntry]:
    """Return the selected query entries for one runner invocation."""
    if limit <= 0:
        msg = f"--limit must be positive, got {limit}"
        raise ValueError(msg)
    selected = [entry for entry in entries if entry.id >= start_id][:limit]
    if not selected:
        msg = "No query entries selected."
        raise ValueError(msg)
    return selected


def _verify_fixture(entry: QueryEntry, fixture: FixtureEntry) -> str:
    """Validate fixture language and SHA-256 against the pool entry."""
    expected_lang = entry.lang.lower()
    if fixture.lang != expected_lang:
        msg = (
            f"Fixture lang mismatch for id={entry.id}: "
            f"manifest={fixture.lang!r}, pool={expected_lang!r}"
        )
        raise ValueError(msg)
    if not fixture.wav_path.is_file():
        msg = f"Fixture WAV does not exist for id={entry.id}: {fixture.wav_path}"
        raise FileNotFoundError(msg)

    actual_sha256 = _sha256_file(fixture.wav_path)
    if actual_sha256.lower() != fixture.sha256:
        msg = (
            f"Fixture SHA-256 mismatch for id={entry.id}: "
            f"manifest={fixture.sha256}, actual={actual_sha256}"
        )
        raise ValueError(msg)
    return actual_sha256


def _load_fixture_wav(path: Path) -> Any:
    """Load and validate a mono 16 kHz fixture WAV."""
    import soundfile as sf  # type: ignore[import-not-found, import-untyped]

    samples, sample_rate = sf.read(path, dtype="float32")
    if sample_rate != EXPECTED_FIXTURE_SAMPLE_RATE:
        msg = f"Expected 16 kHz fixture WAV, got {sample_rate} Hz: {path}"
        raise ValueError(msg)
    if getattr(samples, "ndim", None) != 1:
        msg = f"Expected mono fixture WAV, got shape={getattr(samples, 'shape', None)}: {path}"
        raise ValueError(msg)
    return samples


def _voice_success(result: Any) -> bool:
    """Return the runner-derived voice path success boolean."""
    metrics = result.metrics
    return bool(
        result.success
        and metrics.speech_segments > 0
        and result.user_text != ""
        and not metrics.hotword_hallucination_detected
        and not metrics.stt_script_drift_detected
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Run the accepted PR 5 100-query pool as voice-input turns.",
    )
    parser.add_argument("--pool", type=Path, required=True, help="Path to PR 5 query pool JSON.")
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        required=True,
        help="Directory containing query_NNN_<lang>.wav files and manifest.json.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to fixture manifest JSON. Default: <fixture-dir>/manifest.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for rounds.jsonl, summary.json, tegrastats.log, and WAV files.",
    )
    parser.add_argument("--start-id", type=int, default=1, help="First query id to execute.")
    parser.add_argument("--limit", type=int, default=100, help="Maximum number of queries to run.")
    parser.add_argument("--llm-model-path", type=str, default=None)
    parser.add_argument("--model-dir", type=str, default="")
    parser.add_argument("--voice-style", type=str, default="F1")
    parser.add_argument("--llm-max-tokens", type=int, default=128)
    parser.add_argument("--max-history-turns", type=int, default=2)
    parser.add_argument("--presence-penalty", type=float, default=1.2)
    parser.add_argument(
        "--llm-resident",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Keep LLM resident between turns. Default follows ManagerConfig.",
    )
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument(
        "--no-save-wav",
        action="store_true",
        help="Do not persist per-turn synthesized WAV files.",
    )
    return parser


def _entry_record(
    entry: QueryEntry,
    fixture: FixtureEntry,
    fixture_sha256: str,
    result: Any,
    *,
    wav_path: Path | None,
    peak_memory_kb: int,
    elapsed_s: float,
) -> dict[str, Any]:
    """Serialize one executed voice-input query result."""
    return {
        **asdict(entry),
        "assistant_text": result.response_text,
        "tts_wav": str(wav_path) if wav_path is not None else None,
        "metrics": result.metrics.to_dict(),
        "success": result.success,
        "error": result.error,
        "peak_memory_kb": peak_memory_kb,
        "elapsed_s": round(elapsed_s, 3),
        "input_wav": str(fixture.wav_path),
        "input_wav_sha256": fixture_sha256,
        "raw_stt_text": result.raw_stt_text,
        "user_text": result.user_text,
        "fixture_lang_expected": fixture.lang,
        "voice_success": _voice_success(result),
    }


def _latency_stats(values: list[float]) -> dict[str, float]:
    """Return aggregate latency statistics for a non-empty value list."""
    sorted_values = sorted(values)
    p90_index = max(0, min(len(sorted_values) - 1, int(len(sorted_values) * 0.9 + 0.999) - 1))
    return {
        "mean": round(statistics.fmean(sorted_values), 3),
        "median": round(statistics.median(sorted_values), 3),
        "p90": round(sorted_values[p90_index], 3),
        "max": round(max(sorted_values), 3),
        "std": round(statistics.pstdev(sorted_values), 3) if len(sorted_values) > 1 else 0.0,
    }


def _build_latency_summary(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Build per-stage aggregate latency statistics from JSONL records."""
    summary: dict[str, dict[str, float]] = {}
    for field in SUMMARY_LATENCY_FIELDS:
        values: list[float] = []
        for record in records:
            metrics = record.get("metrics", {})
            if not isinstance(metrics, dict):
                continue
            value = metrics.get(field)
            if isinstance(value, int | float):
                values.append(float(value))
        if values:
            summary[field] = _latency_stats(values)
    return summary


def _build_summary(
    *,
    args: argparse.Namespace,
    manifest_path: Path,
    output_dir: Path,
    results_path: Path,
    thermal_summary_path: Path,
    records: list[dict[str, Any]],
    elapsed_s: float,
    thermal_logging_enabled: bool,
    thermal_summary: dict[str, Any],
    final_peak_memory_kb: int,
) -> dict[str, Any]:
    """Build the final summary.json payload."""
    success_count = sum(1 for record in records if record["success"])
    voice_success_count = sum(1 for record in records if record["voice_success"])
    both_success_count = sum(
        1 for record in records if record["success"] and record["voice_success"]
    )
    peak_values = [int(record["peak_memory_kb"]) for record in records]

    return {
        "pool": str(args.pool),
        "fixture_dir": str(args.fixture_dir),
        "manifest": str(manifest_path),
        "output_dir": str(output_dir),
        "results_jsonl": str(results_path),
        "query_count": len(records),
        "success_count": success_count,
        "voice_success_count": voice_success_count,
        "both_success_count": both_success_count,
        "failure_count": len(records) - success_count,
        "voice_failure_count": len(records) - voice_success_count,
        "elapsed_s": elapsed_s,
        "latency_stats": _build_latency_summary(records),
        "peak_memory_kb": max([final_peak_memory_kb, *peak_values])
        if peak_values
        else final_peak_memory_kb,
        "playback_enabled": False,
        "playback_force_disabled_reason": STAGE1_PLAYBACK_DISABLED_REASON,
        "output_device": None,
        "wav_saved": not args.no_save_wav,
        "tegrastats_log": str(output_dir / "tegrastats.log") if thermal_logging_enabled else None,
        "thermal_summary_json": str(thermal_summary_path) if thermal_logging_enabled else None,
        "thermal_summary": thermal_summary if thermal_logging_enabled else None,
    }


def main() -> int:
    """Run the accepted PR 5 query pool through the voice pipeline."""
    args = _build_parser().parse_args()
    _prepare_imports()

    from core.model_manager import ManagerConfig, ModelManager
    from core.pipeline import ConversationPipeline, PipelineConfig
    from scripts.e2e_60rounds_text_tts import (
        TegrastatsMonitor,
        _build_thermal_summary,
        _save_turn_wav,
        _slugify,
        run_preflight,
    )
    from scripts.utils import get_peak_memory_kb

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    fixture_dir = args.fixture_dir.resolve()
    manifest_path = (args.manifest or fixture_dir / "manifest.json").resolve()
    entries = _select_entries(_load_pool(args.pool), start_id=args.start_id, limit=args.limit)
    fixtures = _load_manifest(manifest_path, fixture_dir=fixture_dir)

    run_preflight(skip=args.skip_preflight)

    output_dir = args.output_dir.resolve()
    wav_dir = output_dir / "tts_wavs"
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "rounds.jsonl"
    summary_path = output_dir / "summary.json"
    tegrastats_log_path = output_dir / "tegrastats.log"
    thermal_summary_path = output_dir / "thermal_summary.json"

    manager_config_kwargs: dict[str, Any] = {
        "model_dir": args.model_dir,
        "tts_voice_style": args.voice_style,
        "llm_model_path": args.llm_model_path,
    }
    if args.llm_resident is not None:
        manager_config_kwargs["llm_resident"] = args.llm_resident

    manager = ModelManager(ManagerConfig(**manager_config_kwargs))
    atexit.register(manager.unload_all)
    manager.initialize()

    pipeline = ConversationPipeline(
        manager,
        PipelineConfig(
            llm_max_tokens=args.llm_max_tokens,
            llm_presence_penalty=args.presence_penalty,
            max_history_turns=args.max_history_turns,
            play_tts_audio=False,  # r2 B3 fix: ignore MUNGI_PLAY_TTS.
            tts_output_device=None,  # r2 B3 fix: ignore MUNGI_AUDIO_OUTPUT_DEVICE.
        ),
    )

    tegrastats = TegrastatsMonitor(tegrastats_log_path)
    thermal_logging_enabled = tegrastats.start()
    started_at = time.monotonic()
    records: list[dict[str, Any]] = []

    logger.info(
        "Starting PR 5 100-query voice run: selected=%d, output=%s", len(entries), output_dir
    )
    logger.info("Stage 1 playback boundary enforced: %s", STAGE1_PLAYBACK_DISABLED_REASON)

    try:
        with results_path.open("w", encoding="utf-8") as results_file:
            for index, entry in enumerate(entries, start=1):
                fixture = fixtures.get(entry.id)
                if fixture is None:
                    msg = f"Missing fixture manifest entry for query id={entry.id}"
                    raise ValueError(msg)

                logger.info(
                    "Voice query %03d/%03d id=%03d lang=%s fixture=%s",
                    index,
                    len(entries),
                    entry.id,
                    entry.lang,
                    fixture.wav_path,
                )
                turn_started_at = time.monotonic()
                fixture_sha256 = _verify_fixture(entry, fixture)
                samples = _load_fixture_wav(fixture.wav_path)
                result = pipeline.run_turn(samples, sample_rate=EXPECTED_FIXTURE_SAMPLE_RATE)

                wav_path: Path | None = None
                if (
                    not args.no_save_wav
                    and result.audio_samples is not None
                    and result.sample_rate > 0
                ):
                    slug = _slugify(entry.query)
                    wav_path = wav_dir / f"query_{entry.id:03d}_{slug}.wav"
                    _save_turn_wav(wav_path, result.audio_samples, result.sample_rate)

                record = _entry_record(
                    entry,
                    fixture,
                    fixture_sha256,
                    result,
                    wav_path=wav_path,
                    peak_memory_kb=get_peak_memory_kb(),
                    elapsed_s=time.monotonic() - turn_started_at,
                )
                records.append(record)
                results_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                results_file.flush()
                logger.info(
                    "Voice query id=%03d complete: success=%s voice_success=%s total=%.3fs",
                    entry.id,
                    record["success"],
                    record["voice_success"],
                    result.metrics.total_time_s,
                )
                pipeline.clear_history()
    finally:
        tegrastats.stop()
        manager.unload_all()

    elapsed_s = round(time.monotonic() - started_at, 3)
    thermal_summary = _build_thermal_summary(tegrastats.snapshots)
    summary = _build_summary(
        args=args,
        manifest_path=manifest_path,
        output_dir=output_dir,
        results_path=results_path,
        thermal_summary_path=thermal_summary_path,
        records=records,
        elapsed_s=elapsed_s,
        thermal_logging_enabled=thermal_logging_enabled,
        thermal_summary=thermal_summary,
        final_peak_memory_kb=get_peak_memory_kb(),
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if thermal_logging_enabled:
        thermal_summary_path.write_text(
            json.dumps(thermal_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    logger.info(
        "PR 5 voice run complete: success=%d/%d voice_success=%d/%d",
        summary["success_count"],
        len(records),
        summary["voice_success_count"],
        len(records),
    )
    logger.info("Summary saved: %s", summary_path)
    return 0 if summary["voice_success_count"] == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
