#!/usr/bin/env python3
"""Pad user-recorded E2E voice fixtures and generate a manifest."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import re
import sys
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np

LOGGER_NAME = "pad_voice_fixtures"
SOURCE_WAV_RE = re.compile(r"^audio_(?P<audio_id>\d+)_(?P<query_fragment>.+)\.wav$")
DEFAULT_PRE_PAD_MS = 500
DEFAULT_POST_PAD_MS = 500
DEFAULT_EXPECTED_SR = 16000
INT16_SCALE = 32767.0


class SoundFileModule(Protocol):
    """Minimal `soundfile` module API used by this script."""

    def read(self, file: str, dtype: str) -> tuple[np.ndarray, int]:
        """Read audio samples and return samples plus sample rate."""

    def write(self, file: str, data: np.ndarray, samplerate: int, subtype: str) -> None:
        """Write audio samples to a WAV file."""


class FixtureBuildError(Exception):
    """Structured fixture build error intended for JSON logging."""

    def __init__(self, payload: Mapping[str, Any]) -> None:
        super().__init__(str(payload.get("error", "fixture build failed")))
        self.payload = dict(payload)


@dataclass(frozen=True)
class SourceFixture:
    """Parsed source WAV fixture metadata."""

    audio_id: int
    path: Path
    query_fragment: str


@dataclass(frozen=True)
class PoolEntry:
    """Query pool entry fields needed to build fixture outputs."""

    query_id: int
    query: str
    lang: str


@dataclass(frozen=True)
class FixturePlan:
    """Validated source-to-pool mapping for one output fixture."""

    source: SourceFixture
    pool_entry: PoolEntry
    output_filename: str


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for fixture padding."""
    parser = argparse.ArgumentParser(
        description="Pad user-recorded voice fixtures with silence and emit a manifest.",
    )
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--pool", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--pre-pad-ms", default=DEFAULT_PRE_PAD_MS, type=int)
    parser.add_argument("--post-pad-ms", default=DEFAULT_POST_PAD_MS, type=int)
    parser.add_argument("--expected-sr", default=DEFAULT_EXPECTED_SR, type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def configure_logging() -> logging.Logger:
    """Configure process logging and return the script logger."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
        force=True,
    )
    return logging.getLogger(LOGGER_NAME)


def log_json(logger: logging.Logger, level: int, payload: Mapping[str, Any]) -> None:
    """Log one structured payload as JSON."""
    logger.log(level, json.dumps(payload, ensure_ascii=False, sort_keys=True))


def load_soundfile_module() -> SoundFileModule:
    """Load the optional `soundfile` dependency used for WAV I/O."""
    try:
        module = importlib.import_module("soundfile")
    except (ImportError, OSError) as exc:
        raise FixtureBuildError(
            {
                "check": "soundfile_dependency",
                "module": "soundfile",
                "error": str(exc),
            }
        ) from exc
    return cast(SoundFileModule, module)


def load_query_pool(path: Path) -> dict[int, PoolEntry]:
    """Load and validate query pool entries keyed by query ID."""
    try:
        raw_pool = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FixtureBuildError(
            {"check": "pool_exists", "path": str(path), "error": "pool file does not exist"}
        ) from exc
    except UnicodeDecodeError as exc:
        raise FixtureBuildError(
            {
                "check": "pool_utf8",
                "path": str(path),
                "byte_offset": exc.start,
                "error": "pool file is not valid UTF-8",
            }
        ) from exc
    except json.JSONDecodeError as exc:
        raise FixtureBuildError(
            {
                "check": "pool_json",
                "path": str(path),
                "line": exc.lineno,
                "column": exc.colno,
                "error": exc.msg,
            }
        ) from exc
    except OSError as exc:
        raise FixtureBuildError(
            {"check": "pool_read", "path": str(path), "error": str(exc)}
        ) from exc

    if not isinstance(raw_pool, list):
        raise FixtureBuildError(
            {
                "check": "pool_shape",
                "path": str(path),
                "expected": "JSON list",
                "actual": type(raw_pool).__name__,
            }
        )

    pool_by_id: dict[int, PoolEntry] = {}
    for index, raw_entry in enumerate(raw_pool):
        entry = _parse_pool_entry(raw_entry, index, path)
        if entry.query_id in pool_by_id:
            raise FixtureBuildError(
                {
                    "check": "pool_duplicate_query_id",
                    "path": str(path),
                    "query_id": entry.query_id,
                    "error": "duplicate query ID in pool",
                }
            )
        pool_by_id[entry.query_id] = entry

    if not pool_by_id:
        raise FixtureBuildError(
            {"check": "pool_nonempty", "path": str(path), "error": "pool is empty"}
        )
    return pool_by_id


def enumerate_source_fixtures(source_dir: Path) -> dict[int, SourceFixture]:
    """Enumerate and parse all source WAV fixtures keyed by audio ID."""
    if not source_dir.is_dir():
        raise FixtureBuildError(
            {
                "check": "source_dir",
                "path": str(source_dir),
                "error": "source directory does not exist or is not a directory",
            }
        )

    fixtures: dict[int, SourceFixture] = {}
    for source_path in sorted(source_dir.glob("*.wav"), key=lambda path: path.name):
        match = SOURCE_WAV_RE.fullmatch(source_path.name)
        if match is None:
            raise FixtureBuildError(
                {
                    "check": "source_filename",
                    "path": str(source_path),
                    "pattern": SOURCE_WAV_RE.pattern,
                    "error": "source WAV filename does not match expected pattern",
                }
            )

        audio_id = int(match.group("audio_id"))
        if audio_id in fixtures:
            raise FixtureBuildError(
                {
                    "check": "source_duplicate_audio_id",
                    "path": str(source_path),
                    "audio_id": audio_id,
                    "existing_path": str(fixtures[audio_id].path),
                    "error": "duplicate source audio ID",
                }
            )
        fixtures[audio_id] = SourceFixture(
            audio_id=audio_id,
            path=source_path,
            query_fragment=match.group("query_fragment"),
        )

    if not fixtures:
        raise FixtureBuildError(
            {
                "check": "source_wavs",
                "path": str(source_dir),
                "error": "no source WAV files found",
            }
        )
    return fixtures


def prepare_output_dir(output_dir: Path, dry_run: bool) -> None:
    """Validate and optionally create the output directory."""
    output_parent = output_dir.parent
    if not output_parent.is_dir():
        raise FixtureBuildError(
            {
                "check": "output_parent",
                "path": str(output_parent),
                "error": "output directory parent does not exist",
            }
        )

    if output_dir.exists() and not output_dir.is_dir():
        raise FixtureBuildError(
            {
                "check": "output_dir_type",
                "path": str(output_dir),
                "error": "output path exists but is not a directory",
            }
        )

    if dry_run:
        return

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FixtureBuildError(
            {
                "check": "output_dir_empty",
                "path": str(output_dir),
                "error": "output directory exists and is non-empty",
            }
        )
    output_dir.mkdir(exist_ok=True)


def build_fixture_plans(
    sources_by_audio_id: Mapping[int, SourceFixture],
    pool_by_query_id: Mapping[int, PoolEntry],
    limit: int | None,
    logger: logging.Logger,
) -> list[FixturePlan]:
    """Build validated fixture processing plans sorted by query ID."""
    if limit is not None and limit < 1:
        raise FixtureBuildError(
            {"check": "limit", "limit": limit, "error": "limit must be at least 1"}
        )

    for audio_id, source in sources_by_audio_id.items():
        query_id = audio_id + 1
        if query_id not in pool_by_query_id:
            raise FixtureBuildError(
                {
                    "check": "pool_query_id",
                    "source_filename": source.path.name,
                    "audio_id": audio_id,
                    "query_id": query_id,
                    "error": "source audio ID maps outside the query pool",
                }
            )

    expected_query_ids = sorted(pool_by_query_id)
    if limit is not None:
        expected_query_ids = expected_query_ids[:limit]

    plans: list[FixturePlan] = []
    for query_id in expected_query_ids:
        audio_id = query_id - 1
        source_fixture = sources_by_audio_id.get(audio_id)
        if source_fixture is None:
            raise FixtureBuildError(
                {
                    "check": "source_missing_audio_id",
                    "audio_id": audio_id,
                    "query_id": query_id,
                    "error": "missing source WAV for query ID",
                }
            )

        pool_entry = pool_by_query_id[query_id]
        _warn_on_query_text_mismatch(source_fixture, pool_entry, logger)
        output_filename = f"query_{query_id:03d}_{pool_entry.lang.lower()}.wav"
        plans.append(
            FixturePlan(
                source=source_fixture,
                pool_entry=pool_entry,
                output_filename=output_filename,
            )
        )

    return plans


def process_fixture_plans(
    plans: Sequence[FixturePlan],
    output_dir: Path,
    pre_pad_ms: int,
    post_pad_ms: int,
    expected_sr: int,
    soundfile_module: SoundFileModule,
    logger: logging.Logger,
) -> list[dict[str, object]]:
    """Pad source WAVs, write outputs, and return manifest entries."""
    if pre_pad_ms < 0 or post_pad_ms < 0:
        raise FixtureBuildError(
            {
                "check": "pad_duration",
                "pre_pad_ms": pre_pad_ms,
                "post_pad_ms": post_pad_ms,
                "error": "pad durations must be non-negative",
            }
        )
    if expected_sr < 1:
        raise FixtureBuildError(
            {
                "check": "expected_sr",
                "expected_sr": expected_sr,
                "error": "expected sample rate must be positive",
            }
        )

    manifest_entries: list[dict[str, object]] = []
    for plan in plans:
        manifest_entries.append(
            _process_one_fixture(
                plan=plan,
                output_dir=output_dir,
                pre_pad_ms=pre_pad_ms,
                post_pad_ms=post_pad_ms,
                expected_sr=expected_sr,
                soundfile_module=soundfile_module,
                logger=logger,
            )
        )
    return manifest_entries


def write_manifest(output_dir: Path, manifest_entries: Sequence[Mapping[str, object]]) -> None:
    """Write manifest entries to `<output-dir>/manifest.json`."""
    manifest_path = output_dir / "manifest.json"
    sorted_entries = sorted(manifest_entries, key=_manifest_query_id)
    try:
        manifest_path.write_text(
            json.dumps(sorted_entries, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise FixtureBuildError(
            {
                "check": "manifest_write",
                "path": str(manifest_path),
                "error": str(exc),
            }
        ) from exc


def _manifest_query_id(entry: Mapping[str, object]) -> int:
    value = entry["query_id"]
    if not isinstance(value, int) or isinstance(value, bool):
        raise FixtureBuildError(
            {
                "check": "manifest_entry_query_id",
                "value": value,
                "error": "manifest query_id is not an integer",
            }
        )
    return value


def _manifest_padded_duration(entry: Mapping[str, object]) -> float:
    value = entry["padded_duration_s"]
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise FixtureBuildError(
            {
                "check": "manifest_entry_padded_duration",
                "value": value,
                "error": "manifest padded_duration_s is not numeric",
            }
        )
    return float(value)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the fixture padding CLI."""
    logger = configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        pool_by_query_id = load_query_pool(args.pool)
        sources_by_audio_id = enumerate_source_fixtures(args.source_dir)
        prepare_output_dir(args.output_dir, args.dry_run)
        plans = build_fixture_plans(
            sources_by_audio_id=sources_by_audio_id,
            pool_by_query_id=pool_by_query_id,
            limit=args.limit,
            logger=logger,
        )

        if args.dry_run:
            _log_dry_run_plans(plans, args.output_dir, logger)
            log_json(
                logger,
                logging.INFO,
                {
                    "event": "summary",
                    "dry_run": True,
                    "total_processed": len(plans),
                    "total_manifest_entries": 0,
                    "total_duration_s": 0.0,
                },
            )
            return 0

        soundfile_module = load_soundfile_module()
        manifest_entries = process_fixture_plans(
            plans=plans,
            output_dir=args.output_dir,
            pre_pad_ms=args.pre_pad_ms,
            post_pad_ms=args.post_pad_ms,
            expected_sr=args.expected_sr,
            soundfile_module=soundfile_module,
            logger=logger,
        )
        write_manifest(args.output_dir, manifest_entries)
        total_duration_s = round(
            sum(_manifest_padded_duration(entry) for entry in manifest_entries), 3
        )
        log_json(
            logger,
            logging.INFO,
            {
                "event": "summary",
                "dry_run": False,
                "total_processed": len(plans),
                "total_manifest_entries": len(manifest_entries),
                "total_duration_s": total_duration_s,
            },
        )
    except FixtureBuildError as exc:
        log_json(logger, logging.ERROR, {"event": "error", **exc.payload})
        return 1

    return 0


def _parse_pool_entry(raw_entry: Any, entry_index: int, pool_path: Path) -> PoolEntry:
    if not isinstance(raw_entry, dict):
        raise FixtureBuildError(
            {
                "check": "pool_entry_shape",
                "path": str(pool_path),
                "entry_index": entry_index,
                "expected": "JSON object",
                "actual": type(raw_entry).__name__,
            }
        )

    raw_query_id = raw_entry.get("query_id", raw_entry.get("id"))
    raw_query = raw_entry.get("query")
    raw_lang = raw_entry.get("lang")
    if not isinstance(raw_query_id, int) or isinstance(raw_query_id, bool) or raw_query_id < 1:
        raise FixtureBuildError(
            {
                "check": "pool_entry_query_id",
                "path": str(pool_path),
                "entry_index": entry_index,
                "value": raw_query_id,
                "error": "query_id/id must be a positive integer",
            }
        )
    if not isinstance(raw_query, str) or not raw_query:
        raise FixtureBuildError(
            {
                "check": "pool_entry_query",
                "path": str(pool_path),
                "entry_index": entry_index,
                "query_id": raw_query_id,
                "error": "query must be a non-empty string",
            }
        )
    if not isinstance(raw_lang, str) or not raw_lang:
        raise FixtureBuildError(
            {
                "check": "pool_entry_lang",
                "path": str(pool_path),
                "entry_index": entry_index,
                "query_id": raw_query_id,
                "error": "lang must be a non-empty string",
            }
        )

    return PoolEntry(query_id=raw_query_id, query=raw_query, lang=raw_lang.lower())


def _warn_on_query_text_mismatch(
    source: SourceFixture,
    pool_entry: PoolEntry,
    logger: logging.Logger,
) -> None:
    source_text = _normalize_query_text(source.query_fragment)
    pool_text = _normalize_query_text(pool_entry.query)
    if pool_text and pool_text not in source_text:
        log_json(
            logger,
            logging.WARNING,
            {
                "event": "filename_pool_text_mismatch",
                "source_filename": source.path.name,
                "query_id": pool_entry.query_id,
                "source_normalized": source_text,
                "pool_normalized": pool_text,
                "warning": "source filename text does not contain pool query text",
            },
        )


def _normalize_query_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).lower()
    return "".join(char for char in normalized if _is_query_text_char(char))


def _is_query_text_char(char: str) -> bool:
    category = unicodedata.category(char)
    return category[0] in {"L", "N"}


def _process_one_fixture(
    plan: FixturePlan,
    output_dir: Path,
    pre_pad_ms: int,
    post_pad_ms: int,
    expected_sr: int,
    soundfile_module: SoundFileModule,
    logger: logging.Logger,
) -> dict[str, object]:
    source_path = plan.source.path
    try:
        samples, decoded_sample_rate = soundfile_module.read(str(source_path), dtype="float32")
    except Exception as exc:
        raise FixtureBuildError(
            {
                "check": "source_wav_read",
                "source_filename": source_path.name,
                "path": str(source_path),
                "error": str(exc),
            }
        ) from exc

    samples = np.asarray(samples, dtype=np.float32)
    if decoded_sample_rate != expected_sr:
        raise FixtureBuildError(
            {
                "check": "sample_rate",
                "source_filename": source_path.name,
                "expected_sr": expected_sr,
                "actual_sr": decoded_sample_rate,
                "error": "source WAV sample rate does not match expected sample rate",
            }
        )
    if samples.ndim != 1:
        raise FixtureBuildError(
            {
                "check": "channels",
                "source_filename": source_path.name,
                "expected_channels": 1,
                "actual_ndim": samples.ndim,
                "shape": list(samples.shape),
                "error": "source WAV is not mono",
            }
        )

    raw_duration_s = len(samples) / expected_sr
    pre_n = int(pre_pad_ms * expected_sr / 1000)
    post_n = int(post_pad_ms * expected_sr / 1000)
    padded = np.concatenate(
        [
            np.zeros(pre_n, dtype=np.float32),
            samples,
            np.zeros(post_n, dtype=np.float32),
        ]
    )
    clipped = np.clip(padded, -1.0, 1.0)
    int16_samples = (clipped * INT16_SCALE).astype(np.int16)

    output_path = output_dir / plan.output_filename
    try:
        soundfile_module.write(str(output_path), int16_samples, expected_sr, subtype="PCM_16")
    except Exception as exc:
        raise FixtureBuildError(
            {
                "check": "output_wav_write",
                "source_filename": source_path.name,
                "path": str(output_path),
                "error": str(exc),
            }
        ) from exc

    padded_duration_s = (len(samples) + pre_n + post_n) / expected_sr
    sha256 = _sha256_file(output_path)
    manifest_entry: dict[str, object] = {
        "query_id": plan.pool_entry.query_id,
        "wav_path": output_path.relative_to(output_dir.parent).as_posix(),
        "voice": f"human-recorded:{source_path.stem}",
        "lang": plan.pool_entry.lang,
        "raw_duration_s": round(raw_duration_s, 3),
        "padded_duration_s": round(padded_duration_s, 3),
        "decoded_sample_rate": expected_sr,
        "sha256": sha256,
    }

    log_json(
        logger,
        logging.INFO,
        {
            "event": "processed_file",
            "source_filename": source_path.name,
            "query_id": plan.pool_entry.query_id,
            "lang": plan.pool_entry.lang,
            "padded_duration_s": round(padded_duration_s, 3),
        },
    )
    return manifest_entry


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as file_handle:
            for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise FixtureBuildError(
            {
                "check": "output_wav_sha256",
                "path": str(path),
                "error": str(exc),
            }
        ) from exc
    return digest.hexdigest()


def _log_dry_run_plans(
    plans: Sequence[FixturePlan],
    output_dir: Path,
    logger: logging.Logger,
) -> None:
    for plan in plans:
        output_path = output_dir / plan.output_filename
        log_json(
            logger,
            logging.INFO,
            {
                "event": "dry_run_mapping",
                "source_filename": plan.source.path.name,
                "query_id": plan.pool_entry.query_id,
                "lang": plan.pool_entry.lang,
                "output_path": str(output_path),
            },
        )


if __name__ == "__main__":
    raise SystemExit(main())
