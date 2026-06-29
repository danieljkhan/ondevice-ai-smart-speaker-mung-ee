"""Ingest raw English and Korean Wikipedia parquet data into cached JSONL files."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_DIR = REPO_ROOT / "assets" / "rag" / "raw material"
DEFAULT_OUTPUT_DIR = DEFAULT_SOURCE_DIR / "_cache"
INGEST_VERSION = "2026-05-16-raw-wikipedia-v1"
MIN_TEXT_LENGTH = 200
MAX_TEXT_LENGTH = 50_000
TITLE_CANDIDATE_MAX_LENGTH = 80
TITLE_CANDIDATE_MAX_WORDS = 14
PARQUET_BATCH_SIZE = 512
SUPPORTED_SOURCES = ("en", "ko")
SENTENCE_FINAL_PUNCTUATION = frozenset(".!?;:")
WHITESPACE_RE = re.compile(r"\s+")
LINE_WHITESPACE_RE = re.compile(r"[ \t]+")
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


@dataclass(frozen=True)
class SourceConfig:
    """Configuration for one raw Wikipedia source."""

    key: str
    relative_dir: Path
    cache_filename: str
    source_name: str
    language: str


@dataclass(frozen=True)
class CacheMetadata:
    """Metadata used to decide whether an ingest cache is still valid."""

    ingest_version: str
    source_hash: str
    record_count: int
    limit: int | None


@dataclass(frozen=True)
class IngestStats:
    """Summary statistics for one source ingest pass."""

    source: str
    language: str
    parquet_files: int
    emitted: int
    skipped_too_short: int
    skipped_too_long: int
    skipped_missing_fields: int
    used_cache: bool
    dry_run: bool
    cache_path: Path


SOURCE_CONFIGS: dict[str, SourceConfig] = {
    "en": SourceConfig(
        key="en",
        relative_dir=Path("simple-wikipedia-en"),
        cache_filename="en_ingested.jsonl",
        source_name="rahular_simple_wikipedia",
        language="en",
    ),
    "ko": SourceConfig(
        key="ko",
        relative_dir=Path("wikipedia-korean-20240501"),
        cache_filename="ko_ingested.jsonl",
        source_name="lcw99_wikipedia_korean_20240501",
        language="ko",
    ),
}


def configure_logging() -> None:
    """Configure CLI logging."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for raw Wikipedia ingest."""

    parser = argparse.ArgumentParser(
        description="Ingest raw EN/KO Wikipedia parquet shards into cached JSONL files.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Directory containing raw Wikipedia parquet source folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where ingest caches will be written.",
    )
    parser.add_argument(
        "--sources",
        default="en,ko",
        help="Comma-separated list of sources to ingest: en, ko, or en,ko.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of emitted records per source.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process records without writing cache files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore fresh cache metadata and rebuild ingest outputs.",
    )
    return parser.parse_args(argv)


def parse_sources(raw_sources: str) -> tuple[str, ...]:
    """Parse and validate the CLI source selection."""

    requested = tuple(part.strip() for part in raw_sources.split(",") if part.strip())
    if not requested:
        raise ValueError("At least one source must be provided")

    invalid = sorted({source for source in requested if source not in SUPPORTED_SOURCES})
    if invalid:
        raise ValueError(f"Unsupported sources: {', '.join(invalid)}")

    deduped: list[str] = []
    for source in requested:
        if source not in deduped:
            deduped.append(source)
    return tuple(deduped)


def normalize_inline_text(value: object) -> str:
    """Collapse arbitrary whitespace into one inline string."""

    if value is None:
        return ""
    return WHITESPACE_RE.sub(" ", str(value)).strip()


def normalize_article_text(value: object) -> str:
    """Normalize article text while preserving paragraph breaks."""

    if value is None:
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n").strip()
    text = LINE_WHITESPACE_RE.sub(" ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def normalize_section_titles(value: object) -> list[str]:
    """Normalize a section-title sequence from a parquet row."""

    if not isinstance(value, list):
        return []
    return [title for title in (normalize_inline_text(item) for item in value) if title]


def is_short_title_candidate(text: str) -> bool:
    """Return whether a paragraph-row string looks like an article title."""

    candidate = normalize_inline_text(text)
    if not candidate:
        return False
    if len(candidate) > TITLE_CANDIDATE_MAX_LENGTH:
        return False
    if len(candidate.split()) > TITLE_CANDIDATE_MAX_WORDS:
        return False
    return candidate[-1] not in SENTENCE_FINAL_PUNCTUATION


def find_parquet_files(source_dir: Path, config: SourceConfig) -> list[Path]:
    """Locate parquet shards for one configured source."""

    dataset_dir = source_dir / config.relative_dir
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Missing source directory: {dataset_dir}")

    parquet_files = sorted(dataset_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {dataset_dir}")
    return parquet_files


def compute_source_hash(parquet_files: Sequence[Path]) -> str:
    """Compute a stable fingerprint for the current raw parquet shard set."""

    hasher = hashlib.sha256()
    for parquet_file in sorted(parquet_files):
        stat = parquet_file.stat()
        hasher.update(parquet_file.as_posix().encode("utf-8"))
        hasher.update(str(stat.st_size).encode("utf-8"))
        hasher.update(str(stat.st_mtime_ns).encode("utf-8"))
    return hasher.hexdigest()


def load_cache_metadata(metadata_path: Path) -> CacheMetadata | None:
    """Load cache metadata if it exists and is well-formed."""

    if not metadata_path.exists():
        return None

    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        LOGGER.warning("Ignoring unreadable cache metadata at %s", metadata_path)
        return None

    if not isinstance(payload, dict):
        LOGGER.warning("Ignoring invalid cache metadata payload at %s", metadata_path)
        return None

    ingest_version = payload.get("ingest_version")
    source_hash = payload.get("source_hash")
    record_count = payload.get("record_count")
    limit = payload.get("limit")
    if not isinstance(ingest_version, str) or not isinstance(source_hash, str):
        return None
    if not isinstance(record_count, int):
        return None
    if limit is not None and not isinstance(limit, int):
        return None
    return CacheMetadata(
        ingest_version=ingest_version,
        source_hash=source_hash,
        record_count=record_count,
        limit=limit,
    )


def write_cache_metadata(
    metadata_path: Path,
    *,
    source_hash: str,
    record_count: int,
    limit: int | None,
) -> None:
    """Write cache metadata for one completed ingest."""

    payload = {
        "ingest_version": INGEST_VERSION,
        "source_hash": source_hash,
        "record_count": record_count,
        "limit": limit,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_output_paths(output_dir: Path, config: SourceConfig) -> tuple[Path, Path]:
    """Return the JSONL cache path and its metadata sidecar path."""

    cache_path = output_dir / config.cache_filename
    metadata_path = output_dir / f"{config.key}_ingested.meta.json"
    return cache_path, metadata_path


def iter_parquet_rows(parquet_path: Path, columns: Sequence[str]) -> Any:
    """Yield parquet rows as dictionaries using batch iteration."""

    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(parquet_path)
    for batch in parquet_file.iter_batches(
        columns=list(columns),
        batch_size=PARQUET_BATCH_SIZE,
    ):
        yield from batch.to_pylist()


def write_jsonl_record(handle: Any, record: dict[str, Any]) -> None:
    """Write a JSONL record with UTF-8-safe JSON encoding."""

    handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def maybe_emit_record(
    record: dict[str, Any],
    *,
    handle: Any | None,
    dry_run: bool,
    stats: dict[str, int],
) -> int:
    """Apply common length filters, update counters, and optionally write a record."""

    text = str(record["text"])
    if len(text) < MIN_TEXT_LENGTH:
        stats["skipped_too_short"] += 1
        return 0
    if len(text) > MAX_TEXT_LENGTH:
        stats["skipped_too_long"] += 1
        return 0
    if handle is not None and not dry_run:
        write_jsonl_record(handle, record)
    stats["emitted"] += 1
    return 1


def flush_en_article(
    *,
    title: str | None,
    paragraphs: list[str],
    source_name: str,
    handle: Any | None,
    dry_run: bool,
    stats: dict[str, int],
) -> int:
    """Flush the current English article buffer into one record if valid."""

    if title is None or not paragraphs:
        return 0

    record = {
        "id": f"en_sw_{stats['emitted']}",
        "title": title,
        "text": "\n\n".join(paragraphs),
        "source": source_name,
        "language": "en",
    }
    return maybe_emit_record(record, handle=handle, dry_run=dry_run, stats=stats)


def ingest_english(
    parquet_files: Sequence[Path],
    *,
    config: SourceConfig,
    handle: Any | None,
    limit: int | None,
    dry_run: bool,
) -> dict[str, int]:
    """Reconstruct article records from Simple Wikipedia paragraph rows."""

    stats = {
        "emitted": 0,
        "skipped_too_short": 0,
        "skipped_too_long": 0,
        "skipped_missing_fields": 0,
    }
    current_title: str | None = None
    current_paragraphs: list[str] = []

    for parquet_file in parquet_files:
        for row in iter_parquet_rows(parquet_file, columns=("text",)):
            raw_text = row.get("text") if isinstance(row, dict) else None
            normalized_inline = normalize_inline_text(raw_text)
            if not normalized_inline:
                stats["skipped_missing_fields"] += 1
                continue

            if is_short_title_candidate(normalized_inline):
                flush_en_article(
                    title=current_title,
                    paragraphs=current_paragraphs,
                    source_name=config.source_name,
                    handle=handle,
                    dry_run=dry_run,
                    stats=stats,
                )
                if limit is not None and stats["emitted"] >= limit:
                    return stats
                current_title = normalized_inline
                current_paragraphs = []
                continue

            if current_title is None:
                stats["skipped_missing_fields"] += 1
                continue

            current_paragraphs.append(normalize_article_text(raw_text))

    flush_en_article(
        title=current_title,
        paragraphs=current_paragraphs,
        source_name=config.source_name,
        handle=handle,
        dry_run=dry_run,
        stats=stats,
    )
    return stats


def build_ko_record(
    *,
    row: dict[str, Any],
    config: SourceConfig,
    row_index: int,
) -> dict[str, Any] | None:
    """Build one Korean ingest record directly from a parquet row."""

    title = normalize_inline_text(row.get("title"))
    text = normalize_article_text(row.get("text"))
    if not title or not text:
        return None

    record: dict[str, Any] = {
        "id": f"ko_w_{row_index}",
        "title": title,
        "text": text,
        "source": config.source_name,
        "language": config.language,
    }
    section_titles = normalize_section_titles(row.get("section_titles"))
    if section_titles:
        record["section_titles"] = section_titles
    return record


def ingest_korean(
    parquet_files: Sequence[Path],
    *,
    config: SourceConfig,
    handle: Any | None,
    limit: int | None,
    dry_run: bool,
) -> dict[str, int]:
    """Ingest Korean Wikipedia article rows directly from parquet shards."""

    stats = {
        "emitted": 0,
        "skipped_too_short": 0,
        "skipped_too_long": 0,
        "skipped_missing_fields": 0,
    }

    for parquet_file in parquet_files:
        for row in iter_parquet_rows(parquet_file, columns=("title", "text", "section_titles")):
            if not isinstance(row, dict):
                stats["skipped_missing_fields"] += 1
                continue

            record = build_ko_record(row=row, config=config, row_index=stats["emitted"])
            if record is None:
                stats["skipped_missing_fields"] += 1
                continue

            maybe_emit_record(record, handle=handle, dry_run=dry_run, stats=stats)
            if limit is not None and stats["emitted"] >= limit:
                return stats

    return stats


def ingest_source(
    *,
    source_dir: Path,
    output_dir: Path,
    config: SourceConfig,
    limit: int | None,
    dry_run: bool,
    force: bool,
) -> IngestStats:
    """Ingest one configured raw Wikipedia source."""

    parquet_files = find_parquet_files(source_dir, config)
    source_hash = compute_source_hash(parquet_files)
    cache_path, metadata_path = build_output_paths(output_dir, config)
    metadata = load_cache_metadata(metadata_path)

    if (
        not dry_run
        and not force
        and cache_path.exists()
        and metadata is not None
        and metadata.ingest_version == INGEST_VERSION
        and metadata.source_hash == source_hash
        and metadata.limit == limit
    ):
        LOGGER.info("Using fresh cache for %s at %s", config.key, cache_path)
        return IngestStats(
            source=config.key,
            language=config.language,
            parquet_files=len(parquet_files),
            emitted=metadata.record_count,
            skipped_too_short=0,
            skipped_too_long=0,
            skipped_missing_fields=0,
            used_cache=True,
            dry_run=False,
            cache_path=cache_path,
        )

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        temp_path = cache_path.with_suffix(f"{cache_path.suffix}.tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            if config.key == "en":
                counts = ingest_english(
                    parquet_files,
                    config=config,
                    handle=handle,
                    limit=limit,
                    dry_run=False,
                )
            else:
                counts = ingest_korean(
                    parquet_files,
                    config=config,
                    handle=handle,
                    limit=limit,
                    dry_run=False,
                )
        temp_path.replace(cache_path)
        write_cache_metadata(
            metadata_path,
            source_hash=source_hash,
            record_count=counts["emitted"],
            limit=limit,
        )
    else:
        if config.key == "en":
            counts = ingest_english(
                parquet_files,
                config=config,
                handle=None,
                limit=limit,
                dry_run=True,
            )
        else:
            counts = ingest_korean(
                parquet_files,
                config=config,
                handle=None,
                limit=limit,
                dry_run=True,
            )

    LOGGER.info(
        "Ingested %s: emitted=%d short=%d long=%d missing=%d dry_run=%s",
        config.key,
        counts["emitted"],
        counts["skipped_too_short"],
        counts["skipped_too_long"],
        counts["skipped_missing_fields"],
        dry_run,
    )
    return IngestStats(
        source=config.key,
        language=config.language,
        parquet_files=len(parquet_files),
        emitted=counts["emitted"],
        skipped_too_short=counts["skipped_too_short"],
        skipped_too_long=counts["skipped_too_long"],
        skipped_missing_fields=counts["skipped_missing_fields"],
        used_cache=False,
        dry_run=dry_run,
        cache_path=cache_path,
    )


def ingest_requested_sources(
    *,
    source_dir: Path,
    output_dir: Path,
    sources: Sequence[str],
    limit: int | None,
    dry_run: bool,
    force: bool,
) -> list[IngestStats]:
    """Ingest all requested sources and return one summary per source."""

    results: list[IngestStats] = []
    for source in sources:
        config = SOURCE_CONFIGS[source]
        results.append(
            ingest_source(
                source_dir=source_dir,
                output_dir=output_dir,
                config=config,
                limit=limit,
                dry_run=dry_run,
                force=force,
            )
        )
    return results


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    configure_logging()
    args = parse_args(argv)

    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be a positive integer")

    try:
        sources = parse_sources(args.sources)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    ingest_requested_sources(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        sources=sources,
        limit=args.limit,
        dry_run=bool(args.dry_run),
        force=bool(args.force),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
