"""Inject official Korean-history chronological order into the runtime manifest.

The curated history content builder emits a runtime manifest with era metadata,
but it does not know the official 우리역사넷 chronological order. This
build-time post-processor reads the committed ``era_order_index.json`` file,
adds an integer ``order`` field to each manifest document, and sorts the
manifest's ``docs`` array by the existing era bucket order and then by that
official order.

The tool is deterministic and idempotent: re-running it against the same
manifest and index rewrites byte-identical JSON. It performs no network I/O and
does not import ``core/``.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from scripts.build_history_content import ERA_ORDER

logger = logging.getLogger("mungi.scripts.history_era_order")

DEFAULT_MANIFEST = Path("assets/history/manifest.json")
DEFAULT_INDEX = Path("assets/history/era_order_index.json")
SOURCE_LEVEL_ID_RE = re.compile(r"^(?P<level_id>eh_[nr]\d{4})_0010\.pdf$")

JsonObject = dict[str, Any]


@dataclass
class EraOrderStats:
    """Counters collected while applying chronological order."""

    docs: int = 0
    docs_ordered: int = 0
    manifest_written: bool = False
    manifest_unchanged: bool = False
    warnings: list[str] = field(default_factory=list)


def source_file_to_level_id(source_file: str) -> str:
    """Return the official site level id for a manifest ``source_file``."""
    match = SOURCE_LEVEL_ID_RE.match(source_file)
    if match is None:
        msg = f"Unsupported history source_file format: {source_file}"
        raise ValueError(msg)
    return match.group("level_id")


def apply_era_order(
    manifest_path: Path = DEFAULT_MANIFEST,
    index_path: Path = DEFAULT_INDEX,
    *,
    dry_run: bool = False,
) -> EraOrderStats:
    """Inject ``order`` into ``manifest_path`` and sort its document list."""
    manifest = _load_json_object(manifest_path)
    index = _load_json_object(index_path)
    order_by_level_id = _order_map(index, index_path)
    era_rank: dict[str, int] = {str(era): rank for rank, era in enumerate(ERA_ORDER)}

    raw_docs = _require_list(manifest, "docs", manifest_path)
    docs: list[JsonObject] = []
    manifest_level_ids: set[str] = set()
    missing_source_files: list[str] = []

    for raw_doc in raw_docs:
        if not isinstance(raw_doc, dict):
            msg = f"Manifest docs must contain objects: {manifest_path}"
            raise ValueError(msg)
        doc = cast(JsonObject, raw_doc)
        source_file = _require_str(doc, "source_file", manifest_path)
        level_id = source_file_to_level_id(source_file)
        manifest_level_ids.add(level_id)
        order = order_by_level_id.get(level_id)
        if order is None:
            missing_source_files.append(source_file)
            continue
        era = _require_str(doc, "era", manifest_path)
        if era not in era_rank:
            msg = f"Manifest doc has era outside ERA_ORDER: {era} ({source_file})"
            raise ValueError(msg)
        doc["order"] = order
        docs.append(doc)

    if missing_source_files:
        missing = ", ".join(sorted(missing_source_files))
        msg = f"Order index missing level_id for manifest source_file(s): {missing}"
        raise ValueError(msg)

    extra_level_ids = sorted(set(order_by_level_id) - manifest_level_ids)
    stats = EraOrderStats(docs=len(docs), docs_ordered=len(docs))
    if extra_level_ids:
        preview = ", ".join(extra_level_ids[:10])
        suffix = " ..." if len(extra_level_ids) > 10 else ""
        warning = (
            f"Order index has {len(extra_level_ids)} level_id(s) not present in manifest: "
            f"{preview}{suffix}"
        )
        logger.warning("%s", warning)
        stats.warnings.append(warning)

    manifest["schema_version"] = 2
    manifest["docs"] = sorted(
        docs,
        key=lambda doc: (
            era_rank[_require_str(doc, "era", manifest_path)],
            _require_int(doc, "order", manifest_path),
        ),
    )

    payload = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if dry_run:
        logger.info("dry-run: would write %s", _display_path(manifest_path))
        return stats

    if manifest_path.read_bytes() == payload:
        stats.manifest_unchanged = True
        logger.info("manifest unchanged: %s", _display_path(manifest_path))
        return stats

    manifest_path.write_bytes(payload)
    stats.manifest_written = True
    logger.info("wrote %s", _display_path(manifest_path))
    return stats


def _order_map(index: JsonObject, index_path: Path) -> dict[str, int]:
    items = _require_list(index, "items", index_path)
    order_by_level_id: dict[str, int] = {}
    for raw_item in items:
        if not isinstance(raw_item, dict):
            msg = f"Order index items must contain objects: {index_path}"
            raise ValueError(msg)
        item = cast(JsonObject, raw_item)
        level_id = _require_str(item, "level_id", index_path)
        order = _require_int(item, "order", index_path)
        if level_id in order_by_level_id:
            msg = f"Duplicate level_id in order index: {level_id}"
            raise ValueError(msg)
        order_by_level_id[level_id] = order
    return order_by_level_id


def _load_json_object(path: Path) -> JsonObject:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = f"Expected JSON object: {path}"
        raise ValueError(msg)
    return cast(JsonObject, payload)


def _require_list(payload: JsonObject, key: str, path: Path) -> list[Any]:
    value = payload.get(key)
    if not isinstance(value, list):
        msg = f"JSON field {key!r} must be a list: {path}"
        raise ValueError(msg)
    return value


def _require_str(payload: JsonObject, key: str, path: Path) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        msg = f"JSON field {key!r} must be non-empty text: {path}"
        raise ValueError(msg)
    return value


def _require_int(payload: JsonObject, key: str, path: Path) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"JSON field {key!r} must be int: {path}"
        raise ValueError(msg)
    return value


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for applying official chronological order."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    try:
        apply_era_order(args.manifest, args.index, dry_run=args.dry_run)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
