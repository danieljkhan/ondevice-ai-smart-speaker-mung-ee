"""Tests for the Korean-history manifest era-order post-processor."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.history_era_order import apply_era_order

JsonObject = dict[str, Any]


def _write_json(path: Path, payload: JsonObject) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _manifest_doc(
    *,
    doc_hash: str,
    source_file: str,
    title: str,
    era: str,
    kind: str = "people",
) -> JsonObject:
    return {
        "doc_hash": doc_hash,
        "source_file": source_file,
        "title": title,
        "kind": kind,
        "era": era,
        "scene_count": 1,
        "section_count": 1,
        "image_count": 0,
        "est_total_ms": 1000,
        "doc_path": f"assets/history/docs/{doc_hash}.json",
        "title_curated": True,
        "era_source": "keyword",
    }


def _index_item(order: int, level_id: str, title: str) -> JsonObject:
    return {
        "order": order,
        "level_id": level_id,
        "source_file": f"{level_id}_0010.pdf",
        "age_code": "eh_age_10",
        "age_name": "고대",
        "title": title,
    }


def _write_manifest(path: Path, docs: list[JsonObject]) -> None:
    _write_json(
        path,
        {
            "schema_version": 2,
            "title": "재미있는 우리역사",
            "era_order": ["선사", "고조선", "삼국"],
            "docs": docs,
        },
    )


def _write_index(path: Path, items: list[JsonObject]) -> None:
    _write_json(
        path,
        {
            "schema_version": 1,
            "source": "fixture",
            "generated_note": "fixture",
            "count": len(items),
            "items": items,
        },
    )


def test_history_era_order_raises_for_manifest_doc_missing_from_index(tmp_path: Path) -> None:
    """A manifest source_file without an index level_id fails the integrity gate."""
    manifest_path = tmp_path / "manifest.json"
    index_path = tmp_path / "era_order_index.json"
    _write_manifest(
        manifest_path,
        [
            _manifest_doc(
                doc_hash="missing",
                source_file="eh_n0001_0010.pdf",
                title="누락 문서",
                era="고조선",
            )
        ],
    )
    _write_index(index_path, [])

    with pytest.raises(ValueError, match="eh_n0001_0010.pdf"):
        apply_era_order(manifest_path, index_path)


def test_history_era_order_injects_order_and_sorts_by_era_then_order(
    tmp_path: Path,
) -> None:
    """Order injection preserves era buckets and sorts within each era by order."""
    manifest_path = tmp_path / "manifest.json"
    index_path = tmp_path / "era_order_index.json"
    _write_manifest(
        manifest_path,
        [
            _manifest_doc(
                doc_hash="samguk",
                source_file="eh_n1000_0010.pdf",
                title="삼국 문서",
                era="삼국",
            ),
            _manifest_doc(
                doc_hash="gojoseon_later",
                source_file="eh_n1001_0010.pdf",
                title="가나다 나중 문서",
                era="고조선",
            ),
            _manifest_doc(
                doc_hash="seonsa",
                source_file="eh_r1002_0010.pdf",
                title="선사 문서",
                era="선사",
                kind="artifact",
            ),
            _manifest_doc(
                doc_hash="gojoseon_earlier",
                source_file="eh_r1003_0010.pdf",
                title="하하 먼저 문서",
                era="고조선",
                kind="artifact",
            ),
        ],
    )
    _write_index(
        index_path,
        [
            _index_item(50, "eh_n1000", "삼국 문서"),
            _index_item(30, "eh_n1001", "가나다 나중 문서"),
            _index_item(40, "eh_r1002", "선사 문서"),
            _index_item(10, "eh_r1003", "하하 먼저 문서"),
        ],
    )

    stats = apply_era_order(manifest_path, index_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    docs = payload["docs"]

    assert stats.docs_ordered == 4
    assert payload["schema_version"] == 2
    assert [doc["doc_hash"] for doc in docs] == [
        "seonsa",
        "gojoseon_earlier",
        "gojoseon_later",
        "samguk",
    ]
    assert [doc["order"] for doc in docs] == [40, 10, 30, 50]


def test_history_era_order_is_idempotent(tmp_path: Path) -> None:
    """Applying the post-processor twice produces byte-identical output."""
    manifest_path = tmp_path / "manifest.json"
    index_path = tmp_path / "era_order_index.json"
    _write_manifest(
        manifest_path,
        [
            _manifest_doc(
                doc_hash="later",
                source_file="eh_n1001_0010.pdf",
                title="가나다 나중 문서",
                era="고조선",
            ),
            _manifest_doc(
                doc_hash="earlier",
                source_file="eh_r1003_0010.pdf",
                title="하하 먼저 문서",
                era="고조선",
                kind="artifact",
            ),
        ],
    )
    _write_index(
        index_path,
        [
            _index_item(30, "eh_n1001", "가나다 나중 문서"),
            _index_item(10, "eh_r1003", "하하 먼저 문서"),
        ],
    )

    apply_era_order(manifest_path, index_path)
    first = manifest_path.read_text(encoding="utf-8")
    apply_era_order(manifest_path, index_path)
    second = manifest_path.read_text(encoding="utf-8")

    assert second == first
