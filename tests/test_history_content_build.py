"""Tests for the Korean history content builder."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image  # type: ignore[import-not-found, import-untyped]

from scripts import build_history_content


def _is_targeted_history_test_run(args: list[str]) -> bool:
    target = Path("tests/test_history_content_build.py")
    return len(args) == 1 and Path(args[0]).as_posix() == target.as_posix()


@pytest.fixture(scope="session", autouse=True)
def _allow_targeted_script_test_coverage(pytestconfig: pytest.Config) -> None:
    cov_plugin = pytestconfig.pluginmanager.get_plugin("_cov")
    if cov_plugin is None or not _is_targeted_history_test_run(pytestconfig.args):
        return
    cov_plugin.options.cov_fail_under = 0


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _scene(
    *,
    seq: int,
    narration: str,
    section_title: str | None = None,
    image_paths: list[str] | None = None,
    image_captions: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "seq": seq,
        "page": seq,
        "section_title": section_title,
        "narration": narration,
        "est_speech_ms": 1000 * seq,
        "tail_silence_ms": 500,
        "image_paths": image_paths or [],
        "image_captions": image_captions or [],
        "image_path": (image_paths or [None])[0],
        "image_caption": (image_captions or [None])[0],
    }


def _document(
    *,
    doc_hash: str,
    source_file: str,
    scenes: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "doc_hash": doc_hash,
        "source_file": source_file,
        "scene_count": len(scenes),
        "scenes": scenes,
    }


def _write_jpeg(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path, format="JPEG")


@pytest.mark.parametrize(
    ("title", "section_title", "source_file", "expected_era", "expected_source"),
    [
        ("고려 사람들이 사용한 도자기, 고려청자", None, "eh_r0140_0010.pdf", "고려", "keyword"),
        ("우리 역사 속 최초의 국가, 고조선", None, "eh_r0005_0010.pdf", "고조선", "keyword"),
        ("조선의 독립을 외치다", None, "eh_r0250_0010.pdf", "일제강점기", "keyword"),
        ("생활 모습의 변화를 가져온 근대 시설", None, "eh_r0362_0010.pdf", "근대", "keyword"),
        ("낯선 제목", None, "eh_n0230_0010.pdf", "고려", "docnum"),
    ],
)
def test_map_era_uses_recent_keyword_order_then_docnum(
    title: str,
    section_title: str | None,
    source_file: str,
    expected_era: str,
    expected_source: str,
) -> None:
    era = build_history_content.map_era(title, section_title, source_file)

    assert era.era == expected_era
    assert era.source == expected_source


def test_derive_title_falls_back_for_trailing_comma() -> None:
    title = build_history_content.derive_title(
        {
            "narration": "김홍도,\n다음 줄",
            "section_title": "조선 백성을 그림에 담아내다",
        }
    )

    assert title.title == "조선 백성을 그림에 담아내다"
    assert title.curated is False


def test_build_stats_count_all_uncurated_title_spills(tmp_path: Path) -> None:
    dataset = tmp_path / "assets" / "dataset_korean history"
    out = tmp_path / "assets" / "history"
    documents = [
        _document(
            doc_hash="hash000000000001",
            source_file="eh_n0010_0010.pdf",
            scenes=[
                _scene(
                    seq=1,
                    narration="단군왕검, 아사달에 나라를 세우다",
                    section_title=None,
                )
            ],
        ),
        _document(
            doc_hash="hash000000000002",
            source_file="eh_n0020_0010.pdf",
            scenes=[
                _scene(
                    seq=1,
                    narration="유일한,\n다음 줄",
                    section_title="나라를 생각한 기업가 유일한",
                )
            ],
        ),
        _document(
            doc_hash="hash000000000003",
            source_file="eh_r0005_0010.pdf",
            scenes=[
                _scene(
                    seq=1,
                    narration="왕",
                    section_title="청동기 시대를 보여 주는 고인돌",
                )
            ],
        ),
    ]
    for document in documents:
        _write_json(
            dataset / "data" / "scenes" / f"{document['doc_hash']}.json",
            document,
        )

    stats = build_history_content.build_history_content(
        build_history_content.BuildOptions(
            dataset=dataset,
            out=out,
            manifest_only=True,
            force=True,
        )
    )
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    entries_by_source = {entry["source_file"]: entry for entry in manifest["docs"]}

    assert stats.title_uncurated == 2
    assert entries_by_source["eh_n0010_0010.pdf"]["title_curated"] is True
    assert entries_by_source["eh_n0010_0010.pdf"]["title"] == "단군왕검, 아사달에 나라를 세우다"
    assert entries_by_source["eh_n0020_0010.pdf"]["title_curated"] is False
    assert entries_by_source["eh_n0020_0010.pdf"]["title"] == "나라를 생각한 기업가 유일한"
    assert entries_by_source["eh_r0005_0010.pdf"]["title_curated"] is False
    assert entries_by_source["eh_r0005_0010.pdf"]["title"] == "청동기 시대를 보여 주는 고인돌"


def test_build_manifest_and_doc_payload_keep_narration_verbatim(tmp_path: Path) -> None:
    dataset = tmp_path / "assets" / "dataset_korean history"
    out = tmp_path / "assets" / "history"
    doc_hash = "abc123def4567890"
    image_rel = f"data/figures/{doc_hash}/fig_001.jpg"
    narration = "단군왕검, 아사달에 나라를 세우다\n“원문 그대로”\n쉼표, 공백 유지"
    _write_jpeg(dataset / image_rel, (120, 80), (200, 20, 10))
    _write_json(
        dataset / "data" / "scenes" / f"{doc_hash}.json",
        _document(
            doc_hash=doc_hash,
            source_file="eh_n0010_0010.pdf",
            scenes=[
                _scene(
                    seq=1,
                    narration=narration,
                    section_title=None,
                    image_paths=[image_rel],
                    image_captions=[],
                )
            ],
        ),
    )

    stats = build_history_content.build_history_content(
        build_history_content.BuildOptions(
            dataset=dataset,
            out=out,
            manifest_only=True,
            force=True,
        )
    )

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    doc_payload = json.loads((out / "docs" / f"{doc_hash}.json").read_text(encoding="utf-8"))

    assert stats.docs == 1
    assert stats.title_uncurated == 0
    assert manifest["schema_version"] == 2
    assert manifest["title"] == "재미있는 우리역사"
    assert manifest["docs"][0]["doc_path"].endswith(f"assets/history/docs/{doc_hash}.json")
    assert manifest["docs"][0]["section_count"] == 1
    assert doc_payload["schema_version"] == 2
    assert doc_payload["section_count"] == 1
    assert doc_payload["sections"] == [
        {
            "section_index": 0,
            "section_title": None,
            "scene_indices": [0],
            "scene_seq": [1],
            "image_captions": [],
            "is_infographic": False,
        }
    ]
    assert doc_payload["scenes"][0]["section_index"] == 0
    assert doc_payload["scenes"][0]["section_title"] is None
    assert doc_payload["scenes"][0]["image_captions"] == []
    assert doc_payload["scenes"][0]["narration"] == narration
    assert doc_payload["scenes"][0]["images"] == [
        {
            "path": f"{out.as_posix()}/images/{doc_hash}/fig_001.jpg",
            "caption": None,
            "letterboxed": True,
            "clean": False,
            "is_infographic": False,
        }
    ]


def test_build_groups_sections_and_preserves_caption_lists(tmp_path: Path) -> None:
    dataset = tmp_path / "assets" / "dataset_korean history"
    out = tmp_path / "assets" / "history"
    doc_hash = "feed1234abcd5678"
    image_paths = [f"data/figures/{doc_hash}/fig_00{index}.jpg" for index in range(1, 4)]
    for index, image_rel in enumerate(image_paths):
        _write_jpeg(dataset / image_rel, (120 + index, 80), (20 * index, 40, 80))
    _write_json(
        dataset / "data" / "scenes" / f"{doc_hash}.json",
        _document(
            doc_hash=doc_hash,
            source_file="eh_n0010_0010.pdf",
            scenes=[
                _scene(seq=1, narration="처음 이야기입니다.", section_title=None),
                _scene(seq=2, narration="고인돌 이야기입니다.", section_title="고인돌 이야기"),
                _scene(
                    seq=3,
                    narration="이어지는 설명입니다.",
                    section_title=None,
                    image_paths=image_paths,
                    image_captions=["우리나라 대표적 유적 한눈에 살펴보기"],
                ),
                _scene(seq=4, narration="다른 이야기입니다.", section_title="다른 이야기"),
            ],
        ),
    )

    stats = build_history_content.build_history_content(
        build_history_content.BuildOptions(
            dataset=dataset,
            out=out,
            manifest_only=True,
            force=True,
        )
    )

    doc_payload = json.loads((out / "docs" / f"{doc_hash}.json").read_text(encoding="utf-8"))

    assert stats.sections == 3
    assert [scene["section_index"] for scene in doc_payload["scenes"]] == [0, 1, 1, 2]
    assert [scene["section_title"] for scene in doc_payload["scenes"]] == [
        None,
        "고인돌 이야기",
        "고인돌 이야기",
        "다른 이야기",
    ]
    assert doc_payload["sections"][0]["section_title"] is None
    assert doc_payload["sections"][0]["scene_indices"] == [0]
    assert doc_payload["sections"][1]["section_title"] == "고인돌 이야기"
    assert doc_payload["sections"][1]["scene_indices"] == [1, 2]
    assert doc_payload["sections"][1]["image_captions"] == ["우리나라 대표적 유적 한눈에 살펴보기"]
    assert doc_payload["scenes"][2]["image_captions"] == ["우리나라 대표적 유적 한눈에 살펴보기"]
    assert [image["caption"] for image in doc_payload["scenes"][2]["images"]] == [
        None,
        None,
        None,
    ]
    assert all(image["is_infographic"] for image in doc_payload["scenes"][2]["images"])


def test_prepare_letterboxed_image_downscales_and_centers(tmp_path: Path) -> None:
    source = tmp_path / "wide.jpg"
    red = (250, 0, 0)
    bg = (27, 27, 31)
    _write_jpeg(source, (1440, 720), red)

    prepared = build_history_content.prepare_letterboxed_image(
        source,
        bg_rgb=bg,
        max_dim=720,
    )

    assert prepared.image.size == (720, 720)
    assert prepared.original_size == (1440, 720)
    assert prepared.content_size == (720, 360)
    assert prepared.downscaled is True
    assert prepared.image.getpixel((10, 179)) == bg
    assert prepared.image.getpixel((10, 180)) != bg
    assert prepared.image.getpixel((10, 539)) != bg
    assert prepared.image.getpixel((10, 540)) == bg
