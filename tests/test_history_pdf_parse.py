"""API-free tests for the Upstage history PDF overlay pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
from PIL import Image

from core.history_mode import HistoryImage, HistoryModeController, HistoryScene
from scripts import build_history_content, history_pdf_parse
from scripts.history_pdf_parse import FigureSignal


def _is_targeted_history_pdf_parse_run(args: list[str]) -> bool:
    target = Path("tests/test_history_pdf_parse.py")
    return len(args) == 1 and Path(args[0]).as_posix() == target.as_posix()


@pytest.fixture(scope="session", autouse=True)
def _allow_targeted_script_test_coverage(pytestconfig: pytest.Config) -> None:
    cov_plugin = pytestconfig.pluginmanager.get_plugin("_cov")
    if cov_plugin is None or not _is_targeted_history_pdf_parse_run(pytestconfig.args):
        return
    cov_plugin.options.cov_fail_under = 0


@pytest.fixture(autouse=True)
def _block_http(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_urlopen(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("HTTP must not be called in history_pdf_parse tests")

    monkeypatch.setattr(history_pdf_parse.urllib.request, "urlopen", fail_urlopen)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jpeg(path: Path, color: tuple[int, int, int] = (20, 80, 120)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Image.new("RGB", (16, 12), color) as image:
        image.save(path, format="JPEG")


def _scene(
    *,
    seq: int,
    page: int | None = None,
    narration: str | None = None,
    section_title: str | None = None,
    section_index: int | None = None,
    image_paths: list[str] | None = None,
    image_captions: list[str] | None = None,
) -> dict[str, Any]:
    paths = image_paths or []
    captions = image_captions or []
    scene: dict[str, Any] = {
        "seq": seq,
        "page": page or seq,
        "section_title": section_title,
        "narration": narration or f"{seq}번 장면입니다.",
        "est_speech_ms": 1000,
        "tail_silence_ms": 100,
        "image_paths": paths,
        "image_captions": captions,
        "image_path": paths[0] if paths else None,
        "image_caption": captions[0] if captions else None,
        "anchor_ratio": 0.25,
    }
    if section_index is not None:
        scene["section_index"] = section_index
    return scene


def _document(scenes: list[dict[str, Any]], doc_hash: str = "doc123") -> dict[str, Any]:
    return {
        "doc_hash": doc_hash,
        "source_file": "eh_n0010_0010.pdf",
        "scene_count": len(scenes),
        "scenes": scenes,
    }


def _element(
    element_id: int,
    category: str,
    text: str,
    *,
    page: int,
    html: str | None = None,
) -> dict[str, Any]:
    html_text = html or f"<h1 id='{element_id}' style='font-size:18px'>{text}</h1>"
    return {
        "id": element_id,
        "category": category,
        "page": page,
        "content": {"text": text, "html": html_text, "markdown": ""},
        "coordinates": [
            {"x": 0.1, "y": 0.1},
            {"x": 0.2, "y": 0.1},
            {"x": 0.2, "y": 0.2},
            {"x": 0.1, "y": 0.2},
        ],
    }


def _response(elements: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "api": "2.0",
        "content": {"text": "", "html": "", "markdown": ""},
        "elements": elements,
        "model": "document-parse-test",
        "ocr": False,
        "usage": {"pages": 1, "standard": [1]},
    }


def _runtime_doc(
    *,
    doc_hash: str = "doc123",
    images: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    scene_images = images or []
    return {
        "schema_version": 2,
        "doc_hash": doc_hash,
        "source_file": "eh_n0010_0010.pdf",
        "title": "테스트 문서",
        "kind": "people",
        "era": "고조선",
        "scene_count": 1,
        "section_count": 1,
        "image_count": len(scene_images),
        "est_total_ms": 1000,
        "sections": [
            {
                "section_index": 0,
                "section_title": "첫 제목",
                "scene_indices": [0],
                "scene_seq": [1],
                "image_captions": [],
                "is_infographic": False,
            }
        ],
        "scenes": [
            {
                "seq": 1,
                "section_index": 0,
                "section_title": "첫 제목",
                "narration": "첫 제목 설명입니다.",
                "est_speech_ms": 1000,
                "tail_silence_ms": 0,
                "image_captions": [],
                "images": scene_images,
            }
        ],
    }


def _pipeline_runtime_doc(
    *,
    out: Path,
    doc_hash: str,
    images: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    scene_images = images or []
    return {
        "schema_version": 2,
        "doc_hash": doc_hash,
        "source_file": "eh_n0010_0010.pdf",
        "title": "테스트 문서",
        "kind": "people",
        "era": "고조선",
        "scene_count": 2,
        "section_count": 1,
        "image_count": len(scene_images),
        "est_total_ms": 2200,
        "sections": [
            {
                "section_index": 0,
                "section_title": None,
                "scene_indices": [0, 1],
                "scene_seq": [1, 2],
                "image_captions": ["첫 그림", "둘째 그림"] if scene_images else [],
                "is_infographic": False,
            }
        ],
        "scenes": [
            {
                "seq": 1,
                "section_index": 0,
                "section_title": None,
                "narration": "첫 제목 본문입니다.",
                "est_speech_ms": 1000,
                "tail_silence_ms": 100,
                "image_captions": ["첫 그림", "둘째 그림"] if scene_images else [],
                "images": scene_images,
            },
            {
                "seq": 2,
                "section_index": 0,
                "section_title": None,
                "narration": "둘째 제목 본문입니다.",
                "est_speech_ms": 1000,
                "tail_silence_ms": 100,
                "image_captions": [],
                "images": [],
            },
        ],
    }


def _sectioned_runtime_doc(
    *,
    doc_hash: str = "doc123",
    source_file: str = "eh_n0010_0010.pdf",
    sections: list[tuple[int, str | None, list[int]]],
) -> dict[str, Any]:
    runtime_sections: list[dict[str, Any]] = []
    runtime_scenes: list[dict[str, Any]] = []
    for section_index, section_title, scene_seq in sections:
        scene_indices: list[int] = []
        for seq in scene_seq:
            scene_indices.append(len(runtime_scenes))
            runtime_scenes.append(
                {
                    "seq": seq,
                    "section_index": section_index,
                    "section_title": section_title,
                    "narration": f"{seq}번 기준 본문입니다.",
                    "est_speech_ms": 1000,
                    "tail_silence_ms": 100,
                    "image_captions": [],
                    "images": [],
                }
            )
        runtime_sections.append(
            {
                "section_index": section_index,
                "section_title": section_title,
                "scene_indices": scene_indices,
                "scene_seq": scene_seq,
                "image_captions": [],
                "is_infographic": False,
            }
        )
    return {
        "schema_version": 2,
        "doc_hash": doc_hash,
        "source_file": source_file,
        "title": "테스트 문서",
        "kind": "people",
        "era": "고조선",
        "scene_count": len(runtime_scenes),
        "section_count": len(runtime_sections),
        "image_count": 0,
        "est_total_ms": len(runtime_scenes) * 1100,
        "sections": runtime_sections,
        "scenes": runtime_scenes,
    }


def _image(path: str, caption: str | None, anchor: float) -> dict[str, Any]:
    return {
        "path": path,
        "caption": caption,
        "letterboxed": True,
        "clean": True,
        "is_infographic": False,
        "anchor_ratio": anchor,
    }


def _write_minimal_pipeline_inputs(
    tmp_path: Path, *, with_images: bool = False
) -> tuple[Path, Path, Path, Path, str]:
    dataset = tmp_path / "assets" / "dataset_korean history"
    out = tmp_path / "assets" / "history"
    pdf_dir = tmp_path / "assets" / "우리역사"
    cache_dir = tmp_path / ".upstage_cache"
    doc_hash = "abc123"
    image_paths: list[str] = []
    image_captions: list[str] = []
    runtime_images: list[dict[str, Any]] = []
    if with_images:
        image_paths = [
            f"data/figures/{doc_hash}/fig_001.jpg",
            f"data/figures/{doc_hash}/fig_002.jpg",
        ]
        image_captions = ["첫 그림", "둘째 그림"]
        for index, raw_path in enumerate(image_paths):
            _write_jpeg(dataset / raw_path, color=(20 + index * 40, 80, 120))
        runtime_images = [
            _image(f"{out.as_posix()}/images/{doc_hash}/fig_001.jpg", "첫 그림", 0.0),
            _image(f"{out.as_posix()}/images/{doc_hash}/fig_002.jpg", "둘째 그림", 0.5),
        ]
    _write_json(
        dataset / "data" / "scenes" / f"{doc_hash}.json",
        _document(
            [
                _scene(
                    seq=1,
                    page=1,
                    narration="첫 제목 본문입니다.",
                    image_paths=image_paths,
                    image_captions=image_captions,
                ),
                _scene(seq=2, page=2, narration="둘째 제목 본문입니다."),
            ],
            doc_hash=doc_hash,
        ),
    )
    _write_json(
        out / "docs" / f"{doc_hash}.json",
        _pipeline_runtime_doc(out=out, doc_hash=doc_hash, images=runtime_images),
    )
    pdf_path = pdf_dir / "eh_n0010_0010.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(b"%PDF-1.4 synthetic")
    return dataset, out, pdf_dir, cache_dir, doc_hash


def _write_sectioning_fallback_inputs(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path, str, dict[str, Any]]:
    dataset = tmp_path / "assets" / "dataset_korean history"
    out = tmp_path / "assets" / "history"
    pdf_dir = tmp_path / "assets" / "우리역사"
    cache_dir = tmp_path / ".upstage_cache"
    doc_hash = "abc123"
    baseline = _sectioned_runtime_doc(
        doc_hash=doc_hash,
        sections=[
            (0, "기준 첫 제목", [1, 2]),
            (1, "기준 둘째 제목", [3]),
        ],
    )
    _write_json(out / "docs" / f"{doc_hash}.json", baseline)
    _write_json(
        dataset / "data" / "scenes" / f"{doc_hash}.json",
        _document(
            [
                _scene(seq=1, section_title="<오염된 사진 제목>", section_index=0),
                _scene(seq=2, section_title="<오염된 이어지는 제목>", section_index=1),
                _scene(seq=3, section_title="<오염된 마지막 제목>", section_index=2),
            ],
            doc_hash=doc_hash,
        ),
    )
    pdf_path = pdf_dir / "eh_n0010_0010.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(b"%PDF-1.4 synthetic")
    cache_path = cache_dir / f"{history_pdf_parse.sha256_file(pdf_path)}.json"
    _write_json(cache_path, _response([_element(1, "heading1", "하나뿐인 제목", page=1)]))
    return dataset, out, pdf_dir, cache_dir, doc_hash, baseline


def test_sectioning_fallback_preserves_baseline_titles() -> None:
    document = _document(
        [
            _scene(seq=1, section_title="기존 첫 제목"),
            _scene(seq=2, section_title=None),
            _scene(seq=3, section_title="기존 둘째 제목"),
        ]
    )
    response = _response([_element(1, "heading1", "하나뿐인 제목", page=1)])

    result = history_pdf_parse.overlay_sections(document, response)

    assert result.bucket == "upstage_fallback"
    assert [scene["section_title"] for scene in result.scenes] == [
        "기존 첫 제목",
        None,
        "기존 둘째 제목",
    ]
    assert [scene["section_index"] for scene in result.scenes] == [0, 0, 1]


def test_sectioning_fallback_mirrors_builder_empty_title_and_section_index() -> None:
    document = _document(
        [
            _scene(seq=1, section_title="", section_index=5),
            _scene(seq=2, section_title="ignored continuation", section_index=5),
            _scene(seq=3, section_title="", section_index=9),
        ]
    )

    result = history_pdf_parse.overlay_sections(document, response_payload=None)

    assert result.bucket == "upstage_fallback"
    assert [scene["section_title"] for scene in result.scenes] == [
        None,
        "ignored continuation",
        None,
    ]
    assert [scene["section_index"] for scene in result.scenes] == [0, 0, 1]
    assert result.section_count_before == 2


def test_sectioning_happy_path_filters_false_heading1_and_stamps_indices() -> None:
    document = _document(
        [
            _scene(seq=1, page=1, narration="첫 제목 이야기가 시작됩니다."),
            _scene(seq=2, page=1, narration="이어지는 본문입니다."),
            _scene(seq=3, page=2, narration="둘째 제목 새 본문입니다."),
        ]
    )
    response = _response(
        [
            _element(1, "heading1", "첫 제목", page=1),
            _element(2, "heading1", "“대화문입니다”", page=1),
            _element(3, "heading1", "김돈중이 정중부의 수염에 촛불을 가져다 댄 거예요.", page=1),
            _element(4, "heading1", "둘째 제목", page=2),
        ]
    )

    result = history_pdf_parse.overlay_sections(document, response)

    assert result.bucket == "upstage_ok"
    assert result.headings_raw == 4
    assert result.headings_kept == 2
    assert [scene["section_title"] for scene in result.scenes] == ["첫 제목", None, "둘째 제목"]
    assert [scene["section_index"] for scene in result.scenes] == [0, 0, 1]


def test_caption_style_heading1_is_not_section_cut_but_still_caption_signal() -> None:
    document = _document(
        [
            _scene(seq=1, page=1, narration="첫 제목 도입입니다."),
            _scene(seq=2, page=1, narration="기준 문장입니다."),
            _scene(seq=3, page=2, narration="둘째 제목 마무리입니다."),
        ]
    )
    response = _response(
        [
            _element(1, "heading1", "첫 제목", page=1),
            _element(2, "paragraph", "기준 문장입니다.", page=1),
            _element(3, "heading1", "<만월대(문화재청)>", page=1),
            _element(4, "figure", "![image](/image/placeholder)", page=1),
            _element(5, "heading1", "둘째 제목", page=2),
            _element(6, "heading1", "〈자료 사진〉", page=2),
            _element(7, "heading1", "《자료 사진》", page=2),
        ]
    )

    result = history_pdf_parse.overlay_sections(document, response)
    elements = history_pdf_parse.parse_elements(response)
    signals = history_pdf_parse.collect_figure_signals(elements, result.heading_texts)
    stats = history_pdf_parse.AnchorStats()
    anchors = history_pdf_parse.compute_title_aware_anchors(
        [{"caption": "만월대(문화재청)"}],
        "도입 문장입니다. 기준 문장입니다. 마무리 문장입니다.",
        None,
        signals,
        stats,
    )

    assert result.bucket == "upstage_ok"
    assert result.headings_raw == 5
    assert result.headings_kept == 2
    assert result.heading_texts == ("첫 제목", "둘째 제목")
    assert [scene["section_title"] for scene in result.scenes] == ["첫 제목", None, "둘째 제목"]
    assert [scene["section_index"] for scene in result.scenes] == [0, 0, 1]
    assert len(signals) == 1
    assert signals[0].caption == "만월대(문화재청)"
    assert signals[0].preceding_text == "기준 문장입니다."
    assert anchors == [0.0]
    assert stats.anchors_real == 1
    assert stats.anchors_fallback == 0


def test_heading_mapping_uses_fuzzy_match_before_first_scene_page_fallback() -> None:
    scenes = [
        _scene(seq=1, page=1, narration="도입 본문입니다."),
        _scene(seq=2, page=2, narration="Second Heeding 본문입니다."),
        _scene(seq=3, page=2, narration="같은 쪽 뒤쪽 장면입니다."),
    ]
    fuzzy_heading = history_pdf_parse.Heading(
        text="Second Heading",
        element_id=1,
        order=1,
        page=2,
        font_size_px=18.0,
    )
    page_heading = history_pdf_parse.Heading(
        text="No matching title",
        element_id=2,
        order=2,
        page=2,
        font_size_px=18.0,
    )

    assert history_pdf_parse._map_heading_to_scene(fuzzy_heading, scenes) == 1
    assert history_pdf_parse._map_heading_to_scene(page_heading, scenes) == 1


def test_overlay_schema_preserves_scene_fields_and_removes_scene_anchor() -> None:
    baseline_scene = _scene(
        seq=1,
        page=3,
        narration="첫 제목 원문 그대로입니다.",
        section_title="잘못된 기존 제목",
        image_paths=["data/figures/doc123/fig_001.jpg"],
        image_captions=["그림 설명"],
    )
    document = _document([baseline_scene])
    response = _response(
        [
            _element(1, "heading1", "첫 제목", page=3),
            _element(2, "heading1", "둘째 제목", page=3),
        ]
    )

    result = history_pdf_parse.overlay_sections(document, response)
    overlaid = result.scenes[0]

    for field in history_pdf_parse.SCENE_COPY_FIELDS:
        assert overlaid[field] == baseline_scene[field]
    assert overlaid["section_title"] == "첫 제목"
    assert overlaid["section_index"] == 0
    assert "anchor_ratio" not in overlaid


def test_narration_segments_match_runtime_and_anchor_formula() -> None:
    scene = HistoryScene(
        seq=1,
        section_index=0,
        section_title="작은 제목",
        narration="앞말입니다. 작은 제목 뒤말입니다.",
        est_speech_ms=1000,
        tail_silence_ms=0,
        image_captions=(),
        images=(),
    )

    expected = HistoryModeController._narration_segments(cast(HistoryModeController, None), scene)

    assert history_pdf_parse.narration_segments(scene.narration, scene.section_title) == list(
        expected
    )
    assert history_pdf_parse.narration_segments("제목 없는 본문입니다.", None) == [
        "제목 없는 본문입니다."
    ]
    assert history_pdf_parse.narration_segments("   ", None) == []

    ratio, confidence = history_pdf_parse._spoken_segment_anchor(
        "뒤말입니다.",
        scene.narration,
        scene.section_title,
    )
    runtime_scene = HistoryScene(
        seq=1,
        section_index=0,
        section_title=scene.section_title,
        narration=scene.narration,
        est_speech_ms=1000,
        tail_silence_ms=0,
        image_captions=(),
        images=(
            HistoryImage(Path("first.jpg"), None, True, True, False, 0.0),
            HistoryImage(Path("second.jpg"), None, True, True, False, ratio),
            HistoryImage(Path("third.jpg"), None, True, True, False, 1.0),
        ),
    )

    assert confidence >= history_pdf_parse.MIN_SEGMENT_CONFIDENCE
    assert ratio == 1.0
    assert HistoryModeController._select_scene_image_by_progress(runtime_scene, ratio).path == Path(
        "third.jpg"
    )


def test_figure_preservation_failure_restores_baseline_anchors(tmp_path: Path) -> None:
    baseline = _runtime_doc(
        images=[
            _image("assets/history/images/doc123/fig_001.jpg", "첫 그림", 0.0),
            _image("assets/history/images/doc123/fig_002.jpg", "둘째 그림", 0.5),
        ]
    )
    rebuilt = _runtime_doc(
        images=[
            _image("assets/history/images/doc123/fig_002.jpg", "둘째 그림", 0.0),
            _image("assets/history/images/doc123/fig_001.jpg", "첫 그림", 1.0),
        ]
    )
    docs_dir = tmp_path / "docs"
    doc_path = docs_dir / "doc123.json"
    _write_json(doc_path, rebuilt)
    task = history_pdf_parse.DocumentTask(
        doc_hash="doc123",
        source_file="eh_n0010_0010.pdf",
        scene_path=tmp_path / "scene.json",
        pdf_path=tmp_path / "doc.pdf",
        pdf_sha256="sha",
    )
    context = history_pdf_parse.DocumentContext(
        task=task,
        bucket="upstage_ok",
        heading_texts=("첫 제목",),
        scene_signals={},
        section_count_before=1,
        section_count_after=1,
        headings_raw=1,
        headings_kept=1,
    )

    stats = history_pdf_parse.annotate_runtime_docs(
        docs_dir=docs_dir,
        baseline_docs={"doc123": baseline},
        contexts={"doc123": context},
    )
    payload = json.loads(doc_path.read_text(encoding="utf-8"))

    assert stats.docs_invariant_fallback == 1
    assert [image["anchor_ratio"] for image in payload["scenes"][0]["images"]] == [0.0, 0.5]


def test_anchor_guarantees_with_title_aware_pins() -> None:
    stats = history_pdf_parse.AnchorStats()
    anchors = history_pdf_parse.compute_title_aware_anchors(
        [
            {"caption": "첫 그림"},
            {"caption": "둘째 그림"},
            {"caption": "셋째 그림"},
        ],
        "도입 문장입니다. 표지 문장입니다. 마무리 문장입니다.",
        None,
        [FigureSignal(page=1, order=1, preceding_text="표지 문장입니다.", caption="둘째 그림")],
        stats,
    )

    assert anchors[0] == 0.0
    assert anchors[-1] == 1.0
    assert all(anchors[index] < anchors[index + 1] for index in range(len(anchors) - 1))
    assert all(anchor == round(anchor, 4) for anchor in anchors)
    assert stats.anchors_real == 1


def test_no_pin_multi_image_fallback_uses_normalized_anchors() -> None:
    stats = history_pdf_parse.AnchorStats()

    anchors = history_pdf_parse.compute_title_aware_anchors(
        [{"caption": None}, {"caption": None}, {"caption": None}],
        "핀으로 쓸 문장이 없는 장면입니다.",
        None,
        [],
        stats,
    )

    assert anchors == [0.0, 0.5, 1.0]
    assert stats.scenes_even_fallback == 1


def test_caption_like_paragraph_still_allows_preceding_text_anchor() -> None:
    elements = history_pdf_parse.parse_elements(
        _response(
            [
                _element(1, "paragraph", "첫 문장입니다.", page=1),
                _element(2, "paragraph", "기준 문장입니다.", page=1),
                _element(3, "figure", "![image](/image/placeholder)", page=1),
                _element(4, "paragraph", "<지도 캡션>", page=1),
            ]
        )
    )
    signals = history_pdf_parse.collect_figure_signals(elements, ())
    stats = history_pdf_parse.AnchorStats()

    anchors = history_pdf_parse.compute_title_aware_anchors(
        [{"caption": "다른 캡션"}],
        "첫 문장입니다. 기준 문장입니다.",
        None,
        signals,
        stats,
    )

    assert signals[0].caption == "지도 캡션"
    assert anchors == [0.0]
    assert stats.anchors_real == 1


def test_runtime_schema_validation_and_anchor_ratio_semantics() -> None:
    payload = _runtime_doc(images=[_image("assets/history/images/doc123/fig_001.jpg", None, 0.0)])

    history_pdf_parse.validate_runtime_document(payload)
    history_pdf_parse.validate_anchor_invariants(payload)

    assert history_pdf_parse.parse_anchor_ratio(None) is None
    assert history_pdf_parse.parse_anchor_ratio(-0.5) == 0.0
    assert history_pdf_parse.parse_anchor_ratio(1.5) == 1.0
    assert history_pdf_parse._normalize("e\u0301") == "\u00e9"
    with pytest.raises(ValueError, match="anchor_ratio"):
        history_pdf_parse.parse_anchor_ratio(float("nan"))


def test_cache_hit_short_circuits_http(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dataset, out, pdf_dir, cache_dir, doc_hash = _write_minimal_pipeline_inputs(
        tmp_path, with_images=True
    )
    pdf_path = pdf_dir / "eh_n0010_0010.pdf"
    cache_path = cache_dir / f"{history_pdf_parse.sha256_file(pdf_path)}.json"
    _write_json(
        cache_path,
        _response(
            [
                _element(1, "heading1", "첫 제목", page=1),
                _element(2, "heading1", "둘째 제목", page=2),
            ]
        ),
    )
    monkeypatch.delenv(history_pdf_parse.UPSTAGE_API_KEY_ENV, raising=False)
    args = history_pdf_parse.build_arg_parser().parse_args(
        [
            "--dataset",
            str(dataset),
            "--out",
            str(out),
            "--pdf-dir",
            str(pdf_dir),
            "--cache-dir",
            str(cache_dir),
            "--only",
            doc_hash,
        ]
    )

    history_pdf_parse.run_pipeline(args)

    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    payload = json.loads((out / "docs" / f"{doc_hash}.json").read_text(encoding="utf-8"))
    assert manifest["docs"][0]["bucket"] == "upstage_ok"
    assert manifest["totals"]["upstage_ok"] == 1
    assert [image["anchor_ratio"] for image in payload["scenes"][0]["images"]] == [0.0, 1.0]


def test_corrupt_cache_without_key_falls_back_and_continues(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset, out, pdf_dir, cache_dir, doc_hash = _write_minimal_pipeline_inputs(tmp_path)
    pdf_path = pdf_dir / "eh_n0010_0010.pdf"
    cache_path = cache_dir / f"{history_pdf_parse.sha256_file(pdf_path)}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("{not json", encoding="utf-8")
    monkeypatch.delenv(history_pdf_parse.UPSTAGE_API_KEY_ENV, raising=False)
    args = history_pdf_parse.build_arg_parser().parse_args(
        [
            "--dataset",
            str(dataset),
            "--out",
            str(out),
            "--pdf-dir",
            str(pdf_dir),
            "--cache-dir",
            str(cache_dir),
            "--only",
            doc_hash,
        ]
    )

    history_pdf_parse.run_pipeline(args)

    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["docs"][0]["bucket"] == "upstage_fallback"


def test_fallback_pipeline_uses_runtime_baseline_and_decontaminates_scene_json(
    tmp_path: Path,
) -> None:
    dataset, out, pdf_dir, cache_dir, doc_hash, baseline = _write_sectioning_fallback_inputs(
        tmp_path
    )
    args = history_pdf_parse.build_arg_parser().parse_args(
        [
            "--dataset",
            str(dataset),
            "--out",
            str(out),
            "--pdf-dir",
            str(pdf_dir),
            "--cache-dir",
            str(cache_dir),
            "--only",
            doc_hash,
        ]
    )

    history_pdf_parse.run_pipeline(args)

    scene_payload = json.loads(
        (dataset / "data" / "scenes" / f"{doc_hash}.json").read_text(encoding="utf-8")
    )
    runtime_payload = json.loads((out / "docs" / f"{doc_hash}.json").read_text(encoding="utf-8"))
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["docs"][0]["bucket"] == "upstage_fallback"
    assert [scene["section_title"] for scene in scene_payload["scenes"]] == [
        scene["section_title"] for scene in baseline["scenes"]
    ]
    assert [scene["section_index"] for scene in scene_payload["scenes"]] == [
        scene["section_index"] for scene in baseline["scenes"]
    ]
    assert history_pdf_parse._sectioning_signature(runtime_payload) == (
        history_pdf_parse._sectioning_signature(baseline)
    )


def test_fallback_pipeline_rerun_is_idempotent(tmp_path: Path) -> None:
    dataset, out, pdf_dir, cache_dir, doc_hash, _baseline = _write_sectioning_fallback_inputs(
        tmp_path
    )
    args = history_pdf_parse.build_arg_parser().parse_args(
        [
            "--dataset",
            str(dataset),
            "--out",
            str(out),
            "--pdf-dir",
            str(pdf_dir),
            "--cache-dir",
            str(cache_dir),
            "--only",
            doc_hash,
        ]
    )

    history_pdf_parse.run_pipeline(args)
    scene_path = dataset / "data" / "scenes" / f"{doc_hash}.json"
    doc_path = out / "docs" / f"{doc_hash}.json"
    manifest_path = cache_dir / "manifest.json"
    first_scene = scene_path.read_text(encoding="utf-8")
    first_doc = doc_path.read_text(encoding="utf-8")
    first_manifest = manifest_path.read_text(encoding="utf-8")

    history_pdf_parse.run_pipeline(args)

    assert scene_path.read_text(encoding="utf-8") == first_scene
    assert doc_path.read_text(encoding="utf-8") == first_doc
    assert manifest_path.read_text(encoding="utf-8") == first_manifest


def test_fallback_sectioning_assertion_checks_runtime_grouping(tmp_path: Path) -> None:
    baseline = _sectioned_runtime_doc(
        sections=[
            (0, "기준 첫 제목", [1]),
            (1, "기준 둘째 제목", [2]),
        ],
    )
    rebuilt = json.loads(json.dumps(baseline, ensure_ascii=False))
    rebuilt["sections"][1]["scene_seq"] = [1, 2]
    docs_dir = tmp_path / "docs"
    doc_path = docs_dir / "doc123.json"
    _write_json(doc_path, rebuilt)
    task = history_pdf_parse.DocumentTask(
        doc_hash="doc123",
        source_file="eh_n0010_0010.pdf",
        scene_path=tmp_path / "scene.json",
        pdf_path=tmp_path / "doc.pdf",
        pdf_sha256="sha",
    )
    context = history_pdf_parse.DocumentContext(
        task=task,
        bucket="upstage_fallback",
        heading_texts=(),
        scene_signals={},
        section_count_before=2,
        section_count_after=2,
        headings_raw=1,
        headings_kept=1,
    )

    with pytest.raises(history_pdf_parse.PipelineError, match="Fallback sectioning changed"):
        history_pdf_parse.annotate_runtime_docs(
            docs_dir=docs_dir,
            baseline_docs={"doc123": baseline},
            contexts={"doc123": context},
        )


def test_corrupt_cache_with_key_refetches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dataset, out, pdf_dir, cache_dir, doc_hash = _write_minimal_pipeline_inputs(tmp_path)
    pdf_path = pdf_dir / "eh_n0010_0010.pdf"
    cache_path = cache_dir / f"{history_pdf_parse.sha256_file(pdf_path)}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("{not json", encoding="utf-8")
    monkeypatch.setenv(history_pdf_parse.UPSTAGE_API_KEY_ENV, "test-key")
    monkeypatch.setattr(
        history_pdf_parse,
        "call_upstage_sync",
        lambda _pdf_path, _api_key: _response(
            [
                _element(1, "heading1", "첫 제목", page=1),
                _element(2, "heading1", "둘째 제목", page=2),
            ]
        ),
    )
    args = history_pdf_parse.build_arg_parser().parse_args(
        [
            "--dataset",
            str(dataset),
            "--out",
            str(out),
            "--pdf-dir",
            str(pdf_dir),
            "--cache-dir",
            str(cache_dir),
            "--only",
            doc_hash,
        ]
    )

    history_pdf_parse.run_pipeline(args)

    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["docs"][0]["bucket"] == "upstage_ok"
    assert json.loads(cache_path.read_text(encoding="utf-8"))["model"] == "document-parse-test"


def test_preflight_invalid_baseline_aborts_before_writes(tmp_path: Path) -> None:
    dataset, out, pdf_dir, cache_dir, doc_hash = _write_minimal_pipeline_inputs(
        tmp_path, with_images=True
    )
    scene_path = dataset / "data" / "scenes" / f"{doc_hash}.json"
    doc_path = out / "docs" / f"{doc_hash}.json"
    baseline = json.loads(doc_path.read_text(encoding="utf-8"))
    del baseline["scenes"][0]["images"][0]["anchor_ratio"]
    _write_json(doc_path, baseline)
    scene_text = scene_path.read_text(encoding="utf-8")
    scene_mtime = scene_path.stat().st_mtime_ns
    doc_text = doc_path.read_text(encoding="utf-8")
    doc_mtime = doc_path.stat().st_mtime_ns
    args = history_pdf_parse.build_arg_parser().parse_args(
        [
            "--dataset",
            str(dataset),
            "--out",
            str(out),
            "--pdf-dir",
            str(pdf_dir),
            "--cache-dir",
            str(cache_dir),
            "--skip-api",
            "--only",
            doc_hash,
        ]
    )

    with pytest.raises(ValueError, match="baseline image anchor_ratio"):
        history_pdf_parse.run_pipeline(args)

    assert scene_path.read_text(encoding="utf-8") == scene_text
    assert scene_path.stat().st_mtime_ns == scene_mtime
    assert doc_path.read_text(encoding="utf-8") == doc_text
    assert doc_path.stat().st_mtime_ns == doc_mtime


def test_rebuild_history_content_does_not_touch_images(tmp_path: Path) -> None:
    dataset, out, _pdf_dir, _cache_dir, doc_hash = _write_minimal_pipeline_inputs(
        tmp_path, with_images=True
    )
    image_path = out / "images" / doc_hash / "fig_001.jpg"
    _write_jpeg(image_path, color=(200, 20, 80))
    before_bytes = image_path.read_bytes()
    before_mtime = image_path.stat().st_mtime_ns

    history_pdf_parse._rebuild_history_content(dataset, out)

    assert image_path.read_bytes() == before_bytes
    assert image_path.stat().st_mtime_ns == before_mtime


def test_concurrency_flag_is_removed() -> None:
    with pytest.raises(SystemExit):
        history_pdf_parse.build_arg_parser().parse_args(["--concurrency", "2"])


def test_skip_api_missing_cache_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset, out, pdf_dir, cache_dir, doc_hash = _write_minimal_pipeline_inputs(tmp_path)
    scene_path = dataset / "data" / "scenes" / f"{doc_hash}.json"
    before = scene_path.read_text(encoding="utf-8")
    monkeypatch.delenv(history_pdf_parse.UPSTAGE_API_KEY_ENV, raising=False)
    args = history_pdf_parse.build_arg_parser().parse_args(
        [
            "--dataset",
            str(dataset),
            "--out",
            str(out),
            "--pdf-dir",
            str(pdf_dir),
            "--cache-dir",
            str(cache_dir),
            "--skip-api",
            "--only",
            doc_hash,
        ]
    )

    history_pdf_parse.run_pipeline(args)

    assert scene_path.read_text(encoding="utf-8") == before
    assert not (cache_dir / "manifest.json").exists()


def test_key_handling_without_api_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dataset, out, pdf_dir, cache_dir, doc_hash = _write_minimal_pipeline_inputs(tmp_path)
    monkeypatch.delenv(history_pdf_parse.UPSTAGE_API_KEY_ENV, raising=False)

    missing_key_exit = history_pdf_parse.main(
        [
            "--dataset",
            str(dataset),
            "--out",
            str(out),
            "--pdf-dir",
            str(pdf_dir),
            "--cache-dir",
            str(cache_dir),
            "--only",
            doc_hash,
        ]
    )
    dry_run_exit = history_pdf_parse.main(
        [
            "--dataset",
            str(dataset),
            "--out",
            str(out),
            "--pdf-dir",
            str(pdf_dir),
            "--cache-dir",
            str(cache_dir),
            "--dry-run",
            "--only",
            doc_hash,
        ]
    )
    skip_api_exit = history_pdf_parse.main(
        [
            "--dataset",
            str(dataset),
            "--out",
            str(out),
            "--pdf-dir",
            str(pdf_dir),
            "--cache-dir",
            str(cache_dir),
            "--skip-api",
            "--only",
            doc_hash,
        ]
    )

    assert missing_key_exit == history_pdf_parse.CONFIG_EXIT
    assert dry_run_exit == history_pdf_parse.SUCCESS_EXIT
    assert skip_api_exit == history_pdf_parse.SUCCESS_EXIT
    assert not (cache_dir / "manifest.json").exists()


def test_builder_hardening_uses_section_index_when_present(tmp_path: Path) -> None:
    dataset = tmp_path / "assets" / "dataset_korean history"
    out = tmp_path / "assets" / "history"
    _write_json(
        dataset / "data" / "scenes" / "doc123.json",
        _document(
            [
                _scene(seq=1, section_title="첫 제목", section_index=0),
                _scene(seq=2, section_title="반복 제목이어도 새 섹션 아님", section_index=0),
                _scene(seq=3, section_title=None, section_index=1),
            ]
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
    payload = json.loads((out / "docs" / "doc123.json").read_text(encoding="utf-8"))

    assert stats.sections == 2
    assert [scene["section_index"] for scene in payload["scenes"]] == [0, 0, 1]
    assert payload["sections"][0]["scene_seq"] == [1, 2]
    assert payload["sections"][1]["scene_seq"] == [3]
