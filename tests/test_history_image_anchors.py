"""Unit tests for the build-time history image-anchor derivation helpers.

These cover the pure-function anchor math (caption parsing, fuzzy location,
monotonic fill, even-spacing fallback, and end-to-end scene annotation) using
synthetic inputs only — no real PDF is required so the suite runs in CI.
"""

from __future__ import annotations

from scripts.history_image_anchors import (
    AnchorStats,
    PdfCaption,
    best_caption_match,
    caption_similarity,
    compute_scene_image_anchors,
    even_anchors,
    locate_anchor_ratio,
    resolve_scene_anchors,
    split_caption_block,
)


def test_split_caption_block_handles_merged_side_by_side_captions() -> None:
    """A merged ``><`` caption block splits into individual captions."""
    text = "<하늘에서 내려오는 환웅><환웅을 만나는 곰과 범>"
    assert split_caption_block(text) == [
        "하늘에서 내려오는 환웅",
        "환웅을 만나는 곰과 범",
    ]


def test_split_caption_block_strips_single_caption_brackets() -> None:
    """A single bracketed caption is unwrapped and trimmed."""
    assert split_caption_block("< 마니산 참성대(인천 강화도) >") == ["마니산 참성대(인천 강화도)"]


def test_caption_similarity_is_whitespace_insensitive() -> None:
    """Identical captions differing only in spacing score a perfect ratio."""
    assert caption_similarity("주먹 도끼", "주먹도끼") == 1.0
    assert caption_similarity("주먹도끼", "전혀다른말") < 0.3


def test_locate_anchor_ratio_finds_midpoint_position() -> None:
    """A snippet from the middle of the narration anchors near the middle."""
    narration = "앞부분 문장입니다. 중간 표지 문장입니다. 끝부분 문장입니다."
    ratio, confidence = locate_anchor_ratio("중간 표지 문장입니다", narration)
    assert confidence >= 0.9
    assert 0.2 < ratio < 0.7


def test_locate_anchor_ratio_reports_zero_confidence_when_absent() -> None:
    """A snippet absent from the narration yields zero confidence."""
    ratio, confidence = locate_anchor_ratio("완전히 없는 문장", "전혀 관계 없는 본문")
    assert confidence < 0.55
    del ratio  # value is unspecified when confidence is low


def test_even_anchors_distribute_uniformly() -> None:
    """Even anchors are uniform and start at zero."""
    assert even_anchors(1) == [0.0]
    assert even_anchors(2) == [0.0, 0.5]
    assert even_anchors(4) == [0.0, 0.25, 0.5, 0.75]


def test_resolve_scene_anchors_pins_real_positions_after_first() -> None:
    """A real pin on the (final) second image lands on the reachable 1.0 end."""
    anchors = resolve_scene_anchors({1: 0.6}, 2)
    assert anchors[0] == 0.0
    # The last slot is forced to 1.0 so the final image is always reachable.
    assert anchors[1] == 1.0


def test_resolve_scene_anchors_is_strictly_increasing_on_collisions() -> None:
    """Colliding pins are nudged apart into a strictly increasing sequence."""
    anchors = resolve_scene_anchors({1: 0.5, 2: 0.5}, 3)
    assert anchors[0] == 0.0
    assert anchors[0] < anchors[1] < anchors[2]
    assert anchors[-1] == 1.0


def test_resolve_scene_anchors_first_image_pin_never_overrides_zero() -> None:
    """A real pin on image 0 cannot push the scene-start image off zero."""
    anchors = resolve_scene_anchors({0: 0.4, 1: 0.7}, 2)
    assert anchors[0] == 0.0
    assert anchors[1] == 1.0


def test_resolve_scene_anchors_falls_back_to_even_without_pins() -> None:
    """With no real pins the tail is spread strictly upward to a 1.0 endpoint."""
    resolved = resolve_scene_anchors({}, 3)
    assert resolved[0] == 0.0
    assert resolved[0] < resolved[1] < resolved[2]
    assert resolved[-1] == 1.0


def test_resolve_scene_anchors_high_pin_spreads_strictly_to_one() -> None:
    """A near-ceiling pin is scaled down so trailing slots stay distinct (F2).

    Repro of the original collision bug: ``{1: 0.95}, 4`` used to emit
    ``[0.0, 0.95, 1.0, 1.0]`` (duplicate trailing 1.0 hid an image). The output
    must now be strictly increasing, in ``[0, 1]``, distinct, and end at 1.0.
    """
    anchors = resolve_scene_anchors({1: 0.95}, 4)
    assert anchors[0] == 0.0
    assert anchors[-1] == 1.0
    assert all(anchors[i] < anchors[i + 1] for i in range(len(anchors) - 1))
    assert len(set(anchors)) == len(anchors)
    assert all(0.0 <= value <= 1.0 for value in anchors)


def test_resolve_scene_anchors_all_ceiling_pins_stay_distinct() -> None:
    """Even all-1.0 pins resolve to a strictly increasing, distinct sequence."""
    anchors = resolve_scene_anchors({1: 1.0, 2: 1.0, 3: 1.0}, 5)
    assert anchors[0] == 0.0
    assert anchors[-1] == 1.0
    assert all(anchors[i] < anchors[i + 1] for i in range(len(anchors) - 1))
    assert len(set(anchors)) == len(anchors)


def test_resolve_scene_anchors_is_idempotent_and_deterministic() -> None:
    """Re-running on the same pins yields the identical strictly-increasing list."""
    assert resolve_scene_anchors({1: 0.95}, 4) == resolve_scene_anchors({1: 0.95}, 4)
    assert resolve_scene_anchors({2: 0.6}, 5) == resolve_scene_anchors({2: 0.6}, 5)


def test_best_caption_match_returns_preceding_text() -> None:
    """The best caption match returns its preceding body text for anchoring."""
    captions = [
        PdfCaption(caption="마니산 참성대(인천 강화도)", preceding_text="첫째 본문"),
        PdfCaption(caption="주먹도끼(국립중앙박물관)", preceding_text="둘째 본문"),
    ]
    ratio, preceding = best_caption_match("주먹도끼(국립중앙박물관)", captions)
    assert ratio >= 0.9
    assert preceding == "둘째 본문"


def test_compute_scene_image_anchors_uses_real_pin_when_caption_matches() -> None:
    """A captioned second image is anchored from its preceding-text position."""
    narration = "도입 문장입니다. 둘째 도구 설명 문장입니다. 마무리 문장입니다."
    images = [
        {"caption": None},
        {"caption": "주먹도끼(국립중앙박물관)"},
    ]
    pdf_captions = [
        PdfCaption(
            caption="주먹도끼(국립중앙박물관)",
            preceding_text="둘째 도구 설명 문장입니다.",
        )
    ]
    stats = AnchorStats()
    anchors = compute_scene_image_anchors(
        images,
        narration,
        pdf_captions,
        stats,
        doc_hash="hash",
        scene_seq=1,
    )
    assert anchors[0] == 0.0
    assert anchors[1] > 0.2
    assert stats.images_real_anchor == 1
    assert stats.images_fallback == 1
    assert stats.scenes_even_fallback == 0


def test_compute_scene_image_anchors_even_fallback_without_captions() -> None:
    """Caption-less multi-image scenes fall back to even spacing and count it."""
    images = [{"caption": None}, {"caption": None}]
    stats = AnchorStats()
    anchors = compute_scene_image_anchors(
        images,
        "본문 문장 하나. 본문 문장 둘.",
        [],
        stats,
        doc_hash="hash",
        scene_seq=2,
    )
    assert anchors == [0.0, 0.5]
    assert stats.images_real_anchor == 0
    assert stats.images_fallback == 2
    assert stats.scenes_even_fallback == 1


def test_compute_scene_image_anchors_single_image_is_zero() -> None:
    """A single-image scene anchors at zero and is not a fallback scene."""
    images = [{"caption": "어떤 그림"}]
    stats = AnchorStats()
    anchors = compute_scene_image_anchors(
        images,
        "본문",
        [],
        stats,
        doc_hash="hash",
        scene_seq=3,
    )
    assert anchors == [0.0]
    assert stats.scenes_even_fallback == 0
