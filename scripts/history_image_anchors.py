"""Derive per-image narration anchors for the curated Korean-history mode.

The runtime ``assets/history/docs/{doc_hash}.json`` files describe each scene's
narration and its attached images, but the images carry no information about
*where* in the narration they belong. Without that, the narration worker can
only cycle images positionally, so the displayed picture drifts away from what
is being spoken.

This build-time tool recovers a narration position (``anchor_ratio`` in
``[0, 1]``) for every image by inspecting the original source PDFs under
``assets/우리역사/``. For each document it:

1. extracts text blocks from every PDF page in column-aware reading order,
2. collects figure captions (``<...>`` style) together with the body text that
   immediately precedes them in reading order,
3. fuzzy-matches each scene image's caption to a PDF caption and locates the
   preceding body text inside the scene narration to obtain a real anchor,
4. fills the remaining (caption-less or unmatched) images with monotonically
   increasing, evenly spaced anchors so the slideshow never flips backward.

When a PDF is missing, an image cannot be correlated, or the fuzzy-match
confidence is too low, the scene's images fall back to pure even spacing
(``anchor_ratio = i / len(images)``) and the event is logged and counted.

The tool is deterministic and idempotent: re-running it rewrites the same
``anchor_ratio`` values. ``fitz`` (PyMuPDF) is imported lazily inside this
build-time module and is never imported by ``core/``.
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("mungi.scripts.history_image_anchors")

DEFAULT_DOCS_DIR = Path("assets/history/docs")
DEFAULT_PDF_DIR = Path("assets/우리역사")

# A text block is treated as a caption when it is fully wrapped in figure
# brackets, e.g. "<연천 전곡리 선사유적지(경기 연천군)>".
CAPTION_RE = re.compile(r"^[<\[(＜].*[>\])＞]\s*$")
# Side-by-side figures often share one caption block joined by "><" / "> <".
CAPTION_SPLIT_RE = re.compile(r">\s*<")
# Only the outermost figure-bracket pair is stripped (inner parens are kept).
CAPTION_OPEN_CHARS = "<[＜"
CAPTION_CLOSE_CHARS = ">]＞"

# Confidence thresholds for accepting a real (PDF-derived) anchor.
MIN_CAPTION_RATIO = 0.72
MIN_LOCATE_CONFIDENCE = 0.55

# Page-geometry heuristics (PDF points). Running headers are tall rotated
# blocks pinned to the top; footers carry page numbers at the very bottom.
HEADER_BAND_RATIO = 0.10
HEADER_MIN_HEIGHT = 80.0
FOOTER_BAND_RATIO = 0.94


@dataclass(frozen=True)
class PdfCaption:
    """One figure caption recovered from a PDF, with its preceding body text."""

    caption: str
    preceding_text: str


@dataclass
class AnchorStats:
    """Counters collected during one anchor build."""

    docs: int = 0
    docs_missing_pdf: int = 0
    images_total: int = 0
    images_real_anchor: int = 0
    images_fallback: int = 0
    scenes_even_fallback: int = 0
    docs_written: int = 0
    docs_unchanged: int = 0
    warnings: list[str] = field(default_factory=list)


def _normalize(text: str) -> str:
    """Collapse all whitespace for whitespace-insensitive comparison."""
    return re.sub(r"\s+", "", text)


def _strip_caption_wrappers(part: str) -> str:
    """Strip surrounding figure brackets while preserving inner parentheses."""
    stripped = part.strip()
    while stripped and stripped[0] in CAPTION_OPEN_CHARS:
        stripped = stripped[1:].strip()
    while stripped and stripped[-1] in CAPTION_CLOSE_CHARS:
        stripped = stripped[:-1].strip()
    return stripped


def split_caption_block(text: str) -> list[str]:
    """Split a (possibly merged) caption block into individual captions."""
    parts = CAPTION_SPLIT_RE.split(text.strip())
    captions: list[str] = []
    for part in parts:
        stripped = _strip_caption_wrappers(part)
        if stripped:
            captions.append(stripped)
    return captions


def caption_similarity(scene_caption: str, pdf_caption: str) -> float:
    """Return the whitespace-insensitive similarity ratio of two captions."""
    return difflib.SequenceMatcher(None, _normalize(scene_caption), _normalize(pdf_caption)).ratio()


def locate_anchor_ratio(snippet: str, narration: str) -> tuple[float, float]:
    """Locate ``snippet`` inside ``narration`` and return (ratio, confidence).

    ``ratio`` is the character offset of the matched region's start divided by
    the narration length, in ``[0, 1]``. ``confidence`` is the fraction of the
    snippet that matched (longest common contiguous run / snippet length).
    """
    narration_norm = _normalize(narration)
    snippet_norm = _normalize(snippet)
    if not narration_norm or not snippet_norm:
        return 0.0, 0.0
    original_index = [i for i, char in enumerate(narration) if not char.isspace()]
    matcher = difflib.SequenceMatcher(None, narration_norm, snippet_norm)
    match = matcher.find_longest_match(0, len(narration_norm), 0, len(snippet_norm))
    if match.size == 0:
        return 0.0, 0.0
    confidence = match.size / len(snippet_norm)
    start = original_index[match.a] if match.a < len(original_index) else len(narration)
    return start / len(narration), confidence


def best_caption_match(scene_caption: str, pdf_captions: list[PdfCaption]) -> tuple[float, str]:
    """Return the (ratio, preceding_text) of the best PDF caption match."""
    best_ratio = 0.0
    best_preceding = ""
    for candidate in pdf_captions:
        ratio = caption_similarity(scene_caption, candidate.caption)
        if ratio > best_ratio:
            best_ratio = ratio
            best_preceding = candidate.preceding_text
    return best_ratio, best_preceding


def even_anchors(count: int) -> list[float]:
    """Return evenly spaced anchors ``[0, 1/count, 2/count, ...]``."""
    if count <= 1:
        return [0.0] * max(count, 0)
    return [index / count for index in range(count)]


def resolve_scene_anchors(
    real_pins: dict[int, float],
    count: int,
) -> list[float]:
    """Combine real (PDF-derived) pins with even fill into ordered anchors.

    Image 0 is always anchored at ``0.0`` (it is shown when the scene begins).
    Real pins position later images; gaps between known pins are filled with
    strictly increasing interpolation and the tail is spread strictly upward to
    end at exactly ``1.0``. When a pin sits too high to leave room for the
    distinct trailing slots it requires, it is scaled down just enough to fit.

    The returned anchors are guaranteed **strictly increasing with no
    duplicates** and confined to ``[0, 1]`` (the first is ``0.0`` and, whenever
    ``count > 1``, the last is ``1.0``), so the runtime can reach every image and
    never flips one backward. The function is deterministic and idempotent.
    """
    if count <= 1:
        return [0.0] * max(count, 0)

    anchors: list[float | None] = [None] * count
    anchors[0] = 0.0
    for index, ratio in real_pins.items():
        if 0 < index < count:
            anchors[index] = float(min(max(ratio, 0.0), 1.0))

    # Force the final slot to land on 1.0 so the last image is always reachable.
    if anchors[count - 1] is None:
        anchors[count - 1] = 1.0

    # Fill caption-less gaps with strictly increasing interpolation between the
    # nearest known pins (the endpoints 0.0 and 1.0 are always known here).
    known = [index for index in range(count) if anchors[index] is not None]
    for current, nxt in zip(known, known[1:], strict=False):
        start_value = anchors[current]
        end_value = anchors[nxt]
        assert start_value is not None and end_value is not None
        span = nxt - current
        for position in range(current + 1, nxt):
            fraction = (position - current) / span
            anchors[position] = start_value + (end_value - start_value) * fraction

    # Enforce a strictly increasing, duplicate-free sequence inside [0, 1] that
    # starts at 0.0 and ends at 1.0. ``step`` is small enough that the tightest
    # back-clamp (``ceiling`` of the first slot, 1 - (count-1)*step = 0.5) stays
    # non-negative, so a feasible strictly increasing assignment always exists
    # even when real pins crowd the ceiling.
    step = 0.5 / (count - 1)
    bounded: list[float] = [value if value is not None else 0.0 for value in anchors]

    # Back pass: cap each slot below its successor (the last is pinned to 1.0)
    # so over-high pins are scaled down enough to leave room for the tail.
    bounded[count - 1] = 1.0
    for position in range(count - 2, 0, -1):
        ceiling = bounded[position + 1] - step
        if bounded[position] > ceiling:
            bounded[position] = ceiling

    # Forward pass: lift colliding/low slots to clear their predecessor. The
    # back pass guarantees enough headroom that this never exceeds 1.0.
    bounded[0] = 0.0
    for position in range(1, count):
        floor = bounded[position - 1] + step
        if bounded[position] < floor:
            bounded[position] = floor

    bounded[count - 1] = 1.0
    resolved = [round(value, 4) for value in bounded]

    # Rounding (4 dp) can re-collide adjacent slots; re-separate from the left
    # while keeping 0.0 and 1.0 fixed at the endpoints.
    for position in range(1, count - 1):
        if resolved[position] <= resolved[position - 1]:
            resolved[position] = round(min(resolved[position - 1] + step, 1.0), 4)
    return resolved


def _open_pdf(pdf_path: Path) -> Any:
    """Open a PDF lazily importing PyMuPDF (build-time only dependency)."""
    import fitz  # type: ignore[import-not-found, import-untyped]

    return fitz.open(pdf_path)


def _page_reading_blocks(page: Any) -> list[tuple[bool, str]]:
    """Return ``(is_caption, text)`` blocks in column-aware reading order."""
    info = page.get_text("dict")
    page_width = page.rect.width
    page_height = page.rect.height
    midline = page_width / 2.0
    rows: list[tuple[int, float, float, bool, str]] = []
    for block in info["blocks"]:
        if block["type"] != 0:
            continue
        x0, y0, x1, y1 = block["bbox"]
        if y0 < page_height * HEADER_BAND_RATIO and (y1 - y0) > HEADER_MIN_HEIGHT:
            continue
        if y0 > page_height * FOOTER_BAND_RATIO:
            continue
        text = "".join(span["text"] for line in block["lines"] for span in line["spans"]).strip()
        if not text:
            continue
        center_x = (x0 + x1) / 2.0
        column = 0 if center_x < midline else 1
        is_caption = bool(CAPTION_RE.match(text))
        rows.append((column, y0, x0, is_caption, text))
    rows.sort(key=lambda row: (row[0], row[1], row[2]))
    return [(row[3], row[4]) for row in rows]


def collect_pdf_captions(pdf_path: Path) -> list[PdfCaption]:
    """Extract figure captions and their preceding body text from a PDF."""
    document = _open_pdf(pdf_path)
    try:
        captions: list[PdfCaption] = []
        for page_number in range(document.page_count):
            preceding = ""
            for is_caption, text in _page_reading_blocks(document[page_number]):
                if not is_caption:
                    preceding = text
                    continue
                for caption in split_caption_block(text):
                    captions.append(PdfCaption(caption=caption, preceding_text=preceding))
        return captions
    finally:
        document.close()


def compute_scene_image_anchors(
    images: list[dict[str, Any]],
    narration: str,
    pdf_captions: list[PdfCaption],
    stats: AnchorStats,
    *,
    doc_hash: str,
    scene_seq: Any,
) -> list[float]:
    """Compute anchors for one scene's images, recording real/fallback counts."""
    count = len(images)
    if count == 0:
        return []
    real_pins: dict[int, float] = {}
    for index, image in enumerate(images):
        caption = image.get("caption")
        if not isinstance(caption, str) or not caption.strip():
            continue
        ratio, preceding = best_caption_match(caption, pdf_captions)
        if ratio < MIN_CAPTION_RATIO or not preceding:
            continue
        anchor, confidence = locate_anchor_ratio(preceding, narration)
        if confidence >= MIN_LOCATE_CONFIDENCE:
            real_pins[index] = anchor

    if real_pins:
        anchors = resolve_scene_anchors(real_pins, count)
    else:
        anchors = even_anchors(count)
        if count > 1:
            stats.scenes_even_fallback += 1
            logger.debug(
                "Even-spaced fallback for %s scene seq=%s (%d images, no caption match)",
                doc_hash,
                scene_seq,
                count,
            )

    stats.images_total += count
    stats.images_real_anchor += len(real_pins)
    stats.images_fallback += count - len(real_pins)
    return anchors


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = f"Expected JSON object: {path}"
        raise ValueError(msg)
    return payload


def _scene_list(payload: dict[str, Any], doc_path: Path) -> list[dict[str, Any]]:
    scenes = payload.get("scenes")
    if not isinstance(scenes, list):
        msg = f"Document scenes must be a list: {doc_path}"
        raise ValueError(msg)
    typed: list[dict[str, Any]] = []
    for scene in scenes:
        if not isinstance(scene, dict):
            msg = f"Document scene must be an object: {doc_path}"
            raise ValueError(msg)
        typed.append(scene)
    return typed


def annotate_document(
    doc_path: Path,
    pdf_dir: Path,
    stats: AnchorStats,
    *,
    dry_run: bool,
) -> None:
    """Annotate one runtime document JSON with per-image ``anchor_ratio``."""
    payload = _load_json(doc_path)
    doc_hash = str(payload.get("doc_hash") or doc_path.stem)
    source_file = payload.get("source_file")
    scenes = _scene_list(payload, doc_path)

    pdf_captions: list[PdfCaption] = []
    pdf_available = False
    if isinstance(source_file, str) and source_file:
        pdf_path = pdf_dir / source_file
        if pdf_path.exists():
            try:
                pdf_captions = collect_pdf_captions(pdf_path)
                pdf_available = True
            except Exception:  # noqa: BLE001 - resilient build tool
                message = f"Failed to read PDF for {doc_hash} ({source_file}); using even spacing"
                logger.warning(message, exc_info=True)
                stats.warnings.append(message)
        else:
            message = f"PDF missing for {doc_hash} ({source_file}); using even spacing"
            logger.warning(message)
            stats.warnings.append(message)
    else:
        message = f"No source_file for {doc_hash}; using even spacing"
        logger.warning(message)
        stats.warnings.append(message)

    if not pdf_available:
        stats.docs_missing_pdf += 1

    for scene in scenes:
        images = scene.get("images")
        if not isinstance(images, list) or not images:
            continue
        narration = scene.get("narration")
        narration_text = narration if isinstance(narration, str) else ""
        anchors = compute_scene_image_anchors(
            images,
            narration_text,
            pdf_captions,
            stats,
            doc_hash=doc_hash,
            scene_seq=scene.get("seq"),
        )
        for image, anchor in zip(images, anchors, strict=True):
            if isinstance(image, dict):
                image["anchor_ratio"] = anchor

    stats.docs += 1
    serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if dry_run:
        logger.info("dry-run: would write %s", doc_path.as_posix())
        return
    if doc_path.read_text(encoding="utf-8") == serialized:
        stats.docs_unchanged += 1
        return
    doc_path.write_text(serialized, encoding="utf-8")
    stats.docs_written += 1


def build_anchors(docs_dir: Path, pdf_dir: Path, *, dry_run: bool) -> AnchorStats:
    """Annotate every runtime document under ``docs_dir`` with image anchors."""
    if not docs_dir.exists():
        msg = f"Missing docs directory: {docs_dir}"
        raise FileNotFoundError(msg)
    stats = AnchorStats()
    for doc_path in sorted(docs_dir.glob("*.json")):
        annotate_document(doc_path, pdf_dir, stats, dry_run=dry_run)
    _log_stats(stats)
    return stats


def _log_stats(stats: AnchorStats) -> None:
    logger.info(
        "docs=%s written=%s unchanged=%s missing_pdf=%s",
        stats.docs,
        stats.docs_written,
        stats.docs_unchanged,
        stats.docs_missing_pdf,
    )
    logger.info(
        "images=%s real_anchor=%s fallback=%s scenes_even_fallback=%s",
        stats.images_total,
        stats.images_real_anchor,
        stats.images_fallback,
        stats.scenes_even_fallback,
    )
    if stats.images_total:
        coverage = 100.0 * stats.images_real_anchor / stats.images_total
        logger.info("real-anchor coverage=%.1f%%", coverage)
    logger.info("warnings=%s", len(stats.warnings))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS_DIR)
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for the anchor build."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    try:
        build_anchors(args.docs_dir, args.pdf_dir, dry_run=args.dry_run)
    except (OSError, ValueError) as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
