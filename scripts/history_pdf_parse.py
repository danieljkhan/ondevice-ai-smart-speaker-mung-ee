"""Overlay Upstage Document Parse structure onto curated history content.

This build-time, one-shot tool reads the existing Korean-history scene JSON,
parses the source PDFs through Upstage Document Parse when a raw-response cache
is missing, overlays only section metadata into the upstream scene layer, then
rebuilds runtime docs and annotates title-aware image anchors post-build.

The script is idempotent by PDF sha256 cache key. ``--dry-run`` performs no API
calls and no file writes. ``--skip-api`` is a cache-only pilot mode: selected
PDFs without cache are reported and left untouched while cached PDFs can still
be processed. Runtime docs outside the processed set have baseline anchors
restored after the in-process rebuild so pilot runs do not erase existing
anchors. If a run is interrupted between rebuild and annotate, recover the
runtime-doc baseline with ``git restore 'assets/history/docs'``.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import logging
import math
import os
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

from models.tts_runner import _split_text_into_sentences
from scripts import build_history_content
from scripts.history_image_anchors import (
    caption_similarity,
    resolve_scene_anchors,
    split_caption_block,
)

logger = logging.getLogger("mungi.scripts.history_pdf_parse")

JsonObject = dict[str, Any]
Bucket = Literal["upstage_ok", "upstage_fallback", "untouched"]
SceneSectionSignature = tuple[int, int, str | None]
SectionRecordSignature = tuple[int, str | None, tuple[int, ...]]
SectioningSignature = tuple[tuple[SceneSectionSignature, ...], tuple[SectionRecordSignature, ...]]

DEFAULT_CACHE_DIR = Path(".upstage_cache")
DEFAULT_PDF_DIR = Path("assets/우리역사")
UPSTAGE_API_KEY_ENV = "UPSTAGE_API_KEY"
UPSTAGE_SYNC_URL = "https://api.upstage.ai/v1/document-digitization"
UPSTAGE_MODEL = "document-parse"
UPSTAGE_OUTPUT_FORMATS = ("text", "html")
UPSTAGE_BASE64_ENCODING = ("figure",)
HTTP_TIMEOUT_S = 60.0
PER_DOC_DEADLINE_S = 300.0
RETRY_DELAYS_S = (2.0, 4.0, 8.0, 16.0, 32.0)
UPSTAGE_MIN_INTERVAL_S = 1.0
MAX_HEADING_CHARS = 40
MIN_HEADING_FUZZY_RATIO = 0.85
MAX_SECTION_SPOKEN_MS = 180_000
MIN_CAPTION_RATIO = 0.72
MIN_SEGMENT_CONFIDENCE = 0.55
CONFIG_EXIT = 2
ERROR_EXIT = 1
SUCCESS_EXIT = 0
SCENE_COPY_FIELDS = (
    "seq",
    "page",
    "narration",
    "est_speech_ms",
    "tail_silence_ms",
    "image_paths",
    "image_captions",
    "image_path",
    "image_caption",
)
TEXT_ELEMENT_CATEGORIES = {"paragraph", "list"}
CAPTION_ELEMENT_CATEGORIES = {"caption"}
CAPTION_HEADING_START_CHARS = ("<", "〈", "《")
QUOTE_START_CHARS = ('"', "'", "“", "‘", "「", "『")
SENTENCE_FINAL_RE = re.compile(
    r"(?:다|요|까|라|죠|지요|어요|예요|이에요|나요|군요|구나|습니다|답니다)[.!?。！？]$"
)
FONT_SIZE_RE = re.compile(r"font-size\s*:\s*(?P<size>\d+(?:\.\d+)?)px")


class ConfigError(Exception):
    """Raised when CLI arguments or local inputs are invalid."""


class PipelineError(Exception):
    """Raised when deterministic overlay or validation fails."""


class UpstageApiError(Exception):
    """Raised when one Upstage request fails after retry policy is exhausted."""


@dataclass(frozen=True)
class Point:
    """One relative coordinate point from Upstage."""

    x: float
    y: float


@dataclass(frozen=True)
class UpstageElement:
    """One normalized Upstage layout element."""

    element_id: int
    order: int
    category: str
    text: str
    html: str
    page: int
    coordinates: tuple[Point, ...]

    @property
    def center(self) -> tuple[float, float] | None:
        """Return the element center from coordinates when present."""
        if not self.coordinates:
            return None
        x_value = sum(point.x for point in self.coordinates) / len(self.coordinates)
        y_value = sum(point.y for point in self.coordinates) / len(self.coordinates)
        return (x_value, y_value)


@dataclass(frozen=True)
class Heading:
    """A kept Upstage heading used as a section cut."""

    text: str
    element_id: int
    order: int
    page: int
    font_size_px: float | None


@dataclass(frozen=True)
class FigureSignal:
    """Reading-order evidence used to anchor one image to narration."""

    page: int
    order: int
    preceding_text: str
    caption: str | None = None


@dataclass(frozen=True)
class SectionCut:
    """One section boundary in scene-index coordinates."""

    scene_index: int
    title: str | None
    from_heading: bool


@dataclass(frozen=True)
class SectioningResult:
    """Outcome of section overlay for one document."""

    bucket: Bucket
    scenes: list[JsonObject]
    headings_raw: int
    headings_kept: int
    heading_texts: tuple[str, ...]
    section_count_before: int
    section_count_after: int
    reason: str | None = None


@dataclass(frozen=True)
class DocumentTask:
    """One selected PDF and its matching scene JSON document."""

    doc_hash: str
    source_file: str
    scene_path: Path
    pdf_path: Path
    pdf_sha256: str


@dataclass
class AnchorStats:
    """Counters collected while annotating rebuilt runtime docs."""

    docs: int = 0
    docs_written: int = 0
    docs_unchanged: int = 0
    docs_anchor_restored: int = 0
    docs_invariant_fallback: int = 0
    images_total: int = 0
    anchors_real: int = 0
    anchors_fallback: int = 0
    scenes_even_fallback: int = 0


@dataclass
class RunTotals:
    """Run-level telemetry counters for the manifest."""

    upstage_ok: int = 0
    upstage_fallback: int = 0
    untouched: int = 0
    missing_cache: int = 0
    headings_raw: int = 0
    headings_kept: int = 0
    anchors_real: int = 0
    anchors_fallback: int = 0


@dataclass
class DocumentContext:
    """Per-document post-process metadata needed after the content rebuild."""

    task: DocumentTask
    bucket: Bucket
    heading_texts: tuple[str, ...]
    scene_signals: dict[int, list[FigureSignal]]
    section_count_before: int
    section_count_after: int
    headings_raw: int
    headings_kept: int
    model: str | None = None
    reason: str | None = None
    anchors_real: int = 0
    anchors_fallback: int = 0


def parse_anchor_ratio(value: Any) -> float | None:
    """Parse runtime ``anchor_ratio`` with the same semantics as history mode."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("history image anchor_ratio must be a number or null")
    numeric = float(value)
    if math.isnan(numeric):
        raise ValueError("history image anchor_ratio must be a number or null")
    return min(max(numeric, 0.0), 1.0)


def narration_segments(narration: str, section_title: str | None) -> list[str | None]:
    """Mirror ``HistoryModeController._narration_segments`` without importing core."""
    title = (section_title or "").strip()
    if not title or title not in narration:
        base_segments: list[str | None] = list(_sentence_segments(narration))
        return base_segments

    before, rest = narration.split(title, 1)
    after = rest
    segments: list[str | None] = []
    segments.extend(_sentence_segments(before))
    if segments:
        segments.append(None)
    segments.extend(_sentence_segments(title))
    if after.strip():
        segments.append(None)
        segments.extend(_sentence_segments(after))
    return segments


def overlay_sections(
    document: JsonObject,
    response_payload: JsonObject | None,
    *,
    baseline_document: JsonObject | None = None,
    max_section_ms: int = MAX_SECTION_SPOKEN_MS,
) -> SectioningResult:
    """Return an upstream scene overlay using Upstage headings or fallback sectioning."""
    scenes = _scene_list(document)
    section_count_before = _source_section_count(scenes, baseline_document)
    if response_payload is None:
        fallback = _fallback_scenes(scenes, baseline_document)
        return SectioningResult(
            bucket="upstage_fallback",
            scenes=fallback,
            headings_raw=0,
            headings_kept=0,
            heading_texts=(),
            section_count_before=section_count_before,
            section_count_after=_baseline_section_count(fallback),
            reason="no upstage response",
        )

    elements = parse_elements(response_payload)
    raw_headings = [element for element in elements if element.category == "heading1"]
    kept_headings = filter_heading_elements(raw_headings)
    if len(kept_headings) < 2:
        fallback = _fallback_scenes(scenes, baseline_document)
        return SectioningResult(
            bucket="upstage_fallback",
            scenes=fallback,
            headings_raw=len(raw_headings),
            headings_kept=len(kept_headings),
            heading_texts=tuple(heading.text for heading in kept_headings),
            section_count_before=section_count_before,
            section_count_after=_baseline_section_count(fallback),
            reason="heading_count < 2",
        )

    cuts = _heading_section_cuts(kept_headings, scenes)
    overlaid = _apply_section_cuts(scenes, cuts, max_section_ms=max_section_ms)
    section_count_after = _section_count_from_indices(overlaid)
    baseline_single_section = section_count_before == 1
    if not _section_count_is_valid(section_count_after, len(scenes), baseline_single_section):
        fallback = _fallback_scenes(scenes, baseline_document)
        return SectioningResult(
            bucket="upstage_fallback",
            scenes=fallback,
            headings_raw=len(raw_headings),
            headings_kept=len(kept_headings),
            heading_texts=tuple(heading.text for heading in kept_headings),
            section_count_before=section_count_before,
            section_count_after=_baseline_section_count(fallback),
            reason="section-count gate failed",
        )
    return SectioningResult(
        bucket="upstage_ok",
        scenes=overlaid,
        headings_raw=len(raw_headings),
        headings_kept=len(kept_headings),
        heading_texts=tuple(heading.text for heading in kept_headings),
        section_count_before=section_count_before,
        section_count_after=section_count_after,
    )


def parse_elements(response_payload: JsonObject) -> list[UpstageElement]:
    """Normalize Upstage ``elements[]`` into typed reading-order elements."""
    raw_elements = response_payload.get("elements")
    if not isinstance(raw_elements, list):
        return []
    elements: list[UpstageElement] = []
    for fallback_order, raw in enumerate(raw_elements):
        if not isinstance(raw, dict):
            continue
        category = raw.get("category")
        if not isinstance(category, str):
            continue
        element_id = _optional_int(raw.get("id"), fallback_order)
        page = _optional_int(raw.get("page"), 0)
        content = raw.get("content")
        content_obj = content if isinstance(content, dict) else {}
        text = content_obj.get("text")
        html = content_obj.get("html")
        elements.append(
            UpstageElement(
                element_id=element_id,
                order=element_id,
                category=category,
                text=text if isinstance(text, str) else "",
                html=html if isinstance(html, str) else "",
                page=page,
                coordinates=_parse_coordinates(raw.get("coordinates")),
            )
        )
    elements.sort(key=lambda element: element.order)
    return elements


def filter_heading_elements(elements: list[UpstageElement]) -> list[Heading]:
    """Apply the mandatory heading1 pre-filter from the methodology."""
    headings: list[Heading] = []
    for element in elements:
        text = _clean_text(element.text)
        font_size = _font_size_px(element.html)
        if not text:
            continue
        if _is_caption_style_heading(text):
            continue
        if _is_quote_wrapped(text):
            continue
        if _is_sentence_final_korean_prose(text):
            continue
        if len(_normalize(text)) > MAX_HEADING_CHARS:
            continue
        headings.append(
            Heading(
                text=text,
                element_id=element.element_id,
                order=element.order,
                page=element.page,
                font_size_px=font_size,
            )
        )
    headings.sort(key=lambda heading: (heading.order, -(heading.font_size_px or 0.0)))
    return headings


def collect_figure_signals(
    elements: list[UpstageElement], headings: tuple[str, ...]
) -> list[FigureSignal]:
    """Collect reading-order figure signals from parsed Upstage elements."""
    heading_set = {_normalize(text) for text in headings}
    section_index = 0
    last_text_by_section: dict[int, str] = {}
    caption_candidates = _caption_candidates(elements)
    signals: list[FigureSignal] = []
    for element in elements:
        normalized = _normalize(_clean_text(element.text))
        if element.category == "heading1" and normalized in heading_set:
            section_index += 1
            continue
        if element.category == "figure":
            caption = _nearest_caption_text(element, caption_candidates)
            signals.append(
                FigureSignal(
                    page=element.page,
                    order=element.order,
                    preceding_text=last_text_by_section.get(section_index, ""),
                    caption=caption,
                )
            )
            continue
        if element.category in TEXT_ELEMENT_CATEGORIES:
            text = _clean_text(element.text)
            if text and not _is_caption_like(text):
                last_text_by_section[section_index] = text
    return signals


def compute_title_aware_anchors(
    images: list[JsonObject],
    narration: str,
    section_title: str | None,
    signals: list[FigureSignal],
    stats: AnchorStats,
) -> list[float]:
    """Compute image anchors from Upstage figure signals and runtime segmentation."""
    count = len(images)
    if count == 0:
        return []
    real_pins: dict[int, float] = {}
    used_signal_indices: set[int] = set()
    for image_index, image in enumerate(images):
        signal_index = _best_signal_index_for_image(
            image, signals, used_signal_indices, image_index
        )
        if signal_index is None:
            continue
        signal = signals[signal_index]
        used_signal_indices.add(signal_index)
        if not signal.preceding_text:
            continue
        ratio, confidence = _spoken_segment_anchor(
            signal.preceding_text,
            narration,
            section_title,
        )
        if confidence >= MIN_SEGMENT_CONFIDENCE:
            real_pins[image_index] = ratio

    if real_pins:
        anchors = resolve_scene_anchors(real_pins, count)
    else:
        anchors = resolve_scene_anchors({}, count)
        if count > 1:
            stats.scenes_even_fallback += 1

    stats.images_total += count
    stats.anchors_real += len(real_pins)
    stats.anchors_fallback += count - len(real_pins)
    return anchors


def validate_runtime_document(payload: JsonObject) -> None:
    """Validate the runtime history document schema subset used by history mode."""
    if payload.get("schema_version") != 2:
        raise ValueError("history document schema_version must be 2")
    _require_str(payload, "doc_hash")
    _require_str(payload, "source_file")
    _require_str(payload, "title")
    _require_str(payload, "kind")
    _require_str(payload, "era")
    _require_int(payload, "scene_count")
    _require_int(payload, "section_count")
    _require_int(payload, "image_count")
    raw_scenes = payload.get("scenes")
    if not isinstance(raw_scenes, list):
        raise ValueError("history document scenes must be a list")
    for fallback_index, raw_scene in enumerate(raw_scenes):
        _validate_runtime_scene(raw_scene, fallback_index)


def validate_anchor_invariants(payload: JsonObject) -> None:
    """Assert post-build anchor invariants for every runtime image list."""
    raw_scenes = payload.get("scenes")
    if not isinstance(raw_scenes, list):
        raise ValueError("history document scenes must be a list")
    for raw_scene in raw_scenes:
        if not isinstance(raw_scene, dict):
            raise ValueError("history scenes must be objects")
        raw_images = raw_scene.get("images", [])
        if not isinstance(raw_images, list):
            raise ValueError("history scene images must be a list")
        anchors: list[float] = []
        for raw_image in raw_images:
            if not isinstance(raw_image, dict):
                raise ValueError("history scene images must be objects")
            if "anchor_ratio" not in raw_image:
                raise ValueError("history image anchor_ratio is required after overlay")
            anchor = parse_anchor_ratio(raw_image.get("anchor_ratio"))
            if anchor is None:
                raise ValueError("history image anchor_ratio must be numeric after overlay")
            if anchor != round(anchor, 4):
                raise ValueError("history image anchor_ratio must be rounded to 4 decimals")
            anchors.append(anchor)
        if not anchors:
            continue
        if anchors[0] != 0.0:
            raise ValueError("history image anchors must start at 0.0")
        if len(anchors) == 1:
            if anchors[0] != 0.0:
                raise ValueError("single-image scene anchor must be 0.0")
            continue
        if anchors[-1] != 1.0:
            raise ValueError("multi-image scene anchors must end at 1.0")
        if any(left >= right for left, right in zip(anchors, anchors[1:], strict=False)):
            raise ValueError("history image anchors must be strictly increasing")


def validate_baseline_anchor_state(payload: JsonObject) -> None:
    """Validate baseline anchors with runtime restore semantics only."""
    raw_scenes = payload.get("scenes")
    if not isinstance(raw_scenes, list):
        raise ValueError("history document scenes must be a list")
    for raw_scene in raw_scenes:
        if not isinstance(raw_scene, dict):
            raise ValueError("history scenes must be objects")
        raw_images = raw_scene.get("images", [])
        if not isinstance(raw_images, list):
            raise ValueError("history scene images must be a list")
        for raw_image in raw_images:
            if not isinstance(raw_image, dict):
                raise ValueError("history scene images must be objects")
            if "anchor_ratio" not in raw_image:
                raise ValueError("baseline image anchor_ratio is required for restore")
            parse_anchor_ratio(raw_image.get("anchor_ratio"))


def load_json_object(path: Path) -> JsonObject:
    """Load a UTF-8 JSON object."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"JSON file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigError(f"Expected JSON object: {path}")
    return cast(JsonObject, payload)


def atomic_write_json(path: Path, payload: JsonObject) -> None:
    """Write JSON with a same-directory temporary file and atomic replace."""
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    try:
        tmp_path.write_text(text, encoding="utf-8")
        tmp_path.replace(path)
    except OSError as exc:
        raise PipelineError(f"Failed to write JSON atomically: {path}") from exc


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--only")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-api", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--dataset", type=Path, default=build_history_content.DEFAULT_DATASET)
    parser.add_argument("--out", type=Path, default=build_history_content.DEFAULT_OUT)
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--max-section-ms", type=int, default=MAX_SECTION_SPOKEN_MS)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        run_pipeline(args)
    except ConfigError as exc:
        logger.error("%s", exc)
        return CONFIG_EXIT
    except (PipelineError, UpstageApiError, OSError, ValueError) as exc:
        logger.error("%s", exc)
        return ERROR_EXIT
    return SUCCESS_EXIT


def run_pipeline(args: argparse.Namespace) -> None:
    """Run the full cache/API, overlay, rebuild, anchor, and manifest pipeline."""
    _validate_args(args)
    _validate_inputs(args.dataset, args.out, args.pdf_dir)
    baseline_docs = _load_runtime_docs(args.out / "docs")
    _validate_baseline_docs(baseline_docs)
    tasks = _select_tasks(
        dataset=args.dataset,
        pdf_dir=args.pdf_dir,
        only=args.only,
        limit=args.limit,
    )
    if not tasks:
        raise ConfigError("No matching history PDF tasks selected")
    cache_missing = [task for task in tasks if not _cache_path(args.cache_dir, task).exists()]
    if cache_missing and not args.dry_run and not args.skip_api:
        _load_upstage_api_key(required=True)

    if args.dry_run:
        logger.info("dry-run: selected=%d cache-missing=%d", len(tasks), len(cache_missing))
        return

    contexts: dict[str, DocumentContext] = {}
    totals = RunTotals()
    last_api_at: float | None = None
    api_key = _load_upstage_api_key(required=False)
    for task in tasks:
        cache_path = _cache_path(args.cache_dir, task)
        response_payload: JsonObject | None = None
        if cache_path.exists():
            try:
                response_payload = load_json_object(cache_path)
            except ConfigError as exc:
                logger.warning("Cache load failed for %s; falling back: %s", task.doc_hash, exc)
                if api_key and not args.skip_api:
                    last_api_at = _pace_api_calls(last_api_at)
                    try:
                        response_payload = call_upstage_sync(task.pdf_path, api_key)
                    except UpstageApiError as api_exc:
                        logger.warning(
                            "Upstage refetch failed for %s; falling back: %s",
                            task.doc_hash,
                            api_exc,
                        )
                    else:
                        atomic_write_json(cache_path, response_payload)
                        last_api_at = time.monotonic()
        elif args.skip_api:
            logger.warning("skip-api: missing cache for %s", task.doc_hash)
            totals.missing_cache += 1
            continue
        else:
            if not api_key:
                raise ConfigError(f"{UPSTAGE_API_KEY_ENV} environment variable is required")
            last_api_at = _pace_api_calls(last_api_at)
            try:
                response_payload = call_upstage_sync(task.pdf_path, api_key)
            except UpstageApiError as exc:
                logger.warning("Upstage failed for %s; falling back: %s", task.doc_hash, exc)
            else:
                atomic_write_json(cache_path, response_payload)
                last_api_at = time.monotonic()

        context = _process_one_task(
            task=task,
            response_payload=response_payload,
            dataset=args.dataset,
            baseline_document=baseline_docs.get(task.doc_hash),
            max_section_ms=args.max_section_ms,
        )
        contexts[task.doc_hash] = context
        _accumulate_context_totals(totals, context)

    if contexts:
        _rebuild_history_content(args.dataset, args.out)
        anchor_stats = annotate_runtime_docs(
            docs_dir=args.out / "docs",
            baseline_docs=baseline_docs,
            contexts=contexts,
        )
        totals.anchors_real = anchor_stats.anchors_real
        totals.anchors_fallback = anchor_stats.anchors_fallback
    else:
        totals.untouched += len(tasks)
        _log_totals(totals)
        return

    _write_manifest(args.cache_dir / "manifest.json", contexts, totals)
    _log_totals(totals)


def call_upstage_sync(pdf_path: Path, api_key: str) -> JsonObject:
    """Call the Upstage sync Document Parse endpoint with timeout and retry."""
    deadline_at = time.monotonic() + PER_DOC_DEADLINE_S
    body, content_type = _multipart_request_body(pdf_path)
    for attempt in range(len(RETRY_DELAYS_S) + 1):
        request = urllib.request.Request(
            UPSTAGE_SYNC_URL,
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": content_type,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_S) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, dict):
                raise UpstageApiError("Upstage response root must be an object")
            return cast(JsonObject, payload)
        except urllib.error.HTTPError as exc:
            if not _is_retryable_http_status(exc.code) or attempt >= len(RETRY_DELAYS_S):
                detail = _http_error_detail(exc)
                raise UpstageApiError(f"HTTP {exc.code} from Upstage: {detail}") from exc
            _sleep_before_retry(attempt, deadline_at)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if attempt >= len(RETRY_DELAYS_S):
                raise UpstageApiError(f"Upstage request failed: {exc}") from exc
            _sleep_before_retry(attempt, deadline_at)
        except json.JSONDecodeError as exc:
            raise UpstageApiError(f"Invalid JSON from Upstage: {exc}") from exc
    raise UpstageApiError("Unexpected Upstage retry exhaustion")


def annotate_runtime_docs(
    *,
    docs_dir: Path,
    baseline_docs: dict[str, JsonObject],
    contexts: dict[str, DocumentContext],
) -> AnchorStats:
    """Annotate rebuilt runtime docs and restore baseline anchors where required."""
    stats = AnchorStats()
    if not docs_dir.exists():
        raise ConfigError(f"Missing runtime docs directory: {docs_dir}")
    for doc_path in sorted(docs_dir.glob("*.json")):
        payload = load_json_object(doc_path)
        doc_hash = str(payload.get("doc_hash") or doc_path.stem)
        baseline = baseline_docs.get(doc_hash)
        context = contexts.get(doc_hash)
        must_restore = context is None or context.bucket != "upstage_ok"
        if baseline is None:
            raise PipelineError(f"Missing baseline runtime doc for {doc_hash}")
        try:
            validate_runtime_document(payload)
            if not must_restore and not _figure_signature_matches(baseline, payload):
                must_restore = True
                stats.docs_invariant_fallback += 1
            if context is not None and context.bucket == "upstage_fallback":
                _assert_baseline_sectioning_preserved(doc_hash, baseline, payload)
            if must_restore:
                _restore_baseline_anchors(payload, baseline)
                restored_count = _runtime_image_count(payload)
                stats.images_total += restored_count
                stats.anchors_fallback += restored_count
                if context is not None:
                    context.anchors_fallback += restored_count
                stats.docs_anchor_restored += 1
            elif context is not None:
                before_real = stats.anchors_real
                before_fallback = stats.anchors_fallback
                _annotate_one_runtime_doc(payload, context, stats)
                context.anchors_real += stats.anchors_real - before_real
                context.anchors_fallback += stats.anchors_fallback - before_fallback
            validate_runtime_document(payload)
            if must_restore:
                validate_baseline_anchor_state(payload)
            else:
                validate_anchor_invariants(payload)
        except ValueError as exc:
            if must_restore:
                raise
            _restore_baseline_anchors(payload, baseline)
            restored_count = _runtime_image_count(payload)
            stats.images_total += restored_count
            stats.anchors_fallback += restored_count
            if context is not None:
                context.anchors_fallback += restored_count
            stats.docs_invariant_fallback += 1
            stats.docs_anchor_restored += 1
            validate_runtime_document(payload)
            validate_baseline_anchor_state(payload)
            logger.warning(
                "Restored baseline anchors for %s after invariant failure: %s", doc_hash, exc
            )
        stats.docs += 1
        _write_doc_if_changed(doc_path, payload, stats)
    return stats


def _runtime_image_count(payload: JsonObject) -> int:
    raw_scenes = payload.get("scenes")
    if not isinstance(raw_scenes, list):
        return 0
    count = 0
    for raw_scene in raw_scenes:
        if not isinstance(raw_scene, dict):
            continue
        raw_images = raw_scene.get("images", [])
        if isinstance(raw_images, list):
            count += len(raw_images)
    return count


def _process_one_task(
    *,
    task: DocumentTask,
    response_payload: JsonObject | None,
    dataset: Path,
    baseline_document: JsonObject | None,
    max_section_ms: int,
) -> DocumentContext:
    document = load_json_object(task.scene_path)
    result = overlay_sections(
        document,
        response_payload,
        baseline_document=baseline_document,
        max_section_ms=max_section_ms,
    )
    elements = parse_elements(response_payload) if response_payload is not None else []
    signals = collect_figure_signals(elements, result.heading_texts)
    scene_signals = _assign_signals_to_scenes(result.scenes, signals)
    context = DocumentContext(
        task=task,
        bucket=result.bucket,
        heading_texts=result.heading_texts,
        scene_signals=scene_signals,
        section_count_before=result.section_count_before,
        section_count_after=result.section_count_after,
        headings_raw=result.headings_raw,
        headings_kept=result.headings_kept,
        model=_response_model(response_payload),
        reason=result.reason,
    )
    payload = {
        "doc_hash": task.doc_hash,
        "source_file": task.source_file,
        "scene_count": len(result.scenes),
        "scenes": result.scenes,
    }
    scene_path = dataset / "data" / "scenes" / f"{task.doc_hash}.json"
    atomic_write_json(scene_path, payload)
    return context


def _validate_args(args: argparse.Namespace) -> None:
    if args.limit is not None and args.limit < 1:
        raise ConfigError("--limit must be positive")
    if args.max_section_ms < 1:
        raise ConfigError("--max-section-ms must be positive")


def _validate_inputs(dataset: Path, out: Path, pdf_dir: Path) -> None:
    scenes_dir = dataset / "data" / "scenes"
    docs_dir = out / "docs"
    for path, label in ((dataset, "dataset"), (scenes_dir, "scene JSON directory")):
        if not path.exists():
            raise ConfigError(f"Missing {label}: {path}")
    if not docs_dir.exists():
        raise ConfigError(f"Missing runtime docs directory: {docs_dir}")
    if not pdf_dir.exists():
        raise ConfigError(f"Missing PDF directory: {pdf_dir}")


def _select_tasks(
    *,
    dataset: Path,
    pdf_dir: Path,
    only: str | None,
    limit: int | None,
) -> list[DocumentTask]:
    scene_docs = _scene_documents_by_source(dataset / "data" / "scenes")
    pdf_files = sorted(path for path in pdf_dir.glob("eh_*.pdf") if path.is_file())
    tasks: list[DocumentTask] = []
    for pdf_path in pdf_files:
        document_path = scene_docs.get(pdf_path.name)
        if document_path is None:
            continue
        document = load_json_object(document_path)
        doc_hash = _require_str(document, "doc_hash")
        if only is not None and doc_hash != only:
            continue
        tasks.append(
            DocumentTask(
                doc_hash=doc_hash,
                source_file=pdf_path.name,
                scene_path=document_path,
                pdf_path=pdf_path,
                pdf_sha256=sha256_file(pdf_path),
            )
        )
    if only is not None and not tasks:
        raise ConfigError(f"--only doc_hash not found among PDFs/scenes: {only}")
    if limit is not None and only is None:
        tasks = tasks[:limit]
    return tasks


def _scene_documents_by_source(scenes_dir: Path) -> dict[str, Path]:
    documents: dict[str, Path] = {}
    for path in sorted(scenes_dir.glob("*.json")):
        payload = load_json_object(path)
        source_file = _require_str(payload, "source_file")
        documents[source_file] = path
    return documents


def sha256_file(path: Path) -> str:
    """Return the SHA256 hex digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_path(cache_dir: Path, task: DocumentTask) -> Path:
    return cache_dir / f"{task.pdf_sha256}.json"


def _load_upstage_api_key(*, required: bool) -> str:
    api_key = os.getenv(UPSTAGE_API_KEY_ENV, "").strip()
    if required and not api_key:
        raise ConfigError(f"{UPSTAGE_API_KEY_ENV} environment variable is required")
    return api_key


def _pace_api_calls(last_api_at: float | None) -> float:
    now = time.monotonic()
    if last_api_at is None:
        return now
    delay = UPSTAGE_MIN_INTERVAL_S - (now - last_api_at)
    if delay > 0:
        time.sleep(delay)
    return time.monotonic()


def _load_runtime_docs(docs_dir: Path) -> dict[str, JsonObject]:
    docs: dict[str, JsonObject] = {}
    if not docs_dir.exists():
        raise ConfigError(f"Missing runtime docs directory: {docs_dir}")
    for path in sorted(docs_dir.glob("*.json")):
        payload = load_json_object(path)
        doc_hash = str(payload.get("doc_hash") or path.stem)
        docs[doc_hash] = payload
    if not docs:
        raise ConfigError(f"No runtime docs found under {docs_dir}")
    return docs


def _validate_baseline_docs(baseline_docs: dict[str, JsonObject]) -> None:
    for doc_hash, payload in sorted(baseline_docs.items()):
        try:
            validate_runtime_document(payload)
            validate_baseline_anchor_state(payload)
        except ValueError as exc:
            raise ValueError(f"Invalid baseline runtime doc {doc_hash}: {exc}") from exc


def _rebuild_history_content(dataset: Path, out: Path) -> None:
    build_history_content.build_history_content(
        build_history_content.BuildOptions(dataset=dataset, out=out, manifest_only=True)
    )


def _write_manifest(
    path: Path,
    contexts: dict[str, DocumentContext],
    totals: RunTotals,
) -> None:
    docs: list[JsonObject] = []
    for context in sorted(contexts.values(), key=lambda item: item.task.source_file):
        docs.append(
            {
                "doc_hash": context.task.doc_hash,
                "source_file": context.task.source_file,
                "pdf_sha256": context.task.pdf_sha256,
                "bucket": context.bucket,
                "reason": context.reason,
                "headings_found": context.headings_kept,
                "headings_raw": context.headings_raw,
                "section_count_before": context.section_count_before,
                "section_count_after": context.section_count_after,
                "anchors_real": context.anchors_real,
                "anchors_fallback": context.anchors_fallback,
                "model": context.model,
            }
        )
    payload = {
        "schema_version": 1,
        "upstage_endpoint": UPSTAGE_SYNC_URL,
        "upstage_model": UPSTAGE_MODEL,
        "docs": docs,
        "totals": {
            "upstage_ok": totals.upstage_ok,
            "upstage_fallback": totals.upstage_fallback,
            "untouched": totals.untouched,
            "missing_cache": totals.missing_cache,
            "headings_raw": totals.headings_raw,
            "headings_kept": totals.headings_kept,
            "anchors_real": totals.anchors_real,
            "anchors_fallback": totals.anchors_fallback,
        },
    }
    atomic_write_json(path, payload)


def _log_totals(totals: RunTotals) -> None:
    logger.info(
        "upstage_ok=%s upstage_fallback=%s untouched=%s missing_cache=%s",
        totals.upstage_ok,
        totals.upstage_fallback,
        totals.untouched,
        totals.missing_cache,
    )
    logger.info(
        "headings_raw=%s headings_kept=%s anchors_real=%s anchors_fallback=%s",
        totals.headings_raw,
        totals.headings_kept,
        totals.anchors_real,
        totals.anchors_fallback,
    )


def _accumulate_context_totals(totals: RunTotals, context: DocumentContext) -> None:
    if context.bucket == "upstage_ok":
        totals.upstage_ok += 1
    elif context.bucket == "upstage_fallback":
        totals.upstage_fallback += 1
    else:
        totals.untouched += 1
    totals.headings_raw += context.headings_raw
    totals.headings_kept += context.headings_kept


def _sentence_segments(text: str) -> list[str]:
    sentences = _split_text_into_sentences(text)
    if sentences:
        return sentences
    stripped = text.strip()
    return [stripped] if stripped else []


def _scene_list(document: JsonObject) -> list[JsonObject]:
    raw_scenes = document.get("scenes")
    if not isinstance(raw_scenes, list):
        raise PipelineError("Expected document scenes to be a list")
    scenes: list[JsonObject] = []
    for raw_scene in raw_scenes:
        if not isinstance(raw_scene, dict):
            raise PipelineError("Expected each scene to be an object")
        scenes.append(cast(JsonObject, raw_scene))
    return scenes


def _source_section_count(scenes: list[JsonObject], baseline_document: JsonObject | None) -> int:
    if baseline_document is None:
        return _baseline_section_count(scenes)
    return _runtime_section_count(baseline_document)


def _runtime_section_count(baseline_document: JsonObject) -> int:
    raw_sections = baseline_document.get("sections")
    if isinstance(raw_sections, list) and raw_sections:
        return len(raw_sections)
    return _baseline_section_count(_scene_list(baseline_document))


def _fallback_scenes(
    scenes: list[JsonObject], baseline_document: JsonObject | None = None
) -> list[JsonObject]:
    if baseline_document is not None:
        return _fallback_scenes_from_runtime_baseline(scenes, baseline_document)
    has_section_indices = all(isinstance(scene.get("section_index"), int) for scene in scenes)
    current_index = -1
    current_raw_index: int | None = None
    output: list[JsonObject] = []
    for scene in scenes:
        raw_title = _scene_section_title(scene)
        raw_index = int(scene["section_index"]) if has_section_indices else None
        starts_indexed_section = has_section_indices and raw_index != current_raw_index
        starts_legacy_section = not has_section_indices and raw_title is not None
        if current_index < 0 or starts_indexed_section or starts_legacy_section:
            current_index += 1
            current_raw_index = raw_index
        copied = _copy_scene_for_overlay(scene)
        copied["section_title"] = raw_title
        copied["section_index"] = current_index
        output.append(copied)
    return output


def _fallback_scenes_from_runtime_baseline(
    scenes: list[JsonObject], baseline_document: JsonObject
) -> list[JsonObject]:
    baseline_fields = _baseline_section_fields_by_seq(baseline_document)
    output: list[JsonObject] = []
    for scene in scenes:
        seq = scene.get("seq")
        if not isinstance(seq, int):
            raise PipelineError("Scene overlay requires integer seq for baseline fallback")
        baseline_field = baseline_fields.get(seq)
        if baseline_field is None:
            raise PipelineError(f"Missing baseline runtime scene for seq={seq}")
        copied = _copy_scene_for_overlay(scene)
        copied["section_index"] = baseline_field[0]
        copied["section_title"] = baseline_field[1]
        output.append(copied)
    return output


def _baseline_section_fields_by_seq(
    baseline_document: JsonObject,
) -> dict[int, tuple[int, str | None]]:
    raw_scenes = baseline_document.get("scenes")
    if not isinstance(raw_scenes, list):
        raise PipelineError("Baseline runtime document scenes must be a list")

    section_index_by_seq = _baseline_section_index_by_seq(baseline_document)
    fields_by_seq: dict[int, tuple[int, str | None]] = {}
    for raw_scene in raw_scenes:
        if not isinstance(raw_scene, dict):
            raise PipelineError("Baseline runtime scenes must be objects")
        seq = raw_scene.get("seq")
        if not isinstance(seq, int):
            raise PipelineError("Baseline runtime scene seq must be int")
        section_index = raw_scene.get("section_index")
        if not isinstance(section_index, int):
            raise PipelineError("Baseline runtime scene section_index must be int")
        section_title = raw_scene.get("section_title")
        if section_title is not None and not isinstance(section_title, str):
            raise PipelineError("Baseline runtime scene section_title must be text or null")
        canonical_index = section_index_by_seq.get(seq, section_index)
        if section_index_by_seq and section_index != canonical_index:
            raise PipelineError("Baseline runtime section grouping conflicts with scenes")
        if seq in fields_by_seq:
            raise PipelineError(f"Duplicate baseline runtime scene seq={seq}")
        fields_by_seq[seq] = (canonical_index, section_title)

    if section_index_by_seq and set(section_index_by_seq) != set(fields_by_seq):
        raise PipelineError("Baseline runtime section grouping does not match scenes")
    return fields_by_seq


def _baseline_section_index_by_seq(baseline_document: JsonObject) -> dict[int, int]:
    raw_sections = baseline_document.get("sections")
    if raw_sections is None:
        return {}
    if not isinstance(raw_sections, list):
        raise PipelineError("Baseline runtime document sections must be a list")

    section_index_by_seq: dict[int, int] = {}
    for raw_section in raw_sections:
        if not isinstance(raw_section, dict):
            raise PipelineError("Baseline runtime sections must be objects")
        section_index = raw_section.get("section_index")
        if not isinstance(section_index, int):
            raise PipelineError("Baseline runtime section_index must be int")
        raw_scene_seq = raw_section.get("scene_seq")
        if not isinstance(raw_scene_seq, list):
            raise PipelineError("Baseline runtime section scene_seq must be a list")
        for raw_seq in raw_scene_seq:
            if not isinstance(raw_seq, int):
                raise PipelineError("Baseline runtime section scene_seq values must be int")
            if raw_seq in section_index_by_seq:
                raise PipelineError(f"Duplicate baseline runtime section seq={raw_seq}")
            section_index_by_seq[raw_seq] = section_index
    return section_index_by_seq


def _copy_scene_for_overlay(scene: JsonObject) -> JsonObject:
    copied: JsonObject = {}
    for key in SCENE_COPY_FIELDS:
        if key in scene:
            copied[key] = scene[key]
    for key, value in scene.items():
        if key not in copied and key not in {"section_title", "section_index", "anchor_ratio"}:
            copied[key] = value
    return copied


def _baseline_section_count(scenes: list[JsonObject]) -> int:
    if not scenes:
        return 0
    if all(isinstance(scene.get("section_index"), int) for scene in scenes):
        count = 0
        current_raw_index: int | None = None
        for scene in scenes:
            raw_index = int(scene["section_index"])
            if count == 0 or raw_index != current_raw_index:
                count += 1
                current_raw_index = raw_index
        return count
    count = 0
    for index, scene in enumerate(scenes):
        if index == 0 or _scene_section_title(scene) is not None:
            count += 1
    return count


def _section_count_from_indices(scenes: list[JsonObject]) -> int:
    return len(
        {
            int(scene["section_index"])
            for scene in scenes
            if isinstance(scene.get("section_index"), int)
        }
    )


def _heading_section_cuts(headings: list[Heading], scenes: list[JsonObject]) -> list[SectionCut]:
    cuts_by_scene: dict[int, SectionCut] = {}
    for heading in headings:
        scene_index = _map_heading_to_scene(heading, scenes)
        cuts_by_scene.setdefault(
            scene_index,
            SectionCut(scene_index=scene_index, title=heading.text, from_heading=True),
        )
    if 0 not in cuts_by_scene:
        cuts_by_scene[0] = SectionCut(scene_index=0, title=None, from_heading=False)
    return [cuts_by_scene[index] for index in sorted(cuts_by_scene)]


def _map_heading_to_scene(heading: Heading, scenes: list[JsonObject]) -> int:
    heading_norm = _normalize(heading.text)
    for index, scene in enumerate(scenes):
        narration = scene.get("narration")
        if isinstance(narration, str) and heading_norm and heading_norm in _normalize(narration):
            return index
    fuzzy_index = _best_fuzzy_heading_scene(heading_norm, scenes)
    if fuzzy_index is not None:
        return fuzzy_index
    best_page_index = 0
    best_page = -1
    for index, scene in enumerate(scenes):
        page = scene.get("page")
        if isinstance(page, int) and page <= heading.page and page > best_page:
            best_page = page
            best_page_index = index
    return best_page_index


def _best_fuzzy_heading_scene(heading_norm: str, scenes: list[JsonObject]) -> int | None:
    if not heading_norm:
        return None
    best_index: int | None = None
    best_ratio = 0.0
    for index, scene in enumerate(scenes):
        narration = scene.get("narration")
        if not isinstance(narration, str):
            continue
        ratio = _heading_similarity(heading_norm, _normalize(narration))
        if ratio > best_ratio:
            best_ratio = ratio
            best_index = index
    if best_index is not None and best_ratio >= MIN_HEADING_FUZZY_RATIO:
        return best_index
    return None


def _heading_similarity(heading_norm: str, narration_norm: str) -> float:
    if not heading_norm or not narration_norm:
        return 0.0
    if len(narration_norm) <= len(heading_norm):
        return difflib.SequenceMatcher(None, heading_norm, narration_norm).ratio()
    window_size = len(heading_norm)
    return max(
        difflib.SequenceMatcher(
            None, heading_norm, narration_norm[start : start + window_size]
        ).ratio()
        for start in range(len(narration_norm) - window_size + 1)
    )


def _apply_section_cuts(
    scenes: list[JsonObject],
    cuts: list[SectionCut],
    *,
    max_section_ms: int,
) -> list[JsonObject]:
    cut_by_scene = {cut.scene_index: cut for cut in cuts}
    current_index = -1
    current_title: str | None = None
    output: list[JsonObject] = []
    section_start = 0
    section_ms = 0
    for scene_index, scene in enumerate(scenes):
        needs_new_section = scene_index in cut_by_scene or current_index < 0
        scene_ms = _scene_spoken_ms(scene)
        if not needs_new_section and section_ms + scene_ms > max_section_ms:
            needs_new_section = True
        if needs_new_section:
            current_index += 1
            section_start = scene_index
            section_ms = 0
            cut = cut_by_scene.get(scene_index)
            current_title = cut.title if cut is not None else None
        copied = _copy_scene_for_overlay(scene)
        copied["section_index"] = current_index
        copied["section_title"] = current_title if scene_index == section_start else None
        output.append(copied)
        section_ms += scene_ms
    return output


def _scene_spoken_ms(scene: JsonObject) -> int:
    value = scene.get("est_speech_ms")
    return value if isinstance(value, int) and value > 0 else 0


def _section_count_is_valid(
    section_count: int, scene_count: int, baseline_single_section: bool
) -> bool:
    if section_count < 1 or section_count > scene_count:
        return False
    if scene_count > 1 and section_count < 2 and not baseline_single_section:
        return False
    return True


def _assign_signals_to_scenes(
    scenes: list[JsonObject],
    signals: list[FigureSignal],
) -> dict[int, list[FigureSignal]]:
    signals_by_page: dict[int, list[FigureSignal]] = {}
    for signal in signals:
        signals_by_page.setdefault(signal.page, []).append(signal)
    for page_signals in signals_by_page.values():
        page_signals.sort(key=lambda signal: signal.order)

    assigned: dict[int, list[FigureSignal]] = {}
    page_offsets: dict[int, int] = {}
    for scene in scenes:
        seq = scene.get("seq")
        page = scene.get("page")
        image_paths = scene.get("image_paths")
        if (
            not isinstance(seq, int)
            or not isinstance(page, int)
            or not isinstance(image_paths, list)
        ):
            continue
        count = len(image_paths)
        if count == 0:
            continue
        available = signals_by_page.get(page, [])
        offset = page_offsets.get(page, 0)
        assigned[seq] = available[offset : offset + count]
        page_offsets[page] = offset + count
    return assigned


def _annotate_one_runtime_doc(
    payload: JsonObject,
    context: DocumentContext,
    stats: AnchorStats,
) -> None:
    raw_scenes = payload.get("scenes")
    if not isinstance(raw_scenes, list):
        raise ValueError("history document scenes must be a list")
    heading_set = set(context.heading_texts)
    for raw_scene in raw_scenes:
        if not isinstance(raw_scene, dict):
            raise ValueError("history scenes must be objects")
        section_title = raw_scene.get("section_title")
        if section_title is not None:
            if not isinstance(section_title, str):
                raise ValueError("history scene section_title must be text or null")
            if section_title and section_title not in heading_set:
                raise ValueError("upstage_ok section_title must be a kept heading")
        raw_images = raw_scene.get("images", [])
        if not isinstance(raw_images, list) or not raw_images:
            continue
        images = [cast(JsonObject, image) for image in raw_images if isinstance(image, dict)]
        narration = raw_scene.get("narration")
        seq = raw_scene.get("seq")
        signals = context.scene_signals.get(seq, []) if isinstance(seq, int) else []
        anchors = compute_title_aware_anchors(
            images,
            narration if isinstance(narration, str) else "",
            section_title if isinstance(section_title, str) else None,
            signals,
            stats,
        )
        for image, anchor in zip(images, anchors, strict=True):
            image["anchor_ratio"] = anchor


def _restore_baseline_anchors(payload: JsonObject, baseline: JsonObject) -> None:
    raw_scenes = payload.get("scenes")
    baseline_scenes = baseline.get("scenes")
    if not isinstance(raw_scenes, list) or not isinstance(baseline_scenes, list):
        raise ValueError("history document scenes must be lists for anchor restore")
    for raw_scene, baseline_scene in zip(raw_scenes, baseline_scenes, strict=False):
        if not isinstance(raw_scene, dict) or not isinstance(baseline_scene, dict):
            continue
        raw_images = raw_scene.get("images", [])
        baseline_images = baseline_scene.get("images", [])
        if not isinstance(raw_images, list) or not isinstance(baseline_images, list):
            continue
        for raw_image, baseline_image in zip(raw_images, baseline_images, strict=False):
            if not isinstance(raw_image, dict) or not isinstance(baseline_image, dict):
                continue
            if "anchor_ratio" in baseline_image:
                raw_image["anchor_ratio"] = baseline_image["anchor_ratio"]
            else:
                raise ValueError("baseline image anchor_ratio is required for restore")


def _figure_signature_matches(baseline: JsonObject, payload: JsonObject) -> bool:
    return _figure_signature(baseline) == _figure_signature(payload)


def _figure_signature(payload: JsonObject) -> tuple[tuple[tuple[str, str | None], ...], int]:
    images: list[tuple[str, str | None]] = []
    raw_scenes = payload.get("scenes")
    if not isinstance(raw_scenes, list):
        return ((), -1)
    for raw_scene in raw_scenes:
        if not isinstance(raw_scene, dict):
            return ((), -1)
        raw_images = raw_scene.get("images", [])
        if not isinstance(raw_images, list):
            return ((), -1)
        for raw_image in raw_images:
            if not isinstance(raw_image, dict):
                return ((), -1)
            path = raw_image.get("path")
            caption = raw_image.get("caption")
            path_name = PurePosixPath(path).name if isinstance(path, str) else ""
            images.append((path_name, caption if isinstance(caption, str) else None))
    image_count = payload.get("image_count")
    count = image_count if isinstance(image_count, int) else len(images)
    return (tuple(images), count)


def _write_doc_if_changed(
    path: Path,
    payload: JsonObject,
    stats: AnchorStats,
) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if path.read_text(encoding="utf-8") == text:
        stats.docs_unchanged += 1
        return
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)
    stats.docs_written += 1


def _best_signal_index_for_image(
    image: JsonObject,
    signals: list[FigureSignal],
    used_signal_indices: set[int],
    fallback_index: int,
) -> int | None:
    caption = image.get("caption")
    if isinstance(caption, str) and caption.strip():
        best_index: int | None = None
        best_ratio = 0.0
        for index, signal in enumerate(signals):
            if index in used_signal_indices or not signal.caption:
                continue
            ratio = caption_similarity(caption, signal.caption)
            if ratio > best_ratio:
                best_index = index
                best_ratio = ratio
        if best_index is not None and best_ratio >= MIN_CAPTION_RATIO:
            return best_index
    if fallback_index < len(signals) and fallback_index not in used_signal_indices:
        return fallback_index
    for index, signal in enumerate(signals):
        if index not in used_signal_indices and signal.preceding_text:
            return index
    return None


def _spoken_segment_anchor(
    snippet: str, narration: str, section_title: str | None
) -> tuple[float, float]:
    segments = [
        segment for segment in narration_segments(narration, section_title) if segment is not None
    ]
    if not segments:
        return 0.0, 0.0
    snippet_norm = _normalize(snippet)
    if not snippet_norm:
        return 0.0, 0.0
    best_index = 0
    best_confidence = 0.0
    for index, segment in enumerate(segments):
        segment_norm = _normalize(segment)
        if not segment_norm:
            continue
        matcher = difflib.SequenceMatcher(None, segment_norm, snippet_norm)
        match = matcher.find_longest_match(0, len(segment_norm), 0, len(snippet_norm))
        confidence = match.size / max(1, len(segment_norm))
        if confidence > best_confidence:
            best_confidence = confidence
            best_index = index
    denominator = max(1, len(segments) - 1)
    return (best_index / denominator, best_confidence)


def _caption_candidates(elements: list[UpstageElement]) -> list[UpstageElement]:
    return [
        element
        for element in elements
        if element.category in CAPTION_ELEMENT_CATEGORIES or _is_caption_like(element.text)
    ]


def _nearest_caption_text(
    figure: UpstageElement,
    candidates: list[UpstageElement],
) -> str | None:
    same_page = [candidate for candidate in candidates if candidate.page == figure.page]
    if not same_page:
        return None
    figure_center = figure.center
    if figure_center is None:
        nearest = min(same_page, key=lambda candidate: abs(candidate.order - figure.order))
    else:
        nearest = min(same_page, key=lambda candidate: _center_distance(figure_center, candidate))
    captions = split_caption_block(_clean_text(nearest.text))
    return captions[0] if captions else None


def _center_distance(center: tuple[float, float], element: UpstageElement) -> float:
    other = element.center
    if other is None:
        return 1.0
    return math.hypot(center[0] - other[0], center[1] - other[1])


def _multipart_request_body(pdf_path: Path) -> tuple[bytes, str]:
    boundary = f"mungi-upstage-{hashlib.sha256(pdf_path.name.encode()).hexdigest()[:16]}"
    parts: list[bytes] = []
    for name, value in (
        ("model", UPSTAGE_MODEL),
        ("output_formats", json.dumps(list(UPSTAGE_OUTPUT_FORMATS))),
        ("coordinates", "true"),
        ("ocr", "auto"),
        ("base64_encoding", json.dumps(list(UPSTAGE_BASE64_ENCODING))),
    ):
        parts.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                value.encode("utf-8"),
                b"\r\n",
            ]
        )
    pdf_bytes = pdf_path.read_bytes()
    parts.extend(
        [
            f"--{boundary}\r\n".encode(),
            (
                f'Content-Disposition: form-data; name="document"; filename="{pdf_path.name}"\r\n'
            ).encode(),
            b"Content-Type: application/pdf\r\n\r\n",
            pdf_bytes,
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )
    return (b"".join(parts), f"multipart/form-data; boundary={boundary}")


def _sleep_before_retry(attempt: int, deadline_at: float) -> None:
    delay_s = RETRY_DELAYS_S[attempt]
    if time.monotonic() + delay_s > deadline_at:
        raise UpstageApiError("Upstage per-doc deadline exceeded")
    logger.warning("Transient Upstage error; retrying in %.1fs", delay_s)
    time.sleep(delay_s)


def _is_retryable_http_status(status: int) -> bool:
    return status == 429 or 500 <= status <= 599


def _http_error_detail(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read(500).decode("utf-8", errors="replace")
    except OSError:
        body = ""
    return body.strip() or exc.reason


def _validate_runtime_scene(raw: Any, fallback_index: int) -> None:
    if not isinstance(raw, dict):
        raise ValueError("history scenes must be objects")
    _require_int(raw, "seq")
    _require_str(raw, "narration")
    _require_int(raw, "est_speech_ms")
    _require_int(raw, "tail_silence_ms")
    section_index_raw = raw.get("section_index", fallback_index)
    if not isinstance(section_index_raw, int):
        raise ValueError("history scene section_index must be int")
    section_title = raw.get("section_title")
    if section_title is not None and not isinstance(section_title, str):
        raise ValueError("history scene section_title must be text or null")
    captions = raw.get("image_captions", [])
    if not isinstance(captions, list) or not all(isinstance(item, str) for item in captions):
        raise ValueError("history scene image_captions must be list[str]")
    raw_images = raw.get("images", [])
    if not isinstance(raw_images, list):
        raise ValueError("history scene images must be a list")
    for raw_image in raw_images:
        _validate_runtime_image(raw_image)


def _validate_runtime_image(raw: Any) -> None:
    if not isinstance(raw, dict):
        raise ValueError("history scene images must be objects")
    _require_str(raw, "path")
    caption = raw.get("caption")
    if caption is not None and not isinstance(caption, str):
        raise ValueError("history image caption must be text or null")
    parse_anchor_ratio(raw.get("anchor_ratio"))


def _require_str(payload: JsonObject, key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f"Expected string field {key!r}")
    return value


def _require_int(payload: JsonObject, key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int):
        raise ValueError(f"Expected integer field {key!r}")
    return value


def _optional_int(value: Any, fallback: int) -> int:
    if isinstance(value, int):
        return value
    return fallback


def _parse_coordinates(value: Any) -> tuple[Point, ...]:
    if not isinstance(value, list):
        return ()
    points: list[Point] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        x_value = item.get("x")
        y_value = item.get("y")
        if isinstance(x_value, (int, float)) and isinstance(y_value, (int, float)):
            points.append(Point(x=float(x_value), y=float(y_value)))
    return tuple(points)


def _font_size_px(html: str) -> float | None:
    match = FONT_SIZE_RE.search(html)
    if match is None:
        return None
    return float(match.group("size"))


def _is_quote_wrapped(text: str) -> bool:
    stripped = text.strip()
    return bool(stripped) and stripped[0] in QUOTE_START_CHARS


def _is_sentence_final_korean_prose(text: str) -> bool:
    return bool(SENTENCE_FINAL_RE.search(_clean_text(text)))


def _is_caption_style_heading(text: str) -> bool:
    stripped = text.strip()
    return stripped.startswith(CAPTION_HEADING_START_CHARS)


def _is_caption_like(text: str) -> bool:
    stripped = text.strip()
    return len(stripped) >= 3 and (
        (stripped.startswith("<") and stripped.endswith(">"))
        or (stripped.startswith("〈") and stripped.endswith("〉"))
    )


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\n", " ")).strip()


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFC", text))


def _scene_section_title(scene: JsonObject) -> str | None:
    title = scene.get("section_title")
    if title is None:
        return None
    if not isinstance(title, str):
        return None
    stripped = title.strip()
    return stripped or None


def _sectioning_signature(payload: JsonObject) -> SectioningSignature:
    raw_scenes = payload.get("scenes")
    if not isinstance(raw_scenes, list):
        raise ValueError("history document scenes must be a list")
    scene_signature: list[SceneSectionSignature] = []
    for raw_scene in raw_scenes:
        if not isinstance(raw_scene, dict):
            raise ValueError("history scenes must be objects")
        seq = raw_scene.get("seq")
        if not isinstance(seq, int):
            raise ValueError("history scene seq must be int")
        section_index = raw_scene.get("section_index")
        if not isinstance(section_index, int):
            raise ValueError("history scene section_index must be int")
        section_title = raw_scene.get("section_title")
        if section_title is not None and not isinstance(section_title, str):
            raise ValueError("history scene section_title must be text or null")
        scene_signature.append((seq, section_index, section_title))
    return (tuple(scene_signature), _section_records_signature(payload))


def _section_records_signature(payload: JsonObject) -> tuple[SectionRecordSignature, ...]:
    raw_sections = payload.get("sections")
    if not isinstance(raw_sections, list):
        raise ValueError("history document sections must be a list")
    signature: list[SectionRecordSignature] = []
    for raw_section in raw_sections:
        if not isinstance(raw_section, dict):
            raise ValueError("history sections must be objects")
        section_index = raw_section.get("section_index")
        if not isinstance(section_index, int):
            raise ValueError("history section section_index must be int")
        section_title = raw_section.get("section_title")
        if section_title is not None and not isinstance(section_title, str):
            raise ValueError("history section section_title must be text or null")
        raw_scene_seq = raw_section.get("scene_seq")
        if not isinstance(raw_scene_seq, list):
            raise ValueError("history section scene_seq must be a list")
        scene_seq: list[int] = []
        for raw_seq in raw_scene_seq:
            if not isinstance(raw_seq, int):
                raise ValueError("history section scene_seq values must be int")
            scene_seq.append(raw_seq)
        signature.append((section_index, section_title, tuple(scene_seq)))
    return tuple(signature)


def _assert_baseline_sectioning_preserved(
    doc_hash: str, baseline: JsonObject, payload: JsonObject
) -> None:
    if _sectioning_signature(payload) != _sectioning_signature(baseline):
        raise PipelineError(f"Fallback sectioning changed for {doc_hash}")


def _response_model(response_payload: JsonObject | None) -> str | None:
    if response_payload is None:
        return None
    model = response_payload.get("model")
    return model if isinstance(model, str) else None


if __name__ == "__main__":
    sys.exit(main())
