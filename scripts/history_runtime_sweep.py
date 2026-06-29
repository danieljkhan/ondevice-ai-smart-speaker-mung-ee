"""Sweep Korean-history runtime assets for parse, render, and optional TTS errors."""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

EXPECTED_DOCS = 240
EXPECTED_SCENES = 3606
EXPECTED_IMAGE_REFS = 1865
HISTORY_FONT_PATH = Path("assets/fonts/pretendard-history-kr.subset.ttf")
FIXED_TEXT_SAMPLES = (
    "재미있는 우리역사",
    "인물",
    "유물",
    "뒤로",
    "전체 끝!",
    "목록이 비어 있어요.",
    "이미지를 준비 중이에요.",
    "다음 이야기 들려줄까? 화면을 톡 누르면 들려줄게.",
    "다음 이야기가 듣고 싶으면 화면을 톡 눌러줘.",
    "좋아! 재미있는 우리역사를 시작할게!",
)


@dataclass
class SweepStats:
    """Counters emitted by the history runtime sweep."""

    docs: int = 0
    scenes: int = 0
    image_refs: int = 0
    missing_images: int = 0
    render_errors: int = 0
    text_render_errors: int = 0


def run_sweep(
    *,
    repo_root: Path = Path("."),
    require_images: bool = False,
    render_images: bool = True,
    tts_longest: int = 0,
    tts_per_era: bool = False,
) -> SweepStats:
    """Run the history runtime asset sweep and return counters."""
    manifest_path = repo_root / "assets" / "history" / "manifest.json"
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") not in (1, 2):
        raise ValueError("history manifest schema_version must be 1 or 2")
    docs = manifest.get("docs")
    if not isinstance(docs, list):
        raise ValueError("history manifest docs must be a list")

    surface: Any | None = None
    pygame_module: Any | None = None
    if render_images:
        try:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            import pygame as pygame_module_import  # type: ignore[import-not-found, import-untyped]
        except ImportError:
            logger.warning("Pygame unavailable; skipping history render validation")
        else:
            pygame_module = pygame_module_import
            pygame_module.init()
            pygame_module.font.init()
            pygame_module.display.set_mode((720, 720))
            surface = pygame_module.Surface((720, 720))

    stats = SweepStats(docs=len(docs))
    tts_candidates: list[tuple[int, str, str]] = []
    try:
        for raw_entry in docs:
            entry = _require_object(raw_entry, "manifest doc")
            era = _require_str(entry, "era")
            doc_path = repo_root / _require_str(entry, "doc_path")
            with doc_path.open(encoding="utf-8") as handle:
                doc = json.load(handle)
            scenes = doc.get("scenes")
            if not isinstance(scenes, list):
                raise ValueError(f"{doc_path} scenes must be a list")
            if len(scenes) != _require_int(entry, "scene_count"):
                raise ValueError(f"{doc_path} scene_count does not match manifest")
            stats.scenes += len(scenes)
            for raw_scene in scenes:
                scene = _require_object(raw_scene, f"{doc_path} scene")
                narration = _require_str(scene, "narration")
                if not narration.strip():
                    raise ValueError(f"{doc_path} has empty narration")
                tts_candidates.append((len(narration), era, narration))
                images = scene.get("images")
                if not isinstance(images, list):
                    raise ValueError(f"{doc_path} scene images must be a list")
                stats.image_refs += len(images)
                for raw_image in images:
                    image = _require_object(raw_image, f"{doc_path} image")
                    image_path = repo_root / _require_str(image, "path")
                    if not image_path.exists():
                        stats.missing_images += 1
                        if require_images:
                            raise FileNotFoundError(f"missing history image: {image_path}")
                        continue
                    if pygame_module is not None and surface is not None:
                        try:
                            _render_image(pygame_module, surface, image_path)
                        except Exception:
                            stats.render_errors += 1
                            logger.warning("Failed to render %s", image_path, exc_info=True)
        if pygame_module is not None and surface is not None:
            try:
                _render_text_samples(
                    pygame_module,
                    surface,
                    _collect_text_samples(manifest, docs),
                    repo_root / HISTORY_FONT_PATH,
                )
            except Exception:
                stats.text_render_errors += 1
                logger.warning("Failed to render history menu/text samples", exc_info=True)
        _validate_expected_counts(stats)
        if tts_longest > 0 or tts_per_era:
            _run_tts_samples(tts_candidates, longest=tts_longest, per_era=tts_per_era)
        return stats
    finally:
        if pygame_module is not None:
            pygame_module.quit()


def _render_image(pygame_module: Any, surface: Any, image_path: Path) -> None:
    image = pygame_module.image.load(str(image_path)).convert()
    source_w, source_h = image.get_size()
    scale = min(720 / source_w, 720 / source_h)
    target_size = (max(1, round(source_w * scale)), max(1, round(source_h * scale)))
    if image.get_size() != target_size:
        image = pygame_module.transform.smoothscale(image, target_size)
    surface.fill((0, 0, 0))
    surface.blit(image, ((720 - target_size[0]) // 2, (720 - target_size[1]) // 2))


def _collect_text_samples(manifest: dict[str, Any], docs: list[Any]) -> list[str]:
    """Return menu/text strings the runtime renderer must tolerate."""
    samples: list[str] = list(FIXED_TEXT_SAMPLES)
    samples.append(_require_str(manifest, "title"))
    era_order = manifest.get("era_order")
    if not isinstance(era_order, list) or not all(isinstance(item, str) for item in era_order):
        raise ValueError("history manifest era_order must be list[str]")
    samples.extend(era_order)
    for raw_entry in docs:
        entry = _require_object(raw_entry, "manifest doc")
        samples.append(_require_str(entry, "title"))
    return samples


def _render_text_samples(
    pygame_module: Any,
    surface: Any,
    samples: list[str],
    font_path: Path,
) -> None:
    """Render each menu/text sample on a headless surface."""
    font = pygame_module.font.Font(str(font_path), 34)
    surface.fill((16, 18, 22))
    for sample in samples:
        text = sample.strip()
        if not text:
            continue
        rendered = font.render(text, True, (244, 246, 248))
        if rendered.get_width() > 680:
            scale = 680 / rendered.get_width()
            target_size = (680, max(1, round(rendered.get_height() * scale)))
            rendered = pygame_module.transform.smoothscale(rendered, target_size)
        surface.blit(rendered, (20, 20))


def _validate_expected_counts(stats: SweepStats) -> None:
    if stats.docs != EXPECTED_DOCS:
        raise ValueError(f"expected {EXPECTED_DOCS} docs, found {stats.docs}")
    if stats.scenes != EXPECTED_SCENES:
        raise ValueError(f"expected {EXPECTED_SCENES} scenes, found {stats.scenes}")
    if stats.image_refs != EXPECTED_IMAGE_REFS:
        raise ValueError(f"expected {EXPECTED_IMAGE_REFS} image refs, found {stats.image_refs}")
    if stats.render_errors:
        raise ValueError(f"history image render errors: {stats.render_errors}")
    if stats.text_render_errors:
        raise ValueError(f"history text render errors: {stats.text_render_errors}")


def _run_tts_samples(
    candidates: list[tuple[int, str, str]],
    *,
    longest: int,
    per_era: bool,
) -> None:
    """Synthesize selected narration samples without playback."""
    from core.model_manager import ManagerConfig, ModelManager, ModelType
    from models.tts_runner import _split_text_into_sentences

    selected: list[tuple[int, str, str]] = sorted(candidates, reverse=True)[:longest]
    if per_era:
        seen_eras: set[str] = set()
        for candidate in sorted(candidates, reverse=True):
            _length, era, _narration = candidate
            if era in seen_eras:
                continue
            selected.append(candidate)
            seen_eras.add(era)
    if not selected:
        return

    manager = ModelManager(ManagerConfig())
    manager.initialize()
    manager.load(ModelType.TTS)
    try:
        tts = manager.tts
        if tts is None:
            raise RuntimeError("TTS did not load for history sweep")
        for _length, era, narration in selected:
            logger.info("tts_sample era=%s chars=%d", era, len(narration))
            for sentence in _split_text_into_sentences(narration):
                tts.synthesize(sentence, language="ko")
    finally:
        manager.unload_tts(force=True)


def _require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _require_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"history payload field {key!r} must be non-empty text")
    return value


def _require_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int):
        raise ValueError(f"history payload field {key!r} must be int")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep Korean-history runtime assets.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--require-images", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--tts-longest", type=int, default=0)
    parser.add_argument("--tts-per-era", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the runtime sweep."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _build_parser().parse_args(argv)
    stats = run_sweep(
        repo_root=args.repo_root,
        require_images=args.require_images,
        render_images=not args.no_render,
        tts_longest=max(0, args.tts_longest),
        tts_per_era=args.tts_per_era,
    )
    logger.info(
        "history_sweep docs=%d scenes=%d image_refs=%d missing_images=%d",
        stats.docs,
        stats.scenes,
        stats.image_refs,
        stats.missing_images,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
