"""Render CSS/HTML emoji animations into PNG frame sequences."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

FPS = 12
LOOP_MS = 10_000
FRAMES = FPS * LOOP_MS // 1_000
VIEWPORT_SIZE = 720
PAGE_SETTLE_MS = 100
FRAME_SETTLE_MS = 20

KOREAN_TO_EXPRESSION = {
    "자연스러움(기본)": "neutral",
    "궁금함": "listening",
    "걱정": "concerned",
    "놀람": "surprised",
    "말하기": "speaking",
    "반가움": "greeting",
    "부끄러움": "shy",
    "삐짐": "sulky",
    "사랑": "affectionate",
    "생각중": "thinking",
    "설레임": "excited",
    "슬픔": "sad",
    "윙크": "winking",
    "졸림": "sleepy",
    "즐거움": "joyful",
    "피곤": "tired",
    "행복함": "happy",
    "화남": "angry",
}

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the offline emoji frame renderer."""
    parser = argparse.ArgumentParser(
        description="Render assets/emoji/HTML/*.html emoji animations to PNG frame sequences.",
    )
    parser.add_argument(
        "--html-dir",
        type=Path,
        default=Path("assets/emoji/HTML"),
        help="Directory containing Korean-named HTML emoji files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("assets/character/frames"),
        help="Output directory for expression frame sequences.",
    )
    parser.add_argument("--fps", type=int, default=FPS, help="Frames per second to capture.")
    parser.add_argument(
        "--loop-ms",
        type=int,
        default=LOOP_MS,
        help="Animation loop duration in milliseconds.",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Optional single emoji stem or expression value to render.",
    )
    return parser.parse_args()


def frame_count(fps: int, loop_ms: int) -> int:
    """Return the number of frames for a loop duration and frame rate."""
    if fps <= 0:
        raise ValueError("--fps must be positive")
    if loop_ms <= 0:
        raise ValueError("--loop-ms must be positive")
    frames = fps * loop_ms // 1_000
    if frames <= 0:
        raise ValueError("--fps and --loop-ms must produce at least one frame")
    return frames


def should_render(stem: str, expression: str, filter_value: str | None) -> bool:
    """Return whether the current emoji matches an optional CLI filter."""
    if filter_value is None:
        return True
    return filter_value in {stem, expression}


def render_frames_for_html(
    page: Any,
    html_path: Path,
    output_dir: Path,
    *,
    frames: int = FRAMES,
    loop_ms: int = LOOP_MS,
) -> None:
    """Render one HTML emoji animation into a numbered PNG frame directory."""
    if frames <= 0:
        raise ValueError("frames must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    page.goto(html_path.resolve().as_uri(), wait_until="load")
    page.wait_for_timeout(PAGE_SETTLE_MS)

    for index in range(frames):
        current_time_ms = index / frames * loop_ms
        page.evaluate(
            "(T)=>{document.getAnimations().forEach(a=>{a.pause();a.currentTime=T;});}",
            current_time_ms,
        )
        page.wait_for_timeout(FRAME_SETTLE_MS)
        page.screenshot(path=str(output_dir / f"{index:03d}.png"))


def render_all(
    html_dir: Path,
    out_dir: Path,
    *,
    fps: int = FPS,
    loop_ms: int = LOOP_MS,
    filter_value: str | None = None,
) -> int:
    """Render all mapped HTML emoji files and return the rendered file count."""
    frames = frame_count(fps, loop_ms)
    html_files = sorted(html_dir.glob("*.html"))
    if not html_files:
        logger.error("No HTML files found in %s", html_dir)
        return 0

    render_queue: list[tuple[Path, str]] = []
    for html_path in html_files:
        expression = KOREAN_TO_EXPRESSION.get(html_path.stem)
        if expression is None:
            logger.warning("Skipping unmapped emoji HTML file: %s", html_path.name)
            continue
        if should_render(html_path.stem, expression, filter_value):
            render_queue.append((html_path, expression))

    if not render_queue:
        if filter_value is not None:
            logger.error("No mapped emoji matched filter %r", filter_value)
        return 0

    from playwright.sync_api import sync_playwright  # type: ignore[import-not-found,import-untyped]

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            for html_path, expression in render_queue:
                expression_dir = out_dir / expression
                logger.info(
                    "Rendering %s -> %s (%s frames at %s fps)",
                    html_path.name,
                    expression_dir,
                    frames,
                    fps,
                )
                page = browser.new_page(
                    viewport={"width": VIEWPORT_SIZE, "height": VIEWPORT_SIZE},
                )
                try:
                    render_frames_for_html(
                        page,
                        html_path,
                        expression_dir,
                        frames=frames,
                        loop_ms=loop_ms,
                    )
                finally:
                    page.close()
        finally:
            browser.close()

    return len(render_queue)


def main() -> int:
    """Run the offline emoji frame renderer CLI."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    try:
        rendered = render_all(
            args.html_dir,
            args.out_dir,
            fps=args.fps,
            loop_ms=args.loop_ms,
            filter_value=args.filter,
        )
    except ValueError as exc:
        logger.error("%s", exc)
        return 2
    return 0 if rendered else 1


if __name__ == "__main__":
    raise SystemExit(main())
