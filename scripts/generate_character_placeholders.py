"""Generate placeholder character expression PNG assets."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any, NamedTuple

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame  # type: ignore[import-not-found,import-untyped]  # noqa: E402

logger = logging.getLogger("mungi.scripts.generate_character_placeholders")

CANVAS_SIZE = (720, 720)
BORDER_WIDTH = 24
FONT_SIZE = 96
BLACK = (0, 0, 0)


class PlaceholderSpec(NamedTuple):
    """Configuration for one generated placeholder image."""

    name: str
    rgb: tuple[int, int, int]


PLACEHOLDERS = (
    PlaceholderSpec("neutral", (128, 128, 128)),
    PlaceholderSpec("idle", (173, 216, 230)),
    PlaceholderSpec("listening", (255, 255, 153)),
    PlaceholderSpec("thinking", (221, 160, 221)),
    PlaceholderSpec("speaking", (144, 238, 144)),
    PlaceholderSpec("happy", (255, 182, 193)),
    PlaceholderSpec("sad", (100, 149, 237)),
    PlaceholderSpec("surprised", (255, 165, 0)),
    PlaceholderSpec("concerned", (210, 180, 140)),
    PlaceholderSpec("joyful", (255, 215, 0)),
    PlaceholderSpec("greeting", (0, 200, 100)),
    PlaceholderSpec("excited", (255, 105, 180)),
    PlaceholderSpec("angry", (220, 20, 60)),
    PlaceholderSpec("sulky", (60, 60, 100)),
    PlaceholderSpec("sleepy", (123, 104, 238)),
    PlaceholderSpec("tired", (169, 169, 169)),
    PlaceholderSpec("shy", (200, 100, 200)),
    PlaceholderSpec("winking", (240, 230, 140)),
    PlaceholderSpec("affectionate", (255, 20, 147)),
)


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for placeholder generation."""
    parser = argparse.ArgumentParser(
        description="Generate 720x720 character expression placeholder PNG assets.",
    )
    parser.add_argument(
        "--asset-dir",
        type=Path,
        default=Path("assets") / "character",
        help="Directory where placeholder PNG files are written.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing placeholder PNG files.",
    )
    return parser


def _render_placeholder(spec: PlaceholderSpec, font: Any) -> Any:
    """Render one placeholder image surface."""
    surface = pygame.Surface(CANVAS_SIZE, pygame.SRCALPHA)
    surface.fill((*spec.rgb, 255))
    pygame.draw.rect(surface, BLACK, surface.get_rect(), BORDER_WIDTH)

    label = spec.name.upper()
    text_surface = font.render(label, True, BLACK)
    text_rect = text_surface.get_rect(center=surface.get_rect().center)
    surface.blit(text_surface, text_rect)
    return surface


def generate_placeholders(asset_dir: Path, *, force: bool = False) -> list[Path]:
    """Generate placeholder PNG files and return the paths that were created."""
    asset_dir.mkdir(parents=True, exist_ok=True)
    created_paths: list[Path] = []

    pygame.init()
    try:
        font = pygame.font.SysFont(None, FONT_SIZE)
        for spec in PLACEHOLDERS:
            output_path = asset_dir / f"{spec.name}.png"
            if output_path.exists() and not force:
                logger.info("skipped %s", output_path)
                continue

            surface = _render_placeholder(spec, font)
            pygame.image.save(surface, str(output_path))
            logger.info("created %s", output_path)
            created_paths.append(output_path)
    finally:
        pygame.quit()

    return created_paths


def main() -> int:
    """Run placeholder generation from the command line."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args()
    generate_placeholders(args.asset_dir, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
