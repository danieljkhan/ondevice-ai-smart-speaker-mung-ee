"""Generate anti-aliased glyph badge PNGs saved as flag_ko.png and flag_en.png."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont  # type: ignore[import-not-found, import-untyped]

LOGGER = logging.getLogger("mungi.scripts.generate_language_badges")

IMAGE_SIZE = 48
SUPERSAMPLE = 4
CANVAS_SIZE = IMAGE_SIZE * SUPERSAMPLE
DISC_MARGIN = 2 * SUPERSAMPLE
RING_WIDTH = 3 * SUPERSAMPLE
DEFAULT_ASSET_DIR = Path("assets") / "character" / "indicator"
DEFAULT_FONT = Path("assets") / "fonts" / "mungi-badge-glyph.subset.ttf"
HALO = (0, 0, 0, 40)
WHITE = (255, 255, 255, 255)
KO_CORAL = (232, 80, 110, 255)
EN_BLUE = (62, 120, 200, 255)


@dataclass(frozen=True)
class BadgeSpec:
    """Configuration for one glyph language badge."""

    filename: str
    glyph: str
    fill: tuple[int, int, int, int]
    font_size_ratio: float
    inner_ring: bool


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for badge generation."""
    parser = argparse.ArgumentParser(
        description="Generate 48x48 anti-aliased KO/EN glyph language badges.",
    )
    parser.add_argument(
        "--asset-dir",
        type=Path,
        default=DEFAULT_ASSET_DIR,
        help=f"Directory where badge PNG files are written (default: {DEFAULT_ASSET_DIR}).",
    )
    parser.add_argument(
        "--font",
        type=Path,
        default=DEFAULT_FONT,
        help=f"Subset font path used for glyph rendering (default: {DEFAULT_FONT}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing badge PNG files.",
    )
    return parser


def _resampling_lanczos() -> int:
    return int(getattr(Image, "Resampling", Image).LANCZOS)


def _centered_text_position(
    glyph: str,
    font: ImageFont.FreeTypeFont,
) -> tuple[float, float]:
    bbox = font.getbbox(glyph)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    x = (CANVAS_SIZE - width) / 2.0 - bbox[0]
    y = (CANVAS_SIZE - height) / 2.0 - bbox[1]
    return x, y


def draw_badge(spec: BadgeSpec, font_path: Path) -> Image.Image:
    """Draw one supersampled glyph badge and return a 48x48 RGBA image."""
    image = Image.new("RGBA", (CANVAS_SIZE, CANVAS_SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    halo_bbox = (
        DISC_MARGIN - 1,
        DISC_MARGIN - 1,
        CANVAS_SIZE - DISC_MARGIN,
        CANVAS_SIZE - DISC_MARGIN,
    )
    disc_bbox = (
        DISC_MARGIN,
        DISC_MARGIN,
        CANVAS_SIZE - DISC_MARGIN,
        CANVAS_SIZE - DISC_MARGIN,
    )
    draw.ellipse(halo_bbox, fill=HALO)
    draw.ellipse(disc_bbox, fill=spec.fill)

    if spec.inner_ring:
        ring_bbox = (
            DISC_MARGIN + RING_WIDTH,
            DISC_MARGIN + RING_WIDTH,
            CANVAS_SIZE - DISC_MARGIN - RING_WIDTH,
            CANVAS_SIZE - DISC_MARGIN - RING_WIDTH,
        )
        draw.ellipse(ring_bbox, outline=WHITE, width=RING_WIDTH)

    font = ImageFont.truetype(str(font_path), int(round(spec.font_size_ratio * CANVAS_SIZE)))
    draw.text(_centered_text_position(spec.glyph, font), spec.glyph, font=font, fill=WHITE)
    return image.resize((IMAGE_SIZE, IMAGE_SIZE), _resampling_lanczos())


def generate_language_badges(
    asset_dir: Path,
    *,
    font_path: Path = DEFAULT_FONT,
    force: bool = False,
) -> list[Path]:
    """Generate glyph badge PNG files and return written paths."""
    if not font_path.exists():
        msg = f"Badge font not found: {font_path}"
        raise FileNotFoundError(msg)

    asset_dir.mkdir(parents=True, exist_ok=True)
    specs = (
        BadgeSpec("flag_ko.png", "한", KO_CORAL, 0.52, False),
        BadgeSpec("flag_en.png", "A", EN_BLUE, 0.60, True),
    )
    written: list[Path] = []
    for spec in specs:
        output_path = asset_dir / spec.filename
        if output_path.exists() and not force:
            LOGGER.info("skipped existing badge %s", output_path)
            continue
        badge = draw_badge(spec, font_path)
        badge.save(output_path)
        LOGGER.info("created %s", output_path)
        written.append(output_path)
    return written


def configure_logging() -> None:
    """Configure CLI logging to stdout for smoke-test capture."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the language badge generator CLI."""
    configure_logging()
    args = build_parser().parse_args(argv)
    try:
        generate_language_badges(
            Path(args.asset_dir),
            font_path=Path(args.font),
            force=bool(args.force),
        )
    except (OSError, ValueError) as exc:
        LOGGER.error("Failed to generate language badges: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
