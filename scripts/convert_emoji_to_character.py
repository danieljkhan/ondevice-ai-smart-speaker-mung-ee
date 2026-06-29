"""One-shot converter: assets/emoji/*.png (Korean) -> assets/character/*.png (English).

Pillow pipeline: open -> verify -> convert("RGBA") -> resize((720,720), LANCZOS)
-> optimize -> save. Files above the waiver limit get a deterministic 7-bit RGB
precision pass while preserving final 720x720 RGBA output.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
import unicodedata
from pathlib import Path

from PIL import Image  # type: ignore[import-not-found, import-untyped]

logger = logging.getLogger("mungi.scripts.convert_emoji_to_character")

CANVAS_SIZE = (720, 720)
PER_FILE_SIZE_TARGET_BYTES = 200 * 1024
PER_FILE_SIZE_WAIVER_BYTES = 300 * 1024

CONVERSIONS: tuple[tuple[str, str], ...] = (
    ("자연스러움(기본).png", "neutral.png"),
    ("편안함.png", "idle.png"),
    ("궁금함.png", "listening.png"),
    ("생각중.png", "thinking.png"),
    ("말하기.png", "speaking.png"),
    ("행복함.png", "happy.png"),
    ("슬픔.png", "sad.png"),
    ("놀람.png", "surprised.png"),
    ("걱정.png", "concerned.png"),
    ("즐거움.png", "joyful.png"),
    ("반가움.png", "greeting.png"),
    ("설레임.png", "excited.png"),
    ("화남.png", "angry.png"),
    ("삐짐.png", "sulky.png"),
    ("졸림.png", "sleepy.png"),
    ("피곤.png", "tired.png"),
    ("부끄러움.png", "shy.png"),
    ("윙크.png", "winking.png"),
    ("사랑.png", "affectionate.png"),
)


def _nfc(name: str) -> str:
    return unicodedata.normalize("NFC", name)


def _save_png(image: Image.Image, dst: Path) -> int:
    """Save one optimized PNG and return its byte size."""
    image.save(dst, "PNG", optimize=True, compress_level=9)
    return dst.stat().st_size


def _reduce_rgb_precision(image: Image.Image) -> Image.Image:
    """Reduce RGB least-significant-bit noise while preserving RGBA mode."""
    red, green, blue, alpha = image.split()
    mask_table = [value & 0xFE for value in range(256)]
    return Image.merge(
        "RGBA",
        (
            red.point(mask_table),
            green.point(mask_table),
            blue.point(mask_table),
            alpha,
        ),
    )


def _save_with_size_policy(image: Image.Image, dst: Path) -> int:
    """Save image, applying a deterministic precision reduction only if needed."""
    size = _save_png(image, dst)
    if size <= PER_FILE_SIZE_WAIVER_BYTES:
        return size

    reduced = _reduce_rgb_precision(image)
    return _save_png(reduced, dst)


def preflight(src_dir: Path) -> dict[str, Path]:
    """Verify count == 19 and all 19 Korean source files exist."""
    if not src_dir.is_dir():
        raise SystemExit(f"PREFLIGHT FAIL: source dir missing: {src_dir}")

    sources = {_nfc(path.name): path for path in src_dir.glob("*.png")}
    if len(sources) != 19:
        raise SystemExit(
            f"PREFLIGHT FAIL: expected 19 PNG sources, found {len(sources)} in {src_dir}"
        )

    missing = [src_name for src_name, _ in CONVERSIONS if _nfc(src_name) not in sources]
    if missing:
        raise SystemExit(f"PREFLIGHT FAIL: missing {len(missing)} expected source(s): {missing}")

    return sources


def convert_one(src: Path, dst: Path) -> int:
    """Convert one PNG and return final file size in bytes."""
    with Image.open(src) as probe:
        probe.verify()

    with Image.open(src) as opened:
        # Widen from ImageFile to Image.Image so convert()/resize() reassignments type-check.
        image: Image.Image = opened
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        if image.size != CANVAS_SIZE:
            image = image.resize(CANVAS_SIZE, Image.Resampling.LANCZOS)
        dst.parent.mkdir(parents=True, exist_ok=True)
        return _save_with_size_policy(image, dst)

    raise RuntimeError(f"unreachable conversion path for {src}")


def atomic_convert_all(src_dir: Path, dst_dir: Path) -> None:
    """Convert all 19 PNGs in a temp dir, then atomically replace outputs."""
    sources = preflight(src_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=dst_dir.parent, prefix="mungi-emoji-convert-") as tmp_str:
        tmp_dir = Path(tmp_str)
        for src_name, dst_name in CONVERSIONS:
            src = sources[_nfc(src_name)]
            tmp_dst = tmp_dir / dst_name
            size = convert_one(src, tmp_dst)
            logger.info("converted %s -> %s (%d bytes, temp)", src.name, dst_name, size)

        postflight(tmp_dir)
        for _, dst_name in CONVERSIONS:
            os.replace(tmp_dir / dst_name, dst_dir / dst_name)

        logger.info("atomic-replace complete: %d files -> %s", len(CONVERSIONS), dst_dir)


def postflight(dst_dir: Path) -> None:
    """Verify 19 output files exist and each satisfies the per-file size policy."""
    outputs = sorted(dst_dir.glob("*.png"))
    if len(outputs) != 19:
        raise SystemExit(
            f"POSTFLIGHT FAIL: expected 19 PNG outputs, found {len(outputs)} in {dst_dir}"
        )

    waivers: list[str] = []
    fails: list[str] = []
    for output in outputs:
        with Image.open(output) as probe:
            probe.verify()
        size = output.stat().st_size
        if size > PER_FILE_SIZE_WAIVER_BYTES:
            fails.append(f"{output.name} {size} bytes > {PER_FILE_SIZE_WAIVER_BYTES} waiver")
        elif size > PER_FILE_SIZE_TARGET_BYTES:
            waivers.append(f"{output.name} {size} bytes (waiver invoked)")

    if fails:
        raise SystemExit("POSTFLIGHT FAIL:\n  " + "\n  ".join(fails))
    if waivers:
        logger.warning(
            "Size waiver invoked (README section 9 target <200KB):\n  %s",
            "\n  ".join(waivers),
        )


def main() -> int:
    """Run the character asset converter from the command line."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-dir", type=Path, default=Path("assets/emoji"))
    parser.add_argument("--dst-dir", type=Path, default=Path("assets/character"))
    args = parser.parse_args()

    atomic_convert_all(args.src_dir, args.dst_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
