"""Generate the committed Korean-history UI font subset."""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Final

logger = logging.getLogger(__name__)

DEFAULT_INPUT_FONT: Final[Path] = Path(
    "assets/font/Pretendard-1.3.9/public/static/alternative/Pretendard-Bold.ttf"
)
DEFAULT_OUTPUT_FONT: Final[Path] = Path("assets/fonts/pretendard-history-kr.subset.ttf")
DEFAULT_MANIFEST: Final[Path] = Path("assets/history/manifest.json")
MODERN_HANGUL_SYLLABLES: Final[range] = range(0xAC00, 0xD7A4)
HANGUL_JAMO: Final[range] = range(0x1100, 0x1200)
HANGUL_COMPATIBILITY_JAMO: Final[range] = range(0x3130, 0x3190)
FIXED_HISTORY_UI_STRINGS: Final[tuple[str, ...]] = (
    "재미있는 우리역사",
    "시대",
    "인물",
    "유물",
    "이야기",
    "뒤로",
    "나가기",
    "전체 끝!",
    "목록이 비어 있어요.",
    "이미지를 준비 중이에요.",
    "다음 이야기 들려줄까? 화면을 톡 누르면 들려줄게.",
    "다음 이야기가 듣고 싶으면 화면을 톡 눌러줘.",
    "지금부터 '' 이야기를 들려줄게.",
    "좋아! 재미있는 우리역사를 시작할게!",
    "0123456789",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
    " .,!?;:()[]{}<>+-_/\\'\"·…—–’‘“”",
    "★☆",
)
DISPLAYABLE_HISTORY_PUNCTUATION: Final[frozenset[str]] = frozenset("·…—–’‘“”★☆")
_EMPTY_BRACKET_PATTERN: Final[re.Pattern[str]] = re.compile(r" *\(\s*\)| *\[\s*\]")
_MULTISPACE_PATTERN: Final[re.Pattern[str]] = re.compile(r" {2,}")


def displayable_history_text(s: str) -> str:
    """Return the renderable display copy for history UI text."""
    filtered = "".join(_displayable_char(char) for char in s)
    previous = None
    while previous != filtered:
        previous = filtered
        filtered = _EMPTY_BRACKET_PATTERN.sub("", filtered)
    filtered = _MULTISPACE_PATTERN.sub(" ", filtered)
    return filtered.strip()


def collect_history_font_text(manifest_path: Path = DEFAULT_MANIFEST) -> str:
    """Return all text whose glyphs must exist in the committed history font subset."""
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError("history manifest must contain a top-level object")
    chunks: list[str] = list(FIXED_HISTORY_UI_STRINGS)
    chunks.append(_require_str(manifest, "title"))
    era_order = manifest.get("era_order")
    if not isinstance(era_order, list) or not all(isinstance(item, str) for item in era_order):
        raise ValueError("history manifest era_order must be list[str]")
    chunks.extend(era_order)
    docs = manifest.get("docs")
    if not isinstance(docs, list):
        raise ValueError("history manifest docs must be a list")
    for raw_doc in docs:
        if not isinstance(raw_doc, dict):
            raise ValueError("history manifest docs entries must be objects")
        chunks.append(_require_str(raw_doc, "title"))
        doc_path = _resolve_doc_path(manifest_path, _require_str(raw_doc, "doc_path"))
        chunks.extend(_collect_document_display_text(doc_path))
    return "".join(displayable_history_text(chunk) for chunk in chunks)


def generate_history_font_subset(
    *,
    input_font: Path = DEFAULT_INPUT_FONT,
    output_font: Path = DEFAULT_OUTPUT_FONT,
    manifest_path: Path = DEFAULT_MANIFEST,
) -> int:
    """Generate the Korean-history UI font subset and return its glyph count."""
    from fontTools import subset  # type: ignore[import-untyped]
    from fontTools.ttLib import TTFont  # type: ignore[import-untyped]

    if not input_font.exists():
        raise FileNotFoundError(f"missing input font: {input_font}")
    required_text = collect_history_font_text(manifest_path)
    codepoints = sorted(
        {ord(char) for char in required_text}
        | set(MODERN_HANGUL_SYLLABLES)
        | set(HANGUL_JAMO)
        | set(HANGUL_COMPATIBILITY_JAMO)
    )

    options = subset.Options()
    options.name_IDs = ["*"]
    options.name_legacy = True
    options.name_languages = ["*"]
    options.layout_features = ["*"]
    font = TTFont(str(input_font))
    subsetter = subset.Subsetter(options=options)
    subsetter.populate(unicodes=codepoints)
    subsetter.subset(font)
    output_font.parent.mkdir(parents=True, exist_ok=True)
    font.save(str(output_font))
    logger.info(
        "Generated history font subset: output=%s glyph_codepoints=%d",
        output_font,
        len(codepoints),
    )
    return len(codepoints)


def _require_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"history manifest field {key!r} must be non-empty text")
    return value


def _displayable_char(char: str) -> str:
    codepoint = ord(char)
    if char.isspace():
        return " "
    if 0x20 <= codepoint <= 0x7E:
        return char
    if 0xAC00 <= codepoint <= 0xD7A3:
        return char
    if 0x3130 <= codepoint <= 0x318F:
        return char
    if char in DISPLAYABLE_HISTORY_PUNCTUATION:
        return char
    return ""


def _resolve_doc_path(manifest_path: Path, raw_path: str) -> Path:
    """Resolve a manifest doc path from either cwd or the manifest history root."""
    path = Path(raw_path)
    if path.is_absolute() or path.exists():
        return path
    if len(path.parts) >= 2 and path.parts[0] == "assets" and path.parts[1] == "history":
        repo_root = manifest_path.parent.parent.parent
        return repo_root / path
    return manifest_path.parent / path


def _collect_document_display_text(doc_path: Path) -> list[str]:
    """Collect text rendered from one runtime history document, excluding narration."""
    with doc_path.open(encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise ValueError("history document must contain a top-level object")
    scenes = document.get("scenes")
    if not isinstance(scenes, list):
        raise ValueError("history document scenes must be a list")
    chunks: list[str] = []
    for raw_scene in scenes:
        if not isinstance(raw_scene, dict):
            raise ValueError("history document scene entries must be objects")
        section_title = raw_scene.get("section_title")
        if section_title is not None:
            if not isinstance(section_title, str):
                raise ValueError("history scene section_title must be text or null")
            chunks.append(section_title)
        raw_captions = raw_scene.get("image_captions", [])
        if not isinstance(raw_captions, list) or not all(
            isinstance(caption, str) for caption in raw_captions
        ):
            raise ValueError("history scene image_captions must be list[str]")
        chunks.extend(raw_captions)
        raw_images = raw_scene.get("images", [])
        if not isinstance(raw_images, list):
            raise ValueError("history scene images must be a list")
        for raw_image in raw_images:
            if not isinstance(raw_image, dict):
                raise ValueError("history scene image entries must be objects")
            caption = raw_image.get("caption")
            if caption is None:
                continue
            if not isinstance(caption, str):
                raise ValueError("history image caption must be text or null")
            chunks.append(caption)
    return chunks


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the Korean-history UI font subset.")
    parser.add_argument("--input-font", type=Path, default=DEFAULT_INPUT_FONT)
    parser.add_argument("--output-font", type=Path, default=DEFAULT_OUTPUT_FONT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the font subset generator."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    args = _build_parser().parse_args(argv)
    generate_history_font_subset(
        input_font=args.input_font,
        output_font=args.output_font,
        manifest_path=args.manifest,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
