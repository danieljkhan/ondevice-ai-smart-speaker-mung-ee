from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

GUIDE_PATH = "docs/templates/codex-task-utf8-korean.md"
MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024
SKIPPED_SUFFIXES = frozenset(
    {
        ".gguf",
        ".onnx",
        ".wav",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".pdf",
        ".zip",
        ".tar",
        ".bz2",
        ".xz",
        ".gz",
        ".npy",
        ".index",
    }
)
MOJIBAKE_PATTERNS = [
    "\u8438",
    "\ub431",
    "\u7d10\u317c",
    "\uc528",
    "\ube18",
    "\uc619",
    "\ucfbe",
    "\ub9a2",
    "\uc501",
]
DIRECT_MATCH_PATTERNS = frozenset({"\u8438", "\ub431", "\u7d10\u317c", "\ucfbe", "\ub9a2"})
CONTEXTUAL_MATCH_PATTERNS = frozenset({"\uc528", "\ube18", "\uc619", "\uc501"})


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect cp949 mojibake glyphs in UTF-8 source files.",
    )
    parser.add_argument(
        "--patterns-file",
        type=Path,
        help="Optional UTF-8 text file with one extra mojibake pattern per line.",
    )
    parser.add_argument("paths", nargs="*", type=Path, help="Files to scan.")
    return parser


def _load_extra_patterns(path: Path) -> list[str]:
    try:
        raw_lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"{path}: invalid UTF-8 in patterns file at byte offset {exc.start}."
        ) from exc

    patterns: list[str] = []
    for line in raw_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        patterns.append(stripped)
    return patterns


def _merged_patterns(extra_patterns: Sequence[str]) -> list[str]:
    return list(dict.fromkeys([*MOJIBAKE_PATTERNS, *extra_patterns]))


def _should_skip_file(path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix in SKIPPED_SUFFIXES:
        return True

    try:
        return path.stat().st_size > MAX_FILE_SIZE_BYTES
    except OSError:
        return False


def _iter_scan_lines(path: Path, text: str) -> Iterable[tuple[int, str]]:
    if path.suffix.lower() != ".md":
        for line_number, line in enumerate(text.splitlines(), start=1):
            yield line_number, line
        return

    in_fenced_block = False
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fenced_block = not in_fenced_block
            yield line_number, ""
            continue
        if in_fenced_block:
            yield line_number, ""
            continue
        if line.startswith("    ") or line.startswith("\t"):
            yield line_number, ""
            continue
        yield line_number, _remove_inline_code_spans(line)


def _remove_inline_code_spans(line: str) -> str:
    output: list[str] = []
    in_code_span = False

    for character in line:
        if character == "`":
            in_code_span = not in_code_span
            continue
        if not in_code_span:
            output.append(character)

    return "".join(output)


def _find_pattern_on_line(
    line: str,
    patterns: Sequence[str],
    extra_patterns: set[str],
) -> str | None:
    for pattern in patterns:
        if pattern not in line:
            continue
        if pattern in extra_patterns or pattern in DIRECT_MATCH_PATTERNS:
            return pattern
        if pattern in CONTEXTUAL_MATCH_PATTERNS and _has_contextual_marker(line):
            return pattern
    return None


def _has_contextual_marker(line: str) -> bool:
    if any(pattern in line for pattern in DIRECT_MATCH_PATTERNS):
        return True

    matches = [pattern for pattern in CONTEXTUAL_MATCH_PATTERNS if pattern in line]
    return len(matches) >= 2


def _scan_file(path: Path, patterns: Sequence[str], extra_patterns: set[str]) -> int:
    if not path.exists() or not path.is_file() or _should_skip_file(path):
        return 0

    data = path.read_bytes()
    try:
        decoded = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        sys.stderr.write(
            f"{path}: invalid UTF-8 at byte offset {exc.start}; "
            f"file must be UTF-8. See {GUIDE_PATH}.\n"
        )
        return 1

    for line_number, line in _iter_scan_lines(path, decoded):
        match = _find_pattern_on_line(line, patterns, extra_patterns)
        if match is None:
            continue
        sys.stderr.write(
            f"{path}:{line_number}: mojibake glyph '{match}' detected; "
            f"file likely written with cp949 encoding instead of UTF-8. "
            f"See {GUIDE_PATH}.\n"
        )
        return 1

    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the mojibake detector for the provided file paths."""

    parser = _build_parser()
    args = parser.parse_args(argv)

    extra_patterns: list[str] = []
    if args.patterns_file is not None:
        try:
            extra_patterns = _load_extra_patterns(args.patterns_file)
        except ValueError as exc:
            sys.stderr.write(f"{exc}\n")
            return 1

    patterns = _merged_patterns(extra_patterns)
    extra_pattern_set = set(extra_patterns)

    for path in args.paths:
        result = _scan_file(path, patterns, extra_pattern_set)
        if result != 0:
            return result

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
