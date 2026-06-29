"""Fetch the official Korean-history chronological order index.

This explicit build-maintenance tool reads the five 우리역사넷 "재미있는
우리역사" age pages in chronological order and emits the committed
``assets/history/era_order_index.json`` schema consumed by
``scripts.history_era_order``.

The module performs network I/O only when its functions are explicitly called
or when run as a CLI. It is never imported by runtime code.
"""

from __future__ import annotations

import argparse
import html as html_lib
import json
import logging
import re
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger("mungi.scripts.history_era_order_fetch")

DEFAULT_OUT = Path("assets/history/era_order_index.json")
DEFAULT_TIMEOUT_S = 25.0
EXPECTED_COUNT = 240
USER_AGENT = "Mozilla/5.0"

AGES: list[tuple[str, str]] = [
    ("eh_age_10", "고대"),
    ("eh_age_20", "고려"),
    ("eh_age_30", "조선"),
    ("eh_age_40", "근대"),
    ("eh_age_50", "현대"),
]

PAIR_RE = re.compile(r"getContent\(['\"](eh_[nr]\d{4})['\"]\)\s*;.*?<dt>(.*?)</dt>", re.S)
TAG_RE = re.compile(r"<[^>]+>")

JsonObject = dict[str, Any]


def fetch_age_html(age_code: str, *, timeout: float = DEFAULT_TIMEOUT_S) -> str:
    """Fetch one official list page as UTF-8 text."""
    url = f"https://contents.history.go.kr/front/newEh/list.do?code={age_code}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data: bytes = response.read()
            return data.decode("utf-8", "replace")
    except (OSError, TimeoutError, urllib.error.URLError) as exc:
        msg = f"Failed to fetch {url}: {exc}"
        raise RuntimeError(msg) from exc


def build_index(
    fetcher: Callable[[str], str] = fetch_age_html,
) -> JsonObject:
    """Build the chronological order index from official list pages."""
    items: list[JsonObject] = []
    seen: set[str] = set()
    order = 0
    for age_code, age_name in AGES:
        html = fetcher(age_code)
        age_count = 0
        for level_id, raw_title in PAIR_RE.findall(html):
            if level_id in seen:
                logger.warning("Skipping duplicate level_id from %s: %s", age_code, level_id)
                continue
            seen.add(level_id)
            items.append(
                {
                    "order": order,
                    "level_id": level_id,
                    "source_file": f"{level_id}_0010.pdf",
                    "age_code": age_code,
                    "age_name": age_name,
                    "title": _clean_title(raw_title),
                }
            )
            order += 1
            age_count += 1
        logger.info("fetched %s items for %s", age_count, age_code)

    index: JsonObject = {
        "schema_version": 1,
        "source": (
            "contents.history.go.kr/front/newEh/list.do (eh_age_10..50) "
            "— official curated chronological order"
        ),
        "generated_note": (
            "Regenerate via scripts/history_era_order_fetch.py. Order is global 0-based site "
            "display sequence across the 5 age pages."
        ),
        "count": len(items),
        "items": items,
    }
    if len(items) != EXPECTED_COUNT:
        logger.warning("expected %s items but fetched %s", EXPECTED_COUNT, len(items))
    return index


def write_index(index: JsonObject, out_path: Path) -> None:
    """Write an order index JSON file using the committed index formatting."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(index, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    logger.info("wrote %s with %s items", _display_path(out_path), index.get("count"))


def _clean_title(raw_title: str) -> str:
    text = TAG_RE.sub(" ", raw_title)
    return re.sub(r"\s+", " ", html_lib.unescape(text)).strip()


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for regenerating the chronological order index."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    try:
        write_index(build_index(), args.out)
    except (OSError, RuntimeError, ValueError) as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
