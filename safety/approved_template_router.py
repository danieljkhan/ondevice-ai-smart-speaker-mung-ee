"""Approved safety template routing for sensitive child questions.

Supports two modes per topic:
- ``block``: bypass LLM entirely; return a fixed safety response (default).
- ``guide``: inject safety guidance into the LLM system prompt so the model
  can answer the child's curiosity while respecting safety constraints.

Each topic may also declare ``exclude_ko`` / ``exclude_en`` lists.  If any
exclude term appears in the user input the keyword match is suppressed,
preventing false positives like "꿀꿀" (pig onomatopoeia) triggering the
honey-safety template.
"""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Final, TypedDict

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_TEMPLATES_PATH = _PROJECT_ROOT / "assets" / "filters" / "approved_templates.json"
_FIXED_TTS_CACHE_TOPIC_IDS: Final[frozenset[str]] = frozenset(
    {
        "mungi_self_intro_child",
        "mungi_product_intro_adult",
    }
)

_EMOJI_RE = re.compile(
    "["
    "\U0001f300-\U0001f9ff"  # Misc Symbols, Emoticons, etc.
    "\U00002702-\U000027b0"  # Dingbats
    "\U0000fe00-\U0000fe0f"  # Variation Selectors
    "\U0000200d"  # ZWJ
    "\U000020e3"  # Combining Enclosing Keycap
    "\U00002600-\U000026ff"  # Misc Symbols
    "]+",
    flags=re.UNICODE,
)


class TemplateMatch(TypedDict):
    """Result of a successful safety-template match."""

    mode: str  # "block" or "guide"
    response: str  # The safety template text (emoji-stripped)
    topic_id: str  # e.g. "volcano", "honey_infant"


def strip_emoji(text: str) -> str:
    """Remove emoji / emoticon characters from *text*."""
    return _EMOJI_RE.sub("", text).strip()


@lru_cache(maxsize=1)
def _load_approved_templates() -> dict[str, dict[str, Any]]:
    """Load approved safety templates from disk once per process."""
    logger.debug("Loading approved templates from %s", _DEFAULT_TEMPLATES_PATH)
    with _DEFAULT_TEMPLATES_PATH.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("approved_templates.json must contain a top-level object")
    return data


def check_approved_template(
    user_input: str,
    language: str = "ko",
) -> TemplateMatch | None:
    """Check whether input matches a pre-approved safety template.

    Phase 2a collects every eligible template candidate, then resolves
    collisions by priority, longest matched keyword, distinct matched keyword
    count, and original JSON order.

    Args:
        user_input: The child's text input.
        language: Response language code ("ko" or "en"). Matching scans
            Korean and English template keywords on every turn.

    Returns:
        A :class:`TemplateMatch` dict when a topic matches, else ``None``.
    """
    if not user_input or not user_input.strip():
        return None

    normalized_language = "en" if language.lower() == "en" else "ko"
    normalized_input = user_input.casefold()
    templates = _load_approved_templates()
    response_field = f"response_{normalized_language}"

    # Plan 2026-04-28 Phase 2a: priority DESC -> longest match DESC ->
    # distinct count DESC -> JSON order ASC.
    candidates: list[tuple[str, int, int, int, str, str, int]] = []
    for original_index, (topic_id, topic) in enumerate(templates.items()):
        if topic_id == "dont_know":
            continue
        keywords = [
            *topic.get("keywords_ko", []),
            *topic.get("keywords_en", []),
        ]
        if not keywords:
            continue

        excludes = [
            *topic.get("exclude_ko", []),
            *topic.get("exclude_en", []),
        ]
        if any(ex.casefold() in normalized_input for ex in excludes):
            continue

        response = topic.get(response_field)
        if not isinstance(response, str):
            continue

        matched_keywords = {kw for kw in keywords if kw.casefold() in normalized_input}
        if not matched_keywords:
            continue

        priority = int(topic.get("priority", 0))
        max_keyword_len = max(len(kw) for kw in matched_keywords)
        distinct_match_count = len(matched_keywords)
        mode = topic.get("mode", "block")
        candidates.append(
            (
                topic_id,
                priority,
                max_keyword_len,
                distinct_match_count,
                mode,
                response,
                original_index,
            )
        )

    if not candidates:
        return None

    candidates.sort(key=lambda x: (-x[1], -x[2], -x[3], x[6]))
    best_topic_id, _, _, _, best_mode, best_response, _ = candidates[0]
    logger.info(
        "Approved template matched topic '%s' (mode=%s)",
        best_topic_id,
        best_mode,
    )
    return TemplateMatch(
        mode=best_mode,
        response=strip_emoji(best_response),
        topic_id=best_topic_id,
    )


def fixed_response_cache_texts() -> frozenset[str]:
    """Return fixed approved-template response texts eligible for TTS cache lookup."""
    templates = _load_approved_templates()
    texts: set[str] = set()
    for topic_id in _FIXED_TTS_CACHE_TOPIC_IDS:
        topic = templates.get(topic_id)
        if not isinstance(topic, dict):
            continue
        response = topic.get("response_ko")
        if isinstance(response, str) and response.strip():
            texts.add(strip_emoji(response))
    return frozenset(texts)
