"""Child-safety content filter for Mungi.

Filters LLM output before TTS to ensure child-safety compliance.
Supports keyword-based blocklist and regex pattern filtering with
configurable severity levels (BLOCK / REPLACE).

Design principles:
    - Biased toward safety: false positives acceptable, false negatives not.
    - Fully offline: no external API calls.
    - Lightweight: suitable for Jetson Orin Nano 8GB memory constraints.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default paths relative to project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_BLOCKLIST = _PROJECT_ROOT / "assets" / "filters" / "blocklist.json"
_DEFAULT_PATTERNS = _PROJECT_ROOT / "assets" / "filters" / "patterns.json"

SAFE_FALLBACK_RESPONSE = "음, 다른 이야기를 해볼까?"


class Severity(Enum):
    """Filter action severity level."""

    BLOCK = "BLOCK"
    REPLACE = "REPLACE"


@dataclass
class FilterResult:
    """Result of content filtering on a single text input.

    Attributes:
        allowed: True if the text passed all filters without BLOCK.
        original: The original input text.
        filtered: The filtered output text (safe fallback if blocked).
        violations: List of human-readable violation descriptions.
    """

    allowed: bool
    original: str
    filtered: str
    violations: list[str] = field(default_factory=list)


@dataclass
class _BlocklistCategory:
    """Internal representation of a single blocklist category."""

    name: str
    severity: Severity
    terms: list[str]


@dataclass
class _PatternRule:
    """Internal representation of a single regex pattern rule."""

    name: str
    severity: Severity
    pattern: re.Pattern[str]
    description: str


class ContentFilter:
    """Filters text for child-safety compliance.

    Loads blocklist keywords and regex patterns from JSON config files
    and applies them to LLM output before TTS synthesis.

    Args:
        blocklist_path: Path to blocklist JSON file.
        patterns_path: Path to patterns JSON file.
    """

    def __init__(
        self,
        blocklist_path: Path | str | None = None,
        patterns_path: Path | str | None = None,
    ) -> None:
        self._blocklist_path = Path(blocklist_path) if blocklist_path else _DEFAULT_BLOCKLIST
        self._patterns_path = Path(patterns_path) if patterns_path else _DEFAULT_PATTERNS
        self._categories: list[_BlocklistCategory] = []
        self._pattern_rules: list[_PatternRule] = []
        self._loaded = False

    @classmethod
    def from_default(cls) -> ContentFilter:
        """Build and eagerly load the production default content filter."""
        content_filter = cls()
        content_filter.load()
        return content_filter

    @property
    def category_count(self) -> int:
        """Return the number of loaded blocklist categories."""
        return len(self._categories)

    @property
    def pattern_count(self) -> int:
        """Return the number of loaded regex pattern rules."""
        return len(self._pattern_rules)

    def load(self) -> None:
        """Load filter configurations from JSON files.

        Raises:
            FileNotFoundError: If a config file does not exist.
            json.JSONDecodeError: If a config file is not valid JSON.
        """
        self._categories = self._load_blocklist(self._blocklist_path)
        self._pattern_rules = self._load_patterns(self._patterns_path)
        self._loaded = True
        logger.info(
            "Content filter loaded: %d categories, %d patterns",
            len(self._categories),
            len(self._pattern_rules),
        )

    def filter(self, text: str | None) -> FilterResult:
        """Filter text for child-safety violations.

        Applies blocklist keyword matching and regex pattern matching.
        If any BLOCK-severity violation is found, returns the safe
        fallback response. REPLACE-severity violations substitute
        matched content with safe placeholders.

        None or empty/whitespace-only input is treated as safe and
        returned immediately without loading filter configs.

        Args:
            text: The text to filter (typically LLM output).
                None and empty strings are treated as safe no-ops.

        Returns:
            FilterResult with filtering outcome and violation details.
        """
        if text is None:
            return FilterResult(
                allowed=True,
                original="",
                filtered="",
                violations=[],
            )

        if not text.strip():
            return FilterResult(
                allowed=True,
                original=text,
                filtered=text,
                violations=[],
            )

        if not self._loaded:
            self.load()

        violations: list[str] = []
        has_block = False
        filtered_text = text

        # Phase 1: Blocklist keyword check
        filtered_text, has_block = self._apply_blocklist(
            filtered_text,
            violations,
            has_block,
        )

        # Phase 2: Regex pattern check
        filtered_text, has_block = self._apply_patterns(
            filtered_text,
            violations,
            has_block,
        )

        if has_block:
            logger.warning(
                "Content BLOCKED: %d violations found",
                len(violations),
            )
            return FilterResult(
                allowed=False,
                original=text,
                filtered=SAFE_FALLBACK_RESPONSE,
                violations=violations,
            )

        if violations:
            logger.info(
                "Content REPLACED: %d violations found",
                len(violations),
            )

        return FilterResult(
            allowed=len(violations) == 0,
            original=text,
            filtered=filtered_text,
            violations=violations,
        )

    def _apply_blocklist(
        self,
        text: str,
        violations: list[str],
        has_block: bool,
    ) -> tuple[str, bool]:
        """Apply blocklist keyword filtering.

        Args:
            text: Current text being filtered.
            violations: Accumulator for violation descriptions.
            has_block: Whether a BLOCK violation has been seen.

        Returns:
            Tuple of (filtered text, updated has_block flag).
        """
        text_lower = text.lower()
        for category in self._categories:
            for term in category.terms:
                if term.lower() in text_lower:
                    violation_msg = f"blocklist:{category.name}:{category.severity.value}:'{term}'"
                    violations.append(violation_msg)
                    if category.severity == Severity.BLOCK:
                        has_block = True
                    else:
                        text = self._replace_term(text, term)
        return text, has_block

    def _apply_patterns(
        self,
        text: str,
        violations: list[str],
        has_block: bool,
    ) -> tuple[str, bool]:
        """Apply regex pattern filtering.

        Args:
            text: Current text being filtered.
            violations: Accumulator for violation descriptions.
            has_block: Whether a BLOCK violation has been seen.

        Returns:
            Tuple of (filtered text, updated has_block flag).
        """
        for rule in self._pattern_rules:
            match = rule.pattern.search(text)
            if match:
                violation_msg = f"pattern:{rule.name}:{rule.severity.value}:'{match.group()}'"
                violations.append(violation_msg)
                if rule.severity == Severity.BLOCK:
                    has_block = True
                else:
                    text = rule.pattern.sub("***", text)
        return text, has_block

    @staticmethod
    def _replace_term(text: str, term: str) -> str:
        """Replace a term in text with asterisks (case-insensitive).

        Args:
            text: The source text.
            term: The term to replace.

        Returns:
            Text with all occurrences of the term replaced.
        """
        escaped = re.escape(term)
        return re.sub(escaped, "***", text, flags=re.IGNORECASE)

    @staticmethod
    def _load_blocklist(path: Path) -> list[_BlocklistCategory]:
        """Load blocklist categories from a JSON file.

        Args:
            path: Path to the blocklist JSON file.

        Returns:
            List of parsed blocklist categories.
        """
        data = _read_json(path)
        categories: list[_BlocklistCategory] = []
        raw_categories: dict[str, Any] = data.get("categories", {})
        for name, info in raw_categories.items():
            severity = Severity(info["severity"])
            terms: list[str] = info.get("terms", [])
            categories.append(
                _BlocklistCategory(
                    name=name,
                    severity=severity,
                    terms=terms,
                ),
            )
            logger.debug(
                "Loaded blocklist category '%s': %d terms, severity=%s",
                name,
                len(terms),
                severity.value,
            )
        return categories

    @staticmethod
    def _load_patterns(path: Path) -> list[_PatternRule]:
        """Load regex pattern rules from a JSON file.

        Args:
            path: Path to the patterns JSON file.

        Returns:
            List of compiled pattern rules.
        """
        data = _read_json(path)
        rules: list[_PatternRule] = []
        for entry in data.get("patterns", []):
            compiled = re.compile(entry["pattern"])
            severity = Severity(entry["severity"])
            rules.append(
                _PatternRule(
                    name=entry["name"],
                    severity=severity,
                    pattern=compiled,
                    description=entry.get("description", ""),
                ),
            )
            logger.debug(
                "Loaded pattern rule '%s': severity=%s",
                entry["name"],
                severity.value,
            )
        return rules


def _read_json(path: Path) -> dict[str, Any]:
    """Read and parse a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON as a dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    logger.debug("Reading filter config: %s", path)
    with open(path, encoding="utf-8") as f:
        result: dict[str, Any] = json.load(f)
    return result
