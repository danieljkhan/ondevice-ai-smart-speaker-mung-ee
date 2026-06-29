"""Filter ingested Wikipedia articles into child-safe topic categories."""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, cast

LOGGER = logging.getLogger(__name__)

Language = Literal["en", "ko"]
DEFAULT_INPUT_DIR: Final[Path] = Path("assets") / "rag" / "raw material" / "_cache"
DEFAULT_OUTPUT_DIR: Final[Path] = DEFAULT_INPUT_DIR
DEFAULT_CONFIG_PATH: Final[Path] = Path("scripts") / "_phaseA_category_patterns.json"
EXPECTED_CATEGORIES: Final[tuple[str, ...]] = (
    "animal",
    "nature",
    "plant",
    "story",
    "science",
    "culture",
    "music",
    "sports",
    "vehicle",
    "weather",
    "body_health",
    "math",
    "world_geography",
    "world_history_light",
    "technology_intro",
    "science_intro_deeper",
    "arts_appreciation_intro",
)


@dataclass(frozen=True)
class TopicRule:
    """Compiled category rule for title matching."""

    category: str
    matched_topic: str
    pattern: re.Pattern[str]


@dataclass(frozen=True)
class CategoryConfig:
    """Compiled configuration for topic filtering."""

    category_order: tuple[str, ...]
    topic_rules: dict[str, tuple[TopicRule, ...]]
    exclude_patterns: dict[str, dict[str, tuple[re.Pattern[str], ...]]]
    history_whitelist: dict[str, re.Pattern[str]]
    violence_saturation_terms: dict[str, tuple[re.Pattern[str], ...]]
    violence_saturation_max_mentions: int


def configure_logging() -> None:
    """Configure default CLI logging."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--categories-config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args(argv)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    """Read JSONL rows from disk."""

    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object rows in {path}")
            rows.append(payload)
    return rows


def write_json(path: Path, payload: object) -> None:
    """Write a JSON payload with trailing newline."""

    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    """Write JSONL rows to disk."""

    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def compile_term(term: str, language: Language) -> re.Pattern[str]:
    """Compile one exclude term for the target language."""

    escaped = re.escape(term).replace(r"\ ", r"\s+")
    if language == "en":
        return re.compile(rf"\b{escaped}\b", re.IGNORECASE)
    return re.compile(escaped, re.IGNORECASE)


def load_category_config(path: Path) -> CategoryConfig:
    """Load and compile the category-filter configuration."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")

    raw_patterns = payload.get("category_patterns")
    raw_excludes = payload.get("exclude_terms")
    raw_translations = payload.get("cross_language_translations")
    raw_whitelist = payload.get("world_history_light_whitelist")
    raw_violence_terms = payload.get("violence_saturation_terms")
    if not all(
        isinstance(item, dict)
        for item in (
            raw_patterns,
            raw_excludes,
            raw_translations,
            raw_whitelist,
            raw_violence_terms,
        )
    ):
        raise ValueError("Category config payload is missing required object sections")
    raw_patterns = cast(dict[str, object], raw_patterns)
    raw_excludes = cast(dict[str, object], raw_excludes)
    raw_translations = cast(dict[str, object], raw_translations)
    raw_whitelist = cast(dict[str, object], raw_whitelist)
    raw_violence_terms = cast(dict[str, object], raw_violence_terms)

    en_patterns = cast(dict[str, object], raw_patterns["en"])
    category_order = tuple(category for category in EXPECTED_CATEGORIES if category in en_patterns)
    topic_rules: dict[str, tuple[TopicRule, ...]] = {}
    exclude_patterns: dict[str, dict[str, tuple[re.Pattern[str], ...]]] = {}
    history_whitelist: dict[str, re.Pattern[str]] = {}
    violence_saturation_terms: dict[str, tuple[re.Pattern[str], ...]] = {}

    for language in ("en", "ko"):
        language_rules: list[TopicRule] = []
        pattern_map = raw_patterns.get(language, {})
        if not isinstance(pattern_map, dict):
            raise ValueError("category_patterns language mappings must be objects")
        for category in EXPECTED_CATEGORIES:
            entries = pattern_map.get(category, [])
            if not isinstance(entries, list):
                raise ValueError("Each category entry must be a list")
            for entry in entries:
                if not isinstance(entry, dict):
                    raise ValueError("Each pattern entry must be an object")
                language_rules.append(
                    TopicRule(
                        category=category,
                        matched_topic=str(entry["topic"]),
                        pattern=re.compile(str(entry["pattern"]), re.IGNORECASE),
                    )
                )
        topic_rules[language] = tuple(language_rules)

        native = raw_excludes.get(language, {})
        translated = raw_translations.get(language, {})
        if not isinstance(native, dict) or not isinstance(translated, dict):
            raise ValueError("Exclude mappings must be objects")
        family_map: dict[str, tuple[re.Pattern[str], ...]] = {}
        for family in sorted(set(native) | set(translated)):
            terms: list[str] = []
            native_terms = native.get(family, [])
            translated_terms = translated.get(family, [])
            if isinstance(native_terms, list):
                terms.extend(str(term) for term in native_terms)
            if isinstance(translated_terms, list):
                terms.extend(str(term) for term in translated_terms)
            family_map[str(family)] = tuple(compile_term(term, language) for term in terms)
        exclude_patterns[language] = family_map

        whitelist_pattern = raw_whitelist.get(language)
        if not isinstance(whitelist_pattern, str):
            raise ValueError("world_history_light_whitelist values must be strings")
        history_whitelist[language] = re.compile(whitelist_pattern, re.IGNORECASE)

        violence_terms = raw_violence_terms.get(language)
        if not isinstance(violence_terms, list):
            raise ValueError("violence_saturation_terms values must be lists")
        violence_saturation_terms[language] = tuple(
            compile_term(str(term), language) for term in violence_terms
        )

    return CategoryConfig(
        category_order=category_order,
        topic_rules=topic_rules,
        exclude_patterns=exclude_patterns,
        history_whitelist=history_whitelist,
        violence_saturation_terms=violence_saturation_terms,
        violence_saturation_max_mentions=3,
    )


def match_topic(title: str, rules: tuple[TopicRule, ...]) -> TopicRule | None:
    """Return the first matching topic rule for the title."""

    normalized = " ".join(title.split())
    for rule in rules:
        if rule.pattern.search(normalized):
            return rule
    return None


def matched_exclude_family(
    *,
    title: str,
    text: str,
    patterns: dict[str, tuple[re.Pattern[str], ...]],
) -> str | None:
    """Return the first matched exclude family, if any."""

    haystack = f"{title}\n{text}"
    for family, family_patterns in patterns.items():
        if any(pattern.search(haystack) for pattern in family_patterns):
            return family
    return None


def count_mentions(text: str, patterns: tuple[re.Pattern[str], ...]) -> int:
    """Count all violence-term mentions in the article text."""

    return sum(len(pattern.findall(text)) for pattern in patterns)


def filter_language_rows(
    *,
    rows: list[dict[str, object]],
    language: Language,
    config: CategoryConfig,
    per_category_counts: Counter[str],
    per_exclude_family_rejections: Counter[str],
    other_rejections: Counter[str],
) -> list[dict[str, object]]:
    """Filter one language's ingested rows into category-tagged outputs."""

    kept: list[dict[str, object]] = []
    for row in rows:
        row_id = str(row.get("id", "")).strip()
        title = str(row.get("title", "")).strip()
        text = str(row.get("text", "")).strip()
        if not row_id or not title or not text:
            other_rejections["invalid_row"] += 1
            continue

        matched_rule = match_topic(title, config.topic_rules[language])
        if matched_rule is None:
            other_rejections["title_no_match"] += 1
            continue

        exclude_family = matched_exclude_family(
            title=title,
            text=text,
            patterns=config.exclude_patterns[language],
        )
        if exclude_family is not None and not (
            matched_rule.category == "world_history_light" and exclude_family == "violence"
        ):
            per_exclude_family_rejections[exclude_family] += 1
            continue

        if matched_rule.category == "world_history_light" and not config.history_whitelist[
            language
        ].search(text):
            other_rejections["world_history_whitelist"] += 1
            continue

        if matched_rule.category == "world_history_light":
            if count_mentions(text, config.violence_saturation_terms[language]) > (
                config.violence_saturation_max_mentions
            ):
                per_exclude_family_rejections["violence"] += 1
                continue

        kept.append(
            {
                "id": row_id,
                "title": title,
                "text": text,
                "category": matched_rule.category,
                "matched_topic": matched_rule.matched_topic,
                "language": language,
            }
        )
        per_category_counts[matched_rule.category] += 1
    return kept


def filter_topic_categories(
    *,
    input_dir: Path,
    output_dir: Path,
    config: CategoryConfig,
) -> dict[str, object]:
    """Filter EN/KO ingested JSONL files and write filtered outputs."""

    output_dir.mkdir(parents=True, exist_ok=True)
    per_category_counts: Counter[str] = Counter()
    per_exclude_family_rejections: Counter[str] = Counter()
    other_rejections: Counter[str] = Counter()
    kept_rows = 0

    for language in ("en", "ko"):
        input_path = input_dir / f"{language}_ingested.jsonl"
        output_path = output_dir / f"{language}_filtered.jsonl"
        rows = read_jsonl(input_path) if input_path.exists() else []
        filtered_rows = filter_language_rows(
            rows=rows,
            language=language,
            config=config,
            per_category_counts=per_category_counts,
            per_exclude_family_rejections=per_exclude_family_rejections,
            other_rejections=other_rejections,
        )
        kept_rows += len(filtered_rows)
        write_jsonl(output_path, filtered_rows)

    summary = {
        "kept_rows": kept_rows,
        "per_category_counts": dict(sorted(per_category_counts.items())),
        "per_exclude_family_rejections": dict(sorted(per_exclude_family_rejections.items())),
        "other_rejections": dict(sorted(other_rejections.items())),
    }
    write_json(output_dir / "filter_summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """Run the topic-category filter CLI."""

    configure_logging()
    args = parse_args(argv)
    config = load_category_config(args.categories_config)
    cache_dir = args.output_dir / "_cache"
    filter_topic_categories(input_dir=args.input_dir, output_dir=cache_dir, config=config)
    LOGGER.info("Wrote filtered outputs to %s", cache_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
