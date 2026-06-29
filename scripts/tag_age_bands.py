"""Tag filtered Wikipedia articles with heuristic age-band hints."""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Final, Literal, cast

LOGGER = logging.getLogger(__name__)

AgeBand = Literal["preschool", "elementary", "middle_school"]

DEFAULT_INPUT_DIR: Final[Path] = Path("assets") / "rag" / "raw material" / "_cache"
DEFAULT_OUTPUT_DIR: Final[Path] = DEFAULT_INPUT_DIR
DEFAULT_OVERRIDES_FILE: Final[Path] = DEFAULT_INPUT_DIR / "age_band_overrides.json"
INPUT_SOURCES: Final[tuple[str, ...]] = ("en", "ko")
EN_WORD_RE: Final[re.Pattern[str]] = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
KO_EOJEOL_RE: Final[re.Pattern[str]] = re.compile(r"\S+")
SENTENCE_SPLIT_RE: Final[re.Pattern[str]] = re.compile(r"(?<=[.!?。！？])\s+|\n+")
HANJA_RE: Final[re.Pattern[str]] = re.compile(r"[\u4e00-\u9fff]")
ALLOWED_AGE_BANDS: Final[tuple[AgeBand, ...]] = ("preschool", "elementary", "middle_school")
COMMON_ENGLISH_WORDS: Final[frozenset[str]] = frozenset(
    {
        "a",
        "about",
        "after",
        "all",
        "also",
        "an",
        "and",
        "animal",
        "are",
        "around",
        "as",
        "at",
        "be",
        "because",
        "body",
        "bright",
        "by",
        "can",
        "cat",
        "children",
        "day",
        "different",
        "do",
        "dog",
        "earth",
        "each",
        "easy",
        "for",
        "from",
        "game",
        "good",
        "grow",
        "has",
        "have",
        "help",
        "how",
        "in",
        "into",
        "is",
        "it",
        "its",
        "learn",
        "like",
        "many",
        "moon",
        "more",
        "most",
        "move",
        "music",
        "nature",
        "new",
        "of",
        "on",
        "one",
        "people",
        "pet",
        "pets",
        "place",
        "plant",
        "play",
        "rain",
        "round",
        "science",
        "school",
        "see",
        "shape",
        "simple",
        "sky",
        "small",
        "sound",
        "space",
        "sport",
        "star",
        "story",
        "sun",
        "team",
        "that",
        "the",
        "their",
        "them",
        "there",
        "they",
        "thing",
        "this",
        "to",
        "use",
        "very",
        "water",
        "way",
        "we",
        "what",
        "when",
        "where",
        "which",
        "with",
        "word",
        "work",
        "world",
        "you",
    }
)


class AgeBandHeuristics:
    """Tunable readability thresholds for the Phase A.0 age-band heuristic."""

    PRESCHOOL_MAX_TOKENS: Final[int] = 200
    PRESCHOOL_MAX_SENTENCE_TOKENS: Final[float] = 12.0
    PRESCHOOL_MAX_COMPLEXITY: Final[float] = 0.05
    ELEMENTARY_MAX_TOKENS: Final[int] = 500
    ELEMENTARY_MAX_SENTENCE_TOKENS: Final[float] = 20.0
    ELEMENTARY_MAX_COMPLEXITY: Final[float] = 0.15


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
    parser.add_argument("--overrides-file", type=Path, default=DEFAULT_OVERRIDES_FILE)
    return parser.parse_args(argv)


def validate_age_band(value: object) -> AgeBand:
    """Validate one age-band string."""

    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in ALLOWED_AGE_BANDS:
            return cast(AgeBand, normalized)
    raise ValueError(f"Unsupported age band: {value}")


def load_overrides(path: Path) -> dict[str, AgeBand]:
    """Load optional per-article age-band overrides."""

    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return {str(key): validate_age_band(value) for key, value in payload.items()}


def en_word_count(text: str) -> int:
    """Count English word tokens."""

    return len(EN_WORD_RE.findall(text))


def en_average_sentence_length(text: str) -> float:
    """Compute average English sentence length in words."""

    sentences = [sentence for sentence in SENTENCE_SPLIT_RE.split(text) if sentence.strip()]
    if not sentences:
        return float(en_word_count(text))
    sentence_lengths = [
        len(EN_WORD_RE.findall(sentence)) for sentence in sentences if sentence.strip()
    ]
    if not sentence_lengths:
        return 0.0
    return sum(sentence_lengths) / len(sentence_lengths)


def en_rare_word_ratio(text: str) -> float:
    """Estimate EN vocabulary difficulty using a common-word allowlist."""

    words = [word.casefold() for word in EN_WORD_RE.findall(text)]
    if not words:
        return 0.0
    rare_count = sum(1 for word in words if word not in COMMON_ENGLISH_WORDS)
    return rare_count / len(words)


def ko_eojeol_count(text: str) -> int:
    """Count Korean eojeol-like whitespace-delimited tokens."""

    return len(KO_EOJEOL_RE.findall(text))


def ko_average_sentence_length(text: str) -> float:
    """Compute average Korean sentence length in eojeol units."""

    sentences = [sentence for sentence in SENTENCE_SPLIT_RE.split(text) if sentence.strip()]
    if not sentences:
        return float(ko_eojeol_count(text))
    sentence_lengths = [
        len(KO_EOJEOL_RE.findall(sentence)) for sentence in sentences if sentence.strip()
    ]
    if not sentence_lengths:
        return 0.0
    return sum(sentence_lengths) / len(sentence_lengths)


def ko_hanja_ratio(text: str) -> float:
    """Estimate vocabulary difficulty using Hanja-character density."""

    eojeol_count = ko_eojeol_count(text)
    if eojeol_count == 0:
        return 0.0
    return len(HANJA_RE.findall(text)) / eojeol_count


def classify_by_thresholds(
    token_count: int, avg_sentence_tokens: float, complexity: float
) -> AgeBand:
    """Classify one article using shared threshold logic."""

    if (
        token_count <= AgeBandHeuristics.PRESCHOOL_MAX_TOKENS
        and avg_sentence_tokens <= AgeBandHeuristics.PRESCHOOL_MAX_SENTENCE_TOKENS
        and complexity <= AgeBandHeuristics.PRESCHOOL_MAX_COMPLEXITY
    ):
        return "preschool"
    if (
        token_count <= AgeBandHeuristics.ELEMENTARY_MAX_TOKENS
        and avg_sentence_tokens <= AgeBandHeuristics.ELEMENTARY_MAX_SENTENCE_TOKENS
        and complexity <= AgeBandHeuristics.ELEMENTARY_MAX_COMPLEXITY
    ):
        return "elementary"
    return "middle_school"


def infer_age_band(text: str, language: str) -> AgeBand:
    """Infer one article's age band from its text and language."""

    if language == "en":
        return classify_by_thresholds(
            en_word_count(text),
            en_average_sentence_length(text),
            en_rare_word_ratio(text),
        )
    if language == "ko":
        return classify_by_thresholds(
            ko_eojeol_count(text),
            ko_average_sentence_length(text),
            ko_hanja_ratio(text),
        )
    raise ValueError(f"Unsupported language: {language}")


def iter_jsonl(path: Path) -> list[dict[str, object]]:
    """Load JSONL records from one file."""

    records: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object rows in {path}")
            records.append(payload)
    return records


def tag_record(record: dict[str, object], overrides: dict[str, AgeBand]) -> dict[str, object]:
    """Apply the heuristic or an override to one article record."""

    row_id = str(record.get("id", "")).strip()
    language = str(record.get("language", "")).strip()
    text = str(record.get("text", "")).strip()
    if not row_id or not language or not text:
        raise ValueError("Each row must include non-empty id, language, and text fields")
    tagged = dict(record)
    tagged["age_band_hint"] = overrides.get(row_id, infer_age_band(text, language))
    return tagged


def process_source(
    *,
    input_dir: Path,
    output_dir: Path,
    source: str,
    overrides: dict[str, AgeBand],
) -> Path:
    """Tag one filtered source file and write the tagged output."""

    input_path = input_dir / f"{source}_filtered.jsonl"
    output_path = output_dir / f"{source}_tagged.jsonl"
    if not input_path.exists():
        LOGGER.warning("Skipping missing input file: %s", input_path)
        return output_path

    output_dir.mkdir(parents=True, exist_ok=True)
    tagged_rows = [tag_record(record, overrides) for record in iter_jsonl(input_path)]
    with output_path.open("w", encoding="utf-8") as handle:
        for row in tagged_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    LOGGER.info("Tagged %d rows for source=%s -> %s", len(tagged_rows), source, output_path)
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    """Run the age-band tagging CLI."""

    configure_logging()
    args = parse_args(argv)
    overrides = load_overrides(args.overrides_file)
    for source in INPUT_SOURCES:
        process_source(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            source=source,
            overrides=overrides,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
