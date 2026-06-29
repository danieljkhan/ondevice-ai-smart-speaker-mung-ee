"""Score Phase 0 confirmable-fact holdout responses offline."""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, cast

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

Verdict = Literal[
    "correct",
    "appropriate-deference",
    "inappropriate-deference",
    "confident-fabrication",
    "other",
]
InjectionState = Literal["off", "on"]
Axis = Literal["matched", "unmatched"]
AgeBand = Literal["under_10", "under_15"]

AXES: Final[tuple[Axis, ...]] = ("matched", "unmatched")
AGE_BANDS: Final[tuple[AgeBand, ...]] = ("under_10", "under_15")

NUMERIC_RE: Final[re.Pattern[str]] = re.compile(
    r"(\d{1,3}(?:,\d{3})*|\d+)\s*(개|명|년|°c|℃|km|m|톤|색|종|동물)?",
    re.IGNORECASE,
)
DEFERENCE_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"모르겠어"),
    re.compile(r"잘 모르겠어요"),
    re.compile(r"확실하지 않"),
    re.compile(r"어른에게 물어"),
    re.compile(r"확실히 모르"),
    re.compile(r"기억이 안 나"),
)
ASSERTIVE_FABRICATION_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r".+(?:이야|야|입니다)(?:[.!?]|\s|$)"),
    re.compile(r"(?:수도는|모두|전부|평균은|약\s*\d)"),
)
VERDICTS: Final[tuple[Verdict, ...]] = (
    "correct",
    "appropriate-deference",
    "inappropriate-deference",
    "confident-fabrication",
    "other",
)
CORRECTNESS_OVERLAP_THRESHOLD: Final[float] = 0.65
MIN_CONTENT_TOKENS_FOR_OVERLAP: Final[int] = 2

KOREAN_SEQUENCE_RE: Final[re.Pattern[str]] = re.compile(r"[가-힣]+")
KOREAN_SINO_DIGITS: Final[dict[str, int]] = {
    "영": 0,
    "일": 1,
    "이": 2,
    "삼": 3,
    "사": 4,
    "오": 5,
    "육": 6,
    "칠": 7,
    "팔": 8,
    "구": 9,
}
KOREAN_NATIVE_ONES: Final[dict[str, tuple[int, bool]]] = {
    "하나": (1, False),
    "한": (1, True),
    "둘": (2, False),
    "두": (2, True),
    "셋": (3, False),
    "세": (3, True),
    "넷": (4, False),
    "네": (4, True),
    "다섯": (5, False),
    "여섯": (6, False),
    "일곱": (7, False),
    "여덟": (8, False),
    "아홉": (9, False),
}
KOREAN_NATIVE_TENS: Final[dict[str, int]] = {
    "열": 10,
    "스물": 20,
    "스무": 20,
    "서른": 30,
    "마흔": 40,
    "쉰": 50,
}
KOREAN_NUMERAL_COUNTERS: Final[tuple[str, ...]] = (
    "번째",
    "가지",
    "마리",
    "사람",
    "개월",
    "개",
    "명",
    "년",
    "살",
    "종",
    "색",
    "톤",
    "번",
    "시",
    "분",
    "초",
    "월",
    "일",
    "달",
    "층",
    "원",
    "도",
)
KOREAN_NUMERAL_WEAK_SUFFIXES: Final[tuple[str, ...]] = (
    "입니다",
    "이야",
    "거야",
    "예요",
    "에요",
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "의",
    "에",
    "와",
    "과",
    "도",
    "만",
    "야",
)
CONTENT_TRAILING_SUFFIXES: Final[tuple[str, ...]] = tuple(
    sorted(KOREAN_NUMERAL_WEAK_SUFFIXES, key=len, reverse=True)
)
CONTENT_TOKEN_STOPLIST: Final[set[str]] = {
    "가장",
    "기본",
    "정답",
    "답변",
    "사실",
    "정보",
    "기준",
    "정도",
    "대략",
    "약",
    "거",
    "것",
    "있어",
    "있다",
}
CONTENT_TOKEN_EQUIVALENTS: Final[dict[str, tuple[str, ...]]] = {
    "도는": ("돌기",),
    "대량": ("많이",),
    "생산하기": ("만들",),
    "자전": ("돌기",),
    "찾는": ("찾아",),
}
CONTENT_TOKEN_STRIP_CHARS: Final[str] = " \t\r\n.,!?;:()[]{}'\"“”‘’"


@dataclass(frozen=True)
class _KoreanNumeralParse:
    value: int
    requires_counter: bool


@dataclass(frozen=True)
class _KoreanNumeralPrefix:
    value: int
    length: int


@dataclass(frozen=True)
class HoldoutRow:
    """One confirmable-fact holdout row."""

    topic: str
    question: str
    category: str
    axis: Axis
    gold_answer: str
    acceptable_variants: tuple[str, ...]
    numeric_tolerance: int | None
    age_band: AgeBand


@dataclass(frozen=True)
class ResponseRow:
    """One generated response row keyed to a holdout question."""

    topic: str
    question: str
    response: str
    ttft_s: float | None = None
    gen_time_s: float | None = None


def iter_jsonl_rows(path: Path) -> Iterator[dict[str, Any]]:
    """Yield JSON object rows from a JSONL file."""

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                msg = f"Expected JSON object at {path}:{line_number}"
                raise ValueError(msg)
            yield row


def load_holdout_rows(path: Path) -> list[HoldoutRow]:
    """Load and validate holdout rows from JSONL."""

    rows: list[HoldoutRow] = []
    for line_number, row in enumerate(iter_jsonl_rows(path), start=1):
        acceptable_variants = row.get("acceptable_variants")
        if not isinstance(acceptable_variants, list) or not all(
            isinstance(item, str) for item in acceptable_variants
        ):
            msg = f"Holdout row {line_number} has invalid acceptable_variants"
            raise ValueError(msg)
        numeric_tolerance = row.get("numeric_tolerance")
        if numeric_tolerance is not None and not isinstance(numeric_tolerance, int):
            msg = f"Holdout row {line_number} has invalid numeric_tolerance"
            raise ValueError(msg)
        axis = _require_axis(row, line_number)
        age_band = _require_age_band(row, line_number)
        rows.append(
            HoldoutRow(
                topic=_require_string(row, "topic", line_number, "holdout"),
                question=_require_string(row, "question", line_number, "holdout"),
                category=_require_string(row, "category", line_number, "holdout"),
                axis=axis,
                gold_answer=_require_string(row, "gold_answer", line_number, "holdout"),
                acceptable_variants=tuple(acceptable_variants),
                numeric_tolerance=cast(int | None, numeric_tolerance),
                age_band=age_band,
            )
        )
    return rows


def load_response_rows(path: Path) -> list[ResponseRow]:
    """Load response rows from JSONL."""

    rows: list[ResponseRow] = []
    for line_number, row in enumerate(iter_jsonl_rows(path), start=1):
        rows.append(
            ResponseRow(
                topic=_require_string(row, "topic", line_number, "response"),
                question=_require_string(row, "question", line_number, "response"),
                response=_require_string(row, "response", line_number, "response"),
            )
        )
    return rows


def score_response_rows(
    holdout_rows: Sequence[HoldoutRow],
    response_rows: Sequence[ResponseRow],
    injection_state: InjectionState,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Score response rows against the Phase 0 holdout schema."""

    holdout_by_key: dict[tuple[str, str], HoldoutRow] = {}
    for row in holdout_rows:
        key = (row.topic, row.question)
        if key in holdout_by_key:
            msg = f"Duplicate holdout key: {row.topic} / {row.question}"
            raise ValueError(msg)
        holdout_by_key[key] = row

    response_by_key: dict[tuple[str, str], ResponseRow] = {}
    for response_row in response_rows:
        key = (response_row.topic, response_row.question)
        if key in response_by_key:
            msg = f"Duplicate response key: {response_row.topic} / {response_row.question}"
            raise ValueError(msg)
        response_by_key[key] = response_row

    missing_keys = sorted(key for key in holdout_by_key if key not in response_by_key)
    if missing_keys:
        topic, question = missing_keys[0]
        msg = f"Missing response row for holdout key: {topic} / {question}"
        raise ValueError(msg)

    extra_keys = sorted(key for key in response_by_key if key not in holdout_by_key)
    if extra_keys:
        topic, question = extra_keys[0]
        msg = f"Response row missing holdout key: {topic} / {question}"
        raise ValueError(msg)

    scored_rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    band_counts: dict[AgeBand, Counter[str]] = {band: Counter() for band in AGE_BANDS}
    for holdout_row in holdout_rows:
        response_row = response_by_key[(holdout_row.topic, holdout_row.question)]
        verdict, matched_variant = classify_response(
            holdout_row,
            response_row.response,
            injection_state=injection_state,
        )
        counts[verdict] += 1
        band_counts[holdout_row.age_band][verdict] += 1
        scored_rows.append(
            {
                "topic": holdout_row.topic,
                "question": holdout_row.question,
                "axis": holdout_row.axis,
                "gold_answer": holdout_row.gold_answer,
                "age_band": holdout_row.age_band,
                "response": response_row.response,
                "injection_state": injection_state,
                "verdict": verdict,
                "matched_variant": matched_variant,
            }
        )

    total_rows = len(scored_rows)
    verdict_counts = {verdict: counts.get(verdict, 0) for verdict in VERDICTS}
    verdict_rates = {
        verdict: (verdict_counts[verdict] / total_rows if total_rows else 0.0)
        for verdict in VERDICTS
    }
    failure_count = (
        verdict_counts["confident-fabrication"] + verdict_counts["inappropriate-deference"]
    )
    summary = {
        "total_rows": total_rows,
        "verdict_counts": verdict_counts,
        "verdict_rates": verdict_rates,
        "failure_rate": (failure_count / total_rows if total_rows else 0.0),
        "age_band_breakdown": _build_age_band_breakdown(
            holdout_rows=holdout_rows,
            band_counts=band_counts,
        ),
    }
    return scored_rows, summary


def classify_response(
    holdout_row: HoldoutRow,
    response: str,
    *,
    injection_state: InjectionState,
) -> tuple[Verdict, str | None]:
    """Classify one model response using the Phase 0 verdict contract."""

    if holdout_row.axis == "unmatched":
        return _classify_unmatched_response(response), None

    for matcher in (
        _match_variant,
        _match_numeral_normalized_variant,
        _match_numeric_tolerance,
        _match_content_overlap,
    ):
        matched_variant = matcher(holdout_row, response)
        if matched_variant is not None:
            return "correct", matched_variant

    if _is_deference(response):
        verdict: Verdict = (
            "inappropriate-deference" if injection_state == "on" else "appropriate-deference"
        )
        return verdict, None

    if _looks_like_confident_fabrication(response):
        return "confident-fabrication", None
    return "other", None


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a JSON payload with repo-standard formatting."""

    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write JSON object rows to JSONL."""

    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def build_parser() -> argparse.ArgumentParser:
    """Build the scorer CLI parser."""

    parser = argparse.ArgumentParser(description="Score Phase 0 fact-shortlist holdout responses.")
    parser.add_argument("--holdout", type=Path, required=True, help="Holdout JSONL path.")
    parser.add_argument("--responses", type=Path, required=True, help="Response JSONL path.")
    parser.add_argument(
        "--injection",
        choices=("off", "on"),
        required=True,
        help="Whether the scored cell ran with injection disabled or enabled.",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        required=True,
        help="Summary JSON output path.",
    )
    parser.add_argument(
        "--output-rows",
        type=Path,
        required=True,
        help="Per-row verdict JSONL output path.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the scorer CLI."""

    args = build_parser().parse_args(argv)
    holdout_rows = load_holdout_rows(args.holdout)
    response_rows = load_response_rows(args.responses)
    scored_rows, summary = score_response_rows(
        holdout_rows,
        response_rows,
        injection_state=cast(InjectionState, args.injection),
    )
    write_json(args.output_summary, summary)
    write_jsonl(args.output_rows, scored_rows)
    return 0


def _require_string(
    row: Mapping[str, Any],
    field_name: str,
    line_number: int,
    label: str,
) -> str:
    value = row.get(field_name)
    if not isinstance(value, str) or not value.strip():
        msg = f"{label.title()} row {line_number} field {field_name} must be a non-empty string"
        raise ValueError(msg)
    return value.strip()


def _require_axis(row: Mapping[str, Any], line_number: int) -> Axis:
    axis = _require_string(row, "axis", line_number, "holdout").casefold()
    if axis not in AXES:
        allowed = ", ".join(AXES)
        msg = f"Holdout row {line_number} field axis must be one of: {allowed}"
        raise ValueError(msg)
    return cast(Axis, axis)


def _require_age_band(row: Mapping[str, Any], line_number: int) -> AgeBand:
    age_band = _require_string(row, "age_band", line_number, "holdout").casefold()
    if age_band not in AGE_BANDS:
        allowed = ", ".join(AGE_BANDS)
        msg = f"Holdout row {line_number} field age_band must be one of: {allowed}"
        raise ValueError(msg)
    return cast(AgeBand, age_band)


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(normalized.split())


def _candidate_answers(holdout_row: HoldoutRow) -> tuple[str, ...]:
    return (holdout_row.gold_answer, *holdout_row.acceptable_variants)


def _match_variant(holdout_row: HoldoutRow, response: str) -> str | None:
    normalized_response = _normalize_text(response)
    for candidate in _candidate_answers(holdout_row):
        normalized_candidate = _normalize_text(candidate)
        if normalized_candidate and normalized_candidate in normalized_response:
            return candidate
    return None


def _match_numeral_normalized_variant(holdout_row: HoldoutRow, response: str) -> str | None:
    normalized_response = _normalize_text_with_korean_numerals(response)
    for candidate in _candidate_answers(holdout_row):
        normalized_candidate = _normalize_text_with_korean_numerals(candidate)
        if normalized_candidate and normalized_candidate in normalized_response:
            return candidate
    return None


def _match_numeric_tolerance(holdout_row: HoldoutRow, response: str) -> str | None:
    if holdout_row.numeric_tolerance is None:
        return None

    response_numbers = _extract_numbers(response)
    if not response_numbers:
        return None

    for candidate in _candidate_answers(holdout_row):
        candidate_numbers = _extract_numbers(candidate)
        for expected in candidate_numbers:
            for observed in response_numbers:
                if abs(observed - expected) <= holdout_row.numeric_tolerance:
                    return candidate
    return None


def _match_content_overlap(holdout_row: HoldoutRow, response: str) -> str | None:
    normalized_response = _normalize_text_with_korean_numerals(response)
    response_tokens = _content_tokens(response)
    for candidate in _candidate_answers(holdout_row):
        if _has_numeric_conflict(
            candidate,
            response,
            numeric_tolerance=holdout_row.numeric_tolerance,
        ):
            continue
        gold_tokens = _content_tokens(candidate)
        if len(gold_tokens) < MIN_CONTENT_TOKENS_FOR_OVERLAP:
            continue
        covered = sum(
            1
            for token in gold_tokens
            if _content_token_is_covered(
                token,
                normalized_response=normalized_response,
                response_tokens=response_tokens,
            )
        )
        if covered / len(gold_tokens) >= CORRECTNESS_OVERLAP_THRESHOLD:
            return candidate
    return None


def _extract_numbers(text: str) -> list[int]:
    numbers: list[int] = []
    normalized = _normalize_korean_numerals(text)
    for match in NUMERIC_RE.finditer(normalized):
        value = match.group(1).replace(",", "")
        try:
            numbers.append(int(value))
        except ValueError:
            continue
    return numbers


def _has_numeric_conflict(
    candidate: str,
    response: str,
    *,
    numeric_tolerance: int | None,
) -> bool:
    candidate_numbers = _extract_numbers(candidate)
    response_numbers = _extract_numbers(response)
    if not candidate_numbers:
        return False

    tolerance = numeric_tolerance if numeric_tolerance is not None else 0
    if not response_numbers:
        return False

    return not any(
        abs(observed - expected) <= tolerance
        for expected in candidate_numbers
        for observed in response_numbers
    )


def _normalize_text_with_korean_numerals(text: str) -> str:
    return _normalize_text(_normalize_korean_numerals(text))


def _normalize_korean_numerals(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    matches = list(KOREAN_SEQUENCE_RE.finditer(normalized))
    if not matches:
        return normalized

    parts: list[str] = []
    previous_end = 0
    for index, match in enumerate(matches):
        parts.append(normalized[previous_end : match.start()])
        next_match = matches[index + 1] if index + 1 < len(matches) else None
        has_following_counter = False
        if next_match is not None:
            separator = normalized[match.end() : next_match.start()]
            has_following_counter = separator.strip() == "" and _starts_with_korean_counter(
                next_match.group(0)
            )
        parts.append(
            _replace_korean_numeral_segment(
                match.group(0),
                has_following_counter=has_following_counter,
            )
        )
        previous_end = match.end()
    parts.append(normalized[previous_end:])

    replaced = "".join(parts)
    counters = "|".join(re.escape(counter) for counter in KOREAN_NUMERAL_COUNTERS)
    return re.sub(rf"(\d+)\s+({counters})", r"\1\2", replaced)


def _replace_korean_numeral_segment(segment: str, *, has_following_counter: bool) -> str:
    prefix = _find_korean_numeral_prefix(
        segment,
        has_following_counter=has_following_counter,
    )
    if prefix is None:
        return segment
    return f"{prefix.value}{segment[prefix.length :]}"


def _find_korean_numeral_prefix(
    segment: str,
    *,
    has_following_counter: bool,
) -> _KoreanNumeralPrefix | None:
    for end_index in range(len(segment), 0, -1):
        candidate = segment[:end_index]
        parsed = _parse_korean_numeral(candidate)
        if parsed is None:
            continue

        suffix = segment[end_index:]
        suffix_context = _korean_numeral_suffix_context(suffix)
        has_counter_context = has_following_counter or suffix_context == "counter"
        if suffix and suffix_context is None:
            continue
        if parsed.requires_counter and not has_counter_context:
            continue
        return _KoreanNumeralPrefix(value=parsed.value, length=end_index)
    return None


def _parse_korean_numeral(word: str) -> _KoreanNumeralParse | None:
    return _parse_native_korean_numeral(word) or _parse_sino_korean_numeral(word)


def _parse_native_korean_numeral(word: str) -> _KoreanNumeralParse | None:
    for ten_word, ten_value in sorted(
        KOREAN_NATIVE_TENS.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        if word == ten_word:
            return _KoreanNumeralParse(value=ten_value, requires_counter=False)
        if word.startswith(ten_word):
            rest = word[len(ten_word) :]
            one = KOREAN_NATIVE_ONES.get(rest)
            if one is not None:
                return _KoreanNumeralParse(value=ten_value + one[0], requires_counter=False)

    one = KOREAN_NATIVE_ONES.get(word)
    if one is None:
        return None
    return _KoreanNumeralParse(value=one[0], requires_counter=one[1])


def _parse_sino_korean_numeral(word: str) -> _KoreanNumeralParse | None:
    if word == "영":
        return _KoreanNumeralParse(value=0, requires_counter=True)
    if not word or any(
        char not in KOREAN_SINO_DIGITS and char not in {"십", "백"} for char in word
    ):
        return None
    if "십" not in word and "백" not in word:
        if len(word) == 1 and word in KOREAN_SINO_DIGITS:
            return _KoreanNumeralParse(
                value=KOREAN_SINO_DIGITS[word],
                requires_counter=True,
            )
        return None
    if word.count("백") > 1 or word.count("십") > 1:
        return None

    total = 0
    rest = word
    if "백" in rest:
        before_hundred, rest = rest.split("백", maxsplit=1)
        if before_hundred:
            hundreds = KOREAN_SINO_DIGITS.get(before_hundred)
            if hundreds is None or hundreds == 0:
                return None
        else:
            hundreds = 1
        total += hundreds * 100

    under_hundred = _parse_sino_under_hundred(rest)
    if under_hundred is None:
        return None
    return _KoreanNumeralParse(value=total + under_hundred, requires_counter=False)


def _parse_sino_under_hundred(word: str) -> int | None:
    if not word:
        return 0
    if "십" in word:
        before_ten, after_ten = word.split("십", maxsplit=1)
        if before_ten:
            tens = KOREAN_SINO_DIGITS.get(before_ten)
            if tens is None or tens == 0:
                return None
        else:
            tens = 1
        if not after_ten:
            return tens * 10
        ones = KOREAN_SINO_DIGITS.get(after_ten)
        if ones is None:
            return None
        return tens * 10 + ones
    return KOREAN_SINO_DIGITS.get(word)


def _starts_with_korean_counter(segment: str) -> bool:
    return any(segment.startswith(counter) for counter in KOREAN_NUMERAL_COUNTERS)


def _korean_numeral_suffix_context(suffix: str) -> Literal["counter", "weak"] | None:
    if not suffix:
        return None
    if _starts_with_korean_counter(suffix):
        return "counter"
    if suffix in KOREAN_NUMERAL_WEAK_SUFFIXES:
        return "weak"
    return None


def _content_tokens(text: str) -> tuple[str, ...]:
    normalized = _normalize_text_with_korean_numerals(text)
    tokens: list[str] = []
    seen: set[str] = set()
    for raw_token in normalized.split():
        token = _normalize_content_token(raw_token)
        if len(token) < 2 or token in CONTENT_TOKEN_STOPLIST or token in seen:
            continue
        tokens.append(token)
        seen.add(token)
    return tuple(tokens)


def _normalize_content_token(raw_token: str) -> str:
    token = raw_token.strip(CONTENT_TOKEN_STRIP_CHARS)
    while token:
        stripped = False
        for suffix in CONTENT_TRAILING_SUFFIXES:
            if token.endswith(suffix) and len(token) - len(suffix) >= 2:
                token = token[: -len(suffix)]
                stripped = True
                break
        if not stripped:
            break
    if token.endswith("겨") and len(token) >= 2:
        token = f"{token[:-1]}기"
    return token.strip(CONTENT_TOKEN_STRIP_CHARS)


def _content_token_is_covered(
    gold_token: str,
    *,
    normalized_response: str,
    response_tokens: tuple[str, ...],
) -> bool:
    if gold_token in normalized_response:
        return True
    if any(_has_content_prefix_match(gold_token, token) for token in response_tokens):
        return True
    equivalents = CONTENT_TOKEN_EQUIVALENTS.get(gold_token, ())
    return any(
        response_token.startswith(equivalent)
        for equivalent in equivalents
        for response_token in response_tokens
    )


def _has_content_prefix_match(gold_token: str, response_token: str) -> bool:
    required_prefix_length = 3 if max(len(gold_token), len(response_token)) >= 3 else 2
    return _common_prefix_length(gold_token, response_token) >= required_prefix_length


def _common_prefix_length(left: str, right: str) -> int:
    length = 0
    for left_char, right_char in zip(left, right, strict=False):
        if left_char != right_char:
            break
        length += 1
    return length


def _build_age_band_breakdown(
    *,
    holdout_rows: Sequence[HoldoutRow],
    band_counts: dict[AgeBand, Counter[str]],
) -> dict[str, Any]:
    totals = Counter(row.age_band for row in holdout_rows)
    breakdown: dict[str, Any] = {}
    for band in AGE_BANDS:
        total_rows = totals.get(band, 0)
        verdict_counts = {verdict: band_counts[band].get(verdict, 0) for verdict in VERDICTS}
        verdict_rates = {
            verdict: (verdict_counts[verdict] / total_rows if total_rows else 0.0)
            for verdict in VERDICTS
        }
        breakdown[band] = {
            "total_rows": total_rows,
            "verdict_counts": verdict_counts,
            "verdict_rates": verdict_rates,
        }
    return breakdown


def _classify_unmatched_response(response: str) -> Verdict:
    normalized = _normalize_text(response)
    if not normalized:
        return "correct"
    if _is_deference(response):
        return "correct"
    if _looks_like_confident_fabrication(response):
        return "confident-fabrication"
    return "correct"


def _is_deference(response: str) -> bool:
    normalized = unicodedata.normalize("NFKC", response)
    return any(pattern.search(normalized) for pattern in DEFERENCE_PATTERNS)


def _looks_like_confident_fabrication(response: str) -> bool:
    normalized = _normalize_text(response)
    if not normalized:
        return False
    if _extract_numbers(response):
        return True
    return any(pattern.search(normalized) for pattern in ASSERTIVE_FABRICATION_PATTERNS)


if __name__ == "__main__":
    raise SystemExit(main())
