"""Validation pipeline for Mungi QLoRA fine-tuning training data.

Applies global and per-category rules to each sample,
outputs PASS/REJECT/FLAG with reason.
Can be used as a module (imported by generate_finetune_data.py) or as CLI.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent

BANNED_ENDINGS_RE = re.compile(
    r"(요|습니다|세요|해요|죠|까요|네요|거예요|줄게요|할게요)"
    r"[\s.!?~…]*$"
)

ALLOWED_ENDINGS_RE = re.compile(
    r"(야|어|해|지|다|까|자|겠다|구나|잖아|거야|인데|할까|볼까|이야"
    r"|니까|했어|좋아|싶어|줄게|할게|보자|같아|될까|는데|았어|었어"
    r"|려고|면서|래|네|군|걸|듯해|셈이야)"
    r"[\s.!?~…]*$"
)

ENGLISH_RE = re.compile(r"[a-zA-Z]{2,}")
EMOJI_RE = re.compile(
    "["
    "\U0001f600-\U0001f64f"
    "\U0001f300-\U0001f5ff"
    "\U0001f680-\U0001f6ff"
    "\U0001f1e0-\U0001f1ff"
    "\U0001f900-\U0001f9ff"
    "\U0001fa00-\U0001fa6f"
    "\U0001fa70-\U0001faff"
    "\U00002702-\U000027b0"
    "\U0000fe00-\U0000fe0f"
    "\U0000200d"
    "]+",
    flags=re.UNICODE,
)

# Known STT noise variants (from core/pipeline.py _STT_ALIAS_MAP)
STT_NOISE_VARIANTS: set[str] = {
    "웅이",
    "문이",
    "멍인",
    "멍이",
    "무이",
    "멍의",
    "붕이",
    "몽이",
    "색칠로이",
    "공용",
    "세종대이",
    "세종대",
}

# Token truncation patterns from v3 plan (17 patterns)
TRUNCATION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p)
    for p in [
        r"이에\s+요",
        r"거예\s+요",
        r"마\s+법",
        r"빵\s+를",
        r"같아욬",
        r"즐\s+있을",
        r"컨데여",
        r"보다은",
        r"에\s+요(?!\s*\w)",
        r"해\s+요(?!\s*\w)",
        r"인\s+요",
        r"는\s+요",
        r"을\s+요",
        r"이\s+요",
        r"줄\s+요",
        r"할\s+요",
        r"볼\s+요",
    ]
]


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

Verdict = Literal["PASS", "REJECT", "FLAG"]


@dataclass
class ValidationResult:
    """Result of validating a single sample."""

    verdict: Verdict
    reasons: list[str] = field(default_factory=list)

    def add(self, verdict: Verdict, reason: str) -> None:
        """Accumulate a finding. REJECT overrides FLAG; FLAG overrides PASS."""
        self.reasons.append(f"[{verdict}] {reason}")
        priority = {"PASS": 0, "FLAG": 1, "REJECT": 2}
        if priority[verdict] > priority[self.verdict]:
            self.verdict = verdict


@dataclass
class ValidationReport:
    """Aggregate validation statistics."""

    total: int = 0
    passed: int = 0
    rejected: int = 0
    flagged: int = 0
    reject_reasons: dict[str, int] = field(default_factory=dict)

    def record(self, result: ValidationResult) -> None:
        """Record one sample result."""
        self.total += 1
        if result.verdict == "PASS":
            self.passed += 1
        elif result.verdict == "REJECT":
            self.rejected += 1
            for r in result.reasons:
                if r.startswith("[REJECT]"):
                    key = r.removeprefix("[REJECT] ")
                    self.reject_reasons[key] = self.reject_reasons.get(key, 0) + 1
        else:
            self.flagged += 1

    def to_dict(self) -> dict[str, object]:
        """Serialize to JSON-friendly dict."""
        return {
            "total": self.total,
            "passed": self.passed,
            "rejected": self.rejected,
            "flagged": self.flagged,
            "pass_rate": (round(self.passed / self.total * 100, 1) if self.total else 0),
            "reject_reasons": dict(
                sorted(
                    self.reject_reasons.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
            ),
        }


# ---------------------------------------------------------------------------
# Blocklist loader
# ---------------------------------------------------------------------------

_blocklist_cache: list[str] | None = None


def _load_blocklist() -> list[str]:
    """Load blocked terms from assets/filters/blocklist.json."""
    global _blocklist_cache  # noqa: PLW0603
    if _blocklist_cache is not None:
        return _blocklist_cache

    path = REPO_ROOT / "assets" / "filters" / "blocklist.json"
    if not path.exists():
        logger.warning("blocklist.json not found at %s", path)
        _blocklist_cache = []
        return _blocklist_cache

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    terms: list[str] = []
    for category in data.values():
        if isinstance(category, dict):
            for entry_list in category.values():
                if isinstance(entry_list, list):
                    terms.extend(str(t) for t in entry_list if isinstance(t, str))
        elif isinstance(category, list):
            terms.extend(str(t) for t in category if isinstance(t, str))

    _blocklist_cache = terms
    logger.info("Loaded %d blocklist terms", len(terms))
    return _blocklist_cache


# ---------------------------------------------------------------------------
# Core validation functions
# ---------------------------------------------------------------------------


def _extract_assistant_texts(sample: dict) -> list[str]:
    """Extract all assistant response texts from a sample."""
    messages = sample.get("messages", [])
    return [m["content"] for m in messages if m.get("role") == "assistant" and m.get("content")]


def _extract_user_texts(sample: dict) -> list[str]:
    """Extract all user input texts from a sample."""
    messages = sample.get("messages", [])
    return [m["content"] for m in messages if m.get("role") == "user" and m.get("content")]


def _split_sentences(text: str) -> list[str]:
    """Split Korean text into sentences."""
    parts = re.split(r"[.!?~…]+", text)
    return [p.strip() for p in parts if p.strip()]


def detect_echo(user_text: str, response_text: str) -> bool:
    """Detect whether a response mostly repeats the user's input.

    Replicates logic from models/llm_runner.py:182-198.
    """
    if not user_text or not response_text:
        return False

    clean_user = re.sub(r"[?!.,~\s]+", "", user_text)
    clean_resp = re.sub(r"[?!.,~\s]+", "", response_text)
    if not clean_user or not clean_resp:
        return False

    overlap = sum(1 for char in clean_user if char in clean_resp)
    ratio = overlap / len(clean_user) if clean_user else 0.0
    return (
        ratio > 0.8
        and len(clean_resp) < len(clean_user) * 1.5
        and (clean_user in clean_resp or clean_resp in clean_user)
    )


def _has_substantive_content(user_text: str, assistant_text: str) -> bool:
    """Check if assistant response has words not in user input."""
    user_words = set(re.findall(r"[\uAC00-\uD7A3]+", user_text))
    assistant_words = set(re.findall(r"[\uAC00-\uD7A3]+", assistant_text))
    new_words = assistant_words - user_words
    return len(new_words) >= 1


def _has_stt_noise(text: str) -> bool:
    """Check if text contains known STT noise variants."""
    return any(variant in text for variant in STT_NOISE_VARIANTS)


# ---------------------------------------------------------------------------
# Global validation
# ---------------------------------------------------------------------------


def validate_global(sample: dict) -> ValidationResult:
    """Apply global validation rules to a sample."""
    result = ValidationResult(verdict="PASS")

    # 1. Structure check
    messages = sample.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        result.add("REJECT", "Invalid messages structure")
        return result

    assistant_texts = _extract_assistant_texts(sample)
    if not assistant_texts:
        result.add("REJECT", "No assistant response found")
        return result

    for resp in assistant_texts:
        # 2. Banned honorific endings
        sentences = _split_sentences(resp)
        for sent in sentences:
            if BANNED_ENDINGS_RE.search(sent):
                result.add("REJECT", f"Honorific ending: '{sent[-15:]}'")

        # 3. Sentence count (1-4 allowed)
        if len(sentences) > 4:
            result.add("REJECT", f"Too many sentences: {len(sentences)}")
        elif len(sentences) == 0:
            result.add("REJECT", "Empty response")

        # 4. English check
        if ENGLISH_RE.search(resp):
            match = ENGLISH_RE.search(resp)
            assert match is not None
            result.add("REJECT", f"English detected: '{match.group()}'")

        # 5. Emoji check
        if EMOJI_RE.search(resp):
            result.add("REJECT", "Emoji detected")

        # 6. Blocklist check
        blocklist = _load_blocklist()
        for term in blocklist:
            if term in resp:
                result.add("REJECT", f"Blocklist term: '{term}'")
                break

    return result


# ---------------------------------------------------------------------------
# Category-specific validation
# ---------------------------------------------------------------------------


def validate_category_b(
    sample: dict,
    subcategory: str | None = None,
) -> ValidationResult:
    """Validate anti-parrot category samples."""
    result = ValidationResult(verdict="PASS")
    user_texts = _extract_user_texts(sample)
    assistant_texts = _extract_assistant_texts(sample)

    for user_text, assistant_text in zip(
        user_texts,
        assistant_texts,
        strict=False,
    ):
        # Echo detection
        if detect_echo(user_text, assistant_text):
            result.add("REJECT", "Echo/parrot response detected")

        # Substantive content check
        if not _has_substantive_content(user_text, assistant_text):
            result.add("REJECT", "No substantive content (all words from user)")

    # STT noise requirement for B1/B4
    if subcategory in ("B1_stt_noise", "B4_multiturn_noise"):
        has_noise = any(_has_stt_noise(t) for t in user_texts)
        if not has_noise:
            result.add("REJECT", f"No STT noise in {subcategory} sample")

    return result


def validate_category_e(
    sample: dict,
    subcategory: str | None = None,
) -> ValidationResult:
    """Validate hallucination prevention samples."""
    result = ValidationResult(verdict="PASS")
    assistant_texts = _extract_assistant_texts(sample)

    if subcategory == "unknown_admission":
        for resp in assistant_texts:
            has_unknown = any(
                phrase in resp for phrase in ["모르겠", "궁금해", "물어보자", "생각해볼까"]
            )
            if not has_unknown:
                result.add(
                    "FLAG",
                    "Unknown-admission sample lacks uncertainty phrase",
                )

            # Should not contain specific numbers
            if re.search(r"\d+(\.\d+)?\s*(미터|킬로|센티|도|km|m|cm)", resp):
                result.add(
                    "REJECT",
                    "Numeric claim in unknown-admission sample",
                )

    return result


def validate_category_f(sample: dict) -> ValidationResult:
    """Validate Korean fluency samples for truncation patterns."""
    result = ValidationResult(verdict="PASS")
    assistant_texts = _extract_assistant_texts(sample)

    for resp in assistant_texts:
        for pattern in TRUNCATION_PATTERNS:
            if pattern.search(resp):
                result.add(
                    "REJECT",
                    f"Token truncation pattern: '{pattern.pattern}'",
                )
                break

    return result


# ---------------------------------------------------------------------------
# Main validation entry point
# ---------------------------------------------------------------------------


def validate_sample(
    sample: dict,
    category: str | None = None,
    subcategory: str | None = None,
) -> ValidationResult:
    """Validate a single training sample with global + category rules.

    Args:
        sample: Training sample with ``messages`` key.
        category: Category letter (A-G) for category-specific checks.
        subcategory: Sub-category identifier for finer rules.

    Returns:
        ValidationResult with verdict and reasons.
    """
    result = validate_global(sample)

    if category == "B":
        cat_result = validate_category_b(sample, subcategory)
        for reason in cat_result.reasons:
            verdict_str = reason[1 : reason.index("]")]
            msg = reason[reason.index("] ") + 2 :]
            result.add(verdict_str, msg)  # type: ignore[arg-type]

    if category == "E":
        cat_result = validate_category_e(sample, subcategory)
        for reason in cat_result.reasons:
            verdict_str = reason[1 : reason.index("]")]
            msg = reason[reason.index("] ") + 2 :]
            result.add(verdict_str, msg)  # type: ignore[arg-type]

    if category == "F":
        cat_result = validate_category_f(sample)
        for reason in cat_result.reasons:
            verdict_str = reason[1 : reason.index("]")]
            msg = reason[reason.index("] ") + 2 :]
            result.add(verdict_str, msg)  # type: ignore[arg-type]

    return result


# ---------------------------------------------------------------------------
# Batch validation
# ---------------------------------------------------------------------------


def validate_jsonl(
    path: Path,
    category: str | None = None,
    subcategory: str | None = None,
) -> ValidationReport:
    """Validate all samples in a JSONL file.

    Args:
        path: Path to JSONL file.
        category: Category letter for category-specific checks.
        subcategory: Sub-category identifier.

    Returns:
        ValidationReport with aggregate statistics.
    """
    report = ValidationReport()

    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                result = ValidationResult(verdict="REJECT")
                result.add("REJECT", f"Invalid JSON at line {line_num}")
                report.record(result)
                continue

            result = validate_sample(sample, category, subcategory)
            report.record(result)

            if result.verdict == "REJECT":
                logger.debug(
                    "Line %d REJECT: %s",
                    line_num,
                    "; ".join(result.reasons),
                )

    return report


# ---------------------------------------------------------------------------
# Deduplication utility
# ---------------------------------------------------------------------------


def _char_ngrams(text: str, n: int = 3) -> set[str]:
    """Generate character n-grams for similarity comparison."""
    cleaned = re.sub(r"\s+", "", text)
    if len(cleaned) < n:
        return {cleaned}
    return {cleaned[i : i + n] for i in range(len(cleaned) - n + 1)}


def compute_similarity(text_a: str, text_b: str) -> float:
    """Compute Jaccard similarity using character trigrams.

    No external dependencies — uses character n-gram overlap as proxy
    for cosine similarity.

    Args:
        text_a: First text.
        text_b: Second text.

    Returns:
        Similarity score between 0.0 and 1.0.
    """
    ngrams_a = _char_ngrams(text_a)
    ngrams_b = _char_ngrams(text_b)
    if not ngrams_a or not ngrams_b:
        return 0.0
    intersection = ngrams_a & ngrams_b
    union = ngrams_a | ngrams_b
    return len(intersection) / len(union)


def deduplicate_samples(
    samples: list[dict],
    threshold: float = 0.85,
) -> list[dict]:
    """Remove near-duplicate samples based on user input similarity.

    Args:
        samples: List of training samples.
        threshold: Jaccard similarity threshold for dedup.

    Returns:
        Deduplicated list of samples.
    """
    before_count = len(samples)
    unique: list[dict] = []
    seen_texts: list[str] = []
    seen_exact_texts: set[str] = set()

    for sample in samples:
        user_texts = _extract_user_texts(sample)
        user_combined = " ".join(user_texts)

        if user_combined in seen_exact_texts:
            continue

        is_dup = False
        for seen in seen_texts:
            if compute_similarity(user_combined, seen) > threshold:
                is_dup = True
                break

        if not is_dup:
            unique.append(sample)
            seen_texts.append(user_combined)
            seen_exact_texts.add(user_combined)

    after_count = len(unique)
    removed = before_count - after_count
    logger.info(
        "Dedup threshold %.2f: %d -> %d samples (%d removed)",
        threshold,
        before_count,
        after_count,
        removed,
    )
    return unique


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file into memory."""
    samples: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                samples.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_num}") from exc
    return samples


def _write_jsonl(
    samples: list[dict[str, Any]],
    path: Path,
    *,
    strip_metadata: bool = False,
) -> None:
    """Write a list of samples as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            payload = {"messages": sample["messages"]} if strip_metadata else sample
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _write_json(data: dict[str, Any] | list[dict[str, Any]], path: Path) -> None:
    """Write JSON data with indentation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _load_validated_samples_with_metadata(
    clean_path: Path,
    reference_path: Path,
) -> list[dict[str, Any]]:
    """Recover metadata for the validated subset from the full batch output."""
    clean_samples = _load_jsonl(clean_path)
    reference_samples = _load_jsonl(reference_path)
    enriched: list[dict[str, Any]] = []
    ref_idx = 0

    for clean_sample in clean_samples:
        clean_messages = clean_sample.get("messages")
        matched = False

        while ref_idx < len(reference_samples):
            reference_sample = reference_samples[ref_idx]
            ref_idx += 1
            if reference_sample.get("messages") == clean_messages:
                enriched.append(dict(reference_sample))
                matched = True
                break

        if not matched:
            raise ValueError(
                "Validated clean dataset is not an ordered subset of batch_all_messages.jsonl"
            )

    return enriched


def _assistant_response_length(sample: dict[str, Any]) -> int:
    """Measure total assistant response length in characters."""
    return sum(len(text) for text in _extract_assistant_texts(sample))


def _assistant_sentence_count(sample: dict[str, Any]) -> int:
    """Count assistant sentences across all assistant turns in a sample."""
    return sum(len(_split_sentences(text)) for text in _extract_assistant_texts(sample))


def _is_multiturn(sample: dict[str, Any]) -> bool:
    """Check whether a sample contains more than one user-assistant pair."""
    user_count = len(_extract_user_texts(sample))
    assistant_count = len(_extract_assistant_texts(sample))
    return min(user_count, assistant_count) > 1


def _round_metric(value: float, digits: int = 4) -> float:
    """Round floating-point metrics for JSON reports."""
    return round(value, digits)


def _percentile(values: list[int], percentile: float) -> float:
    """Compute a percentile using linear interpolation."""
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])

    ordered = sorted(values)
    rank = (len(ordered) - 1) * (percentile / 100)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def _build_distribution(values: list[int]) -> dict[str, float | int]:
    """Build summary stats for a numeric distribution."""
    if not values:
        return {
            "min": 0,
            "max": 0,
            "mean": 0.0,
            "median": 0.0,
            "p95": 0.0,
        }

    return {
        "min": min(values),
        "max": max(values),
        "mean": _round_metric(statistics.mean(values), 2),
        "median": _round_metric(statistics.median(values), 2),
        "p95": _round_metric(_percentile(values, 95), 2),
    }


def _build_sentence_distribution(sentence_counts: list[int]) -> dict[str, int]:
    """Bucket sentence counts into 1/2/3/4+ bins."""
    distribution = {"1": 0, "2": 0, "3": 0, "4+": 0}

    for count in sentence_counts:
        if count <= 1:
            distribution["1"] += 1
        elif count == 2:
            distribution["2"] += 1
        elif count == 3:
            distribution["3"] += 1
        else:
            distribution["4+"] += 1

    return distribution


def _count_by_category(samples: list[dict[str, Any]]) -> Counter[str]:
    """Count samples by category metadata."""
    counts: Counter[str] = Counter()
    for sample in samples:
        category = str(sample.get("_category", "UNKNOWN"))
        counts[category] += 1
    return counts


def _combined_user_text(sample: dict[str, Any]) -> str:
    """Combine all user turns into one normalized string."""
    return re.sub(r"\s+", " ", " ".join(_extract_user_texts(sample))).strip()


def _combined_assistant_text(sample: dict[str, Any]) -> str:
    """Combine all assistant turns into one normalized string."""
    return re.sub(r"\s+", " ", " ".join(_extract_assistant_texts(sample))).strip()


def _build_quality_report(
    deduped_samples: list[dict[str, Any]],
    before_counts: Counter[str],
    after_counts: Counter[str],
) -> dict[str, Any]:
    """Build the statistical quality report for the deduplicated dataset."""
    per_category: dict[str, Any] = {}
    assistant_lengths = [_assistant_response_length(sample) for sample in deduped_samples]
    sentence_counts = [_assistant_sentence_count(sample) for sample in deduped_samples]
    multiturn_count = sum(1 for sample in deduped_samples if _is_multiturn(sample))

    samples_by_category: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in deduped_samples:
        samples_by_category[str(sample["_category"])].append(sample)

    for category in sorted(samples_by_category):
        category_samples = samples_by_category[category]
        lengths = [_assistant_response_length(sample) for sample in category_samples]
        cat_sentence_counts = [_assistant_sentence_count(sample) for sample in category_samples]
        cat_multiturn_count = sum(1 for sample in category_samples if _is_multiturn(sample))
        per_category[category] = {
            "count": len(category_samples),
            "avg_assistant_length": _round_metric(statistics.mean(lengths), 2),
            "avg_sentences": _round_metric(statistics.mean(cat_sentence_counts), 2),
            "multiturn_ratio": _round_metric(
                cat_multiturn_count / len(category_samples),
            ),
            "response_length_distribution": _build_distribution(lengths),
        }

    return {
        "total_samples": len(deduped_samples),
        "total_before_dedup": sum(before_counts.values()),
        "per_category": per_category,
        "response_length_distribution": _build_distribution(assistant_lengths),
        "sentence_count_distribution": _build_sentence_distribution(sentence_counts),
        "multiturn_ratio": _round_metric(
            multiturn_count / len(deduped_samples) if deduped_samples else 0.0,
        ),
        "dedup_removed": sum(before_counts.values()) - len(deduped_samples),
        "dedup_removed_per_category": {
            category: before_counts.get(category, 0) - after_counts.get(category, 0)
            for category in sorted(before_counts)
        },
    }


def _format_spot_check_text(texts: list[str]) -> str:
    """Flatten one or more turns into a readable review string."""
    if len(texts) <= 1:
        return texts[0] if texts else ""
    return "\n".join(f"{index}. {text}" for index, text in enumerate(texts, start=1))


def _build_spot_check_samples(
    deduped_samples: list[dict[str, Any]],
    sample_size: int,
    random_seed: int,
) -> list[dict[str, Any]]:
    """Select reproducible random spot-check samples per category."""
    rng = random.Random(random_seed)
    indexed_by_category: defaultdict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)

    for index, sample in enumerate(deduped_samples, start=1):
        indexed_by_category[str(sample["_category"])].append((index, sample))

    output: list[dict[str, Any]] = []
    for category in sorted(indexed_by_category):
        pool = indexed_by_category[category]
        if len(pool) < sample_size:
            raise ValueError(
                f"Category {category} has only {len(pool)} samples; need {sample_size}"
            )
        selected = sorted(rng.sample(pool, sample_size), key=lambda item: item[0])
        for index, sample in selected:
            output.append(
                {
                    "category": category,
                    "index": index,
                    "user": _format_spot_check_text(_extract_user_texts(sample)),
                    "assistant": _format_spot_check_text(_extract_assistant_texts(sample)),
                    "turn_type": str(sample["_turn_type"]),
                }
            )

    return output


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    """Check whether text contains any keyword fragment."""
    return any(keyword in text for keyword in keywords)


def _build_deep_validation_report(
    deduped_samples: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build category-specific deep validation checks."""
    sad_keywords = (
        "슬퍼",
        "속상",
        "무서",
        "걱정",
        "울",
        "아파",
        "외로",
        "서운",
        "다쳤",
        "힘들",
    )
    celebratory_keywords = (
        "축하",
        "신난",
        "우와",
        "와아",
        "대박",
        "최고",
        "멋지",
        "좋겠다",
        "짝짝",
        "해냈",
        "기쁘",
        "잘했",
    )

    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in deduped_samples:
        grouped[str(sample["_category"])].append(sample)

    category_b = grouped.get("B", [])
    echo_samples = 0
    for sample in category_b:
        pairs = zip(
            _extract_user_texts(sample),
            _extract_assistant_texts(sample),
            strict=False,
        )
        if any(detect_echo(user_text, assistant_text) for user_text, assistant_text in pairs):
            echo_samples += 1

    category_d = grouped.get("D", [])
    sad_context_samples = 0
    emotion_mismatches = 0
    for sample in category_d:
        user_text = _combined_user_text(sample)
        assistant_text = _combined_assistant_text(sample)
        if _contains_any(user_text, sad_keywords):
            sad_context_samples += 1
            if _contains_any(assistant_text, celebratory_keywords):
                emotion_mismatches += 1

    category_f = grouped.get("F", [])
    truncation_samples = 0
    truncation_pattern_counts: Counter[str] = Counter()
    for sample in category_f:
        sample_has_truncation = False
        for assistant_text in _extract_assistant_texts(sample):
            for pattern in TRUNCATION_PATTERNS:
                if pattern.search(assistant_text):
                    truncation_pattern_counts[pattern.pattern] += 1
                    sample_has_truncation = True
        if sample_has_truncation:
            truncation_samples += 1

    diversity_report: dict[str, Any] = {}
    for category in ("H", "I"):
        category_samples = grouped.get(category, [])
        responses_by_user: defaultdict[str, set[str]] = defaultdict(set)
        sample_counts_by_user: Counter[str] = Counter()

        for sample in category_samples:
            user_text = _combined_user_text(sample)
            assistant_text = _combined_assistant_text(sample)
            responses_by_user[user_text].add(assistant_text)
            sample_counts_by_user[user_text] += 1

        repeated_inputs = [
            user_text for user_text, count in sample_counts_by_user.items() if count > 1
        ]
        avg_unique_all = (
            sum(len(responses) for responses in responses_by_user.values()) / len(responses_by_user)
            if responses_by_user
            else 0.0
        )
        avg_unique_repeated = (
            sum(len(responses_by_user[user_text]) for user_text in repeated_inputs)
            / len(repeated_inputs)
            if repeated_inputs
            else 0.0
        )
        diversity_report[category] = {
            "sample_count": len(category_samples),
            "unique_user_inputs": len(responses_by_user),
            "repeated_user_inputs": len(repeated_inputs),
            "avg_unique_responses_per_unique_user_input": _round_metric(
                avg_unique_all,
                2,
            ),
            "avg_unique_responses_per_repeated_user_input": _round_metric(
                avg_unique_repeated,
                2,
            ),
            "max_unique_responses_for_one_input": (
                max((len(responses) for responses in responses_by_user.values()), default=0)
            ),
        }

    return {
        "total_samples": len(deduped_samples),
        "B": {
            "sample_count": len(category_b),
            "echo_samples": echo_samples,
            "echo_rate": _round_metric(
                echo_samples / len(category_b) if category_b else 0.0,
            ),
        },
        "D": {
            "sample_count": len(category_d),
            "sad_context_samples": sad_context_samples,
            "emotion_keyword_mismatches": emotion_mismatches,
            "mismatch_rate": _round_metric(
                emotion_mismatches / sad_context_samples if sad_context_samples else 0.0,
            ),
        },
        "F": {
            "sample_count": len(category_f),
            "truncation_samples": truncation_samples,
            "truncation_rate": _round_metric(
                truncation_samples / len(category_f) if category_f else 0.0,
            ),
            "pattern_counts": dict(sorted(truncation_pattern_counts.items())),
        },
        "H": diversity_report["H"],
        "I": diversity_report["I"],
    }


def run_deep_quality_workflow(
    clean_path: Path,
    reference_path: Path,
    dedup_output: Path,
    quality_report_output: Path,
    spot_check_output: Path,
    deep_report_output: Path,
    *,
    dedup_threshold: float = 0.85,
    sample_size: int = 50,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Generate the deduped dataset and all deep-validation reports."""
    enriched_samples = _load_validated_samples_with_metadata(clean_path, reference_path)
    before_counts = _count_by_category(enriched_samples)
    deduped_samples = deduplicate_samples(enriched_samples, threshold=dedup_threshold)
    after_counts = _count_by_category(deduped_samples)

    quality_report = _build_quality_report(deduped_samples, before_counts, after_counts)
    spot_check_samples = _build_spot_check_samples(
        deduped_samples,
        sample_size,
        random_seed,
    )
    deep_validation_report = _build_deep_validation_report(deduped_samples)

    _write_jsonl(deduped_samples, dedup_output, strip_metadata=True)
    _write_json(quality_report, quality_report_output)
    _write_json(spot_check_samples, spot_check_output)
    _write_json(deep_validation_report, deep_report_output)

    return {
        "before_counts": dict(before_counts),
        "after_counts": dict(after_counts),
        "dedup_total": quality_report["dedup_removed"],
        "total_samples": quality_report["total_samples"],
        "echo_rate": deep_validation_report["B"]["echo_rate"],
        "truncation_rate": deep_validation_report["F"]["truncation_rate"],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _configure_logging(verbose: bool) -> None:
    """Set up structured logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    """CLI entry point for validation pipeline."""
    parser = argparse.ArgumentParser(
        description="Validate Mungi fine-tuning training data",
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        help="Path to JSONL file to validate",
    )
    parser.add_argument(
        "--category",
        "-c",
        choices=list("ABCDEFGHI"),
        help="Category for category-specific validation",
    )
    parser.add_argument(
        "--subcategory",
        "-s",
        help="Sub-category identifier (e.g., B1_stt_noise)",
    )
    parser.add_argument(
        "--report",
        "-r",
        type=Path,
        help="Output JSON report path",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--run-deep-quality-workflow",
        action="store_true",
        help="Generate deduped training data and deep-validation reports",
    )
    parser.add_argument(
        "--reference-jsonl",
        type=Path,
        default=REPO_ROOT / "assets" / "training" / "batch_all_messages.jsonl",
        help="Reference JSONL with category metadata for the validated dataset",
    )
    parser.add_argument(
        "--dedup-output",
        type=Path,
        default=REPO_ROOT / "assets" / "training" / "batch_deduped.jsonl",
        help="Output path for the deduplicated JSONL dataset",
    )
    parser.add_argument(
        "--quality-report-output",
        type=Path,
        default=REPO_ROOT / "assets" / "training" / "quality_report_v4.json",
        help="Output path for the statistical quality report",
    )
    parser.add_argument(
        "--spot-check-output",
        type=Path,
        default=REPO_ROOT / "assets" / "training" / "spot_check_samples.json",
        help="Output path for the random spot-check sample file",
    )
    parser.add_argument(
        "--deep-report-output",
        type=Path,
        default=REPO_ROOT / "assets" / "training" / "deep_validation_report.json",
        help="Output path for the category-specific deep validation report",
    )
    parser.add_argument(
        "--spot-check-per-category",
        type=int,
        default=50,
        help="Number of manual review samples to select per category",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducible spot-check sampling",
    )
    parser.add_argument(
        "--dedup-threshold",
        type=float,
        default=0.85,
        help="Jaccard similarity threshold used for trigram deduplication",
    )
    args = parser.parse_args()
    _configure_logging(args.verbose)

    if args.run_deep_quality_workflow:
        clean_path = args.input or (
            REPO_ROOT / "assets" / "training" / "batch_validated_clean.jsonl"
        )
        if not clean_path.exists():
            logger.error("Validated clean dataset not found: %s", clean_path)
            sys.exit(1)
        if not args.reference_jsonl.exists():
            logger.error("Reference dataset not found: %s", args.reference_jsonl)
            sys.exit(1)

        summary = run_deep_quality_workflow(
            clean_path,
            args.reference_jsonl,
            args.dedup_output,
            args.quality_report_output,
            args.spot_check_output,
            args.deep_report_output,
            dedup_threshold=args.dedup_threshold,
            sample_size=args.spot_check_per_category,
            random_seed=args.random_seed,
        )
        logger.info(
            "Deep validation complete: %d samples after dedup, %d removed",
            summary["total_samples"],
            summary["dedup_total"],
        )
        return

    if args.input is None:
        logger.error("input is required unless --run-deep-quality-workflow is used")
        sys.exit(1)

    if not args.input.exists():
        logger.error("File not found: %s", args.input)
        sys.exit(1)

    report = validate_jsonl(args.input, args.category, args.subcategory)

    logger.info(
        "Validation complete: %d total, %d passed (%.1f%%), %d rejected, %d flagged",
        report.total,
        report.passed,
        report.to_dict()["pass_rate"],
        report.rejected,
        report.flagged,
    )

    if report.reject_reasons:
        logger.info("Top reject reasons:")
        for reason, count in list(report.reject_reasons.items())[:10]:
            logger.info("  %d x %s", count, reason)

    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info("Report saved to %s", args.report)

    sys.exit(1 if report.rejected > 0 else 0)


if __name__ == "__main__":
    main()
