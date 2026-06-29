"""Measure the STT accuracy gap across standalone and pipeline invocation paths.

This script runs the same WAV corpus through three increasingly complete STT paths:

- Path A: direct ``models.stt_runner.run_stt()`` on the original WAV
- Path B: pipeline audio prep + VAD + speech extraction, then STT without alias normalization
- Path C: full ``ConversationPipeline.run_turn()`` stopped immediately after STT text is produced

It writes a markdown report containing per-path CER/WER, A->B and B->C deltas, an
auto-generated dominant-cause comment, and a per-round prediction table.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MethodType
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("mungi.scripts.measure_stt_accuracy_gap")

DEFAULT_PATHS: str = "ABC"
TARGET_SAMPLE_RATE: int = 16000
DOMINANT_DELTA_THRESHOLD: float = 0.005
PATH_DESCRIPTIONS: dict[str, str] = {
    "A": "baseline (stt_runner direct)",
    "B": "+ VAD + resample",
    "C": "+ alias normalization (full pipeline)",
}


@dataclass(frozen=True)
class AudioCase:
    """One WAV input paired with its language and ground-truth text."""

    lang: str
    wav_path: Path
    stem: str
    ground_truth: str


@dataclass(frozen=True)
class MeasurementRuntime:
    """Runtime objects used to execute one or more measurement paths."""

    manager: Any
    pipeline: Any | None
    stt_model_type: Any


@dataclass(frozen=True)
class AudioMeasurement:
    """Per-audio measurement output across the selected invocation paths."""

    case: AudioCase
    source_sample_rate: int
    duration_s: float
    predictions: dict[str, str]

    @property
    def resampled_for_pipeline(self) -> bool:
        """Return ``True`` when pipeline preprocessing must resample the input."""
        return self.source_sample_rate != TARGET_SAMPLE_RATE


@dataclass(frozen=True)
class PathSummaryRow:
    """Aggregated CER/WER for one path-language combination."""

    path_code: str
    lang: str
    cer: float
    wer: float
    count: int
    description: str


@dataclass(frozen=True)
class MetricDelta:
    """Delta between two summary rows for the same language."""

    cer_delta: float
    wer_delta: float


@dataclass(frozen=True)
class GapAnalysis:
    """Auto-generated interpretation of the measured path deltas."""

    ab_by_lang: dict[str, MetricDelta]
    bc_by_lang: dict[str, MetricDelta]
    dominant_cause: str
    interpretation: str
    hypothesis_a: str
    hypothesis_c: str
    other: str


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for STT accuracy-gap measurement."""
    parser = argparse.ArgumentParser(
        description=(
            "Measure STT accuracy across direct, VAD-preprocessed, and full-pipeline "
            "invocation paths, then write a markdown comparison report."
        ),
    )
    parser.add_argument(
        "--ko-dir",
        type=Path,
        required=True,
        help="Directory containing Korean WAV inputs.",
    )
    parser.add_argument(
        "--en-dir",
        type=Path,
        required=True,
        help="Directory containing English WAV inputs.",
    )
    parser.add_argument(
        "--gt-json",
        type=Path,
        required=True,
        help="JSON file mapping WAV filename stems to ground-truth text.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output markdown report path.",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=0,
        help="Maximum WAVs to process from each language directory (0 = all).",
    )
    parser.add_argument(
        "--paths",
        type=str.upper,
        default=DEFAULT_PATHS,
        choices=("A", "B", "C", "ABC"),
        help="Which paths to run: A, B, C, or ABC (default: ABC).",
    )
    parser.add_argument(
        "--skip-jetson-preflight",
        action="store_true",
        help="Skip Linux-only Jetson preflight checks.",
    )
    return parser


def _run_preflight(skip: bool) -> None:
    """Run the shared Jetson/Linux preflight helper."""
    from scripts.e2e_60rounds_text_tts import run_preflight

    run_preflight(skip=skip)


def _get_runtime_classes() -> tuple[Any, Any, Any, Any, Any]:
    """Import runtime classes lazily so ``--help`` stays lightweight."""
    from core.model_manager import ManagerConfig, ModelManager, ModelType
    from core.pipeline import ConversationPipeline, PipelineConfig

    return ManagerConfig, ModelManager, ModelType, PipelineConfig, ConversationPipeline


def _get_stt_runner_helpers() -> tuple[Any, Any]:
    """Import the direct STT helpers lazily."""
    from models.stt_runner import _read_wav_samples, run_stt

    return run_stt, _read_wav_samples


def parse_selected_paths(raw_value: str) -> tuple[str, ...]:
    """Normalize the path selector into an ordered tuple."""
    value = raw_value.strip().upper()
    if value == "ABC":
        return ("A", "B", "C")
    if value in {"A", "B", "C"}:
        return (value,)
    msg = f"Unsupported path selector: {raw_value}"
    raise ValueError(msg)


def load_ground_truth_map(path: Path) -> dict[str, str]:
    """Load the stem -> text mapping from JSON."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        msg = f"Failed to read ground-truth JSON: {path} ({exc})"
        raise ValueError(msg) from exc
    except json.JSONDecodeError as exc:
        msg = f"Ground-truth JSON is invalid: {path} ({exc})"
        raise ValueError(msg) from exc

    if not isinstance(payload, dict):
        msg = f"Ground-truth JSON must be an object mapping filename stems to text: {path}"
        raise ValueError(msg)

    result: dict[str, str] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not isinstance(value, str):
            msg = f"Ground-truth JSON contains a non-string key/value pair: {key!r}"
            raise ValueError(msg)
        result[key] = value
    return result


def _discover_wavs(directory: Path, max_rounds: int) -> list[Path]:
    """Discover WAV files from one language directory."""
    if max_rounds < 0:
        msg = "--max-rounds must be >= 0."
        raise ValueError(msg)
    if not directory.exists():
        msg = f"Audio directory does not exist: {directory}"
        raise FileNotFoundError(msg)
    if not directory.is_dir():
        msg = f"Audio path is not a directory: {directory}"
        raise NotADirectoryError(msg)

    wavs = sorted(directory.rglob("*.wav"), key=lambda path: path.name.lower())
    if max_rounds > 0:
        return wavs[:max_rounds]
    return wavs


def collect_audio_cases(
    ko_dir: Path,
    en_dir: Path,
    gt_map: dict[str, str],
    max_rounds: int,
) -> tuple[list[AudioCase], list[str]]:
    """Collect aligned measurement inputs and skip files missing ground truth."""
    cases: list[AudioCase] = []
    skipped: list[str] = []

    for lang, directory in (("ko", ko_dir), ("en", en_dir)):
        wavs = _discover_wavs(directory, max_rounds)
        if not wavs:
            msg = f"No WAV files found in {directory}"
            raise ValueError(msg)
        for wav_path in wavs:
            ground_truth = gt_map.get(wav_path.stem)
            if ground_truth is None:
                skipped.append(wav_path.stem)
                logger.warning(
                    "Skipping %s: no ground-truth entry for stem '%s'.",
                    wav_path.name,
                    wav_path.stem,
                )
                continue
            cases.append(
                AudioCase(
                    lang=lang,
                    wav_path=wav_path,
                    stem=wav_path.stem,
                    ground_truth=ground_truth,
                ),
            )

    if not cases:
        msg = "No WAVs remain after applying the ground-truth mapping."
        raise ValueError(msg)
    return cases, skipped


def _normalize_text(text: str, lang: str) -> str:
    """Normalize text consistently before CER/WER scoring."""
    collapsed = " ".join(text.strip().split())
    if lang == "en":
        return collapsed.lower()
    return collapsed


def _edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    """Compute Levenshtein edit distance."""
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)

    previous_row = list(range(len(hypothesis) + 1))
    for ref_index, ref_item in enumerate(reference, start=1):
        current_row = [ref_index]
        for hyp_index, hyp_item in enumerate(hypothesis, start=1):
            substitution_cost = 0 if ref_item == hyp_item else 1
            current_row.append(
                min(
                    previous_row[hyp_index] + 1,
                    current_row[hyp_index - 1] + 1,
                    previous_row[hyp_index - 1] + substitution_cost,
                ),
            )
        previous_row = current_row
    return previous_row[-1]


def compute_char_error_rate(reference: str, hypothesis: str, lang: str) -> float:
    """Compute CER after language-aware normalization."""
    normalized_reference = _normalize_text(reference, lang)
    normalized_hypothesis = _normalize_text(hypothesis, lang)
    reference_chars = list(normalized_reference)
    hypothesis_chars = list(normalized_hypothesis)
    edits = _edit_distance(reference_chars, hypothesis_chars)
    denominator = max(len(reference_chars), 1)
    return edits / denominator


def compute_word_error_rate(reference: str, hypothesis: str, lang: str) -> float:
    """Compute WER after language-aware normalization."""
    normalized_reference = _normalize_text(reference, lang)
    normalized_hypothesis = _normalize_text(hypothesis, lang)
    reference_words = normalized_reference.split()
    hypothesis_words = normalized_hypothesis.split()
    edits = _edit_distance(reference_words, hypothesis_words)
    denominator = max(len(reference_words), 1)
    return edits / denominator


def _sum_segment_text(segments: Sequence[Any]) -> str:
    """Join STT segments into one plain-text prediction."""
    parts = [str(getattr(segment, "text", "")).strip() for segment in segments]
    return " ".join(part for part in parts if part)


def _create_runtime_context(paths_run: Sequence[str]) -> MeasurementRuntime:
    """Create the shared runtime objects needed for the requested paths."""
    manager_config_cls, manager_cls, model_type_cls, pipeline_config_cls, pipeline_cls = (
        _get_runtime_classes()
    )

    manager = manager_cls(manager_config_cls())
    pipeline = None
    if any(path in {"B", "C"} for path in paths_run):
        manager.initialize()
        pipeline = pipeline_cls(
            manager,
            pipeline_config_cls(
                enable_content_filter=False,
                enable_stt_preload=False,
                play_tts_audio=False,
            ),
        )

    return MeasurementRuntime(
        manager=manager,
        pipeline=pipeline,
        stt_model_type=model_type_cls.STT,
    )


def _require_pipeline(runtime: MeasurementRuntime) -> Any:
    """Return the configured pipeline or raise if the path does not support it."""
    if runtime.pipeline is None:
        msg = "Selected path requires a ConversationPipeline runtime."
        raise RuntimeError(msg)
    return runtime.pipeline


def run_path_a(runtime: MeasurementRuntime, case: AudioCase) -> str:
    """Run Path A: direct STT on the original WAV."""
    run_stt, _read_wav_samples = _get_stt_runner_helpers()
    del _read_wav_samples

    runtime.manager.load(runtime.stt_model_type)
    segments, _info = run_stt(
        runtime.manager.stt,
        case.wav_path,
        language=case.lang,
    )
    return _sum_segment_text(segments)


def run_path_b(
    runtime: MeasurementRuntime,
    case: AudioCase,
    audio_samples: list[float],
    sample_rate: int,
) -> str:
    """Run Path B: prepare audio, VAD, and speech extraction before STT."""
    pipeline = _require_pipeline(runtime)

    # Measurement-only access to private signal-flow helpers keeps production code unchanged.
    pipeline._config.stt_language = case.lang
    prepared_audio = pipeline._prepare_input_audio(audio_samples, sample_rate)
    if not prepared_audio:
        return ""

    segments = pipeline._run_vad(prepared_audio)
    if not segments:
        return ""

    speech_audio = pipeline._extract_speech(prepared_audio, segments)
    if not speech_audio:
        return ""

    runtime.manager.load(runtime.stt_model_type)
    return str(pipeline._run_stt(speech_audio))


@contextmanager
def _measurement_only_path_c_patch(pipeline: Any) -> Iterator[None]:
    """Stop ``run_turn()`` immediately after STT text is available.

    Path C must reuse the full pipeline STT path, including alias normalization, but it must not
    continue into LLM/TTS or write conversation artifacts for this measurement-only workflow.
    """
    from core.pipeline import PipelineState, TurnResult

    original_respond_to_text = pipeline._respond_to_text
    original_init_session_dir = pipeline._init_session_dir
    original_save_conversation_audio = pipeline._save_conversation_audio

    def _measurement_respond_to_text(
        self: Any,
        user_text: str,
        metrics: Any,
        turn_start: float,
        *,
        turn_num: int | None = None,
        input_wav: str | None = None,
    ) -> Any:
        del turn_num, input_wav
        detected_language = self._detect_turn_language(user_text)
        self._last_detected_language = detected_language
        self._state = PipelineState.IDLE
        metrics.total_time_s = time.monotonic() - turn_start
        return TurnResult(
            user_text=user_text,
            response_text="",
            audio_samples=None,
            sample_rate=0,
            metrics=metrics,
            state=self._state,
            detected_language=detected_language,
        )

    def _measurement_init_session_dir(self: Any) -> None:
        del self
        return None

    def _measurement_save_conversation_audio(self: Any, *args: Any, **kwargs: Any) -> None:
        del self, args, kwargs
        return None

    pipeline._respond_to_text = MethodType(_measurement_respond_to_text, pipeline)
    pipeline._init_session_dir = MethodType(_measurement_init_session_dir, pipeline)
    pipeline._save_conversation_audio = MethodType(_measurement_save_conversation_audio, pipeline)
    try:
        yield
    finally:
        pipeline._respond_to_text = original_respond_to_text
        pipeline._init_session_dir = original_init_session_dir
        pipeline._save_conversation_audio = original_save_conversation_audio


def run_path_c(
    runtime: MeasurementRuntime,
    case: AudioCase,
    audio_samples: list[float],
    sample_rate: int,
) -> str:
    """Run Path C: the full audio pipeline, stopped immediately after STT."""
    pipeline = _require_pipeline(runtime)
    pipeline._config.stt_language = case.lang
    with _measurement_only_path_c_patch(pipeline):
        result = pipeline.run_turn(audio_samples, sample_rate=sample_rate)
    return str(result.user_text)


def measure_audio_cases(
    runtime: MeasurementRuntime,
    cases: Sequence[AudioCase],
    paths_run: Sequence[str],
) -> list[AudioMeasurement]:
    """Measure all selected paths for every audio case."""
    _run_stt, read_wav_samples = _get_stt_runner_helpers()
    del _run_stt

    measurements: list[AudioMeasurement] = []
    for case in cases:
        audio_samples, sample_rate, duration_s = read_wav_samples(case.wav_path)
        predictions: dict[str, str] = {}

        if "A" in paths_run:
            predictions["A"] = run_path_a(runtime, case)
        if "B" in paths_run:
            predictions["B"] = run_path_b(runtime, case, audio_samples, sample_rate)
        if "C" in paths_run:
            predictions["C"] = run_path_c(runtime, case, audio_samples, sample_rate)

        measurements.append(
            AudioMeasurement(
                case=case,
                source_sample_rate=sample_rate,
                duration_s=duration_s,
                predictions=predictions,
            ),
        )

    return measurements


def build_summary_rows(
    measurements: Sequence[AudioMeasurement],
    paths_run: Sequence[str],
) -> list[PathSummaryRow]:
    """Aggregate CER/WER by path and language."""
    totals: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {
            "char_edits": 0.0,
            "char_ref": 0.0,
            "word_edits": 0.0,
            "word_ref": 0.0,
            "count": 0.0,
        }
    )

    for measurement in measurements:
        for path_code in paths_run:
            prediction = measurement.predictions.get(path_code)
            if prediction is None:
                continue

            lang = measurement.case.lang
            normalized_gt = _normalize_text(measurement.case.ground_truth, lang)
            normalized_prediction = _normalize_text(prediction, lang)
            char_edits = _edit_distance(list(normalized_gt), list(normalized_prediction))
            word_edits = _edit_distance(normalized_gt.split(), normalized_prediction.split())

            bucket = totals[(path_code, lang)]
            bucket["char_edits"] += char_edits
            bucket["char_ref"] += max(len(normalized_gt), 1)
            bucket["word_edits"] += word_edits
            bucket["word_ref"] += max(len(normalized_gt.split()), 1)
            bucket["count"] += 1

    rows: list[PathSummaryRow] = []
    for path_code in paths_run:
        for lang in ("ko", "en"):
            key = (path_code, lang)
            if key not in totals:
                continue
            bucket = totals[key]
            cer = bucket["char_edits"] / max(bucket["char_ref"], 1.0)
            wer = bucket["word_edits"] / max(bucket["word_ref"], 1.0)
            rows.append(
                PathSummaryRow(
                    path_code=path_code,
                    lang=lang,
                    cer=cer,
                    wer=wer,
                    count=int(bucket["count"]),
                    description=PATH_DESCRIPTIONS[path_code],
                ),
            )
    return rows


def _build_summary_lookup(rows: Sequence[PathSummaryRow]) -> dict[tuple[str, str], PathSummaryRow]:
    """Convert summary rows into a lookup map."""
    return {(row.path_code, row.lang): row for row in rows}


def _compute_delta(
    before: PathSummaryRow | None, after: PathSummaryRow | None
) -> MetricDelta | None:
    """Compute a summary delta when both rows are available."""
    if before is None or after is None:
        return None
    return MetricDelta(cer_delta=after.cer - before.cer, wer_delta=after.wer - before.wer)


def _delta_score(delta: MetricDelta | None) -> float:
    """Return the strongest absolute delta component."""
    if delta is None:
        return 0.0
    return max(abs(delta.cer_delta), abs(delta.wer_delta))


def _mean_a_to_b_case_delta(measurements: Sequence[AudioMeasurement], *, resampled: bool) -> float:
    """Compute mean per-case A->B delta for either native or resampled inputs."""
    deltas: list[float] = []
    for measurement in measurements:
        if measurement.resampled_for_pipeline is not resampled:
            continue
        prediction_a = measurement.predictions.get("A")
        prediction_b = measurement.predictions.get("B")
        if prediction_a is None or prediction_b is None:
            continue
        cer_a = compute_char_error_rate(
            measurement.case.ground_truth,
            prediction_a,
            measurement.case.lang,
        )
        cer_b = compute_char_error_rate(
            measurement.case.ground_truth,
            prediction_b,
            measurement.case.lang,
        )
        deltas.append(cer_b - cer_a)
    if not deltas:
        return 0.0
    return sum(deltas) / len(deltas)


def build_gap_analysis(
    measurements: Sequence[AudioMeasurement],
    rows: Sequence[PathSummaryRow],
    paths_run: Sequence[str],
) -> GapAnalysis:
    """Interpret the measured path deltas and generate report-friendly comments."""
    summary_lookup = _build_summary_lookup(rows)
    ab_by_lang = {
        lang: delta
        for lang in ("ko", "en")
        if (
            delta := _compute_delta(
                summary_lookup.get(("A", lang)), summary_lookup.get(("B", lang))
            )
        )
        is not None
    }
    bc_by_lang = {
        lang: delta
        for lang in ("ko", "en")
        if (
            delta := _compute_delta(
                summary_lookup.get(("B", lang)), summary_lookup.get(("C", lang))
            )
        )
        is not None
    }

    if not {"A", "B", "C"}.issubset(set(paths_run)):
        dominant_cause = "not enough paths selected"
        interpretation = "Dominant cause: unavailable because not all A/B/C paths were run."
        hypothesis_a = "not-evaluated"
        hypothesis_c = "not-evaluated"
        other = "Run with --paths ABC to localize VAD/resample versus alias-normalization effects."
        return GapAnalysis(
            ab_by_lang=ab_by_lang,
            bc_by_lang=bc_by_lang,
            dominant_cause=dominant_cause,
            interpretation=interpretation,
            hypothesis_a=hypothesis_a,
            hypothesis_c=hypothesis_c,
            other=other,
        )

    ab_score = max((_delta_score(delta) for delta in ab_by_lang.values()), default=0.0)
    bc_score = max((_delta_score(delta) for delta in bc_by_lang.values()), default=0.0)
    largest_score = max(ab_score, bc_score)

    if largest_score < DOMINANT_DELTA_THRESHOLD:
        dominant_cause = "unknown - deltas small"
    elif ab_score > bc_score:
        native_delta = _mean_a_to_b_case_delta(measurements, resampled=False)
        resampled_delta = _mean_a_to_b_case_delta(measurements, resampled=True)
        if resampled_delta > native_delta + DOMINANT_DELTA_THRESHOLD:
            dominant_cause = "resample artifact"
        else:
            dominant_cause = "VAD boundary loss"
    elif bc_score > ab_score:
        dominant_cause = "alias normalization"
    else:
        dominant_cause = "unknown - deltas small"

    interpretation = f"Dominant cause: {dominant_cause}"
    hypothesis_a = "supported" if dominant_cause == "VAD boundary loss" else "not-supported"
    hypothesis_c = "supported" if dominant_cause == "resample artifact" else "not-supported"

    if dominant_cause == "alias normalization":
        other = "Downstream alias normalization appears to contribute more than VAD/resampling."
    elif dominant_cause == "unknown - deltas small":
        other = (
            "If neither A nor B path causes the gap, investigation focus moves to downstream "
            "normalization or run_turn ordering."
        )
    elif dominant_cause == "not enough paths selected":
        other = "Selected paths were insufficient to classify the dominant cause."
    else:
        other = "Focus next on the VAD/resample boundary before downstream pipeline stages."

    return GapAnalysis(
        ab_by_lang=ab_by_lang,
        bc_by_lang=bc_by_lang,
        dominant_cause=dominant_cause,
        interpretation=interpretation,
        hypothesis_a=hypothesis_a,
        hypothesis_c=hypothesis_c,
        other=other,
    )


def _format_metric(value: float | None) -> str:
    """Format a metric or delta value for markdown."""
    if value is None:
        return "N/A"
    return f"{value:.4f}"


def _metric_delta_value(
    delta_map: dict[str, MetricDelta],
    lang: str,
    field_name: str,
) -> float | None:
    """Return one delta field for a language, or ``None`` when unavailable."""
    delta = delta_map.get(lang)
    if delta is None:
        return None
    if field_name == "cer":
        return delta.cer_delta
    if field_name == "wer":
        return delta.wer_delta
    msg = f"Unsupported delta field: {field_name}"
    raise ValueError(msg)


def _markdown_escape(text: str) -> str:
    """Escape text for use inside markdown table cells."""
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")


def render_report(
    args: argparse.Namespace,
    measurements: Sequence[AudioMeasurement],
    rows: Sequence[PathSummaryRow],
    analysis: GapAnalysis,
    skipped_stems: Sequence[str],
    paths_run: Sequence[str],
) -> str:
    """Render the final markdown report."""
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    ko_count = sum(1 for measurement in measurements if measurement.case.lang == "ko")
    en_count = sum(1 for measurement in measurements if measurement.case.lang == "en")
    lines: list[str] = [f"# STT Accuracy Gap Measurement - {timestamp}", ""]

    lines.extend(
        [
            "## Configuration",
            f"- KO dir: {args.ko_dir.resolve()}",
            f"- EN dir: {args.en_dir.resolve()}",
            f"- GT file: {args.gt_json.resolve()}",
            f"- Paths run: {', '.join(paths_run)}",
            (
                f"- Total rounds: {len(measurements)} "
                f"(KO {ko_count}, EN {en_count}, skipped {len(skipped_stems)})"
            ),
            "",
            "## Summary",
            "| Path | Lang | CER | WER | Description |",
            "|---|---|---:|---:|---|",
        ],
    )
    for row in rows:
        lines.append(
            f"| {row.path_code} | {row.lang.upper()} | {row.cer:.4f} | {row.wer:.4f} | "
            f"{row.description} |"
        )

    lines.extend(
        [
            "",
            "## Gap analysis",
            (
                "- A->B delta (VAD + resample effect): "
                f"KO dCER={_format_metric(_metric_delta_value(analysis.ab_by_lang, 'ko', 'cer'))}, "
                f"dWER={_format_metric(_metric_delta_value(analysis.ab_by_lang, 'ko', 'wer'))}; "
                f"EN dCER={_format_metric(_metric_delta_value(analysis.ab_by_lang, 'en', 'cer'))}, "
                f"dWER={_format_metric(_metric_delta_value(analysis.ab_by_lang, 'en', 'wer'))}"
            ),
            (
                "- B->C delta (alias normalization effect): "
                f"KO dCER={_format_metric(_metric_delta_value(analysis.bc_by_lang, 'ko', 'cer'))}, "
                f"dWER={_format_metric(_metric_delta_value(analysis.bc_by_lang, 'ko', 'wer'))}; "
                f"EN dCER={_format_metric(_metric_delta_value(analysis.bc_by_lang, 'en', 'cer'))}, "
                f"dWER={_format_metric(_metric_delta_value(analysis.bc_by_lang, 'en', 'wer'))}"
            ),
            f"- Interpretation comment: {analysis.interpretation}",
            "",
            "## Per-round diff",
            "| Stem | Lang | Hz | GT | A | B | C | Notes |",
            "|---|---|---:|---|---|---|---|---|",
        ],
    )

    for measurement in measurements:
        note_parts = [
            "resampled" if measurement.resampled_for_pipeline else "native 16 kHz",
            f"{measurement.duration_s:.2f}s",
        ]
        lines.append(
            f"| {_markdown_escape(measurement.case.stem)} | {measurement.case.lang.upper()} | "
            f"{measurement.source_sample_rate} | {_markdown_escape(measurement.case.ground_truth)} | "
            f"{_markdown_escape(measurement.predictions.get('A', '-'))} | "
            f"{_markdown_escape(measurement.predictions.get('B', '-'))} | "
            f"{_markdown_escape(measurement.predictions.get('C', '-'))} | "
            f"{_markdown_escape(', '.join(note_parts))} |"
        )

    lines.extend(
        [
            "",
            "## Hypotheses map",
            f"- Hypothesis A (VAD boundary loss): {analysis.hypothesis_a}",
            f"- Hypothesis C (resample quality): {analysis.hypothesis_c}",
            f"- Other: {analysis.other}",
        ],
    )

    if skipped_stems:
        lines.extend(
            [
                "",
                "## Skipped inputs",
                f"- Missing ground truth: {', '.join(skipped_stems)}",
            ],
        )

    return "\n".join(lines) + "\n"


def write_report(path: Path, report_text: str) -> None:
    """Write the markdown report to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report_text, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the full STT accuracy-gap measurement workflow."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = build_parser()
    args = parser.parse_args(argv)
    paths_run = parse_selected_paths(args.paths)

    runtime: MeasurementRuntime | None = None
    try:
        _run_preflight(skip=bool(args.skip_jetson_preflight))
        gt_map = load_ground_truth_map(args.gt_json)
        cases, skipped_stems = collect_audio_cases(
            args.ko_dir, args.en_dir, gt_map, args.max_rounds
        )
        runtime = _create_runtime_context(paths_run)
        measurements = measure_audio_cases(runtime, cases, paths_run)
        summary_rows = build_summary_rows(measurements, paths_run)
        gap_analysis = build_gap_analysis(measurements, summary_rows, paths_run)
        report_text = render_report(
            args=args,
            measurements=measurements,
            rows=summary_rows,
            analysis=gap_analysis,
            skipped_stems=skipped_stems,
            paths_run=paths_run,
        )
        write_report(args.output, report_text)
        logger.info("Wrote STT accuracy-gap report to %s", args.output)
        return 0
    except Exception as exc:
        logger.error("STT accuracy-gap measurement failed: %s", exc, exc_info=True)
        return 1
    finally:
        if runtime is not None:
            try:
                runtime.manager.unload_all(force=True)
            except Exception:
                logger.warning("Failed to unload models after measurement", exc_info=True)


if __name__ == "__main__":
    raise SystemExit(main())
