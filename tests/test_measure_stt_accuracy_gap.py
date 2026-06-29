"""Tests for the STT accuracy-gap measurement script."""

from __future__ import annotations

import json
import struct
import sys
import wave
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    from scripts.measure_stt_accuracy_gap import AudioCase

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _write_test_wav(
    path: Path,
    *,
    sample_rate: int = 16000,
    duration_frames: int = 1600,
) -> None:
    """Write a tiny mono PCM WAV for testing."""
    samples = [0] * duration_frames
    raw_frames = struct.pack(f"<{len(samples)}h", *samples)

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(raw_frames)


def _build_case(tmp_path: Path, *, lang: str, stem: str, text: str) -> tuple[Path, AudioCase]:
    """Create one temporary WAV plus the corresponding AudioCase."""
    from scripts import measure_stt_accuracy_gap as gap

    wav_path = tmp_path / f"{stem}.wav"
    _write_test_wav(wav_path)
    case = gap.AudioCase(lang=lang, wav_path=wav_path, stem=stem, ground_truth=text)
    return wav_path, case


def test_parser_accepts_required_args_and_options() -> None:
    """The CLI parser accepts all required arguments and optional flags."""
    from scripts import measure_stt_accuracy_gap as gap

    parser = gap.build_parser()
    args = parser.parse_args(
        [
            "--ko-dir",
            "ko",
            "--en-dir",
            "en",
            "--gt-json",
            "gt.json",
            "--output",
            "report.md",
            "--max-rounds",
            "3",
            "--paths",
            "B",
            "--skip-jetson-preflight",
        ],
    )

    assert args.ko_dir == Path("ko")
    assert args.en_dir == Path("en")
    assert args.gt_json == Path("gt.json")
    assert args.output == Path("report.md")
    assert args.max_rounds == 3
    assert args.paths == "B"
    assert args.skip_jetson_preflight is True


def test_parser_missing_required_args_exits_with_code_2() -> None:
    """Missing required CLI args must raise SystemExit(2)."""
    from scripts import measure_stt_accuracy_gap as gap

    parser = gap.build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args([])

    assert excinfo.value.code == 2


def test_run_path_a_isolation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Path A must bypass VAD, speech extraction, and alias normalization."""
    from scripts import measure_stt_accuracy_gap as gap

    _wav_path, case = _build_case(tmp_path, lang="ko", stem="ko_round_1", text="hello")
    pipeline = MagicMock()
    manager = MagicMock()
    manager.stt = object()
    runtime = gap.MeasurementRuntime(manager=manager, pipeline=pipeline, stt_model_type="STT")

    run_stt_mock = MagicMock(return_value=([SimpleNamespace(text="path-a text")], {}))
    monkeypatch.setattr(
        gap,
        "_get_stt_runner_helpers",
        lambda: (run_stt_mock, lambda _path: ([0.0], 16000, 0.1)),
    )

    result = gap.run_path_a(runtime, case)

    assert result == "path-a text"
    manager.load.assert_called_once_with("STT")
    pipeline._run_vad.assert_not_called()
    pipeline._extract_speech.assert_not_called()
    pipeline._normalize_stt_text.assert_not_called()


def test_run_path_b_isolation() -> None:
    """Path B must use VAD and speech extraction but must not normalize aliases."""
    from scripts import measure_stt_accuracy_gap as gap

    case = gap.AudioCase(
        lang="en",
        wav_path=Path("dummy.wav"),
        stem="en_round_1",
        ground_truth="hello there",
    )
    pipeline = MagicMock()
    pipeline._config = SimpleNamespace(stt_language="ko")
    pipeline._prepare_input_audio.return_value = [0.1, 0.2, 0.3]
    pipeline._run_vad.return_value = [SimpleNamespace(start=0.0, end=0.1)]
    pipeline._extract_speech.return_value = [0.2, 0.3]
    pipeline._run_stt.return_value = "path-b text"

    manager = MagicMock()
    runtime = gap.MeasurementRuntime(manager=manager, pipeline=pipeline, stt_model_type="STT")

    result = gap.run_path_b(runtime, case, [0.1, 0.2], 22050)

    assert result == "path-b text"
    manager.load.assert_called_once_with("STT")
    pipeline._prepare_input_audio.assert_called_once_with([0.1, 0.2], 22050)
    pipeline._run_vad.assert_called_once()
    pipeline._extract_speech.assert_called_once()
    pipeline._run_stt.assert_called_once_with([0.2, 0.3])
    pipeline._normalize_stt_text.assert_not_called()
    assert pipeline._config.stt_language == "en"


def test_run_path_c_full_path_calls_alias_normalization() -> None:
    """Path C must run through full pipeline STT flow and call alias normalization."""
    from core.pipeline import ConversationPipeline, PipelineConfig
    from scripts import measure_stt_accuracy_gap as gap

    manager = MagicMock()
    manager.guard_stt_resident_memory.return_value = True
    pipeline = ConversationPipeline(
        manager,
        PipelineConfig(enable_content_filter=False, play_tts_audio=False),
    )
    case = gap.AudioCase(
        lang="ko",
        wav_path=Path("dummy.wav"),
        stem="ko_round_1",
        ground_truth="unused",
    )
    runtime = gap.MeasurementRuntime(manager=manager, pipeline=pipeline, stt_model_type="STT")

    with (
        patch.object(
            pipeline,
            "_run_vad",
            MagicMock(return_value=[SimpleNamespace(start=0.0, end=0.1)]),
        ),
        patch.object(pipeline, "_extract_speech", MagicMock(return_value=[0.1] * 100)),
        patch.object(pipeline, "_run_stt", MagicMock(return_value="raw stt text")),
        patch.object(pipeline, "_normalize_stt_text", return_value="normalized text") as mock_norm,
    ):
        result = gap.run_path_c(runtime, case, [0.0] * 1600, 16000)

    assert result == "normalized text"
    assert mock_norm.called


def test_cer_wer_numerical_correctness() -> None:
    """CER/WER helpers should match known edit-distance values."""
    from scripts import measure_stt_accuracy_gap as gap

    cer = gap.compute_char_error_rate("abcd", "abxd", "en")
    wer = gap.compute_word_error_rate("one two three", "one four three", "en")

    assert cer == pytest.approx(0.25, abs=1e-4)
    assert wer == pytest.approx(1.0 / 3.0, abs=1e-4)


def test_identical_inputs_report_unknown_deltas_small() -> None:
    """Identical A/B/C predictions should yield the 'unknown - deltas small' comment."""
    from scripts import measure_stt_accuracy_gap as gap

    measurements = [
        gap.AudioMeasurement(
            case=gap.AudioCase(
                lang="ko",
                wav_path=Path("ko.wav"),
                stem="ko_round_1",
                ground_truth="same text",
            ),
            source_sample_rate=16000,
            duration_s=0.1,
            predictions={"A": "same text", "B": "same text", "C": "same text"},
        ),
        gap.AudioMeasurement(
            case=gap.AudioCase(
                lang="en",
                wav_path=Path("en.wav"),
                stem="en_round_1",
                ground_truth="same words",
            ),
            source_sample_rate=16000,
            duration_s=0.1,
            predictions={"A": "same words", "B": "same words", "C": "same words"},
        ),
    ]

    rows = gap.build_summary_rows(measurements, ("A", "B", "C"))
    analysis = gap.build_gap_analysis(measurements, rows, ("A", "B", "C"))

    assert analysis.dominant_cause == "unknown - deltas small"


def test_collect_audio_cases_skips_missing_ground_truth(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A missing GT entry should warn and skip without crashing."""
    from scripts import measure_stt_accuracy_gap as gap

    ko_dir = tmp_path / "ko"
    en_dir = tmp_path / "en"
    ko_dir.mkdir()
    en_dir.mkdir()

    _write_test_wav(ko_dir / "ko_keep.wav")
    _write_test_wav(en_dir / "en_missing.wav")

    gt_map = {"ko_keep": "hello there"}

    cases, skipped = gap.collect_audio_cases(ko_dir, en_dir, gt_map, max_rounds=0)

    assert len(cases) == 1
    assert cases[0].stem == "ko_keep"
    assert skipped == ["en_missing"]
    assert "no ground-truth entry" in caplog.text


def test_report_generation_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The script should generate a markdown report end-to-end with mocked path outputs."""
    from scripts import measure_stt_accuracy_gap as gap

    ko_dir = tmp_path / "ko"
    en_dir = tmp_path / "en"
    ko_dir.mkdir()
    en_dir.mkdir()

    for stem in ("ko_round_1", "ko_round_2"):
        _write_test_wav(ko_dir / f"{stem}.wav")
    for stem in ("en_round_1", "en_round_2"):
        _write_test_wav(en_dir / f"{stem}.wav")

    gt_map = {
        "ko_round_1": "ko gt 1",
        "ko_round_2": "ko gt 2",
        "en_round_1": "en gt 1",
        "en_round_2": "en gt 2",
    }
    gt_path = tmp_path / "gt.json"
    gt_path.write_text(json.dumps(gt_map), encoding="utf-8")
    output_path = tmp_path / "report.md"

    manager = MagicMock()
    monkeypatch.setattr(gap, "_run_preflight", lambda skip: None)
    monkeypatch.setattr(
        gap,
        "_create_runtime_context",
        lambda _paths: gap.MeasurementRuntime(
            manager=manager,
            pipeline=None,
            stt_model_type="STT",
        ),
    )
    monkeypatch.setattr(
        gap,
        "run_path_a",
        lambda runtime, case: f"{case.lang}-a",
    )
    monkeypatch.setattr(
        gap,
        "run_path_b",
        lambda runtime, case, audio_samples, sample_rate: f"{case.lang}-b-{sample_rate}",
    )
    monkeypatch.setattr(
        gap,
        "run_path_c",
        lambda runtime, case, audio_samples, sample_rate: f"{case.lang}-c",
    )

    exit_code = gap.main(
        [
            "--ko-dir",
            str(ko_dir),
            "--en-dir",
            str(en_dir),
            "--gt-json",
            str(gt_path),
            "--output",
            str(output_path),
            "--skip-jetson-preflight",
        ],
    )

    report_text = output_path.read_text(encoding="utf-8")
    assert exit_code == 0
    assert output_path.exists()
    assert "# STT Accuracy Gap Measurement" in report_text
    assert "## Configuration" in report_text
    assert "## Summary" in report_text
    assert "## Gap analysis" in report_text
    assert "## Per-round diff" in report_text
    assert "## Hypotheses map" in report_text
