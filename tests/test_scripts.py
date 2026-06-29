"""Tests for VAD test script and benchmark framework.

Tests script imports, argparse setup, and utility functions without
requiring actual model files or torch installation.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

# Ensure repo root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------


def test_test_vad_importable() -> None:
    """Verify test_vad module can be imported without errors."""
    from scripts import test_vad  # noqa: F401

    assert hasattr(test_vad, "main")
    assert hasattr(test_vad, "build_parser")
    assert hasattr(test_vad, "read_wav_mono_16k")
    assert hasattr(test_vad, "run_vad")
    assert hasattr(test_vad, "load_vad_model")


def test_bench_model_importable() -> None:
    """Verify bench_model module can be imported without errors."""
    from scripts import bench_model  # noqa: F401

    assert hasattr(bench_model, "main")
    assert hasattr(bench_model, "build_parser")
    assert hasattr(bench_model, "BENCHMARK_REGISTRY")
    assert hasattr(bench_model, "read_proc_memory")
    assert hasattr(bench_model, "parse_tegrastats_line")


# ---------------------------------------------------------------------------
# Argparse tests
# ---------------------------------------------------------------------------


def test_vad_parser_help(capsys: object) -> None:
    """Verify VAD test script parser is configured correctly."""
    from scripts.test_vad import build_parser

    parser = build_parser()
    # Check that essential arguments are present
    actions = {a.dest for a in parser._actions}
    assert "wav_path" in actions
    assert "model_path" in actions
    assert "threshold" in actions
    assert "min_speech_ms" in actions
    assert "min_silence_ms" in actions


def test_bench_parser_help() -> None:
    """Verify bench_model parser is configured correctly."""
    from scripts.bench_model import build_parser

    parser = build_parser()
    actions = {a.dest for a in parser._actions}
    assert "model_type" in actions
    assert "input_path" in actions
    assert "model_path" in actions
    assert "tegrastats_log" in actions
    assert "output" in actions


def test_vad_parser_defaults() -> None:
    """Verify default values for VAD parser arguments."""
    from scripts.test_vad import (
        DEFAULT_THRESHOLD,
        MIN_SILENCE_DURATION_MS,
        MIN_SPEECH_DURATION_MS,
        build_parser,
    )

    parser = build_parser()
    args = parser.parse_args(["test.wav"])
    assert args.wav_path == Path("test.wav")
    assert args.model_path is None
    assert args.threshold == DEFAULT_THRESHOLD
    assert args.min_speech_ms == MIN_SPEECH_DURATION_MS
    assert args.min_silence_ms == MIN_SILENCE_DURATION_MS


def test_vad_parser_custom_args() -> None:
    """Verify custom argument values are parsed correctly."""
    from scripts.test_vad import build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "audio.wav",
            "--model-path",
            "/opt/mungi/ai_models/silero_vad.jit",
            "--threshold",
            "0.7",
            "--min-speech-ms",
            "500",
            "--min-silence-ms",
            "200",
        ]
    )
    assert args.wav_path == Path("audio.wav")
    assert args.model_path == "/opt/mungi/ai_models/silero_vad.jit"
    assert args.threshold == 0.7
    assert args.min_speech_ms == 500
    assert args.min_silence_ms == 200


def test_bench_parser_vad_subcommand() -> None:
    """Verify bench_model parser handles VAD subcommand."""
    from scripts.bench_model import build_parser

    parser = build_parser()
    args = parser.parse_args(["vad", "test.wav"])
    assert args.model_type == "vad"
    assert args.input_path == "test.wav"


def test_bench_parser_stt_subcommand() -> None:
    """Verify bench_model parser handles STT subcommand."""
    from scripts.bench_model import build_parser

    parser = build_parser()
    args = parser.parse_args(["stt", "test.wav"])
    assert args.model_type == "stt"


def test_bench_parser_llm_subcommand() -> None:
    """Verify bench_model parser handles LLM subcommand."""
    from scripts.bench_model import build_parser

    parser = build_parser()
    args = parser.parse_args(["llm", "--prompt", "Hello world"])
    assert args.model_type == "llm"
    assert args.prompt == "Hello world"


def test_bench_parser_tts_subcommand() -> None:
    """Verify bench_model parser handles TTS subcommand."""
    from scripts.bench_model import build_parser

    parser = build_parser()
    args = parser.parse_args(["tts", "--text", "Hello world"])
    assert args.model_type == "tts"
    assert args.text == "Hello world"


# ---------------------------------------------------------------------------
# Data class tests
# ---------------------------------------------------------------------------


def test_speech_segment_dataclass() -> None:
    """Test SpeechSegment data class methods."""
    from models.vad_runner import SpeechSegment

    seg = SpeechSegment(start=1.0, end=2.5)
    assert seg.duration_ms() == 1500.0

    d = seg.to_dict()
    assert d == {"start": 1.0, "end": 2.5}


def test_vad_result_dataclass() -> None:
    """Test VADResult data class serialization."""
    from scripts.test_vad import VADResult

    result = VADResult(
        segments=[{"start": 0.0, "end": 1.0}],
        model_load_time_s=0.5,
        inference_time_s=0.2,
        peak_memory_kb=1024,
        audio_duration_s=3.0,
        total_speech_s=1.0,
    )
    d = result.to_dict()
    assert d["model_load_time_s"] == 0.5
    assert d["inference_time_s"] == 0.2
    assert d["peak_memory_kb"] == 1024
    assert len(d["segments"]) == 1


def test_benchmark_result_dataclass() -> None:
    """Test BenchmarkResult data class serialization."""
    from scripts.bench_model import BenchmarkResult

    result = BenchmarkResult(
        model_type="vad",
        model_path="/opt/mungi/ai_models/silero_vad.jit",
        load_time_s=1.0,
        inference_time_s=0.5,
    )
    d = result.to_dict()
    assert d["model_type"] == "vad"
    assert d["load_time_s"] == 1.0
    assert d["error"] is None


def test_memory_snapshot_dataclass() -> None:
    """Test MemorySnapshot data class."""
    from scripts.bench_model import MemorySnapshot

    snap = MemorySnapshot(rss_kb=1024, vm_peak_kb=2048, gpu_used_mb=256)
    d = snap.to_dict()
    assert d["rss_kb"] == 1024
    assert d["vm_peak_kb"] == 2048
    assert d["gpu_used_mb"] == 256


# ---------------------------------------------------------------------------
# Utility function tests
# ---------------------------------------------------------------------------


def test_read_proc_memory_returns_snapshot() -> None:
    """Verify read_proc_memory returns a valid MemorySnapshot."""
    from scripts.bench_model import MemorySnapshot, read_proc_memory

    snap = read_proc_memory()
    assert isinstance(snap, MemorySnapshot)
    # On Windows, values should be 0 (no /proc)
    # On Linux, values should be non-negative
    assert snap.rss_kb >= 0
    assert snap.vm_peak_kb >= 0


def test_get_peak_memory_kb_returns_int() -> None:
    """Verify get_peak_memory_kb returns a non-negative integer."""
    from scripts.test_vad import get_peak_memory_kb

    result = get_peak_memory_kb()
    assert isinstance(result, int)
    assert result >= 0


def test_parse_tegrastats_line_with_ram() -> None:
    """Test tegrastats line parsing with RAM data."""
    from scripts.bench_model import parse_tegrastats_line

    line = "RAM 3456/7620MB (lfb 23x4MB) SWAP 128/3810MB GR3D_FREQ 45%"
    parsed = parse_tegrastats_line(line)

    assert parsed["ram_used_mb"] == 3456
    assert parsed["ram_total_mb"] == 7620
    assert parsed["swap_used_mb"] == 128
    assert parsed["swap_total_mb"] == 3810
    assert parsed["gr3d_freq_pct"] == 45


def test_parse_tegrastats_line_partial() -> None:
    """Test tegrastats line parsing with partial data."""
    from scripts.bench_model import parse_tegrastats_line

    line = "RAM 2048/7620MB"
    parsed = parse_tegrastats_line(line)

    assert parsed["ram_used_mb"] == 2048
    assert parsed["ram_total_mb"] == 7620
    assert "swap_used_mb" not in parsed
    assert "gr3d_freq_pct" not in parsed


def test_parse_tegrastats_line_temperatures() -> None:
    """CPU and GPU temperatures should be parsed when present."""
    from scripts.bench_model import parse_tegrastats_line

    line = "cpu@59.093C soc0@58.218C gpu@60.687C GR3D_FREQ 69%"
    parsed = parse_tegrastats_line(line)

    assert parsed["cpu_temp_c"] == 59.093
    assert parsed["gpu_temp_c"] == 60.687
    assert parsed["gr3d_freq_pct"] == 69


def test_parse_tegrastats_line_empty() -> None:
    """Test tegrastats line parsing with no matching data."""
    from scripts.bench_model import parse_tegrastats_line

    parsed = parse_tegrastats_line("some random text")
    assert parsed == {}


def test_parse_tegrastats_log_file_not_found() -> None:
    """Test tegrastats log parser raises on missing file."""
    import pytest

    from scripts.bench_model import parse_tegrastats_log

    with pytest.raises(FileNotFoundError):
        parse_tegrastats_log(Path("/nonexistent/tegrastats.log"))


def test_parse_tegrastats_log_valid(tmp_path: Path) -> None:
    """Test tegrastats log parsing with a valid temporary file."""
    from scripts.bench_model import parse_tegrastats_log

    log_content = (
        "RAM 3000/7620MB (lfb 23x4MB) SWAP 0/3810MB GR3D_FREQ 0%\n"
        "RAM 3200/7620MB (lfb 22x4MB) SWAP 0/3810MB GR3D_FREQ 50%\n"
        "\n"
        "RAM 3400/7620MB (lfb 21x4MB) SWAP 10/3810MB GR3D_FREQ 75%\n"
    )
    log_file = tmp_path / "tegrastats.log"
    log_file.write_text(log_content, encoding="utf-8")

    snapshots = parse_tegrastats_log(log_file)
    assert len(snapshots) == 3
    assert snapshots[0]["ram_used_mb"] == 3000
    assert snapshots[1]["gr3d_freq_pct"] == 50
    assert snapshots[2]["swap_used_mb"] == 10


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


def test_benchmark_registry_contains_all_types() -> None:
    """Verify all supported model types are in the registry."""
    from scripts.bench_model import BENCHMARK_REGISTRY, SUPPORTED_MODEL_TYPES

    for model_type in SUPPORTED_MODEL_TYPES:
        assert model_type in BENCHMARK_REGISTRY


def test_benchmark_registry_instances() -> None:
    """Verify registry entries are proper ModelBenchmark subclasses."""
    from scripts.bench_model import BENCHMARK_REGISTRY, ModelBenchmark

    for name, cls in BENCHMARK_REGISTRY.items():
        instance = cls()
        assert isinstance(instance, ModelBenchmark)
        assert instance.model_type() == name


# ---------------------------------------------------------------------------
# Stub benchmark tests
# ---------------------------------------------------------------------------


def test_stt_benchmark_stub() -> None:
    """Verify STT benchmark returns error when no input file given."""
    from scripts.bench_model import STTBenchmark, build_parser

    parser = build_parser()
    args = parser.parse_args(["stt"])

    bench = STTBenchmark()
    result = bench.run(args)
    assert result.error is not None
    assert "input" in result.error.lower() or "WAV" in result.error


def test_llm_benchmark_no_model(tmp_path: Any) -> None:
    """Verify LLM benchmark returns error when no model found."""
    from scripts.bench_model import LLMBenchmark, build_parser

    parser = build_parser()
    args = parser.parse_args(["llm"])

    with patch("scripts.test_llm.DEFAULT_MODEL_DIR", str(tmp_path / "empty")):
        bench = LLMBenchmark()
        result = bench.run(args)
    assert result.error is not None
    assert "No GGUF model found" in result.error


def test_tts_benchmark_no_model(tmp_path: Any) -> None:
    """Verify TTS benchmark returns error when model dir missing."""
    from scripts.bench_model import TTSBenchmark, build_parser

    parser = build_parser()
    args = parser.parse_args(["tts"])

    with patch(
        "scripts.test_tts.DEFAULT_SUPERTONIC_MODEL_DIR",
        str(tmp_path / "empty"),
    ):
        bench = TTSBenchmark()
        result = bench.run(args)
    assert result.error is not None


# ---------------------------------------------------------------------------
# WAV reader validation tests
# ---------------------------------------------------------------------------


def test_read_wav_file_not_found() -> None:
    """Verify FileNotFoundError for missing WAV files."""
    import pytest

    from scripts.test_vad import read_wav_mono_16k

    with pytest.raises(FileNotFoundError):
        read_wav_mono_16k(Path("/nonexistent/audio.wav"))


# ---------------------------------------------------------------------------
# Constants tests
# ---------------------------------------------------------------------------


def test_vad_constants() -> None:
    """Verify VAD module constants have expected values."""
    from models.vad_runner import (
        DEFAULT_THRESHOLD,
        MIN_SILENCE_DURATION_MS,
        MIN_SPEECH_DURATION_MS,
        SAMPLE_RATE,
        WINDOW_SIZE_SAMPLES,
    )

    assert SAMPLE_RATE == 16000
    assert WINDOW_SIZE_SAMPLES == 512
    assert 0.0 < DEFAULT_THRESHOLD < 1.0
    assert MIN_SPEECH_DURATION_MS > 0
    assert MIN_SILENCE_DURATION_MS > 0


def test_bench_constants() -> None:
    """Verify bench_model module constants have expected values."""
    from scripts.bench_model import DEFAULT_MODEL_DIR, SUPPORTED_MODEL_TYPES

    assert DEFAULT_MODEL_DIR == "/opt/mungi/ai_models"
    assert "vad" in SUPPORTED_MODEL_TYPES
    assert "stt" in SUPPORTED_MODEL_TYPES
    assert "llm" in SUPPORTED_MODEL_TYPES
    assert "tts" in SUPPORTED_MODEL_TYPES
