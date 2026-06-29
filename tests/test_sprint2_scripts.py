"""Tests for Sprint 2 inference scripts (STT, LLM, TTS) and bench_model updates.

Tests script imports, argparse setup, dataclasses, and utility functions
without requiring actual model files or Jetson-only packages
(sherpa_onnx, llama_cpp, supertonic, torch).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure repo root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Check numpy availability for TTS write_wav tests
try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# ===================================================================
# test_stt.py tests
# ===================================================================


# --- Constants ---


def test_stt_constants_exist() -> None:
    """Verify test_stt module exposes expected constants."""
    from scripts.test_stt import (
        DEFAULT_BEAM_SIZE,
        DEFAULT_COMPUTE_TYPE,
        DEFAULT_DEVICE,
        DEFAULT_LANGUAGE,
        DEFAULT_MODEL_DIR,
        DEFAULT_MODEL_SIZE,
        SAMPLE_RATE,
    )

    assert SAMPLE_RATE == 16000
    assert isinstance(DEFAULT_MODEL_SIZE, str)
    assert isinstance(DEFAULT_DEVICE, str)
    assert isinstance(DEFAULT_COMPUTE_TYPE, str)
    assert isinstance(DEFAULT_MODEL_DIR, str)
    assert isinstance(DEFAULT_LANGUAGE, str)
    assert isinstance(DEFAULT_BEAM_SIZE, int)
    assert DEFAULT_BEAM_SIZE > 0


def test_stt_constants_values() -> None:
    """Verify test_stt constants have sensible defaults."""
    from scripts.test_stt import (
        DEFAULT_BEAM_SIZE,
        DEFAULT_COMPUTE_TYPE,
        DEFAULT_DEVICE,
        DEFAULT_LANGUAGE,
        DEFAULT_MODEL_DIR,
        DEFAULT_MODEL_SIZE,
    )

    assert DEFAULT_MODEL_SIZE == "small"
    assert DEFAULT_DEVICE in ("cuda", "cpu")
    assert DEFAULT_COMPUTE_TYPE in (
        "float16",
        "int8",
        "float32",
    )
    assert DEFAULT_MODEL_DIR == "/opt/mungi/ai_models"
    assert DEFAULT_LANGUAGE == "ko"
    assert DEFAULT_BEAM_SIZE == 5


# --- Dataclasses ---


def test_stt_result_dataclass() -> None:
    """Verify STTResult dataclass creation and serialization."""
    from scripts.test_stt import STTResult

    result = STTResult(
        segments=[{"start": 0.0, "end": 1.5, "text": "hello"}],
        full_text="hello",
        model_load_time_s=1.23,
        inference_time_s=0.45,
        peak_memory_kb=2048,
        audio_duration_s=3.0,
        rtf=0.15,
        detected_language="ko",
        language_probability=0.95,
        model_size="small",
        device="cuda",
        compute_type="float16",
    )
    d = result.to_dict()
    assert d["full_text"] == "hello"
    assert d["model_load_time_s"] == 1.23
    assert d["inference_time_s"] == 0.45
    assert d["peak_memory_kb"] == 2048
    assert d["rtf"] == 0.15
    assert d["detected_language"] == "ko"
    assert len(d["segments"]) == 1


def test_transcription_segment_dataclass() -> None:
    """Verify TranscriptionSegment dataclass methods."""
    from models.stt_runner import TranscriptionSegment

    seg = TranscriptionSegment(start=1.0, end=3.5, text="test")
    assert seg.duration_s() == 2.5
    d = seg.to_dict()
    assert d == {"start": 1.0, "end": 3.5, "text": "test"}


# --- Audio validation ---


def test_stt_validate_wav_file_missing() -> None:
    """Verify validate_wav_file raises FileNotFoundError."""
    from scripts.test_stt import validate_wav_file

    with pytest.raises(FileNotFoundError):
        validate_wav_file(Path("/nonexistent/audio.wav"))


# --- Parser ---


def test_stt_build_parser() -> None:
    """Verify STT parser has all expected arguments."""
    from scripts.test_stt import build_parser

    parser = build_parser()
    actions = {a.dest for a in parser._actions}
    assert "wav_path" in actions
    assert "model_size" in actions
    assert "device" in actions
    assert "compute_type" in actions
    assert "model_dir" in actions
    assert "language" in actions
    assert "beam_size" in actions


def test_stt_parser_defaults() -> None:
    """Verify STT parser default values match constants."""
    from scripts.test_stt import (
        DEFAULT_BEAM_SIZE,
        DEFAULT_COMPUTE_TYPE,
        DEFAULT_DEVICE,
        DEFAULT_LANGUAGE,
        DEFAULT_MODEL_DIR,
        DEFAULT_MODEL_SIZE,
        build_parser,
    )

    parser = build_parser()
    args = parser.parse_args(["test.wav"])
    assert args.wav_path == Path("test.wav")
    assert args.model_size == DEFAULT_MODEL_SIZE
    assert args.device == DEFAULT_DEVICE
    assert args.compute_type == DEFAULT_COMPUTE_TYPE
    assert args.model_dir == DEFAULT_MODEL_DIR
    assert args.language == DEFAULT_LANGUAGE
    assert args.beam_size == DEFAULT_BEAM_SIZE


def test_stt_parser_custom_args() -> None:
    """Verify STT parser accepts custom argument values."""
    from scripts.test_stt import build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "audio.wav",
            "--model-size",
            "large-v3",
            "--device",
            "cpu",
            "--compute-type",
            "float32",
            "--model-dir",
            "/tmp/models",
            "--language",
            "en",
            "--beam-size",
            "3",
        ]
    )
    assert args.wav_path == Path("audio.wav")
    assert args.model_size == "large-v3"
    assert args.device == "cpu"
    assert args.compute_type == "float32"
    assert args.model_dir == "/tmp/models"
    assert args.language == "en"
    assert args.beam_size == 3


# --- Memory ---


def test_stt_get_peak_memory_kb() -> None:
    """Verify get_peak_memory_kb returns non-negative int."""
    from scripts.test_stt import get_peak_memory_kb

    result = get_peak_memory_kb()
    assert isinstance(result, int)
    assert result >= 0


# ===================================================================
# test_llm.py tests
# ===================================================================


# --- Constants ---


def test_llm_constants_exist() -> None:
    """Verify test_llm module exposes expected constants."""
    from scripts.test_llm import (
        DEFAULT_MAX_TOKENS,
        DEFAULT_MODEL_DIR,
        DEFAULT_N_CTX,
        DEFAULT_N_GPU_LAYERS,
        DEFAULT_PROMPT,
        STOP_SEQUENCES,
    )

    assert isinstance(DEFAULT_MODEL_DIR, str)
    assert isinstance(DEFAULT_PROMPT, str)
    assert isinstance(DEFAULT_MAX_TOKENS, int)
    assert isinstance(DEFAULT_N_GPU_LAYERS, int)
    assert isinstance(DEFAULT_N_CTX, int)
    assert isinstance(STOP_SEQUENCES, list)


def test_llm_constants_values() -> None:
    """Verify test_llm constants have sensible defaults."""
    from scripts.test_llm import (
        DEFAULT_MAX_TOKENS,
        DEFAULT_MODEL_DIR,
        DEFAULT_N_CTX,
        DEFAULT_N_GPU_LAYERS,
    )

    assert DEFAULT_MODEL_DIR == "/opt/mungi/ai_models"
    assert DEFAULT_MAX_TOKENS == 64
    assert DEFAULT_N_GPU_LAYERS == 10  # Jetson 8GB safe max
    assert DEFAULT_N_CTX == 4096


# --- Dataclass ---


def test_llm_result_dataclass() -> None:
    """Verify LLMResult dataclass creation and serialization."""
    from scripts.test_llm import LLMResult

    result = LLMResult(
        model_path="/opt/mungi/ai_models/qwen.gguf",
        prompt="Hello",
        generated_text="World",
        completion_tokens=10,
        model_load_time_s=2.5,
        ttft_s=0.1,
        generation_time_s=1.0,
        tokens_per_s=10.0,
        peak_memory_kb=4096,
    )
    d = result.to_dict()
    assert d["model_path"] == "/opt/mungi/ai_models/qwen.gguf"
    assert d["generated_text"] == "World"
    assert d["completion_tokens"] == 10
    assert d["ttft_s"] == 0.1
    assert d["tokens_per_s"] == 10.0
    assert d["peak_memory_kb"] == 4096


# --- GGUF discovery ---


def test_llm_find_gguf_model_empty_dir(
    tmp_path: Path,
) -> None:
    """Verify find_gguf_model returns None for empty dir."""
    from scripts.test_llm import find_gguf_model

    result = find_gguf_model(str(tmp_path))
    assert result is None


def test_llm_find_gguf_model_nonexistent_dir() -> None:
    """Verify find_gguf_model returns None for missing dir."""
    from scripts.test_llm import find_gguf_model

    result = find_gguf_model("/nonexistent/model/dir")
    assert result is None


def test_llm_find_gguf_model_with_files(
    tmp_path: Path,
) -> None:
    """Verify find_gguf_model discovers .gguf files."""
    from scripts.test_llm import find_gguf_model

    # Create fake gguf files
    (tmp_path / "model_a.gguf").write_text("fake")
    (tmp_path / "model_b.gguf").write_text("fake")
    (tmp_path / "not_a_model.txt").write_text("fake")

    result = find_gguf_model(str(tmp_path))
    assert result is not None
    assert result.suffix == ".gguf"
    assert result.name == "model_a.gguf"


# --- Parser ---


def test_llm_build_parser() -> None:
    """Verify LLM parser has all expected arguments."""
    from scripts.test_llm import build_parser

    parser = build_parser()
    actions = {a.dest for a in parser._actions}
    assert "model_path" in actions
    assert "model_dir" in actions
    assert "prompt" in actions
    assert "max_tokens" in actions
    assert "n_gpu_layers" in actions
    assert "n_ctx" in actions


def test_llm_parser_defaults() -> None:
    """Verify LLM parser default values match constants."""
    from scripts.test_llm import (
        DEFAULT_MAX_TOKENS,
        DEFAULT_MODEL_DIR,
        DEFAULT_N_CTX,
        DEFAULT_N_GPU_LAYERS,
        DEFAULT_PROMPT,
        build_parser,
    )

    parser = build_parser()
    args = parser.parse_args([])
    assert args.model_path is None
    assert args.model_dir == DEFAULT_MODEL_DIR
    assert args.prompt == DEFAULT_PROMPT
    assert args.max_tokens == DEFAULT_MAX_TOKENS
    assert args.n_gpu_layers == DEFAULT_N_GPU_LAYERS
    assert args.n_ctx == DEFAULT_N_CTX


def test_llm_parser_custom_args() -> None:
    """Verify LLM parser accepts custom argument values."""
    from scripts.test_llm import build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "--model-path",
            "/tmp/model.gguf",
            "--prompt",
            "Hi",
            "--max-tokens",
            "64",
            "--n-gpu-layers",
            "20",
            "--n-ctx",
            "4096",
        ]
    )
    assert args.model_path == "/tmp/model.gguf"
    assert args.prompt == "Hi"
    assert args.max_tokens == 64
    assert args.n_gpu_layers == 20
    assert args.n_ctx == 4096


# --- Memory ---


def test_llm_get_peak_memory_kb() -> None:
    """Verify get_peak_memory_kb returns non-negative int."""
    from scripts.test_llm import get_peak_memory_kb

    result = get_peak_memory_kb()
    assert isinstance(result, int)
    assert result >= 0


# ===================================================================
# test_tts.py tests
# ===================================================================


# --- Constants ---


def test_tts_constants_exist() -> None:
    """Verify test_tts module exposes expected constants."""
    from scripts.test_tts import (
        DEFAULT_ENGINE,
        DEFAULT_SAMPLE_RATE,
        DEFAULT_SUPERTONIC_MODEL_DIR,
        DEFAULT_TEXT,
    )

    assert isinstance(DEFAULT_ENGINE, str)
    assert isinstance(DEFAULT_TEXT, str)
    assert isinstance(DEFAULT_SUPERTONIC_MODEL_DIR, str)
    assert isinstance(DEFAULT_SAMPLE_RATE, int)


def test_tts_constants_values() -> None:
    """Verify test_tts constants have sensible defaults."""
    from scripts.test_tts import (
        DEFAULT_ENGINE,
        DEFAULT_SAMPLE_RATE,
        DEFAULT_SUPERTONIC_MODEL_DIR,
        DEFAULT_TEXT,
    )

    assert DEFAULT_ENGINE == "supertonic"
    assert DEFAULT_SAMPLE_RATE == 22050
    assert DEFAULT_TEXT == "안녕하세요, 저는 뭉이예요. 오늘 기분이 어때요?"
    assert DEFAULT_SUPERTONIC_MODEL_DIR == ("/opt/mungi/ai_models/supertonic-2")


# --- Dataclass ---


def test_tts_result_dataclass() -> None:
    """Verify TTSResult dataclass creation and serialization."""
    from scripts.test_tts import TTSResult

    result = TTSResult(
        engine="supertonic",
        text="hello",
        model_load_time_s=1.0,
        synthesis_time_s=0.5,
        audio_duration_s=2.0,
        rtf=0.25,
        peak_memory_kb=1024,
        sample_rate=22050,
        num_samples=44100,
        success=True,
    )
    d = result.to_dict()
    assert d["engine"] == "supertonic"
    assert d["success"] is True
    assert d["error"] is None
    assert d["rtf"] == 0.25
    assert d["num_samples"] == 44100


def test_tts_result_dataclass_with_error() -> None:
    """Verify TTSResult dataclass with error field."""
    from scripts.test_tts import TTSResult

    result = TTSResult(
        engine="supertonic",
        text="test",
        model_load_time_s=0.0,
        synthesis_time_s=0.0,
        audio_duration_s=0.0,
        rtf=0.0,
        peak_memory_kb=0,
        sample_rate=0,
        num_samples=0,
        success=False,
        error="Model not found",
    )
    d = result.to_dict()
    assert d["success"] is False
    assert d["error"] == "Model not found"


# --- Engine factory ---


def test_tts_create_engine_supertonic() -> None:
    """Verify create_engine returns SupertonicEngine."""
    from scripts.test_tts import (
        SupertonicEngine,
        create_engine,
    )

    engine = create_engine(
        engine_name="supertonic",
        model_dir="/tmp/models",
    )
    assert isinstance(engine, SupertonicEngine)
    assert engine.engine_name() == "supertonic"


def test_tts_create_engine_invalid() -> None:
    """Verify create_engine raises for unknown engine."""
    from scripts.test_tts import create_engine

    with pytest.raises(ValueError, match="Unknown engine"):
        create_engine(
            engine_name="invalid_engine",
            model_dir="/tmp/models",
        )


# --- Parser ---


def test_tts_build_parser() -> None:
    """Verify TTS parser has all expected arguments."""
    from scripts.test_tts import build_parser

    parser = build_parser()
    actions = {a.dest for a in parser._actions}
    assert "engine" in actions
    assert "text" in actions
    assert "text_unicode_escape" in actions
    assert "model_dir" in actions
    assert "play" in actions
    assert "output_device" in actions
    assert "output_wav" in actions


def test_tts_parser_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify TTS parser default values match constants."""
    from scripts.test_tts import (
        DEFAULT_ENGINE,
        DEFAULT_SUPERTONIC_MODEL_DIR,
        DEFAULT_TEXT,
        build_parser,
    )

    monkeypatch.delenv("MUNGI_AUDIO_OUTPUT_DEVICE", raising=False)
    parser = build_parser()
    args = parser.parse_args([])
    assert args.engine == DEFAULT_ENGINE
    assert args.text == DEFAULT_TEXT
    assert args.text_unicode_escape is None
    assert args.model_dir == DEFAULT_SUPERTONIC_MODEL_DIR
    assert args.play is False
    assert args.output_device is None
    assert args.output_wav is None


def test_tts_parser_custom_args() -> None:
    """Verify TTS parser accepts custom argument values."""
    from scripts.test_tts import build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "--engine",
            "supertonic",
            "--text",
            "Hello world",
            "--model-dir",
            "/tmp/models",
            "--play",
            "--output-device",
            "USB PnP Audio Device",
            "--output-wav",
            "/tmp/output.wav",
        ]
    )
    assert args.engine == "supertonic"
    assert args.text == "Hello world"
    assert args.text_unicode_escape is None
    assert args.model_dir == "/tmp/models"
    assert args.play is True
    assert args.output_device == "USB PnP Audio Device"
    assert args.output_wav == Path("/tmp/output.wav")


def test_tts_parser_unicode_escape_arg() -> None:
    """Verify TTS parser accepts unicode-escaped text input."""
    from scripts.test_tts import build_parser

    parser = build_parser()
    args = parser.parse_args(["--text-unicode-escape", r"\uc548\ub155\ud558\uc138\uc694"])

    assert args.text_unicode_escape == r"\uc548\ub155\ud558\uc138\uc694"


def test_tts_resolve_text_input_defaults() -> None:
    """Verify text resolver falls back to the default phrase."""
    from scripts.test_tts import DEFAULT_TEXT, resolve_text_input

    assert resolve_text_input(None, None) == DEFAULT_TEXT


def test_tts_resolve_text_input_literal_text() -> None:
    """Verify text resolver preserves literal CLI text."""
    from scripts.test_tts import resolve_text_input

    assert resolve_text_input("Hello world", None) == "Hello world"


def test_tts_resolve_text_input_unicode_escape() -> None:
    """Verify text resolver decodes unicode-escaped text."""
    from scripts.test_tts import resolve_text_input

    assert resolve_text_input(None, r"\uc548\ub155\ud558\uc138\uc694") == "안녕하세요"


def test_tts_resolve_text_input_invalid_unicode_escape() -> None:
    """Verify text resolver rejects malformed unicode escapes."""
    from scripts.test_tts import resolve_text_input

    with pytest.raises(ValueError, match="unicode escapes"):
        resolve_text_input(None, r"\u12")


# --- Memory ---


def test_tts_get_peak_memory_kb() -> None:
    """Verify get_peak_memory_kb returns non-negative int."""
    from scripts.test_tts import get_peak_memory_kb

    result = get_peak_memory_kb()
    assert isinstance(result, int)
    assert result >= 0


# --- write_wav ---


@pytest.mark.skipif(
    not HAS_NUMPY,
    reason="numpy not installed",
)
def test_tts_write_wav(tmp_path: Path) -> None:
    """Verify write_wav creates a valid WAV file."""
    import wave

    from scripts.test_tts import write_wav

    samples = np.zeros(1000, dtype=np.float32)
    out_path = tmp_path / "test_output.wav"
    write_wav(out_path, samples, 22050)

    assert out_path.exists()
    with wave.open(str(out_path), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 22050
        assert wf.getnframes() == 1000


@pytest.mark.skipif(
    not HAS_NUMPY,
    reason="numpy not installed",
)
def test_tts_write_wav_clips_values(tmp_path: Path) -> None:
    """Verify write_wav clips samples outside [-1, 1]."""
    import wave

    from scripts.test_tts import write_wav

    # Values outside [-1, 1] should be clipped
    samples = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    out_path = tmp_path / "clipped.wav"
    write_wav(out_path, samples, 16000)

    assert out_path.exists()
    with wave.open(str(out_path), "rb") as wf:
        assert wf.getnframes() == 5


# ===================================================================
# bench_model.py updated tests (STTBenchmark)
# ===================================================================


def test_bench_stt_benchmark_model_type() -> None:
    """Verify STTBenchmark has correct model_type."""
    from scripts.bench_model import STTBenchmark

    bench = STTBenchmark()
    assert bench.model_type() == "stt"


def test_bench_stt_benchmark_in_registry() -> None:
    """Verify STTBenchmark is registered in BENCHMARK_REGISTRY."""
    from scripts.bench_model import (
        BENCHMARK_REGISTRY,
        STTBenchmark,
    )

    assert "stt" in BENCHMARK_REGISTRY
    assert BENCHMARK_REGISTRY["stt"] is STTBenchmark


def test_bench_parser_stt_with_new_args() -> None:
    """Verify bench_model parser has STT-specific arguments."""
    from scripts.bench_model import build_parser

    parser = build_parser()
    actions = {a.dest for a in parser._actions}
    assert "model_size" in actions
    assert "device" in actions
    assert "compute_type" in actions
    assert "model_dir" in actions
    assert "language" in actions
    assert "beam_size" in actions


def test_bench_parser_stt_subcommand_with_args() -> None:
    """Verify bench_model parser handles STT with extra args."""
    from scripts.bench_model import build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "stt",
            "test.wav",
            "--model-size",
            "large-v3",
            "--device",
            "cpu",
            "--compute-type",
            "float32",
            "--language",
            "en",
            "--beam-size",
            "3",
        ]
    )
    assert args.model_type == "stt"
    assert args.input_path == "test.wav"
    assert args.model_size == "large-v3"
    assert args.device == "cpu"
    assert args.compute_type == "float32"
    assert args.language == "en"
    assert args.beam_size == 3


# ===================================================================
# bench_model.py — None input_path guard tests
# ===================================================================


def test_bench_stt_none_input_returns_error() -> None:
    """Verify STTBenchmark returns a clean error for None input_path.

    The fix adds an early guard in STTBenchmark.run() that returns a
    BenchmarkResult with an error message before attempting heavy
    imports, when args.input_path is None.
    """
    from scripts.bench_model import STTBenchmark, build_parser

    parser = build_parser()
    args = parser.parse_args(["stt"])
    assert args.input_path is None

    bench = STTBenchmark()
    result = bench.run(args)

    assert result.model_type == "stt"
    assert result.error is not None
    assert "input" in result.error.lower() or "WAV" in result.error
    assert result.model_path == "(no input file)"
    assert result.load_time_s == 0.0
    assert result.inference_time_s == 0.0


def test_bench_vad_none_input_returns_error() -> None:
    """Verify VADBenchmark returns a clean error for None input_path.

    The fix adds an early guard in VADBenchmark.run() that returns a
    BenchmarkResult with an error message before attempting file I/O,
    when args.input_path is None (same pattern as STTBenchmark).
    """
    from scripts.bench_model import VADBenchmark, build_parser

    parser = build_parser()
    args = parser.parse_args(["vad"])
    assert args.input_path is None

    bench = VADBenchmark()
    result = bench.run(args)

    assert result.model_type == "vad"
    assert result.error is not None
    assert "input" in result.error.lower() or "WAV" in result.error
    assert result.model_path == "(no input file)"
    assert result.load_time_s == 0.0
    assert result.inference_time_s == 0.0


# ===================================================================
# Module importability tests
# ===================================================================


def test_test_stt_importable() -> None:
    """Verify test_stt module can be imported without errors."""
    from scripts import test_stt  # noqa: F401

    assert hasattr(test_stt, "main")
    assert hasattr(test_stt, "build_parser")
    assert hasattr(test_stt, "validate_wav_file")
    assert hasattr(test_stt, "get_peak_memory_kb")
    assert hasattr(test_stt, "STTResult")
    # TranscriptionSegment moved to models.stt_runner (Sprint 3)


def test_test_llm_importable() -> None:
    """Verify test_llm module can be imported without errors."""
    from scripts import test_llm  # noqa: F401

    assert hasattr(test_llm, "main")
    assert hasattr(test_llm, "build_parser")
    assert hasattr(test_llm, "find_gguf_model")
    assert hasattr(test_llm, "get_peak_memory_kb")
    assert hasattr(test_llm, "LLMResult")


def test_test_tts_importable() -> None:
    """Verify test_tts module can be imported without errors."""
    from scripts import test_tts  # noqa: F401

    assert hasattr(test_tts, "main")
    assert hasattr(test_tts, "build_parser")
    assert hasattr(test_tts, "create_engine")
    assert hasattr(test_tts, "get_peak_memory_kb")
    assert hasattr(test_tts, "write_wav")
    assert hasattr(test_tts, "TTSResult")
    assert hasattr(test_tts, "TTSEngine")
    assert hasattr(test_tts, "SupertonicEngine")


# ===================================================================
# scripts/utils.py — shared utility tests
# ===================================================================


def test_utils_importable() -> None:
    """Verify scripts.utils exposes get_peak_memory_kb."""
    from scripts.utils import get_peak_memory_kb  # noqa: F401

    assert callable(get_peak_memory_kb)


def test_utils_get_peak_memory_kb_returns_int() -> None:
    """Verify get_peak_memory_kb returns a non-negative int."""
    from scripts.utils import get_peak_memory_kb

    result = get_peak_memory_kb()
    assert isinstance(result, int)
    assert result >= 0


# ===================================================================
# Re-exported get_peak_memory_kb identity tests
# ===================================================================


def test_vad_uses_shared_get_peak_memory_kb() -> None:
    """Verify test_vad.get_peak_memory_kb is the shared util."""
    from scripts import test_vad, utils

    assert test_vad.get_peak_memory_kb is utils.get_peak_memory_kb


def test_stt_uses_shared_get_peak_memory_kb() -> None:
    """Verify test_stt.get_peak_memory_kb is the shared util."""
    from scripts import test_stt, utils

    assert test_stt.get_peak_memory_kb is utils.get_peak_memory_kb


def test_llm_uses_shared_get_peak_memory_kb() -> None:
    """Verify test_llm.get_peak_memory_kb is the shared util."""
    from scripts import test_llm, utils

    assert test_llm.get_peak_memory_kb is utils.get_peak_memory_kb


def test_tts_uses_shared_get_peak_memory_kb() -> None:
    """Verify test_tts.get_peak_memory_kb is the shared util."""
    from scripts import test_tts, utils

    assert test_tts.get_peak_memory_kb is utils.get_peak_memory_kb


# ===================================================================
# LLMBenchmark real implementation tests
# ===================================================================


def test_bench_llm_model_type() -> None:
    """Verify LLMBenchmark reports model_type as 'llm'."""
    from scripts.bench_model import LLMBenchmark

    bench = LLMBenchmark()
    assert bench.model_type() == "llm"


def test_bench_llm_is_not_stub() -> None:
    """Verify LLMBenchmark.run() no longer returns 'Not implemented'.

    With the real implementation, running against a nonexistent model
    directory should produce a model-not-found error, not a stub error.
    """
    import argparse

    from scripts.bench_model import LLMBenchmark

    args = argparse.Namespace(
        model_path=None,
        model_dir="/nonexistent/dir/for/stub/check",
        prompt=None,
        max_tokens=None,
        n_gpu_layers=None,
        n_ctx=None,
    )
    bench = LLMBenchmark()
    result = bench.run(args)
    # The old stub returned "Not implemented" — real code does not.
    assert result.error is None or "Not implemented" not in result.error


def test_bench_llm_no_model_returns_error() -> None:
    """Verify LLMBenchmark returns error when no GGUF model is found."""
    import argparse

    from scripts.bench_model import LLMBenchmark

    args = argparse.Namespace(
        model_path=None,
        model_dir="/nonexistent/model/directory",
        prompt=None,
        max_tokens=None,
        n_gpu_layers=None,
        n_ctx=None,
    )
    bench = LLMBenchmark()
    result = bench.run(args)

    assert result.model_type == "llm"
    assert result.error is not None
    assert "No GGUF model found" in result.error
    assert result.model_path == "(no model found)"


# ===================================================================
# TTSBenchmark real implementation tests
# ===================================================================


def test_bench_tts_model_type() -> None:
    """Verify TTSBenchmark reports model_type as 'tts'."""
    from scripts.bench_model import TTSBenchmark

    bench = TTSBenchmark()
    assert bench.model_type() == "tts"


def test_bench_tts_is_not_stub() -> None:
    """Verify TTSBenchmark docstring does not mention 'stub'."""
    from scripts.bench_model import TTSBenchmark

    docstring = TTSBenchmark.__doc__ or ""
    assert "stub" not in docstring.lower()


def test_bench_tts_no_model_dir_returns_error() -> None:
    """Verify TTSBenchmark returns error for nonexistent model dir.

    The real implementation calls create_engine -> engine.load(),
    which should fail when the model directory does not exist.
    """
    import argparse

    from scripts.bench_model import TTSBenchmark

    args = argparse.Namespace(
        engine="supertonic",
        text=None,
        supertonic_model_dir="/nonexistent/supertonic/dir",
        output_wav=None,
    )
    bench = TTSBenchmark()
    result = bench.run(args)

    assert result.model_type == "tts"
    assert result.error is not None


# ===================================================================
# New CLI arguments tests
# ===================================================================


def test_bench_parser_llm_args() -> None:
    """Verify bench_model parser handles LLM-specific arguments."""
    from scripts.bench_model import build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "llm",
            "--max-tokens",
            "256",
            "--n-gpu-layers",
            "-1",
            "--n-ctx",
            "4096",
        ]
    )
    assert args.model_type == "llm"
    assert args.max_tokens == 256
    assert args.n_gpu_layers == -1
    assert args.n_ctx == 4096


def test_bench_parser_tts_args() -> None:
    """Verify bench_model parser handles TTS-specific arguments."""
    from scripts.bench_model import build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "tts",
            "--engine",
            "supertonic",
            "--text-unicode-escape",
            r"\uc548\ub155",
            "--supertonic-model-dir",
            "/tmp/model",
            "--output-wav",
            "/tmp/out.wav",
        ]
    )
    assert args.model_type == "tts"
    assert args.engine == "supertonic"
    assert args.text_unicode_escape == r"\uc548\ub155"
    assert args.supertonic_model_dir == "/tmp/model"
    assert args.output_wav == Path("/tmp/out.wav")


# ===================================================================
# main() exit code change test
# ===================================================================


def test_bench_main_exit_code_no_stub_exception() -> None:
    """Verify main() no longer has a 'Not implemented' exception path.

    The old main() had logic:
        return 1 if result.error and "Not implemented" not in ...
    The new main() simply does:
        return 1 if result.error else 0

    We verify by inspecting the source code of main() to confirm
    'Not implemented' is not referenced.
    """
    import inspect

    from scripts.bench_model import main

    source = inspect.getsource(main)
    assert "Not implemented" not in source
