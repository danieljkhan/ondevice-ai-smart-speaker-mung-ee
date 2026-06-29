"""Model benchmark framework for Mungi on Jetson Orin Nano.

Provides a generic framework for benchmarking AI model load times,
inference times, and memory usage. Supports tegrastats parsing for
GPU memory monitoring.

Usage:
    python scripts/bench_model.py vad /path/to/test.wav
    python scripts/bench_model.py vad /path/to/test.wav \\
        --model-path /opt/mungi/ai_models/silero_vad.jit
    python scripts/bench_model.py stt /path/to/test.wav
    python scripts/bench_model.py llm --prompt "Hello"
    python scripts/bench_model.py tts --text "Hello world"
    python scripts/bench_model.py tts --text-unicode-escape "\uc548\ub155\ud558\uc138\uc694"
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("mungi.scripts.bench_model")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL_DIR: str = "/opt/mungi/ai_models"
SUPPORTED_MODEL_TYPES: list[str] = ["vad", "stt", "llm", "tts"]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class MemorySnapshot:
    """A snapshot of memory usage at a point in time."""

    rss_kb: int = 0
    vm_peak_kb: int = 0
    gpu_used_mb: int = 0

    def to_dict(self) -> dict[str, int]:
        """Serialize to a plain dict."""
        return asdict(self)


@dataclass
class BenchmarkResult:
    """Result of a single model benchmark run."""

    model_type: str
    model_path: str
    load_time_s: float = 0.0
    inference_time_s: float = 0.0
    mem_before: dict[str, int] = field(default_factory=dict)
    mem_after: dict[str, int] = field(default_factory=dict)
    mem_delta_kb: int = 0
    extra: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Memory utilities
# ---------------------------------------------------------------------------


def read_proc_memory() -> MemorySnapshot:
    """Read current process memory from /proc/self/status on Linux.

    Returns:
        MemorySnapshot with RSS and VmPeak values. Returns zeros on
        non-Linux platforms.
    """
    snapshot = MemorySnapshot()
    try:
        status_path = Path("/proc/self/status")
        if not status_path.exists():
            return snapshot
        text = status_path.read_text(encoding="utf-8")
        for line in text.splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2:
                    snapshot.rss_kb = int(parts[1])
            elif line.startswith("VmPeak:"):
                parts = line.split()
                if len(parts) >= 2:
                    snapshot.vm_peak_kb = int(parts[1])
    except (OSError, ValueError):
        pass
    return snapshot


def parse_tegrastats_line(line: str) -> dict[str, Any]:
    """Parse a single tegrastats output line for memory/GPU info.

    Tegrastats lines look like:
        RAM 3456/7620MB (lfb 23x4MB) SWAP 0/3810MB ...
        GR3D_FREQ 0% ...

    Args:
        line: A single line from tegrastats output.

    Returns:
        Dict with parsed values such as ram_used_mb, ram_total_mb,
        swap_used_mb, swap_total_mb, gr3d_freq_pct, cpu_temp_c,
        and gpu_temp_c.
    """
    result: dict[str, Any] = {}

    # RAM pattern: RAM XXXX/YYYYMB
    ram_match = re.search(r"RAM\s+(\d+)/(\d+)MB", line)
    if ram_match:
        result["ram_used_mb"] = int(ram_match.group(1))
        result["ram_total_mb"] = int(ram_match.group(2))

    # SWAP pattern: SWAP XXXX/YYYYMB
    swap_match = re.search(r"SWAP\s+(\d+)/(\d+)MB", line)
    if swap_match:
        result["swap_used_mb"] = int(swap_match.group(1))
        result["swap_total_mb"] = int(swap_match.group(2))

    # GR3D frequency: GR3D_FREQ XX%
    gr3d_match = re.search(r"GR3D_FREQ\s+(\d+)%", line)
    if gr3d_match:
        result["gr3d_freq_pct"] = int(gr3d_match.group(1))

    cpu_temp_match = re.search(r"cpu@([0-9.]+)C", line)
    if cpu_temp_match:
        result["cpu_temp_c"] = float(cpu_temp_match.group(1))

    gpu_temp_match = re.search(r"gpu@([0-9.]+)C", line)
    if gpu_temp_match:
        result["gpu_temp_c"] = float(gpu_temp_match.group(1))

    return result


def parse_tegrastats_log(log_path: Path) -> list[dict[str, Any]]:
    """Parse a tegrastats log file into a list of snapshots.

    Args:
        log_path: Path to a tegrastats log file.

    Returns:
        List of parsed dicts, one per line.

    Raises:
        FileNotFoundError: If the log file does not exist.
    """
    if not log_path.exists():
        msg = f"Tegrastats log not found: {log_path}"
        raise FileNotFoundError(msg)

    text = log_path.read_text(encoding="utf-8")
    snapshots: list[dict[str, Any]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            parsed = parse_tegrastats_line(stripped)
            if parsed:
                snapshots.append(parsed)
    return snapshots


# ---------------------------------------------------------------------------
# Abstract benchmark runner
# ---------------------------------------------------------------------------


class ModelBenchmark(ABC):
    """Abstract base class for model benchmark runners."""

    @abstractmethod
    def model_type(self) -> str:
        """Return the model type identifier."""

    @abstractmethod
    def run(self, args: argparse.Namespace) -> BenchmarkResult:
        """Execute the benchmark and return results.

        Args:
            args: Parsed CLI arguments.

        Returns:
            BenchmarkResult with timing and memory data.
        """


# ---------------------------------------------------------------------------
# VAD benchmark
# ---------------------------------------------------------------------------


class VADBenchmark(ModelBenchmark):
    """Benchmark runner for Silero VAD model."""

    def model_type(self) -> str:
        """Return 'vad' as the model type."""
        return "vad"

    def run(self, args: argparse.Namespace) -> BenchmarkResult:
        """Run the VAD benchmark on a WAV file.

        Args:
            args: CLI args with input_path, model_path, etc.

        Returns:
            BenchmarkResult with VAD timing and segment info.
        """
        from models.vad_runner import (
            SpeechSegment,
            load_vad_model,
            run_vad,
        )
        from scripts.test_vad import read_wav_mono_16k

        if args.input_path is None:
            result = BenchmarkResult(
                model_type="vad",
                model_path="(no input file)",
                error="VAD benchmark requires an input WAV file path.",
            )
            return result
        model_path = getattr(args, "model_path", None)
        input_path = Path(args.input_path)
        result_model_path = model_path or "torch.hub (silero-vad)"

        result = BenchmarkResult(
            model_type="vad",
            model_path=result_model_path,
        )

        try:
            # Read audio
            audio_samples = read_wav_mono_16k(input_path)
            audio_duration = len(audio_samples) / 16000.0

            # Measure model load
            mem_before = read_proc_memory()
            result.mem_before = mem_before.to_dict()

            t0 = time.monotonic()
            model = load_vad_model(model_path)
            result.load_time_s = round(time.monotonic() - t0, 4)

            # Measure inference
            t0 = time.monotonic()
            segments: list[SpeechSegment] = run_vad(audio_samples, model)
            result.inference_time_s = round(time.monotonic() - t0, 4)

            mem_after = read_proc_memory()
            result.mem_after = mem_after.to_dict()
            result.mem_delta_kb = max(mem_after.rss_kb - mem_before.rss_kb, 0)

            result.extra = {
                "audio_duration_s": round(audio_duration, 4),
                "segments_count": len(segments),
                "total_speech_s": round(
                    sum(seg.duration_ms() for seg in segments) / 1000.0,
                    4,
                ),
                "rtf": round(result.inference_time_s / audio_duration, 4)
                if audio_duration > 0
                else 0.0,
            }
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
            logger.error("VAD benchmark failed: %s", exc)

        return result


# ---------------------------------------------------------------------------
# STT benchmark
# ---------------------------------------------------------------------------


class STTBenchmark(ModelBenchmark):
    """Benchmark runner for Sherpa-ONNX STT models."""

    def model_type(self) -> str:
        """Return 'stt' as the model type."""
        return "stt"

    def run(self, args: argparse.Namespace) -> BenchmarkResult:
        """Run the STT benchmark on a WAV file.

        Args:
            args: CLI args with input_path and optional model_path,
                model_size, device, compute_type, model_dir, language.

        Returns:
            BenchmarkResult with STT timing, memory, and transcription.
        """
        from models.stt_runner import (
            DEFAULT_BEAM_SIZE,
            DEFAULT_COMPUTE_TYPE,
            DEFAULT_DEVICE,
            DEFAULT_LANGUAGE,
            DEFAULT_MODEL_DIR,
            DEFAULT_MODEL_SIZE,
            TranscriptionSegment,
            load_stt_model,
            resolve_model_size,
            run_stt,
        )
        from scripts.test_stt import validate_wav_file

        if args.input_path is None:
            result = BenchmarkResult(
                model_type="stt",
                model_path="(no input file)",
                error="STT benchmark requires an input WAV file path.",
            )
            return result
        input_path = Path(args.input_path)
        model_size = getattr(args, "model_size", None) or DEFAULT_MODEL_SIZE
        device = getattr(args, "device", None) or DEFAULT_DEVICE
        compute_type = getattr(args, "compute_type", None) or DEFAULT_COMPUTE_TYPE
        model_dir = getattr(args, "model_dir", None) or DEFAULT_MODEL_DIR
        language = getattr(args, "language", None) or DEFAULT_LANGUAGE
        beam_size = getattr(args, "beam_size", None) or DEFAULT_BEAM_SIZE

        try:
            resolved_model_size = resolve_model_size(model_size)
        except ValueError:
            resolved_model_size = model_size
        result_model_path = f"{model_dir}/{resolved_model_size}"

        result = BenchmarkResult(
            model_type="stt",
            model_path=result_model_path,
        )

        try:
            # Validate audio
            audio_duration = validate_wav_file(input_path)

            # Measure model load
            mem_before = read_proc_memory()
            result.mem_before = mem_before.to_dict()

            t0 = time.monotonic()
            model = load_stt_model(
                model_size=model_size,
                device=device,
                compute_type=compute_type,
                model_dir=model_dir,
                language=language,
            )
            result.model_path = getattr(model, "model_path", result_model_path)
            result.load_time_s = round(time.monotonic() - t0, 4)

            # Measure inference
            t0 = time.monotonic()
            segments: list[TranscriptionSegment]
            segments, info_dict = run_stt(
                model,
                input_path,
                language=language,
                beam_size=beam_size,
            )
            result.inference_time_s = round(time.monotonic() - t0, 4)

            mem_after = read_proc_memory()
            result.mem_after = mem_after.to_dict()
            result.mem_delta_kb = max(mem_after.rss_kb - mem_before.rss_kb, 0)

            full_text = " ".join(seg.text for seg in segments)
            rtf = (
                round(
                    result.inference_time_s / audio_duration,
                    4,
                )
                if audio_duration > 0
                else 0.0
            )

            result.extra = {
                "audio_duration_s": round(audio_duration, 4),
                "segments_count": len(segments),
                "full_text": full_text,
                "rtf": rtf,
                "model_size": model_size,
                "resolved_model_size": info_dict.get("resolved_model_size", resolved_model_size),
                "device": device,
                "compute_type": compute_type,
                "language": language,
                "detected_language": info_dict.get("language", language),
                "language_probability": info_dict.get("language_probability", 0.0),
                "backend": info_dict.get("backend", "unknown"),
                "provider": info_dict.get("provider", device),
            }
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
            logger.error("STT benchmark failed: %s", exc)

        return result


# ---------------------------------------------------------------------------
# LLM benchmark
# ---------------------------------------------------------------------------


class LLMBenchmark(ModelBenchmark):
    """Benchmark runner for Qwen LLM via llama.cpp."""

    def model_type(self) -> str:
        """Return 'llm' as the model type."""
        return "llm"

    def run(self, args: argparse.Namespace) -> BenchmarkResult:
        """Run the LLM benchmark with a text generation prompt.

        Args:
            args: CLI args with optional model_path, prompt,
                max_tokens, n_gpu_layers, n_ctx.

        Returns:
            BenchmarkResult with LLM timing, memory, and
            generation metrics.
        """
        from scripts.test_llm import (
            DEFAULT_MAX_TOKENS,
            DEFAULT_MODEL_DIR,
            DEFAULT_N_CTX,
            DEFAULT_N_GPU_LAYERS,
            find_gguf_model,
            load_llm_model,
            run_generation,
        )

        # Resolve model path
        model_path: str | None = getattr(args, "model_path", None)
        if model_path is None:
            model_dir = getattr(args, "model_dir", None) or DEFAULT_MODEL_DIR
            discovered = find_gguf_model(model_dir)
            if discovered is None:
                return BenchmarkResult(
                    model_type="llm",
                    model_path="(no model found)",
                    error=(f"No GGUF model found in {model_dir}. Provide --model-path explicitly."),
                )
            model_path = str(discovered)

        prompt: str = getattr(args, "prompt", None) or "안녕하세요, 뭉이와 이야기해요."
        max_tokens_raw = getattr(args, "max_tokens", None)
        max_tokens: int = max_tokens_raw if max_tokens_raw is not None else DEFAULT_MAX_TOKENS
        n_gpu_layers_raw = getattr(args, "n_gpu_layers", None)
        n_gpu_layers: int = (
            n_gpu_layers_raw if n_gpu_layers_raw is not None else DEFAULT_N_GPU_LAYERS
        )
        n_ctx_raw = getattr(args, "n_ctx", None)
        n_ctx: int = n_ctx_raw if n_ctx_raw is not None else DEFAULT_N_CTX

        result = BenchmarkResult(
            model_type="llm",
            model_path=model_path,
        )

        try:
            # Measure model load
            mem_before = read_proc_memory()
            result.mem_before = mem_before.to_dict()

            t0 = time.monotonic()
            llm = load_llm_model(
                model_path=model_path,
                n_gpu_layers=n_gpu_layers,
                n_ctx=n_ctx,
            )
            result.load_time_s = round(time.monotonic() - t0, 4)

            # Measure inference (generation)
            t0 = time.monotonic()
            generated_text, token_count, ttft, gen_time = run_generation(
                llm=llm,
                prompt=prompt,
                max_tokens=max_tokens,
            )
            result.inference_time_s = round(time.monotonic() - t0, 4)

            mem_after = read_proc_memory()
            result.mem_after = mem_after.to_dict()
            result.mem_delta_kb = max(mem_after.rss_kb - mem_before.rss_kb, 0)

            tokens_per_s: float = token_count / gen_time if gen_time > 0 else 0.0

            result.extra = {
                "prompt": prompt,
                "generated_text": generated_text,
                "completion_tokens": token_count,
                "ttft_s": round(ttft, 4),
                "tokens_per_s": round(tokens_per_s, 2),
                "n_gpu_layers": n_gpu_layers,
                "n_ctx": n_ctx,
                "max_tokens": max_tokens,
            }
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
            logger.error("LLM benchmark failed: %s", exc)

        return result


# ---------------------------------------------------------------------------
# TTS benchmark
# ---------------------------------------------------------------------------


class TTSBenchmark(ModelBenchmark):
    """Benchmark runner for Supertonic TTS."""

    def model_type(self) -> str:
        """Return 'tts' as the model type."""
        return "tts"

    def run(self, args: argparse.Namespace) -> BenchmarkResult:
        """Run the TTS benchmark with text synthesis.

        Args:
            args: CLI args with optional engine, text,
                supertonic_model_dir, output_wav.

        Returns:
            BenchmarkResult with TTS timing, memory, and
            audio metrics.
        """
        from scripts.test_tts import (
            DEFAULT_ENGINE,
            DEFAULT_SUPERTONIC_MODEL_DIR,
            create_engine,
            resolve_text_input,
            write_wav,
        )

        engine_name: str = getattr(args, "engine", None) or DEFAULT_ENGINE
        text = resolve_text_input(
            getattr(args, "text", None),
            getattr(args, "text_unicode_escape", None),
        )
        supertonic_model_dir: str = (
            getattr(args, "supertonic_model_dir", None) or DEFAULT_SUPERTONIC_MODEL_DIR
        )
        output_wav: Path | None = getattr(args, "output_wav", None)

        result = BenchmarkResult(
            model_type="tts",
            model_path=supertonic_model_dir,
        )

        try:
            # Create engine
            engine = create_engine(
                engine_name=engine_name,
                model_dir=supertonic_model_dir,
            )

            # Measure model load
            mem_before = read_proc_memory()
            result.mem_before = mem_before.to_dict()

            t0 = time.monotonic()
            engine.load()
            result.load_time_s = round(time.monotonic() - t0, 4)

            # Measure synthesis
            t0 = time.monotonic()
            audio, sample_rate = engine.synthesize(text)
            result.inference_time_s = round(time.monotonic() - t0, 4)

            mem_after = read_proc_memory()
            result.mem_after = mem_after.to_dict()
            result.mem_delta_kb = max(mem_after.rss_kb - mem_before.rss_kb, 0)

            num_samples = len(audio)
            audio_duration = num_samples / sample_rate if sample_rate > 0 else 0.0
            rtf = result.inference_time_s / audio_duration if audio_duration > 0 else 0.0

            result.extra = {
                "engine": engine_name,
                "text": text,
                "audio_duration_s": round(audio_duration, 4),
                "rtf": round(rtf, 4),
                "sample_rate": sample_rate,
                "num_samples": num_samples,
            }

            # Save WAV if requested
            if output_wav is not None:
                output_path = Path(output_wav)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                write_wav(output_path, audio, sample_rate)
                logger.info("TTS output saved to: %s", output_path)
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
            logger.error("TTS benchmark failed: %s", exc)

        return result


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

BENCHMARK_REGISTRY: dict[str, type[ModelBenchmark]] = {
    "vad": VADBenchmark,
    "stt": STTBenchmark,
    "llm": LLMBenchmark,
    "tts": TTSBenchmark,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the benchmark framework.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        description=("Mungi model benchmark framework for Jetson Orin Nano."),
    )
    parser.add_argument(
        "model_type",
        choices=SUPPORTED_MODEL_TYPES,
        help="Model type to benchmark: vad, stt, llm, tts.",
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default=None,
        help="Input file path (WAV for vad/stt).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to the model file.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt text for LLM benchmark.",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Text input for TTS benchmark.",
    )
    parser.add_argument(
        "--text-unicode-escape",
        type=str,
        default=None,
        help="Unicode-escaped text input for TTS benchmark.",
    )
    # STT-specific arguments
    parser.add_argument(
        "--model-size",
        type=str,
        default=None,
        help=("Sherpa STT model selector (e.g. small, sense-voice, moonshine-tiny-ko)."),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for STT inference (cuda or cpu).",
    )
    parser.add_argument(
        "--compute-type",
        type=str,
        default=None,
        help="Compatibility-only compute type flag for STT. Sherpa-ONNX ignores it.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Model cache directory for STT.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help=("Language code for STT transcription (e.g. ko, en)."),
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=None,
        help="Beam size for STT decoding.",
    )
    # LLM-specific arguments
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum tokens to generate for LLM benchmark.",
    )
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=None,
        help=("Number of layers to offload to GPU for LLM. -1 = all layers."),
    )
    parser.add_argument(
        "--n-ctx",
        type=int,
        default=None,
        help="Context window size in tokens for LLM.",
    )
    # TTS-specific arguments
    parser.add_argument(
        "--engine",
        type=str,
        default=None,
        choices=["supertonic"],
        help="TTS engine to use (supertonic).",
    )
    parser.add_argument(
        "--supertonic-model-dir",
        type=str,
        default=None,
        help="Path to Supertonic 2 model directory.",
    )
    parser.add_argument(
        "--output-wav",
        type=Path,
        default=None,
        help="Save synthesized TTS audio to this WAV path.",
    )
    # Common arguments
    parser.add_argument(
        "--tegrastats-log",
        type=Path,
        default=None,
        help=("Path to tegrastats log file for GPU memory analysis."),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Save JSON result to this file path.",
    )
    return parser


def main() -> int:
    """Run the benchmark for the specified model type.

    Returns:
        Exit code: 0 on success, 1 on error.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = build_parser()
    args = parser.parse_args()

    model_type: str = args.model_type
    bench_cls = BENCHMARK_REGISTRY.get(model_type)
    if bench_cls is None:
        logger.error("Unknown model type: %s", model_type)
        return 1

    logger.info("Starting benchmark: model_type=%s", model_type)

    benchmark = bench_cls()
    result = benchmark.run(args)

    # Attach tegrastats data if provided
    tegrastats_data: list[dict[str, Any]] | None = None
    if args.tegrastats_log is not None:
        try:
            tegrastats_data = parse_tegrastats_log(args.tegrastats_log)
            logger.info(
                "Parsed %d tegrastats snapshots from %s",
                len(tegrastats_data),
                args.tegrastats_log,
            )
        except FileNotFoundError as exc:
            logger.warning("Tegrastats log error: %s", exc)

    # Build output
    output: dict[str, Any] = {
        "benchmark": result.to_dict(),
    }
    if tegrastats_data is not None:
        output["tegrastats"] = {
            "snapshots_count": len(tegrastats_data),
            "snapshots": tegrastats_data,
        }

    json_str = json.dumps(output, indent=2, ensure_ascii=False)
    logger.info("Benchmark result:\n%s", json_str)

    # Save to file if requested
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json_str + "\n", encoding="utf-8")
        logger.info("Result saved to: %s", args.output)

    return 1 if result.error else 0


if __name__ == "__main__":
    raise SystemExit(main())
