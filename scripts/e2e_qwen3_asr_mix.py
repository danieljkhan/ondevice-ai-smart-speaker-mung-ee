"""Bilingual interleaved audio-input E2E runner for pre-recorded child WAVs.

Discovers already-recorded Korean and English child voice WAVs, interleaves
them round-by-round, runs each clip through Mungi's full conversation pipeline
(STT -> LLM -> TTS), and writes per-round artifacts plus an aggregate summary.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import logging
import os
import re
import shutil
import signal
import sys
import time
import wave
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.model_manager import ModelType  # noqa: E402
from models.tts_runner import (  # noqa: E402
    SentenceSynthesisResult,
    synthesize_to_speaker_by_sentence,
)

if TYPE_CHECKING:
    from core.pipeline import TurnResult

logger = logging.getLogger("mungi.scripts.e2e_qwen3_asr_mix")

DEFAULT_OUTPUT_ROOT = Path(
    os.getenv("MUNGI_E2E_QWEN3_OUTPUT_DIR", "/var/lib/mungi/conversations"),
)
DEFAULT_MODEL_DIR = os.getenv("MUNGI_MODEL_DIR", "/opt/mungi/ai_models")
DEFAULT_OUTPUT_DEVICE = os.getenv("MUNGI_AUDIO_OUTPUT_DEVICE", "").strip() or None
DEFAULT_STT_MODEL = "qwen3-asr-0.6b"
DEFAULT_STT_LANGUAGE = "ko"
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_SKIP_TTS_SAMPLE_RATE = 22050
# Silero VAD onset detection needs hangover; 300 ms is a conservative floor.
VAD_AUDIO_TOO_SHORT_MS = 300
_TRUE_ENV_VALUES = frozenset({"1", "true", "on", "yes"})
_FALSE_ENV_VALUES = frozenset({"0", "false", "off", "no"})
LATENCY_TABLE_HEADER = (
    "| Turn | VAD | STT | LLM로드 | TTFT | LLM추론 | TTS로드 | TTS합성 | 재생 | 첫소리까지 | 전체 |"
)
LATENCY_TABLE_DIVIDER = (
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
)
_AUDIO_INDEX_RE = re.compile(r"^audio_(\d+)_")
_AUDIO_TEXT_RE = re.compile(r"^audio_\d+_(.+)$")
STABILITY_COUNTER_KEYS = (
    "critical_memory_events",
    "stt_force_unload_count",
    "llm_prompt_cache_flush_count",
    "system_state_snapshot_count",
)
_STABILITY_COUNTER_PATTERNS = {
    "critical_memory_events": re.compile(r"CRITICAL memory: \d+ MB", re.MULTILINE),
    "stt_force_unload_count": re.compile(r"forcing STT unload", re.MULTILINE),
    "llm_prompt_cache_flush_count": re.compile(
        r"LLM prompt cache flushed",
        re.MULTILINE,
    ),
    "system_state_snapshot_count": re.compile(
        r"System-state snapshot captured \(\d+ tokens\)",
        re.MULTILINE,
    ),
}
_SHUTDOWN_MANAGER: Any = None


@dataclass(frozen=True)
class RoundInput:
    """One bilingual interleaved E2E round input."""

    round_id: int
    lang: str
    wav_path: Path
    gt_text: str
    sequence_index: int
    source_round_id: int


@dataclass
class LoadCounters:
    """Resident-load counters collected from the shared model manager."""

    stt_load_count: int = 0
    tts_load_count: int = 0
    tts_load_error_count: int = 0


class ExpectedSTTProviderMismatch(RuntimeError):
    """Raised when ``--expect-stt-provider`` does not match turn telemetry."""


def _graceful_shutdown(signum: int, frame: Any) -> None:
    """Unload models before exiting on termination signals."""
    del frame
    logger.warning("Signal %d received, initiating graceful shutdown...", signum)
    manager = _SHUTDOWN_MANAGER
    if manager is not None:
        try:
            manager.unload_all(force=True)
        except Exception:
            logger.warning("Failed to unload models during signal shutdown", exc_info=True)
    raise SystemExit(0)


def _register_signal_handlers() -> None:
    """Register SIGTERM and SIGHUP handlers."""
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, _graceful_shutdown)
    signal.signal(signal.SIGTERM, _graceful_shutdown)


def _positive_int_or_none(value: str) -> int | None:
    """Parse a non-negative integer CLI override."""
    try:
        parsed = int(value)
    except ValueError as exc:
        msg = f"Expected a non-negative integer, got: {value}"
        raise argparse.ArgumentTypeError(msg) from exc
    if parsed < 0:
        msg = f"Expected a non-negative integer, got: {value}"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def _non_negative_int(value: str) -> int:
    """Argparse type validator: accepts int >= 0, rejects negatives."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--input-pad-ms must be an integer, got {value!r}",
        ) from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"--input-pad-ms must be >= 0, got {parsed}")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the bilingual interleaved audio E2E runner."""
    parser = argparse.ArgumentParser(
        description=(
            "Run interleaved Korean and English prerecorded child WAVs through "
            "the full Mungi STT -> LLM -> TTS pipeline."
        ),
    )
    parser.add_argument("--ko-dir", type=Path, required=True, help="Directory containing KO WAVs.")
    parser.add_argument("--en-dir", type=Path, required=True, help="Directory containing EN WAVs.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=(f"Root directory for timestamped run outputs (default: {DEFAULT_OUTPUT_ROOT})."),
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=DEFAULT_MODEL_DIR,
        help=f"Model root directory (default: {DEFAULT_MODEL_DIR}).",
    )
    parser.add_argument(
        "--stt-model",
        type=str,
        default=DEFAULT_STT_MODEL,
        help=f"STT model selector passed to ModelManager (default: {DEFAULT_STT_MODEL}).",
    )
    parser.add_argument(
        "--stt-language",
        type=str,
        default=DEFAULT_STT_LANGUAGE,
        help=(
            "Default STT language hint. Interleaved rounds still use per-round "
            "language-specific pipeline configs."
        ),
    )
    parser.add_argument(
        "--expect-stt-provider",
        choices=("cuda", "cpu"),
        default=None,
        help="Fail fast when a turn reports a different actual STT provider.",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=0,
        help="Maximum interleaved rounds to process (0 = all available).",
    )
    parser.add_argument(
        "--repeat-passes",
        type=int,
        choices=range(1, 21),
        metavar="N",
        default=1,
        help="Run the same capped bilingual round list N times in this process.",
    )
    parser.add_argument(
        "--input-pad-ms",
        type=_non_negative_int,
        default=200,
        help=(
            "Add symmetric leading + trailing silence padding (in ms) to each "
            "loaded WAV before pipeline ingestion. Default 200 ms addresses "
            "Silero VAD onset detection for first-turn audio. Set 0 to disable."
        ),
    )
    parser.add_argument(
        "--llm-n-gpu-layers",
        type=int,
        default=None,
        help="Override LLM GPU layer count for this run. CLI value wins over env.",
    )
    parser.add_argument(
        "--llm-max-tokens",
        type=int,
        default=None,
        help=(
            "Override LLM max_tokens for this run. Supersedes "
            "MUNGI_LLM_MAX_TOKENS env and PipelineConfig default "
            "(80). Range 1-4096."
        ),
    )
    parser.add_argument(
        "--conversation-per-lang",
        action="store_true",
        default=False,
        help=(
            "Suppress per-round pipeline.clear_history() so each language lane "
            "(KO / EN) is a continuous conversation across all rounds. Default: "
            "off (Wave 2 parity - clear every round)."
        ),
    )
    parser.add_argument(
        "--max-history-turns",
        type=_positive_int_or_none,
        default=None,
        help=(
            "Override PipelineConfig.max_history_turns. Default None means use "
            "PipelineConfig default (2)."
        ),
    )
    parser.add_argument(
        "--max-history-tokens",
        type=_positive_int_or_none,
        default=None,
        help=(
            "Override PipelineConfig.max_history_tokens. Default None means use "
            "PipelineConfig default (200)."
        ),
    )
    parser.add_argument(
        "--skip-tts",
        action="store_true",
        help="Skip real TTS loading/synthesis while still exercising STT -> LLM flow.",
    )
    parser.add_argument(
        "--tts-streaming",
        action="store_true",
        help="Enable sentence-level TTS streaming playback.",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip Linux-only Jetson preflight checks.",
    )
    parser.add_argument(
        "--output-device",
        type=str,
        default=DEFAULT_OUTPUT_DEVICE,
        help="Optional TTS output device passthrough.",
    )
    return parser


def _extract_audio_index(path: Path) -> int:
    """Extract the integer index from an ``audio_<N>_...`` filename."""
    match = _AUDIO_INDEX_RE.match(path.name)
    if match is None:
        msg = f"Filename does not match 'audio_<N>_...' pattern: {path.name}"
        raise ValueError(msg)
    return int(match.group(1))


def filename_to_gt_text(path: Path) -> str:
    """Recover readable ground-truth text from an ``audio_<N>_...`` filename."""
    match = _AUDIO_TEXT_RE.match(path.stem)
    if match is None:
        msg = f"Filename does not match 'audio_<N>_...' pattern: {path.name}"
        raise ValueError(msg)
    return " ".join(match.group(1).replace("_", " ").split())


def _discover_language_wavs(directory: Path) -> list[Path]:
    """Return sorted ``audio_*.wav`` files from *directory*."""
    if not directory.exists():
        msg = f"Audio directory does not exist: {directory}"
        raise FileNotFoundError(msg)
    if not directory.is_dir():
        msg = f"Audio path is not a directory: {directory}"
        raise NotADirectoryError(msg)
    return sorted(directory.glob("audio_*.wav"), key=_extract_audio_index)


def discover_round_pairs(
    ko_dir: Path,
    en_dir: Path,
    max_rounds: int = 0,
) -> list[RoundInput]:
    """Discover bilingual WAV pairs and interleave them as KO/EN rounds."""
    if max_rounds < 0:
        msg = "--max-rounds must be >= 0."
        raise ValueError(msg)

    ko_wavs = _discover_language_wavs(ko_dir)
    en_wavs = _discover_language_wavs(en_dir)

    if not ko_wavs and not en_wavs:
        msg = (
            "No audio_*.wav files found in either bilingual input directory: "
            f"ko_dir={ko_dir}, en_dir={en_dir}"
        )
        raise ValueError(msg)
    if not ko_wavs:
        msg = f"No Korean audio_*.wav files found in {ko_dir}"
        raise ValueError(msg)
    if not en_wavs:
        msg = f"No English audio_*.wav files found in {en_dir}"
        raise ValueError(msg)

    pair_count = min(len(ko_wavs), len(en_wavs))
    if pair_count == 0:
        msg = (
            "No bilingual round pairs are available because one language directory is empty: "
            f"ko={len(ko_wavs)}, en={len(en_wavs)}"
        )
        raise ValueError(msg)

    if len(ko_wavs) != len(en_wavs):
        logger.warning(
            "Input WAV count imbalance detected: ko=%d en=%d. Interleaving only %d pairs.",
            len(ko_wavs),
            len(en_wavs),
            pair_count,
        )

    if max_rounds > 0:
        pair_count = min(pair_count, max_rounds)

    rounds: list[RoundInput] = []
    for sequence_index in range(pair_count):
        rounds.append(
            RoundInput(
                round_id=len(rounds) + 1,
                lang="ko",
                wav_path=ko_wavs[sequence_index],
                gt_text=filename_to_gt_text(ko_wavs[sequence_index]),
                sequence_index=sequence_index,
                source_round_id=sequence_index + 1,
            ),
        )
        rounds.append(
            RoundInput(
                round_id=len(rounds) + 1,
                lang="en",
                wav_path=en_wavs[sequence_index],
                gt_text=filename_to_gt_text(en_wavs[sequence_index]),
                sequence_index=sequence_index,
                source_round_id=sequence_index + 1,
            ),
        )

    return rounds


def _load_librosa_module() -> Any:
    """Import and return the librosa module lazily."""
    return importlib.import_module("librosa")


def _load_audio_input(
    path: Path,
    *,
    input_pad_ms: int = 0,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Load WAV, normalize to float32, and apply symmetric silence padding.

    Returns:
        ``(padded_float_audio, raw_pcm_audio, original_duration_ms, padded_duration_ms)``.
        ``raw_pcm_audio`` is derived from the original source, before padding, so trace
        artifacts stay reproducible across different ``--input-pad-ms`` values.
    """
    if input_pad_ms < 0:
        msg = f"input_pad_ms must be >= 0, got {input_pad_ms}"
        raise ValueError(msg)
    librosa = _load_librosa_module()
    audio, _sample_rate = librosa.load(path, sr=DEFAULT_SAMPLE_RATE, mono=True)
    float_audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    pcm_audio = np.clip(np.round(float_audio * 32767.0), -32768, 32767).astype(np.int16)
    original_duration_ms = (
        float_audio.size / float(DEFAULT_SAMPLE_RATE) * 1000.0 if float_audio.size else 0.0
    )
    if input_pad_ms > 0:
        pad_samples = int(round(DEFAULT_SAMPLE_RATE * input_pad_ms / 1000.0))
        silence = np.zeros(pad_samples, dtype=float_audio.dtype)
        padded_float_audio = np.concatenate([silence, float_audio, silence])
        padded_duration_ms = original_duration_ms + 2 * input_pad_ms
    else:
        padded_float_audio = float_audio
        padded_duration_ms = original_duration_ms
    return padded_float_audio, pcm_audio, original_duration_ms, padded_duration_ms


def _configure_logging(output_dir: Path) -> Path:
    """Configure stdout and file logging for one run directory."""
    log_path = output_dir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
        ],
        force=True,
    )
    return log_path


def _run_preflight(skip: bool) -> None:
    """Run the shared Linux/Jetson preflight helper."""
    from scripts.e2e_60rounds_text_tts import run_preflight

    run_preflight(skip=skip)


def _get_runtime_classes() -> tuple[Any, Any, Any, Any]:
    """Import runtime classes lazily so ``--help`` stays lightweight."""
    from core.model_manager import ManagerConfig, ModelManager
    from core.pipeline import ConversationPipeline, PipelineConfig

    return ManagerConfig, ModelManager, PipelineConfig, ConversationPipeline


def _get_write_wav() -> Any:
    """Import the shared WAV writer lazily."""
    from scripts.test_tts import write_wav

    return write_wav


def _env_flag_enabled(name: str) -> bool:
    """Parse a boolean environment flag using common truthy and falsy aliases."""
    raw_value = os.getenv(name, "").strip().lower()
    if not raw_value:
        return False
    if raw_value in _TRUE_ENV_VALUES:
        return True
    if raw_value in _FALSE_ENV_VALUES:
        return False
    logger.warning("Ignoring unrecognized boolean env flag: %s=%r", name, raw_value)
    return False


def _tts_streaming_enabled(args: argparse.Namespace) -> bool:
    """Resolve sentence-streaming enablement from CLI and environment."""
    if bool(args.skip_tts):
        return False
    if bool(args.tts_streaming):
        return True
    return _env_flag_enabled("MUNGI_TTS_STREAMING")


def _apply_llm_max_tokens_override(
    *,
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> None:
    """Validate and apply the per-run LLM token override via environment."""
    if args.llm_max_tokens is None:
        return
    if args.llm_max_tokens <= 0 or args.llm_max_tokens > 4096:
        parser.error(
            f"--llm-max-tokens must be in 1..4096, got {args.llm_max_tokens}",
        )
    os.environ["MUNGI_LLM_MAX_TOKENS"] = str(args.llm_max_tokens)


def _resolve_llm_n_gpu_layers(args: argparse.Namespace) -> int | None:
    """Resolve LLM GPU-layer override with CLI value taking precedence over env."""
    cli_value = getattr(args, "llm_n_gpu_layers", None)
    if cli_value is not None:
        os.environ["MUNGI_LLM_N_GPU_LAYERS"] = str(cli_value)
        return int(cli_value)

    raw_env = os.getenv("MUNGI_LLM_N_GPU_LAYERS", "").strip()
    if not raw_env:
        return None
    try:
        return int(raw_env)
    except ValueError:
        logger.warning("Ignoring invalid MUNGI_LLM_N_GPU_LAYERS=%r", raw_env)
        return None


def _sherpa_onnx_version() -> str | None:
    """Return the installed sherpa-onnx version when importable."""
    try:
        import sherpa_onnx  # type: ignore[import-not-found,import-untyped]
    except ImportError:
        return None
    version = getattr(sherpa_onnx, "__version__", None)
    return str(version) if version is not None else None


def _model_sha256_summary() -> dict[str, str | None]:
    """Return the stable model-hash object; values are null when not computed."""
    return {
        "gemma": None,
        "qwen3_asr": None,
        "supertonic": None,
    }


def _process_rss_mb() -> float | None:
    """Return current process RSS in MiB using psutil or a Linux procfs fallback."""
    try:
        import psutil  # type: ignore[import-not-found,import-untyped]
    except ImportError:
        status_path = Path("/proc/self/status")
        if not status_path.exists():
            return None
        try:
            for line in status_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return round(int(parts[1]) / 1024.0, 3)
        except (OSError, ValueError):
            return None
        return None
    try:
        rss_mb = float(psutil.Process(os.getpid()).memory_info().rss) / 1024.0**2
        return round(rss_mb, 3)
    except (OSError, AttributeError):
        return None


def _wrap_load_counters(manager: Any, counters: LoadCounters) -> None:
    """Wrap ``manager.load`` to count STT/TTS load attempts and TTS load failures."""
    original_load = getattr(manager, "load", None)
    if not callable(original_load):
        return

    def _counting_load(model_type: Any, *args: Any, **kwargs: Any) -> Any:
        model_value = getattr(model_type, "value", str(model_type)).lower()
        is_stt = model_value == "stt"
        is_tts = model_value == "tts"
        if is_stt:
            counters.stt_load_count += 1
        if is_tts:
            counters.tts_load_count += 1
        try:
            return original_load(model_type, *args, **kwargs)
        except Exception:
            if is_tts:
                counters.tts_load_error_count += 1
            raise

    manager.load = _counting_load


def _tts_wav_stats(path: Path | None) -> tuple[int, int]:
    """Return persisted TTS WAV byte and frame counts."""
    if path is None or not path.exists():
        return 0, 0
    byte_count = path.stat().st_size
    try:
        with wave.open(str(path), "rb") as wav_file:
            frame_count = wav_file.getnframes()
    except wave.Error:
        frame_count = 0
    return byte_count, frame_count


def _get_thermal_helpers() -> tuple[Any, Any]:
    """Import the shared tegrastats monitor and summary helper lazily."""
    from scripts.e2e_60rounds_text_tts import TegrastatsMonitor, _build_thermal_summary

    return TegrastatsMonitor, _build_thermal_summary


def _thermal_max_c(summary: dict[str, Any]) -> float | None:
    """Return the maximum CPU/GPU temperature from a thermal summary."""
    values: list[float] = []
    for key in ("cpu_temp_c", "gpu_temp_c"):
        item = summary.get(key)
        if isinstance(item, dict):
            maximum = _optional_float(item.get("max"))
            if maximum is not None:
                values.append(maximum)
    return max(values) if values else None


def _build_thermal_curve(
    snapshots: list[dict[str, Any]],
    *,
    interval_s: float = 30.0,
) -> list[dict[str, float | int | None]]:
    """Downsample tegrastats snapshots to a compact thermal curve."""
    rows: list[dict[str, float | int | None]] = []
    last_elapsed: float | None = None
    for snapshot in snapshots:
        elapsed = _optional_float(snapshot.get("elapsed_s"))
        if elapsed is None:
            elapsed = 0.0 if last_elapsed is None else last_elapsed + interval_s
        if last_elapsed is not None and elapsed - last_elapsed < interval_s:
            continue
        rows.append(
            {
                "t_s": round(elapsed, 3),
                "cpu_temp_c": _optional_float(snapshot.get("cpu_temp_c")),
                "gpu_temp_c": _optional_float(snapshot.get("gpu_temp_c")),
                "ram_used_mb": _optional_int(snapshot.get("ram_used_mb")),
                "gr3d_freq_pct": _optional_int(snapshot.get("gr3d_freq_pct")),
            },
        )
        last_elapsed = elapsed
    return rows


def _start_tegrastats_monitor(log_path: Path) -> tuple[Any | None, bool]:
    """Start tegrastats monitoring, returning ``False`` on graceful degradation."""
    tegrastats_monitor_cls, _summary_helper = _get_thermal_helpers()
    monitor = tegrastats_monitor_cls(log_path)
    try:
        enabled = bool(monitor.start())
    except Exception:
        logger.warning("TegrastatsMonitor start failed; continuing", exc_info=True)
        return monitor, False
    if not enabled:
        logger.warning("TegrastatsMonitor unavailable; continuing without thermal artifacts")
    return monitor, enabled


def _write_thermal_artifacts(
    *,
    monitor: Any,
    summary_path: Path,
    curve_path: Path,
    started_at: float,
) -> None:
    """Stop the monitor and write summary plus downsampled thermal curve artifacts."""
    try:
        monitor.stop()
    except Exception:
        logger.warning("Failed to stop TegrastatsMonitor cleanly", exc_info=True)
    _monitor_cls, build_summary = _get_thermal_helpers()
    snapshots = list(getattr(monitor, "snapshots", []))
    thermal_summary = build_summary(snapshots)
    thermal_summary["thermal_max_c"] = _thermal_max_c(thermal_summary)
    thermal_summary["duration_s"] = round(time.monotonic() - started_at, 3)
    _write_json_atomic(summary_path, thermal_summary)
    _write_json_atomic(curve_path, _build_thermal_curve(snapshots))


def _latest_system_ram_mb(monitor: Any | None) -> float | None:
    """Return the latest tegrastats RAM-used value in MiB."""
    if monitor is None:
        return None
    snapshots = list(getattr(monitor, "snapshots", []))
    for snapshot in reversed(snapshots):
        ram_used_mb = _optional_float(snapshot.get("ram_used_mb"))
        if ram_used_mb is not None:
            return ram_used_mb
    return None


def _load_generated_wav(path: Path) -> tuple[np.ndarray, int]:
    """Load a mono 16-bit PCM WAV artifact written by the streaming helper."""
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        raw_frames = wav_file.readframes(frame_count)

    if sample_width != 2:
        msg = f"Unsupported WAV sample width: {sample_width}"
        raise ValueError(msg)

    audio = np.frombuffer(raw_frames, dtype=np.int16).astype(np.float32) / 32767.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1).astype(np.float32)
    return audio.reshape(-1), sample_rate


@contextlib.contextmanager
def _maybe_externalize_tts(pipeline: Any, enabled: bool) -> Iterator[None]:
    """Patch the pipeline so sentence streaming can run after text generation."""
    if not enabled:
        yield
        return

    missing = object()
    original_run_tts = getattr(pipeline, "_run_tts", missing)
    original_play_audio = getattr(pipeline, "_play_audio_out", missing)
    original_post_tts_unload = getattr(
        pipeline,
        "_maybe_unload_tts_after_success",
        missing,
    )
    mm = getattr(pipeline, "_mm", None)
    original_mm_load = getattr(mm, "load", missing) if mm is not None else missing
    original_mm_unload_tts = getattr(mm, "unload_tts", missing) if mm is not None else missing

    def _patched_run_tts(text: str, language: str = "ko") -> tuple[np.ndarray, int]:
        del text, language
        return np.zeros(0, dtype=np.float32), DEFAULT_SKIP_TTS_SAMPLE_RATE

    def _patched_play_audio(audio_samples: Any, sample_rate: int) -> None:
        del audio_samples, sample_rate
        return

    def _patched_post_tts_unload() -> None:
        return

    def _patched_mm_load(model_type: ModelType, *args: Any, **kwargs: Any) -> Any:
        if model_type == ModelType.TTS:
            return None
        assert original_mm_load is not missing
        return cast(Any, original_mm_load)(model_type, *args, **kwargs)

    def _patched_mm_unload_tts(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        return

    logger.info(
        "Sentence streaming mode: externalizing TTS lifecycle (skip load/unload on pipeline)",
    )
    pipeline._run_tts = _patched_run_tts
    pipeline._play_audio_out = _patched_play_audio
    pipeline._maybe_unload_tts_after_success = _patched_post_tts_unload
    if mm is not None:
        mm.load = _patched_mm_load
        mm.unload_tts = _patched_mm_unload_tts
    try:
        yield
    finally:
        if original_run_tts is missing:
            delattr(pipeline, "_run_tts")
        else:
            pipeline._run_tts = original_run_tts

        if original_play_audio is missing:
            delattr(pipeline, "_play_audio_out")
        else:
            pipeline._play_audio_out = original_play_audio

        if original_post_tts_unload is missing:
            delattr(pipeline, "_maybe_unload_tts_after_success")
        else:
            pipeline._maybe_unload_tts_after_success = original_post_tts_unload

        if mm is not None:
            if original_mm_load is missing:
                delattr(mm, "load")
            else:
                mm.load = original_mm_load

            if original_mm_unload_tts is missing:
                delattr(mm, "unload_tts")
            else:
                mm.unload_tts = original_mm_unload_tts


def _apply_sentence_streaming_tts(
    *,
    result: Any,
    manager: Any,
    output_device: str | None,
) -> SentenceSynthesisResult | None:
    """Run sentence-streaming TTS and hydrate the pipeline result with its metrics."""
    response_text = str(getattr(result, "response_text", "")).strip()
    if not response_text:
        return None

    metrics = getattr(result, "metrics", None)
    previous_tts_time_s = float(getattr(metrics, "tts_time_s", 0.0)) if metrics is not None else 0.0
    manager_config = getattr(manager, "config", None)
    voice_style = str(getattr(manager_config, "tts_voice_style", "F2"))
    model_dir_value = getattr(manager_config, "tts_model_dir", None)
    model_dir = str(model_dir_value) if model_dir_value is not None else None
    sentence_result = synthesize_to_speaker_by_sentence(
        response_text,
        voice_style,
        model_dir=model_dir,
        output_device=output_device,
    )

    if metrics is not None:
        metrics.tts_first_chunk_ms = sentence_result.first_chunk_ms
        metrics.tts_time_s = sentence_result.total_duration_ms / 1000.0
        metrics.total_time_s = max(
            float(getattr(metrics, "total_time_s", 0.0)) - previous_tts_time_s + metrics.tts_time_s,
            0.0,
        )

    full_wav_path = sentence_result.full_wav_path
    if full_wav_path:
        wav_path = Path(full_wav_path)
        try:
            if wav_path.exists():
                audio_out, sample_rate = _load_generated_wav(wav_path)
                result.audio_samples = audio_out
                result.sample_rate = sample_rate
        finally:
            if wav_path.exists():
                wav_path.unlink()
    return sentence_result


@contextlib.contextmanager
def _maybe_skip_tts(pipeline: Any, manager: Any, skip_tts: bool) -> Iterator[None]:
    """Patch the pipeline so ``--skip-tts`` avoids real TTS work."""
    if not skip_tts:
        yield
        return

    original_load = getattr(manager, "load", None)
    original_unload_tts = getattr(manager, "unload_tts", None)
    original_run_tts = getattr(pipeline, "_run_tts", None)
    original_play_audio = getattr(pipeline, "_play_audio_out", None)

    def _patched_load(model_type: Any) -> None:
        if getattr(model_type, "value", None) == "tts":
            return
        if callable(original_load):
            original_load(model_type)

    def _patched_unload_tts() -> None:
        return

    def _patched_run_tts(text: str, language: str = "ko") -> tuple[np.ndarray, int]:
        del text, language
        return np.zeros(0, dtype=np.float32), DEFAULT_SKIP_TTS_SAMPLE_RATE

    def _patched_play_audio(audio_samples: Any, sample_rate: int) -> None:
        del audio_samples, sample_rate
        return

    manager.load = _patched_load
    manager.unload_tts = _patched_unload_tts
    pipeline._run_tts = _patched_run_tts
    pipeline._play_audio_out = _patched_play_audio
    try:
        yield
    finally:
        manager.load = original_load
        manager.unload_tts = original_unload_tts
        pipeline._run_tts = original_run_tts
        pipeline._play_audio_out = original_play_audio


def _make_output_dirs(output_root: Path) -> dict[str, Path]:
    """Create and return the run directory tree."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"qwen3_mix_{timestamp}"
    input_wav_dir = run_dir / "input_wavs"
    response_wav_dir = run_dir / "response_wavs"
    run_dir.mkdir(parents=True, exist_ok=True)
    input_wav_dir.mkdir(parents=True, exist_ok=True)
    response_wav_dir.mkdir(parents=True, exist_ok=True)
    return {
        "run_dir": run_dir,
        "input_wavs": input_wav_dir,
        "response_wavs": response_wav_dir,
    }


def _link_or_copy_input(src: Path, dst: Path) -> None:
    """Create a traceable input artifact using symlink or copy."""
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if os.name == "nt":
        shutil.copy2(src, dst)
        return
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def _write_response_wav(path: Path, audio_samples: Any, sample_rate: int) -> None:
    """Persist synthesized response audio as a WAV file."""
    write_wav = _get_write_wav()
    normalized = np.asarray(audio_samples, dtype=np.float32).reshape(-1)
    write_wav(path, normalized, sample_rate)


def _append_jsonl_record(path: Path, record: dict[str, Any]) -> None:
    """Append one JSON record to a JSONL file."""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _sanitize_error_message(exc: Exception) -> str:
    """Return a compact single-line exception message for JSON artifacts."""
    message = str(exc).strip() or exc.__class__.__name__
    return " ".join(message.split())[:500]


def _write_json_atomic(path: Path, payload: Any) -> None:
    """Atomically write JSON to disk."""
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    temp_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temp_path, path)


def _empty_stability_counters() -> dict[str, int]:
    return {key: 0 for key in STABILITY_COUNTER_KEYS}


def parse_run_log_stability_counters(log_path: Path) -> dict[str, int]:
    """Parse run.log stability event counters without requiring Jetson hardware."""
    if not log_path.exists():
        logger.warning("Missing run.log for stability counters: %s", log_path)
        return _empty_stability_counters()

    counters = _empty_stability_counters()
    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                for key, pattern in _STABILITY_COUNTER_PATTERNS.items():
                    if pattern.search(line):
                        counters[key] += 1
    except OSError as exc:
        logger.warning("Failed to parse run.log stability counters from %s: %s", log_path, exc)
        return _empty_stability_counters()
    return counters


def _stt_total_ms(metrics: Any) -> float:
    """Return STT load + inference time in milliseconds."""
    return (float(metrics.stt_load_time_s) + float(metrics.stt_time_s)) * 1000.0


def _maybe_float(value: Any) -> float | None:
    """Return ``value`` coerced to float, or ``None`` if coercion fails."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_sound_ms(metrics: Any) -> float:
    """Return latency until synthesized audio becomes available in milliseconds.

    When streaming TTS is active (``metrics.tts_first_chunk_ms`` is set
    to a positive value), substitute the first-chunk duration for the
    full synthesis duration because the user perceives first sound at
    first-chunk dispatch, not at full-response completion.
    """
    first_chunk_ms = _maybe_float(getattr(metrics, "tts_first_chunk_ms", None))
    tts_contribution_s: float
    if first_chunk_ms is not None and first_chunk_ms > 0.0:
        tts_contribution_s = first_chunk_ms / 1000.0
    else:
        tts_contribution_s = float(metrics.tts_time_s)

    return (
        float(metrics.vad_time_s)
        + float(metrics.stt_load_time_s)
        + float(metrics.stt_time_s)
        + float(metrics.llm_load_time_s)
        + float(metrics.llm_time_s)
        + float(metrics.tts_load_time_s)
        + tts_contribution_s
    ) * 1000.0


def _timings_ms(metrics: Any) -> dict[str, float | None]:
    """Convert pipeline metrics to a millisecond dictionary."""
    return {
        "vad_ms": float(metrics.vad_time_s) * 1000.0,
        "stt_load_ms": float(metrics.stt_load_time_s) * 1000.0,
        "stt_ms": float(metrics.stt_time_s) * 1000.0,
        "stt_total_ms": _stt_total_ms(metrics),
        "llm_load_ms": float(metrics.llm_load_time_s) * 1000.0,
        "llm_ttft_ms": float(metrics.llm_ttft_s) * 1000.0,
        "llm_ms": float(metrics.llm_time_s) * 1000.0,
        "tts_load_ms": float(metrics.tts_load_time_s) * 1000.0,
        "tts_first_chunk_ms": _maybe_float(getattr(metrics, "tts_first_chunk_ms", None)),
        "tts_ms": float(metrics.tts_time_s) * 1000.0,
        "playback_ms": float(metrics.playback_time_s) * 1000.0,
        "first_sound_ms": _first_sound_ms(metrics),
        "total_ms": float(metrics.total_time_s) * 1000.0,
    }


def _normalize_metric_text(text: str) -> str:
    """Normalize text for simple CER/WER estimation."""
    return " ".join(text.strip().split()).lower()


def _levenshtein_distance(source: Sequence[Any], target: Sequence[Any]) -> int:
    """Compute Levenshtein distance for generic sequences."""
    if not source:
        return len(target)
    if not target:
        return len(source)

    previous = list(range(len(target) + 1))
    for row_index, source_item in enumerate(source, start=1):
        current = [row_index]
        for column_index, target_item in enumerate(target, start=1):
            cost = 0 if source_item == target_item else 1
            current.append(
                min(
                    previous[column_index] + 1,
                    current[column_index - 1] + 1,
                    previous[column_index - 1] + cost,
                ),
            )
        previous = current
    return previous[-1]


def _error_rate(reference: Sequence[Any], hypothesis: Sequence[Any]) -> float:
    """Return normalized edit distance for one reference/hypothesis pair."""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return _levenshtein_distance(reference, hypothesis) / float(len(reference))


def _avg(values: list[float]) -> float:
    """Return the arithmetic mean or ``0.0`` for an empty list."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def _avg_present(values: list[float]) -> float | None:
    """Return the average of positive values, or ``None`` when unavailable."""
    present_values = [value for value in values if value > 0.0]
    if not present_values:
        return None
    return _avg(present_values)


def _avg_defined(values: list[float | None]) -> float | None:
    """Return the average of defined values, or ``None`` when unavailable."""
    present_values = [float(value) for value in values if value is not None]
    if not present_values:
        return None
    return _avg(present_values)


def _optional_int(value: Any) -> int | None:
    """Return an integer when coercion succeeds, else ``None``."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    """Return a float when coercion succeeds, else ``None``."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_language_key(value: Any) -> str | None:
    """Normalize a language key for per-language turn indexing."""
    normalized = str(value).strip().lower()
    return normalized or None


def _vad_miss_reason(vad_miss: bool, audio_duration_ms: float | None) -> str | None:
    """Return the row-level VAD miss reason for a non-detected speech clip."""
    if not vad_miss:
        return None
    if audio_duration_ms is None:
        return "unknown_no_segments"
    if audio_duration_ms < VAD_AUDIO_TOO_SHORT_MS:
        return "audio_too_short"
    return "silence_detected"


def _stt_provider_mismatch_message(actual: Any, expected: str | None) -> str | None:
    """Return a fail-fast message when the actual STT provider mismatches."""
    if expected is None:
        return None
    actual_provider = str(actual).strip().lower() if actual is not None else None
    if actual_provider == expected:
        return None
    return f"STT provider mismatch: expected actual provider {expected!r}, got {actual_provider!r}."


def _summarize_stt_provider_actual(records: list[dict[str, Any]]) -> str | list[str] | None:
    """Return the mode actual STT provider, or distinct providers when mixed."""
    values = [
        str(value).strip().lower()
        for record in records
        for value in (record.get("stt_provider_actual"),)
        if value is not None and str(value).strip()
    ]
    if not values:
        return None
    distinct_values = sorted(set(values))
    if len(distinct_values) == 1:
        return distinct_values[0]
    counts = Counter(values)
    logger.warning(
        "Mixed actual STT providers observed: %s",
        ", ".join(f"{name}={counts[name]}" for name in distinct_values),
    )
    return distinct_values


def _format_latency_cell(value: float | None) -> str:
    """Format one latency table cell, using ``-`` for unavailable stages."""
    if value is None or value <= 0.0:
        return "-"
    return f"{value:.1f}"


def _build_latency_table(records: list[dict[str, Any]]) -> str:
    """Build the required markdown latency table in milliseconds."""
    rows = [LATENCY_TABLE_HEADER, LATENCY_TABLE_DIVIDER]

    def _row(turn_name: str, timings: dict[str, float | None]) -> str:
        return (
            f"| {turn_name} | "
            f"{_format_latency_cell(timings.get('vad_ms'))} | "
            f"{_format_latency_cell(timings.get('stt_total_ms'))} | "
            f"{_format_latency_cell(timings.get('llm_load_ms'))} | "
            f"{_format_latency_cell(timings.get('llm_ttft_ms'))} | "
            f"{_format_latency_cell(timings.get('llm_ms'))} | "
            f"{_format_latency_cell(timings.get('tts_load_ms'))} | "
            f"{_format_latency_cell(timings.get('tts_ms'))} | "
            f"{_format_latency_cell(timings.get('playback_ms'))} | "
            f"{_format_latency_cell(timings.get('first_sound_ms'))} | "
            f"{_format_latency_cell(timings.get('total_ms'))} |"
        )

    for record in records:
        rows.append(_row(f"r{int(record['round_id']):02d}_{record['lang']}", record["timings_ms"]))

    avg_timings: dict[str, float | None] = {
        "vad_ms": _avg_present([record["timings_ms"]["vad_ms"] for record in records]),
        "stt_total_ms": _avg_present([record["timings_ms"]["stt_total_ms"] for record in records]),
        "llm_load_ms": _avg_present([record["timings_ms"]["llm_load_ms"] for record in records]),
        "llm_ttft_ms": _avg_present([record["timings_ms"]["llm_ttft_ms"] for record in records]),
        "llm_ms": _avg_present([record["timings_ms"]["llm_ms"] for record in records]),
        "tts_load_ms": _avg_present([record["timings_ms"]["tts_load_ms"] for record in records]),
        "tts_ms": _avg_present([record["timings_ms"]["tts_ms"] for record in records]),
        "playback_ms": _avg_present([record["timings_ms"]["playback_ms"] for record in records]),
        "first_sound_ms": _avg_present(
            [record["timings_ms"]["first_sound_ms"] for record in records],
        ),
        "total_ms": _avg_present([record["timings_ms"]["total_ms"] for record in records]),
    }
    rows.append(_row("AVG", avg_timings))
    return "\n".join(rows)


def _relative_to(base_dir: Path, path: Path | None) -> str | None:
    """Return a repo-local relative string when *path* is present."""
    if path is None:
        return None
    return str(path.relative_to(base_dir))


def _make_round_record(
    *,
    output_dir: Path,
    round_input: RoundInput,
    duration_s: float,
    audio_duration_ms: float,
    audio_padded_ms: float,
    result: TurnResult,
    input_trace_path: Path,
    response_wav_path: Path | None,
    turn_index_per_lang: int,
    pass_id: str,
    global_turn_id: int,
    system_ram_mb: float | None,
    process_rss_mb: float | None,
    stt_provider_configured: str,
    stt_provider_requested: str | None,
) -> dict[str, Any]:
    """Build one JSON-serializable round record."""
    timings_ms = _timings_ms(result.metrics)
    tts_first_chunk_ms = getattr(result.metrics, "tts_first_chunk_ms", None)
    llm_cache_hit_tokens = getattr(result.metrics, "llm_cache_hit_tokens", None)
    llm_cache_miss_tokens = getattr(result.metrics, "llm_cache_miss_tokens", None)
    tts_wav_bytes, tts_wav_frames = _tts_wav_stats(response_wav_path)
    error_text = str(getattr(result, "error", "") or "").lower()
    tts_synth_error = "tts" in error_text and "load" not in error_text
    tts_load_error = "tts" in error_text and "load" in error_text
    speech_segments = int(getattr(result.metrics, "speech_segments", 0))
    vad_miss = bool(round_input.gt_text) and speech_segments == 0
    core_success = bool(getattr(result, "success", False))
    row_success = core_success and not vad_miss
    if row_success:
        failure_reason = None
    elif not core_success:
        failure_reason = "runtime_error"
    else:
        failure_reason = "vad_miss"
    return {
        "pass_id": pass_id,
        "global_turn_id": global_turn_id,
        "round_id": round_input.round_id,
        "source_round_id": round_input.source_round_id,
        "lang": round_input.lang,
        "sequence_index": round_input.sequence_index,
        "wav_path": str(round_input.wav_path),
        "input_trace_wav": _relative_to(output_dir, input_trace_path),
        "response_wav": _relative_to(output_dir, response_wav_path),
        "gt_text": round_input.gt_text,
        "stt_pred": str(result.user_text),
        "llm_response": str(result.response_text),
        "detected_language": str(getattr(result, "detected_language", round_input.lang)),
        "stt_provider_actual": getattr(result.metrics, "stt_provider_actual", None),
        "stt_provider_configured": stt_provider_configured,
        "stt_provider_requested": stt_provider_requested,
        "core_success": core_success,
        "success": row_success,
        "failure_reason": failure_reason,
        "error": getattr(result, "error", None),
        "duration_s": round(duration_s, 3),
        "audio_duration_ms": round(audio_duration_ms, 3),
        "audio_padded_ms": round(audio_padded_ms, 3),
        "speech_segments": speech_segments,
        "vad_miss": vad_miss,
        "vad_miss_reason": _vad_miss_reason(vad_miss, audio_duration_ms),
        "llm_tokens": int(getattr(result.metrics, "llm_tokens", 0)),
        "llm_model_fallback_used": bool(getattr(result.metrics, "llm_model_fallback_used", False)),
        "llm_model_path_actual": getattr(result.metrics, "llm_model_path_actual", None),
        "llm_model_fallback_reason": getattr(
            result.metrics,
            "llm_model_fallback_reason",
            None,
        ),
        "tts_first_chunk_ms": (
            round(float(tts_first_chunk_ms), 3) if tts_first_chunk_ms is not None else None
        ),
        "llm_cache_hit_tokens": (
            int(llm_cache_hit_tokens) if llm_cache_hit_tokens is not None else None
        ),
        "llm_cache_miss_tokens": (
            int(llm_cache_miss_tokens) if llm_cache_miss_tokens is not None else None
        ),
        "turn_index_per_lang": int(turn_index_per_lang),
        "template_topic_id": getattr(result.metrics, "template_topic_id", None),
        "template_mode": getattr(result.metrics, "template_mode", None),
        "template_matched": bool(getattr(result.metrics, "template_matched", False)),
        "tts_wav_bytes": tts_wav_bytes,
        "tts_wav_frames": tts_wav_frames,
        "tts_synth_error": tts_synth_error,
        "tts_load_error": tts_load_error,
        "system_ram_mb": system_ram_mb,
        "process_rss_mb": process_rss_mb,
        "timings_ms": {
            name: round(value, 3) if value is not None else None
            for name, value in timings_ms.items()
        },
    }


def _make_pass_failure_record(
    *,
    output_dir: Path,
    pass_id: str,
    global_turn_id: int,
    exc: Exception,
    round_input: RoundInput | None = None,
    input_trace_path: Path | None = None,
    turn_index_per_lang: int | None = None,
    system_ram_mb: float | None = None,
    process_rss_mb: float | None = None,
    audio_duration_ms: float | None = None,
    audio_padded_ms: float | None = None,
    stt_provider_configured: str = "",
    stt_provider_requested: str | None = None,
) -> dict[str, Any]:
    """Build a schema-compatible marker for a pass-level failure."""
    error_message = _sanitize_error_message(exc)
    round_id = round_input.round_id if round_input is not None else 0
    source_round_id = round_input.source_round_id if round_input is not None else 0
    lang = round_input.lang if round_input is not None else "unknown"
    sequence_index = round_input.sequence_index if round_input is not None else 0
    wav_path = str(round_input.wav_path) if round_input is not None else None
    gt_text = round_input.gt_text if round_input is not None else ""
    empty_timings = {
        "vad_ms": 0.0,
        "stt_load_ms": 0.0,
        "stt_ms": 0.0,
        "stt_total_ms": 0.0,
        "llm_load_ms": 0.0,
        "llm_ttft_ms": 0.0,
        "llm_ms": 0.0,
        "tts_load_ms": 0.0,
        "tts_first_chunk_ms": None,
        "tts_ms": 0.0,
        "playback_ms": 0.0,
        "first_sound_ms": 0.0,
        "total_ms": 0.0,
    }
    return {
        "record_type": "pass_failure",
        "pass_id": pass_id,
        "global_turn_id": global_turn_id,
        "round_id": round_id,
        "source_round_id": source_round_id,
        "lang": lang,
        "sequence_index": sequence_index,
        "wav_path": wav_path,
        "input_trace_wav": _relative_to(output_dir, input_trace_path),
        "response_wav": None,
        "gt_text": gt_text,
        "stt_pred": "",
        "llm_response": "",
        "detected_language": lang,
        "stt_provider_actual": None,
        "stt_provider_configured": stt_provider_configured,
        "stt_provider_requested": stt_provider_requested,
        "core_success": False,
        "success": False,
        "failure_reason": "runtime_error",
        "error": error_message,
        "error_message": error_message,
        "duration_s": 0.0,
        "audio_duration_ms": round(float(audio_duration_ms), 3)
        if audio_duration_ms is not None
        else None,
        "audio_padded_ms": round(float(audio_padded_ms), 3)
        if audio_padded_ms is not None
        else None,
        "speech_segments": 0,
        "vad_miss": False,
        "vad_miss_reason": None,
        "llm_tokens": 0,
        "llm_model_fallback_used": False,
        "llm_model_path_actual": None,
        "llm_model_fallback_reason": None,
        "tts_first_chunk_ms": None,
        "llm_cache_hit_tokens": None,
        "llm_cache_miss_tokens": None,
        "turn_index_per_lang": turn_index_per_lang,
        "template_topic_id": None,
        "template_mode": None,
        "template_matched": False,
        "tts_wav_bytes": 0,
        "tts_wav_frames": 0,
        "tts_synth_error": False,
        "tts_load_error": False,
        "system_ram_mb": system_ram_mb,
        "process_rss_mb": process_rss_mb,
        "timings_ms": empty_timings,
    }


def _build_summary(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    rounds_path: Path,
    log_path: Path,
    records: list[dict[str, Any]],
    llm_n_gpu_layers_resolved: int | None = None,
    stt_provider_resolved: str | None = None,
    stt_provider_configured: str | None = None,
    stt_provider_requested: str | None = None,
    load_counters: LoadCounters | None = None,
    passes_failed: list[str] | None = None,
) -> dict[str, Any]:
    """Build the aggregate summary payload for the completed run."""
    ko_records = [record for record in records if record["lang"] == "ko"]
    en_records = [record for record in records if record["lang"] == "en"]
    response_lengths = [len(str(record["llm_response"])) for record in records]
    avg_tts_first_chunk_ms = _avg_defined(
        [record.get("tts_first_chunk_ms") for record in records],
    )
    first_turn_ttft: list[float] = []
    after_first_turn_ttft: list[float] = []
    for record in records:
        turn_index = _optional_int(record.get("turn_index_per_lang"))
        llm_ttft_ms = _optional_float(record.get("timings_ms", {}).get("llm_ttft_ms"))
        if turn_index is None or llm_ttft_ms is None:
            continue
        if turn_index == 0:
            first_turn_ttft.append(llm_ttft_ms)
        elif turn_index > 0:
            after_first_turn_ttft.append(llm_ttft_ms)
    cache_hit_rates: list[float] = []
    for record in records:
        hit_tokens = _optional_int(record.get("llm_cache_hit_tokens"))
        miss_tokens = _optional_int(record.get("llm_cache_miss_tokens"))
        if hit_tokens is None or miss_tokens is None:
            continue
        total_tokens = hit_tokens + miss_tokens
        if total_tokens <= 0:
            continue
        cache_hit_rates.append(hit_tokens / float(total_tokens))
    cer_values = [
        _error_rate(
            list(_normalize_metric_text(str(record["gt_text"]))),
            list(_normalize_metric_text(str(record["stt_pred"]))),
        )
        for record in records
    ]
    wer_values = [
        _error_rate(
            _normalize_metric_text(str(record["gt_text"])).split(),
            _normalize_metric_text(str(record["stt_pred"])).split(),
        )
        for record in records
    ]
    latency_table = _build_latency_table(records)
    stability_counters = parse_run_log_stability_counters(log_path)
    counters = load_counters or LoadCounters()
    tts_synth_error_count = sum(1 for record in records if record.get("tts_synth_error"))
    stt_provider_configured_value = stt_provider_configured or stt_provider_resolved

    def _language_summary(language_records: list[dict[str, Any]]) -> dict[str, Any]:
        local_cer = [
            _error_rate(
                list(_normalize_metric_text(str(record["gt_text"]))),
                list(_normalize_metric_text(str(record["stt_pred"]))),
            )
            for record in language_records
        ]
        local_wer = [
            _error_rate(
                _normalize_metric_text(str(record["gt_text"])).split(),
                _normalize_metric_text(str(record["stt_pred"])).split(),
            )
            for record in language_records
        ]
        return {
            "count": len(language_records),
            "avg_total_ms": round(
                _avg([float(record["timings_ms"]["total_ms"]) for record in language_records]),
                3,
            ),
            "avg_llm_ms": round(
                _avg([float(record["timings_ms"]["llm_ms"]) for record in language_records]),
                3,
            ),
            "cer": round(_avg(local_cer), 6),
            "wer": round(_avg(local_wer), 6),
        }

    return {
        "output_dir": str(output_dir),
        "rounds_jsonl": str(rounds_path),
        "run_log": str(log_path),
        "runner": "e2e_qwen3_asr_mix",
        "total_rounds": len(records),
        "ko_count": len(ko_records),
        "en_count": len(en_records),
        "repeat_passes": int(getattr(args, "repeat_passes", 1)),
        "input_pad_ms": int(getattr(args, "input_pad_ms", 0)),
        "passes_failed": list(passes_failed or []),
        "mungi_llm_resident": os.getenv("MUNGI_LLM_RESIDENT"),
        "mungi_stt_resident": os.getenv("MUNGI_STT_RESIDENT"),
        "mungi_tts_resident": os.getenv("MUNGI_TTS_RESIDENT"),
        "llm_n_gpu_layers_resolved": llm_n_gpu_layers_resolved,
        "stt_provider_actual": _summarize_stt_provider_actual(records),
        "stt_provider_configured": stt_provider_configured_value,
        "stt_provider_requested": (
            stt_provider_requested
            if stt_provider_requested is not None
            else os.getenv("MUNGI_STT_PROVIDER")
        ),
        "stt_provider_resolved": stt_provider_configured_value,
        "sherpa_onnx_version": _sherpa_onnx_version(),
        "stt_load_count": counters.stt_load_count,
        "tts_load_count": counters.tts_load_count,
        "tts_load_error_count": counters.tts_load_error_count,
        "tts_synth_error_count": tts_synth_error_count,
        "model_sha256": _model_sha256_summary(),
        "skip_tts": bool(args.skip_tts),
        "tts_file_count": sum(1 for record in records if record["response_wav"] is not None),
        "avg_vad_ms": round(_avg([record["timings_ms"]["vad_ms"] for record in records]), 3),
        "avg_stt_ms": round(_avg([record["timings_ms"]["stt_ms"] for record in records]), 3),
        "avg_stt_total_ms": round(
            _avg([record["timings_ms"]["stt_total_ms"] for record in records]),
            3,
        ),
        "avg_llm_load_ms": round(
            _avg([record["timings_ms"]["llm_load_ms"] for record in records]),
            3,
        ),
        "avg_llm_ttft_ms": round(
            _avg([record["timings_ms"]["llm_ttft_ms"] for record in records]),
            3,
        ),
        "avg_llm_ms": round(_avg([record["timings_ms"]["llm_ms"] for record in records]), 3),
        "avg_tts_load_ms": round(
            _avg([record["timings_ms"]["tts_load_ms"] for record in records]),
            3,
        ),
        "avg_tts_ms": round(_avg([record["timings_ms"]["tts_ms"] for record in records]), 3),
        "avg_playback_ms": round(
            _avg([record["timings_ms"]["playback_ms"] for record in records]),
            3,
        ),
        "avg_first_sound_ms": round(
            _avg([record["timings_ms"]["first_sound_ms"] for record in records]),
            3,
        ),
        "avg_tts_first_chunk_ms": (
            round(avg_tts_first_chunk_ms, 3) if avg_tts_first_chunk_ms is not None else None
        ),
        "avg_llm_ttft_ms_first_turn": (
            round(_avg(first_turn_ttft), 3) if first_turn_ttft else None
        ),
        "avg_llm_ttft_ms_after_first": (
            round(_avg(after_first_turn_ttft), 3) if after_first_turn_ttft else None
        ),
        "avg_llm_cache_hit_rate": (round(_avg(cache_hit_rates), 6) if cache_hit_rates else None),
        "avg_total_ms": round(_avg([record["timings_ms"]["total_ms"] for record in records]), 3),
        "asr_cer": round(_avg(cer_values), 6),
        "asr_wer": round(_avg(wer_values), 6),
        "llm_response_chars": {
            "min": min(response_lengths) if response_lengths else 0,
            "max": max(response_lengths) if response_lengths else 0,
            "avg": round(_avg([float(length) for length in response_lengths]), 3),
        },
        "languages": {
            "ko": _language_summary(ko_records),
            "en": _language_summary(en_records),
        },
        "latency_units": "ms",
        "latency_table_markdown": latency_table,
        "critical_memory_events": stability_counters["critical_memory_events"],
        "stt_force_unload_count": stability_counters["stt_force_unload_count"],
        "llm_prompt_cache_flush_count": stability_counters["llm_prompt_cache_flush_count"],
        "system_state_snapshot_count": stability_counters["system_state_snapshot_count"],
    }


def _build_pipelines(
    *,
    manager: Any,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    """Create one cached conversation pipeline per language."""
    _manager_config_cls, _model_manager_cls, pipeline_config_cls, pipeline_cls = (
        _get_runtime_classes()
    )
    history_config_overrides: dict[str, int] = {}
    if args.max_history_turns is not None:
        history_config_overrides["max_history_turns"] = args.max_history_turns
    if args.max_history_tokens is not None:
        history_config_overrides["max_history_tokens"] = args.max_history_tokens

    pipelines: dict[str, Any] = {}
    for language in ("ko", "en"):
        pipeline_config = pipeline_config_cls(
            stt_language=language,
            play_tts_audio=False,
            tts_output_device=args.output_device,
            enable_stt_preload=False,
            **history_config_overrides,
        )
        if language == "ko":
            logger.info("LLM max_tokens for this run: %d", pipeline_config.llm_max_tokens)
        pipeline = pipeline_cls(
            manager,
            pipeline_config,
        )
        if hasattr(pipeline, "_conversation_dir"):
            pipeline._conversation_dir = output_dir / "_pipeline_sessions" / language
        pipelines[language] = pipeline
    return pipelines


def main(argv: Sequence[str] | None = None) -> int:
    """Run the bilingual interleaved prerecorded-audio E2E workflow."""
    global _SHUTDOWN_MANAGER

    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.max_rounds < 0:
        logger.error("--max-rounds must be >= 0.")
        return 1
    _apply_llm_max_tokens_override(args=args, parser=parser)
    llm_n_gpu_layers_resolved = _resolve_llm_n_gpu_layers(args)
    tts_streaming = _tts_streaming_enabled(args)
    stt_provider_requested = os.getenv("MUNGI_STT_PROVIDER")

    output_dirs = _make_output_dirs(args.output_root.resolve())
    output_dir = output_dirs["run_dir"]
    rounds_path = output_dir / "rounds.jsonl"
    summary_path = output_dir / "summary.json"
    tegrastats_log_path = output_dir / "tegrastats.log"
    thermal_summary_path = output_dir / "thermal_summary.json"
    thermal_curve_path = output_dir / "thermal_curve.json"
    log_path = _configure_logging(output_dir)
    _register_signal_handlers()

    logger.info("Starting bilingual Qwen3-ASR audio E2E run")
    logger.info("KO directory: %s", args.ko_dir)
    logger.info("EN directory: %s", args.en_dir)
    logger.info("Output directory: %s", output_dir)
    logger.info(
        "skip_tts=%s skip_preflight=%s tts_streaming=%s",
        args.skip_tts,
        args.skip_preflight,
        tts_streaming,
    )

    manager: Any = None
    tegrastats_monitor: Any | None = None
    thermal_logging_enabled = False
    thermal_started_at = 0.0
    load_counters = LoadCounters()
    stt_provider_resolved: str | None = None
    stt_provider_configured = ""
    try:
        rounds = discover_round_pairs(args.ko_dir, args.en_dir, args.max_rounds)
        logger.info("Discovered %d interleaved rounds", len(rounds))

        _run_preflight(skip=bool(args.skip_preflight))

        thermal_started_at = time.monotonic()
        tegrastats_monitor, thermal_logging_enabled = _start_tegrastats_monitor(
            tegrastats_log_path,
        )

        manager_config_cls, model_manager_cls, _pipeline_config_cls, _pipeline_cls = (
            _get_runtime_classes()
        )
        manager_config = manager_config_cls(
            model_dir=args.model_dir,
            stt_model_size=args.stt_model,
            llm_resident=True,
            llm_n_gpu_layers=(
                llm_n_gpu_layers_resolved if llm_n_gpu_layers_resolved is not None else -1
            ),
        )
        stt_provider_configured = str(getattr(manager_config, "stt_device", ""))
        stt_provider_resolved = stt_provider_configured
        manager = model_manager_cls(manager_config)
        _wrap_load_counters(manager, load_counters)
        _SHUTDOWN_MANAGER = manager
        manager.initialize()
        pipelines = _build_pipelines(
            manager=manager,
            args=args,
            output_dir=output_dir,
        )

        records: list[dict[str, Any]] = []
        turn_index_by_lang: dict[str, int] = {}
        global_turn_id = 0
        passes_completed: list[str] = []
        passes_failed: list[str] = []
        fatal_error = False
        for pass_number in range(1, int(args.repeat_passes) + 1):
            pass_id = f"pass{pass_number}"
            logger.info("Starting %s/%d", pass_id, args.repeat_passes)
            current_round_input: RoundInput | None = None
            current_input_trace_path: Path | None = None
            current_turn_index_per_lang: int | None = None
            current_system_ram_mb: float | None = None
            current_process_rss_mb: float | None = None
            current_audio_duration_ms: float | None = None
            current_audio_padded_ms: float | None = None
            try:
                for round_input in rounds:
                    current_round_input = round_input
                    current_input_trace_path = None
                    current_turn_index_per_lang = None
                    current_system_ram_mb = None
                    current_process_rss_mb = None
                    current_audio_duration_ms = None
                    current_audio_padded_ms = None
                    pipeline = pipelines[round_input.lang]
                    if not args.conversation_per_lang:
                        pipeline.clear_history()
                    language_key = _normalize_language_key(round_input.lang)
                    turn_index_per_lang = (
                        turn_index_by_lang.get(language_key, 0) if language_key is not None else 0
                    )
                    current_turn_index_per_lang = turn_index_per_lang
                    if language_key is not None:
                        turn_index_by_lang[language_key] = turn_index_per_lang + 1

                    round_prefix = (
                        f"r{round_input.round_id:02d}_{round_input.lang}"
                        if args.repeat_passes == 1
                        else f"{pass_id}_r{round_input.round_id:02d}_{round_input.lang}"
                    )
                    input_trace_path = output_dirs["input_wavs"] / (
                        f"{round_prefix}_{round_input.wav_path.name}"
                    )
                    current_input_trace_path = input_trace_path
                    # input_trace_wav copies the ORIGINAL source WAV (pre-pad). Padded
                    # audio is pipeline-internal; preserving the raw trace allows
                    # analysts to re-run with different --input-pad-ms values. See PR 2.
                    _link_or_copy_input(round_input.wav_path, input_trace_path)

                    float_audio, _pcm_audio, audio_duration_ms, audio_padded_ms = _load_audio_input(
                        round_input.wav_path,
                        input_pad_ms=args.input_pad_ms,
                    )
                    duration_s = audio_duration_ms / 1000.0
                    system_ram_mb = _latest_system_ram_mb(tegrastats_monitor)
                    process_rss_mb = _process_rss_mb()
                    current_system_ram_mb = system_ram_mb
                    current_process_rss_mb = process_rss_mb
                    current_audio_duration_ms = audio_duration_ms
                    current_audio_padded_ms = audio_padded_ms
                    logger.info(
                        "%s round %d/%d [%s] input=%s duration=%.3fs padded=%.3fs",
                        pass_id,
                        round_input.round_id,
                        len(rounds),
                        round_input.lang,
                        round_input.wav_path.name,
                        duration_s,
                        audio_padded_ms / 1000.0,
                    )

                    with (
                        _maybe_skip_tts(pipeline, manager, bool(args.skip_tts)),
                        _maybe_externalize_tts(
                            pipeline,
                            tts_streaming,
                        ),
                    ):
                        result = pipeline.run_turn(float_audio, sample_rate=DEFAULT_SAMPLE_RATE)
                    if tts_streaming and getattr(result, "error", None) is None:
                        _apply_sentence_streaming_tts(
                            result=result,
                            manager=manager,
                            output_device=args.output_device,
                        )

                    mismatch_message = _stt_provider_mismatch_message(
                        getattr(result.metrics, "stt_provider_actual", None),
                        args.expect_stt_provider,
                    )
                    if mismatch_message is not None:
                        raise ExpectedSTTProviderMismatch(mismatch_message)

                    response_wav_path: Path | None = None
                    audio_out = getattr(result, "audio_samples", None)
                    sample_rate = int(getattr(result, "sample_rate", 0))
                    if (
                        not args.skip_tts
                        and audio_out is not None
                        and sample_rate > 0
                        and np.asarray(audio_out).size > 0
                    ):
                        response_wav_path = output_dirs["response_wavs"] / f"{round_prefix}.wav"
                        _write_response_wav(response_wav_path, audio_out, sample_rate)

                    record = _make_round_record(
                        output_dir=output_dir,
                        round_input=round_input,
                        duration_s=duration_s,
                        audio_duration_ms=audio_duration_ms,
                        audio_padded_ms=audio_padded_ms,
                        result=result,
                        input_trace_path=input_trace_path,
                        response_wav_path=response_wav_path,
                        turn_index_per_lang=turn_index_per_lang,
                        pass_id=pass_id,
                        global_turn_id=global_turn_id,
                        system_ram_mb=system_ram_mb,
                        process_rss_mb=process_rss_mb,
                        stt_provider_configured=stt_provider_configured,
                        stt_provider_requested=stt_provider_requested,
                    )
                    _append_jsonl_record(rounds_path, record)
                    records.append(record)
                    logger.info(
                        "%s round %d complete: success=%s stt=%r total_ms=%.1f",
                        pass_id,
                        round_input.round_id,
                        record["success"],
                        record["stt_pred"],
                        record["timings_ms"]["total_ms"],
                    )
                    global_turn_id += 1
            except ExpectedSTTProviderMismatch as exc:
                failing_round_id = (
                    current_round_input.round_id if current_round_input is not None else None
                )
                logger.error(
                    "Fail-fast STT provider check failed: pass_id=%s round_id=%s error=%s",
                    pass_id,
                    failing_round_id,
                    _sanitize_error_message(exc),
                )
                record = _make_pass_failure_record(
                    output_dir=output_dir,
                    pass_id=pass_id,
                    global_turn_id=global_turn_id,
                    exc=exc,
                    round_input=current_round_input,
                    input_trace_path=current_input_trace_path,
                    turn_index_per_lang=current_turn_index_per_lang,
                    system_ram_mb=current_system_ram_mb,
                    process_rss_mb=current_process_rss_mb,
                    audio_duration_ms=current_audio_duration_ms,
                    audio_padded_ms=current_audio_padded_ms,
                    stt_provider_configured=stt_provider_configured,
                    stt_provider_requested=stt_provider_requested,
                )
                _append_jsonl_record(rounds_path, record)
                records.append(record)
                passes_failed.append(pass_id)
                global_turn_id += 1
                fatal_error = True
            except Exception as exc:
                failing_round_id = (
                    current_round_input.round_id if current_round_input is not None else None
                )
                logger.error(
                    "Pass failed: pass_id=%s round_id=%s error=%s",
                    pass_id,
                    failing_round_id,
                    _sanitize_error_message(exc),
                    exc_info=True,
                )
                record = _make_pass_failure_record(
                    output_dir=output_dir,
                    pass_id=pass_id,
                    global_turn_id=global_turn_id,
                    exc=exc,
                    round_input=current_round_input,
                    input_trace_path=current_input_trace_path,
                    turn_index_per_lang=current_turn_index_per_lang,
                    system_ram_mb=current_system_ram_mb,
                    process_rss_mb=current_process_rss_mb,
                    audio_duration_ms=current_audio_duration_ms,
                    audio_padded_ms=current_audio_padded_ms,
                    stt_provider_configured=stt_provider_configured,
                    stt_provider_requested=stt_provider_requested,
                )
                _append_jsonl_record(rounds_path, record)
                records.append(record)
                passes_failed.append(pass_id)
                global_turn_id += 1
                continue
            if fatal_error:
                break
            passes_completed.append(pass_id)

        summary = _build_summary(
            args=args,
            output_dir=output_dir,
            rounds_path=rounds_path,
            log_path=log_path,
            records=records,
            llm_n_gpu_layers_resolved=getattr(
                manager_config,
                "llm_n_gpu_layers",
                llm_n_gpu_layers_resolved,
            ),
            stt_provider_resolved=stt_provider_resolved,
            stt_provider_configured=stt_provider_configured,
            stt_provider_requested=stt_provider_requested,
            load_counters=load_counters,
            passes_failed=passes_failed,
        )
        _write_json_atomic(summary_path, summary)
        logger.info("Latency table (ms):\n%s", summary["latency_table_markdown"])
        logger.info("Summary saved: %s", summary_path)
        if fatal_error:
            logger.error("Run stopped by fail-fast provider validation.")
            return 1
        if not passes_completed and passes_failed:
            logger.error("All repeat passes failed: %s", ", ".join(passes_failed))
            return 1
        return 0
    except Exception as exc:
        logger.error("E2E run failed: %s", exc, exc_info=True)
        return 1
    finally:
        if thermal_logging_enabled and tegrastats_monitor is not None:
            try:
                _write_thermal_artifacts(
                    monitor=tegrastats_monitor,
                    summary_path=thermal_summary_path,
                    curve_path=thermal_curve_path,
                    started_at=thermal_started_at,
                )
            except Exception:
                logger.warning("Failed to write thermal artifacts", exc_info=True)
        if manager is not None:
            try:
                manager.unload_all()
            except Exception:
                logger.warning("Failed to unload models during shutdown", exc_info=True)
        _SHUTDOWN_MANAGER = None


if __name__ == "__main__":
    raise SystemExit(main())
