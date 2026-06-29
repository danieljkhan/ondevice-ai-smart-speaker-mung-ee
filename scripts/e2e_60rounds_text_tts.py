"""60-round text-input / spoken-TTS E2E regression runner.

Uses the same child persona prompt as production, but skips STT by
feeding scripted text turns directly into the conversation pipeline.
Each assistant response is synthesized, optionally played on the local
speaker path, and saved to WAV so the rendered TTS can be reviewed later.
"""

from __future__ import annotations

import argparse
import atexit
import json
import logging
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.model_manager import ManagerConfig, ModelManager  # noqa: E402
from core.pipeline import ConversationPipeline, PipelineConfig  # noqa: E402
from scripts.bench_model import parse_tegrastats_line  # noqa: E402
from scripts.e2e_60rounds import (  # noqa: E402
    DEFAULT_ROUNDS,
    TOPIC_POOL,
    TopicData,
    build_round_messages,
    choose_round_topic,
    load_topic_pool,
    turn_count_for_round,
)
from scripts.test_tts import write_wav  # noqa: E402
from scripts.utils import get_peak_memory_kb  # noqa: E402

logger = logging.getLogger("mungi.scripts.e2e_60rounds_text_tts")

DEFAULT_OUTPUT_ROOT = Path(
    os.getenv("MUNGI_E2E_TTS_OUTPUT_DIR", "/var/lib/mungi/e2e_results"),
)
DEFAULT_OUTPUT_DEVICE = os.getenv("MUNGI_AUDIO_OUTPUT_DEVICE", "").strip() or None
DEFAULT_SILENCE_GAP_S = 0.25
_SHUTDOWN_MANAGER: ModelManager | None = None


def _graceful_shutdown(signum: int, frame: Any) -> None:
    """Clean up model resources on termination signals before exiting."""
    del frame
    logger.warning("Signal %d received, initiating graceful shutdown...", signum)
    manager = _SHUTDOWN_MANAGER
    if manager is not None:
        try:
            manager.unload_all(force=True)
        except Exception:
            pass
    sys.exit(0)


def _register_signal_handlers() -> None:
    """Register process termination handlers before model loading begins."""
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, _graceful_shutdown)
    signal.signal(signal.SIGTERM, _graceful_shutdown)


@dataclass(frozen=True)
class TextTTSTurnRecord:
    """Observable result for one text-input turn."""

    round_num: int
    topic: str
    exchange: int
    user_text: str
    assistant_text: str
    tts_wav: str | None
    llm_tokens: int
    llm_ttft_s: float
    llm_time_s: float
    llm_model_fallback_used: bool
    llm_model_path_actual: str | None
    llm_model_fallback_reason: str | None
    tts_time_s: float
    total_time_s: float
    peak_memory_kb: int
    success: bool
    error: str | None = None


class TegrastatsMonitor:
    """Capture tegrastats output while the scripted run is executing."""

    def __init__(self, log_path: Path, interval_ms: int = 1000) -> None:
        self._interval_ms = interval_ms
        self._log_path = log_path
        self._proc: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None
        self._snapshots: list[dict[str, Any]] = []
        self._start_monotonic = 0.0
        self._log_handle: Any = None

    def start(self) -> bool:
        """Start tegrastats if available on the current machine."""
        if shutil.which("tegrastats") is None:
            logger.info("tegrastats not available; thermal logging disabled")
            return False

        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self._log_path.open("w", encoding="utf-8")
        self._start_monotonic = time.monotonic()
        self._proc = subprocess.Popen(
            ["tegrastats", "--interval", str(self._interval_ms)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        """Stop tegrastats and finalize the raw log file."""
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=5)
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self._log_handle is not None:
            self._log_handle.close()

    @property
    def snapshots(self) -> list[dict[str, Any]]:
        """Return parsed tegrastats snapshots collected so far."""
        return list(self._snapshots)

    def _reader_loop(self) -> None:
        """Read tegrastats lines until the subprocess exits."""
        if self._proc is None or self._proc.stdout is None or self._log_handle is None:
            return

        for line in self._proc.stdout:
            stripped = line.strip()
            if not stripped:
                continue
            self._log_handle.write(stripped + "\n")
            self._log_handle.flush()
            parsed = parse_tegrastats_line(stripped)
            if parsed:
                parsed["elapsed_s"] = round(time.monotonic() - self._start_monotonic, 3)
                self._snapshots.append(parsed)


def _preflight_check_tmux() -> bool:
    """Return True if running inside a tmux session or not on Linux."""
    if sys.platform != "linux":
        return True
    return os.environ.get("TMUX", "") != ""


def _preflight_drop_page_cache() -> None:
    """Best-effort page cache drop for CUDA memory reclaim on Jetson."""
    if sys.platform != "linux":
        return
    try:
        subprocess.run(
            ["sudo", "-n", "tee", "/proc/sys/vm/drop_caches"],
            input=b"1",
            check=True,
            capture_output=True,
        )
        logger.info("Page cache dropped for clean test environment")
    except (subprocess.CalledProcessError, OSError):
        logger.warning("Page cache drop skipped (no passwordless sudo)")


def _preflight_check_memory(min_free_mb: int = 3000) -> bool:
    """Return True if MemFree is above the minimum threshold."""
    if sys.platform != "linux":
        return True
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemFree:"):
                    free_mb = int(line.split()[1]) // 1024
                    logger.info("MemFree: %d MB (minimum: %d MB)", free_mb, min_free_mb)
                    return free_mb >= min_free_mb
    except (OSError, ValueError, IndexError):
        pass
    return True


def _preflight_kill_zombie_python() -> int:
    """Kill orphaned Python processes consuming excessive memory.

    Only kills processes that started more than 60 seconds ago to avoid
    killing the current process tree (self, parent, tee, tmux bash).
    """
    if sys.platform != "linux":
        return 0
    killed = 0
    try:
        # Find e2e python processes older than 60 seconds
        result = subprocess.run(
            ["find", "/proc", "-maxdepth", "1", "-mindepth", "1", "-type", "d", "-mmin", "+1"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        old_pids = {
            p.split("/")[-1]
            for p in result.stdout.strip().split("\n")
            if p.strip() and p.split("/")[-1].isdigit()
        }

        pgrep_result = subprocess.run(
            ["pgrep", "-f", "python.*e2e_60rounds"],
            capture_output=True,
            text=True,
            check=False,
        )
        for pid in pgrep_result.stdout.strip().split("\n"):
            pid = pid.strip()
            if pid and pid in old_pids:
                try:
                    os.kill(int(pid), 9)
                    killed += 1
                    logger.warning("Killed orphaned E2E process: PID %s", pid)
                except (ProcessLookupError, PermissionError):
                    pass
    except (OSError, subprocess.SubprocessError):
        pass
    return killed


def run_preflight(skip: bool = False) -> None:
    """Run all pre-flight checks before E2E test execution.

    Checks (Linux/Jetson only):
    1. tmux session — prevents zombie processes on SSH disconnect
    2. Kill orphaned E2E processes — reclaim leaked memory
    3. Drop page cache — maximize MemFree for CUDA allocation
    4. Memory threshold — ensure sufficient free memory

    Args:
        skip: Bypass all checks (for local/CI environments).
    """
    if skip or sys.platform != "linux":
        return

    logger.info("=== Pre-flight checks ===")

    if not _preflight_check_tmux():
        logger.error(
            "NOT running inside tmux. SSH background E2E tests MUST use tmux "
            "to prevent zombie processes. Run: tmux new-session -d -s e2e '...'"
        )
        raise SystemExit(1)
    logger.info("tmux session: OK")

    # Zombie kill disabled — unreliable PID matching causes self-kill.
    # tmux + page cache drop + memory check is sufficient.

    _preflight_drop_page_cache()

    if not _preflight_check_memory():
        logger.error(
            "Insufficient MemFree (<3000 MB). Kill zombie processes or "
            "reboot Jetson before running E2E tests."
        )
        raise SystemExit(1)
    logger.info("Memory check: OK")
    logger.info("=== Pre-flight complete ===")


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for the text-input 60-round TTS runner."""
    parser = argparse.ArgumentParser(
        description=(
            "Run 60 scripted text rounds with 3-8 turns per round, "
            "synthesize each response with TTS, optionally play audio locally, "
            "and save rendered WAV files."
        ),
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=DEFAULT_ROUNDS,
        help="Number of rounds to run.",
    )
    parser.add_argument("--start", type=int, default=1, help="Starting round number.")
    parser.add_argument(
        "--topic-pool",
        type=Path,
        default=None,
        help="Path to topic pool JSON file. Default: V1 built-in pool",
    )
    parser.add_argument(
        "--llm-model-path",
        type=str,
        default=None,
        help="Explicit GGUF model file path. Overrides auto-discovery.",
    )
    parser.add_argument(
        "--presence-penalty",
        type=float,
        default=1.2,
        help="Presence penalty for LLM generation (Qwen3.5 recommends 1.5).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Directory for JSONL logs and rendered WAV files (default: {DEFAULT_OUTPUT_ROOT}).",
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="Play synthesized audio through the local sounddevice output path.",
    )
    parser.add_argument(
        "--output-device",
        type=str,
        default=DEFAULT_OUTPUT_DEVICE,
        help="Optional sounddevice output device override.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="Optional model root override. Defaults to runtime auto-detect.",
    )
    parser.add_argument(
        "--voice-style",
        type=str,
        default="F1",
        help="Supertonic voice style passed to ModelManager (default: F1).",
    )
    parser.add_argument(
        "--llm-max-tokens",
        type=int,
        default=128,
        help="Maximum tokens per assistant turn.",
    )
    parser.add_argument(
        "--max-history-turns",
        type=int,
        default=2,
        help="Conversation history turn-pairs to keep in prompt context.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible topic shuffling.",
    )
    parser.add_argument(
        "--session-wav",
        action="store_true",
        help=(
            "Build a combined session WAV from saved per-turn WAV files after the run. "
            "Disabled by default to avoid retaining audio in RAM."
        ),
    )
    parser.add_argument(
        "--llm-resident",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Keep LLM loaded between turns (resident mode; default follows ManagerConfig). "
            "Use --no-llm-resident or MUNGI_LLM_RESIDENT=0 to disable. "
            "Eliminates load/unload overhead and GPU memory fragmentation. "
            "Requires sufficient memory for LLM + TTS coexistence."
        ),
    )
    parser.add_argument(
        "--silence-gap-s",
        type=float,
        default=DEFAULT_SILENCE_GAP_S,
        help="Silence gap inserted between turns in the optional combined WAV.",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip tmux/memory pre-flight checks (for local/CI environments).",
    )
    return parser


def _slugify(text: str) -> str:
    """Create a filesystem-safe slug for per-turn WAV filenames."""
    slug = re.sub(r"[^0-9A-Za-z가-힣]+", "-", text.strip())
    slug = slug.strip("-")
    return slug[:40] or "topic"


def _flatten_audio(audio_samples: Any) -> np.ndarray:
    """Normalize audio samples to a mono float32 vector."""
    audio = np.asarray(audio_samples, dtype=np.float32)
    if audio.size == 0:
        return np.zeros(0, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.reshape(-1)
    return audio.astype(np.float32, copy=False)


def _resample_audio(
    audio: np.ndarray,
    source_sample_rate: int,
    target_sample_rate: int,
) -> np.ndarray:
    """Resample mono float audio with linear interpolation."""
    if source_sample_rate <= 0 or target_sample_rate <= 0:
        msg = "Sample rates must be positive integers."
        raise ValueError(msg)
    if source_sample_rate == target_sample_rate or audio.size <= 1:
        return audio.astype(np.float32, copy=False)

    duration_s = audio.size / float(source_sample_rate)
    target_size = max(int(round(duration_s * target_sample_rate)), 1)
    source_positions = np.linspace(0.0, audio.size - 1, num=audio.size, dtype=np.float32)
    target_positions = np.linspace(
        0.0,
        audio.size - 1,
        num=target_size,
        dtype=np.float32,
    )
    result: np.ndarray = np.interp(target_positions, source_positions, audio).astype(np.float32)
    return result


def _append_session_audio(
    session_audio: list[np.ndarray],
    audio_samples: Any,
    sample_rate: int,
    *,
    silence_gap_s: float,
    session_sample_rate: int | None,
) -> int:
    """Append a turn's audio to an in-memory session buffer when explicitly requested."""
    audio = _flatten_audio(audio_samples)
    if audio.size == 0 or sample_rate <= 0:
        return session_sample_rate or sample_rate

    target_sr = session_sample_rate or sample_rate
    if sample_rate != target_sr:
        audio = _resample_audio(audio, sample_rate, target_sr)

    if session_audio and silence_gap_s > 0:
        gap_samples = max(int(round(target_sr * silence_gap_s)), 1)
        session_audio.append(np.zeros(gap_samples, dtype=np.float32))

    session_audio.append(audio)
    return target_sr


def _save_turn_wav(path: Path, audio_samples: Any, sample_rate: int) -> None:
    """Write a synthesized turn to a WAV file."""
    write_wav(path, _flatten_audio(audio_samples), sample_rate)


def _load_turn_wav(path: Path) -> tuple[np.ndarray, int]:
    """Load a mono 16-bit PCM WAV file written by :func:`write_wav`."""
    with wave.open(str(path), "rb") as wav_file:
        channel_count = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        frames = wav_file.readframes(frame_count)

    if channel_count != 1:
        msg = f"Expected mono WAV for session mix, got {channel_count} channels: {path}"
        raise ValueError(msg)
    if sample_width != 2:
        msg = f"Expected 16-bit PCM WAV for session mix, got {sample_width * 8}-bit: {path}"
        raise ValueError(msg)

    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
    return audio, sample_rate


def _build_session_audio_from_files(
    turn_wavs: list[Path],
    silence_gap_s: float,
) -> tuple[np.ndarray, int] | None:
    """Build a combined session buffer from saved per-turn WAV files."""
    session_audio: list[np.ndarray] = []
    session_sample_rate: int | None = None

    for wav_path in turn_wavs:
        audio, sample_rate = _load_turn_wav(wav_path)
        if audio.size == 0 or sample_rate <= 0:
            continue

        target_sr = session_sample_rate or sample_rate
        if sample_rate != target_sr:
            audio = _resample_audio(audio, sample_rate, target_sr)

        if session_audio and silence_gap_s > 0:
            gap_samples = max(int(round(target_sr * silence_gap_s)), 1)
            session_audio.append(np.zeros(gap_samples, dtype=np.float32))

        session_audio.append(audio)
        session_sample_rate = target_sr

    if not session_audio or session_sample_rate is None:
        return None

    return np.concatenate(session_audio).astype(np.float32), session_sample_rate


def _build_thermal_summary(snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize CPU/GPU temperature changes from tegrastats snapshots."""
    summary: dict[str, Any] = {"snapshots_count": len(snapshots)}
    for key in ("cpu_temp_c", "gpu_temp_c", "ram_used_mb", "gr3d_freq_pct"):
        values = [float(s[key]) for s in snapshots if key in s]
        if not values:
            continue
        summary[key] = {
            "start": round(values[0], 3),
            "end": round(values[-1], 3),
            "min": round(min(values), 3),
            "max": round(max(values), 3),
            "avg": round(sum(values) / len(values), 3),
            "delta": round(values[-1] - values[0], 3),
        }
    return summary


def run_round(
    pipeline: ConversationPipeline,
    round_num: int,
    topic_data: TopicData,
    wav_dir: Path,
    *,
    silence_gap_s: float = DEFAULT_SILENCE_GAP_S,
    session_audio: list[np.ndarray] | None = None,
    session_sample_rate: int | None = None,
    manager: ModelManager | None = None,
    session_turn_wavs: list[Path] | None = None,
) -> tuple[dict[str, Any], int | None]:
    """Run one scripted round and return round metadata."""
    topic_name = str(topic_data["topic"])
    messages = build_round_messages(topic_data, round_num)
    topic_slug = _slugify(topic_name)
    turns: list[dict[str, Any]] = []
    total_tokens = 0
    total_time_s = 0.0

    for exchange_idx, user_text in enumerate(messages, start=1):
        result = pipeline.run_text_turn(str(user_text))
        wav_path: Path | None = None

        if result.audio_samples is not None and result.sample_rate > 0:
            wav_path = wav_dir / f"round_{round_num:03d}_{topic_slug}_turn_{exchange_idx:02d}.wav"
            _save_turn_wav(wav_path, result.audio_samples, result.sample_rate)
            if session_audio is not None:
                session_sample_rate = _append_session_audio(
                    session_audio,
                    result.audio_samples,
                    result.sample_rate,
                    silence_gap_s=silence_gap_s,
                    session_sample_rate=session_sample_rate,
                )
            if session_turn_wavs is not None:
                session_turn_wavs.append(wav_path)

        if manager is not None:
            llm_diag = manager.latest_llm_load_diagnostics()
            logger.info(
                (
                    "Round %d turn %d LLM load diagnostics: loaded_n_gpu_layers=%s, "
                    "selected_n_gpu_layers=%s, attempted_n_gpu_layers=%s, fallback_used=%s"
                ),
                round_num,
                exchange_idx,
                llm_diag.get("loaded_n_gpu_layers"),
                llm_diag.get("selected_n_gpu_layers"),
                llm_diag.get("attempted_n_gpu_layers"),
                llm_diag.get("fallback_used"),
            )
        logger.info(
            "Round %d turn %d peak memory: %d KB",
            round_num,
            exchange_idx,
            get_peak_memory_kb(),
        )

        record = TextTTSTurnRecord(
            round_num=round_num,
            topic=topic_name,
            exchange=exchange_idx,
            user_text=result.user_text,
            assistant_text=result.response_text,
            tts_wav=str(wav_path) if wav_path is not None else None,
            llm_tokens=result.metrics.llm_tokens,
            llm_ttft_s=round(result.metrics.llm_ttft_s, 3),
            llm_time_s=round(result.metrics.llm_time_s, 3),
            llm_model_fallback_used=result.metrics.llm_model_fallback_used,
            llm_model_path_actual=result.metrics.llm_model_path_actual,
            llm_model_fallback_reason=result.metrics.llm_model_fallback_reason,
            tts_time_s=round(result.metrics.tts_time_s, 3),
            total_time_s=round(result.metrics.total_time_s, 3),
            peak_memory_kb=get_peak_memory_kb(),
            success=result.success,
            error=result.error,
        )
        turns.append(asdict(record))
        total_tokens += result.metrics.llm_tokens
        total_time_s += result.metrics.total_time_s

    return (
        {
            "round": round_num,
            "planned_turns": len(messages),
            "topics": [{"topic": topic_name, "turns": turns}],
            "total_tokens": total_tokens,
            "total_time_s": round(total_time_s, 3),
        },
        session_sample_rate,
    )


def main() -> int:
    """CLI entry point."""
    global _SHUTDOWN_MANAGER

    args = build_parser().parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _register_signal_handlers()

    run_preflight(skip=args.skip_preflight)

    output_dir = args.output_dir.resolve()
    wav_dir = output_dir / "tts_wavs"
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "rounds.jsonl"
    session_wav_path = output_dir / "tts_session.wav"
    summary_path = output_dir / "summary.json"
    tegrastats_log_path = output_dir / "tegrastats.log"
    thermal_summary_path = output_dir / "thermal_summary.json"

    manager_config_kwargs: dict[str, Any] = {
        "model_dir": args.model_dir,
        "tts_voice_style": args.voice_style,
        "llm_model_path": args.llm_model_path,
    }
    if args.llm_resident is not None:
        manager_config_kwargs["llm_resident"] = args.llm_resident
    manager_config = ManagerConfig(**manager_config_kwargs)
    manager = ModelManager(manager_config)
    _SHUTDOWN_MANAGER = manager
    atexit.register(manager.unload_all)
    manager.initialize()

    pipeline = ConversationPipeline(
        manager,
        PipelineConfig(
            llm_max_tokens=args.llm_max_tokens,
            llm_presence_penalty=args.presence_penalty,
            max_history_turns=args.max_history_turns,
            play_tts_audio=args.play,
            tts_output_device=args.output_device,
        ),
    )

    pool = load_topic_pool(args.topic_pool) if args.topic_pool else list(TOPIC_POOL)
    rng = random.Random(args.seed)
    session_turn_wavs: list[Path] | None = [] if args.session_wav else None
    tegrastats = TegrastatsMonitor(tegrastats_log_path)
    thermal_logging_enabled = tegrastats.start()

    logger.info(
        "Starting text-input 60-round TTS run: rounds=%d, start=%d",
        args.rounds,
        args.start,
    )
    logger.info("Output directory: %s", output_dir)
    logger.info("Local playback: %s", "enabled" if args.play else "disabled")
    logger.info("Combined session WAV: %s", "enabled" if args.session_wav else "disabled")
    if args.output_device:
        logger.info("Output device override: %s", args.output_device)

    try:
        with results_path.open("w", encoding="utf-8") as results_file:
            topic_cursor = 0
            for round_num in range(args.start, args.start + args.rounds):
                selected_topic, topic_cursor = choose_round_topic(
                    pool,
                    cursor=topic_cursor,
                    rng=rng,
                )
                planned_turns = turn_count_for_round(round_num)
                logger.info(
                    "Round %d/%d topic: %s (%d turns)",
                    round_num,
                    args.start + args.rounds - 1,
                    selected_topic["topic"],
                    planned_turns,
                )

                round_result, _ = run_round(
                    pipeline,
                    round_num,
                    selected_topic,
                    wav_dir,
                    manager=manager,
                    session_turn_wavs=session_turn_wavs,
                )
                results_file.write(json.dumps(round_result, ensure_ascii=False) + "\n")
                results_file.flush()
                logger.info(
                    "Round %d complete: %d tokens, %.3fs",
                    round_num,
                    round_result["total_tokens"],
                    round_result["total_time_s"],
                )
                pipeline.clear_history()
    finally:
        tegrastats.stop()

    session_wav_output: str | None = None
    if session_turn_wavs:
        session_audio = _build_session_audio_from_files(session_turn_wavs, args.silence_gap_s)
        if session_audio is not None:
            session_mix, session_sample_rate = session_audio
            write_wav(session_wav_path, session_mix, session_sample_rate)
            session_wav_output = str(session_wav_path)
            logger.info(
                "Combined TTS session WAV saved: %s (%.2fs)",
                session_wav_path,
                session_mix.size / float(session_sample_rate),
            )

    topic_pool_version = args.topic_pool.stem if args.topic_pool else "v1"
    summary = {
        "rounds": args.rounds,
        "start": args.start,
        "topic_pool_version": topic_pool_version,
        "output_dir": str(output_dir),
        "results_jsonl": str(results_path),
        "session_wav": session_wav_output,
        "peak_memory_kb": get_peak_memory_kb(),
        "playback_enabled": args.play,
        "output_device": args.output_device,
        "tegrastats_log": str(tegrastats_log_path) if thermal_logging_enabled else None,
        "thermal_summary_json": str(thermal_summary_path) if thermal_logging_enabled else None,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Summary saved: %s", summary_path)

    if thermal_logging_enabled:
        thermal_summary_path.write_text(
            json.dumps(
                _build_thermal_summary(tegrastats.snapshots),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info("Thermal summary saved: %s", thermal_summary_path)

    manager.unload_all()
    _SHUTDOWN_MANAGER = None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
