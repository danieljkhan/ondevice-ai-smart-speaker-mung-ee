"""Blocking E2E voice runner for Jetson pilot playback tests."""

from __future__ import annotations

import argparse
import enum
import json
import logging
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

try:
    import sounddevice as sd  # type: ignore[import-not-found, import-untyped]
except ImportError:
    sd = ModuleType("sounddevice")  # type: ignore[assignment]

    def _missing_sounddevice(*args: Any, **kwargs: Any) -> Any:
        msg = "sounddevice is required to run scripts.e2e_voice_runner"
        raise RuntimeError(msg)

    sd.query_devices = _missing_sounddevice  # type: ignore[attr-defined]
    sd.rec = _missing_sounddevice  # type: ignore[attr-defined]
    sd.wait = _missing_sounddevice  # type: ignore[attr-defined]
    sys.modules.setdefault("sounddevice", sd)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.model_manager import ManagerConfig, ModelManager
from core.pipeline import ConversationPipeline, PipelineConfig, TurnResult
from scripts.e2e_60rounds_text_tts import TegrastatsMonitor, run_preflight

logger = logging.getLogger("mungi.scripts.e2e_voice_runner")

DEFAULT_ROUNDS = 30
DEFAULT_ENERGY_THRESHOLD_DB = -40.0
DEFAULT_END_GAP_S = 2.0
DEFAULT_MAX_CAPTURE_S = 10.0
DEFAULT_MIN_CAPTURE_S = 1.5
DEFAULT_PROBE_WINDOW_S = 0.3
DEFAULT_INPUT_DEVICE = "USB PnP Audio Device"

_shutdown_requested = False


class RunnerState(enum.Enum):
    """Observable runner states for the live pilot capture loop."""

    STARTUP = "startup"
    ARMED = "armed"
    CAPTURING = "capturing"
    PROCESS = "process"
    SHUTDOWN = "shutdown"


def rms_db(audio: np.ndarray) -> float:
    """Compute RMS level in dB. Returns -inf for silence."""
    rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2))
    if rms < 1e-10:
        return float("-inf")
    return float(20.0 * np.log10(rms))


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the voice pilot runner."""
    parser = argparse.ArgumentParser(
        description="Capture live playback segments and run them through the Mungi pipeline.",
    )
    parser.add_argument("--lang", choices=("ko", "en"), required=True)
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--energy-threshold-db",
        type=float,
        default=DEFAULT_ENERGY_THRESHOLD_DB,
    )
    parser.add_argument("--end-gap-s", type=float, default=DEFAULT_END_GAP_S)
    parser.add_argument("--max-capture-s", type=float, default=DEFAULT_MAX_CAPTURE_S)
    parser.add_argument("--min-capture-s", type=float, default=DEFAULT_MIN_CAPTURE_S)
    parser.add_argument("--probe-window-s", type=float, default=DEFAULT_PROBE_WINDOW_S)
    parser.add_argument("--input-device", type=str, default=DEFAULT_INPUT_DEVICE)
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument(
        "--warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable LLM warmup on startup.",
    )
    return parser


def find_usb_input(device_name: str) -> tuple[int, int, int]:
    """Find USB input device. Returns (device_index, sample_rate, channels)."""
    for index, info in enumerate(sd.query_devices()):
        if int(info.get("max_input_channels", 0)) <= 0:
            continue
        if device_name.lower() not in str(info.get("name", "")).lower():
            continue
        sample_rate = int(float(info.get("default_samplerate", 48000)))
        channels = min(int(info.get("max_input_channels", 1)), 2)
        return index, sample_rate, channels
    msg = f"USB input device '{device_name}' not found"
    raise RuntimeError(msg)


def _graceful_shutdown(signum: int, frame: Any) -> None:
    """Set the shutdown flag for cooperative termination."""
    del frame
    global _shutdown_requested
    _shutdown_requested = True
    logger.info("Shutdown requested (signal %d)", signum)


def _register_signal_handlers() -> None:
    """Register termination handlers used by the long-running capture loop."""
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, _graceful_shutdown)
    signal.signal(signal.SIGTERM, _graceful_shutdown)


def _configure_logging() -> None:
    """Configure INFO-level logging for the runner."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _transition_state(current: RunnerState, next_state: RunnerState) -> RunnerState:
    """Log and return a runner state transition."""
    logger.info("State: %s -> %s", current.name, next_state.name)
    return next_state


def _to_mono_float32(audio: np.ndarray) -> np.ndarray:
    """Downmix captured audio to a mono float32 vector."""
    samples = np.asarray(audio, dtype=np.float32)
    if samples.size == 0:
        return np.zeros(0, dtype=np.float32)
    if samples.ndim == 1:
        return samples.astype(np.float32, copy=False)
    mono = np.mean(samples, axis=1, dtype=np.float32)
    return np.asarray(mono, dtype=np.float32)


def _resample_to_16k(audio: np.ndarray, original_sr: int) -> np.ndarray:
    """Resample mono float32 audio to 16kHz using FFT-domain band limiting."""
    if original_sr <= 0:
        msg = f"original_sr must be positive, got {original_sr}"
        raise ValueError(msg)

    samples = np.asarray(audio, dtype=np.float32)
    if samples.size == 0:
        return np.zeros(0, dtype=np.float32)
    if original_sr == 16000:
        return samples.astype(np.float32, copy=False)

    target_len = max(int(round(samples.shape[0] * 16000 / float(original_sr))), 1)
    source_freq = np.fft.rfft(samples)
    target_freq_len = target_len // 2 + 1

    if target_freq_len <= source_freq.shape[0]:
        resampled_freq = source_freq[:target_freq_len].copy()
    else:
        resampled_freq = np.zeros(target_freq_len, dtype=source_freq.dtype)
        resampled_freq[: source_freq.shape[0]] = source_freq

    resampled = np.fft.irfft(resampled_freq, n=target_len)
    scaled = resampled * (target_len / samples.shape[0])
    return np.asarray(scaled, dtype=np.float32)


def _trim_trailing_silence(audio: np.ndarray, recorded_frames: int) -> np.ndarray:
    """Trim trailing silent frames from a captured segment."""
    clipped = np.asarray(audio[:recorded_frames], dtype=np.float32)
    if clipped.size == 0:
        return clipped
    if clipped.ndim == 1:
        magnitude = np.abs(clipped)
    else:
        magnitude = np.max(np.abs(clipped), axis=1)
    nonzero = np.flatnonzero(magnitude > 1e-8)
    if nonzero.size == 0:
        return clipped[:0]
    return clipped[: int(nonzero[-1]) + 1]


def _capture_probe(
    *,
    sample_rate: int,
    channels: int,
    device_index: int,
    probe_window_s: float,
) -> np.ndarray:
    """Capture one blocking probe window for speech onset detection."""
    probe = sd.rec(
        int(probe_window_s * sample_rate),
        samplerate=sample_rate,
        channels=channels,
        dtype="float32",
        device=device_index,
    )
    sd.wait()
    return np.asarray(probe, dtype=np.float32)


def _capture_segment(
    *,
    sample_rate: int,
    channels: int,
    device_index: int,
    max_capture_s: float,
) -> np.ndarray:
    """Record a fixed-duration segment and trim trailing silence."""
    total_frames = max(int(max_capture_s * sample_rate), 1)
    segment = sd.rec(
        total_frames,
        samplerate=sample_rate,
        channels=channels,
        dtype="float32",
        device=device_index,
    )
    sd.wait()
    return _trim_trailing_silence(np.asarray(segment, dtype=np.float32), total_frames)


def _estimate_playback_duration(result: TurnResult) -> float:
    """Estimate TTS playback duration from generated audio or measured playback time."""
    playback_duration_s = 0.0

    if result.audio_samples is not None and hasattr(result.audio_samples, "__len__"):
        sample_count = len(result.audio_samples)
        if sample_count > 0 and result.sample_rate > 0:
            playback_duration_s = sample_count / float(result.sample_rate)

    if playback_duration_s <= 0.0 and result.metrics.playback_time_s > 0.0:
        playback_duration_s = result.metrics.playback_time_s

    return playback_duration_s


def _apply_post_turn_lockout(
    *,
    success: bool,
    result: TurnResult,
    sample_rate: int,
    channels: int,
    device_index: int,
    probe_window_s: float,
) -> None:
    """Delay microphone re-arming to avoid capturing residual TTS playback."""
    if not success:
        time.sleep(0.5)
        return

    if result.metrics.tts_time_s <= 0.0 and result.metrics.playback_time_s <= 0.0:
        return

    playback_duration_s = _estimate_playback_duration(result)
    if playback_duration_s <= 0.0:
        return

    cooldown_s = playback_duration_s + 2.0
    logger.info(
        "TTS lockout: waiting %.1fs (playback=%.1fs + margin=2.0s)",
        cooldown_s,
        playback_duration_s,
    )
    time.sleep(cooldown_s)

    _ = _capture_probe(
        sample_rate=sample_rate,
        channels=channels,
        device_index=device_index,
        probe_window_s=probe_window_s,
    )


def _save_input_wav(
    *,
    output_dir: Path,
    segment_idx: int,
    audio_array: np.ndarray,
    sample_rate: int,
) -> Path:
    """Persist the captured input segment for post-hoc analysis."""
    import soundfile as sf  # type: ignore[import-not-found, import-untyped]

    wavs_in_dir = output_dir / "wavs_in"
    wavs_in_dir.mkdir(parents=True, exist_ok=True)
    input_path = wavs_in_dir / f"segment_{segment_idx:02d}_input.wav"
    sf.write(str(input_path), audio_array, sample_rate)
    return input_path


def _append_jsonl_record(output_path: Path, record: dict[str, Any]) -> None:
    """Append one JSONL record to the output file."""
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False))
        handle.write("\n")


def _build_pipeline_config(language: str, warmup: bool) -> PipelineConfig:
    """Return a live-runner pipeline configuration."""
    return PipelineConfig(
        stt_language=language,
        play_tts_audio=True,
        enable_warmup=warmup,
        enable_stt_preload=True,
        vad_threshold=0.3,
        vad_min_speech_ms=100,
    )


def _append_summary_record(
    *,
    output_path: Path,
    lang: str,
    requested_rounds: int,
    completed_segments: int,
    success_count: int,
) -> None:
    """Append the final run summary to rounds.jsonl."""
    _append_jsonl_record(
        output_path,
        {
            "record_type": "summary",
            "timestamp_utc": datetime.utcnow().isoformat(),
            "timestamp_monotonic_s": time.monotonic(),
            "lang": lang,
            "requested_rounds": requested_rounds,
            "completed_segments": completed_segments,
            "success_count": success_count,
            "fail_count": completed_segments - success_count,
            "shutdown_requested": _shutdown_requested,
        },
    )


def main() -> int:
    """Run the blocking voice capture loop."""
    global _shutdown_requested
    _shutdown_requested = False

    args = build_parser().parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rounds_path = output_dir / "rounds.jsonl"
    tegrastats_log_path = output_dir / "tegrastats.log"

    _configure_logging()
    _register_signal_handlers()

    state = RunnerState.STARTUP
    tegrastats: TegrastatsMonitor | None = None
    manager: ModelManager | None = None
    completed_segments = 0
    success_count = 0

    try:
        run_preflight(skip=args.skip_preflight)

        manager = ModelManager(ManagerConfig(llm_resident=True))
        manager.initialize()
        pipeline = ConversationPipeline(manager, _build_pipeline_config(args.lang, args.warmup))
        if args.warmup:
            pipeline.warmup_llm()
        manager.preload_stt()

        device_index, sample_rate, channels = find_usb_input(args.input_device)
        device_name = str(sd.query_devices(device_index).get("name", "?"))

        tegrastats = TegrastatsMonitor(tegrastats_log_path)
        tegrastats.start()

        logger.info(
            "Ready for %s batch. Press play on Sony recorder. Input=[%d] %s (%dHz, %dch)",
            args.lang,
            device_index,
            device_name,
            sample_rate,
            channels,
        )
        state = _transition_state(state, RunnerState.ARMED)

        while completed_segments < args.rounds and not _shutdown_requested:
            probe = _capture_probe(
                sample_rate=sample_rate,
                channels=channels,
                device_index=device_index,
                probe_window_s=args.probe_window_s,
            )
            probe_mono = _to_mono_float32(probe)
            probe_level_db = rms_db(probe_mono)
            logger.info("ARMED probe RMS: %.1f dB", probe_level_db)
            if probe_level_db <= args.energy_threshold_db:
                continue

            state = _transition_state(state, RunnerState.CAPTURING)
            captured = _capture_segment(
                sample_rate=sample_rate,
                channels=channels,
                device_index=device_index,
                max_capture_s=args.max_capture_s,
            )
            audio_array = np.concatenate([probe_mono, _to_mono_float32(captured)])
            audio_16k = _resample_to_16k(audio_array, sample_rate)
            captured_duration = audio_array.shape[0] / float(sample_rate)
            logger.info(
                "Captured segment candidate: duration=%.3fs rms=%.1f dB",
                captured_duration,
                rms_db(audio_array) if audio_array.size else float("-inf"),
            )

            if captured_duration < args.min_capture_s:
                logger.info(
                    "Skipping short segment %.3fs (< %.3fs)",
                    captured_duration,
                    args.min_capture_s,
                )
                manager.preload_stt()
                state = _transition_state(state, RunnerState.ARMED)
                continue

            state = _transition_state(state, RunnerState.PROCESS)
            segment_idx = completed_segments + 1
            input_path = _save_input_wav(
                output_dir=output_dir,
                segment_idx=segment_idx,
                audio_array=audio_array,
                sample_rate=sample_rate,
            )
            result: TurnResult = pipeline.run_turn(audio_16k, sample_rate=16000)
            success = bool(result.user_text.strip())
            record = {
                "segment_idx": segment_idx,
                "timestamp_utc": datetime.utcnow().isoformat(),
                "timestamp_monotonic_s": time.monotonic(),
                "lang": args.lang,
                "captured_duration_s": round(captured_duration, 3),
                "captured_rms_db": round(rms_db(audio_array), 1),
                "input_wav": str(input_path.relative_to(output_dir)),
                "user_text": result.user_text,
                "response_text": result.response_text,
                "speech_segments": result.metrics.speech_segments,
                "vad_time_s": round(result.metrics.vad_time_s, 3),
                "stt_time_s": round(result.metrics.stt_time_s, 3),
                "stt_load_time_s": round(result.metrics.stt_load_time_s, 3),
                "llm_time_s": round(result.metrics.llm_time_s, 3),
                "llm_load_time_s": round(result.metrics.llm_load_time_s, 3),
                "llm_model_fallback_used": result.metrics.llm_model_fallback_used,
                "llm_model_path_actual": result.metrics.llm_model_path_actual,
                "llm_model_fallback_reason": result.metrics.llm_model_fallback_reason,
                "tts_time_s": round(result.metrics.tts_time_s, 3),
                "tts_load_time_s": round(result.metrics.tts_load_time_s, 3),
                "total_time_s": round(result.metrics.total_time_s, 3),
                "success": success,
            }
            _append_jsonl_record(rounds_path, record)

            completed_segments += 1
            if success:
                success_count += 1
            logger.info(
                "Segment %d/%d processed: success=%s user_text=%r total=%.3fs",
                segment_idx,
                args.rounds,
                success,
                result.user_text,
                result.metrics.total_time_s,
            )
            _apply_post_turn_lockout(
                success=success,
                result=result,
                sample_rate=sample_rate,
                channels=channels,
                device_index=device_index,
                probe_window_s=args.probe_window_s,
            )
            manager.preload_stt()
            state = _transition_state(state, RunnerState.ARMED)
    except KeyboardInterrupt:
        _shutdown_requested = True
        logger.info("Shutdown requested (keyboard interrupt)")
    finally:
        state = _transition_state(state, RunnerState.SHUTDOWN)
        if tegrastats is not None:
            tegrastats.stop()
        if manager is not None:
            try:
                manager.unload_all()
            except Exception:
                logger.warning("Failed to unload models during shutdown", exc_info=True)
        _append_summary_record(
            output_path=rounds_path,
            lang=args.lang,
            requested_rounds=args.rounds,
            completed_segments=completed_segments,
            success_count=success_count,
        )
        logger.info(
            "Final summary: total=%d success=%d fail=%d shutdown_requested=%s",
            completed_segments,
            success_count,
            completed_segments - success_count,
            _shutdown_requested,
        )

    return 0 if not _shutdown_requested else 130


if __name__ == "__main__":
    raise SystemExit(main())
