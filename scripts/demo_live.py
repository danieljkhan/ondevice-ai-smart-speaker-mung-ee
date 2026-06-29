"""Live demo: mic -> VAD -> STT -> LLM -> TTS -> speaker.

Usage:
    python scripts/demo_live.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import tempfile
import time
import warnings
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from core.audio_capture import AudioCapture
from core.bgm_player import BgmPlayer
from core.event_log import EventLog
from core.funny_english_mode import FunnyEnglishModeController
from core.history_mode import HistoryModeController
from core.model_manager import ManagerConfig, ModelManager
from core.pipeline import ConversationPipeline, PipelineConfig, TurnResult
from core.runtime import detect_runtime_paths
from core.session_manager import (
    CharacterRenderer,
    NullCharacterRenderer,
    PipelineFactory,
    SessionManager,
    SessionModelManager,
)
from core.sound_bank import SoundBank
from core.system_probe import (
    DEFAULT_INPUT_DEVICE_NAME,
)
from core.system_probe import (
    find_usb_input as _find_usb_input,
)
from core.system_probe import (
    read_max_thermal_c as _read_max_thermal_c,
)
from core.system_probe import (
    read_meminfo_snapshot as _read_meminfo_snapshot,
)
from hardware import audio_player
from hardware.touch_input import TouchInputListener
from safety.content_filter import ContentFilter

__all__ = [
    "_find_usb_input",
    "_read_max_thermal_c",
    "_read_meminfo_snapshot",
]

logger = logging.getLogger(__name__)

MAX_SECONDS = int(os.getenv("MUNGI_DEMO_MAX_RECORDING_SECONDS", "20"))
DEMO_LLM_MAX_TOKENS = 80
DEMO_LLM_TEMPERATURE = 0.2
DEMO_LLM_TOP_P = 0.7
DEMO_LLM_REPEAT_PENALTY = 1.3
INPUT_DEVICE_NAME = (
    os.getenv("MUNGI_AUDIO_INPUT_DEVICE", DEFAULT_INPUT_DEVICE_NAME).strip()
    or DEFAULT_INPUT_DEVICE_NAME
)
OUTPUT_DEVICE_NAME = os.getenv("MUNGI_AUDIO_OUTPUT_DEVICE", "").strip() or None
JSONL_PATH_ENV = "MUNGI_DEMO_JSONL"
USB_AUDIO_WAIT_ENV = "MUNGI_USB_AUDIO_WAIT_S"
USB_AUDIO_WAIT_DEFAULT_S = 5.0
_USB_AUDIO_POLL_INTERVAL_S = 0.5
CONVERSATION_LOG_DIR = Path(detect_runtime_paths().mutable_root) / "conversations"
_MODEL_ERROR_MARKERS = (
    "failed to load model",
    "failed to create llama_context",
    "out of memory",
    "nvmapmemallocinternaltagged",
    "cuda",
)

warnings.warn(
    "scripts.demo_live private system probe shims are deprecated; "
    "import from core.system_probe instead.",
    DeprecationWarning,
    stacklevel=2,
)


def _build_pipeline_config() -> PipelineConfig:
    """Return live-demo pipeline settings tuned for responsiveness."""
    base_prompt = PipelineConfig().llm_system_prompt
    # Strip /no_think from base so we can append rules before it
    base_body = base_prompt.replace("/no_think", "").rstrip()
    live_prompt = (
        f"{base_body}\n"
        "\n"
        "ADDITIONAL RULES FOR LIVE DEMO:\n"
        "- The user speaks Korean through a microphone. STT may produce noisy text.\n"
        "- Interpret the user's intent in English internally, then respond in Korean.\n"
        "- Use only simple words a small child can understand.\n"
        "- Focus on one key point per response.\n"
        "- If you said something wrong, correct it immediately.\n"
        "- If unsure, say you don't know instead of guessing.\n"
        "/no_think"
    )
    return PipelineConfig(
        play_tts_audio=True,
        tts_output_device=OUTPUT_DEVICE_NAME,
        llm_max_tokens=DEMO_LLM_MAX_TOKENS,
        llm_temperature=DEMO_LLM_TEMPERATURE,
        llm_top_p=DEMO_LLM_TOP_P,
        llm_repeat_penalty=DEMO_LLM_REPEAT_PENALTY,
        llm_system_prompt=live_prompt,
        max_history_turns=1,
        max_history_entries=20,
        adaptive_history_threshold_s=15.0,
        enable_warmup=True,
        enable_stt_preload=True,
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build the live-demo command-line parser."""
    parser = argparse.ArgumentParser(description="Run the Mungi live microphone demo.")
    parser.add_argument(
        "--stt-model",
        default=None,
        help=(
            "STT model selector or alias. Overrides MUNGI_STT_MODEL_SIZE when provided; "
            "otherwise ManagerConfig falls back to env/default selection."
        ),
    )
    return parser


def _build_manager_config(stt_model: str | None = None) -> ManagerConfig:
    """Build ManagerConfig with CLI/env STT model-size precedence."""
    selected_stt_model = (stt_model or "").strip()
    if not selected_stt_model:
        selected_stt_model = os.getenv("MUNGI_STT_MODEL_SIZE", "").strip()

    if selected_stt_model:
        return ManagerConfig(stt_model_size=selected_stt_model)
    return ManagerConfig()


def _resolve_jsonl_path(now: datetime | None = None) -> Path:
    """Resolve the per-turn JSONL log path for this live-demo session."""
    env_path = os.getenv(JSONL_PATH_ENV, "").strip()
    if env_path:
        return Path(env_path).expanduser()

    if not sys.platform.startswith("linux"):
        return Path(tempfile.gettempdir()) / "mungi-demo.jsonl"

    stamp = (now or datetime.now()).strftime("%Y-%m-%d_%H-%M-%S")
    return CONVERSATION_LOG_DIR / f"demo_{stamp}.jsonl"


def _rotate_jsonl_path(log_path: Path) -> None:
    """Rotate an existing JSONL file to a single ``.bak`` copy."""
    if not log_path.exists():
        return
    backup_path = log_path.with_suffix(f"{log_path.suffix}.bak")
    backup_path.unlink(missing_ok=True)
    log_path.replace(backup_path)


def _friendly_error_message(error: str | None) -> str:
    """Map internal exceptions to short user-facing live-demo guidance."""
    lowered = (error or "").lower()
    if any(marker in lowered for marker in _MODEL_ERROR_MARKERS):
        return "모델 로드에 실패했어. 잠시 후 다시 시도해줘."
    return "문제가 생겼어. 한 번 더 말해줄래?"


def _build_turn_log_entry(
    turn: int,
    recorded_duration_s: float,
    result: TurnResult,
    memory_before_mb: dict[str, int],
    memory_after_mb: dict[str, int],
    llm_diag: dict[str, Any],
) -> dict[str, Any]:
    """Build a JSON-serializable record for one live-demo turn."""
    status = "ok"
    if result.error:
        status = "error"
    elif not result.user_text:
        status = "no_speech"

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "turn": turn,
        "status": status,
        "success": result.success,
        "state": result.state.value,
        "error": result.error,
        "recorded_duration_s": round(recorded_duration_s, 3),
        "mem_free_mb": memory_after_mb.get("memfree_mb", memory_before_mb.get("memfree_mb")),
        "mem_avail_mb": memory_after_mb.get(
            "memavailable_mb",
            memory_before_mb.get("memavailable_mb"),
        ),
        "n_gpu_layers": llm_diag.get("loaded_n_gpu_layers"),
        "selected_n_gpu_layers": llm_diag.get("selected_n_gpu_layers"),
        "fallback_used": llm_diag.get("fallback_used", False),
        "thermal_c": _read_max_thermal_c(),
        "stt_load_s": round(result.metrics.stt_load_time_s, 3),
        "stt_s": round(result.metrics.stt_time_s, 3),
        "llm_load_s": round(result.metrics.llm_load_time_s, 3),
        "llm_s": round(result.metrics.llm_time_s, 3),
        "llm_tokens": result.metrics.llm_tokens,
        "llm_model_fallback_used": result.metrics.llm_model_fallback_used,
        "llm_model_path_actual": result.metrics.llm_model_path_actual,
        "llm_model_fallback_reason": result.metrics.llm_model_fallback_reason,
        "tts_load_s": round(result.metrics.tts_load_time_s, 3),
        "tts_s": round(result.metrics.tts_time_s, 3),
        "play_s": round(result.metrics.playback_time_s, 3),
        "total_s": round(result.metrics.total_time_s, 3),
        "user_text": result.user_text,
        "response_text": result.response_text,
        "user_text_chars": len(result.user_text),
        "response_text_chars": len(result.response_text),
        "response_len": len(result.response_text),
        "content_filter_triggered": result.metrics.content_filter_blocked,
        "metrics": result.metrics.to_dict(),
        "memory_before_mb": memory_before_mb,
        "memory_after_mb": memory_after_mb,
        "llm": llm_diag,
    }


def _append_turn_log(log_path: Path, entry: dict[str, Any]) -> None:
    """Append one JSONL record for later latency/memory analysis."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False))
        handle.write("\n")


def _make_pipeline_factory() -> PipelineFactory:
    """Build the SessionManager pipeline factory for the live demo."""

    def _factory(model_manager: SessionModelManager) -> ConversationPipeline:
        config = _build_pipeline_config()
        content_filter: ContentFilter | None = None
        if config.enable_content_filter:
            content_filter = ContentFilter.from_default()
            logger.info(
                "Content filter active for demo live pipeline: %d categories, %d patterns",
                content_filter.category_count,
                content_filter.pattern_count,
            )
        return ConversationPipeline(
            cast(ModelManager, model_manager),
            config,
            content_filter=content_filter,
        )

    return _factory


def _usb_audio_wait_seconds() -> float:
    raw_value = os.getenv(USB_AUDIO_WAIT_ENV, "").strip()
    if not raw_value:
        return USB_AUDIO_WAIT_DEFAULT_S
    try:
        return max(0.0, float(raw_value))
    except ValueError:
        logger.warning(
            "Invalid %s=%r; using %.1fs",
            USB_AUDIO_WAIT_ENV,
            raw_value,
            USB_AUDIO_WAIT_DEFAULT_S,
        )
        return USB_AUDIO_WAIT_DEFAULT_S


def _refresh_sounddevice_devices() -> None:
    """Best-effort refresh for PortAudio's device enumeration cache."""
    try:
        import sounddevice as sd  # type: ignore[import-not-found, import-untyped]

        terminate = getattr(sd, "_terminate", None)
        initialize = getattr(sd, "_initialize", None)
        if callable(terminate) and callable(initialize):
            terminate()
            initialize()
            return
        sd.query_devices()
    except Exception as exc:
        logger.debug("Sounddevice device refresh failed: %s", exc)


def _build_audio_capture() -> AudioCapture:
    """Create AudioCapture for the configured USB microphone."""
    input_info = _find_usb_input(allow_fallback=False)
    wait_seconds = 0.0
    if input_info is None:
        wait_seconds = _usb_audio_wait_seconds()
        if wait_seconds > 0:
            logger.info(
                "USB input device '%s' not found; waiting up to %.1fs",
                INPUT_DEVICE_NAME,
                wait_seconds,
            )
            deadline = time.monotonic() + wait_seconds
            while input_info is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(_USB_AUDIO_POLL_INTERVAL_S, remaining))
                _refresh_sounddevice_devices()
                input_info = _find_usb_input(allow_fallback=False)
    if input_info is None:
        logger.warning(
            "USB input device '%s' not found after %.1fs; trying first available input",
            INPUT_DEVICE_NAME,
            wait_seconds,
        )
        input_info = _find_usb_input()
    if input_info is None:
        msg = f"No input-capable audio device found after waiting for '{INPUT_DEVICE_NAME}'"
        raise RuntimeError(msg)
    return AudioCapture(
        sample_rate=int(input_info["sample_rate"]),
        channels=int(input_info["channels"]),
        device=int(input_info["index"]),
    )


def _display_available() -> bool:
    """Return whether the environment advertises a usable SDL display."""
    return (
        bool(os.environ.get("DISPLAY"))
        or bool(os.environ.get("WAYLAND_DISPLAY"))
        or os.environ.get("SDL_VIDEODRIVER") in {"kmsdrm", "fbdev", "dummy"}
        or os.environ.get("MUNGI_SDL_DRIVER") in {"kmsdrm", "fbdev", "dummy"}
    )


def _build_renderer() -> CharacterRenderer:
    """Build the configured character renderer for the live demo."""
    mode = os.environ.get("MUNGI_RENDERER", "auto").lower()
    if mode == "null":
        return NullCharacterRenderer()
    if mode == "pygame" or (mode == "auto" and _display_available()):
        try:
            from core.character_renderer import PygameCharacterRenderer

            return PygameCharacterRenderer(
                asset_dir=Path("assets/character"),
                windowed=os.environ.get("MUNGI_RENDERER_WINDOWED") == "1",
                sdl_driver=os.environ.get("MUNGI_SDL_DRIVER"),
            )
        except Exception as exc:
            logger.warning("Pygame renderer init failed, falling back to null: %s", exc)
            return NullCharacterRenderer()
    return NullCharacterRenderer()


def _play_chime(sound_bank: SoundBank) -> None:
    """Play the long-press chime best-effort."""
    try:
        chime_audio, sample_rate = sound_bank.chime()
        audio_player.play_audio(
            chime_audio,
            sample_rate,
            device=OUTPUT_DEVICE_NAME,
            blocking=False,
        )
    except Exception as exc:
        logger.warning("Long-press chime playback failed: %s", exc)


def main(argv: list[str] | None = None) -> int:
    """Run the touchscreen-driven live demo loop."""

    args = _build_parser().parse_args(argv or [])

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )
    logger.info("\n  Mungi Live Demo (Ctrl+C to quit)\n")
    sigtstp = getattr(signal, "SIGTSTP", None)
    if sigtstp is not None:
        signal.signal(sigtstp, signal.SIG_IGN)

    mm = ModelManager(_build_manager_config(args.stt_model))
    touch: TouchInputListener | None = None
    audio_capture: AudioCapture | None = None
    session_manager: SessionManager | None = None
    bgm_player: BgmPlayer | None = None
    try:
        mm.initialize()
        sound_bank = SoundBank(Path(os.getenv("MUNGI_SOUND_DIR", "assets/sounds")))
        paths = detect_runtime_paths()
        event_log = EventLog(Path(paths.mutable_root) / "events" / "touchscreen.jsonl")
        audio_capture = _build_audio_capture()

        def _history_safe_chime() -> None:
            if session_manager is not None and (
                session_manager.is_history_mode_active()
                or session_manager.is_funny_english_mode_active()
            ):
                logger.debug("Suppressing long-press chime during special mode")
                return
            _play_chime(sound_bank)

        touch = TouchInputListener(chime_callback=_history_safe_chime)
        renderer = _build_renderer()
        bgm_player = BgmPlayer(device=OUTPUT_DEVICE_NAME)
        session_manager = SessionManager(
            mm=mm,
            pipeline_factory=_make_pipeline_factory(),
            touch=touch,
            sound_bank=sound_bank,
            audio_capture=audio_capture,
            event_log=event_log,
            renderer=renderer,
        )
        funny_english_controller = FunnyEnglishModeController(
            session=session_manager,
            model_manager=mm,
            renderer=renderer,
            repo_root=Path("."),
            bgm_player=bgm_player,
            bgm_clip=sound_bank.funny_english_bgm(),
        )
        history_controller = HistoryModeController(
            session=session_manager,
            model_manager=mm,
            renderer=renderer,
            repo_root=Path("."),
        )
        session_manager.set_funny_english_controller(funny_english_controller)
        session_manager.set_history_controller(history_controller)
        touch.start()
        session_manager.run()
    except KeyboardInterrupt:
        logger.info("\n\n  Bye!")
    finally:
        if session_manager is not None:
            with suppress(Exception):
                session_manager.shutdown()
        if touch is not None:
            touch.stop()
        if audio_capture is not None:
            audio_capture.close()
        if bgm_player is not None:
            bgm_player.close()
        mm.unload_all()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
