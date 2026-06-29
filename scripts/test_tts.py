"""Supertonic TTS standalone test script for Jetson Orin Nano.

Loads the Supertonic 2 TTS engine, synthesizes speech from text, and
reports timing, RTF, and memory metrics.

Usage:
    python scripts/test_tts.py
    python scripts/test_tts.py --engine supertonic --text "안녕하세요"
    python scripts/test_tts.py --output-wav /tmp/output.wav
    python scripts/test_tts.py --text-unicode-escape "\uc548\ub155\ud558\uc138\uc694"
    python scripts/test_tts.py --play --output-device "USB PnP Audio Device"
"""

from __future__ import annotations

import argparse
import codecs
import json
import logging
import os
import struct
import sys
import time
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if TYPE_CHECKING:
    import numpy as np

from models.tts_runner import (  # noqa: E402
    SupertonicEngine,
    TTSEngine,
    normalize_tts_text,
)
from scripts.utils import get_peak_memory_kb  # noqa: E402

logger = logging.getLogger("mungi.scripts.test_tts")

# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------

DEFAULT_ENGINE: str = "supertonic"
DEFAULT_TEXT: str = "안녕하세요, 저는 뭉이예요. 오늘 기분이 어때요?"
DEFAULT_SUPERTONIC_MODEL_DIR: str = "/opt/mungi/ai_models/supertonic-2"
DEFAULT_SAMPLE_RATE: int = 22050


# -------------------------------------------------------------------
# Text helpers
# -------------------------------------------------------------------


def resolve_text_input(
    text: str | None,
    text_unicode_escape: str | None,
) -> str:
    """Resolve CLI text input, decoding unicode-escaped payloads when needed."""
    if text_unicode_escape is not None:
        try:
            return normalize_tts_text(codecs.decode(text_unicode_escape, "unicode_escape"))
        except UnicodeDecodeError as exc:
            msg = "--text-unicode-escape must contain valid unicode escapes."
            raise ValueError(msg) from exc

    if text is not None:
        return normalize_tts_text(text)

    return DEFAULT_TEXT


# -------------------------------------------------------------------
# Data classes
# -------------------------------------------------------------------


@dataclass(frozen=True)
class TTSResult:
    """Full result of a TTS synthesis run including metrics."""

    engine: str
    text: str
    model_load_time_s: float
    synthesis_time_s: float
    audio_duration_s: float
    rtf: float
    peak_memory_kb: int
    sample_rate: int
    num_samples: int
    success: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result to a plain dict."""
        return asdict(self)


# -------------------------------------------------------------------
# WAV writing
# -------------------------------------------------------------------


def write_wav(
    path: Path,
    samples: np.ndarray,
    sample_rate: int,
) -> None:
    """Write a numpy float32 array to a 16-bit mono WAV file.

    Args:
        path: Output WAV file path.
        samples: Audio samples as float32 in [-1.0, 1.0].
        sample_rate: Sample rate in Hz.
    """
    import numpy as np

    # Clip and convert to 16-bit PCM
    clipped = np.clip(samples, -1.0, 1.0)
    pcm_16 = (clipped * 32767).astype(np.int16)
    raw_data = struct.pack(f"<{len(pcm_16)}h", *pcm_16)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(raw_data)

    logger.info(
        "WAV saved: %s (%d samples, %d Hz)",
        path,
        len(pcm_16),
        sample_rate,
    )


# -------------------------------------------------------------------
# Engine factory
# -------------------------------------------------------------------


def create_engine(
    engine_name: str,
    model_dir: str,
) -> TTSEngine:
    """Create a TTS engine instance by name.

    Args:
        engine_name: Engine identifier ("supertonic").
        model_dir: Path to Supertonic model directory.

    Returns:
        Configured TTSEngine instance.

    Raises:
        ValueError: If engine_name is not recognized.
    """
    if engine_name == "supertonic":
        return SupertonicEngine(model_dir=model_dir)

    msg = f"Unknown engine: '{engine_name}'. Choose 'supertonic'."
    raise ValueError(msg)


# -------------------------------------------------------------------
# Main entry point
# -------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the TTS test script.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        description=(
            "TTS standalone test -- load Supertonic 2, synthesize speech, and report metrics."
        ),
    )
    parser.add_argument(
        "--engine",
        type=str,
        default=DEFAULT_ENGINE,
        choices=["supertonic"],
        help=(f"TTS engine to test (default: {DEFAULT_ENGINE})."),
    )
    parser.add_argument(
        "--text",
        type=str,
        default=DEFAULT_TEXT,
        help="Text to synthesize (default: Korean test phrase).",
    )
    parser.add_argument(
        "--text-unicode-escape",
        type=str,
        default=None,
        help=(
            "Unicode-escaped text input (for example, "
            "'\\uc548\\ub155\\ud558\\uc138\uc694'). Overrides --text when provided."
        ),
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=DEFAULT_SUPERTONIC_MODEL_DIR,
        help=(f"Path to Supertonic 2 model directory (default: {DEFAULT_SUPERTONIC_MODEL_DIR})."),
    )
    parser.add_argument(
        "--output-wav",
        type=Path,
        default=None,
        help="Optional path to save synthesized audio as WAV.",
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="Play synthesized audio locally via sounddevice.",
    )
    parser.add_argument(
        "--output-device",
        type=str,
        default=os.getenv("MUNGI_AUDIO_OUTPUT_DEVICE", "").strip() or None,
        help="Optional sounddevice output device name or index.",
    )
    return parser


def main() -> int:
    """Run the TTS test and report results.

    Returns:
        Exit code: 0 on success, 1 on error.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = build_parser()
    args = parser.parse_args()
    try:
        text = resolve_text_input(args.text, args.text_unicode_escape)
    except ValueError as exc:
        logger.error("Invalid text input: %s", exc)
        return 1

    logger.info(
        "TTS test starting -- engine=%s, text='%s'",
        args.engine,
        text[:50],
    )

    # Create engine
    try:
        engine = create_engine(
            engine_name=args.engine,
            model_dir=args.model_dir,
        )
    except ValueError as exc:
        logger.error("Engine creation failed: %s", exc)
        return 1

    # Load model
    try:
        mem_before = get_peak_memory_kb()
        t0 = time.monotonic()
        engine.load()
        load_time = time.monotonic() - t0
        logger.info("Model loaded in %.3f seconds", load_time)
    except (ImportError, FileNotFoundError, RuntimeError) as exc:
        logger.error("Model load failed: %s", exc)
        result = TTSResult(
            engine=args.engine,
            text=text,
            model_load_time_s=0.0,
            synthesis_time_s=0.0,
            audio_duration_s=0.0,
            rtf=0.0,
            peak_memory_kb=0,
            sample_rate=0,
            num_samples=0,
            success=False,
            error=str(exc),
        )
        output = json.dumps(result.to_dict(), indent=2, ensure_ascii=False)
        logger.info("TTS result:\n%s", output)
        return 1

    # Synthesize
    try:
        t0 = time.monotonic()
        audio, sample_rate = engine.synthesize(text)
        synthesis_time = time.monotonic() - t0
        peak_memory = get_peak_memory_kb()
        logger.info("Synthesis completed in %.3f seconds", synthesis_time)
    except RuntimeError as exc:
        logger.error("Synthesis failed: %s", exc)
        result = TTSResult(
            engine=args.engine,
            text=text,
            model_load_time_s=round(load_time, 4),
            synthesis_time_s=0.0,
            audio_duration_s=0.0,
            rtf=0.0,
            peak_memory_kb=0,
            sample_rate=0,
            num_samples=0,
            success=False,
            error=str(exc),
        )
        output = json.dumps(result.to_dict(), indent=2, ensure_ascii=False)
        logger.info("TTS result:\n%s", output)
        return 1

    # Calculate metrics
    num_samples = len(audio)
    audio_duration = num_samples / sample_rate if sample_rate > 0 else 0.0
    rtf = synthesis_time / audio_duration if audio_duration > 0 else 0.0

    # Build result
    result = TTSResult(
        engine=args.engine,
        text=text,
        model_load_time_s=round(load_time, 4),
        synthesis_time_s=round(synthesis_time, 4),
        audio_duration_s=round(audio_duration, 4),
        rtf=round(rtf, 4),
        peak_memory_kb=max(peak_memory - mem_before, 0),
        sample_rate=sample_rate,
        num_samples=num_samples,
        success=True,
    )

    # Report JSON
    output = json.dumps(result.to_dict(), indent=2, ensure_ascii=False)
    logger.info("TTS result:\n%s", output)

    # Summary
    logger.info("--- Summary ---")
    logger.info("Engine:          %s", args.engine)
    logger.info("Text:            %s", text[:60])
    logger.info("Sample rate:     %d Hz", sample_rate)
    logger.info("Audio samples:   %d", num_samples)
    logger.info("Audio duration:  %.2f s", audio_duration)
    logger.info("Model load time: %.3f s", load_time)
    logger.info("Synthesis time:  %.3f s", synthesis_time)
    logger.info("RTF:             %.4f", rtf)
    logger.info("Peak memory:     %d KB", result.peak_memory_kb)
    if rtf < 1.0:
        logger.info("RTF < 1.0 -- faster than real-time")
    else:
        logger.info("RTF >= 1.0 -- slower than real-time")

    # Save WAV if requested
    if args.output_wav is not None:
        try:
            write_wav(args.output_wav, audio, sample_rate)
        except OSError as exc:
            logger.error("Failed to save WAV: %s", exc)
            return 1

    if args.play:
        try:
            from hardware.audio_player import play_audio

            play_audio(audio, sample_rate, device=args.output_device)
        except (ImportError, RuntimeError) as exc:
            logger.error("Failed to play audio: %s", exc)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
