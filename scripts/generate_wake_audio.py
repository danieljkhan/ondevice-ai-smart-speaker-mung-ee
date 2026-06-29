"""Generate the Phase 1 touchscreen wake voice WAV sound bank."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import wave
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.tts_runner import SupertonicEngine  # noqa: E402

LOGGER = logging.getLogger("mungi.scripts.generate_wake_audio")

DEFAULT_SCRIPTS_JSON = Path("assets/sounds/scripts.json")
DEFAULT_OUTPUT_ROOT = Path("assets/sounds")
DEFAULT_VOICE_JSON = Path(
    os.getenv("MUNGI_TTS_VOICE_STYLE_KO", "").strip() or "/var/lib/mungi/voices/tobi.json"
)
DEFAULT_MODEL_DIR = Path(
    os.getenv("MUNGI_TTS_MODEL_DIR", "").strip() or "/opt/mungi/ai_models/supertonic-2"
)
EXPECTED_VERSION = "1.0"
EXPECTED_SAMPLE_RATE = 44_100
INT16_MAX = 32_767.0
CATEGORIES = (
    "welcome_morning",
    "welcome_afternoon",
    "welcome_evening",
    "welcome_night",
    "wake_ack",
    "stt_load_fail",
    "sleep",
)
EXPECTED_CATEGORY_COUNTS = {
    "welcome_morning": 3,
    "welcome_afternoon": 3,
    "welcome_evening": 4,
    "welcome_night": 3,
    "wake_ack": 8,
    "stt_load_fail": 3,
    "sleep": 3,
}


class TtsEngine(Protocol):
    """Minimal synthesis engine protocol used by this generator."""

    def load(self) -> None:
        """Load model and voice assets."""

    def synthesize(
        self,
        text: str | None,
        language: str = "ko",
        total_steps: int = 7,
    ) -> tuple[np.ndarray, int]:
        """Synthesize text into mono float samples and a sample rate."""


@dataclass(frozen=True)
class SoundScriptEntry:
    """One voice sound-bank script entry."""

    path: str
    text: str
    category: str
    language: str
    duration_target_s: float | None = None


@dataclass(frozen=True)
class SoundScriptsManifest:
    """Parsed sound-bank manifest."""

    version: str
    voice_style: str
    sample_rate: int
    speed: float
    language_default: str
    generated_by: str
    design_notes: str
    files: tuple[SoundScriptEntry, ...]


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the wake-audio generator."""
    parser = argparse.ArgumentParser(
        description="Generate touchscreen wake/error/sleep voice WAVs via Supertonic.",
    )
    parser.add_argument(
        "--scripts-json",
        type=Path,
        default=DEFAULT_SCRIPTS_JSON,
        help=f"Sound-bank script manifest path (default: {DEFAULT_SCRIPTS_JSON}).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Output sound root (default: {DEFAULT_OUTPUT_ROOT}).",
    )
    parser.add_argument(
        "--voice-json",
        type=Path,
        default=DEFAULT_VOICE_JSON,
        help=f"Supertonic voice JSON path (default: {DEFAULT_VOICE_JSON}).",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=f"Supertonic model directory (default: {DEFAULT_MODEL_DIR}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the synthesis plan without invoking Supertonic.",
    )
    parser.add_argument(
        "--filter",
        dest="category_filter",
        choices=CATEGORIES,
        default=None,
        help="Generate only one category.",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=7,
        help="Supertonic total_steps for offline cue rendering (default: 7).",
    )
    return parser


def _expect_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        msg = f"manifest field {key!r} must be a non-empty string"
        raise ValueError(msg)
    return value


def _expect_number(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if not isinstance(value, int | float):
        msg = f"manifest field {key!r} must be numeric"
        raise ValueError(msg)
    return float(value)


def _normalize_manifest_path(path_value: str) -> str:
    wav_path = Path(path_value)
    if wav_path.is_absolute() or any(part == ".." for part in wav_path.parts):
        msg = f"sound-bank path must stay relative to output root: {path_value}"
        raise ValueError(msg)
    if wav_path.suffix.lower() != ".wav":
        msg = f"sound-bank path must end with .wav: {path_value}"
        raise ValueError(msg)
    return path_value.replace("\\", "/")


def _parse_entry(raw_entry: Any, index: int) -> SoundScriptEntry:
    if not isinstance(raw_entry, dict):
        msg = f"files[{index}] must be an object"
        raise ValueError(msg)
    path = _normalize_manifest_path(_expect_string(raw_entry, "path"))
    text = _expect_string(raw_entry, "text")
    category = _expect_string(raw_entry, "category")
    language = _expect_string(raw_entry, "language")
    if category not in CATEGORIES:
        msg = f"files[{index}].category is unsupported: {category}"
        raise ValueError(msg)
    if language != "ko":
        msg = f"files[{index}].language must be 'ko', got {language!r}"
        raise ValueError(msg)

    duration_target_raw = raw_entry.get("duration_target_s")
    duration_target_s: float | None
    if duration_target_raw is None:
        duration_target_s = None
    elif isinstance(duration_target_raw, int | float):
        duration_target_s = float(duration_target_raw)
    else:
        msg = f"files[{index}].duration_target_s must be numeric or null"
        raise ValueError(msg)

    return SoundScriptEntry(
        path=path,
        text=text,
        category=category,
        language=language,
        duration_target_s=duration_target_s,
    )


def load_scripts_manifest(path: Path) -> SoundScriptsManifest:
    """Read and validate the sound-bank scripts manifest."""
    try:
        raw_manifest = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        msg = f"scripts manifest not found: {path}"
        raise FileNotFoundError(msg) from exc
    except json.JSONDecodeError as exc:
        msg = f"scripts manifest is invalid JSON: {path}:{exc.lineno}:{exc.colno}"
        raise ValueError(msg) from exc
    except OSError as exc:
        msg = f"failed to read scripts manifest: {path}"
        raise OSError(msg) from exc

    if not isinstance(raw_manifest, dict):
        msg = "scripts manifest must be a JSON object"
        raise ValueError(msg)

    files_raw = raw_manifest.get("files")
    if not isinstance(files_raw, list) or not files_raw:
        msg = "manifest field 'files' must be a non-empty list"
        raise ValueError(msg)
    files = tuple(_parse_entry(raw_entry, index) for index, raw_entry in enumerate(files_raw))
    _validate_phase1_inventory(files)

    version = _expect_string(raw_manifest, "version")
    sample_rate = int(_expect_number(raw_manifest, "sample_rate"))
    if version != EXPECTED_VERSION:
        msg = f"manifest version must be {EXPECTED_VERSION!r}, got {version!r}"
        raise ValueError(msg)
    if sample_rate != EXPECTED_SAMPLE_RATE:
        msg = f"manifest sample_rate must be {EXPECTED_SAMPLE_RATE}, got {sample_rate}"
        raise ValueError(msg)

    return SoundScriptsManifest(
        version=version,
        voice_style=_expect_string(raw_manifest, "voice_style"),
        sample_rate=sample_rate,
        speed=_expect_number(raw_manifest, "speed"),
        language_default=_expect_string(raw_manifest, "language_default"),
        generated_by=_expect_string(raw_manifest, "generated_by"),
        design_notes=_expect_string(raw_manifest, "design_notes"),
        files=files,
    )


def _validate_phase1_inventory(files: tuple[SoundScriptEntry, ...]) -> None:
    paths = [entry.path for entry in files]
    texts = [entry.text for entry in files]
    if len(files) != 27:
        msg = f"Phase 1 voice manifest must contain 27 files, got {len(files)}"
        raise ValueError(msg)
    if len(set(paths)) != len(paths):
        msg = "Phase 1 voice manifest contains duplicate output paths"
        raise ValueError(msg)
    if len(set(texts)) != len(texts):
        msg = "Phase 1 voice manifest contains duplicate text"
        raise ValueError(msg)
    counts = Counter(entry.category for entry in files)
    if counts != EXPECTED_CATEGORY_COUNTS:
        msg = f"Phase 1 category counts mismatch: {dict(counts)}"
        raise ValueError(msg)


def select_entries(
    manifest: SoundScriptsManifest,
    category_filter: str | None,
) -> tuple[SoundScriptEntry, ...]:
    """Return manifest entries matching the optional category filter."""
    if category_filter is None:
        return manifest.files
    return tuple(entry for entry in manifest.files if entry.category == category_filter)


def resolve_output_path(output_root: Path, relative_path: str) -> Path:
    """Resolve one relative manifest path under the output root."""
    normalized_path = _normalize_manifest_path(relative_path)
    root = output_root.resolve()
    candidate = (output_root / Path(normalized_path)).resolve()
    if not candidate.is_relative_to(root):
        msg = f"resolved output path escaped output root: {relative_path}"
        raise ValueError(msg)
    return candidate


def render_synthesis_plan(
    entries: Sequence[SoundScriptEntry],
    output_root: Path,
) -> list[str]:
    """Render human-readable synthesis-plan lines for dry runs."""
    lines = [f"Synthesis plan: {len(entries)} voice file(s)"]
    for index, entry in enumerate(entries, start=1):
        output_path = resolve_output_path(output_root, entry.path)
        lines.append(
            f"{index:02d}/{len(entries):02d} category={entry.category} "
            f"target={output_path} text={entry.text}"
        )
    return lines


def _validate_runtime_paths(model_dir: Path, voice_json: Path) -> None:
    if not model_dir.is_dir():
        msg = f"Supertonic model dir not found: {model_dir}"
        raise FileNotFoundError(msg)
    if not voice_json.is_file():
        msg = f"Supertonic voice JSON not found: {voice_json}"
        raise FileNotFoundError(msg)


def _build_engine(model_dir: Path, voice_json: Path) -> TtsEngine:
    engine = SupertonicEngine(model_dir=str(model_dir), voice_style=str(voice_json))
    return cast(TtsEngine, engine)


def write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    """Write mono float samples to a 16-bit PCM WAV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    flattened = np.asarray(samples, dtype=np.float32).reshape(-1)
    clipped = np.clip(flattened, -1.0, 1.0)
    pcm16 = np.round(clipped * INT16_MAX).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())


def generate_voice_files(
    *,
    entries: Sequence[SoundScriptEntry],
    output_root: Path,
    model_dir: Path,
    voice_json: Path,
    expected_sample_rate: int,
    total_steps: int = 7,
) -> int:
    """Generate voice WAV files for the selected manifest entries."""
    _validate_runtime_paths(model_dir, voice_json)
    engine = _build_engine(model_dir, voice_json)
    try:
        engine.load()
    except ImportError as exc:
        msg = "supertonic package is not installed; run this script on the Jetson runtime"
        raise RuntimeError(msg) from exc

    for index, entry in enumerate(entries, start=1):
        output_path = resolve_output_path(output_root, entry.path)
        LOGGER.info(
            "Synthesizing %d/%d category=%s target=%s",
            index,
            len(entries),
            entry.category,
            output_path,
        )
        audio, sample_rate = engine.synthesize(
            entry.text,
            language=entry.language,
            total_steps=total_steps,
        )
        if sample_rate != expected_sample_rate:
            msg = (
                f"sample rate mismatch for {entry.path}: "
                f"expected {expected_sample_rate}, got {sample_rate}"
            )
            raise RuntimeError(msg)
        write_wav(output_path, audio, sample_rate)
    return len(entries)


def configure_logging() -> None:
    """Configure CLI logging to stdout for smoke-test capture."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the wake-audio generator CLI."""
    configure_logging()
    args = build_parser().parse_args(argv)

    try:
        manifest = load_scripts_manifest(Path(args.scripts_json))
        entries = select_entries(manifest, cast(str | None, args.category_filter))
        for line in render_synthesis_plan(entries, Path(args.output_root)):
            LOGGER.info(line)
        if bool(args.dry_run):
            LOGGER.info("Dry run complete; no Supertonic synthesis invoked")
            return 0

        generated_count = generate_voice_files(
            entries=entries,
            output_root=Path(args.output_root),
            model_dir=Path(args.model_dir),
            voice_json=Path(args.voice_json),
            expected_sample_rate=manifest.sample_rate,
            total_steps=int(args.total_steps),
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        LOGGER.error("Wake audio generation failed: %s", exc)
        return 1

    LOGGER.info("Generated %d voice WAV file(s)", generated_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
