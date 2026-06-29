"""Synthesize Typecast E2E pilot voice inputs on Windows and build batch WAVs.

This Windows-only utility reads the curated pilot JSON, synthesizes per-message
WAV files through the Typecast HTTP API, concatenates them into per-language
batch WAVs, and emits a manifest for replay/manual evaluation.

Usage:
    python scripts/synth_e2e_voices_typecast.py \
        --pilot-json Dev_Plan/e2e_voice_pilot_scripts.json \
        --out-dir assets/e2e_voice_inputs \
        [--lang ko|en|both] \
        [--skip-existing] \
        [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
import urllib.error
import urllib.request
import wave
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np

logger = logging.getLogger("mungi.scripts.synth_e2e_voices_typecast")

ENV_TYPECAST_API_KEY = "TYPECAST_API_KEY"
SUCCESS_EXIT = 0
CONFIG_EXIT = 2
API_ERROR_EXIT = 1
TYPECAST_URL = "https://api.typecast.ai/v1/text-to-speech"
EXPECTED_SAMPLE_RATE = 44100
EXPECTED_CHANNELS = 1
EXPECTED_SAMPLE_WIDTH = 2
HTTP_TIMEOUT_S = 60.0
MAX_ATTEMPTS = 6
RETRY_DELAYS_S = (2, 4, 8, 16, 32)
SILENCE_PREFIX_S = 2.0
SILENCE_PRE_S = 0.2
SILENCE_POST_S = 0.2
SILENCE_GAP_S = 8.0


class ConfigError(Exception):
    """Raised when CLI arguments or input files are invalid."""


class ApiRequestError(Exception):
    """Raised when the Typecast API cannot be reached successfully."""


@dataclass(frozen=True)
class VoiceConfig:
    """Voice synthesis configuration for one language."""

    voice_id: str
    voice_name: str
    model: str
    emotion_preset: str
    emotion_intensity: float
    audio_tempo: float
    audio_pitch: int


@dataclass(frozen=True)
class MessageTask:
    """One message to synthesize from the pilot JSON."""

    lang: str
    round_id: str
    msg_idx: int
    transcript: str
    output_path: Path


@dataclass
class MessageArtifact:
    """Per-message artifact metadata recorded into the manifest."""

    file: str
    transcript: str
    lang: str
    round_id: str
    msg_idx: int
    duration_s: float
    sha256: str
    batch_offset_s: float = 0.0


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Synthesize Typecast E2E pilot WAVs and batch inputs.",
    )
    parser.add_argument(
        "--pilot-json",
        type=Path,
        required=True,
        help="Relative path to the curated pilot JSON file.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Relative output directory for per-message WAVs, batches, and manifest.",
    )
    parser.add_argument(
        "--lang",
        choices=("ko", "en", "both"),
        default="both",
        help="Language subset to synthesize.",
    )
    parser.add_argument(
        "--skip-existing",
        dest="skip_existing",
        action="store_true",
        default=True,
        help="Skip verified existing WAV files when a matching manifest entry exists.",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Force re-synthesis even when verified outputs already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and log the planned work without API calls or file writes.",
    )
    return parser


def main() -> int:
    """Run the Typecast synthesis pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args()

    try:
        _validate_relative_path(args.pilot_json, "--pilot-json")
        _validate_relative_path(args.out_dir, "--out-dir")
        pilot_payload = _load_pilot_json(args.pilot_json)
        selected_langs = _selected_languages(args.lang)
        api_key = _load_api_key(dry_run=args.dry_run)
        manifest_path = args.out_dir / "manifest.json"
        previous_manifest = _load_previous_manifest(manifest_path)
        pilot_sha256 = _sha256_file(args.pilot_json)

        if args.dry_run:
            _log_dry_run_plan(
                pilot_payload=pilot_payload,
                out_dir=args.out_dir,
                selected_langs=selected_langs,
                skip_existing=args.skip_existing,
                previous_manifest=previous_manifest,
            )
            return SUCCESS_EXIT

        args.out_dir.mkdir(parents=True, exist_ok=True)

        lang_manifest_payloads: dict[str, dict[str, Any]] = {}
        for lang in selected_langs:
            voice_config = _load_voice_config(pilot_payload, lang)
            tasks = _build_message_tasks(pilot_payload, lang, args.out_dir)
            artifacts = _synthesize_language(
                tasks=tasks,
                voice_config=voice_config,
                lang=lang,
                api_key=api_key,
                previous_manifest=previous_manifest,
                skip_existing=args.skip_existing,
                out_dir=args.out_dir,
            )
            batch_path, total_duration_s = _build_batch_wav(
                out_dir=args.out_dir,
                lang=lang,
                artifacts=artifacts,
            )
            lang_manifest_payloads[lang] = _build_language_manifest(
                artifacts=artifacts,
                batch_path=batch_path,
                voice_config=voice_config,
                total_duration_s=total_duration_s,
                out_dir=args.out_dir,
            )

        manifest_payload = _build_manifest(
            pilot_json_path=args.pilot_json,
            pilot_json_sha256=pilot_sha256,
            selected_langs=selected_langs,
            lang_manifest_payloads=lang_manifest_payloads,
        )
        manifest_path.write_text(
            json.dumps(manifest_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        logger.info("Wrote manifest: %s", manifest_path)
        return SUCCESS_EXIT
    except ConfigError as exc:
        logger.error("%s", exc)
        return CONFIG_EXIT
    except ApiRequestError as exc:
        logger.error("%s", exc)
        return API_ERROR_EXIT


def _validate_relative_path(path: Path, argument_name: str) -> None:
    """Reject absolute paths per repository policy."""
    if path.is_absolute():
        raise ConfigError(f"{argument_name} must be a relative path: {path}")


def _selected_languages(lang_arg: str) -> tuple[str, ...]:
    """Expand the CLI language selection into concrete language codes."""
    if lang_arg == "both":
        return ("ko", "en")
    return (lang_arg,)


def _load_pilot_json(path: Path) -> dict[str, Any]:
    """Load and validate the pilot JSON payload."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Pilot JSON file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Pilot JSON is not valid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigError(f"Pilot JSON root must be an object: {path}")

    _require_key_path(payload, ("typecast_config", "ko", "voice_id"))
    _require_key_path(payload, ("typecast_config", "en", "voice_id"))
    ko_rounds = _require_key_path(payload, ("rounds", "ko"))
    en_rounds = _require_key_path(payload, ("rounds", "en"))
    ko_count = _count_messages("ko", ko_rounds)
    en_count = _count_messages("en", en_rounds)
    if ko_count != 30 or en_count != 30:
        raise ConfigError(
            "Pilot JSON must contain exactly 30 KO messages and 30 EN messages; "
            f"found ko={ko_count}, en={en_count}"
        )
    return payload


def _require_key_path(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    """Resolve a nested key path or raise a configuration error."""
    current: Any = payload
    current_path: list[str] = []
    for key in path:
        current_path.append(key)
        if not isinstance(current, dict) or key not in current:
            joined = ".".join(current_path)
            raise ConfigError(f"Missing required key: {joined}")
        current = current[key]
    return current


def _count_messages(lang: str, rounds_value: Any) -> int:
    """Count messages for one language while validating the round structure."""
    if not isinstance(rounds_value, list):
        raise ConfigError(f"rounds.{lang} must be a list")
    total = 0
    for index, round_value in enumerate(rounds_value):
        if not isinstance(round_value, dict):
            raise ConfigError(f"rounds.{lang}[{index}] must be an object")
        messages = round_value.get("messages")
        if not isinstance(messages, list):
            raise ConfigError(f"rounds.{lang}[{index}].messages must be a list")
        if len(messages) != 3:
            raise ConfigError(f"rounds.{lang}[{index}].messages must contain exactly 3 items")
        for msg_index, message in enumerate(messages):
            if not isinstance(message, str) or not message.strip():
                raise ConfigError(
                    f"rounds.{lang}[{index}].messages[{msg_index}] must be a non-empty string"
                )
        total += len(messages)
    return total


def _load_api_key(*, dry_run: bool) -> str:
    """Read the Typecast API key from the environment."""
    api_key = os.getenv(ENV_TYPECAST_API_KEY, "").strip()
    if not api_key and not dry_run:
        raise ConfigError(f"{ENV_TYPECAST_API_KEY} environment variable is required")
    if api_key:
        logger.debug(
            "Loaded %s from environment; request logs use X-API-KEY: ***",
            ENV_TYPECAST_API_KEY,
        )
    return api_key


def _load_previous_manifest(path: Path) -> dict[str, Any] | None:
    """Load a previous manifest if present and valid."""
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring unreadable existing manifest %s: %s", path, exc)
        return None
    if not isinstance(payload, dict):
        logger.warning("Ignoring invalid manifest root in %s", path)
        return None
    return payload


def _log_dry_run_plan(
    *,
    pilot_payload: dict[str, Any],
    out_dir: Path,
    selected_langs: tuple[str, ...],
    skip_existing: bool,
    previous_manifest: dict[str, Any] | None,
) -> None:
    """Log the planned work without touching the network or filesystem."""
    logger.info("Dry-run enabled; no API calls and no file writes will occur.")
    logger.info("Output directory: %s", out_dir)
    logger.info("Skip existing: %s", skip_existing)
    logger.info("Previous manifest detected: %s", previous_manifest is not None)
    for lang in selected_langs:
        rounds = cast(list[dict[str, Any]], _require_key_path(pilot_payload, ("rounds", lang)))
        total_messages = sum(len(cast(list[str], round_item["messages"])) for round_item in rounds)
        logger.info("Language %s: %d rounds, %d messages", lang, len(rounds), total_messages)


def _load_voice_config(pilot_payload: dict[str, Any], lang: str) -> VoiceConfig:
    """Extract the voice configuration for one language."""
    config = _require_key_path(pilot_payload, ("typecast_config", lang))
    if not isinstance(config, dict):
        raise ConfigError(f"typecast_config.{lang} must be an object")
    try:
        return VoiceConfig(
            voice_id=str(config["voice_id"]),
            voice_name=str(config["voice_name"]),
            model=str(config["model"]),
            emotion_preset=str(config["emotion_preset"]),
            emotion_intensity=float(config["emotion_intensity"]),
            audio_tempo=float(config["audio_tempo"]),
            audio_pitch=int(config["audio_pitch"]),
        )
    except KeyError as exc:
        raise ConfigError(f"Missing voice config key for {lang}: {exc.args[0]}") from exc
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Invalid voice config value for {lang}: {exc}") from exc


def _build_message_tasks(
    pilot_payload: dict[str, Any],
    lang: str,
    out_dir: Path,
) -> list[MessageTask]:
    """Create message tasks in round and message order."""
    rounds = _require_key_path(pilot_payload, ("rounds", lang))
    if not isinstance(rounds, list):
        raise ConfigError(f"rounds.{lang} must be a list")
    tasks: list[MessageTask] = []
    for round_item in rounds:
        if not isinstance(round_item, dict):
            raise ConfigError(f"rounds.{lang} entries must be objects")
        round_id = round_item.get("round_id")
        messages = round_item.get("messages")
        if not isinstance(round_id, str) or not isinstance(messages, list):
            raise ConfigError(f"Invalid round entry in rounds.{lang}")
        for msg_idx, transcript in enumerate(messages, start=1):
            if not isinstance(transcript, str):
                raise ConfigError(f"Invalid transcript type in {round_id} message {msg_idx}")
            output_path = out_dir / lang / f"{round_id}_m{msg_idx}.wav"
            tasks.append(
                MessageTask(
                    lang=lang,
                    round_id=round_id,
                    msg_idx=msg_idx,
                    transcript=transcript,
                    output_path=output_path,
                )
            )
    return tasks


def _synthesize_language(
    *,
    tasks: list[MessageTask],
    voice_config: VoiceConfig,
    lang: str,
    api_key: str,
    previous_manifest: dict[str, Any] | None,
    skip_existing: bool,
    out_dir: Path,
) -> list[MessageArtifact]:
    """Synthesize or reuse per-message WAV files for one language."""
    artifacts: list[MessageArtifact] = []
    previous_lookup = _previous_manifest_lookup(previous_manifest, lang)
    language_dir = out_dir / lang
    language_dir.mkdir(parents=True, exist_ok=True)

    for task in tasks:
        started_at = time.perf_counter()
        if skip_existing:
            reused = _reuse_existing_artifact(task, out_dir, previous_lookup)
            if reused is not None:
                elapsed_s = time.perf_counter() - started_at
                logger.info(
                    "Reused %s msg=%d from existing output in %.2fs",
                    task.round_id,
                    task.msg_idx,
                    elapsed_s,
                )
                artifacts.append(reused)
                continue

        wav_bytes = _request_tts_bytes(
            voice_config=voice_config,
            lang=lang,
            transcript=task.transcript,
            api_key=api_key,
            round_id=task.round_id,
            msg_idx=task.msg_idx,
        )
        task.output_path.write_bytes(wav_bytes)
        sha256 = _sha256_file(task.output_path)
        duration_s = _read_wav_duration(task.output_path)
        elapsed_s = time.perf_counter() - started_at
        logger.info(
            "Synthesized %s msg=%d in %.2fs",
            task.round_id,
            task.msg_idx,
            elapsed_s,
        )
        artifacts.append(
            MessageArtifact(
                file=_relative_posix(task.output_path, out_dir),
                transcript=task.transcript,
                lang=task.lang,
                round_id=task.round_id,
                msg_idx=task.msg_idx,
                duration_s=duration_s,
                sha256=sha256,
            )
        )
    return artifacts


def _previous_manifest_lookup(
    previous_manifest: dict[str, Any] | None,
    lang: str,
) -> dict[tuple[str, int], dict[str, Any]]:
    """Index previous manifest message entries by round/message id."""
    if previous_manifest is None:
        return {}
    lang_payload = previous_manifest.get(lang)
    if not isinstance(lang_payload, dict):
        return {}
    messages = lang_payload.get("messages")
    if not isinstance(messages, list):
        return {}
    lookup: dict[tuple[str, int], dict[str, Any]] = {}
    for item in messages:
        if not isinstance(item, dict):
            continue
        round_id = item.get("round_id")
        msg_idx = item.get("msg_idx")
        if isinstance(round_id, str) and isinstance(msg_idx, int):
            lookup[(round_id, msg_idx)] = item
    return lookup


def _reuse_existing_artifact(
    task: MessageTask,
    out_dir: Path,
    previous_lookup: dict[tuple[str, int], dict[str, Any]],
) -> MessageArtifact | None:
    """Reuse an existing WAV only when it matches the previous manifest SHA-256."""
    if not task.output_path.exists():
        return None
    previous_entry = previous_lookup.get((task.round_id, task.msg_idx))
    if previous_entry is None:
        return None
    previous_sha = previous_entry.get("sha256")
    if not isinstance(previous_sha, str):
        return None
    current_sha = _sha256_file(task.output_path)
    if current_sha != previous_sha:
        logger.warning(
            "Existing file hash mismatch for %s msg=%d; regenerating",
            task.round_id,
            task.msg_idx,
        )
        return None
    duration_s = _read_wav_duration(task.output_path)
    return MessageArtifact(
        file=_relative_posix(task.output_path, out_dir),
        transcript=task.transcript,
        lang=task.lang,
        round_id=task.round_id,
        msg_idx=task.msg_idx,
        duration_s=duration_s,
        sha256=current_sha,
    )


def _request_tts_bytes(
    *,
    voice_config: VoiceConfig,
    lang: str,
    transcript: str,
    api_key: str,
    round_id: str,
    msg_idx: int,
) -> bytes:
    """Call the Typecast API with retries for transient failures."""
    language_code = "kor" if lang == "ko" else "eng"
    payload = {
        "voice_id": voice_config.voice_id,
        "text": transcript,
        "model": voice_config.model,
        "language": language_code,
        "prompt": {
            "emotion_type": "preset",
            "emotion_preset": voice_config.emotion_preset,
            "emotion_intensity": voice_config.emotion_intensity,
        },
        "output": {
            "audio_format": "wav",
            "audio_tempo": voice_config.audio_tempo,
            "audio_pitch": voice_config.audio_pitch,
            "volume": 100,
        },
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        TYPECAST_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "X-API-KEY": api_key,
        },
        method="POST",
    )

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_S) as response:
                return cast(bytes, response.read())
        except urllib.error.HTTPError as exc:
            response_body = exc.read().decode("utf-8", errors="replace")
            if 400 <= exc.code < 500:
                raise ApiRequestError(
                    f"Typecast 4xx for {round_id} msg={msg_idx}: "
                    f"status={exc.code}, body={response_body}"
                ) from exc
            if attempt == MAX_ATTEMPTS:
                raise ApiRequestError(
                    f"Typecast 5xx for {round_id} msg={msg_idx} after {attempt} attempts: "
                    f"status={exc.code}, body={response_body}"
                ) from exc
            _log_retry(
                reason=f"HTTP {exc.code}",
                response_body=response_body,
                round_id=round_id,
                msg_idx=msg_idx,
                attempt=attempt,
            )
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if attempt == MAX_ATTEMPTS:
                raise ApiRequestError(
                    f"Typecast request failed for {round_id} msg={msg_idx} after {attempt} attempts: "
                    f"{exc}"
                ) from exc
            _log_retry(
                reason=str(exc),
                response_body=None,
                round_id=round_id,
                msg_idx=msg_idx,
                attempt=attempt,
            )

        time.sleep(RETRY_DELAYS_S[attempt - 1])

    raise ApiRequestError(f"Unexpected retry exhaustion for {round_id} msg={msg_idx}")


def _log_retry(
    *,
    reason: str,
    response_body: str | None,
    round_id: str,
    msg_idx: int,
    attempt: int,
) -> None:
    """Log one transient retry with exponential backoff information."""
    delay_s = RETRY_DELAYS_S[attempt - 1]
    if response_body is None:
        logger.warning(
            "Retrying %s msg=%d after attempt %d/%d in %ds due to %s",
            round_id,
            msg_idx,
            attempt,
            MAX_ATTEMPTS,
            delay_s,
            reason,
        )
        return
    logger.warning(
        "Retrying %s msg=%d after attempt %d/%d in %ds due to %s; body=%s",
        round_id,
        msg_idx,
        attempt,
        MAX_ATTEMPTS,
        delay_s,
        reason,
        response_body,
    )


def _sha256_file(path: Path) -> str:
    """Compute the SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_wav_duration(path: Path) -> float:
    """Read WAV metadata and validate the required audio format."""
    try:
        with wave.open(str(path), "rb") as handle:
            framerate = handle.getframerate()
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            frames = handle.getnframes()
    except wave.Error as exc:
        raise ApiRequestError(f"Invalid WAV output at {path}: {exc}") from exc

    if framerate != EXPECTED_SAMPLE_RATE:
        raise ApiRequestError(
            f"Unexpected sample rate for {path}: {framerate} != {EXPECTED_SAMPLE_RATE}"
        )
    if channels != EXPECTED_CHANNELS:
        raise ApiRequestError(f"Unexpected channels for {path}: {channels} != {EXPECTED_CHANNELS}")
    if sample_width != EXPECTED_SAMPLE_WIDTH:
        raise ApiRequestError(
            f"Unexpected sample width for {path}: {sample_width} != {EXPECTED_SAMPLE_WIDTH}"
        )
    return frames / framerate


def _build_batch_wav(
    *,
    out_dir: Path,
    lang: str,
    artifacts: list[MessageArtifact],
) -> tuple[Path, float]:
    """Concatenate per-message WAVs into one stereo PCM16 batch file."""
    sf_module = _import_soundfile()
    prefix_frames = _seconds_to_frames(SILENCE_PREFIX_S)
    pre_frames = _seconds_to_frames(SILENCE_PRE_S)
    post_frames = _seconds_to_frames(SILENCE_POST_S)
    gap_frames = _seconds_to_frames(SILENCE_GAP_S)

    segments: list[np.ndarray[Any, np.dtype[np.float32]]] = [
        np.zeros(prefix_frames, dtype=np.float32)
    ]
    cursor_frames = prefix_frames

    for index, artifact in enumerate(artifacts):
        mono, sample_rate = sf_module.read(out_dir / Path(artifact.file), dtype="float32")
        if sample_rate != EXPECTED_SAMPLE_RATE:
            raise ApiRequestError(
                f"Unexpected sample rate for {artifact.file}: {sample_rate} != {EXPECTED_SAMPLE_RATE}"
            )
        mono_array = _ensure_mono_float32(np.asarray(mono), artifact.file)
        artifact.batch_offset_s = (cursor_frames + pre_frames) / EXPECTED_SAMPLE_RATE
        segments.append(np.zeros(pre_frames, dtype=np.float32))
        segments.append(mono_array)
        segments.append(np.zeros(post_frames, dtype=np.float32))
        cursor_frames += pre_frames + len(mono_array) + post_frames
        if index != len(artifacts) - 1:
            segments.append(np.zeros(gap_frames, dtype=np.float32))
            cursor_frames += gap_frames

    mono_concat = np.concatenate(segments, axis=0)
    stereo = np.stack([mono_concat, mono_concat], axis=-1)
    stereo = np.clip(stereo, -1.0, 1.0)
    pcm16 = (stereo * 32767.0).astype(np.int16)

    batch_path = out_dir / f"pilot_{lang}_batch.wav"
    sf_module.write(batch_path, pcm16, samplerate=EXPECTED_SAMPLE_RATE, subtype="PCM_16")
    info = sf_module.info(batch_path)
    if info.samplerate != EXPECTED_SAMPLE_RATE:
        raise ApiRequestError(
            f"Unexpected batch sample rate for {batch_path}: {info.samplerate} != {EXPECTED_SAMPLE_RATE}"
        )
    if info.channels != 2:
        raise ApiRequestError(f"Unexpected batch channels for {batch_path}: {info.channels} != 2")
    total_duration_s = info.frames / info.samplerate
    logger.info("Wrote batch WAV for %s: %s (%.2fs)", lang, batch_path, total_duration_s)
    return batch_path, total_duration_s


def _seconds_to_frames(seconds: float) -> int:
    """Convert a fixed-second padding interval into frame counts."""
    return int(round(seconds * EXPECTED_SAMPLE_RATE))


def _import_soundfile() -> Any:
    """Import soundfile lazily so dry-run mode works without the optional runtime dep."""
    try:
        import soundfile  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise ConfigError("soundfile package is required for batch WAV generation") from exc
    return soundfile


def _ensure_mono_float32(
    data: np.ndarray[Any, Any],
    artifact_file: str,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Normalize soundfile output to a one-dimensional float32 mono array."""
    if data.ndim == 1:
        return np.asarray(data, dtype=np.float32)
    if data.ndim == 2 and data.shape[1] == 1:
        return np.asarray(data[:, 0], dtype=np.float32)
    raise ApiRequestError(f"Expected mono audio for {artifact_file}; got shape {data.shape}")


def _build_language_manifest(
    *,
    artifacts: list[MessageArtifact],
    batch_path: Path,
    voice_config: VoiceConfig,
    total_duration_s: float,
    out_dir: Path,
) -> dict[str, Any]:
    """Build the manifest payload for one language."""
    messages: list[dict[str, Any]] = []
    for idx, artifact in enumerate(artifacts):
        messages.append(
            {
                "idx": idx,
                "round_id": artifact.round_id,
                "msg_idx": artifact.msg_idx,
                "text": artifact.transcript,
                "wav_file": artifact.file,
                "sha256": artifact.sha256,
                "duration_s": artifact.duration_s,
                "batch_offset_s": artifact.batch_offset_s,
            }
        )
    return {
        "batch_file": _relative_posix(batch_path, out_dir),
        "voice_id": voice_config.voice_id,
        "voice_name": voice_config.voice_name,
        "model": voice_config.model,
        "total_duration_s": total_duration_s,
        "sample_rate": EXPECTED_SAMPLE_RATE,
        "channels": 2,
        "subtype": "PCM_16",
        "messages": messages,
    }


def _build_manifest(
    *,
    pilot_json_path: Path,
    pilot_json_sha256: str,
    selected_langs: tuple[str, ...],
    lang_manifest_payloads: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build the top-level manifest payload."""
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "generated_at": datetime.now().astimezone().isoformat(),
        "pilot_json_path": pilot_json_path.as_posix(),
        "pilot_json_sha256": pilot_json_sha256,
    }
    for lang in selected_langs:
        manifest[lang] = lang_manifest_payloads[lang]
    return manifest


def _relative_posix(path: Path, root: Path) -> str:
    """Return a root-relative POSIX path string for manifest storage."""
    return path.relative_to(root).as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
