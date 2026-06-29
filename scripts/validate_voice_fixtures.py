from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, cast

EXPECTED_SAMPLE_RATE = 16000
EXPECTED_CHANNELS = 1
LOGGER_NAME = "validate_voice_fixtures"


class SoundFileInfo(Protocol):
    """Minimal `soundfile.info()` result used by this validator."""

    samplerate: int
    channels: int


class SoundFileModule(Protocol):
    """Minimal `soundfile` module API used by this validator."""

    def info(self, file: str) -> SoundFileInfo:
        """Return metadata for an audio file."""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate voice fixture manifest entries against transferred WAV files.",
    )
    parser.add_argument(
        "--fixture-dir",
        required=True,
        type=Path,
        help="Directory containing manifest.json and fixture WAV files.",
    )
    parser.add_argument(
        "--expected-count",
        default=100,
        type=int,
        help="Expected number of manifest entries.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Optional manifest path. Defaults to <fixture-dir>/manifest.json.",
    )
    return parser


def _configure_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
        force=True,
    )
    return logging.getLogger(LOGGER_NAME)


def _log_json(logger: logging.Logger, level: int, payload: Mapping[str, Any]) -> None:
    logger.log(level, json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _load_manifest(path: Path) -> tuple[list[Any] | None, dict[str, Any] | None]:
    try:
        raw_manifest = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, {
            "check": "manifest_exists",
            "path": str(path),
            "error": "manifest file does not exist",
        }
    except UnicodeDecodeError as exc:
        return None, {
            "check": "manifest_utf8",
            "path": str(path),
            "byte_offset": exc.start,
            "error": "manifest file is not valid UTF-8",
        }
    except json.JSONDecodeError as exc:
        return None, {
            "check": "manifest_json",
            "path": str(path),
            "line": exc.lineno,
            "column": exc.colno,
            "error": exc.msg,
        }
    except OSError as exc:
        return None, {
            "check": "manifest_read",
            "path": str(path),
            "error": str(exc),
        }

    if not isinstance(raw_manifest, list):
        return None, {
            "check": "manifest_shape",
            "path": str(path),
            "expected": "JSON list",
            "actual": type(raw_manifest).__name__,
        }

    return raw_manifest, None


def _resolve_wav_path(fixture_dir: Path, wav_path_value: Any) -> tuple[Path | None, str | None]:
    if not isinstance(wav_path_value, str) or not wav_path_value:
        return None, "wav_path must be a non-empty string"

    wav_path = Path(wav_path_value)
    if wav_path.is_absolute():
        return wav_path, None

    candidates = [
        fixture_dir / wav_path,
        fixture_dir.parent / wav_path,
        fixture_dir / wav_path.name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate, None
    return candidates[0], None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_soundfile_module() -> tuple[SoundFileModule | None, dict[str, Any] | None]:
    try:
        module = importlib.import_module("soundfile")
    except (ImportError, OSError) as exc:
        return None, {
            "check": "soundfile_dependency",
            "module": "soundfile",
            "error": str(exc),
        }
    return cast(SoundFileModule, module), None


def _validate_entry(
    entry: Any,
    entry_index: int,
    fixture_dir: Path,
    soundfile_module: SoundFileModule,
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    if not isinstance(entry, dict):
        return [
            {
                "check": "entry_shape",
                "entry_index": entry_index,
                "expected": "JSON object",
                "actual": type(entry).__name__,
            }
        ]

    query_id = entry.get("query_id")
    wav_path, wav_path_error = _resolve_wav_path(fixture_dir, entry.get("wav_path"))
    if wav_path_error is not None or wav_path is None:
        return [
            {
                "check": "wav_path",
                "entry_index": entry_index,
                "query_id": query_id,
                "error": wav_path_error,
            }
        ]

    if not isinstance(entry.get("sha256"), str) or not entry["sha256"]:
        errors.append(
            {
                "check": "sha256_manifest",
                "entry_index": entry_index,
                "query_id": query_id,
                "path": str(wav_path),
                "error": "sha256 must be a non-empty string",
            }
        )
        return errors

    try:
        actual_sha256 = _sha256_file(wav_path)
    except FileNotFoundError:
        errors.append(
            {
                "check": "wav_exists",
                "entry_index": entry_index,
                "query_id": query_id,
                "path": str(wav_path),
                "error": "WAV file does not exist",
            }
        )
        return errors
    except OSError as exc:
        errors.append(
            {
                "check": "wav_read",
                "entry_index": entry_index,
                "query_id": query_id,
                "path": str(wav_path),
                "error": str(exc),
            }
        )
        return errors

    if actual_sha256 != entry["sha256"]:
        errors.append(
            {
                "check": "sha256_match",
                "entry_index": entry_index,
                "query_id": query_id,
                "path": str(wav_path),
                "expected": entry["sha256"],
                "actual": actual_sha256,
            }
        )

    try:
        info = soundfile_module.info(str(wav_path))
    except RuntimeError as exc:
        errors.append(
            {
                "check": "soundfile_info",
                "entry_index": entry_index,
                "query_id": query_id,
                "path": str(wav_path),
                "error": str(exc),
            }
        )
        return errors

    if info.samplerate != EXPECTED_SAMPLE_RATE:
        errors.append(
            {
                "check": "sample_rate",
                "entry_index": entry_index,
                "query_id": query_id,
                "path": str(wav_path),
                "expected": EXPECTED_SAMPLE_RATE,
                "actual": info.samplerate,
            }
        )
    if info.channels != EXPECTED_CHANNELS:
        errors.append(
            {
                "check": "channels",
                "entry_index": entry_index,
                "query_id": query_id,
                "path": str(wav_path),
                "expected": EXPECTED_CHANNELS,
                "actual": info.channels,
            }
        )

    return errors


def validate_voice_fixtures(
    fixture_dir: Path,
    expected_count: int,
    manifest_path: Path | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Validate voice fixtures against a JSON-list manifest."""

    resolved_fixture_dir = fixture_dir.resolve()
    resolved_manifest_path = (manifest_path or fixture_dir / "manifest.json").resolve()

    manifest, manifest_error = _load_manifest(resolved_manifest_path)
    if manifest_error is not None or manifest is None:
        error = manifest_error or {
            "check": "manifest_load",
            "path": str(resolved_manifest_path),
            "error": "manifest could not be loaded",
        }
        return False, {
            "status": "fail",
            "fixture_dir": str(resolved_fixture_dir),
            "manifest": str(resolved_manifest_path),
            "expected_count": expected_count,
            "actual_count": None,
            "validated_count": 0,
            "error_count": 1,
            "errors": [error],
        }

    errors: list[dict[str, Any]] = []
    if len(manifest) != expected_count:
        errors.append(
            {
                "check": "manifest_count",
                "expected": expected_count,
                "actual": len(manifest),
            }
        )

    soundfile_module: SoundFileModule | None = None
    if manifest:
        soundfile_module, soundfile_error = _load_soundfile_module()
        if soundfile_error is not None or soundfile_module is None:
            error = soundfile_error or {
                "check": "soundfile_dependency",
                "module": "soundfile",
                "error": "soundfile could not be imported",
            }
            errors.append(error)
            status = "fail"
            return False, {
                "status": status,
                "fixture_dir": str(resolved_fixture_dir),
                "manifest": str(resolved_manifest_path),
                "expected_count": expected_count,
                "actual_count": len(manifest),
                "validated_count": 0,
                "required_sample_rate": EXPECTED_SAMPLE_RATE,
                "required_channels": EXPECTED_CHANNELS,
                "error_count": len(errors),
                "errors": errors,
            }

    for entry_index, entry in enumerate(manifest, start=1):
        if soundfile_module is None:
            continue
        errors.extend(_validate_entry(entry, entry_index, resolved_fixture_dir, soundfile_module))

    status = "pass" if not errors else "fail"
    return not errors, {
        "status": status,
        "fixture_dir": str(resolved_fixture_dir),
        "manifest": str(resolved_manifest_path),
        "expected_count": expected_count,
        "actual_count": len(manifest),
        "validated_count": len(manifest),
        "required_sample_rate": EXPECTED_SAMPLE_RATE,
        "required_channels": EXPECTED_CHANNELS,
        "error_count": len(errors),
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the voice fixture validator CLI."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    logger = _configure_logging()

    is_valid, summary = validate_voice_fixtures(
        fixture_dir=args.fixture_dir,
        expected_count=args.expected_count,
        manifest_path=args.manifest,
    )
    _log_json(logger, logging.INFO if is_valid else logging.ERROR, summary)
    return 0 if is_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
