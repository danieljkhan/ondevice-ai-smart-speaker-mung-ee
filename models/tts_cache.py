"""Validated read-only lookup for pre-rendered TTS WAV cache entries."""

from __future__ import annotations

import hashlib
import json
import os
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from models import tts_runner
from models.tts_runner import normalize_tts_text

CACHE_DIR_ENV = "MUNGI_TTS_CACHE_DIR"
DEFAULT_CACHE_DIR = Path("/var/lib/mungi/tts_cache")
META_FILENAME = "cache_meta.json"
MANIFEST_FILENAME = "manifest.json"
SCHEMA_VERSION = 1
ENGINE_ID = "supertonic"
TTS_SPEED = 0.95
CACHE_TOTAL_STEPS = 30
CACHE_SAMPLE_RATE = 16_000
CACHE_AUDIO_FORMAT = "pcm_s16le"
_CACHE_WAV_CHANNELS = 1
_CACHE_WAV_SAMPLE_WIDTH = 2
_HASH_CHUNK_SIZE = 1024 * 1024
_KEY_SEPARATOR = "\x1f"
_DEFAULT_TTS_MODEL_DIR = Path("/opt/mungi/ai_models/supertonic-2")


@dataclass(frozen=True)
class _ManifestEntry:
    wav: str
    sha256: str
    bytes: int
    lang: str


@dataclass(frozen=True)
class _CacheState:
    cache_dir: Path
    enabled: bool
    manifest: dict[str, _ManifestEntry]


_cache_state: _CacheState | None = None
_runtime_identity: dict[str, Any] | None = None


def compute_key(text: str, lang: str) -> str:
    """Return the stable cache key for normalized TTS text and language."""
    normalized_lang = _normalize_lang(lang)
    normalized_text = normalize_tts_text(text)
    payload = f"{normalized_lang}{_KEY_SEPARATOR}{normalized_text}".encode()
    return hashlib.sha256(payload).hexdigest()


def lookup(text: str, lang: str) -> Path | None:
    """Return a validated cached WAV path for ``text`` and ``lang``, or ``None``."""
    try:
        state = _load_cache_state()
        if not state.enabled:
            return None
        normalized_lang = _normalize_lang(lang)
        key = compute_key(text, normalized_lang)
        entry = state.manifest.get(key)
        if entry is None or entry.lang != normalized_lang:
            return None
        wav_path = state.cache_dir / f"{key}.wav"
        if entry.wav != wav_path.name:
            return None
        if not _is_valid_wav_hit(wav_path, entry):
            return None
        return wav_path
    except Exception:
        return None


def _reset_for_tests() -> None:
    global _cache_state, _runtime_identity
    _cache_state = None
    _runtime_identity = None


def _load_cache_state() -> _CacheState:
    global _cache_state
    if _cache_state is not None:
        return _cache_state

    cache_dir = _cache_dir()
    meta_path = cache_dir / META_FILENAME
    if not meta_path.is_file():
        _cache_state = _CacheState(cache_dir=cache_dir, enabled=False, manifest={})
        return _cache_state

    try:
        meta = _read_json_object(meta_path)
        enabled = _identity_matches(meta)
        manifest = _read_manifest(cache_dir / MANIFEST_FILENAME) if enabled else {}
    except Exception:
        enabled = False
        manifest = {}

    _cache_state = _CacheState(cache_dir=cache_dir, enabled=enabled, manifest=manifest)
    return _cache_state


def _identity_matches(meta: dict[str, Any]) -> bool:
    expected = _current_runtime_identity()
    if not _has_hash_backed_identity(expected):
        return False
    return all(meta.get(key) == value for key, value in expected.items())


def _has_hash_backed_identity(identity: dict[str, Any]) -> bool:
    return (
        _is_sha256_hex(identity.get("model_id"))
        and _is_sha256_hex(identity.get("voice_id_ko"))
        and _is_sha256_hex(identity.get("voice_id_en"))
    )


def _is_sha256_hex(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(char in "0123456789abcdefABCDEF" for char in value)


def _current_runtime_identity() -> dict[str, Any]:
    global _runtime_identity
    if _runtime_identity is None:
        _runtime_identity = _build_runtime_identity()
    return dict(_runtime_identity)


def _build_runtime_identity(
    *,
    model_dir: Path | None = None,
    voice_style_ko: str | None = None,
    voice_style_en: str | None = None,
) -> dict[str, Any]:
    engine = _active_supertonic_engine()
    resolved_model_dir = model_dir or _resolve_model_dir(engine)
    ko_selector = (
        voice_style_ko if voice_style_ko is not None else _resolve_voice_selector(engine, "ko")
    )
    en_selector = (
        voice_style_en if voice_style_en is not None else _resolve_voice_selector(engine, "en")
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "engine_id": _resolve_engine_id(engine),
        "model_id": _hash_directory(resolved_model_dir),
        "voice_id_ko": _voice_id(ko_selector),
        "voice_id_en": _voice_id(en_selector),
        "speed": TTS_SPEED,
        "total_steps": CACHE_TOTAL_STEPS,
        "sample_rate": CACHE_SAMPLE_RATE,
        "audio_format": CACHE_AUDIO_FORMAT,
    }


def _active_supertonic_engine() -> Any | None:
    lock = getattr(tts_runner, "_ACTIVE_SUPERTONIC_ENGINE_LOCK", None)
    if lock is None:
        return getattr(tts_runner, "_ACTIVE_SUPERTONIC_ENGINE", None)
    with lock:
        return getattr(tts_runner, "_ACTIVE_SUPERTONIC_ENGINE", None)


def _resolve_engine_id(engine: Any | None) -> str:
    if engine is None:
        return ENGINE_ID
    engine_name = getattr(engine, "engine_name", None)
    if not callable(engine_name):
        return ENGINE_ID
    value = engine_name()
    return value if isinstance(value, str) and value else ENGINE_ID


def _resolve_model_dir(engine: Any | None) -> Path:
    if engine is not None:
        model_dir = getattr(engine, "_model_dir", "")
        if isinstance(model_dir, str) and model_dir.strip():
            return Path(model_dir)

    env_tts_model_dir = os.getenv("MUNGI_TTS_MODEL_DIR", "").strip()
    if env_tts_model_dir:
        return Path(env_tts_model_dir)

    env_model_root = os.getenv("MUNGI_MODEL_ROOT", "").strip()
    if env_model_root:
        return Path(env_model_root) / "supertonic-2"

    env_model_dir = os.getenv("MUNGI_MODEL_DIR", "").strip()
    if env_model_dir:
        return Path(env_model_dir) / "supertonic-2"

    return _DEFAULT_TTS_MODEL_DIR


def _resolve_voice_selector(engine: Any | None, lang: str) -> str:
    if engine is not None:
        per_language_attr = f"_voice_style_{lang}_name"
        selector = getattr(engine, per_language_attr, None)
        if isinstance(selector, str) and selector.strip():
            return selector
        legacy_selector = getattr(engine, "_voice_style_name", None)
        if isinstance(legacy_selector, str) and legacy_selector.strip():
            return legacy_selector

    env_specific = os.getenv(f"MUNGI_TTS_VOICE_STYLE_{lang.upper()}", "").strip()
    if env_specific:
        return env_specific
    env_legacy = os.getenv("MUNGI_TTS_VOICE_STYLE", "").strip()
    if env_legacy:
        return env_legacy
    return "F2"


def _voice_id(selector: str) -> str:
    if not selector:
        return ""
    path = Path(selector).expanduser()
    if _is_voice_json_selector(selector) and path.is_file():
        return _hash_file(path)
    if _is_voice_json_selector(selector):
        return f"missing:{selector}"
    return f"preset:{selector}"


def _is_voice_json_selector(selector: str) -> bool:
    return selector.endswith(".json") or "/" in selector or "\\" in selector


def _cache_dir() -> Path:
    override = os.getenv(CACHE_DIR_ENV, "").strip()
    return Path(override) if override else DEFAULT_CACHE_DIR


def _normalize_lang(lang: str) -> str:
    return (lang or "").strip().lower()


def _read_json_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        msg = f"{path} must contain a JSON object"
        raise ValueError(msg)
    return payload


def _read_manifest(path: Path) -> dict[str, _ManifestEntry]:
    payload = _read_json_object(path)
    manifest: dict[str, _ManifestEntry] = {}
    for key, raw in payload.items():
        if not isinstance(key, str) or not isinstance(raw, dict):
            continue
        entry = _parse_manifest_entry(raw)
        if entry is not None:
            manifest[key] = entry
    return manifest


def _parse_manifest_entry(raw: dict[str, Any]) -> _ManifestEntry | None:
    wav = raw.get("wav")
    checksum = raw.get("sha256")
    byte_count = raw.get("bytes")
    lang = raw.get("lang")
    if not isinstance(wav, str):
        return None
    if not isinstance(checksum, str):
        return None
    if isinstance(byte_count, bool) or not isinstance(byte_count, int):
        return None
    if byte_count <= 0:
        return None
    if not isinstance(lang, str):
        return None
    return _ManifestEntry(
        wav=wav,
        sha256=checksum,
        bytes=byte_count,
        lang=_normalize_lang(lang),
    )


def _is_valid_wav_hit(path: Path, entry: _ManifestEntry) -> bool:
    try:
        if not path.is_file():
            return False
        if path.stat().st_size != entry.bytes:
            return False
        if not _has_expected_wav_header(path):
            return False
        return _hash_file(path) == entry.sha256
    except (OSError, EOFError, wave.Error):
        return False


def _has_expected_wav_header(path: Path) -> bool:
    with wave.open(str(path), "rb") as handle:
        return (
            handle.getnchannels() == _CACHE_WAV_CHANNELS
            and handle.getsampwidth() == _CACHE_WAV_SAMPLE_WIDTH
            and handle.getframerate() == CACHE_SAMPLE_RATE
            and handle.getnframes() > 0
        )


def _hash_directory(path: Path) -> str:
    if path.is_file():
        return _hash_file(path)
    if not path.is_dir():
        return f"missing:{path}"

    digest = hashlib.sha256()
    for child in sorted(path.rglob("*"), key=lambda item: item.relative_to(path).as_posix()):
        if not child.is_file():
            continue
        relative = child.relative_to(path).as_posix()
        stat = child.stat()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["compute_key", "lookup"]
