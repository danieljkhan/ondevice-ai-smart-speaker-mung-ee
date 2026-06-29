"""Bake the validated offline TTS cache for fixed runtime speech inventories."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.funny_english_mode import FE_KOREAN_SPEECH_LINES, FunnyEnglishModeController
from core.history_mode import (
    CONSENT_PROMPT,
    CONSENT_REPROMPT,
    history_narration_segments,
)
from models import tts_cache

logger = logging.getLogger(__name__)

DEFAULT_OUT_DIR = tts_cache.DEFAULT_CACHE_DIR
DEFAULT_HISTORY_DOCS_DIR = Path("assets/history/docs")
DEFAULT_FUNNY_ENGLISH_MANIFEST = Path("assets/funny_english/manifest.json")
DEFAULT_APPROVED_TEMPLATES_PATH = Path("assets/filters/approved_templates.json")
DEFAULT_CHECKPOINT_EVERY = 10
APPROVED_TEMPLATE_FIXED_TOPIC_IDS = (
    "mungi_self_intro_child",
    "mungi_product_intro_adult",
)


class SynthesisEngine(Protocol):
    """Small protocol required by the cache baker."""

    def synthesize(
        self,
        text: str | None,
        language: str = "ko",
        total_steps: int = 10,
    ) -> tuple[Any, int]:
        """Synthesize ``text`` and return audio samples plus sample rate."""


@dataclass(frozen=True)
class BakeText:
    """One unique text unit to bake."""

    text: str
    lang: str
    key: str
    source: str


@dataclass(frozen=True)
class CollectionInventory:
    """Raw and unique inventory counts for the bake collection."""

    history_segments: int
    history_lead_ins: int
    history_consent: int
    fe_ko: int
    fe_en: int
    approved_template_fixed: int
    total_raw: int
    total_unique: int


@dataclass(frozen=True)
class CollectionResult:
    """Collected bake items plus inventory metadata."""

    items: tuple[BakeText, ...]
    inventory: CollectionInventory


@dataclass(frozen=True)
class BakeSkippedError:
    """One cache bake item skipped because synthesis or cache writing failed."""

    source: str
    error: str


@dataclass(frozen=True)
class BakeSummary:
    """Summary of one bake invocation."""

    considered: int
    rendered: int
    skipped: int
    skipped_existing: int
    skipped_error: int
    error_sources: tuple[str, ...]
    skipped_errors: tuple[BakeSkippedError, ...]
    inventory: CollectionInventory


def collect_bake_texts(repo_root: Path = REPO_ROOT) -> CollectionResult:
    """Collect all runtime-equivalent text units for the cache bake."""
    raw_items: list[BakeText] = []
    raw_items.extend(_collect_history_items(repo_root))
    raw_items.extend(_collect_funny_english_ko_items())
    raw_items.extend(_collect_funny_english_en_items(repo_root))
    raw_items.extend(_collect_approved_template_fixed_items(repo_root))

    unique_items = _dedupe_by_key(raw_items)
    inventory = CollectionInventory(
        history_segments=sum(1 for item in raw_items if item.source.startswith("history:scene:")),
        history_lead_ins=sum(1 for item in raw_items if item.source.startswith("history:lead_in:")),
        history_consent=sum(1 for item in raw_items if item.source.startswith("history:consent:")),
        fe_ko=sum(1 for item in raw_items if item.source.startswith("funny_english:ko:")),
        fe_en=sum(1 for item in raw_items if item.source.startswith("funny_english:en:")),
        approved_template_fixed=sum(
            1 for item in raw_items if item.source.startswith("approved_template:fixed:")
        ),
        total_raw=len(raw_items),
        total_unique=len(unique_items),
    )
    return CollectionResult(items=tuple(unique_items), inventory=inventory)


def bake_cache(
    *,
    out_dir: Path,
    steps: int = tts_cache.CACHE_TOTAL_STEPS,
    limit: int | None = None,
    only: str | None = None,
    device: str = "cuda",
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
    repo_root: Path = REPO_ROOT,
    engine: SynthesisEngine | None = None,
    match: tuple[str, ...] | None = None,
    force: bool = False,
) -> BakeSummary:
    """Bake missing or invalid cache entries into ``out_dir``.

    When ``match`` is given, only items whose text contains at least one of the
    substrings are considered. When ``force`` is True, items that are already a
    valid manifest hit are re-synthesized and overwritten instead of skipped.
    """
    _validate_bake_args(steps=steps, limit=limit, only=only, checkpoint_every=checkpoint_every)
    out_dir.mkdir(parents=True, exist_ok=True)

    collection = collect_bake_texts(repo_root)
    items = _filter_items(collection.items, only=only, limit=limit, match=match)
    _write_json_atomic(out_dir / tts_cache.META_FILENAME, _bake_identity())
    manifest_path = out_dir / tts_cache.MANIFEST_FILENAME
    manifest = _read_manifest_payload(manifest_path)

    owned_engine = engine is None
    active_engine = engine if engine is not None else _load_engine(device)
    rendered = 0
    skipped_existing = 0
    skipped_errors: list[BakeSkippedError] = []
    dirty_since_checkpoint = 0

    _log_event(
        "bake_start",
        out_dir=str(out_dir),
        considered=len(items),
        only=only or "all",
        limit=limit,
        device=device,
        match=list(match) if match else None,
        force=force,
        inventory=asdict(collection.inventory),
    )

    try:
        for index, item in enumerate(items, start=1):
            valid_hit = _is_valid_manifest_hit(out_dir, manifest, item.key)
            if valid_hit and not force:
                skipped_existing += 1
                _log_event(
                    "cache_skip", index=index, key=item.key, lang=item.lang, source=item.source
                )
                continue
            if valid_hit and force:
                _log_event(
                    "cache_force_rebake",
                    index=index,
                    key=item.key,
                    lang=item.lang,
                    source=item.source,
                )

            wav_path = out_dir / f"{item.key}.wav"
            try:
                audio_samples, sample_rate = active_engine.synthesize(
                    item.text,
                    language=item.lang,
                    total_steps=steps,
                )
                _write_cache_wav_atomic(wav_path, audio_samples, sample_rate)
                manifest[item.key] = _manifest_entry(item, wav_path, steps=steps)
            except Exception as exc:
                skipped_error = BakeSkippedError(source=item.source, error=str(exc))
                skipped_errors.append(skipped_error)
                _log_event(
                    "cache_skip_error",
                    index=index,
                    key=item.key,
                    lang=item.lang,
                    source=item.source,
                    error=str(exc),
                )
                continue

            rendered += 1
            dirty_since_checkpoint += 1
            _log_event(
                "cache_rendered",
                index=index,
                key=item.key,
                lang=item.lang,
                source=item.source,
                bytes=wav_path.stat().st_size,
            )

            if dirty_since_checkpoint >= checkpoint_every:
                _write_json_atomic(manifest_path, manifest)
                dirty_since_checkpoint = 0
                _log_event(
                    "manifest_checkpoint",
                    rendered=rendered,
                    skipped_existing=skipped_existing,
                    skipped_error=len(skipped_errors),
                )
    finally:
        if dirty_since_checkpoint:
            _write_json_atomic(manifest_path, manifest)
            _log_event(
                "manifest_checkpoint",
                rendered=rendered,
                skipped_existing=skipped_existing,
                skipped_error=len(skipped_errors),
            )
            dirty_since_checkpoint = 0
        if owned_engine:
            unload = getattr(active_engine, "unload", None)
            if callable(unload):
                unload()

    if dirty_since_checkpoint or not manifest_path.exists():
        _write_json_atomic(manifest_path, manifest)
        _log_event(
            "manifest_checkpoint",
            rendered=rendered,
            skipped_existing=skipped_existing,
            skipped_error=len(skipped_errors),
        )

    skipped_error_items = tuple(skipped_errors)
    summary = BakeSummary(
        considered=len(items),
        rendered=rendered,
        skipped=skipped_existing,
        skipped_existing=skipped_existing,
        skipped_error=len(skipped_error_items),
        error_sources=tuple(item.source for item in skipped_error_items),
        skipped_errors=skipped_error_items,
        inventory=collection.inventory,
    )
    _log_event("bake_done", **asdict(summary))
    return summary


def main(argv: list[str] | None = None) -> int:
    """Run the TTS cache bake CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--steps", type=int, default=tts_cache.CACHE_TOTAL_STEPS)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--only", choices=("ko", "en"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--match",
        action="append",
        metavar="TEXT",
        help=(
            "Restrict baking to items whose text contains this substring. "
            "Repeatable; an item matches if its text contains ANY provided substring."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-bake and overwrite items even if they are already a valid manifest hit.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(message)s",
    )
    match = tuple(args.match) if args.match else None
    bake_cache(
        out_dir=args.out_dir,
        steps=args.steps,
        limit=args.limit,
        only=args.only,
        device=args.device,
        checkpoint_every=args.checkpoint_every,
        repo_root=args.repo_root,
        match=match,
        force=args.force,
    )
    return 0


def _collect_history_items(repo_root: Path) -> list[BakeText]:
    docs_dir = repo_root / DEFAULT_HISTORY_DOCS_DIR
    items: list[BakeText] = []
    for doc_path in sorted(docs_dir.glob("*.json")):
        with doc_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        title = _required_str(payload, "title", doc_path)
        items.append(
            _make_item(
                text=f"지금부터 '{title}' 이야기를 들려줄게.",
                lang="ko",
                source=f"history:lead_in:{doc_path.stem}",
            )
        )
        scenes = payload.get("scenes")
        if not isinstance(scenes, list):
            raise ValueError(f"{doc_path}: scenes must be a list")
        for scene_index, raw_scene in enumerate(scenes):
            if not isinstance(raw_scene, dict):
                raise ValueError(f"{doc_path}: scene {scene_index} must be an object")
            narration = _required_str(raw_scene, "narration", doc_path)
            section_title = raw_scene.get("section_title")
            if section_title is not None and not isinstance(section_title, str):
                raise ValueError(f"{doc_path}: section_title must be text or null")
            seq = raw_scene.get("seq", scene_index)
            for segment_index, segment in enumerate(
                history_narration_segments(narration, section_title)
            ):
                if segment is None:
                    continue
                items.append(
                    _make_item(
                        text=segment,
                        lang="ko",
                        source=f"history:scene:{doc_path.stem}:{seq}:{segment_index}",
                    )
                )
    items.append(_make_item(text=CONSENT_PROMPT, lang="ko", source="history:consent:prompt"))
    items.append(_make_item(text=CONSENT_REPROMPT, lang="ko", source="history:consent:reprompt"))
    return items


def _collect_funny_english_ko_items() -> list[BakeText]:
    return [
        _make_item(text=text, lang="ko", source=f"funny_english:ko:{index}")
        for index, text in enumerate(_funny_english_ko_texts())
    ]


def _collect_funny_english_en_items(repo_root: Path) -> list[BakeText]:
    controller = object.__new__(FunnyEnglishModeController)
    controller._repo_root = repo_root  # type: ignore[attr-defined]
    manifest_path = repo_root / DEFAULT_FUNNY_ENGLISH_MANIFEST
    stages = FunnyEnglishModeController._load_manifest(controller, manifest_path)
    items: list[BakeText] = []
    for stage in stages:
        for card in stage.cards:
            items.append(
                _make_item(
                    text=card.text,
                    lang="en",
                    source=f"funny_english:en:{stage.stage}:{card.card_id}",
                )
            )
    return items


def _collect_approved_template_fixed_items(repo_root: Path) -> list[BakeText]:
    templates_path = repo_root / DEFAULT_APPROVED_TEMPLATES_PATH
    with templates_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{templates_path}: approved templates must be an object")

    items: list[BakeText] = []
    for topic_id in APPROVED_TEMPLATE_FIXED_TOPIC_IDS:
        raw_topic = payload.get(topic_id)
        if not isinstance(raw_topic, dict):
            raise ValueError(f"{templates_path}: missing fixed template {topic_id!r}")
        if raw_topic.get("mode") != "block":
            raise ValueError(f"{templates_path}: fixed template {topic_id!r} must be block mode")
        response = _required_str(raw_topic, "response_ko", templates_path)
        items.append(
            _make_item(
                text=response,
                lang="ko",
                source=f"approved_template:fixed:{topic_id}",
            )
        )
    return items


def _funny_english_ko_texts() -> tuple[str, ...]:
    return FE_KOREAN_SPEECH_LINES


def _make_item(*, text: str, lang: str, source: str) -> BakeText:
    normalized_lang = lang.strip().lower()
    if normalized_lang not in ("ko", "en"):
        raise ValueError(f"unsupported bake language: {lang!r}")
    return BakeText(
        text=text,
        lang=normalized_lang,
        key=tts_cache.compute_key(text, normalized_lang),
        source=source,
    )


def _dedupe_by_key(items: list[BakeText]) -> list[BakeText]:
    seen: set[str] = set()
    unique: list[BakeText] = []
    for item in items:
        if item.key in seen:
            continue
        seen.add(item.key)
        unique.append(item)
    return unique


def _filter_items(
    items: tuple[BakeText, ...],
    *,
    only: str | None,
    limit: int | None,
    match: tuple[str, ...] | None = None,
) -> tuple[BakeText, ...]:
    filtered = tuple(
        item
        for item in items
        if (only is None or item.lang == only) and _matches_any(item.text, match)
    )
    if limit is None:
        return filtered
    return filtered[:limit]


def _matches_any(text: str, match: tuple[str, ...] | None) -> bool:
    """Return True if no substrings are given or ``text`` contains any of them."""
    if not match:
        return True
    return any(substring in text for substring in match)


def _validate_bake_args(
    *,
    steps: int,
    limit: int | None,
    only: str | None,
    checkpoint_every: int,
) -> None:
    if steps != tts_cache.CACHE_TOTAL_STEPS:
        raise ValueError(f"steps must be {tts_cache.CACHE_TOTAL_STEPS} to match tts_cache identity")
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    if only is not None and only not in ("ko", "en"):
        raise ValueError("only must be 'ko', 'en', or None")
    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive")


def _load_engine(device: str) -> SynthesisEngine:
    _configure_device(device)
    from models.tts_runner import SupertonicEngine

    model_dir = tts_cache._resolve_model_dir(None)
    voice_ko = tts_cache._resolve_voice_selector(None, "ko")
    voice_en = tts_cache._resolve_voice_selector(None, "en")
    engine = SupertonicEngine(
        model_dir=str(model_dir),
        voice_style_ko=voice_ko,
        voice_style_en=voice_en,
    )
    engine.load()
    return engine


def _configure_device(device: str) -> None:
    if device == "cuda":
        os.environ["MUNGI_TTS_ONNX_PROVIDER"] = "cuda"
        _set_supertonic_providers(["CUDAExecutionProvider", "CPUExecutionProvider"])
        return
    if device == "cpu":
        os.environ["MUNGI_TTS_ONNX_PROVIDER"] = "cpu"
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
        _set_supertonic_providers(["CPUExecutionProvider"])
        return
    raise ValueError(f"unsupported device: {device!r}")


def _set_supertonic_providers(providers: list[str]) -> None:
    try:
        import supertonic.config as supertonic_config  # type: ignore[import-not-found]
    except ImportError:
        return
    if hasattr(supertonic_config, "DEFAULT_ONNX_PROVIDERS"):
        supertonic_config.DEFAULT_ONNX_PROVIDERS = providers


def _bake_identity() -> dict[str, Any]:
    model_dir = tts_cache._resolve_model_dir(None)
    voice_ko = tts_cache._resolve_voice_selector(None, "ko")
    voice_en = tts_cache._resolve_voice_selector(None, "en")
    return tts_cache._build_runtime_identity(
        model_dir=model_dir,
        voice_style_ko=voice_ko,
        voice_style_en=voice_en,
    )


def _read_manifest_payload(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    manifest: dict[str, dict[str, Any]] = {}
    for key, value in payload.items():
        if isinstance(key, str) and isinstance(value, dict):
            manifest[key] = dict(value)
    return manifest


def _is_valid_manifest_hit(
    out_dir: Path,
    manifest: dict[str, dict[str, Any]],
    key: str,
) -> bool:
    raw = manifest.get(key)
    if raw is None or raw.get("status") != "done":
        return False
    entry = tts_cache._parse_manifest_entry(raw)
    wav_path = out_dir / f"{key}.wav"
    return (
        entry is not None
        and entry.wav == wav_path.name
        and tts_cache._is_valid_wav_hit(wav_path, entry)
    )


def _manifest_entry(item: BakeText, wav_path: Path, *, steps: int) -> dict[str, Any]:
    return {
        "wav": wav_path.name,
        "text": item.text,
        "lang": item.lang,
        "steps": steps,
        "sr": tts_cache.CACHE_SAMPLE_RATE,
        "speed": tts_cache.TTS_SPEED,
        "bytes": wav_path.stat().st_size,
        "sha256": _sha256_file(wav_path),
        "status": "done",
        "source": item.source,
    }


def _write_cache_wav_atomic(path: Path, audio_samples: Any, sample_rate: int) -> None:
    pcm16 = _to_cache_pcm16(audio_samples, sample_rate)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("wb") as raw_handle:
        with wave.open(raw_handle, "wb") as wav_handle:
            wav_handle.setnchannels(1)
            wav_handle.setsampwidth(2)
            wav_handle.setframerate(tts_cache.CACHE_SAMPLE_RATE)
            wav_handle.writeframes(pcm16.tobytes())
        raw_handle.flush()
        os.fsync(raw_handle.fileno())
    os.replace(tmp_path, path)


def _to_cache_pcm16(audio_samples: Any, sample_rate: int) -> Any:
    import numpy as np

    audio = np.asarray(audio_samples)
    if audio.ndim > 1:
        audio = audio.reshape(-1, audio.shape[-1]).mean(axis=1)
    if np.issubdtype(audio.dtype, np.integer):
        integer_info = np.iinfo(audio.dtype)
        scale = float(max(abs(integer_info.min), integer_info.max))
        audio = audio.astype(np.float32) / scale
    else:
        audio = audio.astype(np.float32, copy=False)
    np.nan_to_num(audio, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    if sample_rate != tts_cache.CACHE_SAMPLE_RATE:
        try:
            import soxr  # type: ignore[import-not-found, import-untyped]
        except ImportError as exc:
            msg = "soxr is required to resample baked TTS cache audio to 16 kHz"
            raise RuntimeError(msg) from exc
        audio = soxr.resample(audio, sample_rate, tts_cache.CACHE_SAMPLE_RATE)
    audio = np.clip(audio, -1.0, 1.0)
    return np.rint(audio * 32767.0).astype(np.int16)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    with tmp_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _required_str(payload: dict[str, Any], key: str, path: Path) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}: {key} must be non-empty text")
    return value


def _log_event(event: str, **fields: Any) -> None:
    payload = {"event": event, **fields}
    logger.info(json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
