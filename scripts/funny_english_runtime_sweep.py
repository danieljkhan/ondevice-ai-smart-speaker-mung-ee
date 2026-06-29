"""Sweep Funny English runtime assets for parse, asset, scoring, and gate errors."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image  # type: ignore[import-not-found, import-untyped]

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.funny_english_match import match_funny_english_attempt

logger = logging.getLogger(__name__)

WHITELISTED_LICENSES = frozenset({"PD", "CC-BY", "CC0", "OFL", "MIT"})
LICENSE_REQUIRED_FIELDS = frozenset({"license", "source", "title", "author", "notice"})


@dataclass
class SweepStats:
    """Counters emitted by the Funny English runtime sweep."""

    stages: int = 0
    cards: int = 0
    missing_images: int = 0
    missing_audio: int = 0
    missing_bgm: int = 0
    image_errors: int = 0
    audio_errors: int = 0
    scoring_errors: int = 0
    license_errors: int = 0

    @property
    def error_count(self) -> int:
        """Return the sum of all error counters."""
        return (
            self.missing_images
            + self.missing_audio
            + self.missing_bgm
            + self.image_errors
            + self.audio_errors
            + self.scoring_errors
            + self.license_errors
        )


def run_sweep(*, repo_root: Path = Path(".")) -> SweepStats:
    """Run the Funny English runtime asset sweep and return counters."""
    manifest_path = repo_root / "assets" / "funny_english" / "manifest.json"
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != 1:
        raise ValueError("Funny English manifest schema_version must be 1")
    raw_stages = manifest.get("stages")
    if not isinstance(raw_stages, list):
        raise ValueError("Funny English manifest stages must be a list")

    stats = SweepStats(stages=len(raw_stages))
    for raw_stage_entry in raw_stages:
        stage_entry = _require_object(raw_stage_entry, "manifest stage")
        stage_path = repo_root / _require_str(stage_entry, "stage_path")
        with stage_path.open(encoding="utf-8") as handle:
            stage = json.load(handle)
        if stage.get("schema_version") != 1:
            raise ValueError(f"{stage_path} schema_version must be 1")
        raw_cards = stage.get("cards")
        if not isinstance(raw_cards, list):
            raise ValueError(f"{stage_path} cards must be a list")
        if len(raw_cards) != _require_int(stage_entry, "card_count"):
            raise ValueError(f"{stage_path} card_count does not match manifest")
        for raw_card in raw_cards:
            card = _require_object(raw_card, f"{stage_path} card")
            stats.cards += 1
            _validate_license(card, stats)
            _validate_image(repo_root / _require_str(card, "image"), stats)
            _validate_audio(repo_root / _require_str(card, "model_audio"), stats)
            _validate_scoring(card, stats)

    bgm_path = repo_root / "assets" / "funny_english" / "music" / "bgm_loop.wav"
    if not bgm_path.exists():
        stats.missing_bgm += 1
    else:
        _validate_audio(bgm_path, stats)
    _validate_notice(repo_root / "assets" / "funny_english" / "NOTICE", stats)
    if stats.error_count:
        raise ValueError(f"Funny English sweep failed with {stats.error_count} errors: {stats}")
    return stats


def _validate_license(card: dict[str, Any], stats: SweepStats) -> None:
    license_info = card.get("asset_license")
    if not isinstance(license_info, dict):
        stats.license_errors += 1
        return
    missing = LICENSE_REQUIRED_FIELDS - set(license_info)
    if missing:
        stats.license_errors += 1
        return
    if license_info.get("license") not in WHITELISTED_LICENSES:
        stats.license_errors += 1


def _validate_image(path: Path, stats: SweepStats) -> None:
    if not path.exists():
        stats.missing_images += 1
        return
    try:
        with Image.open(path) as image:
            image.verify()
    except OSError:
        stats.image_errors += 1
        logger.warning("Failed to decode Funny English image %s", path, exc_info=True)


def _validate_audio(path: Path, stats: SweepStats) -> None:
    if not path.exists():
        stats.missing_audio += 1
        return
    try:
        with wave.open(str(path), "rb") as handle:
            if handle.getnchannels() < 1 or handle.getframerate() <= 0 or handle.getnframes() < 1:
                stats.audio_errors += 1
    except (wave.Error, OSError):
        stats.audio_errors += 1
        logger.warning("Failed to decode Funny English audio %s", path, exc_info=True)


def _validate_scoring(card: dict[str, Any], stats: SweepStats) -> None:
    try:
        tokens = _require_str_list(card, "tokens")
        text = _require_str(card, "text")
        pass_result = match_funny_english_attempt(text, tokens)
        low_result = match_funny_english_attempt("zzz qqq", tokens)
        silent_result = match_funny_english_attempt("", tokens)
        if pass_result.band != "pass":
            stats.scoring_errors += 1
        if low_result.band not in {"low", "silent_junk"}:
            stats.scoring_errors += 1
        if silent_result.band != "silent_junk":
            stats.scoring_errors += 1
    except (TypeError, ValueError):
        stats.scoring_errors += 1
        logger.warning("Funny English scoring validation failed for %s", card, exc_info=True)


def _validate_notice(path: Path, stats: SweepStats) -> None:
    if not path.exists():
        stats.license_errors += 1
        return
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        stats.license_errors += 1
        return
    if "Funny English NOTICE" not in text:
        stats.license_errors += 1


def _require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _require_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Funny English payload field {key!r} must be non-empty text")
    return value


def _require_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int):
        raise ValueError(f"Funny English payload field {key!r} must be int")
    return value


def _require_str_list(payload: dict[str, Any], key: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"Funny English payload field {key!r} must be list[str]")
    return list(value)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep Funny English runtime assets.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the runtime sweep."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _build_parser().parse_args(argv)
    try:
        stats = run_sweep(repo_root=args.repo_root)
    except Exception as exc:
        logger.error("funny_english_sweep failed: %s", exc)
        return 1
    logger.info(
        "funny_english_sweep stages=%d cards=%d errors=%d",
        stats.stages,
        stats.cards,
        stats.error_count,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
