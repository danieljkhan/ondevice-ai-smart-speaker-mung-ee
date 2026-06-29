"""Build runtime assets for the Funny English read-along mode."""

from __future__ import annotations

import argparse
import json
import logging
import math
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont  # type: ignore[import-not-found, import-untyped]

logger = logging.getLogger("mungi.scripts.build_funny_english_content")

JsonObject = dict[str, Any]
DEFAULT_OUT = Path("assets/funny_english")
WHITELISTED_LICENSES = frozenset({"PD", "CC-BY", "CC0", "OFL", "MIT"})
LICENSE_REQUIRED_FIELDS = frozenset({"license", "source", "title", "author", "notice"})
STAGE_TITLES = {
    0: "Alphabet sounds",
    1: "CVC words",
    2: "First sight words",
    3: "Word pairs",
    4: "Short sentences",
    5: "Tiny readers",
    6: "The Hare and the Tortoise",
    7: "The Lion and the Mouse",
}
# Story stages adapt public-domain Aesop fables (original simplified text + AI-generated art).
STORY_STAGES = frozenset({6, 7})
LICENSE_TEXTS = {
    "MIT.txt": (
        "MIT License\n\n"
        "Permission is hereby granted, free of charge, to any person obtaining a copy "
        'of this software and associated documentation files (the "Software"), to deal '
        "in the Software without restriction, including without limitation the rights "
        "to use, copy, modify, merge, publish, distribute, sublicense, and/or sell "
        "copies of the Software, and to permit persons to whom the Software is "
        "furnished to do so, subject to the following conditions:\n\n"
        "The above copyright notice and this permission notice shall be included in all "
        "copies or substantial portions of the Software.\n\n"
        'THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR '
        "IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, "
        "FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE "
        "AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER "
        "LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, "
        "OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE "
        "SOFTWARE.\n"
    ),
    "PD.txt": "Public Domain notice: the referenced educational text is free of known copyright restrictions.\n",
    "CC0.txt": "CC0 notice: the referenced material is dedicated to the public domain where legally possible.\n",
    "CC-BY.txt": "CC BY notice: attribution is required for the referenced material.\n",
    "OFL.txt": "SIL Open Font License notice: bundled font assets must retain their OFL terms.\n",
}


class BuildError(RuntimeError):
    """Raised when the Funny English content build cannot produce valid output."""


@dataclass(frozen=True)
class BuildOptions:
    """Command-line options for one Funny English content build."""

    out: Path = DEFAULT_OUT
    max_dim: int = 720
    sample_rate: int = 16000
    force: bool = False
    dry_run: bool = False


@dataclass
class BuildStats:
    """Counters collected during one content build."""

    stages: int = 0
    cards: int = 0
    images_written: int = 0
    audio_written: int = 0
    json_written: int = 0
    notice_written: bool = False


def build_funny_english_content(options: BuildOptions) -> BuildStats:
    """Build manifest JSON, placeholder images/audio, NOTICE, and licenses."""
    if options.max_dim <= 0:
        raise BuildError("--max-dim must be positive")
    if options.sample_rate <= 0:
        raise BuildError("--sample-rate must be positive")

    cards_by_stage = _starter_curriculum(options.out)
    _validate_curriculum(cards_by_stage)
    stats = BuildStats(
        stages=len(cards_by_stage), cards=sum(len(v) for v in cards_by_stage.values())
    )

    stage_entries: list[JsonObject] = []
    for stage, cards in sorted(cards_by_stage.items()):
        stage_path = options.out / "stages" / f"stage_{stage}.json"
        stage_payload = {
            "schema_version": 1,
            "stage": stage,
            "title": STAGE_TITLES[stage],
            "cards": cards,
        }
        if _write_json_if_changed(stage_path, stage_payload, options=options):
            stats.json_written += 1
        stage_entries.append(
            {
                "stage": stage,
                "title": STAGE_TITLES[stage],
                "stage_path": _json_path(stage_path),
                "card_count": len(cards),
            }
        )
        for card in cards:
            if _ensure_placeholder_image(card, options=options):
                stats.images_written += 1
            if _ensure_placeholder_audio(card, options=options):
                stats.audio_written += 1

    manifest = {
        "schema_version": 1,
        "title": "Funny English",
        "stages": stage_entries,
        "mastery_successes": 2,
        "stage_advance_pct": 0.8,
    }
    if _write_json_if_changed(options.out / "manifest.json", manifest, options=options):
        stats.json_written += 1
    if _write_text_if_changed(
        options.out / "NOTICE", _notice_text(cards_by_stage), options=options
    ):
        stats.notice_written = True
    for filename, text in LICENSE_TEXTS.items():
        _write_text_if_changed(options.out / "licenses" / filename, text, options=options)
    _ensure_bgm(options, stats)
    _log_stats(stats)
    return stats


def _starter_curriculum(out: Path) -> dict[int, list[JsonObject]]:
    cards: dict[int, list[JsonObject]] = {stage: [] for stage in STAGE_TITLES}
    stage_specs: dict[int, list[tuple[str, str, str, list[str], str]]] = {
        0: [
            ("fe_s0_a", "A", "apple", ["a"], "애플의 첫 소리"),
            ("fe_s0_b", "B", "ball", ["b"], "공의 첫 소리"),
            ("fe_s0_c", "C", "cat", ["c"], "고양이의 첫 소리"),
            ("fe_s0_m", "M", "moon", ["m"], "달의 첫 소리"),
        ],
        1: [
            ("fe_s1_cat", "cat", "cat", ["cat"], "고양이"),
            ("fe_s1_dog", "dog", "dog", ["dog"], "강아지"),
            ("fe_s1_sun", "sun", "sun", ["sun"], "해"),
            ("fe_s1_hat", "hat", "hat", ["hat"], "모자"),
            ("fe_s1_map", "map", "map", ["map"], "지도"),
        ],
        2: [
            ("fe_s2_the", "the", "the", ["the"], "자주 나오는 말"),
            ("fe_s2_and", "and", "and", ["and"], "그리고"),
            ("fe_s2_see", "see", "see", ["see"], "보다"),
            ("fe_s2_play", "play", "play", ["play"], "놀다"),
            ("fe_s2_like", "like", "like", ["like"], "좋아하다"),
        ],
        3: [
            ("fe_s3_ship", "ship", "ship", ["ship"], "배"),
            ("fe_s3_fish", "fish", "fish", ["fish"], "물고기"),
            ("fe_s3_star", "star", "star", ["star"], "별"),
            ("fe_s3_tree", "tree", "tree", ["tree"], "나무"),
        ],
        4: [
            ("fe_s4_i_see_cat", "I see a cat", "cat", ["i", "see", "cat"], "고양이가 보여"),
            ("fe_s4_we_play", "We play", "play", ["we", "play"], "우리는 놀아"),
            ("fe_s4_the_sun", "The sun is up", "sun", ["the", "sun", "is", "up"], "해가 떴어"),
        ],
        5: [
            ("fe_s5_moon_1", "I see the moon", "moon", ["i", "see", "the", "moon"], "달이 보여"),
            ("fe_s5_moon_2", "The moon is big", "moon", ["the", "moon", "is", "big"], "달이 커"),
            ("fe_s5_moon_3", "I like the moon", "moon", ["i", "like", "the", "moon"], "달이 좋아"),
        ],
        6: [
            (
                "fe_aesop_hare_01",
                "The hare can run very fast.",
                "hare",
                ["the", "hare", "can", "run", "very", "fast"],
                "토끼는 아주 빨리 달릴 수 있어요.",
            ),
            (
                "fe_aesop_hare_02",
                "The slow tortoise walks step by step.",
                "tortoise",
                ["the", "slow", "tortoise", "walks", "step", "by", "step"],
                "느린 거북이는 한 걸음씩 걸어요.",
            ),
            (
                "fe_aesop_hare_03",
                '"Let\'s run a race!" says the hare.',
                "race",
                ["let's", "run", "a", "race", "says", "the", "hare"],
                '토끼가 "달리기 시합하자!"라고 말해요.',
            ),
            (
                "fe_aesop_hare_04",
                "The hare runs far ahead and sleeps.",
                "sleeps",
                ["the", "hare", "runs", "far", "ahead", "and", "sleeps"],
                "토끼는 멀리 앞서가서 잠을 자요.",
            ),
            (
                "fe_aesop_hare_05",
                "The tortoise keeps going and never stops.",
                "tortoise",
                ["the", "tortoise", "keeps", "going", "and", "never", "stops"],
                "거북이는 멈추지 않고 계속 가요.",
            ),
            (
                "fe_aesop_hare_06",
                "The little tortoise wins the big race!",
                "wins",
                ["the", "little", "tortoise", "wins", "the", "big", "race"],
                "작은 거북이가 큰 시합에서 이겨요!",
            ),
            (
                "fe_aesop_hare_07",
                "Slow and steady wins the race.",
                "steady",
                ["slow", "and", "steady", "wins", "the", "race"],
                "느려도 꾸준하면 이겨요.",
            ),
        ],
        7: [
            (
                "fe_aesop_lion_01",
                "A big lion sleeps under a tree.",
                "lion",
                ["a", "big", "lion", "sleeps", "under", "a", "tree"],
                "큰 사자가 나무 아래서 자요.",
            ),
            (
                "fe_aesop_lion_02",
                "A little mouse runs across his nose.",
                "mouse",
                ["a", "little", "mouse", "runs", "across", "his", "nose"],
                "작은 쥐가 사자 코 위로 달려가요.",
            ),
            (
                "fe_aesop_lion_03",
                "The lion wakes up and feels angry.",
                "lion",
                ["the", "lion", "wakes", "up", "and", "feels", "angry"],
                "사자가 깨어나서 화가 나요.",
            ),
            (
                "fe_aesop_lion_04",
                '"Please let me go!" cries the mouse.',
                "mouse",
                ["please", "let", "me", "go", "cries", "the", "mouse"],
                '쥐가 "제발 보내 주세요!"라고 울어요.',
            ),
            (
                "fe_aesop_lion_05",
                "The kind lion lets the mouse go.",
                "lion",
                ["the", "kind", "lion", "lets", "the", "mouse", "go"],
                "착한 사자가 쥐를 보내 줘요.",
            ),
            (
                "fe_aesop_lion_06",
                "One day a net traps the lion.",
                "net",
                ["one", "day", "a", "net", "traps", "the", "lion"],
                "어느 날 그물이 사자를 가둬요.",
            ),
            (
                "fe_aesop_lion_07",
                "The little mouse bites the ropes away.",
                "mouse",
                ["the", "little", "mouse", "bites", "the", "ropes", "away"],
                "작은 쥐가 밧줄을 물어 끊어요.",
            ),
            (
                "fe_aesop_lion_08",
                "Small friends can help big friends too.",
                "friends",
                ["small", "friends", "can", "help", "big", "friends", "too"],
                "작은 친구도 큰 친구를 도울 수 있어요.",
            ),
        ],
    }
    for stage, specs in stage_specs.items():
        for card_id, text, keyword, tokens, gloss in specs:
            cards[stage].append(
                _card_payload(
                    out=out,
                    card_id=card_id,
                    stage=stage,
                    text=text,
                    keyword=keyword,
                    tokens=tokens,
                    ko_gloss=gloss,
                    license_info=_story_license(text) if stage in STORY_STAGES else None,
                )
            )
    return cards


def _story_license(text: str) -> JsonObject:
    """Return the asset-license block for an Aesop story card (PD-derived + AI art)."""
    return {
        "license": "PD",
        "source": "Mungi Funny English - Aesop story cards",
        "title": text,
        "author": "Mungi project",
        "notice": (
            "Card text: original simplified adaptation of public-domain Aesop. "
            "Illustration: AI-generated (Nano Banana 2, 2026-06-17), Mungi project."
        ),
    }


def _card_payload(
    *,
    out: Path,
    card_id: str,
    stage: int,
    text: str,
    keyword: str,
    tokens: list[str],
    ko_gloss: str,
    license_info: JsonObject | None = None,
) -> JsonObject:
    image_path = out / "images" / f"{card_id}.jpg"
    audio_path = out / "audio" / f"{card_id}.wav"
    asset_license = (
        license_info
        if license_info is not None
        else {
            "license": "PD",
            "source": "Mungi Funny English starter curriculum",
            "title": text,
            "author": "Mungi project",
            "notice": "Original/PD placeholder card generated for Funny English v1.",
        }
    )
    return {
        "schema_version": 1,
        "card_id": card_id,
        "stage": stage,
        "type": "reader" if stage >= 5 else "word",
        "text": text,
        "tokens": tokens,
        "ko_gloss": ko_gloss,
        "syllables": list(text.replace(" ", "")) if len(tokens) == 1 else tokens,
        "first_sound": tokens[0][0],
        "image": _json_path(image_path),
        "model_audio": _json_path(audio_path),
        "accept_threshold": 0.6,
        "hotwords": tokens,
        "asset_license": asset_license,
    }


def _validate_curriculum(cards_by_stage: dict[int, list[JsonObject]]) -> None:
    for stage, cards in cards_by_stage.items():
        if stage not in STAGE_TITLES:
            raise BuildError(f"unknown stage: {stage}")
        if not cards:
            raise BuildError(f"stage {stage} has no cards")
        for card in cards:
            license_info = card.get("asset_license")
            if not isinstance(license_info, dict):
                raise BuildError(f"{card.get('card_id')} missing asset_license")
            missing = sorted(LICENSE_REQUIRED_FIELDS - set(license_info))
            if missing:
                raise BuildError(f"{card.get('card_id')} missing license fields: {missing}")
            license_id = license_info.get("license")
            if license_id not in WHITELISTED_LICENSES:
                raise BuildError(f"{card.get('card_id')} has unknown license: {license_id}")


def _ensure_placeholder_image(card: JsonObject, *, options: BuildOptions) -> bool:
    path = Path(_require_str(card, "image"))
    if path.exists() and not options.force:
        return False
    if options.dry_run:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (options.max_dim, options.max_dim), "#1B1B1F")
    draw = ImageDraw.Draw(image)
    keyword = _require_str(card, "text")
    font = _load_font(72)
    small_font = _load_font(32)
    draw.rounded_rectangle((56, 72, 664, 470), radius=24, fill="#F4F6F8")
    draw.text((360, 250), keyword[:18], anchor="mm", fill="#20242A", font=font)
    draw.text((360, 560), "Funny English", anchor="mm", fill="#F4F6F8", font=small_font)
    image.save(path, format="JPEG", quality=90, optimize=True)
    return True


def _ensure_placeholder_audio(card: JsonObject, *, options: BuildOptions) -> bool:
    path = Path(_require_str(card, "model_audio"))
    if path.exists() and not options.force:
        return False
    if options.dry_run:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_silent_wav(path, duration_s=0.45, sample_rate=options.sample_rate)
    return True


def _ensure_bgm(options: BuildOptions, stats: BuildStats) -> None:
    path = options.out / "music" / "bgm_loop.wav"
    if path.exists() and not options.force:
        return
    if options.dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    duration_s = 2.0
    t = np.linspace(0.0, duration_s, int(options.sample_rate * duration_s), endpoint=False)
    wave_a = np.sin(2 * math.pi * 220.0 * t) * 0.07
    wave_b = np.sin(2 * math.pi * 330.0 * t) * 0.04
    samples = np.asarray((wave_a + wave_b) * 32767, dtype=np.int16)
    _write_wav(path, samples, options.sample_rate)
    stats.audio_written += 1


def _write_silent_wav(path: Path, *, duration_s: float, sample_rate: int) -> None:
    frames = np.zeros(max(1, int(round(duration_s * sample_rate))), dtype=np.int16)
    _write_wav(path, frames, sample_rate)


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(samples.astype("<i2").tobytes())


def _notice_text(cards_by_stage: dict[int, list[JsonObject]]) -> str:
    lines = ["Funny English NOTICE", ""]
    seen: set[tuple[str, str, str]] = set()
    for cards in cards_by_stage.values():
        for card in cards:
            license_info = card["asset_license"]
            key = (
                str(license_info["license"]),
                str(license_info["title"]),
                str(license_info["source"]),
            )
            if key in seen:
                continue
            seen.add(key)
            lines.append(
                f"- {license_info['title']} ({license_info['license']}), "
                f"{license_info['author']}, {license_info['source']}: {license_info['notice']}"
            )
    lines.append("")
    return "\n".join(lines)


def _write_json_if_changed(path: Path, payload: JsonObject, *, options: BuildOptions) -> bool:
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    return _write_text_if_changed(path, text, options=options)


def _write_text_if_changed(path: Path, text: str, *, options: BuildOptions) -> bool:
    if options.dry_run:
        logger.info("dry-run: would write %s", _display_path(path))
        return False
    if not options.force and path.exists() and path.read_text(encoding="utf-8") == text:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    logger.info("wrote %s", _display_path(path))
    return True


def _load_font(size: int) -> Any:
    font_path = Path("assets/fonts/pretendard-history-kr.subset.ttf")
    try:
        return ImageFont.truetype(str(font_path), size)
    except OSError:
        return ImageFont.load_default()


def _require_str(payload: JsonObject, key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise BuildError(f"Expected non-empty string field {key!r}")
    return value


def _json_path(path: Path) -> str:
    return _display_path(path)


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _log_stats(stats: BuildStats) -> None:
    logger.info("stages=%s cards=%s", stats.stages, stats.cards)
    logger.info(
        "json_written=%s images_written=%s audio_written=%s notice_written=%s",
        stats.json_written,
        stats.images_written,
        stats.audio_written,
        stats.notice_written,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-dim", type=int, default=720)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _options_from_args(args: argparse.Namespace) -> BuildOptions:
    return BuildOptions(
        out=args.out,
        max_dim=args.max_dim,
        sample_rate=args.sample_rate,
        force=args.force,
        dry_run=args.dry_run,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        build_funny_english_content(_options_from_args(args))
    except BuildError as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
