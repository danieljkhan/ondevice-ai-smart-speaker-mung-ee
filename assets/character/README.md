# Character Sprite Asset Spec

## 1. Purpose

This directory holds Mungi's character sprite assets; the placeholder generator script remains as a CI/dev fallback safety net.

## 2. Character Species

Mungi is a dog character. The final design should be child-friendly, soft, and calm enough that it will not frighten children under 8.

## 3. Technical Spec

Each sprite must be a 720x720 PNG with an RGBA alpha channel. The filename must match the lowercase `CharacterExpression` enum value: `neutral.png`, `idle.png`, `listening.png`, `thinking.png`, `speaking.png`, `happy.png`, `sad.png`, `surprised.png`, `concerned.png`, `joyful.png`, `greeting.png`, `excited.png`, `angry.png`, `sulky.png`, `sleepy.png`, `tired.png`, `shy.png`, `winking.png`, and `affectionate.png`.

## 4. Expression Mapping Table

| Expression enum | Source PNG | MP4 available (out of scope) | Dog behavior description |
|---|---|---|---|
| `NEUTRAL` | 자연스러움(기본).png | Yes | Calm forward gaze |
| `IDLE` | 편안함.png | No | Sitting comfortably |
| `LISTENING` | 궁금함.png | Yes | Perked ears, tilted head, curious attention |
| `THINKING` | 생각중.png | Yes | Slight head tilt while thinking |
| `SPEAKING` | 말하기.png | Yes | Mouth slightly open as if speaking |
| `HAPPY` | 행복함.png | Yes | Deep, stable satisfaction with smiling eyes |
| `SAD` | 슬픔.png | Yes | Drooped ears and low mood |
| `SURPRISED` | 놀람.png | Yes | Wide eyes and strongly perked ears |
| `CONCERNED` | 걱정.png | Yes | Ears slightly back, worried brow |
| `JOYFUL` | 즐거움.png | Yes | Immediate playful enjoyment |
| `GREETING` | 반가움.png | Yes | Warm hello, tail-wagging feel |
| `EXCITED` | 설레임.png | Yes | Anticipation and eager energy |
| `ANGRY` | 화남.png | Yes | Downturned mouth, growly feeling |
| `SULKY` | 삐짐.png | Yes | Pouty expression, gaze turned aside |
| `SLEEPY` | 졸림.png | No | Half-closed eyes, head lowering |
| `TIRED` | 피곤.png | No | Fatigued expression, lowered mouth corners |
| `SHY` | 부끄러움.png | Yes | Eyes slightly down, gentle closeness |
| `WINKING` | 윙크.png | No | One eye closed, light friendly cue |
| `AFFECTIONATE` | 사랑.png | Yes | Deep closeness, heart motif allowed |

LISTENING maps to `궁금함.png` because the asset reads as curious attention. If a future
visual review concludes the image reads as curious rather than attentive listening, a
re-pairing decision may move `LISTENING` to a dedicated asset and re-purpose
`궁금함.png` to a future `CURIOUS` enum.

## 5. Source-File Format

Keep the layered original outside the repo in PSD, SVG, Procreate, `.ai`, or an equivalent editable format. This repo stores only the final 720x720 PNG exports.

## 6. Canvas And Safe Zone

Use a 720x720 canvas. Keep the character inside the central 600x600 safe zone, leaving a 60px bezel margin on every side for fullscreen LCD display safety.

## 7. Transparent Background Policy

Final sprites must preserve a real RGBA alpha channel. The character should sit on transparent background pixels so the renderer can control the display fill independently.

## 8. No-Baked-Text Policy

Final sprites must not contain labels, captions, UI text, or speech text.

## 9. Export Checklist

Export as PNG-24 RGBA in sRGB color space, 8-bit per channel. Strip metadata and target a final file size under 200KB per sprite when practical. Files exceeding 200KB after Pillow LANCZOS resize and PNG optimization are tolerated up to 300KB; above 300KB requires PM review.

## 10. QA Criteria

The same dog identity must be recognizable across all 19 sprites. The color palette should stay harmonious, child-friendly, and not oversaturated. Expression differences must be immediately visible to children under 8. Use a central gaze by default, with head tilt allowed for `LISTENING` and `THINKING`. Edges should have smooth anti-aliasing.

## 11. External Illustrator Handoff Guide

Before starting, read this README and the Phase 2 B-2 plan's expression mapping table. Deliver exactly 19 PNG files with names matching the enum values listed in section 3.

## 12. Asset-Replacement Procedure

Use `scripts/convert_emoji_to_character.py` to convert Korean-named illustrator PNG files from `assets/emoji/` into English enum filenames under `assets/character/`. Future additions must follow the same convert-and-commit procedure so committed runtime assets remain 720x720 RGBA PNGs.

## 13. NEUTRAL Role

`NEUTRAL` is the universal fallback when another sprite is unavailable. Its design must read as the default gaze.

## 14. Asset Status Marker

| Enum | Asset status | Source |
|---|---|---|
| `NEUTRAL` | Illustration | External illustrator, 2026-05-27 handoff |
| `IDLE` | Illustration | External illustrator, 2026-05-27 handoff |
| `LISTENING` | Illustration | External illustrator, 2026-05-27 handoff |
| `THINKING` | Illustration | External illustrator, 2026-05-27 handoff |
| `SPEAKING` | Illustration | External illustrator, 2026-05-28 addition |
| `HAPPY` | Illustration | External illustrator, 2026-05-27 handoff |
| `SAD` | Illustration | External illustrator, 2026-05-27 handoff |
| `SURPRISED` | Illustration | External illustrator, 2026-05-27 handoff |
| `CONCERNED` | Illustration | External illustrator, 2026-05-27 handoff |
| `JOYFUL` | Illustration | External illustrator, 2026-05-27 handoff |
| `GREETING` | Illustration | External illustrator, 2026-05-27 handoff |
| `EXCITED` | Illustration | External illustrator, 2026-05-27 handoff |
| `ANGRY` | Illustration | External illustrator, 2026-05-27 handoff |
| `SULKY` | Illustration | External illustrator, 2026-05-27 handoff |
| `SLEEPY` | Illustration | External illustrator, 2026-05-27 handoff |
| `TIRED` | Illustration | External illustrator, 2026-05-27 handoff |
| `SHY` | Illustration | External illustrator, 2026-05-27 handoff |
| `WINKING` | Illustration | External illustrator, 2026-05-27 handoff |
| `AFFECTIONATE` | Illustration | External illustrator, 2026-05-27 handoff |
