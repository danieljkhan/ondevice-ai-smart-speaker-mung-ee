# Curated Korean-History Scene Schema

This document describes the schema of the runtime document files produced for
the "재미있는 우리역사" (Korean history) picture-storytelling mode and consumed by
`core/history_mode.py`.

## Source vs. runtime layers

| Layer | Location | Producer |
|-------|----------|----------|
| Source scenes | `assets/dataset_korean history/data/scenes/{doc_hash}.json` | upstream dataset |
| Source figures | `assets/dataset_korean history/data/figures/{doc_hash}/fig_NNN.jpg` | upstream dataset |
| Runtime manifest | `assets/history/manifest.json` | `scripts/build_history_content.py` + `scripts/history_era_order.py` |
| Runtime docs | `assets/history/docs/{doc_hash}.json` | `scripts/build_history_content.py` |
| Runtime images | `assets/history/images/{doc_hash}/fig_NNN.jpg` | `scripts/build_history_content.py` |
| Image anchors | `assets/history/docs/{doc_hash}.json` (`images[].anchor_ratio`) | `scripts/history_image_anchors.py` |
| Chronological order index | `assets/history/era_order_index.json` | `scripts/history_era_order_fetch.py` |

The original source PDFs live under `assets/우리역사/{source_file}` and are used
only at build time (never at runtime). Both the source dataset and the PDFs are
git-ignored build inputs; only the derived `assets/history/` tree ships.

## Runtime manifest (`assets/history/manifest.json`)

The manifest is the catalogue consumed by `core/history_mode.py`.

Top-level fields:

| Field | Type | Notes |
|-------|------|-------|
| `schema_version` | int | currently `2` |
| `title` | str | runtime catalogue title |
| `era_order` | list[str] | display order for era buckets |
| `docs` | list | manifest document entries (see below) |

### Manifest document entry (`docs[]`)

| Field | Type | Notes |
|-------|------|-------|
| `doc_hash` | str | document identifier (also the runtime doc file stem) |
| `source_file` | str | original PDF name, e.g. `eh_r0030_0010.pdf` |
| `title` | str | runtime display title |
| `kind` | str | `people` or `artifact`; retained as metadata, not a menu level |
| `era` | str | one of the curated era buckets |
| `scene_count` | int | number of scenes |
| `section_count` | int | number of consent-paced sections |
| `image_count` | int | total images across all scenes |
| `est_total_ms` | int | estimated total narration duration |
| `doc_path` | str | repo-relative path to the runtime document JSON |
| `title_curated` | bool | whether the runtime title was curated from known metadata |
| `era_source` | str | `keyword` or `docnum` |
| `order` | int | official-site chronological sequence, global and 0-based across `eh_age_10`..`eh_age_50` |

Runtime menu behavior:

- The catalogue menu is **Era → Document** (2 levels).
- Era options follow `era_order`.
- Within an era, `people` and `artifact` documents are merged into one list and ordered by `order`.

## Runtime document (`assets/history/docs/{doc_hash}.json`)

Top-level fields:

| Field | Type | Notes |
|-------|------|-------|
| `schema_version` | int | currently `2` |
| `doc_hash` | str | document identifier (also the file stem) |
| `source_file` | str | original PDF name, e.g. `eh_r0030_0010.pdf` |
| `title` | str | runtime display title |
| `kind` | str | `people` or `artifact` |
| `era` | str | one of the curated era buckets |
| `era_source` | str | `keyword` or `docnum` |
| `scene_count` | int | number of scenes |
| `section_count` | int | number of consent-paced sections |
| `image_count` | int | total images across all scenes |
| `est_total_ms` | int | estimated total narration duration |
| `sections` | list | section grouping metadata |
| `scenes` | list | the narrated scenes (see below) |

### Scene object (`scenes[]`)

| Field | Type | Notes |
|-------|------|-------|
| `seq` | int | 1-based scene sequence within the document |
| `section_index` | int | the section this scene belongs to |
| `section_title` | str \| null | the section heading (may repeat across scenes) |
| `narration` | str | paragraphs joined by `\n`; TTS-only, never rendered as text |
| `est_speech_ms` | int | estimated narration duration |
| `tail_silence_ms` | int | trailing silence after narration |
| `image_captions` | list[str] | scene-level captions (reading order) |
| `images` | list | the images attached to the scene (see below) |

### Image object (`scenes[].images[]`)

| Field | Type | Notes |
|-------|------|-------|
| `path` | str | repo-relative path to the letterboxed runtime image |
| `caption` | str \| null | per-image caption (preferred for display) |
| `letterboxed` | bool | always `true` for runtime images |
| `clean` | bool | image needs no character-margin handling |
| `is_infographic` | bool | image is a map/table/diagram |
| `anchor_ratio` | float \| null | **narration position of the image, in `[0, 1]`** |

#### `anchor_ratio` (added 2026-06-09)

`anchor_ratio` records *where in the scene narration an image belongs*, as a
fraction of the narration length (`0.0` = scene start, `1.0` = scene end). It is
derived at build time by `scripts/history_image_anchors.py` from the original
PDF and lets the runtime show each image when the narration reaches its
position, instead of cycling images positionally.

Guarantees:

- Within a scene, anchors are **strictly increasing** in image order, so the
  slideshow never flips an image backward.
- The first image of every scene is anchored at `0.0` (shown at scene start);
  when any image carries a real PDF-derived anchor, the last image is anchored
  at `1.0` so it is always reached at the end of narration.
- Images whose PDF caption can be matched receive a **real** narration-derived
  anchor; the rest are filled with monotonically increasing, evenly spaced
  anchors, which is also the fallback for an entire scene when no caption
  matches (or the source PDF is unavailable).

Runtime behavior (`core/history_mode.py`):

- The narration worker tracks playback progress (fraction of spoken segments,
  with the final segment mapping to `1.0`) and displays the last image whose
  `anchor_ratio <= progress`.
- When `anchor_ratio` is absent on any image of a scene (older data), the
  runtime falls back to even spacing over the scene's display ordering.

Regenerate after rebuilding content:

```bash
python -m scripts.build_history_content      # writes docs + images
python -m scripts.history_image_anchors       # adds images[].anchor_ratio
python -m scripts.history_era_order           # adds manifest docs[].order and sorts docs
```

`history_image_anchors` is deterministic and idempotent — re-running rewrites
the same anchors.

`assets/history/era_order_index.json` is the committed build-time ordering
source for `history_era_order`. Regenerate it explicitly with:

```bash
python -m scripts.history_era_order_fetch --out assets/history/era_order_index.json
```
