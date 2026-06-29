# ADR 0109: Upstage Document Parse API as a Build-Time-Only Dependency for `재미있는 우리역사` Content Regeneration

## Status

Proposed

(Will move to Accepted after the one-shot keyed regeneration runs, the regenerated runtime docs pass the validation invariants + existing tests, and the content ships. Methodology: `Dev_Plan/2026-06-10-history-upstage-pdf-parse-methodology.md` v4, converged through Codex 3-round review — r1 PUSH BACK → r2 PUSH BACK → r3 APPROVE WITH NOTES.)

## Context

The `재미있는 우리역사` history mode (ADR 0107) plays 240 curated documents built from an upstream scene-JSON layer (`assets/dataset_korean history/data/scenes/{doc_hash}.json`) — an externally-produced artifact with no PDF→scene extraction script in the repo. Two open defects from the r3 content review share that single root cause:

1. **Narration↔image sync**: each figure is attached to the PDF page it physically appears on (a `page` field), not to the narration sentence that discusses it, so images drift out of sync with the spoken narration.
2. **Subtitle/consent pacing**: each scene's `section_title` is the first sentence of that scene's narration, and `scripts/build_history_content.py` opens a new section on any non-null `section_title` → roughly one section per scene → the inter-section consent prompt fires constantly.

Both defects can only be fixed upstream of the builder: the scene layer needs real document structure (headings, figure/caption typing, reading order) extracted from the source PDFs (`assets/우리역사/eh_*.pdf`). The Upstage Document Parse API provides exactly this — Layout Element Detection returns real `heading1` elements, typed `figure`/`caption` elements with bounding boxes, and a global reading order, with first-class Korean/CJK support.

This introduces a new external API dependency and an API key, which requires an ADR (CLAUDE.md §7 rule 4) and explicit user approval for key usage (`feedback_no_api_key_without_approval` — granted in principle; key provided at implementation time).

## Decision

### 1. Adopt Upstage Document Parse as a BUILD-TIME-ONLY, one-shot, dev-PC-only dependency

- Endpoint: `POST https://api.upstage.ai/v1/document-digitization` (sync) with `model=document-parse`, `output_formats=["text","html"]`, `coordinates=true`, `ocr=auto`. The batch runs **sequentially over the sync endpoint** (each PDF is ~6–10 pages, far under the sync 100-page cap; the async variant carries an up-to-72 h peak-queue risk with no SLA and is not used). Live-verified rate limits (2026-06-10): default tier 1 request/s, 300 pages/min.
- Tier/cost: a 2026-06-10 one-PDF live smoke test billed all pages at the **Standard tier ($0.01/page)** by default while still returning `heading1` elements (over-detecting, even — the fix is a heading pre-filter, not a higher tier), so the expected cost is **~$24 one-time** (240 PDFs ≈ ~2,400 pages); the earlier Enhanced-tier estimate (~$0.03/page ≈ ~$72) is retained only as the upper bound. No standing free-page credit is assumed.
- The dependency lives in ONE new build-time script, `scripts/history_pdf_parse.py`, run once on the dev PC. **`core/` never imports it** (same discipline as PyMuPDF in `scripts/history_image_anchors.py`). Zero runtime change for the happy path.
- **CI stays key-free**: CI validates only the committed runtime docs (`assets/history/docs/*.json` — the single committed artifact and authoritative baseline). CI never runs `history_pdf_parse.py`, never calls Upstage, and never needs the key, the source PDFs, or the API cache (all gitignored). Raw Upstage responses are cached locally keyed by `sha256(pdf bytes)` so re-runs never re-call the API.

### 2. OVERLAY architecture — preserve content, overlay structure

`scripts/history_pdf_parse.py` does **not** regenerate narration, scenes, timings, or figures. It overlays Upstage-derived structure onto the existing content:

- **Preserved verbatim**: per-scene `narration`, `est_speech_ms`, `tail_silence_ms`, `page`, `seq`, `image_paths`/`image_captions` (+ singular legacy fields), and the existing `fig_NNN.jpg` files. Timings are copied (Upstage cannot produce TTS durations); figures are reused, never re-cropped.
- **Overlaid (the only recomputed fields)**:
  - real-heading `section_index`/`section_title` — sections cut at each `heading1` in reading order, written into the upstream scene JSON (builder INPUT), so consent fires only at real heading changes;
  - title-aware `images[].anchor_ratio` — each figure anchored to the spoken segment that discusses it, computed over the runtime's `_narration_segments` segmentation (so the anchor matches `_select_scene_image_by_progress`), and **written post-build directly into the runtime docs** by the same script (reusing the existing `resolve_scene_anchors` helper for strictly-increasing / first=0.0 / last=1.0 guarantees).
- The post-build annotator is the **sole anchor writer**: the old `scripts/history_image_anchors.py` PDF-heuristic anchor pass is **RETIRED for these docs** (not run in this flow).
- The downstream contract is untouched: `build_history_content.py`, the runtime-doc schema (`schema_version==2`), and `core/history_mode.py` section grouping / anchor selection all keep their existing behavior — only their inputs improve.

### 3. API-key handling (user-approved)

The script reads `UPSTAGE_API_KEY` from the environment only. The key is never committed, never placed in config files, and never needed by CI or the Jetson runtime. The script errors out with a clear message if the variable is absent. Key usage was explicitly approved by the user per `feedback_no_api_key_without_approval`.

### 4. Validation invariants and safeguards (must hold for the 0-fix bar)

1. **Bucket-conditional sections**: `upstage_ok` docs → every `section_title` is a real `heading1` text; `upstage_fallback` docs → today's heuristic output (explicitly allowed). Telemetry reports the split (`upstage_ok` / `upstage_fallback` / `headings_found`).
2. **Heading fallback**: per-doc fallback to today's sectioning heuristic on Upstage error/timeout or when `heading_count < 2` (a single-heading parse would yield one giant section). No regression for un-parseable PDFs.
3. **Figure-preservation invariant**: each regenerated runtime doc's `image_paths` basenames + order, the `fig_NNN.jpg` set, per-image captions, and the doc/manifest `image_count` are identical to the committed baseline (figures are only re-anchored, never added/removed/reordered). A doc failing this falls back to today's anchors.
4. **Section bounds**: `section_count ≤ scene_count`, per-doc `section_count` within a band of today's distribution, a max single-section spoken-duration cap, and **never 0 inter-section consent gates** unless the doc is genuinely single-section today.
5. Anchors strictly increasing with first=0.0 / last=1.0 (post-build assertion); all 240 regenerated docs load through `HistoryModeController._load_document` without raising; existing `tests/test_history_content_build.py` + `tests/test_history_mode.py` stay green.

## Consequences

**Positive**
- Both r3 content defects (narration↔image sync, consent-prompt pacing) are fixed at their shared root — the upstream scene structure — without any runtime code change or runtime dependency.
- The Jetson device, `core/`, and CI remain fully offline/key-free, consistent with the local-first product principle; the cloud dependency is confined to a one-shot ~$72 dev-PC build step.
- The OVERLAY design plus the figure-preservation and bucket-conditional invariants bound the blast radius: curated narration, timings, and figures cannot silently change.

**Negative / risks**
- A new external API + key enters the build toolchain (dev-side only). Mitigated: one-shot run, local response cache (re-runs are free and offline), committed runtime docs as the reproducible baseline, env-var-only key handling.
- Korean heading misclassification cuts BOTH ways: missed `heading1`s are handled by the `heading_count < 2` heuristic fallback + per-doc telemetry, and the live smoke test showed **over-detection** (dialogue quotes / narrative sentences returned as `heading1`) → a deterministic pre-filter (quote-wrapped / sentence-final / over-long headings dropped; font-size as tie-break) runs before sectioning; fallback-bucket docs ship today's behavior, never worse.
- Anchor coverage is an uplift metric, not a total fix: multi-figure-per-segment scenes and cross-scene-referent figures fall back to even spacing by design.
- The retired `history_image_anchors.py` anchor pass must not be re-run over these docs (it would overwrite the title-aware anchors); the regeneration flow in `history_pdf_parse.py` is the sole pipeline.

**Alternatives rejected / deferred**
- *Keep the heuristic-only pipeline (no Upstage)* — rejected: both r3 defects root in the upstream scene JSON (page-attached figures, first-sentence pseudo-titles); no downstream heuristic can recover real headings or figure→sentence linkage that was never extracted.
- *Regenerate figures from Upstage `base64_encoding=["figure"]` crops* — **deferred to v2**: reusing the existing `fig_NNN.jpg` files is lower-risk and is enforced by the figure-preservation invariant; crop replacement would expand the blast radius for no current defect.

## References
- Methodology (authoritative): `Dev_Plan/2026-06-10-history-upstage-pdf-parse-methodology.md` (v4; discussion records `…-discussion-r1.md`, `…-discussion-r2.md`; Codex r3 APPROVE WITH NOTES).
- Pipeline code: `scripts/build_history_content.py`, `scripts/history_image_anchors.py` (`resolve_scene_anchors` helper reused; heuristic anchor pass retired for these docs), `core/history_mode.py` (`_narration_segments`, `_select_scene_image_by_progress`).
- Upstage Document Parse: console.upstage.ai/docs (document-parsing), upstage.ai/pricing (Standard $0.01 / Enhanced $0.03 per page); live smoke-test evidence 2026-06-10 (`.upstage_cache/3994fac….json`, local) — response `model: document-parse-260128`, billed Standard, rate-limit header `X-Upstage-Ratelimit-Limit-Pages: 300`/min.
- Related ADRs: 0107 (`재미있는 우리역사` curated-content player — the mode whose content this regenerates), 0108 (sibling curated-mode scope precedent).
- Memory/policy: `feedback_no_api_key_without_approval` (user-approved `UPSTAGE_API_KEY` usage).
