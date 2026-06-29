# ADR 0111: Pre-rendered TTS Cache for 재미있는 우리역사 + 퍼니 잉글리시 (16 kHz / step 30; live synthesis retained as validated fallback)

## Status

Accepted

(Implemented and merged to `dev` via PR #225 — `models/tts_cache.py` loader, history/FE wiring, and
`scripts/bake_tts_cache.py` — with normalization hardening in PR #226 and #227. The full on-device P4
bake completed 2026-06-17 (26,504 / 26,504 entries, `skipped_error=0`, `missing_wav=0`); P5 on-device
verification PASSED the same day. Plan:
`Dev_Plan/2026-06-16-history-fe-prebaked-tts-cache-plan.md` v4, converged through Codex 3-round
review — r1 6/6 ACCEPT → r2 2/2 ACCEPT → r3 1 residual PM-closed at the 3-round cap; user pre-approved
all actions 2026-06-16. Two residuals remain open and are tracked in **Consequences → Follow-ups**: the
FE Aesop (stage 6/7) English cards added later by PR #229 are not yet in the cache, and the GPU bake
path defect is deferred. Amended 2026-06-20 to include two explicitly enumerated fixed approved-template
responses: Mungi child self-intro and adult product-intro.)

- **Date**: 2026-06-16 (ratified after P5 verification 2026-06-17)
- **Decision owner**: Claude Code (primary orchestrator) + user direction (2026-06-16: pre-render ALL
  history + Funny English spoken text; play pre-baked WAV in the study modes; zero negative impact on the
  live system as a hard gate)
- **Related**: ADR 0100 (voice mung-ee v2 spec — the cached voice identity), ADR 0107 (curated history
  player + content-filter scope), ADR 0108 (Funny English read-along player + shared renderer), ADR 0099
  (boot-persistent kiosk runtime — the service the bake interlock brackets)

## Context

Live Korean narration in `재미있는 우리역사` (history) and `퍼니 잉글리시` (Funny English) under-converged
("mushy" / dropped syllables) at the runtime TTS `total_steps`, and a census found the runtime sentence
splitter failed to break on `종결부호 + 닫는따옴표` (`!”`, `.”`, …): **1,082 / 3,606 scenes (30%), 1,932
boundaries, 239 / 240 docs** read as run-on phrasing. These two modes speak a **fixed, pre-authored**
inventory (no free-form LLM output), so their spoken text is fully enumerable ahead of time.

On-device measurement (mung-ee voice) showed 16 kHz ≈ 44.1 kHz on the mono speaker, and a 20→32 step
sweep selected `total_steps=30`. Rendering every utterance at step 30 live is too slow for a responsive
child experience (CPU RTF@30 ≈ 0.49), but the inventory is fixed — so it can be **pre-rendered once** and
played back instantly, while the live synthesis path is retained unchanged as a fallback.

Three constraints shaped the design:

1. **Zero negative impact on the live system (hard gate).** The cache must never make any outcome worse
   than today: a miss or a corrupt entry must always fall through to a known-good live path, and the bake
   must not destabilize the resident runtime or the kiosk.
2. **LLM-generated conversation TTS must stay uncached.** The free-form generated response path still
   routes directly through live synthesis and never reads `tts_cache`. As of the 2026-06-20 amendment, the
   shared fixed-response path may do a validated cache lookup only for fixed, pre-authored strings that are
   explicitly enumerated in the bake inventory; a miss falls through to the same live-synthesis path.
3. **Content/voice drift must invalidate stale audio.** A changed model, voice, speed, step count, format,
   or text must not silently serve outdated audio.

## Decision

Adopt a **read-only, content-addressed, pre-rendered TTS WAV cache** for the history and Funny English
study modes, with live synthesis retained as a validated multi-tier fallback.

1. **Inventory (pre-baked).** History scene narration (per `_narration_segments` unit, `None` pause
   sentinels skipped), the 240 history lead-ins, the 2 consent prompts, the 8 fixed FE Korean utterances,
   and the FE English model words (`FunnyEnglishCard.text`) — all at **mung-ee voice, lang per item,
   speed = 0.95, total_steps = 30, 16 kHz / 16-bit / mono PCM**. The two mode-entry confirmations and the
   SoundBank cues are **excluded** (spoken via the conversational fixed-response path / already pre-recorded).

2. **Content-hash key.** `sha256(schema_version | engine_id | model_id | voice_id(lang) | lang | speed |
   steps | format | sample_rate | normalized_text)`, where `model_id` is the Supertonic model-dir hash and
   `voice_id` is the mung-ee JSON sha256. Any of these changing invalidates the cache. Identical texts dedup
   by key.

3. **Validated hit (not path-present).** `models/tts_cache.py::lookup(text, lang) -> Path | None` returns a
   path only if the cache is **enabled** (runtime identity matches `cache_meta.json`), the manifest entry
   exists for the key and language, the WAV file exists, its header parses (mono / 16-bit / 16 kHz, frames
   > 0), its byte length matches the manifest, and its sha256 matches. Any mismatch is treated as a miss.
   The loader performs **no synthesis**.

4. **Defence-in-depth live fallback.** `history_mode` and `funny_english_mode` are cache-first-then-live:
   a valid hit is loaded and played; a miss **or** an invalid/corrupt hit falls through to the existing live
   `tts.synthesize`. FE English uses a 3-tier order: validated cache WAV → committed `card.model_audio_path`
   WAV → live English TTS. A bad cache entry never errors and never plays garbage.

5. **Cache is scoped to fixed, pre-authored text only.** `tts_cache.lookup` is invoked from
   `core/history_mode.py`, `core/funny_english_mode.py`, and the fixed-response helper in
   `core/pipeline.py`. The pipeline helper is cache-first only for deterministic fixed responses, including
   the two 2026-06-20 approved-template additions; the generated-response `_run_tts` path remains direct live
   synthesis and continues to avoid `tts_cache`.

6. **Resumable bake.** `scripts/bake_tts_cache.py` collects the in-scope texts, and for each key not already
   `status=done` with a valid WAV: synth (step 30) → 16 kHz resample → temp-write + fsync + atomic rename →
   manifest append. Per-item synthesis failure is isolated (logged, item left not-`done` so a resume retries
   it) so one bad item never aborts the run. Persistent writes occur **only** under
   `/var/lib/mungi/tts_cache/` (`<key>.wav` + `manifest.json` + `cache_meta.json`).

7. **Unsupported-character normalization (PR #226 / #227).** `normalize_tts_text` is the single choke point
   feeding both the cache key and live synthesis: it strips/normalizes CJK glosses and bare ideographs,
   separators (`· ㆍ • ・ ∼ ▲`), `℃ → 도` (number expansion preserved), Hiragana/Katakana/PUA/placeholders,
   and empty brackets. `SupertonicEngine.synthesize` additionally strips any char outside the engine's
   `supported_chars` as a final safety net, so arbitrary live LLM input can never crash synthesis.

8. **2026-06-20 approved-template amendment.** Extend the bake inventory to the two fixed Korean
   approved-template responses `mungi_self_intro_child` and `mungi_product_intro_adult`. These strings are
   authored content, not model output, and are loaded from `assets/filters/approved_templates.json` so the
   router response and the baked cache source stay byte-equivalent. This preserves the invariant that
   LLM-generated conversation TTS is never cached.

## Implementation deviations from the plan (recorded)

- **ADR number 0101 → 0111.** The plan reserved "ADR 0101", but 0101 was already taken
  (`0101-crisis-distress-mandatory-escalation.md`). Reassigned to the next free number, **0111** (highest
  existing ADR was 0110).
- **Bake ran on CPU, not GPU.** GPU bake (`--device cuda`) does not reach the Supertonic ONNX providers and
  falls back to CPU; this defect is **deferred**, not fixed (CPU wall-time ~26 h was acceptable for a
  one-time bake). See Follow-ups.
- **Bake run directly via `bake_tts_cache.py`, not the sudo-gated `_run.sh` safe runner.** With the kiosk
  intentionally stopped and a CPU-only bake, the GPU interlock (the `ConditionPathExists` drop-in + lock
  bracket in plan §6) was moot, and the device's narrow passwordless-sudo scope made the runner
  impractical. The interlock design is retained in the plan/runner for any future GPU bake.

## Consequences

### Positive
- History and FE study modes play instantly at full step-30 quality with no live-synthesis latency.
- The phrasing/segmentation defect is fixed at the source (`_split_text_into_sentences` closer-aware split)
  and baked once.
- The hard "never worse than today" gate holds: validated-hit + multi-tier fallback means a miss or
  corruption degrades to live synthesis, never to an error or garbage audio.
- The conversation pipeline is provably unaffected (cache scoped to two call sites; guard test).

### Negative / costs
- ~6.4 GB of device storage under `/var/lib/mungi/tts_cache/` (gitignored, device-local).
- Any change to the voice, model, step count, or content requires a re-bake of the affected keys (mitigated
  by the content-hash key auto-invalidating stale entries → safe live fallback until re-baked).

### Follow-ups (open)
- **FE Aesop English cards (PR #229).** The 15 new `stage 6/7` (Hare & Tortoise / Lion & Mouse) English
  cards were merged after the P4 bake started and are **not yet cached** (inventory enumerates 26,519 vs
  26,504 cached). They currently resolve via live fallback; a device deploy + incremental re-bake will fold
  them in (`fe_en` 24 → 39).
- **GPU bake path.** `--device cuda` → Supertonic provider reach is deferred; revisit only if CPU bake
  wall-time becomes unacceptable.

## Verification (P4 + P5, 2026-06-17)
- **P4 bake**: process completed cleanly (~26 h CPU); manifest 26,504 / 26,504 `done`; 26,504 WAV files
  (`missing_wav=0`); `skipped_error=0`, `error_sources=[]`; cache 6.4 GB.
- **P5 on-device**: cache `enabled=True` (runtime identity matches); history (lead-in / scene / consent) and
  FE (KO / EN) lookups all HIT with non-silent valid WAV; a synthetic miss returns `None` (live fallback);
  `lookup()` call-site audit confirmed the generated conversation path never read the cache at the time of
  ratification; representative samples exported for subjective listening; kiosk restored to
  `active (running)`. The 2026-06-20 amendment supersedes this call-site scope only for enumerated fixed
  approved-template responses, not for generated conversation TTS.

## References
- Plan: `Dev_Plan/2026-06-16-history-fe-prebaked-tts-cache-plan.md` (v4)
- Runbook: `docs/runbooks/2026-06-16-tts-cache-bake-runbook.md`
- Code: `models/tts_cache.py`, `scripts/bake_tts_cache.py`, `core/history_mode.py`,
  `core/funny_english_mode.py`, `core/pipeline.py`, `models/tts_runner.py`
- PRs: #225 (cache + bake), #226 (CJK normalization + skip-on-error), #227 (unsupported-char normalization
  + synth safety net), #229 (FE Aesop stages — pending re-bake)
- ADRs: 0100, 0107, 0108, 0099
