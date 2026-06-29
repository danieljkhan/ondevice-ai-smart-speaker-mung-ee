# ADR 0107: 재미있는 우리역사 — Curated-Content History Player and Content-Filter Scope

## Status

Proposed

(Will move to Accepted after the `재미있는 우리역사` plan converges through Codex review, the mode is implemented + verified, and it ships. Plan: `Dev_Plan/2026-06-08-jaemiittneun-uri-yeoksa-plan.md`.)

## Context

`재미있는 우리역사` is a new on-device, offline Korean-history **picture-storytelling player** for children under 10, built from the curated dataset `assets/dataset_korean history/` (240 documents → 3,606 scenes → 1,865 images). Each scene pairs a pre-authored, child-friendly Korean `narration` string with one or more images; the mode shows the image on the 720×720 panel and speaks the narration via Supertonic TTS, advancing on tap.

Two architectural questions require a decision recorded here:

1. **It is a new runtime path** — a non-conversational content player layered on the session machine, with its own `HISTORY_*` states, a new renderer text/image layer, and a deterministic entry matcher in `safety/`. CLAUDE.md §7 rule 4 / §1 require an ADR for new runtime-path + safety-family changes.
2. **Content-filter scope** — the conversational pipeline routes free-form LLM output through `safety/content_filter.py` (ContentFilter), crisis/parent-disclosure routers, and approved-template gates. ADR 0098 made the live ContentFilter activation **fail-closed** (a missing/broken filter blocks output rather than passing it). The history player plays **pre-curated historical narration**, not LLM-generated or user-derived text. The product owner (user) has directed that this curated historical narration is **historical fact, requires no age-softening or content filtering, and is license-confirmed**, and that it should be played verbatim.

## Decision

### 1. New mode is a curated-content player (no generation)
`재미있는 우리역사` runs as a fixed-content player: it never invokes the LLM and never runs STT for navigation (tap-only). The only speech-recognition it uses is the **deterministic entry/exit trigger phrase** matcher (`safety/history_mode_router.py`, whole-turn-anchored, mirroring the language-switch matcher) — the narration body is never user-derived.

### 2. Curated narration bypasses the conversational content filter — by scope, not by disabling it
The historical `narration` is played verbatim and **does not pass through** `_filter_text` (ContentFilter), the crisis/parent-disclosure routers, or the approved-template gate. This is a **scope decision**, explicitly bounded:

- The conversational pipeline's ContentFilter remains **unchanged and fail-closed** (ADR 0098 is not weakened): every LLM turn still goes through it. This ADR does **not** disable, relax, or alter that filter for conversation.
- The bypass applies **only** to the new curated-history-player path, whose content source is a **pre-authored, human-curated, license-confirmed educational dataset** — not user input and not model output.
- **There is no path by which a child's utterance becomes narrated content.** The child can only (a) trigger the mode by a fixed phrase and (b) tap to navigate. No free-text from the child is ever synthesized or displayed.
- Therefore the safety surface of this mode is reduced to a single question — *"is the curated dataset content appropriate?"* — which is owned by **content curation + the product owner's directive**, not by a runtime filter designed for unbounded LLM output.

### 3. Product-owner directive (recorded)
The user, as product owner of this non-commercial / academic-PoC track, has directed that the historical narration is treated as historical fact, is **not age-softened and not content-filtered**, and that licensing is user-confirmed. This ADR records that directive and the bounded scope under which it applies. Burn-in text on some dataset images is handled as a **display/layout** concern only (a `clean` flag steering where Mungi's character sits), never as content removal.

## Consequences

**Positive**
- Children can hear the full historical record (no censoring), consistent with the educational intent and the product owner's directive.
- The conversational safety architecture (ADR 0098 fail-closed ContentFilter, crisis/parent routers) is untouched and still governs all LLM output.
- The mode's safety surface is small and well-bounded (no generation, no user-derived content, tap-only navigation).

**Negative / risks**
- The runtime filter no longer acts as a backstop for this content path; correctness depends on **dataset curation** and the product owner's judgment. Mitigation: the dataset is pre-authored child-friendly narration; provenance/licensing is user-confirmed; the mode performs no generation.
- A future decision to expand the player to non-curated or generated content would **invalidate this scope** and require a new ADR (the bypass is justified *only* because the content is curated + non-generated).

**Alternatives rejected**
- *Route narration through ContentFilter* — rejected per the product-owner directive (historical fact needs no filtering) and because the filter is tuned for unbounded LLM output, not curated educational text (risk of false-positive censoring of legitimate history).
- *Age-soften the narration at build time* — rejected per directive; would alter the historical record.

## References
- Plan: `Dev_Plan/2026-06-08-jaemiittneun-uri-yeoksa-plan.md`
- Related: ADR 0098 (runtime ContentFilter activation fail-closed — preserved), `safety/content_filter.py`, `safety/language_switch_router.py` (matcher precedent).
