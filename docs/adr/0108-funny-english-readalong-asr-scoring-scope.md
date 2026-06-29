# ADR 0108: Funny English — Curated English Read-Along Player, ASR-Scoring Scope, and Shared Renderer

## Status

Proposed

(Will move to Accepted after the `Funny English` plan converges through Codex review, the mode is implemented + verified, and it ships. Plan: `Dev_Plan/2026-06-08-funny-english-plan.md`; pedagogy: `Dev_Plan/2026-06-08-funny-english-pedagogy-research.md`.)

## Context

`Funny English` is a new on-device, offline English **read-along learning mode** for Korean children under 10 learning English as a foreign language (EFL). The child sees a target word/phrase + image, hears a Text-To-Speech model voice, reads aloud, and the device gives gentle, encouraging feedback. It is built natively on Mungi's stack and **reimplements the published learning science of Google Read Along without copying any Google IP** (no Diya character/brand/UI/art/sound, no Google corpora/word-lists/levels, no Google ASR/pronunciation models). Pedagogy ideas/methods are not copyrightable, are documented publicly by Google, and are independently validated by academic literature (see References).

Four decisions require recording here.

1. **New runtime path.** Funny English is a non-conversational learning player layered on the session machine, with its own `FE_*` states (`FE_SELECT → FE_PROMPT → FE_LISTEN → FE_SCORE → FE_FEEDBACK → FE_DONE`), a deterministic entry matcher in `safety/`, and a per-prompt STT-scoring loop. CLAUDE.md §7 rule 4 / §1 require an ADR for new runtime-path + safety-family changes.
2. **Content-filter scope** (parallels ADR 0107). The conversational pipeline routes free-form LLM output through `safety/content_filter.py` (ContentFilter, made fail-closed by ADR 0098), crisis/parent-disclosure routers, and approved-template gates. Funny English generates no LLM output: the displayed/spoken targets are a **pre-authored, license-clean curriculum** (Dolch sight words — Public Domain; synthetic-phonics CVC — authored; Stage-5 readers — per-title-verified CC-BY/CC0), and the child's spoken attempt is processed by **STT only**, for coarse word-matching against the known card target.
3. **Pedagogy decision — coarse ASR is a participation/approximation gate, not a pronunciation grade.** Qwen3-ASR is transcription-only (no phoneme/GOP scoring) and is unreliable on children's L2 speech. Both engineering reality and the learning-science literature point the same way for this age group: replace accuracy-grading with **encouraging re-modeling (recasts)**.
4. **Shared renderer primitive** (shared with ADR 0107). Both Funny English and `재미있는 우리역사` need a text + full-screen-image rendering layer in `core/character_renderer.py` (which has none today). To avoid two conflicting paths, ONE generic primitive layer is defined; the history mode (ahead in the pipeline) **lands and owns** it; Funny English **consumes** it.

## Decision

### 1. New mode is a curated read-along player (no generation)
Funny English never invokes the LLM. It plays a fixed curriculum and listens to the child read. The only speech recognition is (a) the **deterministic entry/exit trigger-phrase** matcher (`safety/funny_english_router.py`, whole-turn-anchored, mirroring the language-switch matcher), and (b) the **per-prompt coarse word-match** scoring of the child's read-aloud against the card's known target tokens. No free text from the child is ever generated, displayed, or synthesized as content.

### 2. Curriculum content and STT-scoring bypass the conversational content filter — by scope, not by disabling it
- The conversational ContentFilter remains **unchanged and fail-closed** (ADR 0098 preserved); every LLM turn still passes through it. This ADR does not disable, relax, or alter it.
- The bypass applies **only** to the Funny English path. Its content source is a **pre-authored, license-confirmed educational curriculum**, not user input or model output. The child's spoken attempt is consumed by STT for an approximate word-match decision and is **never** routed to `_filter_text`/crisis/parent/approved-template gates or to the LLM. (Child read-aloud audio is still stored under `/var/lib/mungi/conversations/` per the standing data policy, as a per-prompt learning record for parental review — not as generated content.)
- Therefore the mode's safety surface reduces to **"is the curated curriculum appropriate and license-clean?"** — owned by content curation + the build-time NOTICE/license gate (`scripts/build_funny_english_content.py` fails the build if **any non-original asset** lacks `license/source/title/author/notice` metadata or carries an **unknown/non-whitelisted** license, and it **bundles the full OFL/MIT/CC license texts**), not by a runtime filter designed for unbounded LLM output.

### 3. Coarse ASR = participation gate; feedback is recast-by-re-modeling (pedagogy, recorded)
- The scorer maps the ASR result to lenient bands (PASS / CLOSE / LOW / SILENT-JUNK), never a numeric score shown or spoken to the child.
- On any non-match, the mascot **warmly re-models** the correct word/phrase via TTS and invites a retry (a recast), following an **"I do / we do / you do"** gradual-release loop with a **fading hint ladder**. After a bounded number of supportive retries the mode models once and advances warmly (never stuck, never punished). No buzzer / red-X / failure state is ever shown.
- This is the honest engineering choice (coarse ASR cannot adjudicate phonemes) AND the pedagogically optimal one for EFL beginners under 10 — the two align. Grounded in: comprehensible input + affective filter (Krashen); ZPD/scaffolding (Vygotsky); gradual release (Pearson & Gallagher); corrective-feedback/recast research (Lyster & Ranta lineage); and the three papers in References.

### 4. Shared renderer primitive — history owns, Funny English consumes
The generic API (`show_image`/`clear_image`/`show_text`/`clear_text`/`_get_font` + `core/_tap_gesture.py`, all Pygame on the UI thread, lock+pending+wake pattern) is landed and owned by the history runtime impl (ADR 0107 mode). The landed renderer renders image XOR text; because Funny English's dual-coded prompt needs an image and a word **together**, FE extends the SAME generic layer with a composite super-primitive `show_card(...)` (image region + text region in one render branch; history's image-only/text-only become degenerate cases). FE does **not** build a second/FE-only text/image path. Landing order: history first (already merged, #175).

## Consequences

**Positive**
- Children get encouraging, low-anxiety read-along practice consistent with reading science and Mungi's safety-first product vision; the device's weakest capability (pronunciation grading) is deliberately de-emphasized exactly where the literature says it should be.
- The conversational safety architecture (ADR 0098 fail-closed ContentFilter, crisis/parent routers) is untouched and still governs all LLM output.
- The mode's safety surface is small and well-bounded (no generation, curated content, STT used only for coarse matching, license gate at build time).
- One renderer primitive serves two modes — no duplicated/conflicting Pygame paths.

**Negative / risks**
- ASR accuracy on children's L2 English is the dominant risk; mitigated by per-prompt hotword biasing + lenient set-match/Levenshtein + a participation-attempt floor + a timeout-then-gentle-cue window + Hangul-junk → encourage. Needs on-device tuning.
- Correctness of the content path depends on **curriculum curation + the build-time license gate**, not a runtime filter. Mitigation: PD/CC-BY/CC0/OFL/MIT whitelist; the NOTICE gate requires `license/source/title/author/notice` for every non-original asset, fails on unknown/non-whitelisted licenses, and bundles the full license texts.
- Expanding the mode to generated or non-curated content would invalidate this scope and require a new ADR.

**Alternatives rejected**
- *Route the child's read-aloud or the curriculum through ContentFilter/LLM* — rejected: the child's attempt is a known-target word-match, not free conversation; the curriculum is pre-curated; routing through an unbounded-output filter risks false-positive censoring and adds no safety.
- *Grade pronunciation with the ASR (phoneme/GOP feedback)* — rejected: Qwen3-ASR is transcription-only and unreliable on children's L2 speech; misgrading would be technically dishonest and affectively harmful (raises the affective filter). Re-modeling replaces grading.
- *Two independent renderer text/image paths (one per mode)* — rejected: duplicated Pygame code and divergence risk; one shared primitive (history owns) is the decision.

## References
- Plan: `Dev_Plan/2026-06-08-funny-english-plan.md`; pedagogy synthesis: `Dev_Plan/2026-06-08-funny-english-pedagogy-research.md`.
- Academic grounding: Mostow et al. (2003), *Evaluation of an automated reading tutor that listens*, JECR 29(1) (+ Poulsen et al. 2007, JECR 36(2), L2 corroboration); Soleimani, Mohammaddokht & Fathi (2022), *Assisted repeated reading… in an EFL context*, Frontiers in Psychology 13:851812; Ehri et al. (2001), *Phonemic awareness instruction…*, RRQ 36(3) (NRP meta-analysis).
- Related ADRs: 0107 (curated-history player, content-filter scope, shared renderer — sibling), 0098 (runtime ContentFilter fail-closed — preserved), 0106 (KO-EN `한영전환` — reused for EN mode + KO coaching).
