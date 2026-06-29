# ADR 0103 — Deterministic parent-disclosure guardrail + universal scope + Rule 5 belief-preservation

- Status: Proposed (pending implementation per Plan v4 + PM verification)
- Date: 2026-06-07
- Decision owner: Claude Code (primary orchestrator) + user final-plan approval (2026-06-07)
- Related: ADR 0072 (parent-disclosure invariance — this **supersedes its rejected mechanism alternatives** and **broadens its scope**), ADR 0101 (crisis matcher pattern — mirrored; `CRISIS_RESPONSE_*` left unchanged), ADR 0102 (E4B primary + E2B fallback — the mobile-quantization context), `safety/crisis_router.py`, `core/safety_rules.py`, `core/pipeline.py`, `Dev_Plan/2026-06-07-parent-disclosure-guardrail-gentle-tone-rule5-plan.md` (v4) + discussion records r1/r2/r3.

## Context

ADR 0072 made parent-disclosure invariance an absolute rule but enforced it **PROMPT-ONLY** (persona Rule 8) and explicitly **rejected** (its §"Alternatives considered and rejected") both (1) an output keyword-filter and (2) deterministic template routing — on two assumptions: *the model reliably follows the prompt*, and *post-processing breaks live latency*.

The 2026-06-07 multi-sample E4B A/B (`.codex/ab2_*`, `.codex/ab3_*`) refuted both assumptions:

- **Model-following refuted.** The 2-bit E4B-mobile quantization (a drop-in GGUF swap) VIOLATED the invariant — generated `우리 둘만의 비밀로 하자` and started disclosure probes with `응!` (prohibited prefix) — on the secret-pact suite; E4B-std held. A child-safety invariant cannot depend on which quantization is loaded.
- **Latency refuted (inverted).** A pre-LLM matcher *bypasses* generation → the matched path is faster, not slower; the post-LLM validator swaps in a fixed constant with no retry-generate.
- **Concrete gap found.** The current `PARENT_DISCLOSURE_KO_BLOCKERS` does NOT contain `둘만의 비밀`, so the existing assertion set would not catch the exact mobile failure phrase.
- **Rule 5 weak on mobile.** Santa/tooth-fairy probes drew evasive "잘 몰라" answers on the 2-bit model.

A deterministic guardrail prototype (`.codex/pd_guardrail_proto.py`) drove mobile's invariant violations **3 → 0** with responses **identical to std**, at **matcher FN=0 / FP=0** over 22 probes (incl. benign secrets and mixed belief+secret-pact utterances).

## Decision

Adopt a **deterministic, model-independent parent-disclosure guardrail**, superseding ADR 0072's rejected mechanism alternatives for this validated failure mode, and **broaden the invariant scope to UNIVERSAL**.

### Mechanism
1. **Pre-LLM matcher** (`safety/parent_disclosure_router.py`, sibling of `crisis_router.py`): two-tier — **explicit parent non-disclosure PROBE (FP-immune)** > **bare-relational FRIENDSHIP (benign-play FP-vetoed)** → fixed gentle response (probe vs friendship branch per ADR 0072), bypassing the LLM. Plus a **narrow belief matcher** (direct Santa/tooth-fairy/Easter-bunny "is it real?" probes) → fixed warm affirming response.
2. **Ordering**: crisis (highest) → explicit PROBE → bare FRIENDSHIP → belief. A secret-pact embedded in a belief probe routes to parent-disclosure, never the belief reply; an explicit probe fires even when a benign-play noun co-occurs.
3. **Post-LLM output validator** (production code in `core/`): replaces any response containing a secrecy-promise (expanded blocker set, incl. the `둘만의 비밀` gap) or a prohibited yes/no prefix in a secret-pact context, with the fixed PROBE response.
4. **Gentle tone**: firm-core + warm-softener constants — never start yes/no, never a blocker phrase, always name parent/trusted-adult; no hard `꼭 ~할 거야` command. `CRISIS_RESPONSE_*` (ADR 0101) are NOT softened.
5. **Rule 5 hardening**: persona-prompt strengthening + the narrow deterministic belief matcher; belief responses affirm the wonder warmly without doubt-casting / make-believe framing, and without asserting a falsehood as fact (Rule 2 honesty preserved).

### Scope change (supersedes ADR 0072 §"Scope boundary")
ADR 0072 limited the invariant to **dangerous-topic threads** (Rule 3 categories). **ADR 0103 broadens it to UNIVERSAL**: the matcher fires on ANY secret-pact / "don't tell my parents" probe regardless of triggering topic.
- **Rationale**: grooming and abuse secret-pacts are frequently **keyword-free** — the secret-pact itself is the danger signal; topic-gating would miss the highest-risk cases.
- **FP control**: bounded + harmless (a gentle, parent-positive reply); explicit benign-play FP guards (`비밀기지/암호/놀이`, `깜짝 선물`, `secret code/fort`) exclude clear play; bare-relational secret-pacts fire by design; benign-FP rate monitored via control fixtures.

### Retained from ADR 0072
The **policy** (parent involvement non-negotiable; probe-vs-friendship branch; never start yes/no; never promise secrecy) is retained and **strengthened**. Only the **mechanism** (deterministic vs prompt-only) and **scope** (universal vs dangerous-topic-only) change.

## Consequences

### Positive
- Parent-disclosure invariant is now **model-independent** — survives any LLM/quantization swap; unblocks E4B-mobile re-evaluation (the −1GB lever).
- Closes the 2-bit regression class and the `둘만의 비밀` blocker gap.
- Matched path is faster (LLM bypass); deterministic, testable contract (AC1–AC10).

### Negative / trade-offs
- Universal scope raises the benign false-positive rate (bounded, harmless gentle reply; monitored).
- Persona byte-identity fixtures regenerate on the Rule 5 + Rule 8 text change (PM regenerates + reviews).
- A new module + JSON template to maintain alongside `crisis_router`.

### Alternatives reconsidered
- **ADR 0072 prompt-only** — refuted by the A/B (model-dependent).
- **Output-filter only** (no pre-LLM matcher) — insufficient; still lets the LLM be asked and rely on scrubbing. The pre-LLM matcher + validator defense-in-depth is cleaner.
- **Topic-gated scope** (keep ADR 0072 boundary) — rejected; misses keyword-free grooming/abuse, the highest-risk class.

## Implementation reference
Tracked in `Dev_Plan/2026-06-07-parent-disclosure-guardrail-gentle-tone-rule5-plan.md` (v4, converged r1→r3, PM-closed at the 3-round cap). Validated prototype: `.codex/pd_guardrail_proto.py`. Implementation artifacts (router, templates, constants, pipeline wiring, prompts, tests) update this Status line when merged.

## Validation
Validated once the implementation PR:
1. Passes AC1–AC10 (Plan v4), incl. pipeline reachability (LLM bypass, ordering, `_return_fixed_tts_response`, metrics, history), the production validator, the FP-override (AC10), and universal scope (AC7).
2. Reruns the guardrail A/B regression with mobile invariant violations = 0.
3. Keeps `CRISIS_RESPONSE_*` byte-unchanged; crisis_router + ADR 0101 tests green; full `pytest` + `ruff`/`mypy` clean.
4. Jetson live re-check of a secret-pact + belief + mixed chain.

Until all are satisfied, Status = Proposed (policy commitment only).
