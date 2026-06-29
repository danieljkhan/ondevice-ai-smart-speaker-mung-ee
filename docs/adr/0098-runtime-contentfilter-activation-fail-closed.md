# ADR 0098: Runtime ContentFilter activation with fail-closed startup

- **Status**: Accepted (2026-05-31 — user final-plan approval of the code-integrity remediation plan; Codex plan review r3 APPROVE WITH NOTES; F1 implemented + code re-review r3 PASS, 수정사항 0건)
- **Date**: 2026-05-31
- **Authority**: `Dev_Plan/2026-05-31-code-integrity-remediation-plan.md` (v3) + `Dev_Plan/2026-05-31-code-integrity-remediation-discussion-r1.md` (Codex r1→r3 convergence)
- **Related**: `safety/content_filter.py` (the filter implementation), `assets/filters/blocklist.json` + `assets/filters/patterns.json` (policy assets), CLAUDE.md §1 (product vision: "the safest AI friend"), §6 (safety scope), §8 (safety role)
- **Memory refs**: `feedback_adr_immutability` (this Decision is amended only via an Update section, never the original)

## Context

A Codex code-integrity review (`mungi-code-review-integrity-r1`, reviewer role) found a BLOCK-severity safety gap (finding F1): the dedicated child-safety `ContentFilter` (`safety/content_filter.py`, unit-tested) was **never instantiated in the live runtime**.

PM validation confirmed it:
- `ContentFilter(` is instantiated nowhere in non-test code.
- The live pipeline factories (`core/session_manager.py` `_default_pipeline_factory`, `scripts/demo_live.py` `_make_pipeline_factory`) constructed `ConversationPipeline` without a `content_filter=` argument.
- `ConversationPipeline._filter_text()` returns `None` when `enable_content_filter` is `True` but `_content_filter is None`, and the input/output guards treat `None` as "allowed" → **fail-open**: child-facing input and LLM output flowed unfiltered despite `enable_content_filter=True`.

For a product whose first principle is child safety, a silently-inert content filter is unacceptable. (Other defense layers — persona safety module, output sanitization — remained active, but the dedicated keyword/pattern filter was bypassed.)

## Decision

### D1 — Wire ContentFilter into the live runtime

The live pipeline factories MUST build and inject a loaded `ContentFilter` when `PipelineConfig.enable_content_filter` is `True`. A `ContentFilter.from_default()` classmethod constructs the filter against the canonical `assets/filters/` policy and eagerly calls `.load()`.

### D2 — Fail-closed contract: hard-fail at startup

When content filtering is enabled, the filter is loaded **eagerly in the factory at construction time**. If the policy assets are missing or fail to load, the exception **propagates and the device refuses to start** (hard-fail at startup).

Rejected alternatives:
- **Fail-open** (the prior behavior): unacceptable — a misconfigured device would serve children unfiltered with no signal.
- **Block-every-turn with safe fallback**: inferior — it degrades every turn to a fallback (an input block returns fallback text with no audio) while masking the misconfiguration instead of surfacing it loudly at startup.

A misconfigured safety device must fail loudly and refuse service, not silently degrade.

### D3 — Preserve `_filter_text` None semantics for non-live callers

`_filter_text` keeps returning `None` when no filter is injected, so library/unit-test callers that construct `ConversationPipeline` without a filter are unaffected. The fail-closed guarantee is enforced at the **live factory layer**, not inside `_filter_text`. A one-shot `logger.warning` fires if filtering is enabled but no filter instance is present (regression canary).

## Consequences

**Positive:**
- The child-safety content filter is now actually applied to live input and output.
- A device with missing/corrupt filter assets refuses to start rather than serving children unfiltered.
- Future regressions (filter not injected) are flagged by the one-shot warning and the live-wiring tests.

**Negative / risk:**
- Startup now surfaces filter-load errors that were previously masked. Deployment MUST verify `assets/filters/blocklist.json` + `patterns.json` are present on the Jetson before cutover.

**Scope note:**
- Benchmark / E2E harness scripts (e.g. `scripts/e2e_live_test.py`, `scripts/e2e_60rounds_text_tts.py`, `scripts/run_pr5_100_voice.py`) construct `ConversationPipeline` without injecting the filter. These are developer/measurement tools, not the child-facing device path; their non-injection is intentional and out of scope for this ADR. The child-facing paths are `SessionManager._default_pipeline_factory` and the demo-live factory (both covered by D1).

## Verification

- `tests/test_live_content_filter_wiring.py`: live factory injects a loaded filter; factory raises on filter load failure (hard-fail); input/output blocked paths return the safe fallback; a no-filter pipeline still works (library path).
- Code re-review `mungi-code-review-integrity-r2`/`r3`: F1 confirmed RESOLVED.
- Implementing commit on branch `fix/code-integrity-remediation`.
