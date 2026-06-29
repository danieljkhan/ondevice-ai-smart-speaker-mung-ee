# ADR 0078 - LLM generation-config layering: caller-explicit wins, backend defaults fill

- **Status**: Accepted (validated)
- **Date**: 2026-04-27
- **Decision owner**: Claude Code (primary orchestrator) + user approval
- **Related**: `core/pipeline.py` (`PipelineConfig`,
  `_apply_llm_backend_generation_config`), `core/llm_backend_config.py`
  (`LLMBackendConfig`), ADR 0073 (Gemma primary swap), ADR 0076 (L1 LLM
  resident default), Plan v4
  (`Dev_Plan/2026-04-25-llm-primary-gemma4-swap-plan.md`) section 3.3
  follow-up #2 and section 10 R9

## Context

PR #49 (ADR 0073) made `gemma4_text` the default LLM backend. To apply Gemma's
validated generation defaults (`max_tokens=256`, `temperature=0.4`) instead of
`PipelineConfig`'s legacy defaults (`max_tokens=80`, `temperature=1.0`),
`ConversationPipeline.__init__` ran `_apply_llm_backend_generation_config()`
(`core/pipeline.py:666-669` pre-fix), which **unconditionally overwrote**
`self._config.llm_max_tokens` and `self._config.llm_temperature` with the
backend's defaults whenever Gemma was active.

This caused a real bug surfaced in Plan v4 section 3.3 follow-up #2 and section
10 R9: any caller passing explicit generation values, such as
`PipelineConfig(llm_temperature=0.15, llm_max_tokens=200)`, had those values
silently discarded under the Gemma default backend. PR #49 worked around this
in two test fixtures (`tests/test_hallucination_fix.py`,
`tests/test_sprint3_core.py:test_sampling_controls_forwarded`) by explicitly
patching the backend to `qwen3_legacy`, but the fix was test-only and the
underlying production bug remained.

Audit also found three production scripts passing explicit generation values
silently overwritten on Gemma:

- `scripts/demo_live.py:81-85` - `llm_max_tokens=80, llm_temperature=0.2`
- `scripts/e2e_60rounds_text_tts.py:716-724` - `args.llm_max_tokens`
  (CLI default 128)
- `scripts/e2e_bilingual_test.py:260-269` - inherits text+TTS parser

Plan v4 section 3.3 #2 filed for an architectural fix at the layering level.

## Decision

Adopt **caller-explicit wins; backend defaults fill only when unset** (Design F
per Plan v3 alternatives table).

Implementation:

1. **Sentinel marker** at module level in `core/pipeline.py`:

   ```python
   class _UnsetSentinel:
       __slots__ = ()

       def __repr__(self) -> str:
           return "_UNSET"

   _UNSET: Final[_UnsetSentinel] = _UnsetSentinel()
   ```

2. **PipelineConfig generation field defaults switch to sentinel**, public types
   remain concrete `int`/`float`:

   ```python
   llm_max_tokens: int = field(default=_UNSET)  # type: ignore[assignment]
   llm_temperature: float = field(default=_UNSET)  # type: ignore[assignment]
   _llm_max_tokens_explicit: bool = field(init=False, default=False, repr=False)
   _llm_temperature_explicit: bool = field(init=False, default=False, repr=False)
   ```

3. **`PipelineConfig.__post_init__` resolves sentinels + sets explicit flags**
   (merged into existing `__post_init__` at `core/pipeline.py:474-510`, which
   already handled `MUNGI_DROP_CACHES_PER_TURN`):

   - If field is `_UNSET`: resolve to legacy default (`_resolve_llm_max_tokens()`
     for max_tokens; `1.0` for temperature) and set `_llm_*_explicit = False`.
   - Else: caller passed explicit value; set `_llm_*_explicit = True`.
   - After `__post_init__`, public fields are concrete `int`/`float` matching
     their type annotations.

4. **`ConversationPipeline._apply_llm_backend_generation_config()` becomes
   conditional fill**:

   ```python
   if not self._config._llm_max_tokens_explicit:
       self._config.llm_max_tokens = self._llm_backend_config.max_tokens
   if not self._config._llm_temperature_explicit:
       self._config.llm_temperature = self._llm_backend_config.temperature
   ```

## Layering ownership (binding)

- **`PipelineConfig`** owns **caller / request intent** (per-pipeline overrides).
  Public fields are concrete `int`/`float` post-`__post_init__`.
- **`LLMBackendConfig`** owns **backend selection + implicit backend defaults**
  with env / config.json / code-default precedence chain.
- **`ConversationPipeline`** owns **resolution** of the effective runtime
  generation config: caller-explicit wins, backend default fills, both backends
  honored symmetrically.

## Future backend addition checklist

When adding a new LLM backend, such as `gemma5_text` or `phi4_text`:

1. Define backend-specific defaults in `LLMBackendConfig` if validated values
   differ from `PipelineConfig`.
2. Add explicit-caller-wins test mirror (mirror of
   `test_pipeline_explicit_generation_config_wins_under_gemma_default`) using the
   new backend.
3. Add implicit-backend-default fill test mirror (mirror of
   `test_pipeline_implicit_generation_config_filled_from_gemma_backend`).
4. Verify legacy default preservation still passes
   (`test_pipeline_implicit_generation_config_preserves_legacy_defaults` remains
   green).
5. Confirm typed access works at all generation call sites without new
   `# type: ignore` annotations.

## Consequences

- **Caller intent honored**: production scripts (`scripts/demo_live.py`,
  `scripts/e2e_60rounds_text_tts.py`, `scripts/e2e_bilingual_test.py`) and test
  fixtures passing explicit generation values now flow through to the LLM call
  regardless of active backend.
- **Backend defaults preserved when caller silent**: pipelines constructed with
  `PipelineConfig()` (no kwargs) still receive Gemma's validated defaults under
  default Gemma backend; legacy backend still uses `PipelineConfig`'s pre-ADR
  defaults.
- **No public API change**: external readers (existing tests at
  `tests/test_pipeline.py:442-479, 501-510`; logs at
  `scripts/e2e_qwen3_asr_mix.py:1114`) see concrete `int`/`float` exactly as
  before. Sentinel never escapes `__post_init__`.
- **E2E baseline shift**: production scripts that pass explicit `llm_max_tokens`
  / `llm_temperature` previously had their values silently overwritten by Gemma
  defaults (256 / 0.4); post-ADR-0078 they receive their explicit values. **E2E
  latency / quality baselines collected pre-ADR-0078 are NOT directly comparable
  to post-ADR-0078 baselines**; operators must re-baseline. Affected:
  `scripts/demo_live.py` (max_tokens 256 -> 80, temperature 0.4 -> 0.2),
  `scripts/e2e_60rounds_text_tts.py` (max_tokens 256 -> 128 or CLI override),
  `scripts/e2e_bilingual_test.py` (same).
- **Internal mypy concession**: 2 `# type: ignore[assignment]` comments at the
  sentinel-default field declarations (`field(default=_UNSET)` against
  `int`/`float` annotations). No `# type: ignore` at any of the 8 generation call
  sites.

## Alternatives considered

- **Design A (public sentinel union)**: PipelineConfig fields typed as
  `int | _UnsetType`. **Rejected** because it leaks sentinel to external readers
  (Codex Round 1 BLOCK 1: `tests/test_pipeline.py:442-479` direct field access;
  `scripts/e2e_qwen3_asr_mix.py:1114` pre-init log).
- **Design B (`__post_init__` introspection)**: track explicit fields via
  dataclass introspection. **Rejected** because there is no clean way to detect
  "was this set explicitly" without sentinel; harder to test.
- **Design C (backend wins, current pre-fix behavior)**: simple but bug-prone;
  the bug being fixed.
- **Design D (`PipelineConfig.from_backend(backend)` factory)**: factory with
  backend-aware defaults. **Rejected** because it only helps callers using the
  factory and does not fix existing direct `PipelineConfig(...)` callers without
  broader refactor.
- **Design E (drop overwrite + change PipelineConfig defaults to Gemma's)**:
  simplest code change but causes legacy regression: `qwen3_legacy` callers'
  default max_tokens move from 80 to 256; default temperature from 1.0 to 0.4.
- **Design G (separate effective-config object)**: cleaner separation but heavier
  refactor; deferred for now.

## Validation

This ADR is validated once the implementation PR
(`feature/llm-generation-config-overwrite-fix`):

1. Passes the new
   `tests/test_pipeline.py::TestADR0078GenerationConfigLayering` (3 tests
   covering explicit-Gemma-wins, implicit-Gemma-fills,
   implicit-legacy-preserves).
2. Passes the un-wrapped
   `tests/test_hallucination_fix.py::test_parameter_propagation_to_llm_runner`
   and `tests/test_sprint3_core.py::test_sampling_controls_forwarded` under
   default Gemma backend (proving the bug is fixed for the original
   explicit-caller use case).
3. Passes `tests/test_sprint3_core.py::test_max_tokens_forwarded` with the new
   value 200 (proving the bug-detection sensitivity that was hidden by
   256-coincidence).
4. No regression in the full pytest suite (>=0% coverage threshold).
5. Pre-existing direct-field-access tests
   (`tests/test_pipeline.py:442-479, 501-510`) and external readers
   (`scripts/e2e_qwen3_asr_mix.py:1114`) work unchanged.

Status flips from `Proposed` to `Accepted (validated)` post-merge via a separate
small `[docs] ADR 0078 status sync` PR following the ADR 0072 / 0073 / 0076
precedent.

## Update - 2026-04-27 - Implementation validated

- Implementation PR: #55 (`feature/llm-generation-config-overwrite-fix` -> `dev`), squash-merged as commit `625f0ec`.
- Plan: `Dev_Plan/2026-04-27-llm-generation-config-overwrite-design-plan.md` (v3, post-Codex 3-round review cycle: R1 PUSH BACK 6 BLOCK + 3 NOTE all ACCEPTED -> R2 PUSH BACK 2 BLOCK + 5 cleanup + 3 hardening all ACCEPTED -> R3 APPROVE AS-IS). See `Dev_Plan/2026-04-27-llm-generation-config-overwrite-design-plan-discussion-round{1,2}.md` and `*-codex-review-round{1,2,3}.md`.
- Implementation artifacts shipped (11 files, +1417 / -6):
  - `core/pipeline.py` - `Final` import + `_UnsetSentinel` class + `_UNSET` module-level singleton + `PipelineConfig.llm_max_tokens` and `llm_temperature` switched to `field(default=_UNSET)` + 2 NEW private fields `_llm_max_tokens_explicit` / `_llm_temperature_explicit` + merged `__post_init__` (preserves existing `MUNGI_DROP_CACHES_PER_TURN` env override) + `_apply_llm_backend_generation_config()` becomes conditional fill.
  - `tests/test_hallucination_fix.py` - un-wrap qwen3_legacy `LLMBackendConfig.load` patch; change `@patch` target from `models.llm_runner.run_generation` to `run_chat_generation` with 6-tuple return value; rename mock variable to `mock_chat_gen`.
  - `tests/test_sprint3_core.py` - `test_max_tokens_forwarded` value 256->200; `test_sampling_controls_forwarded` qwen-wrap removed.
  - `tests/test_pipeline.py` - NEW `class TestADR0078GenerationConfigLayering` with 3 tests covering explicit-Gemma-wins / implicit-Gemma-fills / implicit-legacy-preserves.
- Validation outcomes against original Validation criteria:
  1. `tests/test_pipeline.py::TestADR0078GenerationConfigLayering` - 3/3 PASS (explicit Gemma 200/0.15 preserved; implicit Gemma 300/0.55 backend-fill; implicit legacy 80/1.0 preserved).
  2. `tests/test_hallucination_fix.py::test_parameter_propagation_to_llm_runner` PASS under default Gemma backend (un-wrapped); `tests/test_sprint3_core.py::test_sampling_controls_forwarded` PASS under default Gemma (un-wrapped).
  3. `tests/test_sprint3_core.py::test_max_tokens_forwarded` PASS with new value 200 (proves bug-detection sensitivity that was hidden by the 256-coincidence pre-fix).
  4. Full pytest suite: 997 passed / 17 skipped / 0 failed / coverage 79.56% (`core/pipeline.py` 90%); >=0% threshold satisfied.
  5. Pre-existing direct-field-access tests (`tests/test_pipeline.py:442-479, 501-510`) and external readers (`scripts/e2e_qwen3_asr_mix.py:1114`) continue to PASS - no public API regression. Sentinel never escapes `__post_init__`; public fields are concrete `int`/`float` post-construction.
- Operator follow-up: per ADR 0078 Consequences (E2E baseline shift), production scripts `scripts/demo_live.py`, `scripts/e2e_60rounds_text_tts.py`, `scripts/e2e_bilingual_test.py` previously had explicit `llm_max_tokens` / `llm_temperature` silently overwritten on Gemma. Post-ADR-0078 those values flow through. E2E latency / quality baselines collected pre-ADR-0078 are NOT directly comparable to post-ADR-0078 baselines; operators must re-baseline.
- Runbook: `docs/runbooks/weekly/archive/2026-04-27-daily-worklog.md` (Session 16 entry).
