# ADR 0039: Production Model Cleanup and Formal Acceptance of Qualitative Selection Override

- **Status**: Accepted
- **Date**: 2026-04-05
- **Decision makers**: Claude Code PM, maintainer
- **Related**: ADR 0034, ADR 0036, ADR 0038

## Context

During the 2026-04-05 afternoon session, a routine "production model deployment verification" task on Jetson (`mungi@jetson.local`) revealed three related governance issues that needed formal resolution:

### Issue 1: Documentation inconsistency between two model selection reports

Two separate reports dated 2026-04-05 contained **different production model selections**:

| Dimension | Runbook (quantitative) | Dev_Plan Report v1.0 (qualitative) |
|---|---|---|
| File | `docs/runbooks/weekly/archive/2026-04-05-12model-comparison-report.md` | `Dev_Plan/Mungi_Model_Selection_Report_v1.md` |
| Git status | Committed (`9c8cff2`, 2026-04-05 02:04 KST) | Untracked (`??`) |
| Generation | Auto via `scripts/generate_e2e_report.py` | Manual |
| Evaluation criteria | p50/p95 latency, peak memory, success rate, honorific rate, unique response count | Factual accuracy, safety, English retention, hallucination avoidance |
| Bilingual consideration | None | Yes (English retention axis) |
| Safety incident weighting | None | Yes (Q8_0 lightning misinformation, Q4_K_M Ariadne Oliver hallucination) |
| 1st rank | **Q35 Q4_K_M DPO** (p95 7.46s, Peak 3788 MB) | **Q35 Q6_K DPO** (Best factual + safety) |
| 2nd rank | **Q3 Q6_K SFT** | **Q35 Q6_K SFT** (Q35 Q6_K FT) |
| 3rd rank | Q35 Q4_K_M SFT | Q35 Q4_K_M DPO |

### Issue 2: Broken ADR cross-references

ADRs 0034, 0036, and 0038 already reference `Dev_Plan/Mungi_Model_Selection_Report_v1.md` as the authoritative source for the production model decision:

- **ADR 0034** (line 3): "Status: Superseded by Mungi_Model_Selection_Report_v1.md (2026-04-04)"
- **ADR 0036** (line 88): "Superseded By: ADR 0038 + Mungi_Model_Selection_Report_v1.md (2026-04-04)"
- **ADR 0038** (line 80): "Dev_Plan/Mungi_Model_Selection_Report_v1.md: Production model decision"

However, the referenced file had **never been committed to git**. Anyone following these cross-references would encounter a missing source. The override decision was recorded informally in ADR text but without the backing document.

### Issue 3: Jetson model directory sprawl

The actual Jetson deployment matched the Dev_Plan Report selection (`Qwen3.5-2B-DPO.Q6_K.gguf` was loaded as production), but the `/opt/mungi/ai_models/` directory contained 13 GGUF files totaling 17.3 GB:

- Top-level `Qwen3.5-2B-DPO.Q6_K.gguf` (1.5 GB) — real file, duplicated (same SHA256) with a subfolder copy
- `qwen35-2b/` subfolder: 6 Q35 variants (9.1 GB total)
- `qwen3-1.7b-compare/` subfolder: 6 Q3 variants (8.2 GB total)

Two of the Q35 Q8_0 variants (DPO and FT) contained documented safety-dangerous misinformation per Dev_Plan Report Appendix B.

### Issue 4: Stale CLAUDE.md section 3

CLAUDE.md section 3 listed `Qwen3-4B-Q4_K_M.gguf` as the active model, but that file was neither present on Jetson nor consistent with the auto-discovery exclusion list in `models/llm_runner.py:_EXCLUDED_GGUF`.

## Decision

### D1: Formally accept the Dev_Plan Report v1.0 selection over the runbook ranking

**Production model**: `Qwen3.5-2B-DPO.Q6_K.gguf` (Q35 Q6_K DPO)
**Backup model (2nd rank)**: `Qwen3.5-2B-FT.Q6_K.gguf` (Q35 Q6_K SFT)

**Rationale for the override** (from Dev_Plan Report §2.2, §2.3, §3, Appendix B):

1. **Factual accuracy is baked into model weights**. Emotional warmth can be tuned via system prompts after deployment, but hallucinations cannot. Select for what cannot be changed later.
2. **Q4_K_M quantization degrades English retention** in 2B-class models. With the bilingual architecture (ADR 0038) introduced on the same day, English quality became a primary evaluation axis that the runbook did not weigh.
3. **Q4_K_M also exhibits the "Ariadne Oliver" hallucination** (Agatha Christie character presented as an elephant biology fact), which is a child-safety concern.
4. **Q8_0 variants produce dangerous safety misinformation** ("rubbing your body protects from lightning", "elephants eat butter"). Both Q35 Q8_0 SFT and DPO produced byte-identical outputs, suggesting a training pipeline error. These models are categorically excluded.
5. **Q3 (Qwen3-1.7B) variants are all eliminated** per Dev_Plan Report §2.1: gibberish output, topic injection, severe factual errors, and hallucinations across all six combinations.

The runbook's quantitative ranking was **not wrong** — it correctly identified Q35 Q4_K_M DPO as the fastest model with lowest memory and 100% success rate within its measurement axes. But those axes do not capture factual accuracy, safety, or bilingual English quality, which are decisive for a child-facing product.

### D2: Commit `Dev_Plan/Mungi_Model_Selection_Report_v1.md` to resolve broken cross-references

The report has been referenced by three ADRs (0034, 0036, 0038) since its creation but was never committed. It will be committed in the same changeset as this ADR so that all existing cross-references resolve.

### D3: Jetson model directory cleanup

**Target structure**:
```
/opt/mungi/ai_models/
├── Qwen3.5-2B-DPO.Q6_K.gguf → qwen35-2b/Qwen3.5-2B-DPO.Q6_K.gguf   (symlink, auto-discovered)
└── qwen35-2b/
    ├── Qwen3.5-2B-DPO.Q6_K.gguf   (1.5 GB, canonical 1st rank)
    └── Qwen3.5-2B-FT.Q6_K.gguf    (1.5 GB, 2nd rank, auto-discovery excluded)
```

**Symlink rationale**: The top-level duplicate (separate inode with identical SHA256) wasted 1.5 GB. Replacing it with a symlink to the canonical subfolder copy retains `find_gguf_model()` auto-discovery behavior (`Path.glob("*.gguf")` resolves symlinks to real files) while eliminating the duplicate storage.

**2nd rank in subfolder rationale**: `models/llm_runner.py:290` uses `model_path.glob("*.gguf")` which is **non-recursive**. Keeping the backup model in `qwen35-2b/` subfolder prevents accidental auto-selection; it requires manual loading.

**Deleted models** (14 GB freed):
- `qwen35-2b/Qwen3.5-2B-DPO.Q4_K_M.gguf` (3rd rank with hallucinations)
- `qwen35-2b/Qwen3.5-2B-DPO.Q8_0.gguf` (dangerous safety — excluded)
- `qwen35-2b/Qwen3.5-2B-FT.Q4_K_M.gguf` (factual errors)
- `qwen35-2b/Qwen3.5-2B-FT.Q8_0.gguf` (dangerous safety — excluded)
- `qwen3-1.7b-compare/` entire directory (6 files: Q4_K_M, Q4_K_M-dpo, Q6_K, Q6_K-dpo, Q8_0, Q8_0-dpo — all eliminated per Dev_Plan Report §2.1)

### D4: Update CLAUDE.md section 3 to match reality

Section 3 "Baseline Technical Stack" has been updated to:
- Pipeline item 3: `Qwen3.5-2B DPO Q6_K via llama.cpp (active model: Qwen3.5-2B-DPO.Q6_K.gguf, symlinked to qwen35-2b/Qwen3.5-2B-DPO.Q6_K.gguf)`
- Inactive models subsection: restructured into "Backup model (2nd rank)" and "Removed models (deleted from Jetson 2026-04-05)" with per-model deletion rationale and cross-reference to the Dev_Plan Report.

### D5: Correct date references in existing ADRs

ADR 0034 and ADR 0036 reference the Dev_Plan Report as "2026-04-04". The actual file version header states "Version 1.0 | 2026-04-05" and the file system timestamp is 2026-04-05 02:40 KST. The `(2026-04-04)` annotations in ADRs 0034 and 0036 will be corrected to `(2026-04-05)` as part of this changeset.

## Alternatives Considered

### A1: Redeploy Q35 Q4_K_M DPO per the runbook's quantitative ranking

**Rejected.** Would satisfy formal process but:
- Introduces English retention degradation that conflicts with the bilingual architecture launched the same day (ADR 0038)
- Reintroduces the "Ariadne Oliver" factual hallucination
- Does not address the safety incidents recorded in Q8_0 variants (which the runbook ranking also did not explicitly weigh)
- Discards two days of Dev_Plan Report v1.0 analysis work

### A2: Re-run the 12-model comparison with factual accuracy and English retention scoring axes added

**Rejected for this cycle.** Would be methodologically cleaner, but:
- Re-running 12 × 60 rounds × 2 languages on Jetson consumes days of test time
- Phase 0 timeline does not permit this delay
- The qualitative analysis in Dev_Plan Report v1.0 is already documented and was the basis for the actual Jetson deployment and the downstream ADR 0038 bilingual architecture decision
- Deferred to Phase 1 as part of the standardized evaluation rubric redesign

### A3: Keep all 12 models on Jetson for A/B testing

**Rejected.** 17.3 GB of model files on Jetson:
- Leaves known-dangerous models (Q8_0 variants) available for accidental loading
- Complicates backup strategy (no clear 2nd rank designation)
- Wastes 14 GB disk for models with documented elimination reasons
- A/B testing can be done from the Colab + HuggingFace side without Jetson storage

### A4: Delete the Dev_Plan Report entirely and restore the runbook's Q35 Q4_K_M DPO selection

**Rejected.** Same reasons as A1, plus the report contains useful qualitative analysis (safety findings, hallucination catalog) that should not be discarded even if the selection decision were to change.

## Consequences

### Positive

- **Single source of truth**: Dev_Plan Report v1.0 committed, ADR cross-references resolve.
- **14 GB disk reclaimed** on Jetson (67 GB → 53 GB, from 30% to 24% usage).
- **Safety hardening**: Q8_0 variants with documented dangerous misinformation removed from Jetson — cannot be accidentally loaded.
- **Clear backup strategy**: 2nd rank model preserved in subfolder, immune to auto-discovery drift.
- **CLAUDE.md consistency**: Section 3 matches actual runtime state, eliminating stale-doc confusion for future sessions.
- **Override justification recorded**: Future reviewers can see why the quantitative ranking was overridden.

### Negative

- **Process deviation from generate_adr_draft.py workflow**: This ADR was written directly by the PM (Claude Code) per explicit user instruction, bypassing the standard script-generated draft workflow defined in CLAUDE.md §8. This is a one-time override for a documentation-governance ADR where the PM has full session context.
- **Runbook auto-generator remains quantitative-only**: `scripts/generate_e2e_report.py` will continue to produce rankings that do not account for factual accuracy, safety, or English retention. Future runbook-driven decisions must be cross-checked against qualitative criteria.
- **2nd rank backup is manual-only**: Requires explicit configuration change to load. No automatic fallback on 1st rank failure.

### Risks

- **If `find_gguf_model()` is changed to recursive glob in the future**, the 2nd rank file in `qwen35-2b/` would become auto-discoverable and the selection could become ambiguous. Mitigation: add a recursive-discovery guard test in `tests/` (deferred, logged as follow-up).
- **Symlink portability**: If the repository or deployment mechanism ever copies `/opt/mungi/ai_models/` to another host without preserving symlinks, the production model path would break. Mitigation: deployment scripts must use `rsync -L` or equivalent to dereference symlinks, OR must preserve them explicitly.
- **Evaluation axis gap persists**: The next model evaluation cycle will still produce quantitative-only runbooks unless the scoring rubric is extended. Logged as Phase 1 task.

## Implementation Record

| Step | Action | Verification |
|---|---|---|
| 1 | Gate 2 load test for 2nd rank model (`Qwen3.5-2B-FT.Q6_K.gguf`) | GPU full offload load + inference OK after `drop_caches` |
| 2 | `mv` top-level DPO.Q6_K to `.pre-symlink` backup | File preserved for rollback |
| 3 | `ln -s qwen35-2b/Qwen3.5-2B-DPO.Q6_K.gguf Qwen3.5-2B-DPO.Q6_K.gguf` | `readlink -f` resolves correctly, SHA256 matches |
| 4 | `find_gguf_model()` discovery re-test | `IS_SYMLINK: True`, loads on GPU, inference OK |
| 5 | Remove `.pre-symlink` backup | 1.5 GB reclaimed |
| 6 | Delete 4 files from `qwen35-2b/` + entire `qwen3-1.7b-compare/` directory | 14 GB total reclaimed |
| 7 | Final `find_gguf_model()` + Gate 2 re-verification | Discovery returns symlinked path, load OK |
| 8 | CLAUDE.md section 3 update via Codex delegation (task `claude-md-sec3-model-update`) | 2-round Codex self-verification + Claude Code 2nd filter PASS |
| 9 | CLAUDE.md orphan label cleanup via Codex delegation (task `claude-md-remove-orphan-label`) | Structure cleaned, grep verification PASS |

## Follow-ups (tracked separately)

- **CLAUDE.md §2 config path discrepancy**: `/var/lib/mungi/config/config.json` is documented but does not exist on Jetson. Needs §2 update or filesystem creation. (Not in this ADR's scope.)
- **Runbook rubric extension**: Add factual accuracy, safety, and bilingual English axes to the auto-generated comparison report. Phase 1 task.
- **Recursive glob regression test**: Add `tests/test_llm_runner_discovery.py` case to ensure `find_gguf_model` remains non-recursive so the 2nd rank backup cannot be auto-selected. Low priority.

## Related

- `Dev_Plan/Mungi_Model_Selection_Report_v1.md` (Version 1.0, 2026-04-05) — authoritative source for model selection
- `docs/runbooks/weekly/archive/2026-04-05-12model-comparison-report.md` — quantitative runbook (superseded for production selection, retained as evaluation record)
- `docs/runbooks/weekly/archive/2026-04-05-model-cleanup-worklog.md` — session log for the cleanup operation
- `docs/archived/adr-superseded/0034-qwen35-feasibility-evaluation.md` — superseded (status updated)
- `docs/adr/0036-dpo-evaluation-q4-q6.md` — superseded (status updated)
- `docs/adr/0038-bilingual-mode-architecture.md` — bilingual mode depends on this model decision
- `models/llm_runner.py` (DEFAULT_MODEL_DIR, \_EXCLUDED_GGUF, find_gguf_model)
- `CLAUDE.md` section 3 — baseline technical stack
