# ADR 0094 — STT fine-tuning track adoption — child speech data stack on Colab Pro+

- **Status**: **Proposed** (validation criteria to be MET via downstream Gate 1 Plan execution + first fine-tune evaluation)
- **Date**: 2026-05-20 (Session 57)
- **Decision owner**: Claude Code (primary orchestrator) + user direction (2026-05-20)
- **Related**: ADR 0055 (Qwen3-ASR adoption — STT engine unchanged), ADR 0076 (LLM-resident memory + 4.50 s TTFT invariant — must be preserved post fine-tune), ADR 0089 (docs-only commit abbreviation — applies to this ADR + feasibility/manual commit), ADR 0090 (Confirmable-fact grounding — STT misrecognition is upstream of fact grounding and worsens its inputs)

## Context

Session 56 (2026-05-20) anchor-focused E2E acceptance test ran 24 Typecast-synthesized child voice fixtures (KO 12 + EN 12) through the production pipeline with `MUNGI_FACT_SHORTLIST=p2` default. Of 24 turns, 6 failed grounded response delivery specifically due to STT misrecognition (F-3):

- **Korean** (4 cases): consonant weakening (`뭉이야 → 니야`, `한글 → 그걸 / 나`), space-loss compound (`조선 왕은 → 조선왕은`).
- **English** (3 cases): proper-noun mishears (`insect → tech`, `blue whale → blue list`, `cheetah → cheat`).

The Session 56 fixtures were synthetic adult-voice TTS (Typecast Hobin/Ella). Real child users will produce phonetic patterns further from Qwen3-ASR's training distribution (smaller vocal tract, higher pitch, immature articulation, L2 accent for English). The F-3 failure surface is therefore expected to widen — not narrow — under production deployment.

Three F-3 mitigation paths were identified in the Session 56 close handoff:
1. **Hotword tuning** (`MUNGI_QWEN3_ASR_HOTWORDS`) — bounded to known anchor words; cannot address general phonetic drift.
2. **Matcher trigger variants** — bounded to confirmable-fact topics; addresses LLM-side robustness on a fixed shortlist, does not improve STT accuracy.
3. **STT fine-tuning** — the only mitigation that targets the STT layer directly and addresses the root cause class.

## Problem

STT misrecognition is upstream of every downstream pipeline component (matcher, LLM, persona, content filter, TTS). A character/word error in STT propagates to:
- Confirmable-fact matcher failures (Phase D §3.6 design assumes correct lexical anchors).
- LLM input distribution shift (Korean grammatical drift, English noun-substitution).
- Inappropriate response delivery (the user's actual question is not received).

For a child-targeted device where the user-facing experience must be the "safest first AI friend" (vision §1), STT accuracy on actual child voices is a non-negotiable quality dimension. Continuing with the stock Qwen3-ASR-0.6B INT8 model — trained primarily on adult speech — leaves a structural quality gap that grows under real deployment.

## Decision

**Adopt STT fine-tuning as a project track. Fine-tune Qwen3-ASR-0.6B on a child-speech-focused data stack centred on AI-Hub Korean + Korean-L2-English corpora. Train on Google Colab Pro+ using parameter-efficient (LoRA) methods. Do not change the STT engine; do not augment with native English open-source data.**

Detailed sub-decisions:

1. **Engine preserved.** Qwen3-ASR-0.6B (ADR 0055 active) remains the sole STT engine. Fine-tuning produces a derived bundle deployed through the same sherpa-onnx INT8 path. The engine choice is out of scope for this ADR.

2. **Data stack — AI-Hub trio:**
   - **AI-Hub 95** — 명령어 음성(소아·유아), 3,000 h, command domain, multi-device + multi-noise → **primary KO**.
   - **AI-Hub 71502** — 어린이 방송 음성(EBS/KBS), 1,001.70 h → **supplemental KO** (broadcast register, outdoor/home subset selectable).
   - **AI-Hub 541** — 학습용 아동 영어 음성, 5,000 h, Korean children's L2 English → **primary EN**.
   - **Total: 9,001.70 hours.** Subsampling policy (~800–1,150 h) deferred to Gate 1 Plan.

3. **Native English open-source augmentation: DEFER.** AI-Hub 541's Korean L2 English distribution directly matches Mungi's actual user population. Native English child corpora (MyST, OGI Kids, CMU Kids) train phonetic patterns Mungi users do not produce. MyST retained as the single optional augmentation candidate if production data later surfaces non-Korean-accent EN demand.

4. **Training infrastructure: Google Colab Pro+** (user decision 2026-05-20). A100 40 GB / 80 GB / H100 with background execution. Jetson Orin Nano 8 GB is non-viable for training. Data staging via KR-region Google Drive.

5. **Method: LoRA (parameter-efficient) first.** Reduces catastrophic-forgetting risk on Qwen3-ASR's general multilingual ASR capability. Full fine-tune only if LoRA results are insufficient. Specific configuration (rank, target modules, learning schedule, KO+EN joint vs sequential, subsampling policy) is the Gate 1 Plan's responsibility.

6. **Scope boundary — this ADR adopts the TRACK, not the codebase change.** The output of this track is a validated sherpa-onnx INT8 bundle (the implementation manual ends there). **Mungi codebase modifications (`/opt/mungi/ai_models/` swap, `models/stt_runner.py` adjustment, Session 56 anchor fixture re-validation, new STT-bundle ADR Update) require a separate Gate 1 Plan with Codex review + mutual-discussion + user final approval per CLAUDE.md §1.**

7. **Parallel-track coordination.** The Mungi E2E latency-reduction track runs concurrently with this STT track (user direction 2026-05-20). Coordination invariants (baseline tagging, STT-runtime commit prefix, joint integration landing) are recorded in the feasibility report §8 and are normative for both tracks.

## Alternatives considered

1. **Hotword tuning only (`MUNGI_QWEN3_ASR_HOTWORDS`)** — DEFERRED, not rejected. Effective for known anchor words but cannot address general phonetic drift on unknown vocabulary. Remains in scope as an operational tuning lever; does not substitute for fine-tuning at the layer level.

2. **Matcher trigger-variant augmentation** — DEFERRED, not rejected. Bounded to confirmable-fact topics. Addresses LLM-side robustness on a fixed shortlist; does not improve STT input quality for general dialogue.

3. **Replace STT engine** — REJECTED. ADR 0055 committed Qwen3-ASR-0.6B as the sole engine (Update 2026-04-29 removed SenseVoice fallback). Switching engines would invalidate Phase D / Session 56 acceptance evidence and is out of scope.

4. **Native English open-source as primary EN source (MyST + OGI Kids + CMU Kids)** — REJECTED. Phonetic distribution mismatch with Mungi's Korean L2 English user population. AI-Hub 541's 5,000 h of Korean children speaking English directly targets the actual deployment distribution.

5. **QLoRA on Wikidata / SimpleWiki for general ASR knowledge (analogous to ADR 0090 alternative #1)** — REJECTED. STT is a phonetic / acoustic adaptation problem, not a factual knowledge problem. Text-based knowledge augmentation does not address the F-3 failure class.

## Consequences

### Positive

- **Apache 2.0 base model** — Qwen3-ASR-0.6B HuggingFace checkpoint is permissively licensed; no engine-license blocker for commercial deployment.
- **Community fine-tune precedent** — 22 derivative fine-tunes exist on HuggingFace under `base_model:finetune:Qwen/Qwen3-ASR-0.6B`; proves technical feasibility.
- **Data stack volume sufficient** — 9,000+ hours dwarfs the published Whisper-on-MyST 400-hour fine-tune precedent (33–38% relative WER reduction); subsampling will be required, not augmentation.
- **Domain match — Mungi's Korean child + Korean L2 English population** — AI-Hub 95 command domain matches voice-AI device usage; AI-Hub 541 covers exactly the L2 English Mungi will receive in production.
- **Track decoupled from latency-reduction work** — different teams of resources (Colab vs Jetson), different timelines (data-application-bound vs sprint-cycle); see §8 coordination invariants.
- **Existing eval harness reusable** — Session 56 anchor fixture (24 WAV) + Phase D LLM-judge protocol can be reused directly.

### Negative / trade-offs

- **AI-Hub data application lead time** — days to weeks per dataset. Project velocity is data-acquisition-bound during this window.
- **Official fine-tune scripts absent** — Qwen3-ASR upstream README is inference-only. Training pipeline must be reconstructed from community derivatives or from scratch (Codex `feature` delegation candidate in Gate 1 Plan).
- **Multi-week effort** — Gate 1 Plan + applications + downloads + preprocessing + training + evaluation + sherpa-onnx re-export + Gate 1 Plan for integration → realistic horizon is multiple weeks to ~1–2 months.
- **Catastrophic forgetting risk** — fine-tuning a multilingual foundation model on a narrow distribution risks general-ASR regression. LoRA mitigates substantially; mitigation effectiveness must be measured.
- **Persona hotword (`뭉이,Moongee`) regression risk** — ADR 0055 baseline established KO 5/5 + EN 5/5 hotword recall. Fine-tune must preserve this.
- **TTFT invariant (ADR 0076 ≤ 4.50 s strict) — preserved by construction.** Model size is unchanged; only weights differ. Re-export INT8 quantization fidelity must be verified before any merge.

### Neutral

- **Parallel latency-reduction track confounding** — addressed deterministically by baseline tagging + STT-runtime commit prefix + joint final merge (feasibility §8). Does not reduce to a risk; it is a coordination protocol.

## Validation criteria (to promote this ADR to Accepted)

The ADR remains **Proposed** until ALL of the following are MET:

1. **Gate 1 Plan user approval** — a separate Plan covers LoRA configuration, subsampling policy, evaluation methodology, integration sequencing, risk register, Codex delegation specs. Plan undergoes Codex `reviewer` review + mutual-discussion cycle (up to 3 rounds) + user final approval per CLAUDE.md §1 Gate 1.
2. **Data acquisition complete** — AI-Hub 95 + 71502 + 541 applications approved, datasets downloaded, preprocessing complete, manifests packed.
3. **First fine-tune evaluation passes** — on the Session 56 anchor fixture (24 KO + EN), the fine-tuned model demonstrates meaningful CER/WER improvement over the stock Qwen3-ASR-0.6B INT8 baseline. Threshold (e.g., relative WER reduction ≥ X% on EN, CER ≥ Y% on KO) decided in the Gate 1 Plan.
4. **No persona-hotword regression** — `뭉이,Moongee` recall remains at the ADR 0055 baseline (KO 5/5, EN 5/5).
5. **No general-ASR catastrophic forgetting** — auxiliary held-out evaluation (e.g., LibriSpeech dev-clean 100 sample) shows acceptable bounded drift.
6. **sherpa-onnx INT8 re-export validated** — bundle loads via `OfflineRecognizer.from_qwen3_asr(...)` and produces equivalent or improved transcriptions vs the merged BF16 reference on the eval fixture.
7. **Mungi integration PR landed** — separate Gate 1 Plan executes the runtime swap; PR passes CI; user approves merge per CLAUDE.md §1 Gate 5.

Once 1–7 MET, an ADR Update appends the evidence and the ADR is promoted to **Accepted**.

## References

- `Dev_Plan/2026-05-20-stt-finetune-child-speech-data-feasibility.md` (feasibility report — the evidence base for this ADR)
- `Dev_Plan/2026-05-20-stt-finetune-implementation-manual.md` (Korean operational manual — execution path until the validated sherpa-onnx bundle)
- `docs/adr/0055-stt-engine-qwen3-asr-adoption.md` (current STT engine; Update 2026-04-29 removed SenseVoice)
- `docs/adr/0076-l1-llm-resident-default-mode.md` (4.50 s TTFT invariant — preserved through fine-tune)
- `docs/adr/0089-docs-only-commit-verification-abbreviation.md` (docs-only commit abbreviation — applies to feasibility + manual + this ADR commit)
- `docs/adr/0090-confirmable-fact-grounding-curated-shortlist.md` (STT misrecognition is upstream of fact-grounding inputs)
- `docs/runbooks/2026-05-20-session56-e2e-acceptance-report.md` (F-3 evidence — the precipitating finding)
- `docs/runbooks/weekly/2026-05-20-session56-close-handoff.md` (F-3 carry-forward → this track)
- `docs/runbooks/weekly/2026-05-20-session57-close-handoff.md` (this session's close)
- Qwen3-ASR HF model card: https://huggingface.co/Qwen/Qwen3-ASR-0.6B (Apache 2.0)
- AI-Hub datasets: 95 / 71502 / 541
- Whisper child-speech fine-tune literature: arXiv 2307.13008, 2309.07927, 2309.11756 (precedent for the method class)

---

## Update 2026-05-21 — Decision #2 (data stack) amended

**Trigger**: User direction during Gate 1 Plan v3 → v4 transition (`Dev_Plan/2026-05-21-stt-finetune-gate1-plan.md` v4). Drive 195.2 GiB free constraint observed; user opted for scope reduction over Drive upgrade.

**Original Decision #2 (preserved above)**: 3-dataset stack — AI-Hub 95 (3,000 h) + 71502 (1,001.70 h) + 541 (5,000 h) totaling 9,001.70 h.

**Amended Decision #2**: AI-Hub 95 AI-assistant filekey subset only. Specific filekeys: `43600`, `43603`, `43604`, `43605`, `43606`, `43607`, `43608`, `43609`, `43610`, `43611`, `43554`, `43557`, `43562` (13 filekeys). Estimated raw compressed ~50-80 GiB; estimated decompressed 16 kHz training corpus ~70-150 h.

**Datasets dropped from scope**:
- AI-Hub 71502 (broadcast children's speech) — out of scope for v4; potential re-introduction in a future data-stack-expansion track if production domain breadth becomes a concern.
- AI-Hub 541 (Korean L2 English children's speech) — out of scope for v4. **EN fine-tuning is postponed to a separate later track.** The Mungi Session 56 F-3 EN-side findings (`insect→tech`, `blue whale→blue list`, `cheetah→cheat`) will NOT improve from this Plan's deliverable.

**Validation criteria impact**:
- **Criterion 3 (KO/EN improvement)**: amended — KO CER target relative-reduction threshold relaxed from ≥20% to ≥15% (training corpus ~10% of original size). EN WER threshold (C3b) DROPPED. LibriSpeech dev-clean drift (C3e) ELEVATED to critical: it becomes the primary EN-side regression gate since no EN training is performed.
- **Criterion 5 (no catastrophic forgetting)**: tightened — without EN co-training, LoRA's parameter-efficient property must be the sole protection against EN-side regression. Two complementary gates apply (per Plan v4.1+v4.2): (a) **C3e LibriSpeech drift ≤ +2.0 pp absolute AND absolute WER ≤ 5.0%** (tightened from v3's +3.0 pp per Codex r4-1 A3); (b) **C3h Session 56 EN 12-utt anchor no-regression ≤ +1.0 pp absolute drift** (NEW in v4.1 per Codex r4-1 E2 — closer Mungi-domain-matched EN protection than LibriSpeech read US English). If EITHER gate fails, the fine-tune is rejected even if C3a passes.
- All other criteria (1, 2, 4, 6, 7) substantively unchanged.

**Rationale for accepting the amendment**:
1. **User-acknowledged trade-off**: user explicitly confirmed (2026-05-21) that postponing EN fine-tuning is acceptable. The product-vision impact is bounded (Mungi remains usable in KO; EN STT errors persist but are not blocking core function).
2. **Infrastructure constraint**: user's Google Drive plan is 225.8 GiB total / 195.2 GiB free, which cannot accommodate the 3-dataset stack's processed-data footprint (~280 GiB minimum including dual processed/shards rotation).
3. **Project velocity**: reduced scope shortens the data-acquisition + preprocessing + training cycle from a multi-month horizon to a several-week horizon, enabling earlier production validation of the KO improvement.
4. **Reversibility**: the dropped datasets remain on AI-Hub. A future data-stack-expansion Plan can re-introduce ds71502 or ds541 if production data identifies a need.

**Process compliance**:
- This Update preserves the original Decision-section text (ADR immutability per memory `feedback_adr_immutability`).
- CLAUDE.md §7 ADR requirement met: architecture-affecting change (data stack) documented before downstream Plan iteration finalizes.
- CLAUDE.md §1 Gate 1 process: Plan v4 + this ADR Update undergo Codex round-4 review + mutual discussion + user final M3 approval before any implementation commits.

**Cross-references for this Update**:
- `Dev_Plan/2026-05-21-stt-finetune-gate1-plan.md` v4 / v4.1 (the amended Plan)
- `.codex/specs/stt-finetune-gate1-plan-discussion-r1.md` (v3 round-1 PM responses)
- `.codex/specs/stt-finetune-gate1-plan-discussion-r2.md` (v3 round-2 PM responses)
- v3 round-3 had no formal discussion record — Codex explicitly authorized inline patches without a further round per its handoff (recorded only in §15 Changelog v3 entries; no `discussion-r3.md` file exists)
- `.codex/specs/stt-finetune-gate1-plan-discussion-r4-1.md` (v4 round-1 PM responses)
- `.codex/specs/stt-finetune-gate1-plan-discussion-r4-2.md` (v4 round-2 PM responses)
- `.codex/specs/stt-finetune-gate1-plan-review-v4-r4.md` (Codex v4 round-1 review spec)
- `.codex/specs/stt-finetune-gate1-plan-review-v4-r2.md` (Codex v4 round-2 review spec)
- `.codex/specs/stt-finetune-gate1-plan-review-v4-r3.md` (Codex v4 round-3 FINAL review spec)

---

## Update 2026-05-21 (second) — academic-context exemption for Validation criterion 3-related ToU artifact

**Trigger**: User direction during Plan v4.3 hotfix (`Dev_Plan/2026-05-21-stt-finetune-gate1-plan.md` v4.3) — "상용화 안할꺼야. 대학원 졸업프로젝트용이야" (this Plan iteration is a graduate-research project; not a commercial deployment).

**Amendment scope**: The validation criteria 1-7 list above remains structurally intact. The amendment narrows the operational implementation of the ToU verification gate:

- **Consequences §Positive "AI-Hub general policy permits commercial use of trained models with attribution"** — remains TRUE in policy terms, but no longer load-bearing for THIS Plan iteration's exit criteria. Plan v4.3 does NOT require commercial-use ToU evidence (R-14 retired in Plan v4.3 §9; ToU artifact `Dev_Plan/2026-05-21-aihub-tou-verification.md` dropped from M3 prereq).
- **What replaces it**: AI-Hub standard academic-use terms agreed at M4 application time (during the dataset application form). Plan v4.3 §3.5 Cell 12 records a non-commercial acknowledgement JSON log on Drive (`tou_acknowledgement.json`) for audit-trail traceability only — no PDF screenshot, no commercial-use language verification.

**Conditional re-introduction**: If the Mungi product's commercialisation posture changes later (e.g., the academic-output fine-tuned model is migrated to a commercial Mungi release), a follow-up Plan iteration must:
1. Re-introduce commercial-use ToU verification artifact for ds95 (and any datasets added back into scope at that point).
2. Append a third ADR 0094 Update revising this exemption.
3. Re-evaluate ADR 0094 Consequences §Positive attribution claim against the then-current AI-Hub policy.

**Process compliance**:
- ADR immutability preserved: original Decision body untouched; first Update (2026-05-21) untouched; this second Update appended only.
- CLAUDE.md §7 ADR requirement met: scope change (commercial → academic context) documented before Plan v4.3 commits.
- CLAUDE.md §1 Gate process: Plan v4.3 is a hotfix during M7 execution (not a new Plan iteration). Codex round skipped per memory `feedback_codex_round1_pushback_design` (this is a bug fix + user directive, not a contested design plan).

**Memory cross-references**:
- `project_academic_context` (NEW 2026-05-21) — STT fine-tune track의 직접 목적은 대학원 졸업 프로젝트; 상업화 미예정; ToU 검증 절차 면제

---

## Update 2026-05-22 (third) — data-acquisition architecture amended (Colab-direct → local-PC + Drive upload)

**Trigger**: Session 59 (2026-05-22) root-cause confirmation of Session 58 M7 download silent-failure. Diagnostic cell (single-filekey clean-dir aihubshell test) surfaced AI-Hub server response `"Download failed with HTTP status 502. Error msg: AI 허브는 해외에서의 데이터 다운로드를 제한하고 있습니다."` from Colab Pro+ VM (US/EU hosted). aihubshell returns exit code 0 even on HTTP 502 — `feedback_aihubshell_exit0_bug` captures the wrapping pattern (returncode + stdout regex + artifact size assertion). The geo-IP block is server-side: AI-Hub's listing API (`-mode l`) remains accessible from any IP, but the data-download endpoint (`-mode d`) is restricted to Korean IPs.

**Amendment scope**: The Decision body remains structurally intact. The amendment narrows the operational implementation of the **data acquisition path** (Decision #4 implicitly assumed Colab-direct `aihubshell` invocation; that assumption is now invalidated):

- **What was assumed (v4.x)**: Colab Pro+ VM directly invokes `aihubshell -mode d -datasetkey 95 -filekey ...` to fetch raw data into `/content/aihub_work/`, then archive to Drive `raw/`.
- **What replaces it (v5)**: Data acquisition is performed on the user's **Korean-IP local PC** using AI-Hub's official desktop download application. The resulting 13 ZIPs (1 script + 2 labels + 9 training source + 1 validation source, user-measured 176 GB compressed total) are uploaded **manually to Google Drive** by the user (Google Drive desktop sync app recommended). Colab consumes ZIPs from Drive (read-only) — Colab never invokes any AI-Hub API.
- **Reason this was not anticipated upstream**: AI-Hub's geo-IP restriction is not documented in `aihubshell -help` output, nor in the AI-Hub developer pages cross-referenced by the upstream Qwen3-ASR + sherpa-onnx documentation. The constraint surfaced only via the diagnostic stdout/stderr capture.

**Validation criteria impact**:
- **Criterion 2 (Data acquisition complete)**: operational path amended — "applications approved, datasets downloaded, preprocessing complete, manifests packed" now interpreted as (a) applications approved [done 2026-05-22 — ds95 자동승인], (b) datasets downloaded on user local PC + manually uploaded to Drive [in progress 2026-05-22; 176 GB], (c) preprocessing complete via §10.1 Deliverable B (preprocess_and_pack.ipynb) [pending Codex dispatch], (d) manifests packed [pending].
- **Criterion 4 (No persona-hotword regression)**: unchanged.
- **Criterion 5 (No general-ASR catastrophic forgetting)**: unchanged.
- **Criterion 6 (sherpa-onnx INT8 re-export validated)**: unchanged. Stock baseline reference bundle for comparison is `csukuangfj/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25` (SHA256 `393f8a14e2f5fb96746aaab342997a40641001fbd5bf9592a080a8329178ee96`, 878,702,423 bytes). **Canonical download source**: `https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25.tar.bz2` — k2-fsa upstream GitHub release. The HuggingFace mirror (`huggingface.co/csukuangfj/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25`) returned HTTP/2 401 on 2026-05-22 probe (gated/private post-initial public release); GitHub release URL is the publicly-accessible canonical going forward (memory `project_sherpa_onnx_qwen3_asr_bundle_source`).
- All other criteria (1, 3, 7) substantively unchanged.

**Consequences delta**:
- **NEW positive**: download bandwidth is bound by user's Korean home internet (typically 100-500 Mbps upload to Drive), not Colab Pro+ VM bandwidth. Drive desktop sync's auto-resume handles partial uploads robustly.
- **NEW negative**: critical-path adds a manual user step (download + upload). M6 (acquisition + upload complete) shifts from "Colab autonomous" to "user-execution-dependent" timeline. Estimated 1-day to 1-week M6 horizon depending on user's network and availability.
- **NEW neutral**: AI-Hub's listing API (`-mode l`) still works from Colab — useful for sanity checks but not required by v5 §3.5 (which inspects ZIP structure on Drive directly).

**Plan v5 changes**: see `Dev_Plan/2026-05-21-stt-finetune-gate1-plan.md` v5 — §3.4 rewritten, §3.5 reduced 12-cell → 5-cell, §3.6 raw layout for 13 ZIPs, §4.1/§4.4/§7.2 updated for AI-Hub built-in split adoption, §9 R-19 (lexical-narrow) NEW, §10.1 Codex deliverable split into A (inventory) + B (preprocess), §15 Changelog v5 entry.

**Process compliance**:
- ADR immutability preserved: original Decision body untouched; first Update (2026-05-21) untouched; second Update (2026-05-21) untouched; this third Update appended only.
- CLAUDE.md §7 ADR requirement met: architecture-affecting change (data acquisition path) documented at the same time as Plan v5 draft.
- CLAUDE.md §1 Gate 1 process: Plan v5 + this ADR Update undergo PM 5-step self-review → Codex `reviewer` v5-r1 → mutual discussion (up to 3 rounds) → user M3 approval before any further implementation commits. This Update is committed alongside the v5 draft as session-close documentation; the Gate 1 cycle continues in the next session.

**Memory cross-references**:
- `project_aihub_geo_restriction` (NEW 2026-05-22) — server-side geo-IP block scope (download endpoint only, listing API works); 우회 경로 (local PC + Drive); aihubshell exit-0 bug과의 관계
- `feedback_aihubshell_exit0_bug` (NEW 2026-05-22) — 3-layer wrap pattern: returncode + stdout pattern match (`"Download failed"`, `"AI 허브는 해외"`, `"HTTP status 4|5"`) + post-execution artifact size gate
- `project_sherpa_onnx_qwen3_asr_bundle_source` (NEW 2026-05-22) — canonical bundle source pin (GitHub release URL + SHA256 + size); HF mirror GATED post-initial-deploy
- `project_academic_context` (carry from 2nd Update) — academic context retained

**Cross-references for this Update**:
- `Dev_Plan/2026-05-21-stt-finetune-gate1-plan.md` v5 / v5.1 / v5.2 (the amended Plan; v5.2 = Codex v5-r1 mutual-discussion patches landed 2026-05-23)
- `docs/runbooks/weekly/2026-05-21-session58-close-handoff.md` (Session 58 close — captured M7 silent failure)
- `docs/runbooks/weekly/2026-05-22-session59-close-handoff.md` (Session 59 close — this amendment's session)
- `docs/runbooks/weekly/2026-05-22-session60-close-handoff.md` (Session 60 close — v5.1 PM self-review patches)
- `docs/runbooks/weekly/2026-05-22-daily-worklog.md` (Session 59 + 60 worklog)
- **Mutual-discussion record (v5 iteration)**: `.codex/specs/stt-finetune-gate1-plan-discussion-r5-1.md` (round 1; APPROVE WITH NOTES; 7 ACCEPT + 1 ACCEPT-as-annotation; 0 REJECT; 0 escalation) — added by Plan v5.2 per Codex v5-r1 C3 NOTE
- Memory: `C:\Users\danie\.claude\projects\E--Python-vscode-mungi\memory\project_aihub_geo_restriction.md`, `feedback_aihubshell_exit0_bug.md`, `project_sherpa_onnx_qwen3_asr_bundle_source.md`

---

## Update 2026-06-15 (fourth) — FT INT8 bundle DEPLOYED (criterion 7 MET) + hotword hallucination → hotwords disabled (criterion 4 superseded)

**Trigger**: Gate-1 Plan execution completed. The fine-tuned Qwen3-ASR-0.6B INT8 bundle was integrated (PR #221) and deployed to the Jetson runtime this session, with live on-device validation.

**Criterion 7 (Mungi integration PR landed): MET.**
- PR #221 (`<asr_text>` template-leak strip + empty-STT re-prompt + hotword env) merged to dev (`8f000f0`); CI passed; user-approved merge per CLAUDE.md §1 Gate 5.
- FT INT8 bundle `sherpa-onnx-qwen3-asr-0.6B-int8-ft` deployed to device `/opt/mungi/ai_models/` via safe cutover (stage → sha256 verify `d8d8…8c81` → retire stock to `_retired-sherpa-…-2026-03-25` [+ `.tar.bz2` backup] → atomic rename → off-kiosk CPU load-test → `mungi-kiosk` restart). Stock bundle preserved for rollback.
- On-device: FT bundle loads (`…-int8-ft`); STT runs on **CPU** (`CPUExecutionProvider`; GPU reserved for the LLM); validated held-out metrics (N=300/lang) **KO CER 5.78% / EN WER 20.32%** vs stock 14.03% / 28.20%.

**Criterion 4 (no persona-hotword regression): SUPERSEDED — wake word abandoned, hotwords disabled.**
- Live on-device testing surfaced a **hotword hallucination** failure: on low-confidence audio the FT bundle echoes the configured hotword list verbatim (`달은 숨바꼭질 뭉이 Moongee`) instead of transcribing the actual utterance. Confirmed root cause = hotword biasing (re-transcribing the same audio with hotwords OFF recovered the real utterances); consistent with the known Qwen3-ASR F31-3 issue (empty-hotword probe = 0/100).
- This is worse than the §Negative "persona hotword regression risk" anticipated in the original ADR (it produces wrong transcriptions, not merely reduced recall).
- **User decision**: abandon the wake word (`뭉이`/`Moongee`); disable hotwords (`MUNGI_QWEN3_ASR_HOTWORDS` default → empty, PR #222 `7d8fa84`). Re-validated on-device: hotwords `(none)`, real utterances transcribed correctly, no hallucination. Criterion 4's hotword-preservation requirement is therefore **moot/superseded**, not met. Memory: `project_stt_hotwords_disabled`.
- Related: the #221 hotword-hallucination DETECTOR (`_is_hotword_hallucination`) missed the echo due to a trailing-punctuation tokenization bug; fixed in PR #223 (`a993438`). The `_detect_hotword_list_recitation` legacy-vocab gap remains (separate scoped task; low priority with hotwords off).

**Promotion status**: criteria 1, 2, 3, 6, 7 substantively MET; criterion 5 (catastrophic forgetting) covered by the held-out eval (FT-session evidence); criterion 4 superseded by the hotword-disable decision. **Recommend user confirm promotion of ADR 0094 to Accepted** with criterion 4 noted as superseded. Header Status left **Proposed** pending that confirmation.

**Process compliance**:
- ADR immutability preserved: original Decision body + Updates 1-3 untouched; this fourth Update appended only.
- CLAUDE.md §1 Gate 5 (PR merge) + Gate 4 (Jetson deploy) executed with user approval this session.
- Verification gate for PR #223 advanced via ADR 0089 abbreviated step 3 (user-approved).

**Cross-references for this Update**:
- PRs #221 (FT integration), #222 (hotword disable), #223 (detector fix).
- `docs/runbooks/weekly/2026-06-15-stt-ft-int8-integration-next-session-handoff.md` (FT integration handoff — bundle metrics + cutover spec).
- `docs/runbooks/weekly/2026-06-15-stt-ft-deploy-hotword-session-close-handoff.md` (this session's close).
- Memory: `project_stt_hotwords_disabled` (NEW 2026-06-15) — wake word abandoned + hotwords disabled + FT hotword hallucination.
