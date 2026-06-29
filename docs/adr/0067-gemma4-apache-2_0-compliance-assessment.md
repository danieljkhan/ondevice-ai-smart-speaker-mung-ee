# ADR 0067 — Gemma 4 Apache 2.0 license compliance assessment for Mungi child-facing offline companion

- **Status**: **APPROVED 2026-04-21** (user Gate 1 승인). Codex R3 R3-F1/R3-F2/R3-F3 integrated. 10-round live E2E confirmed. Pipeline decision: Gemma 4 E2B Q5_K_M text LLM via llama-cpp-python 0.3.20 + llama.cpp b8772.
- **Date**: 2026-04-21
- **Authors**: Claude Code PM (Opus 4.7)
- **Context plan**: `docs/archived/dev-plan/2026-04-21-gemma4-text-llm-mungi-integration-plan-v3_1.md` §4.1
- **Supersedes / related**: —

## Context

Mungi is an offline edge AI conversation device for children under age 10, running on Jetson Orin Nano Super 8 GB. The product vision is "the safest AI friend — a child's very first." This ADR documents the license + policy compliance assessment for integrating `google/gemma-4-E2B-it` (Gemma 4 E2B instruction-tuned) as the LLM stage of Mungi's conversation pipeline.

The assessment is required as Gate 1 prerequisite per Plan v3.1 §4.1, in response to Codex R2-F3-BLOCK finding.

## License regime

- Official Google documentation indicates that **Gemma 4 is released under the Apache License 2.0**, replacing the earlier bespoke "Gemma Terms" that applied to Gemma 1 / 2 / 3.
- References:
  - Gemma Terms page (legacy redirect): `https://ai.google.dev/gemma/terms`
  - Gemma 4 Apache 2.0 license page: `https://ai.google.dev/gemma/apache_2`
  - Prohibited Use Policy (distinct from license, applied separately): `https://ai.google.dev/gemma/prohibited_use_policy`
  - Intended Use Statement: `https://ai.google.dev/gemma/intended_use_statement`

## Apache 2.0 obligations analysis for Mungi's use case

### Clause 2 (Grant of Copyright License)

Grants permission to use, reproduce, modify, merge, distribute. **Satisfied by default**; Mungi uses the model for local inference.

### Clause 4(a) (Retention of notices)

Any distribution of the Work or Derivative Works must retain the copyright notices, patent notices, trademark notices, attribution notices present in the Work.

**Action**: Mungi must include an Apache 2.0 `NOTICE` file in deployments that reference Gemma 4. Implementation:
- `/opt/mungi/licenses/NOTICE` documents: "This product includes Gemma 4 E2B from Google LLC, used under the Apache License 2.0. Gemma 4 copyright notices are preserved in the GGUF model file metadata."
- Mungi's public documentation (when product ships) credits Gemma usage.

### Clause 4(b) (Distribution statement) — revised per R3-F1

Derivative Works must carry a notice describing modifications.

**Prior (v1) position**: Mungi uses Gemma 4 "unmodified" because GGUF quantization is a format conversion and system-prompt layering is instruction wrapping, therefore no derivative-work notice required.

**Revised (v2, safer posture)**: Treat the Q5_K_M GGUF file as a **transformed redistributed artifact** even though Mungi is not the quantizer.
- Q5_K_M quantization alters the weight representation (bf16 → q5_K) — arguably a "modification" in the Apache 2.0 sense even if lossless-format conversion is arguable.
- Mungi did not quantize; the file originates from unsloth on HuggingFace. However, by shipping this GGUF on `/opt/mungi/ai_models/` in a product distribution, Mungi acts as a secondary redistributor.
- Even if a legal determination ultimately says quantization ≠ modification, the cost of including a prominent attribution notice is trivial and the cost of omitting it if a court later disagrees is material.

**Action (revised)**: Ship a prominent attribution notice (NOTICE file at `/opt/mungi/licenses/NOTICE`) that records:
- Source model: `google/gemma-4-E2B-it` (Google LLC, Apache 2.0).
- Quantization format: Q5_K_M GGUF.
- Quantizer / origin: unsloth (HuggingFace).
- File SHA256: `f281a529f9272d1febd75c242b94c69d54f577268d87a15cf175ed7ffa5bc73c`.
- Date deployed to Mungi: `<Phase 3 cutover date>`.
- Apache 2.0 license full text **shipped** alongside at `/opt/mungi/licenses/apache-2.0-license.txt` per clause 4(a) (Mungi redistributes the model in binary form; recipients get the license copy on device).

System-prompt layering ("너는 뭉이야") remains non-modification of weights.

### Clause 4(c) (NOTICE pass-through)

If the Work's original distribution contains a `NOTICE` file, Mungi distributions must include readable attribution.

**Action**: Download and include Gemma 4's upstream NOTICE content (if any) from HuggingFace model card in Mungi's `/opt/mungi/licenses/gemma4-NOTICE`. If no upstream NOTICE exists, create one describing "Gemma 4 E2B, Google LLC, Apache 2.0" attribution.

### Clause 5 (Submission of contributions)

Not applicable — Mungi does not contribute modifications back.

### Clause 6 (Trademarks)

License does NOT grant trademark rights for "Google", "Gemma", or related marks.

**Action**: Mungi's product branding MUST NOT imply endorsement or origin from Google.
- Product name "Mungi" does not include "Gemma" or "Google".
- Persona name "뭉이" is independent.
- Public-facing marketing materials (future) must NOT suggest Google/Gemma partnership.
- Internal documentation (this ADR, plan files) may reference "Gemma 4" descriptively.

### Clause 7 (Disclaimer of warranty)

Apache 2.0 disclaims all warranties. Mungi's product terms (when shipped) must clearly state that Gemma 4 is distributed under this disclaimer.

**Action**: Mungi end-user documentation (future) carries a disclaimer reflecting Apache 2.0 clause 7.

### Clause 8 (Limitation of liability)

Google has no liability for damages arising from use of Gemma 4.

**Action**: Consistent with the child-safety product vision — Mungi's `safety/` filter chain is the primary safety mechanism. The model vendor (Google) is not relied upon as a liability backstop.

## Google Prohibited Use Policy analysis (R3-F2 + R3-F3 revised)

The Prohibited Use Policy is a separate document from the Apache 2.0 license. Google states usage of Gemma 4 is subject to both. Reference: `https://ai.google.dev/gemma/prohibited_use_policy`. The policy covers the following clauses (this enumeration is exhaustive for Mungi's use-case scope):

1. **Child sexual exploitation / CSAM** — MUST be blocked by Mungi's `safety/` filter chain before TTS. Test coverage in Phase 1: `tests/test_safety_gemma4_prohibited_use.py` case `csam_adjacent_content`.
2. **Sexual content generally (beyond CSAM)** — child product blanket prohibition on all sexual/romantic content regardless of age context. Test coverage: `sexual_content_child_product_inapplicable`.
3. **Self-harm / suicide promotion or facilitation** — blocked; referral language to trusted adult. Test coverage: `self_harm_facilitation`.
4. **Harassment / bullying** — blocked at safety filter + persona prompt steers away. Test coverage: `harassment_bullying`.
5. **Illegal activities (weapons, drugs, hacking)** — safety filter. Test coverage: `illegal_activity_instructions`.
6. **Discrimination based on protected class** — safety filter + persona prompt. Test coverage: `discriminatory_content`.
7. **Fraud / impersonation** — safety filter. Test coverage: `fraud_impersonation_scenario`.
8. **Misinformation / disinformation** — persona prompt steers tentative/child-friendly language; safety filter checks for authoritative false claims. Test coverage: `misinformation_health_safety`.
9. **Unauthorized professional advice (medical / legal / financial)** — persona redirects to "trusted adult" per Mungi persona. Test coverage: `professional_advice_medical_legal_financial`.
10. **PII generation / distribution** — persona blocks; safety filter flags. Test coverage: `pii_generation_inference`.
11. **Safety-filter bypass attempts (prompt injection)** — Mungi's existing prompt-injection defenses from T3.3 work; new cases cover Gemma-4-specific jailbreak strings if discovered in Phase 2 red-teaming. Test coverage: `safety_filter_bypass_attempts`.
12. **Sensitive personal / private information gathering / processing / inferring without required rights / authorizations / consents** — this clause is broader than "no cloud telemetry." It covers **any** collection/storage of sensitive personal information, including on-device storage of children's voice and conversation content. See §R3-F3 resolution below.

**R3-F3 revision — parental consent re-scoped as Phase 3 cutover BLOCKER (not "future product note")**:

The v1 ADR said "parental consent ... is not applicable to on-device private storage." This was too narrow. Google's Prohibited Use Policy clause on sensitive personal data is broader. Because:

- Mungi permanently stores children's conversation audio and text at `/var/lib/mungi/conversations/` per CLAUDE.md §6.
- The data subjects are children under 10.
- Google's policy requires "rights / authorizations / consents" for gathering/processing/inferring sensitive personal information.

The Mungi product MUST collect documented parental consent before enabling Gemma-4-based conversation logging for a specific device.

**Mungi operational requirement (blocker for Phase 3 cutover, not Phase 1)**:
- First-run setup flow collects and records parental consent.
- Consent record stored at `/var/lib/mungi/config/parental_consent.json` with timestamp, persona name, Mungi device serial.
- Consent revocation flow stops Gemma-4-based logging; legacy data tagged with `_legacy_preserved_` rather than deleted (CLAUDE.md §6 permanence).
- Phase 3 cutover cannot proceed without this consent-collection infrastructure verified on the target Jetson.

**Posture**: Prohibited-use obligations are enforced via (a) Mungi `safety/` filter chain + new Phase 1 tests for 11 clause cases above, and (b) parental consent infrastructure as a Phase 3 cutover blocker for clause 12. No license-level Apache 2.0 blocker.

## Child-facing offline product specifics

1. **No telemetry / offline guarantee**: Apache 2.0 does NOT require upstream telemetry. Mungi's offline-first vision is compatible.
2. **Permanent conversation storage**: CLAUDE.md §6 mandates. Apache 2.0 and Gemma Prohibited Use Policy do not prohibit permanent storage; Google's sensitive-personal-data clause requires consent for third-party processing, which is not applicable to on-device private storage.
3. **Parental consent**: A product-level operational requirement (not an ADR deliverable) — Mungi's first-run setup must collect parental consent before enabling conversation logging.
4. **Under-13 regulatory overlap**: US COPPA / EU GDPR-K may apply. Out of scope for this ADR — a separate legal review is required before commercial launch. This ADR concerns only Gemma 4 license compliance, not broader regulatory posture.

## NOTICE file draft (R3-F1 revised)

```
Mungi AI Companion
Copyright 2026 Daniel J. K. Han

This product includes Gemma 4 E2B from Google LLC, distributed under the
Apache License, Version 2.0 (see `/opt/mungi/licenses/apache-2.0-license.txt`
on this device for the full license text).

Model artifact as deployed:
- Upstream: google/gemma-4-E2B-it
- Format: Q5_K_M GGUF quantization
- Quantizer: unsloth (via HuggingFace)
- SHA256: f281a529f9272d1febd75c242b94c69d54f577268d87a15cf175ed7ffa5bc73c
- Deployed: <Phase 3 cutover date>

Use of Gemma 4 is also subject to Google's Prohibited Use Policy at
https://ai.google.dev/gemma/prohibited_use_policy.
Mungi enforces these obligations via the on-device `safety/` filter chain
and the Mungi persona prompt (see /opt/mungi/licenses/prohibited-use-enforcement.md).

"Gemma" is a trademark of Google LLC; used here solely to identify the
model's upstream origin, not to imply endorsement or partnership.
```

Placement: `/opt/mungi/licenses/NOTICE` during Phase 3 cutover. Full Apache 2.0 license text at `/opt/mungi/licenses/apache-2.0-license.txt`. End-user documentation (shipped with Mungi product) references both.

## Decision (revised v2)

1. Gemma 4 Apache 2.0 license is **compatible** with Mungi's use case for the child-facing offline companion.
2. **No license-level blocker exists for Phase 1 integration** (default-off backend switch, text-only inference).
3. **Apache 2.0 obligations** (clauses 4a/4b/4c/6/7) are fulfilled via:
   - NOTICE file with attribution, quantization provenance, SHA256, and Mungi device-local license copy.
   - Ship the full Apache 2.0 license text on device at `/opt/mungi/licenses/apache-2.0-license.txt` (clause 4a).
   - Trademark discipline: no implication of Google endorsement in public-facing branding.
   - Disclaimer of warranty + limitation of liability reflected in Mungi's end-user terms (product obligation, Phase 3).
4. **Prohibited Use Policy obligations** enforced via (a) Mungi `safety/` filter chain + Phase 1 `tests/test_safety_gemma4_prohibited_use.py` covering 11 clause cases (see §R3-F2), and (b) **parental consent infrastructure as Phase 3 cutover BLOCKER** (see §R3-F3) for sensitive-personal-data clause.
5. Regulatory compliance (COPPA / GDPR-K) remains a separate pre-commercial legal engagement — this ADR does not substitute.

## Consequences

- Phase 1 PR includes:
  - `/opt/mungi/licenses/NOTICE` draft (deployed Phase 3).
  - `/opt/mungi/licenses/apache-2.0-license.txt` bundled with product (deployed Phase 3).
  - `tests/test_safety_gemma4_prohibited_use.py` with 11 explicit Prohibited Use Policy clause cases.
- Phase 3 cutover **blocks** until parental consent infrastructure is verified on target Jetson:
  - `/var/lib/mungi/config/parental_consent.json` schema + first-run flow.
  - Consent revocation preserves legacy data with `_legacy_preserved_` tag (CLAUDE.md §6 permanence).
- Pre-commercial legal review (separate engagement) remains required for broader regulatory posture.
- ADR 0067 final version (post-Phase-3) captures observed compliance experience.

## Status update (after Gate 1)

TBD. Update after user approval + Phase 1 deployment.

## References

- Apache License 2.0 full text: `http://www.apache.org/licenses/LICENSE-2.0`
- Google Gemma Terms (legacy): `https://ai.google.dev/gemma/terms`
- Gemma 4 Apache 2.0 page: `https://ai.google.dev/gemma/apache_2`
- Gemma Prohibited Use Policy: `https://ai.google.dev/gemma/prohibited_use_policy`
- Gemma Intended Use Statement: `https://ai.google.dev/gemma/intended_use_statement`
- CLAUDE.md §6 — Mungi conversation permanence
- Plan v3.1 §4.1 — referring plan section
- ADR 0066 — Wave 3 PARTIAL close (context)
