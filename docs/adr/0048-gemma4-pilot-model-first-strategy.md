# ADR 0048: Gemma 4 Pilot — "Model First, Infra Second" Evaluation Strategy

- **Status**: Accepted
- **Date**: 2026-04-09
- **Decision makers**: User + Claude Code (PM)

## Context

Gemma 4 (released 2026-04-02) is a candidate replacement or supplement for the current Qwen3.5-2B DPO Q6_K LLM in the Mungi pipeline. The Qwen model has known safety issues (10 CRITICAL, 28+ HIGH per `safety/mungi-script-qa-report.md`). Gemma 4 E2B offers multimodal capabilities (text + image + audio) with Google's safety training.

Two approaches were considered for evaluation:

1. **OTA upgrade + Docker pilot**: Upgrade JetPack 6.2.1 → 6.2.2 via APT, install Gemma 4 via NVIDIA Docker container, run comparative evaluation against Qwen on current SD card setup, then migrate to NVMe after model is confirmed.

2. **NVMe re-flash first**: Flash JetPack 6.2.2 to NVMe SSD via SDK Manager (4-6 hours, requires Ubuntu Live USB host), then swap model images for benchmarking on NVMe.

## Decision

**Option 1: "Model first, infra second."**

- Stage 1 (Pilot): OTA APT upgrade → Docker Gemma 4 E2B → Qwen vs Gemma comparative evaluation (quantitative + qualitative, 20 prompts, 4-dimension scoring) → final model decision.
- Stage 2 (Infrastructure): NVMe SSD boot migration → confirmed model optimized deployment. Deferred to separate session.

## Rationale

1. **Model inference performance is GPU-bound, not storage-bound.** Tokens/sec, Korean quality, and safety scores are identical on SD card vs NVMe. Storage only affects initial model loading time (~30s SD vs ~5s NVMe), which is a secondary metric.

2. **OTA preserves data, NVMe re-flash does not.** APT upgrade takes ~30 minutes with zero data loss. SDK Manager re-flash requires full backup/restore cycle (4-6 hours).

3. **Risk ordering matters.** If Gemma 4 fails evaluation (OOM, poor Korean, safety issues), NVMe migration effort would have been wasted. Evaluating first eliminates this risk.

4. **JetPack 6.2.2 CUDA bug fix is critical.** JetPack 6.2.1 has a known CUDA memory allocation bug that may cause false OOM errors. OTA upgrade resolves this before evaluation begins.

## Model Selection: Gemma 4 E2B Q8_0

| Variant | Size | Orin Nano 8GB Feasibility | Source |
|---------|------|:---:|--------|
| **E2B Q8_0** | 4.97 GB | **Yes** (~3 GB headroom) | Recommended |
| E4B Q4_K_M | 5.3 GB | **No** — OOM confirmed | NVIDIA forum |
| E4B Q8_0 | ~8 GB | Impossible | Math |

Installation via NVIDIA official Docker container (`ghcr.io/nvidia-ai-iot/llama_cpp:gemma4-jetson-orin`), recommended by NVIDIA engineer AastaLLL on developer forums. vLLM does not support Gemma 4 architecture on Jetson.

## Consequences

### Positive
- Evaluation can start immediately (next Jetson session, ~3-4 hours total)
- Zero risk to existing Mungi setup (OTA is non-destructive)
- Model decision informs whether NVMe migration is even necessary
- Docker-based serving opens path to easier model updates

### Negative
- Model loading will be slower on SD card (~30s vs ~5s on NVMe) during evaluation — acceptable for pilot
- Docker adds HTTP overhead compared to native llama-cpp-python — measured during evaluation
- If Gemma is confirmed, Mungi pipeline architecture change is needed (native → HTTP API)

### Neutral
- Existing NVMe migration plan (`docs/archived/dev-plan/2026-04-09-JetPack-622-NVMe-Upgrade-Plan.md`) remains valid and is deferred, not cancelled

## References

- `docs/archived/dev-plan/2026-04-09-Gemma4-Installation-Plan.md` — Full pilot plan with evaluation framework
- `docs/archived/dev-plan/2026-04-09-JetPack-622-NVMe-Upgrade-Plan.md` — Deferred NVMe plan (Stage 2)
- `safety/mungi-script-qa-report.md` — Qwen safety issues driving model evaluation
- [NVIDIA Forum: Gemma 4 on Orin Nano](https://forums.developer.nvidia.com/t/no-luck-with-gemma-4-on-jetson-nano-super/365620)
- [Jetson AI Lab: Gemma 4 E2B](https://www.jetson-ai-lab.com/models/gemma4-e2b/)
