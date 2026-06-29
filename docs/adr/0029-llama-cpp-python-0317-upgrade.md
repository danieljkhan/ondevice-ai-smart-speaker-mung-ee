# ADR 0029: llama-cpp-python Upgrade to 0.3.17 for KV Cache Fix

- **Status**: Accepted
- **Date**: 2026-03-31
- **Author**: Claude Orchestrator (Opus 4.6)

## Context

llama-cpp-python 0.3.14 (llama.cpp b5904) exhibited KV cache corruption when using `n_gpu_layers=-1` (full GPU offload) in the E2E pipeline. The corruption manifested as repeating digit sequences (`333...`) after 2+ conversation turns, triggered by GPU memory fragmentation during LLM↔TTS model load/unload cycles.

The issue was absent in isolated LLM tests (model loaded once, kept resident) but appeared consistently (67% of turns) in the full pipeline where models are loaded and unloaded each turn.

## Decision

Upgrade llama-cpp-python through a 3-stage verified path:

1. **0.3.14 → 0.3.16** (b5904 → b~6170): Intermediate step. Added built-in `seq_id` clamp, `ggml_set_rows()` default ON, Flash Attention KV_max fix. **Result**: KV corruption persisted in non-resident mode — `ggml_cpy()` fallback still present.

2. **0.3.16 → 0.3.17** (b~6170 → b8475): Final target. `ggml_cpy()` legacy path fully removed, `ggml_set_rows()` is the only KV cache path. Jetson iGPU buffer detection bug fixed. **Result**: KV corruption eliminated in ALL modes.

All versions built from source on Jetson (`scripts/build_llama_cpp.sh`) with `CUDA_ARCHITECTURES=87`.

## Alternatives Considered

- **Keep 0.3.14 + `n_gpu_layers=20` cap**: Worked but sacrificed ~35% LLM speed (4.91s vs 1.69s avg). Acceptable as workaround but not as permanent solution.
- **Keep 0.3.14 + LLM resident mode only**: Eliminated the trigger (no load/unload) but didn't fix the root cause. Non-resident mode remained broken.
- **Upgrade to 0.3.16 only**: Insufficient — `ggml_cpy()` fallback still caused corruption in non-resident mode.
- **Upgrade to 0.3.19 (latest)**: Larger jump with no additional KV cache benefit over 0.3.17. Higher regression risk for no gain.

## Consequences

### Positive

- Full GPU offload (`n_gpu_layers=-1`) safe in all modes (resident and non-resident)
- LLM inference ~2x faster with full offload vs partial (20 layers)
- `LLAMA_SET_ROWS` environment variable no longer needed
- `seq_id` monkey-patch no longer needed (retained as no-op for rollback)
- Garbage detection filter retained as defense-in-depth layer
- Non-resident mode viable as fallback (~6.9s avg vs ~4.6s resident)

### Negative

- Source build required on Jetson (35 min, no pre-built aarch64 CUDA wheel)
- llama.cpp jumped ~2,571 builds — large change surface for potential regressions
- `numpy` version conflict during build (auto-resolved by reinstall)
- Monkey-patch code becomes dead code (retained for rollback safety)

### Performance Impact

| Mode | Before (0.3.14) | After (0.3.17) |
|------|:---------------:|:--------------:|
| Resident full offload | N/A (not implemented) | **avg 3.88s** (60-round) |
| Non-resident full offload | **BROKEN** (67% garbage) | **avg 6.92s** (stable) |
| Non-resident partial (20) | avg 9.55s | N/A (no longer needed) |

### Rollback

Pre-built wheels for 0.3.14 and 0.3.16 stored in `wheelhouse/`:
```bash
pip install wheelhouse/llama_cpp_python-0.3.14-cp310-cp310-linux_aarch64.whl
pip install wheelhouse/llama_cpp_python-0.3.16-cp310-cp310-linux_aarch64.whl
```

### Known Interaction: RAG + 0.3.17 CUDA Graph Deadlock

0.3.17's CUDA Graph optimization conflicts with `onnxruntime-gpu`'s import-time CUDA probing. When RAG embedding is active, the transitive import chain (`sentence_transformers` → `onnxruntime`) creates a competing CUDA context that deadlocks with llama.cpp's CUDA Graph capture. **Resolved in ADR 0030** by migrating RAG embedding from ONNX Runtime to sentence-transformers with an ORT sys.modules stub.

## References

- ADR 0005: Jetson CUDA package build policy
- ADR 0009: Sequential GPU loading
- **ADR 0030: RAG embedding ONNX RT → sentence-transformers migration**
- `docs/runbooks/weekly/archive/2026-03-30-kv-cache-full-offload-issue.md`
- `docs/runbooks/weekly/archive/2026-03-30-kv-cache-fix-feasibility-report.md`
- `docs/runbooks/weekly/archive/2026-03-31-llama-cpp-0317-upgrade-result.md`
- `docs/runbooks/weekly/archive/2026-03-31-llm-resident-mode-report.md`
- `docs/runbooks/weekly/archive/2026-03-31-e2e-60round-rag-resident-report.md`

## Update (2026-04-15) - Build infrastructure alignment

Discovery on 2026-04-15 confirmed that the 2026-03-31 upgrade commit (`4440eaa`) modified runtime code only. The default in `scripts/build_llama_cpp.sh` and the pin in `Dev_Plan/requirements-jetson.txt` remained at 0.3.14 for roughly 3.5 months, so a fresh clone followed by the default build path still produced a broken 0.3.14 wheel.

Action in this change:

- `scripts/build_llama_cpp.sh` default bumped to 0.3.17.
- `Dev_Plan/requirements-jetson.txt` pin bumped to 0.3.17.
- `scripts/install_llama_cpp.sh` added with `--from-release` (preferred) and `--from-source` (fallback) modes.

Release artifact:

- Tag: `v0.3.17-llama`
- Asset: `llama_cpp_python-0.3.17-cp310-cp310-linux_aarch64.whl`
- Size: 413 MB
- SHA256: `12bffe3f7c5a3c445debb34b235f22276090a3d0aac1d806b29a32cd17c4f503`
- Build host: Jetson Orin Nano Super, JetPack 6.2, CUDA 12.6, Python 3.10, sm_87
- Build date: 2026-04-13

### Reproducibility matrix

| Install path | Command | Expected duration | Expected SHA256 |
|---|---|---|---|
| GitHub Release wheel | `bash scripts/install_llama_cpp.sh --from-release` | ~5-10 min | `12bffe3f7c5a3c445debb34b235f22276090a3d0aac1d806b29a32cd17c4f503` |
| Local source build | `bash scripts/install_llama_cpp.sh --from-source` | ~35 min | `12bffe3f7c5a3c445debb34b235f22276090a3d0aac1d806b29a32cd17c4f503` |
| Jetson AI Lab prebuilt (if available) | `pip install llama-cpp-python==0.3.17 --index-url https://pypi.jetson-ai-lab.io/jp6/cu126` | ~5-10 min | `12bffe3f7c5a3c445debb34b235f22276090a3d0aac1d806b29a32cd17c4f503` |

Full implementation details and rollout steps are documented in `docs/archived/dev-plan/2026-04-15-llama-cpp-tech-debt-resolution-plan.md`.
