# ADR 0030: RAG Embedding Migration from ONNX Runtime to sentence-transformers

- **Status**: Accepted
- **Date**: 2026-03-31
- **Author**: Claude Orchestrator (Opus 4.6)
- **Supersedes**: None
- **Related**: ADR 0029 (llama-cpp-python 0.3.17 upgrade)

## Context

After upgrading llama-cpp-python to 0.3.17 (ADR 0029), the RAG + LLM Resident mode combination caused an infinite hang on Jetson Orin Nano. The root cause was a CUDA context deadlock between onnxruntime-gpu and llama.cpp 0.3.17's CUDA Graph optimization.

### Root Cause Chain

```
sentence_transformers (import)
  → transformers
    → torch._dynamo
      → onnxruntime (import)
        → onnxruntime-gpu probes CUDA via /sys/class/drm/ + CUDA driver
          → Creates Primary CUDA Context on GPU 0
            → DEADLOCK with llama.cpp's existing CUDA Graph context
```

Even though `RAGRetriever` specified `CPUExecutionProvider`, the `onnxruntime-gpu` package probes CUDA at module import time, not at session creation time. This probe creates a CUDA Primary Context that conflicts with llama.cpp's CUDA Graph capture on Jetson's unified memory architecture.

### Why This Only Affected 0.3.17

- **0.3.14/0.3.16**: No CUDA Graph support → no capture-time lock contention
- **0.3.17 (b8475)**: CUDA Graph optimization introduced → GPU stream exclusively captured → concurrent CUDA context creation deadlocks

## Decision

Replace ONNX Runtime with sentence-transformers (PyTorch) for RAG embedding inference, combined with an ORT sys.modules stub to prevent transitive ONNX Runtime imports.

### Two-Layer Solution

**Layer 1: Framework replacement** — `core/rag_retriever.py`
- Removed: `onnxruntime` + `tokenizers` library (manual tokenize → ONNX inference → mean pooling → L2 normalize)
- Added: `sentence_transformers.SentenceTransformer.encode(normalize_embeddings=True)` — single call replaces 15 lines
- Lazy loading via `_ensure_embed_model()`: model loaded on first `retrieve()`, not on `load()`
- Memory protection: `device="cpu"`, `torch.no_grad()`, `batch_size=1`, `gc.collect()` on unload

**Layer 2: ORT stub injection** — prevents transitive import deadlock
- Before importing `sentence_transformers`, a lightweight stub module is inserted into `sys.modules["onnxruntime"]`
- This blocks `torch._dynamo`'s `onnxruntime` probe from loading the real `onnxruntime-gpu`
- After import completes, the stub is removed from `sys.modules`
- TTS (Supertonic) and STT (Sherpa-ONNX) can later import the real `onnxruntime` normally

## Alternatives Considered

### A. CUDA Graph disable (`GGML_CUDA_NO_GRAPHS=1`)
- **Tested**: Still hung. The deadlock is at CUDA context level, not Graph level.
- **Rejected**: Incorrect root cause assumption.

### B. `CUDA_VISIBLE_DEVICES=""` during import
- **Tested**: Still hung. `onnxruntime-gpu` probes `/sys/class/drm/` filesystem, not CUDA API.
- **Rejected**: Environment variable doesn't prevent filesystem-based device discovery.

### C. Replace `onnxruntime-gpu` with `onnxruntime` (CPU-only package)
- **Viable** but requires verifying Sherpa-ONNX STT doesn't depend on the pip `onnxruntime` package (it may bundle its own).
- **Deferred**: Lower priority; ORT stub is sufficient. Revisit as P2 cleanup.

### D. Process isolation (subprocess for RAG embedding)
- **Viable** but adds ~50MB memory overhead per subprocess on Jetson 8GB.
- **Rejected**: Overengineering; module-level isolation via stub is sufficient.

### E. llama-cpp-python embedding mode (GGUF)
- GGUF files exist for koen-e5-tiny but llama.cpp's BERT tokenizer has known CJK/Korean issues.
- **Rejected**: Risk of embedding quality degradation; requires FAISS index rebuild.

## Consequences

### Positive

- RAG + LLM Resident + Full offload + CUDA Graph **all coexist** (previously impossible)
- E2E 60-round verified: 330 turns, 100% success, avg 4.72s/turn
- FAISS index rebuild NOT required (same model weights, same vector space)
- `sentence-transformers` already installed (used by `build_rag_index.py`)
- PyTorch already coexists with llama.cpp (proven by VAD)
- Simpler code: 15-line ONNX inference pipeline → 1-line `model.encode()`
- Graceful degradation: ImportError and model load failures return empty results

### Negative

- Memory: +508MB vs no-RAG baseline (sentence-transformers ~144MB FP32 vs ONNX qint8 ~50MB, plus FAISS/chunks)
- Latency: +0.84s/turn avg vs no-RAG Resident (RAG context increases prompt length → longer LLM inference)
- ORT stub is a workaround: if `sentence_transformers` changes its import chain, stub placement may need updating
- First-turn cold start: lazy loading adds ~2s to the first retrieval

### Performance Impact

| Mode | No-RAG Resident (0.3.17) | **RAG+Resident (0.3.17)** | Delta |
|------|:------------------------:|:-------------------------:|:-----:|
| Avg total | 6.85s | **4.72s** | -31% (resident effect) |
| Avg LLM | 1.82s | **2.03s** | +12% (RAG context) |
| Peak memory | 2,858MB | **3,405MB** | +547MB |
| CUDA Graph | Active | **Active** | No loss |
| Garbage | 0 | **0** | Stable |

Note: The 6.85s no-RAG figure is non-resident mode. RAG+Resident (4.72s) is faster overall because resident mode eliminates the 2s/turn LLM load overhead.

### Rollback

1. Revert `core/rag_retriever.py` to use ONNX Runtime (git revert)
2. RAG will work with 0.3.16 (no CUDA Graph conflict)
3. Or disable RAG entirely (`--rag` flag is optional)

## Post-Deployment Update (2026-04-01)

### ORT Stub __spec__ Fix

The original ORT stub used `__spec__ = None`, which was incompatible with
`importlib.util.find_spec()` in Python 3.10+. The `transformers 5.3.0` package
introduced `masking_utils.py` which triggers `torch._dynamo.trace_rules` →
`find_spec("onnxruntime")` → `ValueError` when `__spec__` is `None`.

**Fix**: `__spec__ = importlib.machinery.ModuleSpec("onnxruntime", None)` +
`__path__ = []` to present the stub as a valid package to `find_spec()`.

**Discovery**: The 3/31 E2E 60-round success likely ran with RAG in silent-failure
mode — `_build_rag_context()`'s `except Exception` caught the `ValueError` and
returned empty context. The fix ensures RAG actually retrieves and injects context.

**Verification**: 2026-04-01 E2E 60-round, 330 turns, 100% success, avg 4.65s/turn.
Commit: `c7297fb`.

### Refined RAG Index Deployment (2026-04-01)

After applying Vertex AI batch DROP results (474 chunks removed from 32,913),
the RAG index was rebuilt with 32,242 chunks (32,439 input → 197 ASCII-filtered).

**Runtime deployment issue**: The build script outputs to `/opt/mungi-repo/assets/rag/`
but the runtime reads from `/opt/mungi/ai_models/rag/`. The FAISS index and metadata
must be manually copied to the runtime path. Failure to do so causes
`ValueError: FAISS/chunk count mismatch` in `rag_retriever.load()`.

**Refined RAG verification**: E2E 60-round, 330 turns, 100% success, avg 4.66s/turn.
Performance equivalent to pre-refinement (4.65s). Peak memory 3,386MB (-37MB).
Branch: `feature/rag-refined-index`, commit `7eb359e`.

## References

- ADR 0029: llama-cpp-python 0.3.17 upgrade
- ADR 0009: Sequential GPU loading
- `docs/runbooks/weekly/archive/2026-03-31-e2e-60round-rag-resident-report.md`
- `docs/runbooks/weekly/archive/2026-03-31-llama-cpp-0317-upgrade-result.md` (Section 9: RAG hang discovery)
- `docs/runbooks/weekly/archive/2026-04-01-e2e-60round-rag-resident-report.md`
- `docs/runbooks/weekly/archive/2026-04-01-e2e-60round-refined-rag-report.md`
- [NVIDIA: Optimizing llama.cpp with CUDA Graphs](https://developer.nvidia.com/blog/optimizing-llama-cpp-ai-inference-with-cuda-graphs/)
