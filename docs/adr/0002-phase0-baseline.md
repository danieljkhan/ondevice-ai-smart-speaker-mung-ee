# ADR 0002: Phase 0 Baseline Technical Decisions

- Status: Accepted (updated 2026-03-13)
- Date: 2026-03-12
- Updated: 2026-03-13 — onnxruntime finding corrected, LD_LIBRARY_PATH expanded
- Decision makers: Daniel (user), Claude PM

## Context

Phase 0 of the Mungi project requires establishing a verified technical baseline
on Jetson Orin Nano Super 8GB before proceeding to MVP development. The baseline
must confirm that every component in the default AI pipeline (Silero VAD,
Faster-Whisper STT, Qwen via llama.cpp, Supertonic TTS 2, Piper TTS fallback) can
run within the 8 GB memory budget, and that the development toolchain (CI, linting,
type checking) is stable enough to support multi-agent team development.

This ADR records the five key technical decisions made during Phase 0 investigation.

## Decisions

### 1. onnxruntime-gpu: PyPI Wheel Works with Correct LD_LIBRARY_PATH

> **2026-03-13 UPDATE**: The original conclusion was incorrect. The PyPI wheel
> DOES support CUDA — the issue was a missing `LD_LIBRARY_PATH` entry.

**Context:**
`onnxruntime-gpu 1.23.0` installed from PyPI appeared to lack
`CUDAExecutionProvider`. Running `onnxruntime.get_available_providers()` returned
only `['AzureExecutionProvider', 'CPUExecutionProvider']`.

**Root cause (corrected 2026-03-13):**
The PyPI aarch64 wheel ships with CUDA support, but the CUDA runtime libraries
bundled inside the pip-installed `nvidia` namespace package were not discoverable.
The missing path was:

```
/opt/mungi-repo/.venv/lib/python3.10/site-packages/nvidia/cu12/lib/
```

Adding this path to `LD_LIBRARY_PATH` causes the PyPI wheel to correctly expose:
`['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']`

**Decision (revised):**
The PyPI wheel is acceptable. The critical requirement is that `LD_LIBRARY_PATH`
includes the `nvidia/cu12/lib` directory from the venv's site-packages.
The `mungidev` shell function must export this path. The local wheelhouse backup
remains available as a fallback but is no longer the primary source.

### 2. System CUDA Paths Required in LD_LIBRARY_PATH

**Context:**
Even after installing the correct `onnxruntime-gpu` wheel,
`CUDAExecutionProvider` fails to initialize unless system CUDA shared libraries
are discoverable at runtime.

**Required paths (updated 2026-03-13):**

| Variable | Paths |
|----------|-------|
| `LD_LIBRARY_PATH` | `/usr/local/cuda/lib64`, `/usr/lib/aarch64-linux-gnu`, `$VENV/lib/python3.10/site-packages/nvidia/cu12/lib` |
| `PATH` | `/usr/local/cuda/bin` (for `nvcc`, `cuda-gdb`, etc.) |

> **2026-03-13 UPDATE**: The `nvidia/cu12/lib` path is critical — without it,
> both `onnxruntime-gpu` CUDA EP and `torch` fail to load (`libcudss.so.0` not
> found). This single missing path was the root cause of the Phase 0 onnxruntime
> blocker.

**Resolution:**
The `mungidev()` shell function in `~/.bashrc` must be updated to export all
three `LD_LIBRARY_PATH` entries every time the development environment is
activated.

**Decision:**
`mungidev` must always export system CUDA paths plus the venv nvidia library
path. This is a mandatory dev setup step. Any documentation or onboarding guide
must call this out explicitly. Developers must run `mungidev` before any Jetson
development work.

### 3. Silero VAD Confirmed Viable for Jetson

**Context:**
Silero VAD is the first stage of the AI pipeline. It must run efficiently on
Jetson, handle both English and Korean speech, and leave enough memory headroom
for downstream models.

**Model details:**

- Source: `torch.hub` (snakers4/silero-vad)
- Additional dependency: `torchaudio` (must be installed separately)

**Test results:**

| Test case | Duration | Segments | Speech detected | Verdict |
|-----------|----------|----------|-----------------|---------|
| Pure tone (440 Hz sine) | 8.0 s | 0 | 0 s | Correct rejection |
| English speech | 33.6 s | 10 | 20.58 s | Correct detection |
| Korean speech | 11.1 s | 3 | 9.44 s | Correct detection |

**Performance metrics:**

| Metric | Value (Sprint 1) | Value (Sprint 2 re-test) | Notes |
|--------|-------------------|--------------------------|-------|
| Model load time | 3.7 -- 4.2 s | 1.46 s | Cached after first load |
| Inference RTF | 0.05 -- 0.07 | 0.32 | See note below |

> **RTF measurement methodology difference:**
> Sprint 1 (0.05~0.07): Estimated on dev PC (x86), range from multiple runs.
> Sprint 2 (0.32): Measured on Jetson Orin Nano Super via `test_vad.py`
> (11.37s Korean audio, RTF = 3.662s / 11.37s). The difference is due to
> measurement environment (x86 PC vs ARM64 Jetson), not performance regression.
> The Jetson measurement (0.32) is the authoritative baseline.
| Peak memory | 430 -- 440 MB | +17 MB delta | Measured as RSS delta |

**Decision:**
Silero VAD is confirmed viable for Jetson Orin Nano Super. It is
language-agnostic and handles Korean speech correctly. The memory footprint
(~430 MB) leaves approximately 7.5 GB for downstream models, which is within
the memory budget.

### 4. CI Dependency Separation

**Context:**
The GitHub Actions CI workflow was installing `requirements-core.txt`, which
contains runtime-only packages (`sounddevice`, `flask`, `supertonic`, etc.)
that are unnecessary for linting and testing. This caused installation failures
in CI and obscured real issues.

**Decision:**
Create a standalone `requirements-ci.txt` containing only packages needed for CI:

- `ruff`
- `mypy`
- `pytest`
- Other linting/testing utilities

The CI workflow (`ci.yml`) installs only `requirements-ci.txt`. Runtime
dependencies are verified separately on Jetson hardware.

### 5. Mypy Optional Dependency Overrides

**Context:**
Jetson-specific packages (`torch`, `onnxruntime`, `faster_whisper`, `llama_cpp`,
`supertonic`, `sounddevice`, `torchaudio`, etc.) are not installed in the CI
environment. This causes `mypy` to emit `import-not-found` errors for every
file that imports these packages, blocking the CI pipeline.

**Decision:**
Configure `mypy` overrides in `pyproject.toml` with `ignore_missing_imports = true`
for all optional Jetson-only packages. This approach:

- Eliminates the need for per-file `# type: ignore` comments
- Keeps `mypy` strict for all non-optional imports
- Allows CI to pass without installing heavy GPU packages

Affected modules configured as overrides:

```
torch, torchaudio, onnxruntime, faster_whisper, llama_cpp,
sounddevice, supertonic, piper
```

> **2026-03-13 UPDATE (Sprint 3 Day 2)**: Sprint 3에서 `models/` 레이어가
> 신설되어 모델 추론 로직이 `scripts/`에서 `models/`로 이동했다
> (`vad_runner.py`, `stt_runner.py`, `llm_runner.py`, `tts_runner.py`).
> 이 모듈들은 위와 동일한 Jetson 전용 패키지를 import하므로,
> 동일한 mypy override 패턴이 적용된다. 신규 모듈 추가 시 override 목록
> 유지보수가 필요하다. 아키텍처 결정은 ADR 0006 참조.

## Consequences

### Positive

- The P0 blocker (`onnxruntime-gpu` CUDA provider) is fully resolved with a
  reproducible, documented solution.
- Silero VAD viability is confirmed with quantitative evidence, giving confidence
  to proceed to STT/LLM/TTS integration in Sprint 2.
- CI is stable and fast, with no false positives from missing GPU packages.
- The `mungidev` setup procedure is documented and enforced, reducing onboarding
  friction for future contributors.
- Memory baseline (~430 MB for VAD alone) establishes a concrete starting point
  for memory budget tracking.

### Negative

- ~~The local wheel approach for `onnxruntime-gpu` is not automatically updatable~~
  (2026-03-13: resolved — PyPI wheel works with correct LD_LIBRARY_PATH).
- `torchaudio` is a new dependency that has not yet been added to
  `requirements-jetson.txt`; this must be addressed in Sprint 2.
- The `mypy` override list must be maintained manually as new optional packages
  are added.
- Audio hardware testing was blocked (USB sound card not available), so the
  full end-to-end pipeline has not been validated yet.
- Additional Jetson-specific source builds are required for some packages;
  see ADR 0005 for details.

## Later References

- ADR 0012: LLM model upgraded from Qwen3-1.7B to Qwen3-4B-Q4_K_M.

## Update (2026-05-13)

Piper TTS fallback retired per ADR 0088. The baseline reference to Piper in line 13
(smoke-test stack enumeration) and line 159 (sounddevice list) are superseded. See
ADR 0088 for the retirement rationale and the formalized Supertonic-only contract.
