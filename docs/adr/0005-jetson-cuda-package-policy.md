# ADR 0005: Jetson CUDA Package Build and Version Policy

- Status: Accepted
- Date: 2026-03-13
- Decision makers: Daniel (user), Claude PM

## Context

Sprint 2 Jetson 실기기 추론 검증 과정에서, PyPI에서 설치한 여러 패키지가
Jetson Orin Nano (aarch64, CUDA 12.6)에서 CUDA를 올바르게 사용하지 못하는
문제가 발생했다. 구체적으로 두 가지 패키지에서 이슈가 확인되었다:

### CTranslate2 (Faster-Whisper 의존성)

- **PyPI wheel (`ctranslate2==4.7.1`)**: aarch64 빌드이지만 **CPU-only**.
  `ldd`로 확인 시 CUDA/cuDNN 라이브러리 링크 없음.
- **영향**: Faster-Whisper가 `device="cuda"`로 동작하지 못함.
  STT 추론이 CPU fallback으로만 가능 (성능 대폭 저하).
- **해결**: 소스 빌드 필요 (`-DWITH_CUDA=ON -DWITH_CUDNN=ON -DWITH_MKL=OFF`).

### llama-cpp-python (LLM 추론)

- **Jetson AI Lab 프리빌드 (`0.3.14`)**: Python 바인딩이 `kv_cache_seq_rm(seq_id=-1)`를
  호출하고, 번들 llama.cpp unified KV cache가 `seq_id >= 0`만 허용해
  `llama_kv_cache_unified::seq_rm` GGML_ASSERT 크래시가 발생할 수 있음.
- **영향**: generation 경로에서 `SIGABRT`가 발생할 수 있음. 모델 종류로 해결되는 문제가 아니다.
- **채택한 해결책**: Jetson AI Lab 프리빌드 0.3.14를 유지하고,
  `models/llm_runner.py`에서 `seq_id < 0`을 `0`으로 보정하는 런타임 호환 패치를 적용한다.
  이 동작은 llama-cpp-python 0.3.16 `_internals.py` 업스트림 수정과 동일하다.
- **장기 옵션**: 휠 자체에 업스트림 수정이 포함되게 하려면 0.3.16+ 소스 빌드를 선택할 수 있다.

### 추가 발견: Qwen3.5 아키텍처 미지원

- `Qwen3.5-2B` GGUF 파일의 `general.architecture`는 `qwen35`인데,
  llama-cpp-python 어떤 릴리스에서도 `qwen35` 아키텍처를 인식하지 못함 (llama.cpp b8233+ 필요).
- `Qwen3-1.7B` (`architecture: qwen3`)는 정상 동작.
- 향후 llama.cpp 업데이트로 해결될 수 있으나, 현재는 `qwen3` 계열 모델 사용.

> **2026-03-13 UPDATE (Sprint 3 Day 2 조사 결과)**:
> - llama-cpp-python v0.3.16이 PyPI 최신이나, Jetson AI Lab 프리빌드는 **v0.3.14까지** 제공.
>   프로젝트가 7개월간 업데이트 없이 사실상 중단 상태.
> - 번들 llama.cpp: ~b6170 (2025-08-14).
> - Qwen3.5는 Gated DeltaNet + Attention 하이브리드 아키텍처 (2026-02 출시).
>   llama.cpp **b8233** (2026-03-07) 이후에서만 CUDA 가속 지원.
>   현재 llama-cpp-python과의 빌드 갭: **~2,063 빌드**.
> - 선택지: (1) 소스 빌드 시 llama.cpp submodule을 b8233+ 핀 후 빌드 (~30분),
>   (2) llama.cpp C++ 바이너리 직접 사용, (3) 현재 Qwen3-1.7B 유지 (권장).
> - PM 권고: Qwen3.5 전환은 Sprint 4 이후 별도 기술 검토로 진행.
>
> **2026-04-02 UPDATE**: llama-cpp-python 0.3.17 (b8475) 소스 빌드 완료 (ADR 0029).
> b8475 > b8233이므로 **Qwen3.5 로딩 가능**. 위 "불가" 판정은 무효.
> → ADR 0034 (Qwen3.5 적용 가능성 평가) 참조.

## Decision

### 1. CTranslate2는 반드시 소스 빌드

PyPI wheel을 사용하지 않고, 다음 옵션으로 소스 빌드한다:

```bash
cmake .. \
  -DWITH_CUDA=ON \
  -DWITH_CUDNN=ON \
  -DWITH_MKL=OFF \
  -DCMAKE_BUILD_TYPE=Release

# Python 바인딩 설치 시 include/lib 경로 지정 필요
CFLAGS="-I<build>/include" \
LDFLAGS="-L<build>/build -Wl,-rpath,<build>/build" \
pip install --no-build-isolation .
```

빌드된 wheel은 `wheelhouse/`에 백업하여 재현성을 보장한다.

### 2. llama-cpp-python 공식 버전: 0.3.14

- Jetson AI Lab 프리빌드 0.3.14 사용 (소스 빌드 불필요).
- `models/llm_runner.py`에서 `seq_id < 0 -> 0` 호환 패치를 적용한다.
- 이 패치는 llama-cpp-python 0.3.16 업스트림 수정과 동작이 같다.
- 휠 레벨 수정이 필요할 때만 `LLAMA_CPP_VERSION=0.3.16` 소스 빌드를 선택한다.
- CUDA 빌드 시 반드시 `CMAKE_ARGS="-DGGML_CUDA=on"` 및
  `CUDACXX=/usr/local/cuda/bin/nvcc` 설정.
- 최적 성능을 위해 `CUDA_ARCHITECTURES=87` (Orin compute capability) 지정 권장.

### 3. Wheelhouse 관리 정책

Jetson 전용 소스 빌드 wheel은 `/opt/mungi-repo/wheelhouse/`에 보관한다.

| 패키지 | 파일명 | 비고 |
|--------|--------|------|
| ctranslate2 | `ctranslate2-4.7.1-cuda-cp310-linux_aarch64.whl` | CUDA+cuDNN 빌드 |
| llama-cpp-python | (직접 설치, wheel 미캐시) | 재빌드 시 ~30분 소요 |

Wheelhouse 파일은 Git에 커밋하지 않는다 (`.gitignore`).
재빌드 절차는 `docs/runbooks/`에 문서화한다.

### 4. CUDA Compute Architecture 지침

Jetson Orin Nano는 **compute capability 8.7**이다.

| 빌드 대상 | 권장 CUDA arch 설정 |
|-----------|-------------------|
| CTranslate2 | cmake 자동 감지 (compute_87 정상 인식) |
| llama-cpp-python | `CUDA_ARCHITECTURES=87` 또는 기본값 (80 fallback 허용) |

기본 빌드 (compute_50~89 전체)는 빌드 시간이 길지만 호환성이 높다.
Orin 전용 빌드 (87만)는 빌드 시간 단축 + 바이너리 크기 감소 + 성능 최적화.

## Consequences

### 긍정적

- 4개 AI 파이프라인 모델 (VAD, STT, LLM, TTS) 전부 Jetson CUDA에서 동작 확인.
- 소스 빌드 절차가 문서화되어 환경 재현 가능.
- Wheelhouse 백업으로 빌드 실패 시 즉시 복구 가능.
- Phase 0 최대 블로커 (CTranslate2 CUDA) 해결.

### 부정적

- 소스 빌드에 시간 소요 (CTranslate2 ~20분). llama-cpp-python은 프리빌드 사용으로 해소.
- CTranslate2 패키지 업데이트 시 소스 재빌드 필요.
- Wheel 파일은 JetPack 버전에 종속적 — JetPack 업그레이드 시 재빌드 필요.
- Qwen3.5 아키텍처 사용 불가 (llama.cpp 업데이트 대기).

## Verified Performance Baseline (2026-03-13)

Sprint 2 실기기 테스트로 확인된 성능:

| 단계 | 모델 | 로드 | 추론 성능 | 메모리 delta |
|------|------|------|-----------|-------------|
| VAD | Silero VAD (PyTorch) | 1.46s | RTF 0.32 | +17 MB |
| STT | Faster-Whisper Small (CUDA fp16) | 2.13s | RTF 0.28 | +723 MB |
| LLM | Qwen3-1.7B Q5_K_M (CUDA) | 1.32s | 27.1 TPS | +1,895 MB |
| TTS | Supertonic 2 (ONNX, CUDA EP) | 1.42s | RTF 0.67 | +477 MB |
| Safety | ContentFilter (CPU, JSON+regex) | <0.05s | <1ms/call | ~5 MB |

**총 추정 메모리: ~3.1 GB / 7.4 GB (여유 58%)**

> ContentFilter는 GPU 비사용 (CPU-only). JSON 설정 1회 로드 + regex 1회
> 컴파일. 메모리 영향 무시 가능. 아키텍처 결정은 ADR 0007 참조.

## Related documents

- ADR 0002 (Phase 0 baseline — onnxruntime 결론 수정됨)
- `docs/runbooks/weekly/archive/2026-03-13-sprint2-day2-worklog.md`
- `CLAUDE.md` §3 (Baseline Technical Stack)
- `CLAUDE.md` §6 (Architecture Constraints — 8GB 메모리 제한)
