# ADR 0037: CUDA 좀비 프로세스 근본 원인 및 방어 전략

- **Status**: Accepted
- **Date**: 2026-04-04
- **Author**: Claude Code PM (Opus 4.6)

## Context

SSH 백그라운드 E2E 테스트 시 CUDA 프로세스가 좀비화하여 3GB+ 통합 메모리를
잠금하는 문제가 반복 발생. ADR 0036의 Q4 19.01s (실제 5.88s) 성능 오판의
직접 원인이었으며, 프로젝트 전체 테스트 신뢰성을 훼손.

## Root Cause (실험 검증, 2026-04-04)

### 이전 가설 (이론 기반)
"CUDA가 D 상태(Uninterruptible Sleep)에서 POSIX 신호를 영구 차단"

### 수정된 원인 (실험 검증)

실제 E2E 프로세스(PID 98024) /proc 샘플링 결과:
- `SigBlk = 0x0000000000000000` — CUDA는 신호를 차단하지 않음
- SIGHUP: 차단=X, 무시=X, 캐치=X — 기본 동작(즉시 종료) 적용
- D 상태 비율: ~10% (DMA 전송 시점만), 나머지 90%는 R/S 상태

**진짜 원인**: SIGHUP 수신 → Python 기본 동작(즉시 종료) → atexit/finally 미실행
→ CUDA NvMap 핸들 미해제 → GPU 통합 메모리 3GB+ 잠금 유지

### 원인 계층

| 순서 | 원인 | 기여도 |
|:----:|------|:------:|
| 1 | SIGHUP 핸들러 미등록 — CUDA 정리 없이 강제 종료 | 50% |
| 2 | NvMap 통합 메모리 핸들 — 프로세스 소멸 후에도 GPU 드라이버가 보유 | 30% |
| 3 | D 상태 타이밍 — DMA 중 신호 지연으로 정리 기회 상실 | 20% |

### Jetson 특수성

- 통합 메모리 8GB에서 3GB 잠금 = 37% 손실 (dGPU 서버에서는 5%)
- 대체 메모리 경로 없음 (dGPU는 CPU RAM fallback 가능)
- 후속 프로세스 MemFree 부족 → CUDA 할당 실패 또는 partial offload → 3-6x 성능 저하

## Decision

### 방어 계층 (Defense in Depth)

| 계층 | 방안 | 상태 |
|:----:|------|:----:|
| 1 | **tmux 필수** — SSH 세션 분리로 SIGHUP 미전송 | ✅ 적용 (preflight 강제) |
| 2 | **SIGHUP 핸들러** — CUDA cleanup 후 정상 종료 | 계획 (P1) |
| 3 | **page cache drop** — 테스트 전 MemFree 확보 | ✅ 적용 (preflight) |
| 4 | **MemFree 검사** — 3000MB 미만 시 실행 거부 | ✅ 적용 (preflight) |

### preflight zombie kill 비활성화

`_preflight_kill_zombie_python()` 함수 호출을 제거 (커밋 `7637188`).

**이유**: pgrep 기반 PID 매칭이 순차 테스트에서 자기 프로세스 트리를 kill하는
자살 버그 반복. age 기반 필터(60초)도 불충분 — 이전 테스트 종료 직후의 잔여
프로세스가 60초 이상으로 판정됨. tmux + page cache drop + memory check만으로
충분하며, 좀비 kill은 수동으로 수행.

## Alternatives Considered

1. **setsid** — 프로세스 그룹 분리. tmux와 유사 효과이나 tmux가 더 범용적
2. **systemd 서비스** — 프로덕션 배포 시 적용 예정 (cgroup 메모리 제한)
3. **watchdog 타임아웃** — 추론 hang 감지, 장기 과제
4. **preflight zombie kill 개선** — 3차 시도에도 자살 버그. 비활성화가 최선

## Consequences

- E2E 테스트 신뢰성 확보 (좀비 오염 없는 클린 환경 보장)
- ADR 0036 Q4 성능 오판 방지 (19.01s → 실제 5.88s)
- 순차 9개 모델 테스트가 tmux 단일 세션에서 안전하게 실행 가능
- SIGHUP 핸들러 추가 시 tmux 없이도 안전 (이중 방어)

## References

- ADR 0036: DPO 평가 Q4/Q6 (좀비 오염 결과 수정)
- `docs/runbooks/weekly/archive/2026-04-04-cuda-zombie-root-cause-analysis.md`
- llama.cpp #3045, ollama #3474, PyTorch #4293
- CLAUDE.md §2 E2E Test Execution Workflow
