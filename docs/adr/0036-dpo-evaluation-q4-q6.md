# ADR 0036: DPO 평가 — Q4/Q6 양자화 × SFT/DPO 교차 비교

- **Status**: Concluded
- **Date**: 2026-04-03
- **Author**: Claude Code PM

## Context

Colab에서 Qwen3-1.7B SFT (12,365건) + DPO (451 페어) 파인튜닝을 완료하고,
Q4_K_M과 Q6_K 양자화 × SFT/DPO 4가지 조합을 Jetson E2E 30라운드로 비교 평가.

## Decision

**Q6_K SFT를 프로덕션 모델로 채택.**

## Evaluation Results

| Model | Success | Avg Total | Avg LLM | Avg Tok | Peak |
|-------|---------|-----------|---------|---------|------|
| Q4_K_M (SFT) | 98.3% | 19.01s | 12.17s | 85 | 3,390MB |
| Q4_K_M-dpo | 100% | 15.21s | 9.74s | 93 | 3,354MB |
| **Q6_K (SFT)** | **100%** | **5.78s** | **2.08s** | **28** | 3,640MB |
| Q6_K-dpo | 100% | 7.12s | 2.91s | 46 | 3,672MB |

## Rationale

1. **Q6_K SFT 최적 성능**: 5.78s/턴, 2.08s LLM — 모든 모델 중 가장 빠름
2. **Q4 비정상**: 12-19s/턴으로 Q6 대비 5.9배 느림. 토큰 85개 (Q6의 3배) — GGUF 변환 시 thinking 토큰 포함 의심
3. **DPO 효과 양면적**: Q6에서 DPO 적용 시 응답 길어짐 (28→46 tok, +64%), 턴 시간 23% 증가. 응답 품질은 향상되나 속도 저하
4. **존댓말 0%**: 4개 모델 모두 FT 효과 확인
5. **메모리 안전**: 모든 모델 3,700MB 이하

## Alternatives Considered

1. **Q6_K-dpo** — 풍부한 응답이지만 23% 느림. 속도 우선 시 SFT 선택
2. **Q4_K_M-dpo** — Q4 중 최선이나 15.21s로 여전히 느림
3. **Q4_K_M** — 2턴 실패, 비정상 성능

## Post-Decision Update (2026-04-04): Q4 비정상 성능 원인 확정

### 근본 원인: 좀비 프로세스 메모리 오염

4/3 DPO 비교 테스트 시 SSH 백그라운드 실행(tmux 미사용)으로 CUDA 좀비 프로세스가
3GB+ 메모리를 점유한 상태에서 Q4 테스트가 실행됨. Q6는 Q4 이후 실행되어 좀비가
kill된 클린 환경에서 측정됨 → 불공정 비교.

### 재검증 결과 (tmux + preflight, 30R/120T)

| Model | Avg Total | Avg LLM | Avg Tok | Peak Mem |
|-------|:---------:|:-------:|:-------:|:--------:|
| **Q4_K_M (SFT)** | **5.88s** | **2.04s** | 32.4 | **3,480MB** |
| Q6_K (SFT) | 6.12s | 2.30s | 31.6 | 3,766MB |

- Q4가 Q6보다 **4% 빠르고 286MB 메모리 절약**
- ADR 0036의 "Q4 5.9배 느림" 결론은 **완전 무효**
- 양자화 수준(Q4 vs Q6)이 성능/품질에 미치는 영향은 **미미**

### 수정된 판정

**Q6_K SFT 프로덕션 유지, but Q4_K_M도 동등 후보.**

| 기준 | Q4_K_M | Q6_K | 우위 |
|------|:------:|:----:|:----:|
| 성능 | 5.88s | 6.12s | **Q4** (+4%) |
| 메모리 | 3,480MB | 3,766MB | **Q4** (-286MB) |
| 파일 크기 | 1.1GB | 1.4GB | **Q4** (-22%) |
| 양자화 품질 | 4-bit | 6-bit | Q6 (이론상) |
| 토큰 수 | 32.4 | 31.6 | 동일 |
| 존댓말 | 0% | 0% | 동일 |

현재 Q6_K를 프로덕션으로 유지하되, 메모리 부족 시 Q4_K_M으로 전환 가능.

### 교훈

E2E 테스트는 반드시 tmux + preflight(메모리 정리) 후 실행해야 함.
`run_preflight()` 함수가 코드에 강제화됨 (커밋 `d941e61`).

## Consequences

- `qwen3-1.7b.Q6_K.gguf` (SFT)를 Jetson 프로덕션 모델로 유지
- Q4_K_M도 동등 성능으로 확인됨 — 메모리 부족 시 대안
- ~~Q4 GGUF 변환 문제 별도 조사 필요~~ → 변환 문제 없음, 좀비 오염이었음
- DPO 모델은 응답 풍부함이 필요한 경우 대안으로 보관
- **E2E 테스트 필수 워크플로우**: tmux + preflight (CLAUDE.md §2 E2E Test Execution Workflow)

## Superseded By

**ADR 0038 + ADR 0039 + Mungi_Model_Selection_Report_v1.md (2026-04-05)**: 12개 모델 종합 비교 후
프로덕션 모델이 **Qwen3.5-2B-DPO Q6_K**로 변경됨. Qwen3-1.7B 계열은 2026-04-05 Jetson 정리 작업으로
전량 삭제됨 (ADR 0039 참조). 바이링구얼 모드 아키텍처는 ADR 0038, 선정 override 공식화 및
Jetson 모델 디렉터리 정리 기록은 ADR 0039 참조.
