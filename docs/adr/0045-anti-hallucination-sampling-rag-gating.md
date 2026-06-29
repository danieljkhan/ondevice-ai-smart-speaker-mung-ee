# ADR-0045: Anti-Hallucination — min-p Sampling Recalibration + RAG Gating

- **Status**: Accepted (updated 2026-05-10; Layer 2 retired by ADR 0085)
- **Date**: 2026-04-08
- **Context**: Qwen3.5-2B-DPO Q6_K on Jetson Orin Nano 8GB

## Problem

E2E 30-round bilingual test (v1~v3)에서 3가지 환각 패턴 확인:

1. **RAG 오염** — 무관한 RAG 청크가 프롬프트에 주입 (score_threshold=0.5 과도하게 허용적)
2. **사실 날조** — 2B 모델이 temperature=1.5에서 자신감 있게 허위 정보 생성
3. **맥락 혼선** — RAG 컨텍스트가 user 메시지에 합쳐져 모델이 구분 불가

## Decision

### Layer 1: 샘플링 파라미터 재조정

min-p 논문 (ICLR 2025, arXiv:2407.01082v8) 분석 결과:

- temp=1.5 + min_p=0.1은 **7B+ 모델의 창작 글쓰기** 최적값
- 2B 모델의 아동 팩트 Q&A에는 temp=1.0이 적절
- min-p와 top-p/top-k **동시 사용 금지** (double normalization 경고, Table 12)
- min_p=0.1을 유일한 truncation method로 사용 권장

| Parameter | Before | After | Rationale |
|-----------|--------|-------|-----------|
| temperature | 1.5 | 1.0 | 2B 모델 팩트 Q&A에 맞는 분포 |
| min_p | 0.1 | 0.1 | 논문 권장값 유지 |
| top_p | 1.0 | 1.0 | min-p 단독 사용 |
| top_k | 0 | 0 | min-p 단독 사용 |
| presence_penalty | 1.2 | 1.5 | QWEN35_PRESENCE_PENALTY에 맞춤, RAG 앵무새 방지 |
| repeat_penalty | 1.0 | 1.15 | 응답 내 반복 억제 (1.5는 과거 milk-fixation 버그) |

### Layer 2: RAG 게이팅 강화

> **RETIRED 2026-05-XX**: Layer 2 retired by ADR 0085 (Wiki RAG removal). The wiki
> RAG path no longer exists in the runtime as of PR 4-B; threshold gating is moot.
> Layers 1, 3, 4 remain active. See ADR 0085 §Decision for the rationale.

- `DEFAULT_SCORE_THRESHOLD`: 0.5 → 0.70
- 코사인 유사도 0.5는 "같은 도메인 다른 주제" 수준 → 무관한 청크 주입 원인
- 0.70 이상만 주입하여 확실히 관련 있는 컨텍스트만 허용
- 프리픽스에 "질문과 관련된 경우에만 참고" 조건 추가

### Layer 3: RAG-쿼리 구조 분리

Before:
```python
user_content = f"{rag_context}\n\n{user_text}"
messages.append({"role": "user", "content": user_content})
```

After:
```python
if rag_context:
    messages.append({"role": "system", "content": rag_context})
messages.append({"role": "user", "content": user_text})
```

모델이 참고자료(system)와 사용자 질문(user)을 명확히 구분.

### Layer 4: 시스템 프롬프트 강화

REFERENCE INFORMATION RULES 섹션 추가:
- 참고 정보가 다른 주제면 완전 무시
- 참고 정보와 지식을 섞어 새 주장 생성 금지
- 동물/음식 사실은 참고 정보에 없으면 언급 금지

## Consequences

### E2E v4 테스트 결과 (30 rounds, 120 turns)

| Metric | v3 (before) | v4 (after) |
|--------|-------------|------------|
| Success rate | 100% | 100% |
| RAG 오염 환각 | 5건+ (zip-line, Cinderella 등) | **0건** |
| 사실 날조 환각 | 3건+ (곤충, 긴 꼬리 등) | **1건** (경미) |
| Peak memory | 6,071 MB | **5,802 MB** (-269 MB) |
| EN avg tokens | 26.7 | 21.3 (불필요한 반복 감소) |

### 잔존 리스크

- 2B 모델의 근본적 지식 한계는 샘플링으로 완전 해결 불가
- 향후 모델 업그레이드(3B+) 또는 RAG 커버리지 확대로 보완 필요

## Related

- ADR-0044: Wiki Filter Thinking Budget Workaround
- ADR-0019: Wiki RAG Factual Grounding
- **ADR 0085**: Wiki RAG removal — retires Layer 2 of this ADR.
- min-p Paper: arXiv:2407.01082v8 (ICLR 2025)
