# ADR-0044: Wiki Filter Gemini Thinking Budget Workaround

- **Status**: Accepted
- **Date**: 2026-04-08
- **Context**: Vertex AI batch prediction for wiki KEEP/DROP classification

## Problem

Gemini 2.5 Flash의 thinking 모드가 `maxOutputTokens` 한도 내에서 내부 추론(thinking) 토큰을
먼저 소비하여, 실제 출력(KEEP/DROP)이 생성되지 않는 문제 발생.

353,596건 배치에서 **50.7% (179,363건)**가 텍스트 응답 없이 score만 반환됨.

### 증상

```json
{
  "candidates": [{
    "content": {"role": "model"},
    "finishReason": "MAX_TOKENS",
    "score": -0.05098
  }],
  "usageMetadata": {
    "promptTokenCount": 69,
    "thoughtsTokenCount": 2,
    "totalTokenCount": 71
  }
}
```

- `thoughtsTokenCount: 2` → thinking에 2토큰 소비
- 출력 토큰: **0개** (KEEP/DROP 미출력)
- `finishReason: "MAX_TOKENS"` → 토큰 한도 도달로 종료

## Decision

### 1차 시도: `maxOutputTokens` 증가 (5 → 20)

- 결과: 179,363건 중 **16,158건(9%)만 추가 판정** 획득
- 163,205건 여전히 미판정 → thinking이 20토큰 내에서도 출력을 잡아먹음

### 2차 시도: `thinkingBudget: 0` 설정

```json
{
  "generationConfig": {
    "temperature": 0.0,
    "maxOutputTokens": 20,
    "thinkingConfig": {"thinkingBudget": 0}
  }
}
```

- thinking 완전 비활성화하여 출력 토큰 전부를 KEEP/DROP 응답에 할당
- 163,205건 재배치 제출 (Job: `wiki-filter-retry-notext-nothink-163k`)

## Consequences

### Score 기반 판정은 부적합

텍스트 응답이 있는 항목으로 score 분포 분석 결과:

| 판정 | score median | score range |
|------|-------------|-------------|
| KEEP | -0.2551 | -1.01 ~ -0.02 |
| DROP | -0.1642 | -0.99 ~ -0.01 |

→ KEEP/DROP 간 score 분포가 크게 겹쳐 **score만으로 KEEP/DROP 분류 불가능**.

### 향후 배치 설정 권장사항

Gemini 2.5 Flash로 단답형(KEEP/DROP) 분류 배치 실행 시:

1. **`thinkingBudget: 0`** 필수 설정 (단답형에 thinking 불필요)
2. `maxOutputTokens: 10~20` (여유분 확보)
3. `temperature: 0.0` (결정론적 응답)

## Related

- `scripts/apply_rag_batch_results.py` — 배치 결과 적용 파이프라인
- `Dev_Plan/submit_vertex_wiki_filter.py` — 배치 제출 스크립트
- ADR-0019: Wiki RAG Factual Grounding
- ADR-0031: Vertex Batch Metadata Flatten
