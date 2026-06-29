# ADR 0033: E2E Topic Pool V2 — Vertex AI Batch Generation

- **Status**: Accepted
- **Date**: 2026-04-02
- **Author**: Claude Code PM (Opus 4.6)

## Context

The E2E test script (`scripts/e2e_60rounds.py`) uses a hardcoded topic pool of
146 topics (V1). To increase test coverage and prevent overfitting to a single
topic set, a second independent topic pool (V2) with 150 new topics is needed.

Manual topic creation is slow and prone to bias. Automated generation via LLM
ensures diversity and consistency while respecting the exclusion list.

## Decision

### Generation Method
- **API**: Vertex AI Batch Prediction API (global endpoint)
- **Model**: `gemini-3-flash-preview`
- **Endpoint**: `aiplatform.googleapis.com/v1/projects/.../locations/global/...`
  (regional endpoints like `us-central1` return 404 for Gemini 3.x preview models)

### Topic Pool Design
- 150 topics, each with 3 seed messages in Korean informal speech (반말)
- 12 categories: animals(20), nature/science(15), food(10), play/toys(10),
  family/friends(15), emotions(15), school(10), imagination(15), daily life(15),
  seasons/weather(10), body/health(10), vehicles(5)
- Zero overlap with V1 (146 topics excluded)
- Turn schedule unchanged: R1-10=3t, R11-20=4t, R21-30=5t, R31-40=6t, R41-50=7t, R51-60=8t

### Execution Summary
- Round 1: 12 batch requests → 150 topics, 16 V1 overlaps + 4 duplicates
- Supplement 1: 25 topics → 18 used, reached 146
- Supplement 2: 10 topics → output truncated, 0 used
- Supplement 3: 6 topics → 4 used, reached **150 (PASS)**
- Final validation: 150 unique topics, 450 messages, 0 V1 overlap, 0 duplicates

### Runtime Integration
V2 topics are loaded from `assets/training/e2e_topic_pool_v2.json` at runtime,
replacing `TOPIC_POOL` in `scripts/e2e_60rounds.py` via monkey-patch. A formal
`e2e_60rounds_v2.py` script will be created in a future iteration.

## Alternatives Considered

1. **Manual topic creation** — Rejected: too slow for 150 topics, human bias
2. **Anthropic Messages Batches API** — Considered, but Vertex AI already set up
   and cheaper for this task ($0.01 vs $0.05 estimated)
3. **gemini-3.1-flash-lite-preview** — 404 on both regional and global endpoints;
   may require project allowlisting
4. **gemini-2.5-flash-lite** — Works on us-central1, but gemini-3-flash-preview
   produces higher quality Korean output on global endpoint

## Consequences

- V2 topic pool stored at `assets/training/e2e_topic_pool_v2.json` (150 topics)
- E2E tests can now alternate between V1 and V2 topic pools
- V2 10-round test: 100% success, avg 5.3s total (comparable to V1 4.98s)
- Quality issues remain: hallucination 16.7% — RAG coverage expansion needed
- Batch generation script: `scripts/generate_e2e_v2_batch_input.py`
- Total cost: < $0.01 (4 batch jobs, ~30K tokens)
