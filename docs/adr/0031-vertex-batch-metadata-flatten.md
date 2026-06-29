# ADR 0031: Vertex AI Batch Input Metadata Flattening

- **Status**: Accepted (historical; Vertex path superseded for Phase A by ADR 0092 v1.5 amendment — Gemini API direct)
- **Date**: 2026-04-01
- **Author**: Claude Orchestrator (Opus 4.6)

## Context

Vertex AI batch prediction API requires `metadata` field values to be primitive
strings. Our batch input JSONL files contained nested objects: `{"key": "78"}`.
Two batch jobs failed with metadata validation errors.

## Decision

Created `scripts/fix_batch_metadata.py` to:
1. Download JSONL from GCS
2. Flatten `{"key": "N"}` → `"N"` for all metadata fields
3. Upload fixed files back to GCS
4. Submit replacement batch prediction jobs

CLI modes: `--dry-run`, `--fix-only`, full run.

## Results

- Wiki: 443,131 records flattened, SUCCEEDED in ~30 min
- RAG: 688 records flattened, SUCCEEDED in ~10 min
- KEEP/DROP distribution: Wiki 85,197 KEEP (19.2%), RAG 144 KEEP (20.9%)

## Consequences

- GCS files are now clean for any future batch re-submission
- Script is reusable for future batch jobs with different models
- Wiki KEEP results (85K English chunks) stored for future Korean translation pipeline
