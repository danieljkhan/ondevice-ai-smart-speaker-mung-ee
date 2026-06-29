# ADR 0082: Conversation Memory RAG — koen-e5-tiny ONNX Embedding (Plan-Level Alignment)

- **Status**: Accepted (plan-level)
- **Date**: 2026-04-29
- **Authority**: `docs/archived/dev-plan/2026-04-28-Plan-v21-v27-Update-Plan-v4.md` (Gate 1 final-approval, 2026-04-29)

## Context

The conversation memory RAG layer enables the agent to recall prior turns/sessions for continuity in child conversations. `CLAUDE.md §3` already declares the architecture:

- Embedding model: `koen-e5-tiny` ONNX (~30–60 MB)
- Index: FAISS, **separate** from the wiki RAG FAISS index (per `CLAUDE.md §6` source-contamination prevention)
- Prompt token budget: 100 tokens (independent of wiki RAG's 150-token budget)

Plan v2.1 did not document this layer, and several active runbooks/Dev_Plan docs still reference an older RAG architecture (FAISS + E5-small). This ADR ratifies the conversation-memory-specific architecture at plan level so subsequent Dev_Plan rewrites and the runtime build can cite a stable reference.

## Decision

Adopt the following architecture for conversation memory RAG:

1. **Embedding**: `koen-e5-tiny` ONNX model (Korean+English bilingual, tiny variant for Jetson 8 GB constraint), ~30–60 MB. Loaded via `onnxruntime`.
2. **Index**: FAISS, file `conversation_memory.faiss`. **Separate from** `wiki_faiss.index` (wiki RAG, 150-token budget).
3. **Prompt budget**: 100 tokens injected into the system/user prompt context. Independent of wiki RAG's 150-token budget. Total RAG context = 250 tokens combined.
4. **Storage location**: `/opt/mungi/ai_models/koen-e5-tiny-onnx/` for the ONNX bundle; `/var/lib/mungi/conversation_memory/` (or equivalent runtime path) for the FAISS index. Final paths confirmed at runtime build time.
5. **Index population**: conversation turns from `/var/lib/mungi/conversations/` (per `CLAUDE.md §6` permanent storage rule) are embedded and indexed asynchronously via a background timer (deferred to follow-up Plan; Session-12 P2 carryover).
6. **Source contamination prevention**: queries to wiki RAG and conversation memory RAG MUST NOT cross-pollinate; the indices are physically separate FAISS files and the embedding model is shared but the index handles are distinct.

## Consequences

### Positive
- Conversation continuity across sessions without retraining the LLM.
- Bilingual (Korean+English) embedding aligns with the bilingual product persona.
- Separate index prevents wiki facts from leaking into "what the child told me" recall.
- Memory footprint (~60 MB steady) is acceptable within Jetson 8 GB.

### Negative
- Embedding inference per turn adds latency (~tens of ms expected; runtime measurement pending).
- Index growth is unbounded over time; rotation/pruning policy is owed (deferred to runtime Plan).
- Separate FAISS index doubles the storage path complexity vs single shared index.

## Verification

Phased verification by PR landing:

**PR-1 (this ADR + initial truth-layer docs)**:
- `CLAUDE.md §3` lists item 6 "Conversation Memory RAG (koen-e5-tiny ONNX + separate FAISS index)".
- `CLAUDE.md §6` declares "The Conversation Memory FAISS index must be separate from the wiki FAISS index".
- `docs/runbooks/baseline-stack-and-models.md` mirrors §3 item 6.
- `docs/PROJECT_STATUS.md` AI pipeline ASCII + model table include the conversation memory RAG row referencing `koen-e5-tiny`.
- `docs/runbooks/jetson-setup-guide.md §7-6` documents the koen-e5-tiny bundle path placeholder pending ADR References pin.

**Deferred to a follow-up Phase A commit on the same PR-1 branch (4 Dev_Plan target files A4–A7)**:
- `docs/archived/dev-plan/Mungi_Development_Plan_v2_1_clean.md` architecture diagram + module tree include `koen-e5-tiny` and the conversation memory FAISS row.
- `docs/archived/dev-plan/Mungi_SW_Build_Plan_v2_7_Part2_Dev_Environment.md` requirements + dependency tables list `onnxruntime` for the koen-e5-tiny path.
- `docs/archived/dev-plan/Mungi_SW_Build_Plan_v2_7_Part3_Product_Build.md` `models/` directory listing adds `koen-e5-tiny-onnx/` + `conversation_memory.faiss`.
- The 4 Dev_Plan target files have not yet been rewritten as of this ADR's first landing on the PR-1 branch; A4–A7 remain pending in Plan v4 Phase A.

**Deferred to runtime build (separate Plan, beyond Plan v4)**:
- Concrete HuggingFace identifier and version pin for `koen-e5-tiny`.
- `conversation_memory.faiss` index population pipeline (background timer, Session-12 P2 carryover).
- Runtime latency measurements on Jetson.

## Open items (deferred to runtime build Plan)

- Concrete HuggingFace model identifier and version pin for `koen-e5-tiny`.
- Background indexing timer (Session-12 P2 carryover).
- Index rotation / pruning policy for unbounded growth.
- Latency measurement on Jetson runtime.

## Related

- `docs/archived/dev-plan/2026-04-28-Plan-v21-v27-Update-Plan-v4.md` §3 model stack mapping (RAG row added)
- `CLAUDE.md §3` (item 6) and `CLAUDE.md §6` (separate-index rule)
