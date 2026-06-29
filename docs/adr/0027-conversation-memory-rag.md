# ADR 0027: Conversation Memory RAG via Separate FAISS Index

- **Status**: Accepted
- **Date**: 2026-03-27
- **Extends**: ADR 0019 (Wiki-based RAG for Factual Grounding)

## Context

Children naturally ask follow-up questions such as "what did we talk about
yesterday?" or "do you remember my dinosaur story?" The current pipeline can
restore short-term session context and store transcripts for parental review,
but it cannot semantically retrieve older conversation turns for reuse in the
prompt.

The feature must remain fully on-device, privacy-preserving, and compatible
with Jetson Orin Nano Super 8GB unified memory limits. The project already
adopted `koen-e5-tiny` ONNX embeddings for wiki RAG (ADR 0026), so adding a
second embedding stack for conversation memory would waste RAM and increase
startup complexity.

We also need to keep factual wiki grounding and personal conversation recall
separate. Mixing them into one shared index would blur source provenance,
complicate prompt formatting, and increase the risk of conversation snippets
being treated as factual evidence.

## Decision

Implement Conversation Memory RAG as a separate local FAISS index in
`core/conversation_memory.py`, while reusing the existing `koen-e5-tiny`
ONNX embedding session and tokenizer.

### 1. Separate Conversation Index

- Store conversation vectors in `/var/lib/mungi/conversations/conv_faiss.index`
- Store conversation chunk metadata in
  `/var/lib/mungi/conversations/conv_chunks.jsonl`
- Use a dedicated FAISS `IndexFlatL2` wrapped with `IndexIDMap` so old entries
  can be pruned with `remove_ids()`

### 2. Shared Embedding Runtime

- Reuse the same 384-dimension `koen-e5-tiny` embedding model used by wiki RAG
- Do not load a second embedding model or maintain duplicate tokenizer/session
  state
- Format conversation passages as:

```text
passage: {user_text} -> {mungi_text}
```

- Format retrieval queries as:

```text
query: {user_text}
```

### 3. Prompt Integration

- Retrieve top-2 conversation results per user turn
- Inject them into a dedicated `[이전 대화 기억]` prompt section
- Cap conversation memory prompt contribution at 100 tokens
- Keep this budget independent from the wiki RAG 150-token budget

### 4. Incremental Indexing and Pruning

- Embed and add turns after each session ends
- Expected indexing latency: ~1-2 seconds for a 10-turn session
- Apply a +20% recency boost to turns from the last 7 days
- Retain only the last 30 days of conversation-memory entries

## Alternatives Considered

### 1. Unified FAISS Index for Wiki + Conversation Data

Rejected. This mixes factual reference material with personal conversation
history, making source tagging and prompt separation harder. It increases the
risk of source contamination, where remembered dialogue is surfaced as if it
were factual grounding.

### 2. SQLite FTS5 Only

Rejected. Keyword-based full-text search is a poor fit for Korean morphology
and paraphrased child speech. Semantic recall of earlier conversations would be
weaker than vector retrieval.

### 3. Hybrid SQLite Metadata + FAISS Search

Rejected. This adds coordination complexity without a clear benefit for the
current 30-day bounded dataset size. The additional moving parts are not
justified for an on-device child companion MVP.

### 4. Sliding Window History Only

Rejected. A prompt-only sliding window can preserve only recent turns and
cannot answer requests about older sessions once they fall out of the active
context budget.

## Consequences

### Positive

- Enables long-term semantic recall of prior child conversations
- Preserves clear separation between factual wiki grounding and personal memory
- Adds almost no model-memory overhead because embeddings are shared
- Estimated conversation index footprint remains small at ~460 KB with a
  30-day cap
- Estimated retrieval latency stays below 10 ms per turn

### Negative

- Adds post-session indexing work after each conversation ends
- Requires pruning logic and ID bookkeeping through `IndexIDMap`
- Introduces another local runtime artifact that must remain aligned with
  transcript storage

### Neutral

- Conversation memory remains an optional retrieval path; the pipeline can
  degrade gracefully if the conversation index is unavailable

## Implementation

- New module: `core/conversation_memory.py`
- Pipeline integration: `core/pipeline.py`
- Embedding/runtime accessors: `core/rag_retriever.py`
- Rebuild utility: `scripts/build_conv_index.py`
- Tests: `tests/test_conversation_memory.py`

## References

- ADR 0019: Wiki-based RAG for factual grounding
- ADR 0021: Token-based adaptive history
- ADR 0026: 1.7B QLoRA + KoEn-E5-Tiny RAG optimization plan
