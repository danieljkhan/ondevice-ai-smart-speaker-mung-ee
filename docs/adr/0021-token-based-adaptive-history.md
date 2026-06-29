# ADR 0021: Token-based Adaptive History Management

- **Status**: Accepted
- **Date**: 2026-03-25
- **Context**: KV cache pressure management for multi-turn conversation on Jetson 8GB

## Context

The original history management used a fixed turn-count window
(`max_history_turns`). This approach has two problems on Jetson 8GB:

1. **Unbounded token growth**: Long user inputs or verbose LLM responses
   cause the history window to consume disproportionate tokens even with
   few turns, increasing KV cache pressure and LLM inference time.
2. **No runtime adaptation**: When LLM inference slows (indicating memory
   pressure or KV cache saturation), the fixed window does not respond.

E2E testing showed degradation starting around turn 54 (LLM time rising
from 2.8s to 7.2s), indicating KV cache buildup beyond the model's
efficient operating range.

## Decision

Replace pure turn-count history with a token-budget-capped, adaptive
history management system in `core/pipeline.py`.

### 1. Token Estimation

`_estimate_tokens(text)` (line 818): conservative heuristic using Korean
character density.

```python
def _estimate_tokens(self, text: str) -> int:
    return max(1, (len(text) + 2) // 3)
```

Approximately 1 token per 3 characters — conservative for Korean text
where actual tokenization yields ~2.5 chars/token. Avoids loading the
full tokenizer to save memory and latency.

### 2. History Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_history_turns` | 2 | Maximum turn-pairs in context window |
| `max_history_tokens` | 200 | Token budget for all history content |
| `max_history_entries` | 100 | Absolute cap on stored entries (ring buffer) |
| `adaptive_history_threshold_s` | 15.0 | LLM time threshold for adaptive reduction |

### 3. Adaptive Scaling

`_should_reduce_history()` (line 823): monitors last LLM inference time.

When LLM inference exceeds `adaptive_history_threshold_s` (15.0s):
- History window reduces from `max_history_turns` to 1 turn-pair
- Provides immediate relief for KV cache pressure
- Automatically restores when inference time normalizes

### 4. Token Budget Enforcement

During `_build_prompt()` (line 853), after selecting turn-pairs within the
turn window:

1. Compute token estimate for each turn-pair
2. Accumulate from most recent to oldest
3. Drop oldest turn-pairs when cumulative tokens exceed `max_history_tokens`

This ensures the prompt stays within a predictable size regardless of
individual turn verbosity.

### 5. Removal Order

FIFO within the window: oldest conversation pairs are removed first,
preserving the most recent context for conversational coherence.

## Consequences

- Predictable prompt size regardless of turn content length
- Automatic adaptation to memory pressure without manual intervention
- Conservative token estimation may over-prune history in some cases
  (slightly fewer turns retained than actual token budget would allow)
- Does not address KV cache management at the llama.cpp level — the
  model's internal KV cache can still grow within a single turn
- Late-session degradation (turn 54+) indicates additional optimization
  may be needed at the inference engine level

## References

- ADR 0009: Sequential GPU loading (memory management context)
- ADR 0012: LLM upgrade to Qwen3-4B (model memory characteristics)
- ADR 0014: Qwen3 chat template (prompt structure)
- ADR 0016: Unified memory stage unload (Jetson memory budget)
- `core/pipeline.py:818`: `_estimate_tokens()` implementation
- `core/pipeline.py:823`: `_should_reduce_history()` implementation
- `docs/runbooks/weekly/archive/2026-03-25-e2e-text-tts-60round-summary.md`
