# ADR 0011: PyTorch Runtime Dependency on Jetson

- **Status**: Accepted
- **Date**: 2026-03-17
- **Context**: Whether to include PyTorch in the production runtime

## Context

The Jetson Orin Nano Super has 8 GB of unified memory. PyTorch
consumes approximately 300–500 MB when loaded. Two core subsystems
depend on it:

1. **Silero VAD** — uses a torch model for voice activity detection.
   No stable ONNX export of Silero VAD is currently available.
2. **VRAM verification** — `torch.cuda.memory_allocated()` is used
   by the sequential GPU loading protocol (ADR 0009) to confirm GPU
   memory release before loading the next model.

Alternatives considered:

- **ONNX-only VAD**: No production-ready ONNX Silero export exists.
  Switching would require an unvalidated custom export or a
  different VAD engine.
- **Remove torch entirely**: Would break both VAD inference and
  VRAM monitoring, requiring replacements for two critical paths.

## Decision

Keep PyTorch as a runtime dependency on Jetson.

1. **Silero VAD requires torch** — the upstream model is distributed
   as a torch JIT archive with no stable ONNX alternative.
2. **VRAM monitoring requires torch.cuda** — the sequential GPU
   loading protocol relies on `torch.cuda.memory_allocated()` for
   double verification before model transitions.
3. **JetPack compatibility** — JetPack 6.2 ships PyTorch-compatible
   wheels (`torch` for aarch64 + CUDA 12.6). No custom build or
   vendor fork is needed.
4. **Acceptable memory cost** — the ~300–500 MB footprint is
   manageable because the sequential loading protocol (ADR 0009)
   guarantees only one large GPU model is resident at a time,
   leaving sufficient headroom within the 8 GB budget.

### Future consideration

If the Silero team releases a stable ONNX export of the VAD model,
the torch dependency can be re-evaluated. Removing it would reclaim
~300 MB of memory and simplify the dependency tree. VRAM monitoring
could be replaced with `/proc/meminfo` or `tegrastats` parsing.

## Consequences

- **Positive**: Retains proven VAD accuracy; preserves reliable
  VRAM monitoring; uses vendor-supported wheels with no build
  complexity.
- **Negative**: ~300–500 MB permanent memory overhead; adds a
  large transitive dependency tree (`torch`, `torchvision`,
  CUDA runtime libs).
- **Risks**: Future PyTorch version bumps may break JetPack wheel
  compatibility, requiring pinning or manual builds.

## Related

- ADR 0009: Sequential GPU loading (VRAM double verification)
- CLAUDE.md §3: Baseline technical stack (Silero VAD)
