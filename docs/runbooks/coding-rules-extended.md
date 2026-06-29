# Coding Rules Extended Reference

> Extracted from `CLAUDE.md §5`. Rule authority: `CLAUDE.md`.
> Post-mortem reference: `docs/runbooks/weekly/archive/2026-04-11-ci-mypy-env-asymmetry-postmortem.md`

- Optional dependencies: any lazy import of an optional runtime package
  (`soundfile`, `sounddevice`, `onnxruntime`, `sherpa_onnx`, `llama_cpp`,
  `faster_whisper`, `supertonic`, etc. — packages that are NOT in
  `requirements-ci.txt` but ARE in `requirements-core.txt` or
  `requirements-jetson.txt`) MUST use combined mypy ignore codes:
  ```python
  import soundfile as sf  # type: ignore[import-not-found, import-untyped]
  ```
  Rationale: CI runs on a clean `requirements-ci.txt` (module missing →
  `import-not-found`), while the local dev/Jetson venv has the package
  installed but without stubs (→ `import-untyped`). A single code covers
  only one environment and fails in the other. See post-mortem
  `docs/runbooks/weekly/archive/2026-04-11-ci-mypy-env-asymmetry-postmortem.md`
  for full root-cause analysis. Exception: a plain `import pkg` (no
  alias, no attribute access) may use `# type: ignore[import-not-found]`
  alone because mypy does not emit `import-untyped` for unaliased
  module references.
- Anthropic API: always use the Message Batches API (`messages.batches.create`) for bulk operations.
  Real-time `messages.create` is only allowed for interactive/single-request use cases.
  The Batch API provides a 50% cost reduction and is mandatory for data generation, batch processing, etc.
