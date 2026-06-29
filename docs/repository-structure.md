# Repository Structure

> Extracted from `CLAUDE.md §4`. Rule authority: `CLAUDE.md`.

## 4. Repository Structure (Current)

```text
mungi/
  core/
    conversation_memory.py       # Conversation memory RAG (separate FAISS index)
  models/
  safety/
  hardware/
  parental/
  assets/
    filters/
    prompts/
    sounds/
  scripts/
  systemd/
  tests/
  docs/
    adr/
    agents/
      instructions/
    runbooks/
      weekly/
      releases/
    templates/
  ai_models/
  Dev_Plan/                      # Authoritative requirements source + dev plan docs
    requirements-core.txt
    requirements-jetson.txt
    requirements-dev.txt
  Design/                        # Design documents
  PPT/                           # Presentation materials
  .github/
    workflows/
  .claude/
    hooks/                       # Verification-enforcement hooks (e.g. enforce_verification.py)
  .codex/
    config.json                  # Codex CLI settings (model, sandbox, rules)
    current-task.md              # Claude → Codex task spec
    chat/
      run-task.py                # thin wrapper → mungi-codex-plugin/scripts/run-task.py
      codex-output.log           # Real-time Codex output (tail -f target, runtime-generated)
      handoff.md                 # Codex deliverable (Claude reads, runtime-generated)
      status.json                # Execution state tracking (runtime-generated)
      history.jsonl              # Task execution history (runtime-generated)
  requirements-core.txt          # → redirects to Dev_Plan/ via -r
  requirements-jetson.txt        # → redirects to Dev_Plan/ via -r
  requirements-dev.txt           # → redirects to Dev_Plan/ via -r
  requirements-ci.txt            # CI-only dependencies
  pyproject.toml                 # ruff, mypy, pytest configuration
  README.md
```
