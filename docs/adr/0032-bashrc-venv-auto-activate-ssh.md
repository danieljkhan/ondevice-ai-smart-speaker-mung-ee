# ADR 0032: Bashrc Venv Auto-Activation for Non-Interactive SSH

- **Status**: Accepted
- **Date**: 2026-04-02
- **Author**: Claude Code PM (Opus 4.6)

## Context

When running commands on the Jetson via SSH (`ssh mungi@jetson.local "command"`),
the Python virtual environment at `/opt/mungi-repo/.venv/` was not activated.
This caused `ModuleNotFoundError: No module named 'torch'` when executing E2E
test scripts remotely.

**Root cause**: Jetson's `~/.bashrc` contained a non-interactive shell guard at
the top:

```bash
case $- in
    *i*) ;;
      *) return;;
esac
```

SSH remote commands run in non-interactive mode, so `.bashrc` returned immediately
before reaching any venv activation or `mungidev` function definitions. The system
Python (`/usr/bin/python3`) was used instead of the venv Python, which lacks torch
and other ML dependencies.

## Decision

Insert the venv auto-activation block **before** the non-interactive guard in
`~/.bashrc`:

```bash
# === Mungi venv auto-activate (interactive + non-interactive SSH) ===
if [ -f /opt/mungi-repo/.venv/bin/activate ]; then
    source /opt/mungi-repo/.venv/bin/activate
fi

# If not running interactively, don't do anything
case $- in
    *i*) ;;
      *) return;;
esac
```

This ensures that both interactive terminals and non-interactive SSH commands use
the correct Python environment with all ML dependencies available.

## Alternatives Considered

1. **Always prefix SSH commands with `source .venv/bin/activate &&`**
   - Rejected: error-prone, requires every caller to remember the prefix.

2. **Use `.venv/bin/python3` absolute path directly**
   - Rejected: does not set `LD_LIBRARY_PATH` or other env vars from activation.

3. **Use `~/.ssh/environment` file**
   - Rejected: requires `PermitUserEnvironment yes` in sshd_config (security concern),
     and cannot run `source` commands.

4. **Use `ForceCommand` in sshd_config**
   - Rejected: too invasive, affects all SSH sessions.

## Consequences

- All SSH remote commands now use `/opt/mungi-repo/.venv/bin/python3` automatically
- `torch`, `onnxruntime`, `sentence-transformers`, and all other venv packages
  are available in non-interactive SSH sessions
- Verified: `ssh mungi@jetson.local "which python3"` returns
  `/opt/mungi-repo/.venv/bin/python3` and `torch 2.10.0` imports successfully
- No impact on interactive terminal sessions (venv was already activated via
  `mungidev` function)
- If the venv path changes, only one line in `.bashrc` needs updating
