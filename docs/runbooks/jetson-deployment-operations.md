# Jetson Deployment Operations

> Extracted from `CLAUDE.md §2`. These are NON-NEGOTIABLE operational procedures.
> Rule authority: `CLAUDE.md`. This document is the detailed reference.

### Jetson Deployment Verification (NON-NEGOTIABLE)

Every deployment to Jetson must pass these 3 gates before proceeding to tests:

**Gate 1: Completeness check** — After file transfer, compare the remote file list against the source:
```bash
ssh jetson "ls /deployed/path/" | sort
ls /local/source/path/ | sort
# Must match (excluding build artifacts like .cache, ggml-*)
```

**Gate 2: Load test** — Verify the deployed model/data actually loads on Jetson:
```bash
# For ML models (SentenceTransformer, ONNX, GGUF):
ssh jetson "cd /opt/mungi-repo && scripts/mungidev.sh python -c \"<load test code>\""
# For FAISS indexes:
ssh jetson "python -c \"import faiss; idx = faiss.read_index('/path/index'); print(idx.ntotal)\""
```
If the load test fails, fix before proceeding. Do NOT start E2E tests on an unverified deployment.

**Gate 3: Startup log scan** — After launching any long-running test, check the first 30 lines of the log for:
- `error`, `unavailable`, `inactive`, `FAIL`, `exception`, `cannot open`
- If ANY warning/error related to the deployment is found → **stop immediately**, diagnose, fix, restart.
- Do NOT let a 60-round test run with a silently broken component.

Skipping any gate is a process violation.

### E2E Test Execution Workflow (NON-NEGOTIABLE)

Every SSH remote E2E test on Jetson must follow this workflow:

1. **tmux required** — All E2E tests MUST run inside a tmux session. Direct SSH background execution (`nohup &`, `bash -c &`) causes CUDA zombie processes (3GB+ memory leak) when SSH disconnects.
   ```bash
   tmux new-session -d -s e2e 'cd /opt/mungi-repo && source .venv/bin/activate && python3 scripts/e2e_60rounds_text_tts.py ...'
   ```

2. **Kill zombie processes** — Before starting, check for and kill orphaned Python/CUDA processes:
   ```bash
   ps aux | grep 'python.*e2e' | grep -v grep
   # Kill any found zombies
   ```

3. **Drop page cache** — Free kernel page cache for CUDA memory:
   ```bash
   sync && echo 1 | sudo -S tee /proc/sys/vm/drop_caches > /dev/null 2>&1
   ```

4. **Memory check** — Verify MemFree ≥ 3000 MB before starting:
   ```bash
   free -m | head -2
   ```

The `e2e_60rounds_text_tts.py` script enforces these checks automatically via `run_preflight()`. On non-Linux or CI environments, use `--skip-preflight` to bypass.

**Root cause**: SSH background CUDA processes block POSIX signals → zombie on disconnect → 3GB+ memory leak → subsequent tests run with insufficient memory → 3-6x performance degradation (confirmed 2026-04-03: Q4 19.01s → 6.77s after fix).

### SSH Auto-tmux Persistence (deployed 2026-04-13)

Jetson `~/.bashrc` includes an auto-attach snippet (installed via `scripts/install_jetson_ssh_tmux.sh`) so every interactive SSH login lands inside a persistent `mungi` tmux session. Detach (`Ctrl+A D`) closes SSH but keeps the session running on Jetson; reconnect resumes the same session. Routine work uses windows (`Ctrl+A C`) inside `mungi`; long-running or high-risk jobs (60R+ E2E, 72h training, production deploys, GPU-exclusive experiments) get dedicated sessions with the `deploy-*` / `e2e-*` / `pilot-*` / `train-*` naming convention.

- **Prefix is `C-a`** (not the tmux default `C-b`). Reason: the Windows PowerShell SSH client does not reliably forward `Ctrl+B` to remote tmux, and VS Code's integrated terminal binds `Ctrl+B` to the sidebar toggle. `C-a` (the GNU screen prefix) avoids both conflicts.
- **Auto-tmux does NOT replace the preflight checks above.** The preflight script still verifies `$TMUX` is set, page cache is dropped, and MemFree is sufficient. Auto-attach simply removes the operator burden of remembering `tmux new-session -d -s ...` before every long run.
- **Full setup, usage patterns, and rollback procedure**: `docs/runbooks/2026-04-13-jetson-ssh-tmux-persistence.md`.

### Passwordless sudo for page cache drop (MANDATORY for Jetson, configured 2026-04-14)

The Jetson preflight drop_caches step (`_preflight_drop_page_cache` in `scripts/e2e_60rounds_text_tts.py`) uses `sudo -n tee /proc/sys/vm/drop_caches`, which requires passwordless sudo for that specific command. Without it, page cache cannot be dropped via non-interactive SSH, MemFree stays depleted after previous runs, and LLM load falls back to `n_gpu_layers=10` (safe partial offload), which roughly doubles per-turn LLM latency versus full GPU offload.

Impact observed (2026-04-14 Wave 1 validation): with passwordless sudo NOT configured, `avg_first_sound_s = 13.767` (Wave 1 gate miss by 1.77 s). After configuring it, the same workload yielded `avg_first_sound_s = 11.004` (gate PASS). The difference came entirely from LLM loading with `n_gpu_layers=-1` (full offload) instead of `=10`.

**Required sudoers line** (created via `sudo visudo -f /etc/sudoers.d/mungi-drop-caches`):

```
mungi ALL=(ALL) NOPASSWD: /usr/bin/tee /proc/sys/vm/drop_caches
```

After saving, `sudo visudo -c` must report `parsed OK`, and `sudo -n tee /proc/sys/vm/drop_caches < /dev/null` must succeed without prompting for a password. This single line grants passwordless access ONLY for writes to `/proc/sys/vm/drop_caches`; all other `sudo` commands still require the normal password. Security impact is negligible because `drop_caches` writes do not touch persistent state — the kernel simply discards the page cache.

**One-liner install (PowerShell/Windows, TTY allocated for password prompt):**

```powershell
ssh -t mungi@jetson.local "echo 'mungi ALL=(ALL) NOPASSWD: /usr/bin/tee /proc/sys/vm/drop_caches' | sudo tee /etc/sudoers.d/mungi-drop-caches && sudo chmod 0440 /etc/sudoers.d/mungi-drop-caches && sudo visudo -c && sudo -n tee /proc/sys/vm/drop_caches < /dev/null && echo PASSWORDLESS_SUDO_OK"
```

Operator verification: `PASSWORDLESS_SUDO_OK` must appear in the output. If not, the E2E workflow is not correctly configured — investigate before proceeding with Jetson measurements.
