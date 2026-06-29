# ADR 0057: Jetson passwordless sudo for page-cache drop as mandatory E2E prerequisite

- **Status**: Accepted
- **Date**: 2026-04-15

## Context

Mungi E2E validation runs on Jetson via non-interactive SSH commands launched from the PM
workstation. The E2E helpers scripts/e2e_60rounds_text_tts.py and scripts/e2e_qwen3_asr_mix.py run a
mandatory preflight that drops the Linux page cache so CUDA cudaMalloc on Jetson unified memory can
reclaim enough MemFree for full LLM GPU offload. That preflight call uses subprocess.run with sudo
-n tee, which fails silently when passwordless sudo is not configured (only a warning is logged and
cache stays undropped). During the 2026-04-14 Wave 1 Jetson validation session, the first resident-
mode run demonstrated a 1.6 s LLM regression vs the baseline T1.2-only run. Three-layer root cause
traced the regression to: (1) proximate — LLM loaded at n_gpu_layers=10 because _select_gpu_layers()
saw MemFree=1098 MB which was below the partial threshold (2500 MB); (2) intermediate — STT resident
held approximately 1 GB of memory so LLM load had less headroom; (3) distal (true root cause) —
_drop_page_cache() failed because sudo -n tee /proc/sys/vm/drop_caches was not permitted. Fixing the
third layer alone (adding the sudoers entry) without any code change recovered MemFree to 3579 MB at
LLM load time, LLM loaded at n_gpu_layers=-1 (full offload), and first_sound dropped from 13.767 s
to 11.004 s. The decision to require passwordless sudo therefore codifies a system-setup practice
that, from a code perspective, is invisible but operationally decisive.

## Decision

Configure a narrow-scope passwordless sudo entry on every Jetson that runs Mungi E2E measurements,
limited to the single command /usr/bin/tee /proc/sys/vm/drop_caches. The setup is maintained via
/etc/sudoers.d/mungi-drop-caches with exactly one line: mungi ALL=(ALL) NOPASSWD: /usr/bin/tee
/proc/sys/vm/drop_caches. This is a MANDATORY prerequisite for every Jetson that participates in E2E
measurement — documented in CLAUDE.md section 2 Passwordless sudo for page cache drop subsection.
Without this configuration, the E2E preflight helper _preflight_drop_page_cache cannot reclaim
kernel page cache on non-interactive SSH sessions; MemFree stays depleted after prior runs; the LLM
falls back from full GPU offload (n_gpu_layers=-1) to safe partial offload (n_gpu_layers=10) during
load; and per-turn LLM latency roughly doubles. The 2026-04-14 Wave 1 validation measured this
effect precisely: without passwordless sudo, avg_first_sound_s was 13.767 (Wave 1 gate missed by
1.77 s); with it, same workload ran at avg_first_sound_s = 11.004 (Wave 1 gate passed with 1.0 s
margin).

## Consequences

Positive: (a) Wave 1 first_sound and total gates are reliably achievable on Jetson when the sudoers
entry is in place; (b) future E2E measurement sessions do not need ad-hoc manual sudo cache drops
before each run, which previously added an error-prone dependency on interactive operator input; (c)
the runtime _drop_page_cache() path called from check_memory_health works end-to-end instead of
silently skipping; (d) the scope of granted privilege is narrow (a single tee write to one kernel
pseudo-file) so the blast radius of a compromised mungi account is unchanged in practice. Trade-offs:
(a) the setup adds a one-time system-level prerequisite that must be applied to every Jetson used
for Mungi E2E — operators new to the project must run the install one-liner before they can
reproduce Wave 1 numbers; (b) the sudoers entry, while narrow, is still elevated privilege that must
be reviewed if sysadmin policies change; (c) alternative approaches exist (tuning
llm_partial_offload_memfree_mb down to 1000; reducing STT resident footprint; running E2E as root;
abandoning STT resident entirely) but all of them carry worse trade-offs than a scoped passwordless
sudo line. Operational impact: CLAUDE.md section 2 now documents the PowerShell one-liner installer.
ssh -t is required so sudo can prompt for the user's password during initial setup; after that, all
subsequent sudo tee /proc/sys/vm/drop_caches calls bypass the prompt. visudo -c is used for syntax
validation. The setup is verified with sudo -n tee /proc/sys/vm/drop_caches < /dev/null which must
print PASSWORDLESS_SUDO_OK during installation.

## Related ADRs

- 0052 — Jetson ALSA default routing override (similar Jetson system-config decision)
- 0055 — STT engine Qwen3-ASR adoption (Wave 1 baseline)

## References

- CLAUDE.md section 2 Passwordless sudo for page cache drop
- docs/runbooks/weekly/archive/2026-04-14-daily-worklog.md Part 6 section 16 (regression RCA and fix)
- docs/archived/dev-plan/2026-04-14-E2E-Bottleneck-Improvement-Plan.md section 4.1 (STT resident + MemAvailable guard)
- scripts/e2e_60rounds_text_tts.py _preflight_drop_page_cache
- core/model_manager.py _select_gpu_layers / _drop_page_cache / _ensure_cuda_memory
- docs/runbooks/weekly/archive/2026-04-14-e2e-wave1-report.md (thermal and memory evidence)
