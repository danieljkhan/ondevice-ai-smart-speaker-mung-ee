# ADR 0047: JetPack 6.2.2 Upgrade with NVMe SSD Boot

**Status**: Accepted  
**Date**: 2026-04-09  
**Author**: Claude Code PM (Opus 4.6)  
**Related**:
- ADR 0002 — Phase 0 baseline (JetPack 6.2, CUDA 12.6)
- ADR 0039 — Model cleanup (Jetson storage optimization)
- ADR 0040 — Gemma 4 candidate evaluation
- `docs/archived/dev-plan/2026-04-09-JetPack-622-NVMe-Upgrade-Plan.md` — Full upgrade plan

## Context

Mungi runs on Jetson Orin Nano Super 8GB with JetPack 6.2.1+b38. JetPack 6.2.2 is
required for Gemma 4 model support, which is the next-generation LLM candidate for Mungi
(see ADR 0040).

The current system boots from the default storage device (SD card / eMMC). NVMe SSDs
provide significantly better I/O performance (sequential read ~3,000 MB/s vs ~100 MB/s
for SD cards), which benefits model loading times and overall system responsiveness.

The user also wants to consolidate boot and runtime onto a single NVMe SSD, eliminating
the SD card dependency entirely.

## Decision

Upgrade from JetPack 6.2.1 to **JetPack 6.2.2** via SDK Manager full flash, with
**NVMe SSD as the sole boot and storage device**.

Key parameters:
- **Flash target**: NVMe SSD (not eMMC, not SD card)
- **Boot mode**: SSD-only boot (bootloader + OS + CUDA + data all on NVMe)
- **Host environment**: Ubuntu 22.04 Live USB (Windows-only constraint)
- **Backup strategy**: Full backup to host PC / external drive before flash

## Alternatives Considered

### 1. OTA upgrade (`apt dist-upgrade`)
- **Pros**: No reflash needed, preserves all data
- **Cons**: May not deliver 6.2.2 (only patch-level updates), cannot change boot device
- **Rejected**: Does not achieve NVMe boot transition

### 2. SDK Manager via WSL2 on Windows
- **Pros**: No Ubuntu host PC needed
- **Cons**: NVMe flash is NOT supported via WSL2 (eMMC only, confirmed via NVIDIA docs)
- **Rejected**: Our primary goal is NVMe boot

### 3. SDK Manager via VM (VMware/VirtualBox)
- **Pros**: Runs on any host OS
- **Cons**: USB passthrough unreliable, NVMe flash timeout risk, NVIDIA unsupported
- **Rejected**: Too fragile for a destructive operation

### 4. Ubuntu Live USB + SDK Manager (Selected)
- **Pros**: Native Ubuntu environment, full NVMe support, no Windows modification
- **Cons**: Must complete in one session (Live USB is ephemeral)
- **Selected**: Most reliable option for Windows-only host

## Consequences

### Positive
- Gemma 4 model support enabled (JetPack 6.2.2 requirement)
- NVMe SSD boot: faster model loading, better overall I/O performance
- SD card dependency eliminated (single storage device)
- Clean OS install: opportunity to remove accumulated cruft

### Negative
- Full reflash required: ~4-6 hours total including backup/restore
- All data wiped: requires comprehensive backup and restore procedure
- Python environment must be rebuilt from scratch (requirements files preserved)
- SSH keys regenerated: GitHub SSH key must be re-registered

### Risks
- Jetson AI Lab PyPI index compatibility with JetPack 6.2.2 unverified
- llama-cpp-python 0.3.14 prebuilt may not exist for 6.2.2 (source build fallback available)
- Live USB session is ephemeral: flash must complete without host PC reboot

## Implementation

Full step-by-step plan in:
`docs/archived/dev-plan/2026-04-09-JetPack-622-NVMe-Upgrade-Plan.md`

Summary: Phase 0 (Live USB prep) → Phase 1 (pre-check) → Phase 2 (backup) →
Phase 3 (SDK Manager flash to NVMe) → Phase 4 (system setup) → Phase 5 (restore) →
Phase 6 (verification).
