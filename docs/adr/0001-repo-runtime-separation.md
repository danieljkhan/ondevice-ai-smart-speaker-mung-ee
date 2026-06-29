# ADR 0001: Separate repository source from mutable runtime data

## Status

Accepted

## Context

The current Jetson device already has a manually prepared `/opt/mungi` directory that reflects real installation work.
The GitHub repository was created later and now represents the development source of truth.

Replacing the live directory in one step is risky because the current device may contain:

- manually installed scripts
- local environment fixes
- model files
- mutable configuration
- logs and conversation history

We need a safe path from the current device state to a repository-managed deployment layout.

## Decision

Use this target layout:

- source repository: `/opt/mungi`
- temporary staging clone during migration: `/opt/mungi-repo`
- mutable config and state: `/var/lib/mungi`
- logs: `/var/log/mungi`
- large model assets: `/opt/mungi/ai_models` or external storage

During migration, keep `/opt/mungi` and `/opt/mungi-repo` separate.
Do not overwrite the live directory until backup, diff, and classification are complete.

## Classification rule

When reviewing files in the live directory:

- source code and templates move into the Git repository
- mutable config moves into `/var/lib/mungi`
- logs move into `/var/log/mungi`
- large downloaded models stay out of Git
- secrets never enter Git

## Consequences

Benefits:

- lower migration risk
- clear boundary between code and device state
- easier GitHub workflow for local PC and Jetson
- better foundation for backup, rollback, and release automation

Costs:

- one temporary duplicate directory on Jetson
- short-term need to compare and classify files
- delayed full convergence until validation is complete
