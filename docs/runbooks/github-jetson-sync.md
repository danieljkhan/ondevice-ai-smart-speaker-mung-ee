# GitHub to Jetson Sync

## Purpose

Use GitHub as the handoff point between local PC development and Jetson hardware validation.

## Local PC flow

1. Make changes in the repository.
2. Commit and push the branch to GitHub.

## Jetson flow

1. Pull the same branch on Jetson.
2. Install or update dependencies only when required.
3. Run hardware validation scripts.
4. Record results back into the repository.

## Shell environment note

Jetson validation commands depend on the CUDA and venv runtime libraries being
present in `LD_LIBRARY_PATH`. The canonical setup lives in the Jetson user's
`~/.bashrc` via `mungidev()`.

- Interactive shell:
  - run `mungidev`
- Non-interactive SSH command:
  - use `bash -lc 'mungidev; ...'`
  - or export `LD_LIBRARY_PATH` explicitly before invoking Python

Without this step, `torch` and ONNX runtime may fail to import with errors like
`libcudss.so.0: cannot open shared object file`.

## Recommended commands

```bash
ssh mungi@<jetson-host> "bash -lc 'cd /opt/mungi-repo && mungidev && git pull --ff-only origin <branch>'"
```

For one-off remote validation without `mungidev`, export the runtime path
explicitly:

```bash
ssh mungi@<jetson-host> "cd /opt/mungi-repo && \
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:/opt/mungi-repo/.venv/lib/python3.10/site-packages/nvidia/cu12/lib:\$LD_LIBRARY_PATH && \
.venv/bin/python scripts/test_tts.py --play --output-device 'USB PnP Audio Device'"
```

## Validation expectation

Jetson-only checks should include:

- CUDA availability
- ONNX provider availability
- inference smoke test
- service startup validation
- thermal and memory baseline capture
