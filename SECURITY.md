# Security Policy

This is a research / educational proof-of-concept, **not** a certified or
commercial product, and **not** a safety-certified children's device. It is
provided *as is* under the Apache-2.0 license (see `LICENSE`). It must not be
deployed with real children without independent safety, privacy, and legal
review.

## Reporting a vulnerability

If you discover a security or privacy issue, please report it **privately** —
do not open a public issue for anything that could expose a user.

- Use GitHub's **"Report a vulnerability"** (Security Advisories) on this
  repository, or
- contact the maintainer privately.

Please include reproduction steps and the affected files/commit. We will
acknowledge and respond as time permits; this is a personal research project
with no SLA.

## Scope notes

- **Secrets:** No API keys, tokens, passwords, private keys, or device PINs are
  stored in this repository or its history. Runtime secrets (parent-portal PIN,
  device environment) live only on the device and are git-ignored.
- **Child data:** Conversation/audio data is stored on-device only and is never
  committed.
- **Models/data:** Model weights, custom voices, and restricted training corpora
  are not included (see `NOTICE`).

If you believe any of the above has been violated by a file in this repository,
please report it privately as above.
