# Persona CEP P0 Operator Guide

This guide runs the Persona Module CEP P0 baseline after the Jetson voice run has
produced the raw artifacts. P0 is measurement-only: do not edit persona prompts,
safety templates, or runtime code while following these steps.

## Prerequisites

- PR #94 / commit `900a3d2` helper scripts are merged into `dev`.
- Jetson is reachable by SSH and has the 100-turn voice fixture at
  `/tmp/mungi_pr5_100_fixtures/voice-fixtures-pr5-100/`.
- Gemma 4 GGUF is accessible on the operator host for tokenizer-only counting.
- Local checkout is on the P0 measurement branch.

Estimated wall time: 5 minutes.
Sanity check: `git rev-parse HEAD` on `dev` reports `900a3d2` before creating
the measurement branch.

## Step 1 - Jetson Voice Run

Run the existing 100-turn voice fixture on the Jetson using the Session 31 helper
pattern. Replace placeholders with the live Jetson host and output timestamp.

```bash
ssh <jetson-host> "cd /opt/mungi && python scripts/e2e_qwen3_asr_mix.py \
  --fixture-dir /tmp/mungi_pr5_100_fixtures/voice-fixtures-pr5-100 \
  --output-dir artifacts/persona-cep-p0-baseline-<timestamp> \
  --repeat-passes 1 \
  --conversation-per-lang \
  --tegrastats-log tegrastats.log"
```

Estimated wall time: 25-45 minutes.
Sanity check: Jetson output directory contains `rounds.jsonl`, `gates.json`, and
`tegrastats.log`.

## Step 2 - Pull Artifacts

Create a local artifact bundle and copy the raw Jetson outputs into it.

```bash
mkdir -p artifacts/persona-cep-p0-baseline-<timestamp>
scp <jetson-host>:/opt/mungi/artifacts/persona-cep-p0-baseline-<timestamp>/rounds.jsonl \
  artifacts/persona-cep-p0-baseline-<timestamp>/
scp <jetson-host>:/opt/mungi/artifacts/persona-cep-p0-baseline-<timestamp>/gates.json \
  artifacts/persona-cep-p0-baseline-<timestamp>/
scp <jetson-host>:/opt/mungi/artifacts/persona-cep-p0-baseline-<timestamp>/tegrastats.log \
  artifacts/persona-cep-p0-baseline-<timestamp>/
```

Estimated wall time: 2-5 minutes.
Sanity check: `rounds.jsonl` has 100 non-empty lines.

## Step 3 - Tokenizer Post-Process

Generate per-turn prompt token counts. If the GGUF is unavailable, omit
`--gemma-model-path`; the CSV will still include heuristic counts.

```bash
python scripts/persona_cep_p0_tokenize.py \
  --rounds-jsonl artifacts/persona-cep-p0-baseline-<timestamp>/rounds.jsonl \
  --output-csv artifacts/persona-cep-p0-baseline-<timestamp>/prompt_tokens.csv \
  --gemma-model-path <relative-or-local-gemma4-gguf-path>
```

Estimated wall time: 1-5 minutes with tokenizer model load.
Sanity check: `prompt_tokens.csv` has per-turn rows plus a `summary` row.

## Step 4 - Intent Template

Generate the auto-labeled seed sheet for manual review.

```bash
python scripts/persona_cep_p0_intent_template.py \
  --rounds-jsonl artifacts/persona-cep-p0-baseline-<timestamp>/rounds.jsonl \
  --output-csv artifacts/persona-cep-p0-baseline-<timestamp>/intent_labels_template.csv
```

Estimated wall time: under 1 minute.
Sanity check: stderr reports the number of auto-labeled turns and the CSV has
`auto_or_manual=auto` on generated rows.

## Step 5 - Guide Tokens

Measure safety-guide injection overhead by topic and language.

```bash
python scripts/persona_cep_p0_guide_tokens.py \
  --templates-json assets/filters/approved_templates.json \
  --output-csv artifacts/persona-cep-p0-baseline-<timestamp>/guide_tokens.csv \
  --gemma-model-path <relative-or-local-gemma4-gguf-path>
```

Estimated wall time: 1-5 minutes with tokenizer model load.
Sanity check: `guide_tokens.csv` includes `volcano`, `earthquake`, and per-lang
summary rows.

## Step 6 - Manual Intent Review

Open `intent_labels_template.csv`, review every auto label, and save the
finalized file as `intent_labels.csv` in the same artifact directory. Change
`auto_or_manual` to `manual` for any row you revise.

Estimated wall time: 20-40 minutes.
Sanity check: every non-empty turn has a reviewed intent row and no remaining
operator notes that need resolution.

## Step 7 - Summary Report

Create `summary.md` comparing the new bundle to Session 31 baseline
`pr5_100_voice_20260511_204700`. Include gate status, latency deltas, tokenizer
deviation, intent-label review notes, and guide-token maxima.

Estimated wall time: 20-30 minutes.
Sanity check: summary states whether the P0 run is rollback-equivalent to
Session 31 and calls out any measurement caveats.

## Step 8 - Commit Artifacts

Open the PR with the generated CSV outputs and `summary.md` after PM review.
Do not include Jetson WAV files or large transient logs unless the PM explicitly
requests them.

Estimated wall time: 5-10 minutes.
Sanity check: PR file list includes `prompt_tokens.csv`, `intent_labels.csv`,
`guide_tokens.csv`, and `summary.md`.
