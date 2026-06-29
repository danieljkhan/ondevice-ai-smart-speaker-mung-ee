# ADR 0041: Colab Finetune Notebook Path Convention

- **Status**: Accepted
- **Date**: 2026-04-07
- **Author**: Claude Code PM

## Context

`Mungi_Qwen3_1_7B_Q6_K_Convert.ipynb` was cloned from the Q8_0 notebook with
`DRIVE_BASE` changed from `mungi-finetune5` to `mungi-finetune7`. Because both
input data paths (`DATA_PATH`, `DPO_PATH`) and output paths (`CHECKPOINT_DIR`,
`GGUF_DIR`) derived from the same `DRIVE_BASE` variable, the notebook failed
with `FileNotFoundError` — the training data only existed in `mungi-finetune5`.

This is a recurring risk: every new finetune experiment creates a fresh Drive
folder for outputs, but the training data lives in a prior folder.

## Decision

**Separate input data paths from output paths in all Colab finetune notebooks.**

Standard pattern:

```python
# Input data (read from existing folder)
DATA_BASE = Path('/content/drive/MyDrive/mungi-finetune5')
DATA_PATH = DATA_BASE / 'sft_good_only.jsonl'
DPO_PATH  = DATA_BASE / 'dpo_pairs.jsonl'

# Output (write to new experiment folder)
DRIVE_BASE = Path('/content/drive/MyDrive/mungi-finetune7')
CHECKPOINT_DIR = DRIVE_BASE / 'checkpoints-...'
GGUF_DIR       = DRIVE_BASE / 'gguf-...'
```

## Rules

1. `DATA_BASE` and `DRIVE_BASE` must be separate variables.
2. `DATA_BASE` points to the folder containing the canonical training data.
3. `DRIVE_BASE` points to the experiment-specific output folder.
4. Training data files (`sft_good_only.jsonl`, `dpo_pairs.jsonl`,
   `batch_deduped.jsonl`) are never copied to output folders.
5. When cloning a notebook for a new experiment, only change `DRIVE_BASE` and
   output-derived paths. `DATA_BASE` stays the same unless the training data
   itself has been updated to a new version folder.

## Affected Notebooks

| Notebook | DATA_BASE | DRIVE_BASE | Status |
|----------|-----------|------------|--------|
| `Mungi_Qwen3_1.7B_Q8_0_Convert.ipynb` | `mungi-finetune5` | `mungi-finetune5` | Legacy (same folder) |
| `Mungi_Qwen3_1_7B_Q6_K_Convert.ipynb` | `mungi-finetune5` | `mungi-finetune7` | **Fixed** |
| `Mungi_QLoRA_Finetune_v6_Qwen35_2B.ipynb` | `mungi-finetune6` | `mungi-finetune6` | Review needed |
| `Mungi_QLoRA_Gemma4_E2B_SFT.ipynb` | `mungi-finetune6` | `mungi-finetune6` | Review needed |

## Consequences

- New finetune notebooks must follow this convention.
- Existing notebooks where `DATA_BASE == DRIVE_BASE` work correctly but should
  adopt the separated pattern when next modified.
- Code review of new notebooks should verify `DATA_BASE` points to a folder
  with confirmed training data.
