# Stage-2 Measurement-Infrastructure Patch — Operator Guide

- **Authoritative source**: `docs/archived/dev-plan/2026-05-05-stage2-measurement-infrastructure-patch-plan.md` (v4 FINAL); patch merged to `dev` 2026-05-05 (PR #80, `e01bd04`).
- **Parent plan**: `docs/archived/dev-plan/2026-05-03-stage2-jetson-full-audio-rag-resident-measurement-plan.md` (v4 FINAL).
- **Anchor ADR**: `docs/adr/0083-stage2-measurement-plan-architectural-decisions.md`.
- **Audience**: operators running Stage-2 measurement campaigns on Jetson Orin Nano Super 8 GB.

This guide is a quick reference for the new CLI flags, output artifacts, schema fields, and the gate verdict scaffold introduced by the patch. It does NOT replace the Plan or ADR; it summarizes user-facing changes for daily operations.

> [HISTORICAL UPDATE: PR 4-B (2026-05-XX) removed wiki RAG from the runtime.
> RAG-specific CLI flags, summary fields, row fields, FAISS metadata, and the
> G5 RAG gate from the original Stage-2 patch are retired.]

## 1. What changed (high level)

| Layer | Change |
|---|---|
| `core/pipeline.py` | `TurnMetrics` dataclass gains router/template observability fields; guide-mode template matches now set `template_matched=True` (was block-only). |
| `scripts/e2e_qwen3_asr_mix.py` | New repeat/GPU-layer/input-padding CLI flags; UTF-8 mojibake fix at latency table header; `RoundInput` gains `source_round_id` (1-based); `discover_round_pairs` caps `pair_count` BEFORE interleaving (intentional behavior change); TegrastatsMonitor wire-in + `thermal_curve.json` artifact; same-process repeat-passes loop; additive `summary.json` + `rounds.jsonl` schemas. |
| `scripts/e2e_60rounds_text_tts.py` | `_build_thermal_summary` extended to also emit `avg` per nested temp series (additive). |
| `scripts/generate_e2e_report.py` | mix-runner per-turn renderer parses canonical column header from `docs/templates/e2e-report-format.md` at runtime; per-pass aggregate + thermal curve + memory envelope (per-turn + per-source-round) sub-sections; 5-column gate verdict scaffold; reproducibility appendix. |

## 2. New CLI flags (`scripts/e2e_qwen3_asr_mix.py`)

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--repeat-passes N` | `int`, `1..20` | `1` | Same-process repeat. Single `ModelManager` + single per-language `ConversationPipeline` pair are constructed once before pass-1 and reused. |
| `--llm-n-gpu-layers N` | `int`, optional | unset | CLI flag **takes precedence over** `MUNGI_LLM_N_GPU_LAYERS` env var. Resolved value recorded in `summary.json`. |

`--max-rounds M` semantics — **intentional behavior change**:
- Pre-patch: `M` was the count of interleaved turns → `--max-rounds 3` yielded 3 turns total (KO/EN/KO).
- Post-patch: `M` is the count of bilingual round-pairs → `--max-rounds 3` yields 6 turns (3 KO + 3 EN). Per pass.
- Total turns across N passes = `N × 2 × M`.

Environment overrides:

| Variable | Purpose |
|---|---|
| `MUNGI_LLM_N_GPU_LAYERS` | Provides the LLM GPU-layer default when `--llm-n-gpu-layers` is not passed. |

Example invocation (Configuration A, parent Plan v4 §6):

```
mungidev
MUNGI_LLM_BACKEND=gemma4_text \
MUNGI_LLM_RESIDENT=1 MUNGI_STT_RESIDENT=1 MUNGI_TTS_RESIDENT=1 \
python scripts/e2e_qwen3_asr_mix.py \
    --ko-dir /home/mungi/qwen3_test/20260412_뭉이야,_너는_바삭/ \
    --en-dir /home/mungi/qwen3_test/20260413_Hi,_moong-/ \
    --output-root /var/lib/mungi/e2e_results/2026-05-XX-stage2-cfgA-coexistence/ \
    --repeat-passes 1 --max-rounds 0 \
    --llm-max-tokens 128 --llm-n-gpu-layers 99
```

Configuration B uses `--repeat-passes 5` (5 × 24 = 120 turns sustained) to a separate output dir (parent Plan v4 §6 Configuration B).

## 3. Output artifacts

Run-output directory contains the following files (additions in **bold**):

| File | Contents |
|---|---|
| `summary.json` | Run-level summary; existing keys preserved including `latency_table_markdown` (UTF-8 markdown rendering of the latency table for quick log inspection). New keys listed in §4. |
| `rounds.jsonl` | Per-turn records (one JSON object per line). New keys listed in §5. |
| `tegrastats.log` | Raw tegrastats sampling output (existing precedent). |
| `thermal_summary.json` | **Backward-compatible superset**: existing nested `cpu_temp_c`/`gpu_temp_c`/`ram_used_mb`/`gr3d_freq_pct` series now include `avg` (alongside `start/end/min/max/delta`). NEW flat top-level fields: `thermal_max_c` (= `max(cpu_temp_c.max, gpu_temp_c.max)`), `duration_s`. |
| **`thermal_curve.json`** | NEW. Downsampled time-series snapshots (~30 s interval): `{interval_s, samples: [{t_s, cpu_temp_c, gpu_temp_c, ram_used_mb, gr3d_freq_pct}, ...]}`. Used by the renderer's thermal curve sub-section. |
| `runner.log` | Standard runner log (existing). G2b verdict counts `MemoryHealth.CRITICAL` events from this log. |

## 4. `summary.json` new fields

All additive; existing keys (including `latency_table_markdown`) unchanged.

| Field | Type | Notes |
|---|---|---|
| `runner` | `str` | Sentinel `"e2e_qwen3_asr_mix"` (renderer detects mix-runner output via this). |
| `mungi_llm_resident` | `str` | env-var echo (`"1"` / `"0"` / absent). |
| `mungi_stt_resident` | `str` | env-var echo. |
| `mungi_tts_resident` | `str` | env-var echo. |
| `llm_n_gpu_layers_resolved` | `int` | CLI > env > config precedence. |
| `stt_provider_actual` | `str | list[str] | null` | Mode of per-turn providers captured from `run_stt()` `info["provider"]`; list when mixed providers are observed. |
| `stt_provider_configured` | `str` | Manager configuration (`ManagerConfig.stt_device`). |
| `stt_provider_requested` | `str | null` | Raw `MUNGI_STT_PROVIDER` env-var echo, e.g., `"cuda"` / `"cpu"` / null. |
| `stt_provider_resolved` | `str` | Back-compat alias of `stt_provider_configured`; retained through PR 5+ consumers. |
| `sherpa_onnx_version` | `str` | `sherpa_onnx.__version__`. |
| `stt_load_count` | `int` | should be 1 across the run when STT-resident. |
| `tts_load_count` | `int` | same semantics for TTS. |
| `tts_load_error_count` | `int` | TTS load errors. |
| `tts_synth_error_count` | `int` | TTS synthesis errors. |
| `repeat_passes` | `int` | resolved `--repeat-passes N`. |
| `input_pad_ms` | `int` | PR 2 NEW: resolved `--input-pad-ms` value at session start. Defaults to 200 ms; set to 0 for legacy behavior. |
| `model_sha256` | `object` | keys: `gemma`, `qwen3_asr`, `supertonic`. SHA-256 strings or `null`. |

## 5. `rounds.jsonl` new fields

All additive. New keys per row:

| Field | Type | Notes |
|---|---|---|
| `pass_id` | `str` | `"pass1"` ... `"passN"`. |
| `global_turn_id` | `int` | Monotonic from 0 across all passes. |
| **`source_round_id`** | `int` | NEW (R3-F3 A). 1-based bilingual-pair identity (= `sequence_index + 1`). Distinct from existing `round_id` (interleaved turn ID, preserved). |
| `stt_provider_actual` | `str | null` | Per-turn actual provider captured from the STT runner info dict. |
| `stt_provider_configured` | `str` | Manager-configured provider for the session. |
| `stt_provider_requested` | `str | null` | Raw provider requested via `MUNGI_STT_PROVIDER`. |
| `vad_miss` | `bool` | True when `gt_text` is non-empty and `speech_segments == 0`. |
| `vad_miss_reason` | `str | null` | `audio_too_short` when `audio_duration_ms < 300`; otherwise `silence_detected` for the default no-segment path. |
| `audio_duration_ms` | `float` | ORIGINAL source WAV duration in milliseconds (pre-pad). Used by `_vad_miss_reason()` for `audio_too_short` classification at < 300 ms. PRESERVED across PR 1 and PR 2; semantics unchanged. |
| `audio_padded_ms` | `float` | PR 2 NEW: post-pad pipeline input duration in milliseconds. Equals `audio_duration_ms + 2 * input_pad_ms` when `input_pad_ms > 0`, else equals `audio_duration_ms`. Reflects what VAD/STT actually saw. |
| `input_trace_wav` | `str | null` | Clarified PR 2: link/copy of ORIGINAL source WAV (pre-pad). Padded audio is pipeline-internal; raw trace allows analysts to re-run with different padding values without re-sourcing. |
| `core_success` | `bool` | Original `TurnResult.success` no-error signal. |
| `failure_reason` | `str | null` | Null on success, `vad_miss` for overlaid VAD misses, or `runtime_error` for core failures. |
| `template_topic_id` | `str | null` | Copied. Populated for any approved-template match (guide or block). |
| `template_mode` | `str | null` | Copied. |
| `template_matched` | `bool` | Copied. **NEW behavior**: now `True` for both guide-mode and block-mode (was block-only pre-patch). |
| `tts_wav_bytes` | `int` | TTS output WAV byte count. |
| `tts_wav_frames` | `int` | TTS output WAV frame count. |
| `tts_synth_error` | `str | null` | Per-turn TTS synthesis error (if any). |
| `tts_load_error` | `str | null` | Per-turn TTS load error (if any). |
| `system_ram_mb` | `int` | tegrastats-aligned snapshot at turn start. |
| `process_rss_mb` | `int` | `psutil.Process(os.getpid()).memory_info().rss / 1024**2` at turn start. |

`VAD_AUDIO_TOO_SHORT_MS = 300` is the named threshold for `audio_too_short`.
Silero VAD onset detection needs roughly 30-100 ms hangover; 300 ms is the
measurement-run safety margin. Use `--expect-stt-provider {cuda,cpu}` to fail
fast when a run's per-turn actual STT provider differs from the expected
provider.

## 6. R44 operational identity (for G7 evaluation)

R44-class is operationally defined as `template_topic_id == "swimming"` AND `template_mode == "guide"` AND `template_matched == true` (ADR 0083 D2).

Any other approved-template match is NOT R44 and does NOT count toward the Phase 0.5 ≥ 5-stimuli threshold or the G7 success condition.

## 7. Generating reports (`scripts/generate_e2e_report.py`)

```
python scripts/generate_e2e_report.py --input-dir <run-output-dir> --output <run-output-dir>/report.md
```

The mix-runner per-turn renderer is detected via the `runner` sentinel field in `summary.json`. The rendered `report.md` includes:

1. **Latency table** — canonical column header parsed at runtime from `docs/templates/e2e-report-format.md:6-20` (D7 strict equality). Columns:

   ```
   | Turn | VAD | STT | LLM로드 | TTFT | LLM추론 | TTS로드 | TTS합성 | 재생 | 첫소리까지 | 전체 |
   ```

   Cell values are seconds, 3 decimal places. The `Turn` cell encodes pass + source-pair + language: `pass{N}.sr{source_round_id:02d}.{lang}` (e.g., `pass1.sr01.ko`).
2. **Per-pass aggregate** — one small table grouping by `pass_id`.
3. **Thermal section** — backward-compatible nested rendering + flat `thermal_max_c` summary line.
4. **Thermal curve** — small time-series table from `thermal_curve.json`. If absent: `"Thermal curve not available."`.
5. **Memory envelope** — per-turn + per-source-round + run/per-pass peak summary.
6. **Gate verdict scaffold** — see §8.
7. **Reproducibility appendix** — env vars, commit SHA, model SHA-256s, `sherpa_onnx_version`.

## 8. Gate verdict scaffold (5-column row format)

Each gate row has 5 columns: `gate_id | threshold | observed_value | verdict | evidence_artifact_path`. `verdict` is one of `PASS` / `FAIL` / `SKIP`. SKIP is used ONLY when input data is unavailable (e.g., `tegrastats.log` absent → G3 SKIP).

| gate_id | threshold | observed_value source | verdict derivation | evidence path |
|---|---|---|---|---|
| G1 | `peak_system_ram_mb < 5500` | run-peak `system_ram_mb` from `rounds.jsonl` | PASS if `< 5500` else FAIL; SKIP if `rounds.jsonl` absent | `rounds.jsonl` |
| G2a | `peak_system_ram_mb < 6000` | run-peak `system_ram_mb` | PASS if `< 6000` else FAIL; SKIP if absent | `rounds.jsonl` |
| G2b | `MemoryHealth.CRITICAL` event count `== 0` | parsed from `runner.log` | PASS if `== 0` else FAIL; SKIP if log absent | `runner.log` |
| G3 | `thermal_max_c < 80.0 °C` | `thermal_summary.json["thermal_max_c"]` | PASS if `< 80.0` else FAIL; SKIP if `thermal_summary.json` absent | `thermal_summary.json` |
| G4 | turn success rate `≥ 95 %` | computed from `success` flags in `rounds.jsonl` | PASS if rate `≥ 0.95` else FAIL; SKIP if 0 turns | `rounds.jsonl` |
| G6 | first-turn positive `*_load_ms`; `stt_load_count == 1`; `tts_load_count == 1` | `summary.json` + first row of `rounds.jsonl` | PASS if all hold else FAIL; SKIP if `summary.json` missing relevant fields | `summary.json` + `rounds.jsonl` |
| G7 | `≥ 5` rows where R44 identity holds | mechanically counted from `rounds.jsonl` | PASS if count `≥ 5` else FAIL; SKIP if `rounds.jsonl` absent | `rounds.jsonl` |
| G8 | KO hot-turn TTFT mean `≤ 4.50 s` (filter `lang == "ko"` AND `template_matched == false`) | mechanically computed | PASS if mean `≤ 4.50` else FAIL; SKIP if 0 matching rows | `rounds.jsonl` |
| G9 | `tts_synth_error_count == 0` AND `tts_load_error_count == 0` AND every `success=true` row has `tts_wav_bytes > 0` AND `tts_wav_frames > 0` | `summary.json` + per-row | PASS if all hold else FAIL; SKIP if no rows | `summary.json` + `rounds.jsonl` |
| G10 | `stt_provider_resolved` recorded AND `sherpa_onnx_version` recorded AND `stt_load_count == 1` | `summary.json` | PASS if all 3 + count `== 1` else FAIL; SKIP if `summary.json` missing | `summary.json` |

The renderer is the **mechanical computer** of these verdicts. The final ADR 0076 status decision text (whether to flip status to "Accepted (validated for full-audio + STT-resident)") is a manual call by Claude orchestrator during Phase 3 of the parent Stage-2 measurement campaign.

## 9. Failure modes and recovery

| Symptom | Likely cause | Action |
|---|---|---|
| `tegrastats.log` empty or absent in run-output | TegrastatsMonitor unavailable (binary missing, permission) | Runner emits structured warning + continues. Phase 3 G3 = SKIP. Investigate Jetson tegrastats install if needed. |
| `thermal_curve.json` absent | TegrastatsMonitor failed to start (same as above) | Renderer emits `"Thermal curve not available."` and continues. |
| `rounds.jsonl` missing `source_round_id` | Older runner (pre-`e01bd04`) on Jetson | `mungiup` to sync `/opt/mungi-repo` to `e01bd04` or later. |
| Renderer `report.md` missing canonical column header | Template file at `docs/templates/e2e-report-format.md` corrupted or moved | Restore from dev. The renderer parses lines 6-20 at runtime. |
| Multi-pass run shows monotonic memory growth across passes | Pipeline state leak (KV cache, model cache, etc.) | Inspect `system_ram_mb` + `process_rss_mb` per-turn envelope; per-pass peaks; flag as Stage-3 candidate per parent Plan v4 §10 R2 mitigation. |

## 10. Verification commands (post-Jetson-sync)

After `mungiup` syncs `/opt/mungi-repo` to dev HEAD `e01bd04` or later:

```
ssh mungi@jetson.local 'cd /opt/mungi-repo && git rev-parse HEAD'
# Expect: e01bd04... (or later)

ssh mungi@jetson.local 'cd /opt/mungi-repo && python -c "from core.pipeline import TurnMetrics; m = TurnMetrics(); print(\"template_topic_id\" in m.to_dict())"'
# Expect: True

ssh mungi@jetson.local 'cd /opt/mungi-repo && python scripts/e2e_qwen3_asr_mix.py --help 2>&1 | grep -E "(--repeat-passes|--llm-n-gpu-layers)"'
# Expect: lines listing the retained measurement flags
```

## 11. Cross-references

- `docs/archived/dev-plan/2026-05-05-stage2-measurement-infrastructure-patch-plan.md` (v4 FINAL — full deliverable spec)
- `docs/archived/dev-plan/2026-05-03-stage2-jetson-full-audio-rag-resident-measurement-plan.md` (v4 FINAL — parent Stage-2 measurement campaign plan; defines G1-G10 thresholds and Phase 0/1/2/3 procedure)
- `docs/adr/0083-stage2-measurement-plan-architectural-decisions.md` (D1 split-PR, D2 R44 identity, D3 TurnMetrics extension, D4 single-process 5-pass, D7 template-direct reference)
- `docs/templates/e2e-report-format.md` (canonical mandatory latency table column header — source-of-truth for D7)
- `docs/runbooks/jetson-deployment-operations.md` (deployment gates, E2E workflow, tmux, passwordless sudo)
- `docs/runbooks/baseline-stack-and-models.md` (active model selection, hardware interfaces)
- ADR 0076 (L1 LLM resident default — sets the residency policy this patch measures)
- ADR 0073 (Gemma 4 swap)
- ADR 0058 (TTS-resident opt-in)
- ADR 0055 (Qwen3-ASR primary)
- ADR 0082 (conversation-memory FAISS — deferred update path)

## 12. When to update this doc

This is a runbook entry; update when:

- A future patch adds new CLI flags or schema fields.
- The G1-G10 threshold or derivation logic changes (would also require a new ADR amending 0083).
- The parent Stage-2 plan is replaced by a Stage-3 plan (failure recovery scope expansion).
- Operator workflow on Jetson changes (e.g., new deploy script, new sudo gate, new tmux convention).

For Plan-level revisions, edit the canonical Plan markdown (`Dev_Plan/...`) and amend ADR 0083 via append-only (per `feedback_adr_immutability.md`); update this guide as a follow-up.
