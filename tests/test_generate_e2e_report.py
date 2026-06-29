"""Tests for the E2E report generator."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from scripts import generate_e2e_report


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON file for one test fixture."""
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write a JSONL fixture file."""
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


def _build_report_data(summary: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a minimal report payload for direct rendering tests."""
    turns: list[generate_e2e_report.TurnRecord] = []
    rounds = generate_e2e_report._group_rounds(turns)
    return {
        "input_dir": Path("fixture"),
        "summary": summary,
        "thermal": {"snapshots_count": 0},
        "turns": turns,
        "rounds": rounds,
        "metrics": generate_e2e_report._summarize_turns(turns),
        "quality": generate_e2e_report._detect_honorific_issues(turns),
    }


def _mix_record(
    *,
    pass_id: str = "pass1",
    global_turn_id: int = 0,
    source_round_id: int = 1,
    round_id: int = 1,
    lang: str = "ko",
    success: bool = True,
    system_ram_mb: float | None = 5000.0,
    process_rss_mb: float | None = 500.0,
    template_topic_id: str | None = None,
    template_mode: str | None = None,
    template_matched: bool = False,
    tts_wav_bytes: int = 1024,
    tts_wav_frames: int = 512,
    llm_ttft_ms: float = 300.0,
    total_ms: float = 1000.0,
) -> dict[str, Any]:
    """Build one flat mix-runner row for report fixtures."""
    return {
        "pass_id": pass_id,
        "global_turn_id": global_turn_id,
        "source_round_id": source_round_id,
        "round_id": round_id,
        "lang": lang,
        "stt_pred": "child input",
        "llm_response": "assistant reply",
        "llm_tokens": 5,
        "success": success,
        "error": None if success else "failed",
        "turn_index_per_lang": source_round_id - 1,
        "system_ram_mb": system_ram_mb,
        "process_rss_mb": process_rss_mb,
        "template_topic_id": template_topic_id,
        "template_mode": template_mode,
        "template_matched": template_matched,
        "tts_wav_bytes": tts_wav_bytes,
        "tts_wav_frames": tts_wav_frames,
        "tts_synth_error": False,
        "tts_load_error": False,
        "timings_ms": {
            "vad_ms": 10.0,
            "stt_load_ms": 20.0,
            "stt_total_ms": 50.0,
            "llm_load_ms": 40.0,
            "llm_ttft_ms": llm_ttft_ms,
            "llm_ms": 200.0,
            "tts_load_ms": 30.0,
            "tts_ms": 100.0,
            "playback_ms": 70.0,
            "first_sound_ms": 450.0,
            "total_ms": total_ms,
        },
    }


def _gate_row(rendered: list[str], gate_id: str) -> str:
    """Return one rendered gate row by gate id."""
    return next(line for line in rendered if line.startswith(f"| {gate_id} |"))


def test_parser_detects_mix_jsonl_format(
    tmp_path: Path,
    monkeypatch: Any,
    caplog: Any,
) -> None:
    """Flat mix-runner JSONL should use the mix flattener and render new sections."""
    _write_jsonl(
        tmp_path / "rounds.jsonl",
        [
            {
                "round_id": 1,
                "lang": "ko",
                "stt_pred": "hello",
                "llm_response": "reply one",
                "llm_tokens": 5,
                "turn_index_per_lang": 0,
                "success": True,
                "error": None,
                "timings_ms": {
                    "llm_ttft_ms": 100.0,
                    "llm_ms": 200.0,
                    "tts_ms": 50.0,
                    "total_ms": 400.0,
                },
            },
            {
                "round_id": 2,
                "lang": "en",
                "stt_pred": "hi",
                "llm_response": "reply two",
                "llm_tokens": 6,
                "turn_index_per_lang": 1,
                "success": True,
                "error": None,
                "timings_ms": {
                    "llm_ttft_ms": 150.0,
                    "llm_ms": 250.0,
                    "tts_ms": 60.0,
                    "total_ms": 500.0,
                },
            },
        ],
    )
    _write_json(
        tmp_path / "summary.json",
        {
            "avg_first_sound_ms": 320.0,
            "avg_tts_first_chunk_ms": 45.0,
            "avg_llm_ttft_ms_first_turn": 100.0,
            "avg_llm_ttft_ms_after_first": 150.0,
            "avg_llm_cache_hit_rate": 0.75,
        },
    )

    original_mix = generate_e2e_report._flatten_mix_jsonl
    original_nested = generate_e2e_report._flatten_turns
    calls = {"mix": 0, "nested": 0}

    def _wrapped_mix(raw_rounds: list[dict[str, Any]]) -> list[generate_e2e_report.TurnRecord]:
        calls["mix"] += 1
        return original_mix(raw_rounds)

    def _wrapped_nested(
        raw_rounds: list[dict[str, Any]],
    ) -> list[generate_e2e_report.TurnRecord]:
        calls["nested"] += 1
        return original_nested(raw_rounds)

    monkeypatch.setattr(generate_e2e_report, "_flatten_mix_jsonl", _wrapped_mix)
    monkeypatch.setattr(generate_e2e_report, "_flatten_turns", _wrapped_nested)

    with caplog.at_level(logging.INFO):
        data = generate_e2e_report._load_dataset(tmp_path)

    report = generate_e2e_report._render_report(data, None)

    assert calls["mix"] == 1
    assert calls["nested"] == 0
    assert "Detected mix-runner JSONL format (2 rounds)" in caplog.text
    assert "First Sound Breakdown" in report
    assert "TTFT Turn Index Split" in report
    assert "LLM Cache Hit Rate" in report


def test_parser_falls_back_to_nested_for_60round(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Nested 60-round JSONL should keep using the legacy flattener path."""
    _write_jsonl(
        tmp_path / "rounds.jsonl",
        [
            {
                "round": 1,
                "topic": "animals",
                "topics": [
                    {
                        "topic": "animals",
                        "turns": [
                            {
                                "round_num": 1,
                                "topic": "animals",
                                "exchange": 1,
                                "user_text": "hello",
                                "assistant_text": "hi",
                                "llm_tokens": 4,
                                "llm_ttft_s": 0.1,
                                "llm_time_s": 0.2,
                                "tts_time_s": 0.3,
                                "total_time_s": 0.4,
                                "success": True,
                                "language": "ko",
                            }
                        ],
                    }
                ],
            }
        ],
    )
    _write_json(tmp_path / "summary.json", {"rounds": 60})

    original_mix = generate_e2e_report._flatten_mix_jsonl
    original_nested = generate_e2e_report._flatten_turns
    calls = {"mix": 0, "nested": 0}

    def _wrapped_mix(raw_rounds: list[dict[str, Any]]) -> list[generate_e2e_report.TurnRecord]:
        calls["mix"] += 1
        return original_mix(raw_rounds)

    def _wrapped_nested(
        raw_rounds: list[dict[str, Any]],
    ) -> list[generate_e2e_report.TurnRecord]:
        calls["nested"] += 1
        return original_nested(raw_rounds)

    monkeypatch.setattr(generate_e2e_report, "_flatten_mix_jsonl", _wrapped_mix)
    monkeypatch.setattr(generate_e2e_report, "_flatten_turns", _wrapped_nested)

    data = generate_e2e_report._load_dataset(tmp_path)
    report = generate_e2e_report._render_report(data, None)

    assert calls["mix"] == 0
    assert calls["nested"] == 1
    assert "First Sound Breakdown" not in report
    assert "TTFT Turn Index Split" not in report
    assert "LLM Cache Hit Rate" not in report


def test_render_first_sound_breakdown_section() -> None:
    """Render should show the first-sound breakdown when the metric exists."""
    report = generate_e2e_report._render_report(
        _build_report_data(
            {
                "avg_first_sound_ms": 320.0,
                "avg_tts_first_chunk_ms": 45.0,
            },
        ),
        None,
    )

    assert "First Sound Breakdown" in report
    assert "320.000 ms (0.320 s)" in report
    assert "45.000 ms (0.045 s)" in report


def test_render_ttft_turn_index_split_section() -> None:
    """Render should show first-turn and later-turn TTFT split values."""
    report = generate_e2e_report._render_report(
        _build_report_data(
            {
                "avg_llm_ttft_ms_first_turn": 100.0,
                "avg_llm_ttft_ms_after_first": 250.0,
            },
        ),
        None,
    )

    assert "TTFT Turn Index Split" in report
    assert "100.000 ms (0.100 s)" in report
    assert "250.000 ms (0.250 s)" in report


def test_render_llm_cache_hit_rate_section() -> None:
    """Render should show the cache hit-rate percentage when provided."""
    report = generate_e2e_report._render_report(
        _build_report_data(
            {
                "avg_llm_cache_hit_rate": 0.75,
            },
        ),
        None,
    )

    assert "LLM Cache Hit Rate" in report
    assert "75.0%" in report


def test_render_stability_section_with_counters_present() -> None:
    """Render should show Stability rows when T3.0 counters exist."""
    report = generate_e2e_report._render_report(
        _build_report_data(
            {
                "critical_memory_events": 0,
                "stt_force_unload_count": 0,
                "llm_prompt_cache_flush_count": 3,
                "system_state_snapshot_count": 2,
            },
        ),
        None,
    )

    assert "## Stability" in report
    assert "| critical_memory_events | 0 | - | - |" in report
    assert "| stt_force_unload_count | 0 | - | - |" in report
    assert "| llm_prompt_cache_flush_count | 3 | - | - |" in report
    assert "| system_state_snapshot_count | 2 | - | - |" in report


def test_render_stability_section_with_legacy_summary() -> None:
    """Legacy summaries should render Stability with unavailable counters."""
    report = generate_e2e_report._render_report(
        _build_report_data(
            {
                "avg_first_sound_ms": 320.0,
            },
        ),
        None,
    )

    assert "## Stability" in report
    for key in generate_e2e_report.STABILITY_COUNTER_KEYS:
        assert f"| {key} | - | - | - |" in report
    assert "(summary.json predates T3.0 - counters unavailable)" in report


def test_render_stability_section_placement() -> None:
    """Stability should appear after thermal analysis and before error details."""
    report = generate_e2e_report._render_report(
        _build_report_data(
            {
                "critical_memory_events": 0,
                "stt_force_unload_count": 0,
                "llm_prompt_cache_flush_count": 3,
                "system_state_snapshot_count": 2,
            },
        ),
        None,
        bilingual=True,
    )

    thermal_index = report.index("## 열 분석")
    stability_index = report.index("## Stability")
    error_index = report.index("## 오류 상세")
    assert thermal_index < stability_index < error_index


def test_render_hides_new_sections_when_missing() -> None:
    """Render should omit Wave 2 sections when the new summary metrics are unavailable."""
    report = generate_e2e_report._render_report(
        _build_report_data(
            {
                "avg_first_sound_ms": 320.0,
                "avg_tts_first_chunk_ms": None,
                "avg_llm_ttft_ms_first_turn": None,
                "avg_llm_ttft_ms_after_first": None,
                "avg_llm_cache_hit_rate": None,
            },
        ),
        None,
    )

    assert "First Sound Breakdown" not in report
    assert "TTFT Turn Index Split" not in report
    assert "LLM Cache Hit Rate" not in report


def test_canonical_latency_header_matches_template() -> None:
    """Rendered mix latency header should equal the runtime-parsed template header."""
    raw_rounds = [_mix_record()]
    turns = generate_e2e_report._flatten_mix_jsonl(raw_rounds)
    rendered = generate_e2e_report._render_mix_latency_table(turns)
    header, _divider, _cells = generate_e2e_report._parse_canonical_latency_table_header()

    assert rendered[0] == header
    assert rendered[2].startswith("| pass1.sr01.ko |")


def test_canonical_latency_row_populates_all_columns_and_avg() -> None:
    """Canonical latency rows and AVG row should contain seconds values."""
    turns = generate_e2e_report._flatten_mix_jsonl(
        [
            _mix_record(global_turn_id=0, total_ms=1000.0, llm_ttft_ms=300.0),
            _mix_record(
                pass_id="pass1",
                global_turn_id=1,
                source_round_id=1,
                round_id=2,
                lang="en",
                total_ms=2000.0,
                llm_ttft_ms=500.0,
            ),
        ],
    )

    rendered = generate_e2e_report._render_mix_latency_table(turns)
    first_data_cells = [cell.strip() for cell in rendered[2].strip("|").split("|")]
    avg_cells = [cell.strip() for cell in rendered[-1].strip("|").split("|")]

    assert len(first_data_cells) == 11
    assert all(cell != "-" for cell in first_data_cells)
    assert first_data_cells[1:] == [
        "0.010",
        "0.050",
        "0.040",
        "0.300",
        "0.200",
        "0.030",
        "0.100",
        "0.070",
        "0.450",
        "1.000",
    ]
    assert avg_cells[0] == "AVG"
    assert avg_cells[-1] == "1.500"


def test_render_pass_aggregates() -> None:
    """Per-pass aggregate table should summarize turns by pass id."""
    turns = generate_e2e_report._flatten_mix_jsonl(
        [
            _mix_record(pass_id="pass1", global_turn_id=0, total_ms=1000.0),
            _mix_record(pass_id="pass2", global_turn_id=1, total_ms=2000.0),
        ],
    )

    rendered = generate_e2e_report._render_pass_aggregates(turns, {"thermal_max_c": 64.5})

    assert (
        "| pass_id | turns | avg_total_s | success_rate | avg_ttft_ko_s | thermal_max_c |"
        in rendered
    )
    assert "| pass1 | 1 | 1.000 | 100.0% | 0.300 | 64.500 |" in rendered
    assert "| pass2 | 1 | 2.000 | 100.0% | 0.300 | 64.500 |" in rendered


def test_render_thermal_nested_avg_and_flat_max() -> None:
    """Thermal rendering should surface nested averages and flat max temperature."""
    rendered = generate_e2e_report._render_thermal_section(
        {
            "snapshots_count": 2,
            "thermal_max_c": 61.0,
            "cpu_temp_c": {
                "start": 50.0,
                "end": 60.0,
                "min": 50.0,
                "max": 60.0,
                "avg": 55.0,
                "delta": 10.0,
            },
            "gpu_temp_c": {
                "start": 51.0,
                "end": 61.0,
                "min": 51.0,
                "max": 61.0,
                "avg": 56.0,
                "delta": 10.0,
            },
        },
    )

    joined = "\n".join(rendered)
    assert "thermal_max_c: 61.000" in joined
    assert "avg 55.000" in joined
    assert "avg 56.000" in joined


def test_render_thermal_curve_presence_and_absence() -> None:
    """Thermal curve rendering should support samples and missing artifacts."""
    present = generate_e2e_report._render_thermal_curve(
        [
            {
                "t_s": 0.0,
                "cpu_temp_c": 50.0,
                "gpu_temp_c": 51.0,
                "ram_used_mb": 1000,
                "gr3d_freq_pct": 10,
            },
            {
                "t_s": 30.0,
                "cpu_temp_c": 52.0,
                "gpu_temp_c": 53.0,
                "ram_used_mb": 1100,
                "gr3d_freq_pct": 20,
            },
        ],
    )
    absent = generate_e2e_report._render_thermal_curve([])

    assert present[0] == "| t_s | CPU°C | GPU°C | RAM MB | GR3D % |"
    assert len(present) == 4
    assert absent == ["Thermal curve not available."]


def test_render_memory_envelope_groups_by_source_round_id() -> None:
    """Memory envelope should include per-turn rows and source-round rollups."""
    raw_rounds = [
        _mix_record(
            pass_id="pass1",
            global_turn_id=0,
            source_round_id=1,
            system_ram_mb=5000.0,
            process_rss_mb=500.0,
        ),
        _mix_record(
            pass_id="pass2",
            global_turn_id=1,
            source_round_id=1,
            system_ram_mb=5200.0,
            process_rss_mb=540.0,
        ),
        _mix_record(
            pass_id="pass2",
            global_turn_id=2,
            source_round_id=2,
            system_ram_mb=5300.0,
            process_rss_mb=560.0,
        ),
    ]

    rendered = "\n".join(generate_e2e_report._render_memory_envelope(raw_rounds))

    assert "delta_from_previous_mb" in rendered
    assert "| 1 | 2 | 5200.000 | 540.000 |" in rendered
    assert "run_peak_system_ram_mb: 5300.000" in rendered
    assert "| pass2 | 5300.000 | 560.000 |" in rendered


def test_gate_verdicts_have_five_columns_and_pass_fail_skip() -> None:
    """Gate rows should expose five columns and include PASS, FAIL, and SKIP verdicts."""
    raw_rounds = [
        _mix_record(
            system_ram_mb=6100.0,
            tts_wav_bytes=0,
            tts_wav_frames=0,
        ),
        _mix_record(
            global_turn_id=1,
            round_id=2,
            lang="en",
            system_ram_mb=5900.0,
            success=False,
        ),
    ]
    summary = {
        "critical_memory_events": 0,
        "stt_load_count": 1,
        "tts_load_count": 1,
        "tts_synth_error_count": 0,
        "tts_load_error_count": 0,
        "stt_provider_resolved": "cpu",
        "sherpa_onnx_version": None,
    }

    rendered = generate_e2e_report._render_gate_verdicts(raw_rounds, summary, {})
    verdicts = "\n".join(rendered)

    assert rendered[0] == (
        "| gate_id | threshold | observed_value | verdict | evidence_artifact_path |"
    )
    for line in rendered[2:]:
        assert len([cell for cell in line.split("|") if cell.strip()]) == 5
    assert "| G2b | == 0 CRITICAL events | 0 | PASS | run.log |" in verdicts
    assert "| G1 | < 5500 MB | 6100.000 | FAIL | rounds.jsonl |" in verdicts
    assert "| G3 | < 80.0 °C | - | SKIP | thermal_summary.json |" in verdicts


def test_g10_prefers_stt_provider_actual_over_legacy_resolved() -> None:
    """G10 should evaluate actual provider when the new summary field exists."""
    summary = {
        "stt_provider_actual": "cuda",
        "stt_provider_resolved": "cpu",
        "sherpa_onnx_version": "1.2.3",
        "stt_load_count": 1,
    }

    rendered = generate_e2e_report._render_gate_verdicts([_mix_record()], summary, {})

    assert _gate_row(rendered, "G10") == (
        "| G10 | STT provider == cpu, version recorded, stt_load_count == 1 | "
        "provider=cuda | FAIL | summary.json |"
    )


def test_g10_falls_back_to_legacy_stt_provider_resolved() -> None:
    """G10 should still pass legacy summaries that only recorded resolved provider."""
    summary = {
        "stt_provider_resolved": "cpu",
        "sherpa_onnx_version": "1.2.3",
        "stt_load_count": 1,
    }

    rendered = generate_e2e_report._render_gate_verdicts([_mix_record()], summary, {})

    assert _gate_row(rendered, "G10") == (
        "| G10 | STT provider == cpu, version recorded, stt_load_count == 1 | "
        "provider=cpu | PASS | summary.json |"
    )


def test_vad_miss_gate_fails_when_vad_miss_rows_exist() -> None:
    """VAD-miss telemetry should surface as a gate failure when present."""
    row = _mix_record(success=False)
    row["vad_miss"] = True
    row["failure_reason"] = "vad_miss"

    rendered = generate_e2e_report._render_gate_verdicts([row], {}, {})

    assert _gate_row(rendered, "G_VAD_MISS") == (
        "| G_VAD_MISS | == 0 rows with failure_reason=vad_miss | 1 | FAIL | rounds.jsonl |"
    )


def test_vad_miss_gate_passes_when_telemetry_has_no_misses() -> None:
    """VAD-miss gate should pass when new telemetry is present and clean."""
    row = _mix_record()
    row["vad_miss"] = False
    row["failure_reason"] = None

    rendered = generate_e2e_report._render_gate_verdicts([row], {}, {})

    assert _gate_row(rendered, "G_VAD_MISS") == (
        "| G_VAD_MISS | == 0 rows with failure_reason=vad_miss | 0 | PASS | rounds.jsonl |"
    )


def test_reproducibility_appendix_renders_recorded_and_missing_fields() -> None:
    """Reproducibility appendix should render all required fields with fallback text."""
    rendered = "\n".join(
        generate_e2e_report._render_reproducibility_appendix(
            {
                "mungi_llm_resident": "1",
                "mungi_stt_resident": "0",
                "mungi_tts_resident": "1",
                "llm_n_gpu_layers_resolved": 99,
                "sherpa_onnx_version": "1.2.3",
                "stt_provider_actual": "cpu",
                "stt_provider_configured": "cpu",
                "stt_provider_requested": "cuda",
                "stt_provider_resolved": "cpu",
                "repeat_passes": 5,
                "model_sha256": {"gemma": None},
            },
        ),
    )

    for field in (
        "mungi_llm_resident",
        "mungi_stt_resident",
        "mungi_tts_resident",
        "llm_n_gpu_layers_resolved",
        "commit_sha",
        "sherpa_onnx_version",
        "stt_provider_actual",
        "stt_provider_configured",
        "stt_provider_requested",
        "stt_provider_resolved",
        "repeat_passes",
        "model_sha256",
    ):
        assert field in rendered
    assert "| commit_sha | not recorded |" in rendered
    assert rendered.index("stt_provider_actual") < rendered.index("stt_provider_configured")
    assert rendered.index("stt_provider_configured") < rendered.index("stt_provider_requested")
    assert rendered.index("stt_provider_requested") < rendered.index("stt_provider_resolved")
