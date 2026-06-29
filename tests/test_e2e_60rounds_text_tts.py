"""Tests for the text-input / spoken-TTS 60-round runner."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np


def test_e2e_text_tts_importable() -> None:
    """The scripted TTS runner should import without side effects."""
    from scripts import e2e_60rounds_text_tts

    assert e2e_60rounds_text_tts is not None


def test_e2e_text_tts_parser_defaults() -> None:
    """Parser defaults should match the intended long-run test mode."""
    from scripts.e2e_60rounds_text_tts import build_parser

    args = build_parser().parse_args([])
    assert args.rounds == 60
    assert args.start == 1
    assert args.play is False
    assert args.llm_resident is None
    assert args.llm_max_tokens == 128
    assert args.silence_gap_s == 0.25


def test_e2e_text_tts_llm_resident_boolean_flags() -> None:
    """The runner should expose explicit enable/disable flags without masking defaults."""
    from scripts.e2e_60rounds_text_tts import build_parser

    parser = build_parser()

    assert parser.parse_args(["--llm-resident"]).llm_resident is True
    assert parser.parse_args(["--no-llm-resident"]).llm_resident is False


def test_run_round_saves_turn_wavs(tmp_path: Path) -> None:
    """Each text turn should save a rendered WAV and join the session mix."""
    from core.pipeline import ConversationPipeline, PipelineState, TurnMetrics, TurnResult
    from scripts.e2e_60rounds_text_tts import run_round

    class FakePipeline:
        def __init__(self) -> None:
            self.calls = 0

        def run_text_turn(self, user_text: str) -> TurnResult:
            self.calls += 1
            metrics = TurnMetrics(
                llm_time_s=0.2,
                llm_ttft_s=0.05,
                llm_tokens=11,
                llm_model_fallback_used=True,
                llm_model_path_actual="/models/gemma-e2b.gguf",
                llm_model_fallback_reason="primary missing",
                tts_time_s=0.3,
                total_time_s=0.6,
            )
            audio = np.linspace(-0.1, 0.1, 2205, dtype=np.float32)
            return TurnResult(
                user_text=user_text,
                response_text=f"{user_text} 응답",
                audio_samples=audio,
                sample_rate=22050,
                metrics=metrics,
                state=PipelineState.IDLE,
            )

    wav_dir = tmp_path / "tts_wavs"
    wav_dir.mkdir()
    session_audio: list[np.ndarray] = []
    round_result, session_sr = run_round(
        cast(ConversationPipeline, FakePipeline()),
        1,
        {"topic": "테스트", "messages": ["하나", "둘"]},
        wav_dir,
        silence_gap_s=0.1,
        session_audio=session_audio,
        session_sample_rate=None,
    )

    assert round_result["round"] == 1
    assert round_result["planned_turns"] == 3
    assert round_result["total_tokens"] == 33
    assert session_sr == 22050
    assert len(round_result["topics"]) == 1
    turns = round_result["topics"][0]["turns"]
    assert len(turns) == 3
    assert turns[0]["tts_wav"] is not None
    assert turns[0]["llm_model_fallback_used"] is True
    assert turns[0]["llm_model_path_actual"] == "/models/gemma-e2b.gguf"
    assert turns[0]["llm_model_fallback_reason"] == "primary missing"
    assert turns[1]["tts_wav"] is not None
    assert turns[2]["user_text"].startswith("뭉이야 테스트")
    assert len(list(wav_dir.glob("*.wav"))) == 3
    assert len(session_audio) == 5


def test_build_thermal_summary() -> None:
    """Thermal helper should summarize CPU/GPU temperature changes."""
    from scripts.e2e_60rounds_text_tts import _build_thermal_summary

    summary = _build_thermal_summary(
        [
            {"cpu_temp_c": 55.0, "gpu_temp_c": 56.0, "ram_used_mb": 1000, "gr3d_freq_pct": 10},
            {"cpu_temp_c": 57.5, "gpu_temp_c": 60.0, "ram_used_mb": 1100, "gr3d_freq_pct": 20},
            {"cpu_temp_c": 58.0, "gpu_temp_c": 59.5, "ram_used_mb": 1200, "gr3d_freq_pct": 30},
        ],
    )

    assert summary["snapshots_count"] == 3
    assert summary["cpu_temp_c"]["start"] == 55.0
    assert summary["cpu_temp_c"]["end"] == 58.0
    assert summary["cpu_temp_c"]["max"] == 58.0
    assert summary["cpu_temp_c"]["avg"] == 56.833
    assert summary["gpu_temp_c"]["delta"] == 3.5
    assert summary["gpu_temp_c"]["avg"] == 58.5
    assert summary["ram_used_mb"]["avg"] == 1100.0
    assert summary["gr3d_freq_pct"]["avg"] == 20.0
    assert set(summary["cpu_temp_c"]) == {"start", "end", "min", "max", "avg", "delta"}
