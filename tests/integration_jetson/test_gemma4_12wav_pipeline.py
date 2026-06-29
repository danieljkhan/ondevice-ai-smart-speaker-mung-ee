"""Jetson-only 12-WAV STT plus Gemma 4 LLM benchmark harness."""

from __future__ import annotations

import json
import os
import tempfile
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("MUNGI_JETSON_INTEGRATION"),
    reason="Jetson-only integration test",
)


def _normalize_text(text: str) -> str:
    """Normalize transcript text for simple exact and near-exact comparison."""
    return "".join(text.casefold().split())


def _load_ground_truth(gt_path: Path) -> dict[str, str]:
    """Load ground-truth transcript text keyed by WAV filename."""
    rows: dict[str, str] = {}
    for index, line in enumerate(gt_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if "\t" in stripped:
            name, text = stripped.split("\t", 1)
        elif "," in stripped and stripped.split(",", 1)[0].endswith(".wav"):
            name, text = stripped.split(",", 1)
        else:
            name, text = f"audio_{index:02d}.wav", stripped
        rows[name.strip()] = text.strip()
    return rows


def _transcript_is_accurate(observed: str, expected: str) -> bool:
    """Return True for exact or near-perfect normalized transcript matches."""
    normalized_observed = _normalize_text(observed)
    normalized_expected = _normalize_text(expected)
    if normalized_observed == normalized_expected:
        return True
    if not normalized_observed or not normalized_expected:
        return False
    return SequenceMatcher(None, normalized_observed, normalized_expected).ratio() >= 0.9


def _summary_path(prefix: str) -> Path:
    """Return a timestamped JSON evidence path under the system temp dir."""
    return Path(tempfile.gettempdir()) / f"{prefix}_{int(time.time())}.json"


def test_gemma4_12wav_pipeline_harness() -> None:
    """Run the PM 12-WAV STT plus Gemma 4 text LLM integration benchmark."""
    bench_dir = Path.home() / "gemma4_bench_wavs"
    wav_paths = sorted(bench_dir.glob("audio_*.wav"))
    gt_path = bench_dir / "gt.txt"
    assert len(wav_paths) == 12
    assert gt_path.exists()

    try:
        from core.llm_backend_config import LLMBackendConfig
        from core.pipeline import _assert_no_gemma4_marker_leak
        from models.llm_runner import build_llm_from_config, run_chat_generation
        from models.stt_runner import _read_wav_samples, load_stt_model, run_stt
    except ImportError as exc:
        pytest.fail(f"missing runtime deps: {exc}")

    config = LLMBackendConfig.load()
    assert config.backend == "gemma4_text"
    stt_model = load_stt_model(model_size=os.getenv("MUNGI_STT_MODEL_SIZE", "qwen3-asr"))
    backend, llm = build_llm_from_config(config)
    assert backend == "gemma4_text"

    ground_truth = _load_ground_truth(gt_path)
    rows: list[dict[str, Any]] = []
    accurate_count = 0
    warm_latencies: list[float] = []

    for wav_path in wav_paths:
        samples, _sample_rate, _duration = _read_wav_samples(wav_path)
        start_time = time.monotonic()
        segments, _info = run_stt(stt_model, wav_path)
        transcript = " ".join(segment.text for segment in segments)
        response, tokens, ttft, generation_time, _cache_hit, _cache_miss = run_chat_generation(
            llm,
            [{"role": "user", "content": transcript}],
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        )
        latency_s = time.monotonic() - start_time
        _assert_no_gemma4_marker_leak(response)
        expected = ground_truth[wav_path.name]
        accurate = _transcript_is_accurate(transcript, expected)
        accurate_count += int(accurate)
        warm_latencies.append(latency_s)
        rows.append(
            {
                "wav": str(wav_path),
                "sample_count": len(samples),
                "expected": expected,
                "transcript": transcript,
                "transcript_accurate": accurate,
                "response": response,
                "llm_tokens": tokens,
                "llm_ttft_s": ttft,
                "llm_generation_time_s": generation_time,
                "stt_llm_latency_s": latency_s,
            }
        )

    warm_turn_latency_avg_s = sum(warm_latencies) / len(warm_latencies)
    summary = {
        "accurate_transcripts": accurate_count,
        "total_wavs": len(wav_paths),
        "warm_turn_latency_avg_s": warm_turn_latency_avg_s,
        "rows": rows,
    }
    output_path = _summary_path("gemma4_12wav_bench")
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    assert accurate_count >= 10
    assert warm_turn_latency_avg_s <= 10.0
