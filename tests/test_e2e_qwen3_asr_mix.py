"""Tests for the bilingual interleaved Qwen3-ASR E2E runner."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from core.model_manager import ModelType
from scripts import e2e_qwen3_asr_mix


def _touch_audio_files(directory: Path, names: list[str]) -> None:
    """Create placeholder audio files under *directory*."""
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).touch()


def _patch_librosa_samples(monkeypatch: pytest.MonkeyPatch, samples: np.ndarray) -> None:
    """Patch librosa loading to return a deterministic mono float array."""

    def _load(path: Path, sr: int, mono: bool) -> tuple[np.ndarray, int]:
        del path
        assert sr == e2e_qwen3_asr_mix.DEFAULT_SAMPLE_RATE
        assert mono is True
        return np.asarray(samples, dtype=np.float32).copy(), sr

    monkeypatch.setattr(
        e2e_qwen3_asr_mix,
        "_load_librosa_module",
        lambda: SimpleNamespace(load=_load),
    )


def _write_source_bytes(path: Path, payload: bytes) -> None:
    """Write deterministic source bytes for raw trace-copy assertions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _patch_runtime(
    monkeypatch: Any,
    turn_results: list[SimpleNamespace | Exception],
    *,
    tts_call_log: list[tuple[str, str]] | None = None,
    clear_history_log: list[str] | None = None,
    audio_input_log: list[np.ndarray] | None = None,
    librosa_audio: np.ndarray | None = None,
) -> None:
    """Patch runtime dependencies so ``main()`` runs without real models."""
    audio_fixture = (
        np.asarray(librosa_audio, dtype=np.float32).reshape(-1)
        if librosa_audio is not None
        else np.zeros(1600, dtype=np.float32)
    )

    class FakeManagerConfig:
        """Capture manager config kwargs for assertions when needed."""

        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            for key, value in kwargs.items():
                setattr(self, key, value)
            if not hasattr(self, "stt_device"):
                self.stt_device = "cpu"
            if not hasattr(self, "tts_model_dir"):
                model_dir = str(kwargs.get("model_dir", ""))
                self.tts_model_dir = f"{model_dir}/supertonic-2" if model_dir else ""
            if not hasattr(self, "tts_voice_style"):
                self.tts_voice_style = "F2"

    class FakePipelineConfig:
        """Capture pipeline config kwargs."""

        def __init__(self, **kwargs: object) -> None:
            self.stt_language = "ko"
            self.llm_max_tokens = int(e2e_qwen3_asr_mix.os.getenv("MUNGI_LLM_MAX_TOKENS", "80"))
            self.max_history_turns = 2
            self.max_history_tokens = 200
            for key, value in kwargs.items():
                setattr(self, key, value)

    class FakeModelManager:
        """Minimal model manager stub."""

        def __init__(self, config: FakeManagerConfig) -> None:
            self.config = config
            self.initialized = False

        def initialize(self) -> None:
            self.initialized = True

        def unload_all(self, force: bool = False) -> None:
            self.initialized = False
            del force

        def load(self, model_type: object) -> None:
            del model_type

        def unload_tts(self) -> None:
            return

    class FakeConversationPipeline:
        """Minimal conversation pipeline stub."""

        _cursor = 0

        def __init__(
            self,
            manager: FakeModelManager,
            config: FakePipelineConfig,
        ) -> None:
            self.manager = manager
            self.config = config
            self._conversation_dir = Path(".")

        def clear_history(self) -> None:
            if clear_history_log is not None:
                clear_history_log.append(self.config.stt_language)
            return

        def _run_tts(self, text: str, language: str = "ko") -> tuple[np.ndarray, int]:
            if tts_call_log is not None:
                tts_call_log.append((language, text))
            return np.linspace(-0.2, 0.2, 800, dtype=np.float32), 22050

        def _play_audio_out(self, audio_samples: np.ndarray, sample_rate: int) -> None:
            del audio_samples, sample_rate
            return

        def run_turn(self, audio_samples: np.ndarray, sample_rate: int = 16000) -> SimpleNamespace:
            del sample_rate
            if audio_input_log is not None:
                audio_input_log.append(np.asarray(audio_samples, dtype=np.float32).copy())
            result = turn_results[FakeConversationPipeline._cursor]
            FakeConversationPipeline._cursor += 1
            if isinstance(result, Exception):
                raise result
            response_text = str(getattr(result, "response_text", ""))
            language = str(getattr(result, "detected_language", self.config.stt_language))
            audio_out, output_sample_rate = self._run_tts(response_text, language=language)
            self._play_audio_out(audio_out, output_sample_rate)
            result.audio_samples = audio_out
            result.sample_rate = output_sample_rate
            return result

    monkeypatch.setattr(
        e2e_qwen3_asr_mix,
        "_get_runtime_classes",
        lambda: (
            FakeManagerConfig,
            FakeModelManager,
            FakePipelineConfig,
            FakeConversationPipeline,
        ),
    )
    monkeypatch.setattr(e2e_qwen3_asr_mix, "_run_preflight", lambda skip: None)
    monkeypatch.setattr(
        e2e_qwen3_asr_mix,
        "_load_librosa_module",
        lambda: SimpleNamespace(
            load=lambda path, sr, mono: (audio_fixture.copy(), sr),
        ),
    )


def _fake_turn_result(
    *,
    user_text: str,
    response_text: str,
    total_time_s: float,
    llm_time_s: float,
    language: str,
    llm_ttft_s: float = 0.050,
    tts_first_chunk_ms: float | None = None,
    llm_cache_hit_tokens: int | None = None,
    llm_cache_miss_tokens: int | None = None,
    llm_model_fallback_used: bool = False,
    llm_model_path_actual: str | None = None,
    llm_model_fallback_reason: str | None = None,
    template_topic_id: str | None = None,
    template_mode: str | None = None,
    template_matched: bool = False,
    speech_segments: int = 1,
    stt_provider_actual: str | None = "cpu",
    success: bool = True,
    error: str | None = None,
) -> SimpleNamespace:
    """Build a minimal fake pipeline turn result."""
    metrics = SimpleNamespace(
        vad_time_s=0.010,
        stt_load_time_s=0.020,
        stt_time_s=0.030,
        llm_load_time_s=0.040,
        llm_ttft_s=llm_ttft_s,
        llm_time_s=llm_time_s,
        tts_load_time_s=0.0,
        tts_time_s=0.0,
        playback_time_s=0.0,
        total_time_s=total_time_s,
        llm_tokens=12,
        speech_segments=speech_segments,
        stt_provider_actual=stt_provider_actual,
        tts_first_chunk_ms=tts_first_chunk_ms,
        llm_cache_hit_tokens=llm_cache_hit_tokens,
        llm_cache_miss_tokens=llm_cache_miss_tokens,
        llm_model_fallback_used=llm_model_fallback_used,
        llm_model_path_actual=llm_model_path_actual,
        llm_model_fallback_reason=llm_model_fallback_reason,
        template_topic_id=template_topic_id,
        template_mode=template_mode,
        template_matched=template_matched,
    )
    return SimpleNamespace(
        user_text=user_text,
        response_text=response_text,
        audio_samples=None,
        sample_rate=0,
        metrics=metrics,
        detected_language=language,
        success=success,
        error=error,
    )


def _summary_record(
    *,
    lang: str,
    llm_ttft_ms: float,
    turn_index_per_lang: int | None = None,
    tts_first_chunk_ms: float | None = None,
    llm_cache_hit_tokens: int | None = None,
    llm_cache_miss_tokens: int | None = None,
    llm_model_fallback_used: bool = False,
    llm_model_path_actual: str | None = None,
    llm_model_fallback_reason: str | None = None,
    stt_provider_actual: str | None = "cpu",
) -> dict[str, Any]:
    """Build a minimal per-round record for summary aggregation tests."""
    record: dict[str, Any] = {
        "round_id": 1,
        "source_round_id": 1,
        "pass_id": "pass1",
        "global_turn_id": 0,
        "lang": lang,
        "sequence_index": 0,
        "wav_path": f"{lang}.wav",
        "input_trace_wav": f"{lang}_input.wav",
        "response_wav": None,
        "gt_text": "hello",
        "stt_pred": "hello",
        "llm_response": "reply",
        "detected_language": lang,
        "stt_provider_actual": stt_provider_actual,
        "stt_provider_configured": "cpu",
        "stt_provider_requested": None,
        "core_success": True,
        "success": True,
        "failure_reason": None,
        "error": None,
        "duration_s": 0.1,
        "audio_duration_ms": 100.0,
        "audio_padded_ms": 500.0,
        "speech_segments": 1,
        "vad_miss": False,
        "vad_miss_reason": None,
        "llm_tokens": 10,
        "tts_first_chunk_ms": tts_first_chunk_ms,
        "llm_cache_hit_tokens": llm_cache_hit_tokens,
        "llm_cache_miss_tokens": llm_cache_miss_tokens,
        "llm_model_fallback_used": llm_model_fallback_used,
        "llm_model_path_actual": llm_model_path_actual,
        "llm_model_fallback_reason": llm_model_fallback_reason,
        "template_topic_id": None,
        "template_mode": None,
        "template_matched": False,
        "tts_wav_bytes": 0,
        "tts_wav_frames": 0,
        "tts_synth_error": False,
        "tts_load_error": False,
        "system_ram_mb": 1000.0,
        "process_rss_mb": 200.0,
        "timings_ms": {
            "vad_ms": 10.0,
            "stt_load_ms": 20.0,
            "stt_ms": 30.0,
            "stt_total_ms": 50.0,
            "llm_load_ms": 40.0,
            "llm_ttft_ms": llm_ttft_ms,
            "llm_ms": 60.0,
            "tts_load_ms": 0.0,
            "tts_ms": 0.0,
            "playback_ms": 0.0,
            "first_sound_ms": 120.0,
            "total_ms": 180.0,
        },
    }
    if turn_index_per_lang is not None:
        record["turn_index_per_lang"] = turn_index_per_lang
    return record


def _read_run_records(run_dir: Path) -> list[dict[str, Any]]:
    """Read the mix runner JSONL records from a completed fake run."""
    return [
        json.loads(line)
        for line in (run_dir / "rounds.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _round_record_for_test(
    tmp_path: Path,
    result: SimpleNamespace,
    *,
    gt_text: str = "hello",
    audio_duration_ms: float = 100.0,
    audio_padded_ms: float = 100.0,
) -> dict[str, Any]:
    """Build one round record through the production row writer."""
    return e2e_qwen3_asr_mix._make_round_record(
        output_dir=tmp_path,
        round_input=e2e_qwen3_asr_mix.RoundInput(
            round_id=1,
            lang="ko",
            wav_path=tmp_path / "audio_0_hello.wav",
            gt_text=gt_text,
            sequence_index=0,
            source_round_id=1,
        ),
        duration_s=audio_duration_ms / 1000.0,
        audio_duration_ms=audio_duration_ms,
        audio_padded_ms=audio_padded_ms,
        result=result,
        input_trace_path=tmp_path / "input.wav",
        response_wav_path=None,
        turn_index_per_lang=0,
        pass_id="pass1",
        global_turn_id=0,
        system_ram_mb=None,
        process_rss_mb=None,
        stt_provider_configured="cpu",
        stt_provider_requested=None,
    )


def test_load_audio_input_padding_adds_expected_samples(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Input padding should add 200 ms of leading and trailing silence."""
    samples = np.linspace(-0.5, 0.5, 16000, dtype=np.float32)
    _patch_librosa_samples(monkeypatch, samples)

    padded_audio, raw_pcm_audio, original_ms, padded_ms = e2e_qwen3_asr_mix._load_audio_input(
        tmp_path / "audio.wav",
        input_pad_ms=200,
    )

    assert padded_audio.shape == (22400,)
    assert raw_pcm_audio.shape == (16000,)
    assert original_ms == 1000.0
    assert padded_ms == 1400.0
    assert np.count_nonzero(padded_audio[:3200]) == 0
    assert np.count_nonzero(padded_audio[-3200:]) == 0


def test_load_audio_input_zero_padding_passes_through(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Zero input padding should return the normalized input without extra samples."""
    samples = np.linspace(-0.25, 0.25, 16000, dtype=np.float32)
    _patch_librosa_samples(monkeypatch, samples)

    padded_audio, _raw_pcm_audio, original_ms, padded_ms = e2e_qwen3_asr_mix._load_audio_input(
        tmp_path / "audio.wav",
        input_pad_ms=0,
    )

    np.testing.assert_array_equal(padded_audio, samples)
    assert original_ms == 1000.0
    assert padded_ms == 1000.0


def test_load_audio_input_padding_preserves_float32_dtype(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Padded pipeline input should keep the post-normalization float32 dtype."""
    samples = np.zeros(16000, dtype=np.float32)
    _patch_librosa_samples(monkeypatch, samples)

    padded_audio, _raw_pcm_audio, _original_ms, _padded_ms = e2e_qwen3_asr_mix._load_audio_input(
        tmp_path / "audio.wav",
        input_pad_ms=200,
    )

    assert padded_audio.dtype == np.float32


def test_load_audio_input_padding_short_audio_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Short source audio should be padded symmetrically without truncation."""
    samples = np.zeros(8000, dtype=np.float32)
    _patch_librosa_samples(monkeypatch, samples)

    padded_audio, _raw_pcm_audio, original_ms, padded_ms = e2e_qwen3_asr_mix._load_audio_input(
        tmp_path / "audio.wav",
        input_pad_ms=200,
    )

    assert padded_audio.shape == (14400,)
    assert original_ms == 500.0
    assert padded_ms == 900.0


def test_conversation_per_lang_flag_parse() -> None:
    """The continuous per-language conversation flag should default off."""
    parser = e2e_qwen3_asr_mix.build_parser()

    default_args = parser.parse_args(["--ko-dir", "/tmp/ko", "--en-dir", "/tmp/en"])
    enabled_args = parser.parse_args(
        [
            "--ko-dir",
            "/tmp/ko",
            "--en-dir",
            "/tmp/en",
            "--conversation-per-lang",
        ],
    )

    assert default_args.conversation_per_lang is False
    assert enabled_args.conversation_per_lang is True


def test_expect_stt_provider_flag_parse() -> None:
    """The provider expectation flag should accept only supported providers."""
    parser = e2e_qwen3_asr_mix.build_parser()

    args = parser.parse_args(
        ["--ko-dir", "/tmp/ko", "--en-dir", "/tmp/en", "--expect-stt-provider", "cpu"],
    )
    assert args.expect_stt_provider == "cpu"

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--ko-dir",
                "/tmp/ko",
                "--en-dir",
                "/tmp/en",
                "--expect-stt-provider",
                "metal",
            ],
        )


def test_input_pad_ms_cli_default_parse() -> None:
    """The input padding flag should default to the PR 2 value."""
    parser = e2e_qwen3_asr_mix.build_parser()

    args = parser.parse_args(["--ko-dir", "/tmp/ko", "--en-dir", "/tmp/en"])

    assert args.input_pad_ms == 200


def test_input_pad_ms_cli_negative_rejected() -> None:
    """Negative input padding should fail argparse validation."""
    parser = e2e_qwen3_asr_mix.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--ko-dir", "/tmp/ko", "--en-dir", "/tmp/en", "--input-pad-ms", "-1"],
        )


def test_input_pad_ms_cli_zero_parse() -> None:
    """Zero input padding should be accepted for legacy behavior."""
    parser = e2e_qwen3_asr_mix.build_parser()

    args = parser.parse_args(
        ["--ko-dir", "/tmp/ko", "--en-dir", "/tmp/en", "--input-pad-ms", "0"],
    )

    assert args.input_pad_ms == 0


def test_input_pad_ms_cli_explicit_200_parse() -> None:
    """Explicit 200 ms input padding should parse cleanly."""
    parser = e2e_qwen3_asr_mix.build_parser()

    args = parser.parse_args(
        ["--ko-dir", "/tmp/ko", "--en-dir", "/tmp/en", "--input-pad-ms", "200"],
    )

    assert args.input_pad_ms == 200


def test_repeat_passes_accepts_only_supported_range() -> None:
    """Repeat passes should accept 1..20 and reject values outside that range."""
    parser = e2e_qwen3_asr_mix.build_parser()

    for value in ("1", "5", "20"):
        args = parser.parse_args(
            ["--ko-dir", "/tmp/ko", "--en-dir", "/tmp/en", "--repeat-passes", value],
        )
        assert args.repeat_passes == int(value)

    for value in ("0", "-1", "21"):
        with pytest.raises(SystemExit):
            parser.parse_args(
                ["--ko-dir", "/tmp/ko", "--en-dir", "/tmp/en", "--repeat-passes", value],
            )


def test_llm_n_gpu_layers_flag_parse_and_rejects_non_int() -> None:
    """The LLM GPU-layer override should accept integers only."""
    parser = e2e_qwen3_asr_mix.build_parser()

    args = parser.parse_args(
        ["--ko-dir", "/tmp/ko", "--en-dir", "/tmp/en", "--llm-n-gpu-layers", "99"],
    )
    assert args.llm_n_gpu_layers == 99

    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--ko-dir", "/tmp/ko", "--en-dir", "/tmp/en", "--llm-n-gpu-layers", "many"],
        )


def test_latency_table_markdown_header_is_utf8_canonical() -> None:
    """The mix-runner latency table should preserve the canonical UTF-8 header."""
    table = e2e_qwen3_asr_mix._build_latency_table(
        [
            _summary_record(
                lang="ko",
                llm_ttft_ms=100.0,
            ),
        ],
    )

    first_line = table.splitlines()[0]
    assert first_line == (
        "| Turn | VAD | STT | LLM로드 | TTFT | LLM추론 | TTS로드 | "
        "TTS합성 | 재생 | 첫소리까지 | 전체 |"
    )
    assert "�" not in table
    assert "濡" not in table


def test_max_history_turns_flag_parse() -> None:
    """The max-history-turns override should parse only when explicitly provided."""
    parser = e2e_qwen3_asr_mix.build_parser()

    default_args = parser.parse_args(["--ko-dir", "/tmp/ko", "--en-dir", "/tmp/en"])
    override_args = parser.parse_args(
        [
            "--ko-dir",
            "/tmp/ko",
            "--en-dir",
            "/tmp/en",
            "--max-history-turns",
            "6",
        ],
    )

    assert default_args.max_history_turns is None
    assert override_args.max_history_turns == 6


def test_max_history_turns_negative_rejected() -> None:
    """Negative history-turn overrides should fail argparse validation."""
    parser = e2e_qwen3_asr_mix.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--ko-dir",
                "/tmp/ko",
                "--en-dir",
                "/tmp/en",
                "--max-history-turns",
                "-1",
            ],
        )


def test_max_history_turns_zero_accepted() -> None:
    """Zero is valid and means no retained history turn-pairs."""
    parser = e2e_qwen3_asr_mix.build_parser()

    args = parser.parse_args(
        [
            "--ko-dir",
            "/tmp/ko",
            "--en-dir",
            "/tmp/en",
            "--max-history-turns",
            "0",
        ],
    )

    assert args.max_history_turns == 0


def test_max_history_tokens_flag_parse() -> None:
    """The max-history-tokens override should parse only when explicitly provided."""
    parser = e2e_qwen3_asr_mix.build_parser()

    default_args = parser.parse_args(["--ko-dir", "/tmp/ko", "--en-dir", "/tmp/en"])
    override_args = parser.parse_args(
        [
            "--ko-dir",
            "/tmp/ko",
            "--en-dir",
            "/tmp/en",
            "--max-history-tokens",
            "800",
        ],
    )

    assert default_args.max_history_tokens is None
    assert override_args.max_history_tokens == 800


def test_pipeline_config_history_override_build_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pipeline construction should pass history overrides only when present."""
    from core.pipeline import PipelineConfig

    class CapturingPipeline:
        """Minimal pipeline that exposes the built config."""

        def __init__(self, manager: object, config: PipelineConfig) -> None:
            self.manager = manager
            self.config = config
            self._conversation_dir = Path(".")

    monkeypatch.setattr(
        e2e_qwen3_asr_mix,
        "_get_runtime_classes",
        lambda: (object, object, PipelineConfig, CapturingPipeline),
    )

    default_pipelines = e2e_qwen3_asr_mix._build_pipelines(
        manager=object(),
        args=argparse.Namespace(
            output_device=None,
            max_history_turns=None,
            max_history_tokens=None,
        ),
        output_dir=tmp_path,
    )
    override_pipelines = e2e_qwen3_asr_mix._build_pipelines(
        manager=object(),
        args=argparse.Namespace(
            output_device=None,
            max_history_turns=6,
            max_history_tokens=800,
        ),
        output_dir=tmp_path,
    )

    assert default_pipelines["ko"].config.max_history_turns == PipelineConfig().max_history_turns
    assert default_pipelines["ko"].config.max_history_tokens == PipelineConfig().max_history_tokens
    assert override_pipelines["ko"].config.max_history_turns == 6
    assert override_pipelines["ko"].config.max_history_tokens == 800


def test_conversation_per_lang_controls_clear_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mix runner should clear per round only when conversation mode is disabled."""
    ko_dir = tmp_path / "ko"
    en_dir = tmp_path / "en"
    _touch_audio_files(ko_dir, ["audio_0_ko_zero.wav"])
    _touch_audio_files(en_dir, ["audio_0_en_zero.wav"])

    conversation_clear_calls: list[str] = []
    _patch_runtime(
        monkeypatch,
        [
            _fake_turn_result(
                user_text="ko text",
                response_text="ko reply",
                total_time_s=0.100,
                llm_time_s=0.060,
                language="ko",
            ),
            _fake_turn_result(
                user_text="en text",
                response_text="en reply",
                total_time_s=0.200,
                llm_time_s=0.070,
                language="en",
            ),
        ],
        clear_history_log=conversation_clear_calls,
    )
    conversation_result = e2e_qwen3_asr_mix.main(
        [
            "--ko-dir",
            str(ko_dir),
            "--en-dir",
            str(en_dir),
            "--output-root",
            str(tmp_path / "conversation_output"),
            "--skip-tts",
            "--skip-preflight",
            "--conversation-per-lang",
        ],
    )

    default_clear_calls: list[str] = []
    _patch_runtime(
        monkeypatch,
        [
            _fake_turn_result(
                user_text="ko text",
                response_text="ko reply",
                total_time_s=0.100,
                llm_time_s=0.060,
                language="ko",
            ),
            _fake_turn_result(
                user_text="en text",
                response_text="en reply",
                total_time_s=0.200,
                llm_time_s=0.070,
                language="en",
            ),
        ],
        clear_history_log=default_clear_calls,
    )
    default_result = e2e_qwen3_asr_mix.main(
        [
            "--ko-dir",
            str(ko_dir),
            "--en-dir",
            str(en_dir),
            "--output-root",
            str(tmp_path / "default_output"),
            "--skip-tts",
            "--skip-preflight",
        ],
    )

    assert conversation_result == 0
    assert conversation_clear_calls == []
    assert default_result == 0
    assert default_clear_calls == ["ko", "en"]


def test_discover_round_pairs_interleaves_ko_en(tmp_path: Path) -> None:
    """Round discovery should interleave KO and EN files in index order."""
    ko_dir = tmp_path / "ko"
    en_dir = tmp_path / "en"
    _touch_audio_files(
        ko_dir,
        [
            "audio_0_ko_zero.wav",
            "audio_1_ko_one.wav",
            "audio_2_ko_two.wav",
        ],
    )
    _touch_audio_files(
        en_dir,
        [
            "audio_0_en_zero.wav",
            "audio_1_en_one.wav",
            "audio_2_en_two.wav",
        ],
    )

    rounds = e2e_qwen3_asr_mix.discover_round_pairs(ko_dir, en_dir)

    assert [round_item.lang for round_item in rounds] == ["ko", "en", "ko", "en", "ko", "en"]
    assert [round_item.sequence_index for round_item in rounds] == [0, 0, 1, 1, 2, 2]
    assert [round_item.source_round_id for round_item in rounds] == [1, 1, 2, 2, 3, 3]
    assert [round_item.wav_path.name for round_item in rounds] == [
        "audio_0_ko_zero.wav",
        "audio_0_en_zero.wav",
        "audio_1_ko_one.wav",
        "audio_1_en_one.wav",
        "audio_2_ko_two.wav",
        "audio_2_en_two.wav",
    ]


def test_discover_round_pairs_handles_unequal_counts(tmp_path: Path, caplog: Any) -> None:
    """Unequal KO/EN counts should warn and interleave only complete pairs."""
    ko_dir = tmp_path / "ko"
    en_dir = tmp_path / "en"
    _touch_audio_files(
        ko_dir,
        [
            "audio_0_ko_zero.wav",
            "audio_1_ko_one.wav",
            "audio_2_ko_two.wav",
            "audio_3_ko_three.wav",
            "audio_4_ko_four.wav",
        ],
    )
    _touch_audio_files(
        en_dir,
        [
            "audio_0_en_zero.wav",
            "audio_1_en_one.wav",
            "audio_2_en_two.wav",
        ],
    )

    with caplog.at_level(logging.WARNING):
        rounds = e2e_qwen3_asr_mix.discover_round_pairs(ko_dir, en_dir)

    assert len(rounds) == 6
    assert "imbalance" in caplog.text.lower()
    assert [round_item.lang for round_item in rounds] == ["ko", "en", "ko", "en", "ko", "en"]


def test_filename_to_gt_strips_audio_prefix_and_underscores() -> None:
    """Filename parsing should strip the audio prefix and convert underscores to spaces."""
    path = Path("audio_0_Mungi_is_really_fun.wav")

    assert e2e_qwen3_asr_mix.filename_to_gt_text(path) == "Mungi is really fun"


def test_main_creates_output_directory_structure(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """The runner should create the expected output tree and files."""
    ko_dir = tmp_path / "ko"
    en_dir = tmp_path / "en"
    output_root = tmp_path / "output"
    _touch_audio_files(ko_dir, ["audio_0_ko_zero.wav"])
    _touch_audio_files(en_dir, ["audio_0_en_zero.wav"])

    _patch_runtime(
        monkeypatch,
        [
            _fake_turn_result(
                user_text="ko text",
                response_text="ko reply",
                total_time_s=0.100,
                llm_time_s=0.060,
                language="ko",
            ),
            _fake_turn_result(
                user_text="en text",
                response_text="en reply",
                total_time_s=0.200,
                llm_time_s=0.070,
                language="en",
            ),
        ],
    )

    result = e2e_qwen3_asr_mix.main(
        [
            "--ko-dir",
            str(ko_dir),
            "--en-dir",
            str(en_dir),
            "--output-root",
            str(output_root),
            "--skip-tts",
            "--skip-preflight",
        ],
    )

    assert result == 0
    run_dirs = list(output_root.glob("qwen3_mix_*"))
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert (run_dir / "rounds.jsonl").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "run.log").exists()
    assert (run_dir / "input_wavs").is_dir()
    assert (run_dir / "response_wavs").is_dir()
    assert list((run_dir / "response_wavs").iterdir()) == []


def test_mocked_run_writes_stage2_summary_and_round_schema(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """A mocked smoke run should include the additive Stage-2 schemas."""
    ko_dir = tmp_path / "ko"
    en_dir = tmp_path / "en"
    output_root = tmp_path / "output"
    _touch_audio_files(ko_dir, ["audio_0_ko_zero.wav"])
    _touch_audio_files(en_dir, ["audio_0_en_zero.wav"])

    _patch_runtime(
        monkeypatch,
        [
            _fake_turn_result(
                user_text="ko text",
                response_text="ko reply",
                total_time_s=0.100,
                llm_time_s=0.060,
                language="ko",
                llm_model_fallback_used=True,
                llm_model_path_actual="/models/gemma-e2b.gguf",
                llm_model_fallback_reason="primary missing",
                template_topic_id="swimming",
                template_mode="guide",
                template_matched=True,
            ),
            _fake_turn_result(
                user_text="en text",
                response_text="en reply",
                total_time_s=0.200,
                llm_time_s=0.070,
                language="en",
            ),
        ],
    )

    result = e2e_qwen3_asr_mix.main(
        [
            "--ko-dir",
            str(ko_dir),
            "--en-dir",
            str(en_dir),
            "--output-root",
            str(output_root),
            "--skip-tts",
            "--skip-preflight",
        ],
    )

    assert result == 0
    run_dir = next(output_root.glob("qwen3_mix_*"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    records = _read_run_records(run_dir)

    for key in (
        "mungi_llm_resident",
        "mungi_stt_resident",
        "mungi_tts_resident",
        "llm_n_gpu_layers_resolved",
        "stt_provider_actual",
        "stt_provider_configured",
        "stt_provider_requested",
        "stt_provider_resolved",
        "sherpa_onnx_version",
        "stt_load_count",
        "tts_load_count",
        "tts_load_error_count",
        "tts_synth_error_count",
        "repeat_passes",
        "input_pad_ms",
        "runner",
        "model_sha256",
    ):
        assert key in summary
    assert summary["runner"] == "e2e_qwen3_asr_mix"
    assert summary["repeat_passes"] == 1
    assert summary["input_pad_ms"] == 200
    assert summary["stt_provider_actual"] == "cpu"
    assert summary["stt_provider_configured"] == "cpu"
    assert summary["stt_provider_resolved"] == "cpu"
    assert summary["latency_table_markdown"].startswith("| Turn | VAD | STT | LLM로드 |")

    row = records[0]
    for key in (
        "pass_id",
        "global_turn_id",
        "source_round_id",
        "stt_provider_actual",
        "stt_provider_configured",
        "stt_provider_requested",
        "vad_miss",
        "vad_miss_reason",
        "audio_duration_ms",
        "audio_padded_ms",
        "core_success",
        "failure_reason",
        "llm_model_fallback_used",
        "llm_model_path_actual",
        "llm_model_fallback_reason",
        "template_topic_id",
        "template_mode",
        "template_matched",
        "tts_wav_bytes",
        "tts_wav_frames",
        "tts_synth_error",
        "tts_load_error",
        "system_ram_mb",
        "process_rss_mb",
    ):
        assert key in row
    assert row["source_round_id"] == 1
    assert row["template_topic_id"] == "swimming"
    assert row["stt_provider_actual"] == "cpu"
    assert row["stt_provider_configured"] == "cpu"
    assert row["core_success"] is True
    assert row["success"] is True
    assert row["failure_reason"] is None
    assert row["llm_model_fallback_used"] is True
    assert row["llm_model_path_actual"] == "/models/gemma-e2b.gguf"
    assert row["llm_model_fallback_reason"] == "primary missing"
    assert row["vad_miss"] is False
    assert row["vad_miss_reason"] is None
    assert row["audio_duration_ms"] == 100.0
    assert row["audio_padded_ms"] == 500.0


def test_mocked_run_records_original_and_padded_audio_durations(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Row telemetry should keep original duration and add the padded duration."""
    ko_dir = tmp_path / "ko"
    en_dir = tmp_path / "en"
    output_root = tmp_path / "output"
    _touch_audio_files(ko_dir, ["audio_0_ko_one_second.wav"])
    _touch_audio_files(en_dir, ["audio_0_en_one_second.wav"])
    audio_input_log: list[np.ndarray] = []
    one_second_audio = np.zeros(16000, dtype=np.float32)
    _patch_runtime(
        monkeypatch,
        [
            _fake_turn_result(
                user_text="ko text",
                response_text="ko reply",
                total_time_s=0.100,
                llm_time_s=0.060,
                language="ko",
            ),
            _fake_turn_result(
                user_text="en text",
                response_text="en reply",
                total_time_s=0.200,
                llm_time_s=0.070,
                language="en",
            ),
        ],
        audio_input_log=audio_input_log,
        librosa_audio=one_second_audio,
    )

    result = e2e_qwen3_asr_mix.main(
        [
            "--ko-dir",
            str(ko_dir),
            "--en-dir",
            str(en_dir),
            "--output-root",
            str(output_root),
            "--skip-tts",
            "--skip-preflight",
            "--input-pad-ms",
            "200",
        ],
    )

    assert result == 0
    run_dir = next(output_root.glob("qwen3_mix_*"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    records = _read_run_records(run_dir)

    assert summary["input_pad_ms"] == 200
    assert records[0]["audio_duration_ms"] == 1000.0
    assert records[0]["audio_padded_ms"] == 1400.0
    assert audio_input_log[0].shape == (22400,)


def test_input_trace_wav_preserves_raw_source_with_padding(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """The trace WAV artifact should copy/link the source file, not padded audio."""
    ko_dir = tmp_path / "ko"
    en_dir = tmp_path / "en"
    output_root = tmp_path / "output"
    ko_source = ko_dir / "audio_0_ko_source.wav"
    en_source = en_dir / "audio_0_en_source.wav"
    _write_source_bytes(ko_source, b"ko-original-wav-bytes")
    _write_source_bytes(en_source, b"en-original-wav-bytes")
    _patch_runtime(
        monkeypatch,
        [
            _fake_turn_result(
                user_text="ko text",
                response_text="ko reply",
                total_time_s=0.100,
                llm_time_s=0.060,
                language="ko",
            ),
            _fake_turn_result(
                user_text="en text",
                response_text="en reply",
                total_time_s=0.200,
                llm_time_s=0.070,
                language="en",
            ),
        ],
        librosa_audio=np.zeros(16000, dtype=np.float32),
    )

    result = e2e_qwen3_asr_mix.main(
        [
            "--ko-dir",
            str(ko_dir),
            "--en-dir",
            str(en_dir),
            "--output-root",
            str(output_root),
            "--skip-tts",
            "--skip-preflight",
            "--input-pad-ms",
            "200",
        ],
    )

    assert result == 0
    run_dir = next(output_root.glob("qwen3_mix_*"))
    records = _read_run_records(run_dir)
    trace_path = run_dir / records[0]["input_trace_wav"]

    assert trace_path.read_bytes() == ko_source.read_bytes()
    assert trace_path.stat().st_size == ko_source.stat().st_size


def test_round_record_overlays_success_for_vad_audio_too_short(tmp_path: Path) -> None:
    """A non-empty ground-truth row with no VAD segments should fail the row overlay."""
    result = _fake_turn_result(
        user_text="",
        response_text="",
        total_time_s=0.100,
        llm_time_s=0.0,
        language="ko",
        speech_segments=0,
    )

    record = _round_record_for_test(tmp_path, result, audio_duration_ms=299.0)

    assert record["core_success"] is True
    assert record["success"] is False
    assert record["vad_miss"] is True
    assert record["vad_miss_reason"] == "audio_too_short"
    assert record["failure_reason"] == "vad_miss"


def test_vad_miss_reason_uses_original_duration_not_padded_duration(tmp_path: Path) -> None:
    """Padded duration should not mask the audio-too-short VAD miss classification."""
    result = _fake_turn_result(
        user_text="",
        response_text="",
        total_time_s=0.100,
        llm_time_s=0.0,
        language="ko",
        speech_segments=0,
    )

    record = _round_record_for_test(
        tmp_path,
        result,
        audio_duration_ms=200.0,
        audio_padded_ms=600.0,
    )

    assert record["audio_duration_ms"] == 200.0
    assert record["audio_padded_ms"] == 600.0
    assert record["vad_miss"] is True
    assert record["vad_miss_reason"] == "audio_too_short"


def test_round_record_classifies_silence_and_unknown_no_segments(tmp_path: Path) -> None:
    """VAD miss classification should distinguish adequate audio from unknown duration."""
    result = _fake_turn_result(
        user_text="",
        response_text="",
        total_time_s=0.500,
        llm_time_s=0.0,
        language="ko",
        speech_segments=0,
    )

    record = _round_record_for_test(tmp_path, result, audio_duration_ms=500.0)

    assert record["vad_miss"] is True
    assert record["vad_miss_reason"] == "silence_detected"
    assert e2e_qwen3_asr_mix._vad_miss_reason(True, None) == "unknown_no_segments"


def test_round_record_runtime_error_failure_reason(tmp_path: Path) -> None:
    """Runtime failures should keep the core result failure reason."""
    result = _fake_turn_result(
        user_text="hello",
        response_text="",
        total_time_s=0.100,
        llm_time_s=0.0,
        language="ko",
        success=False,
        error="runtime boom",
    )

    record = _round_record_for_test(tmp_path, result, audio_duration_ms=500.0)

    assert record["core_success"] is False
    assert record["success"] is False
    assert record["vad_miss"] is False
    assert record["failure_reason"] == "runtime_error"


def test_round_record_writes_stt_telemetry(tmp_path: Path) -> None:
    """The row writer should serialize STT provider telemetry."""
    result = _fake_turn_result(
        user_text="hello",
        response_text="reply",
        total_time_s=0.100,
        llm_time_s=0.050,
        language="ko",
        stt_provider_actual="cuda",
    )

    record = _round_record_for_test(tmp_path, result, audio_duration_ms=500.0)

    assert record["stt_provider_actual"] == "cuda"
    assert record["stt_provider_configured"] == "cpu"
    assert record["stt_provider_requested"] is None


def test_expect_stt_provider_cpu_passes_when_actual_cpu(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """The provider expectation should pass when actual turn telemetry matches."""
    ko_dir = tmp_path / "ko"
    en_dir = tmp_path / "en"
    output_root = tmp_path / "output"
    _touch_audio_files(ko_dir, ["audio_0_ko_zero.wav"])
    _touch_audio_files(en_dir, ["audio_0_en_zero.wav"])
    _patch_runtime(
        monkeypatch,
        [
            _fake_turn_result(
                user_text="ko text",
                response_text="ko reply",
                total_time_s=0.100,
                llm_time_s=0.060,
                language="ko",
                stt_provider_actual="cpu",
            ),
            _fake_turn_result(
                user_text="en text",
                response_text="en reply",
                total_time_s=0.200,
                llm_time_s=0.070,
                language="en",
                stt_provider_actual="cpu",
            ),
        ],
    )

    result = e2e_qwen3_asr_mix.main(
        [
            "--ko-dir",
            str(ko_dir),
            "--en-dir",
            str(en_dir),
            "--output-root",
            str(output_root),
            "--skip-tts",
            "--skip-preflight",
            "--expect-stt-provider",
            "cpu",
        ],
    )

    assert result == 0


def test_expect_stt_provider_cuda_fails_fast_when_actual_cpu(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """The provider expectation should stop the run on the first mismatch."""
    ko_dir = tmp_path / "ko"
    en_dir = tmp_path / "en"
    output_root = tmp_path / "output"
    _touch_audio_files(ko_dir, ["audio_0_ko_zero.wav"])
    _touch_audio_files(en_dir, ["audio_0_en_zero.wav"])
    _patch_runtime(
        monkeypatch,
        [
            _fake_turn_result(
                user_text="ko text",
                response_text="ko reply",
                total_time_s=0.100,
                llm_time_s=0.060,
                language="ko",
                stt_provider_actual="cpu",
            ),
        ],
    )

    result = e2e_qwen3_asr_mix.main(
        [
            "--ko-dir",
            str(ko_dir),
            "--en-dir",
            str(en_dir),
            "--output-root",
            str(output_root),
            "--skip-tts",
            "--skip-preflight",
            "--expect-stt-provider",
            "cuda",
        ],
    )

    assert result == 1
    run_dir = next(output_root.glob("qwen3_mix_*"))
    records = _read_run_records(run_dir)
    assert [record.get("record_type") for record in records] == ["pass_failure"]
    assert records[0]["failure_reason"] == "runtime_error"
    assert "STT provider mismatch" in records[0]["error_message"]


def test_repeat_passes_two_passes_with_monotonic_global_turn_ids(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Repeat passes should reuse one process and write monotonic global turn ids."""
    ko_dir = tmp_path / "ko"
    en_dir = tmp_path / "en"
    output_root = tmp_path / "output"
    _touch_audio_files(ko_dir, ["audio_0_ko_zero.wav"])
    _touch_audio_files(en_dir, ["audio_0_en_zero.wav"])
    _patch_runtime(
        monkeypatch,
        [
            _fake_turn_result(
                user_text=f"turn {index}",
                response_text="reply",
                total_time_s=0.100,
                llm_time_s=0.060,
                language="ko" if index % 2 == 0 else "en",
            )
            for index in range(4)
        ],
    )

    result = e2e_qwen3_asr_mix.main(
        [
            "--ko-dir",
            str(ko_dir),
            "--en-dir",
            str(en_dir),
            "--output-root",
            str(output_root),
            "--skip-tts",
            "--skip-preflight",
            "--max-rounds",
            "0",
            "--repeat-passes",
            "2",
        ],
    )

    assert result == 0
    records = _read_run_records(next(output_root.glob("qwen3_mix_*")))
    assert [record["pass_id"] for record in records] == ["pass1", "pass1", "pass2", "pass2"]
    assert [record["global_turn_id"] for record in records] == [0, 1, 2, 3]


def test_repeat_passes_continue_after_pass_level_exception(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """A failing pass should emit a marker and allow later passes to run."""
    ko_dir = tmp_path / "ko"
    en_dir = tmp_path / "en"
    output_root = tmp_path / "output"
    _touch_audio_files(ko_dir, ["audio_0_ko_zero.wav"])
    _touch_audio_files(en_dir, ["audio_0_en_zero.wav"])
    _patch_runtime(
        monkeypatch,
        [
            RuntimeError("pass1 deterministic failure"),
            _fake_turn_result(
                user_text="pass2 ko",
                response_text="reply",
                total_time_s=0.100,
                llm_time_s=0.060,
                language="ko",
            ),
            _fake_turn_result(
                user_text="pass2 en",
                response_text="reply",
                total_time_s=0.100,
                llm_time_s=0.060,
                language="en",
            ),
        ],
    )

    result = e2e_qwen3_asr_mix.main(
        [
            "--ko-dir",
            str(ko_dir),
            "--en-dir",
            str(en_dir),
            "--output-root",
            str(output_root),
            "--skip-tts",
            "--skip-preflight",
            "--repeat-passes",
            "2",
        ],
    )

    assert result == 0
    run_dir = next(output_root.glob("qwen3_mix_*"))
    records = _read_run_records(run_dir)
    failure_records = [record for record in records if record.get("record_type") == "pass_failure"]
    pass2_records = [record for record in records if record["pass_id"] == "pass2"]
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

    assert len(failure_records) == 1
    assert failure_records[0]["pass_id"] == "pass1"
    assert failure_records[0]["success"] is False
    assert failure_records[0]["error_message"] == "pass1 deterministic failure"
    assert failure_records[0]["audio_duration_ms"] == 100.0
    assert failure_records[0]["audio_padded_ms"] == 500.0
    assert failure_records[0]["llm_model_fallback_used"] is False
    assert failure_records[0]["llm_model_path_actual"] is None
    assert failure_records[0]["llm_model_fallback_reason"] is None
    assert len(pass2_records) == 2
    assert [record["global_turn_id"] for record in records] == [0, 1, 2]
    assert summary["passes_failed"] == ["pass1"]


def test_repeat_passes_all_failed_returns_nonzero_and_records_summary(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """A run with no successful pass should return non-zero and list all failed passes."""
    ko_dir = tmp_path / "ko"
    en_dir = tmp_path / "en"
    output_root = tmp_path / "output"
    _touch_audio_files(ko_dir, ["audio_0_ko_zero.wav"])
    _touch_audio_files(en_dir, ["audio_0_en_zero.wav"])
    _patch_runtime(
        monkeypatch,
        [
            RuntimeError("pass1 deterministic failure"),
            RuntimeError("pass2 deterministic failure"),
        ],
    )

    result = e2e_qwen3_asr_mix.main(
        [
            "--ko-dir",
            str(ko_dir),
            "--en-dir",
            str(en_dir),
            "--output-root",
            str(output_root),
            "--skip-tts",
            "--skip-preflight",
            "--repeat-passes",
            "2",
        ],
    )

    assert result == 1
    run_dir = next(output_root.glob("qwen3_mix_*"))
    records = _read_run_records(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

    assert [record.get("record_type") for record in records] == [
        "pass_failure",
        "pass_failure",
    ]
    assert [record["global_turn_id"] for record in records] == [0, 1]
    assert summary["passes_failed"] == ["pass1", "pass2"]
    assert len(summary["passes_failed"]) == 2


def test_llm_n_gpu_layers_cli_overrides_env(monkeypatch: Any) -> None:
    """CLI GPU-layer value should win over the environment value."""
    parser = e2e_qwen3_asr_mix.build_parser()
    monkeypatch.setenv("MUNGI_LLM_N_GPU_LAYERS", "10")
    args = parser.parse_args(
        ["--ko-dir", "ko", "--en-dir", "en", "--llm-n-gpu-layers", "99"],
    )

    assert e2e_qwen3_asr_mix._resolve_llm_n_gpu_layers(args) == 99
    assert e2e_qwen3_asr_mix.os.environ["MUNGI_LLM_N_GPU_LAYERS"] == "99"


def test_tegrastats_unavailable_logs_warning_and_continues(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Unavailable tegrastats should warn and continue without thermal artifacts."""
    ko_dir = tmp_path / "ko"
    en_dir = tmp_path / "en"
    output_root = tmp_path / "output"
    _touch_audio_files(ko_dir, ["audio_0_ko_zero.wav"])
    _touch_audio_files(en_dir, ["audio_0_en_zero.wav"])

    class FakeMonitor:
        snapshots: list[dict[str, Any]] = []

        def __init__(self, log_path: Path) -> None:
            self.log_path = log_path

        def start(self) -> bool:
            return False

    monkeypatch.setattr(
        e2e_qwen3_asr_mix,
        "_get_thermal_helpers",
        lambda: (FakeMonitor, lambda snapshots: {"snapshots_count": len(snapshots)}),
    )
    _patch_runtime(
        monkeypatch,
        [
            _fake_turn_result(
                user_text="ko text",
                response_text="reply",
                total_time_s=0.100,
                llm_time_s=0.060,
                language="ko",
            ),
            _fake_turn_result(
                user_text="en text",
                response_text="reply",
                total_time_s=0.100,
                llm_time_s=0.060,
                language="en",
            ),
        ],
    )

    result = e2e_qwen3_asr_mix.main(
        [
            "--ko-dir",
            str(ko_dir),
            "--en-dir",
            str(en_dir),
            "--output-root",
            str(output_root),
            "--skip-tts",
            "--skip-preflight",
        ],
    )

    assert result == 0
    run_dir = next(output_root.glob("qwen3_mix_*"))
    assert "TegrastatsMonitor unavailable" in (run_dir / "run.log").read_text(encoding="utf-8")
    assert not (run_dir / "tegrastats.log").exists()


def test_repeat_passes_with_capped_pair_count_preserves_source_round_ids(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Capping before interleave should produce full KO/EN pairs per pass."""
    ko_dir = tmp_path / "ko"
    en_dir = tmp_path / "en"
    output_root = tmp_path / "output"
    _touch_audio_files(ko_dir, [f"audio_{index}_ko_{index}.wav" for index in range(5)])
    _touch_audio_files(en_dir, [f"audio_{index}_en_{index}.wav" for index in range(5)])
    _patch_runtime(
        monkeypatch,
        [
            _fake_turn_result(
                user_text=f"turn {index}",
                response_text="reply",
                total_time_s=0.100,
                llm_time_s=0.060,
                language="ko" if index % 2 == 0 else "en",
            )
            for index in range(12)
        ],
    )

    result = e2e_qwen3_asr_mix.main(
        [
            "--ko-dir",
            str(ko_dir),
            "--en-dir",
            str(en_dir),
            "--output-root",
            str(output_root),
            "--skip-tts",
            "--skip-preflight",
            "--max-rounds",
            "3",
            "--repeat-passes",
            "2",
        ],
    )

    assert result == 0
    records = _read_run_records(next(output_root.glob("qwen3_mix_*")))
    assert len(records) == 12
    assert [
        sum(1 for record in records if record["pass_id"] == pass_id)
        for pass_id in ("pass1", "pass2")
    ] == [6, 6]
    assert sorted({record["pass_id"] for record in records}) == ["pass1", "pass2"]
    assert [record["global_turn_id"] for record in records] == list(range(12))
    for pass_id in ("pass1", "pass2"):
        pass_records = [record for record in records if record["pass_id"] == pass_id]
        assert [record["source_round_id"] for record in pass_records] == [1, 1, 2, 2, 3, 3]
        assert [record["round_id"] for record in pass_records] == [1, 2, 3, 4, 5, 6]


def test_tegrastats_success_writes_summary_and_curve_artifacts(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Successful thermal monitoring should write log, summary, and curve artifacts."""
    ko_dir = tmp_path / "ko"
    en_dir = tmp_path / "en"
    output_root = tmp_path / "output"
    _touch_audio_files(ko_dir, ["audio_0_ko_zero.wav"])
    _touch_audio_files(en_dir, ["audio_0_en_zero.wav"])

    class FakeMonitor:
        def __init__(self, log_path: Path) -> None:
            self.log_path = log_path
            self.snapshots = [
                {
                    "elapsed_s": 0.0,
                    "cpu_temp_c": 50.0,
                    "gpu_temp_c": 51.0,
                    "ram_used_mb": 1000,
                    "gr3d_freq_pct": 10,
                },
                {
                    "elapsed_s": 31.0,
                    "cpu_temp_c": 52.0,
                    "gpu_temp_c": 53.0,
                    "ram_used_mb": 1100,
                    "gr3d_freq_pct": 20,
                },
            ]

        def start(self) -> bool:
            self.log_path.write_text("tegrastats sample\n", encoding="utf-8")
            return True

        def stop(self) -> None:
            return

    def _fake_summary(snapshots: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "snapshots_count": len(snapshots),
            "cpu_temp_c": {
                "start": 50.0,
                "end": 52.0,
                "min": 50.0,
                "max": 52.0,
                "avg": 51.0,
                "delta": 2.0,
            },
            "gpu_temp_c": {
                "start": 51.0,
                "end": 53.0,
                "min": 51.0,
                "max": 53.0,
                "avg": 52.0,
                "delta": 2.0,
            },
        }

    monkeypatch.setattr(
        e2e_qwen3_asr_mix, "_get_thermal_helpers", lambda: (FakeMonitor, _fake_summary)
    )
    _patch_runtime(
        monkeypatch,
        [
            _fake_turn_result(
                user_text="ko text",
                response_text="reply",
                total_time_s=0.100,
                llm_time_s=0.060,
                language="ko",
            ),
            _fake_turn_result(
                user_text="en text",
                response_text="reply",
                total_time_s=0.100,
                llm_time_s=0.060,
                language="en",
            ),
        ],
    )

    result = e2e_qwen3_asr_mix.main(
        [
            "--ko-dir",
            str(ko_dir),
            "--en-dir",
            str(en_dir),
            "--output-root",
            str(output_root),
            "--skip-tts",
            "--skip-preflight",
        ],
    )

    assert result == 0
    run_dir = next(output_root.glob("qwen3_mix_*"))
    assert (run_dir / "tegrastats.log").exists()
    thermal_summary = json.loads((run_dir / "thermal_summary.json").read_text(encoding="utf-8"))
    thermal_curve = json.loads((run_dir / "thermal_curve.json").read_text(encoding="utf-8"))
    assert thermal_summary["cpu_temp_c"]["avg"] == 51.0
    assert thermal_summary["thermal_max_c"] == 53.0
    assert "duration_s" in thermal_summary
    assert len(thermal_curve) == 2


def test_summary_aggregates_per_language_metrics(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Summary output should include language counts and aggregate latency metrics."""
    ko_dir = tmp_path / "ko"
    en_dir = tmp_path / "en"
    output_root = tmp_path / "output"
    _touch_audio_files(
        ko_dir,
        [
            "audio_0_ko_zero.wav",
            "audio_1_ko_one.wav",
        ],
    )
    _touch_audio_files(
        en_dir,
        [
            "audio_0_en_zero.wav",
            "audio_1_en_one.wav",
        ],
    )

    _patch_runtime(
        monkeypatch,
        [
            _fake_turn_result(
                user_text="ko one",
                response_text="reply one",
                total_time_s=0.100,
                llm_time_s=0.050,
                language="ko",
            ),
            _fake_turn_result(
                user_text="en one",
                response_text="reply two",
                total_time_s=0.200,
                llm_time_s=0.060,
                language="en",
            ),
            _fake_turn_result(
                user_text="ko two",
                response_text="reply three",
                total_time_s=0.300,
                llm_time_s=0.070,
                language="ko",
            ),
            _fake_turn_result(
                user_text="en two",
                response_text="reply four",
                total_time_s=0.400,
                llm_time_s=0.080,
                language="en",
            ),
        ],
    )

    result = e2e_qwen3_asr_mix.main(
        [
            "--ko-dir",
            str(ko_dir),
            "--en-dir",
            str(en_dir),
            "--output-root",
            str(output_root),
            "--skip-tts",
            "--skip-preflight",
        ],
    )

    assert result == 0
    run_dir = next(output_root.glob("qwen3_mix_*"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["ko_count"] == 2
    assert summary["en_count"] == 2
    assert summary["avg_total_ms"] == 250.0
    assert summary["languages"]["ko"]["avg_total_ms"] == 200.0
    assert summary["languages"]["en"]["avg_total_ms"] == 300.0


def test_turn_index_per_lang_assigned_in_mix_runner(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Main runner should assign per-language turn indices across interleaved rounds."""
    ko_dir = tmp_path / "ko"
    en_dir = tmp_path / "en"
    output_root = tmp_path / "output"
    _touch_audio_files(
        ko_dir,
        [
            "audio_0_ko_zero.wav",
            "audio_1_ko_one.wav",
        ],
    )
    _touch_audio_files(
        en_dir,
        [
            "audio_0_en_zero.wav",
            "audio_1_en_one.wav",
        ],
    )

    _patch_runtime(
        monkeypatch,
        [
            _fake_turn_result(
                user_text="ko one",
                response_text="reply one",
                total_time_s=0.100,
                llm_time_s=0.050,
                language="ko",
            ),
            _fake_turn_result(
                user_text="en one",
                response_text="reply two",
                total_time_s=0.200,
                llm_time_s=0.060,
                language="en",
            ),
            _fake_turn_result(
                user_text="ko two",
                response_text="reply three",
                total_time_s=0.300,
                llm_time_s=0.070,
                language="ko",
            ),
            _fake_turn_result(
                user_text="en two",
                response_text="reply four",
                total_time_s=0.400,
                llm_time_s=0.080,
                language="en",
            ),
        ],
    )

    result = e2e_qwen3_asr_mix.main(
        [
            "--ko-dir",
            str(ko_dir),
            "--en-dir",
            str(en_dir),
            "--output-root",
            str(output_root),
            "--skip-tts",
            "--skip-preflight",
        ],
    )

    assert result == 0
    run_dir = next(output_root.glob("qwen3_mix_*"))
    records = [
        json.loads(line)
        for line in (run_dir / "rounds.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [record["turn_index_per_lang"] for record in records] == [0, 0, 1, 1]


def test_summary_aggregation_computes_turn_index_split(tmp_path: Path) -> None:
    """Summary aggregation should split TTFT by first and later same-language turns."""
    records = [
        _summary_record(
            lang="ko",
            llm_ttft_ms=100.0,
            turn_index_per_lang=0,
            tts_first_chunk_ms=40.0,
            llm_cache_hit_tokens=8,
            llm_cache_miss_tokens=2,
        ),
        _summary_record(
            lang="en",
            llm_ttft_ms=200.0,
            turn_index_per_lang=0,
            tts_first_chunk_ms=60.0,
            llm_cache_hit_tokens=2,
            llm_cache_miss_tokens=6,
        ),
        _summary_record(
            lang="ko",
            llm_ttft_ms=300.0,
            turn_index_per_lang=1,
        ),
        _summary_record(
            lang="en",
            llm_ttft_ms=500.0,
            turn_index_per_lang=1,
        ),
    ]

    summary = e2e_qwen3_asr_mix._build_summary(
        args=argparse.Namespace(skip_tts=True),
        output_dir=tmp_path,
        rounds_path=tmp_path / "rounds.jsonl",
        log_path=tmp_path / "run.log",
        records=records,
    )

    assert summary["avg_tts_first_chunk_ms"] == 50.0
    assert summary["avg_llm_ttft_ms_first_turn"] == 150.0
    assert summary["avg_llm_ttft_ms_after_first"] == 400.0
    assert summary["avg_llm_cache_hit_rate"] == 0.525


def test_summary_aggregation_includes_observability_fields(tmp_path: Path) -> None:
    """Summary aggregation should include PR 1 observability values and legacy alias."""
    records = [
        _summary_record(
            lang="ko",
            llm_ttft_ms=100.0,
            stt_provider_actual="cpu",
        ),
        _summary_record(
            lang="en",
            llm_ttft_ms=200.0,
            stt_provider_actual="cuda",
        ),
        _summary_record(
            lang="ko",
            llm_ttft_ms=300.0,
            stt_provider_actual="cpu",
        ),
    ]

    summary = e2e_qwen3_asr_mix._build_summary(
        args=argparse.Namespace(skip_tts=True),
        output_dir=tmp_path,
        rounds_path=tmp_path / "rounds.jsonl",
        log_path=tmp_path / "run.log",
        records=records,
        stt_provider_configured="cpu",
        stt_provider_requested="cuda",
    )

    assert summary["stt_provider_actual"] == ["cpu", "cuda"]
    assert summary["stt_provider_configured"] == "cpu"
    assert summary["stt_provider_requested"] == "cuda"
    assert summary["stt_provider_resolved"] == "cpu"


def test_summary_aggregation_handles_missing_new_fields(tmp_path: Path) -> None:
    """Summary aggregation should return ``None`` for unavailable Wave 2 metrics."""
    records = [
        _summary_record(lang="ko", llm_ttft_ms=100.0),
        _summary_record(lang="en", llm_ttft_ms=200.0),
    ]

    summary = e2e_qwen3_asr_mix._build_summary(
        args=argparse.Namespace(skip_tts=True),
        output_dir=tmp_path,
        rounds_path=tmp_path / "rounds.jsonl",
        log_path=tmp_path / "run.log",
        records=records,
    )

    assert summary["avg_tts_first_chunk_ms"] is None
    assert summary["avg_llm_ttft_ms_first_turn"] is None
    assert summary["avg_llm_ttft_ms_after_first"] is None
    assert summary["avg_llm_cache_hit_rate"] is None


def test_first_sound_ms_uses_first_chunk_when_streaming() -> None:
    """First-sound latency should use first chunk timing when streaming is active."""
    metrics = SimpleNamespace(
        vad_time_s=0.373,
        stt_load_time_s=0.120,
        stt_time_s=5.708,
        llm_load_time_s=0.160,
        llm_time_s=3.608,
        tts_load_time_s=0.0,
        tts_time_s=1.700,
        tts_first_chunk_ms=978.0,
    )

    streaming_tts_ms = e2e_qwen3_asr_mix._maybe_float(metrics.tts_first_chunk_ms)
    assert streaming_tts_ms is not None
    streaming_expected_ms = (
        metrics.vad_time_s
        + metrics.stt_load_time_s
        + metrics.stt_time_s
        + metrics.llm_load_time_s
        + metrics.llm_time_s
        + metrics.tts_load_time_s
    ) * 1000.0 + streaming_tts_ms
    assert e2e_qwen3_asr_mix._first_sound_ms(metrics) == pytest.approx(
        streaming_expected_ms,
        abs=1e-3,
    )

    metrics.tts_first_chunk_ms = 0.0
    fallback_expected_ms = (
        metrics.vad_time_s
        + metrics.stt_load_time_s
        + metrics.stt_time_s
        + metrics.llm_load_time_s
        + metrics.llm_time_s
        + metrics.tts_load_time_s
        + metrics.tts_time_s
    ) * 1000.0
    assert e2e_qwen3_asr_mix._first_sound_ms(metrics) == pytest.approx(
        fallback_expected_ms,
        abs=1e-3,
    )


def test_apply_sentence_streaming_tts_passes_model_dir(monkeypatch: Any) -> None:
    """Sentence-streaming TTS should forward the manager's configured model directory."""
    helper_calls: list[tuple[str, str, str | None, str | None]] = []
    result = _fake_turn_result(
        user_text="ko text",
        response_text="First reply. Second reply.",
        total_time_s=0.100,
        llm_time_s=0.060,
        language="ko",
    )
    manager = SimpleNamespace(
        config=SimpleNamespace(
            tts_voice_style="F2",
            tts_model_dir="/fake/path",
        )
    )

    def _fake_sentence_helper(
        text: str,
        voice_style: str,
        model_dir: str | None = None,
        output_device: str | None = None,
    ) -> e2e_qwen3_asr_mix.SentenceSynthesisResult:
        """Capture helper inputs and return a deterministic streaming result."""
        helper_calls.append((text, voice_style, model_dir, output_device))
        return e2e_qwen3_asr_mix.SentenceSynthesisResult(
            total_duration_ms=800.0,
            first_chunk_ms=350.0,
            sentence_count=2,
            full_wav_path=None,
        )

    monkeypatch.setattr(
        e2e_qwen3_asr_mix,
        "synthesize_to_speaker_by_sentence",
        _fake_sentence_helper,
    )

    sentence_result = e2e_qwen3_asr_mix._apply_sentence_streaming_tts(
        result=result,
        manager=manager,
        output_device="speaker0",
    )

    assert sentence_result is not None
    assert helper_calls == [
        ("First reply. Second reply.", "F2", "/fake/path", "speaker0"),
    ]


def test_tts_streaming_flag_enables_helper(tmp_path: Path, monkeypatch: Any) -> None:
    """The CLI flag should route TTS through the sentence-streaming helper."""
    ko_dir = tmp_path / "ko"
    en_dir = tmp_path / "en"
    output_root = tmp_path / "output"
    _touch_audio_files(ko_dir, ["audio_0_ko_zero.wav"])
    _touch_audio_files(en_dir, ["audio_0_en_zero.wav"])

    helper_calls: list[tuple[str, str, str | None]] = []
    tts_call_log: list[tuple[str, str]] = []
    _patch_runtime(
        monkeypatch,
        [
            _fake_turn_result(
                user_text="ko text",
                response_text="First reply. Second reply.",
                total_time_s=0.100,
                llm_time_s=0.060,
                language="ko",
            ),
            _fake_turn_result(
                user_text="en text",
                response_text="First answer. Second answer.",
                total_time_s=0.200,
                llm_time_s=0.070,
                language="en",
            ),
        ],
        tts_call_log=tts_call_log,
    )

    def _fake_sentence_helper(
        text: str,
        voice_style: str,
        model_dir: str | None = None,
        output_device: str | None = None,
    ) -> e2e_qwen3_asr_mix.SentenceSynthesisResult:
        """Capture helper calls while preserving the streaming return payload."""
        del model_dir
        helper_calls.append((text, voice_style, output_device))
        return e2e_qwen3_asr_mix.SentenceSynthesisResult(
            total_duration_ms=800.0,
            first_chunk_ms=350.0,
            sentence_count=2,
            full_wav_path=None,
        )

    monkeypatch.setattr(
        e2e_qwen3_asr_mix,
        "synthesize_to_speaker_by_sentence",
        _fake_sentence_helper,
    )

    result = e2e_qwen3_asr_mix.main(
        [
            "--ko-dir",
            str(ko_dir),
            "--en-dir",
            str(en_dir),
            "--output-root",
            str(output_root),
            "--skip-preflight",
            "--tts-streaming",
        ],
    )

    assert result == 0
    assert len(helper_calls) == 2
    assert tts_call_log == []


def test_tts_streaming_env_var_enables_helper(tmp_path: Path, monkeypatch: Any) -> None:
    """The env alias should enable sentence streaming when the CLI flag is absent."""
    ko_dir = tmp_path / "ko"
    en_dir = tmp_path / "en"
    output_root = tmp_path / "output"
    _touch_audio_files(ko_dir, ["audio_0_ko_zero.wav"])
    _touch_audio_files(en_dir, ["audio_0_en_zero.wav"])

    helper_calls: list[str] = []
    _patch_runtime(
        monkeypatch,
        [
            _fake_turn_result(
                user_text="ko text",
                response_text="First reply. Second reply.",
                total_time_s=0.100,
                llm_time_s=0.060,
                language="ko",
            ),
            _fake_turn_result(
                user_text="en text",
                response_text="First answer. Second answer.",
                total_time_s=0.200,
                llm_time_s=0.070,
                language="en",
            ),
        ],
    )
    monkeypatch.setenv("MUNGI_TTS_STREAMING", "TrUe")

    def _fake_sentence_helper(
        text: str,
        voice_style: str,
        model_dir: str | None = None,
        output_device: str | None = None,
    ) -> e2e_qwen3_asr_mix.SentenceSynthesisResult:
        """Capture streamed text while returning the expected fake synthesis."""
        del voice_style, model_dir, output_device
        helper_calls.append(text)
        return e2e_qwen3_asr_mix.SentenceSynthesisResult(
            total_duration_ms=800.0,
            first_chunk_ms=350.0,
            sentence_count=2,
            full_wav_path=None,
        )

    monkeypatch.setattr(
        e2e_qwen3_asr_mix,
        "synthesize_to_speaker_by_sentence",
        _fake_sentence_helper,
    )

    result = e2e_qwen3_asr_mix.main(
        [
            "--ko-dir",
            str(ko_dir),
            "--en-dir",
            str(en_dir),
            "--output-root",
            str(output_root),
            "--skip-preflight",
        ],
    )

    assert result == 0
    assert helper_calls == ["First reply. Second reply.", "First answer. Second answer."]


def test_maybe_externalize_tts_skips_mm_lifecycle() -> None:
    """Streaming patching should skip only TTS lifecycle calls and restore originals."""
    original_run_tts = MagicMock(
        return_value=(np.ones(3, dtype=np.float32), 16000),
    )
    original_play_audio = MagicMock()
    original_post_tts_unload = MagicMock()
    original_mm_load = MagicMock(side_effect=lambda model_type: model_type.name)
    original_mm_unload_tts = MagicMock(return_value=None)
    pipeline = SimpleNamespace(
        _mm=SimpleNamespace(
            load=original_mm_load,
            unload_tts=original_mm_unload_tts,
        ),
        _run_tts=original_run_tts,
        _play_audio_out=original_play_audio,
        _maybe_unload_tts_after_success=original_post_tts_unload,
    )

    with e2e_qwen3_asr_mix._maybe_externalize_tts(pipeline, enabled=True):
        outer_run_tts = pipeline._run_tts
        outer_play_audio = pipeline._play_audio_out
        outer_post_tts_unload = pipeline._maybe_unload_tts_after_success
        outer_mm_load = pipeline._mm.load
        outer_mm_unload_tts = pipeline._mm.unload_tts

        with e2e_qwen3_asr_mix._maybe_externalize_tts(pipeline, enabled=True):
            assert pipeline._run_tts is not outer_run_tts
            assert pipeline._play_audio_out is not outer_play_audio
            assert pipeline._maybe_unload_tts_after_success is not outer_post_tts_unload
            assert pipeline._mm.load is not outer_mm_load
            assert pipeline._mm.unload_tts is not outer_mm_unload_tts

        assert pipeline._run_tts is outer_run_tts
        assert pipeline._play_audio_out is outer_play_audio
        assert pipeline._maybe_unload_tts_after_success is outer_post_tts_unload
        assert pipeline._mm.load is outer_mm_load
        assert pipeline._mm.unload_tts is outer_mm_unload_tts

        skipped_audio, skipped_sample_rate = pipeline._run_tts("hello")
        assert isinstance(skipped_audio, np.ndarray)
        assert skipped_audio.size == 0
        assert skipped_sample_rate == e2e_qwen3_asr_mix.DEFAULT_SKIP_TTS_SAMPLE_RATE

        assert pipeline._mm.load(ModelType.TTS) is None
        original_mm_load.assert_not_called()

        assert pipeline._mm.load(ModelType.STT) == "STT"
        original_mm_load.assert_called_once_with(ModelType.STT)

        assert pipeline._mm.unload_tts(force=True) is None
        original_mm_unload_tts.assert_not_called()

        assert pipeline._maybe_unload_tts_after_success() is None
        original_post_tts_unload.assert_not_called()

    restored_audio, restored_sample_rate = pipeline._run_tts("x")
    assert restored_sample_rate == 16000
    assert np.array_equal(restored_audio, np.ones(3, dtype=np.float32))
    original_run_tts.assert_called_once_with("x")
    assert pipeline._play_audio_out is original_play_audio
    assert pipeline._maybe_unload_tts_after_success is original_post_tts_unload
    assert pipeline._mm.load is original_mm_load
    assert pipeline._mm.unload_tts is original_mm_unload_tts

    original_run_tts.reset_mock()
    original_mm_load.reset_mock()
    original_mm_unload_tts.reset_mock()
    original_post_tts_unload.reset_mock()

    with e2e_qwen3_asr_mix._maybe_externalize_tts(pipeline, enabled=False):
        assert pipeline._run_tts is original_run_tts
        assert pipeline._play_audio_out is original_play_audio
        assert pipeline._maybe_unload_tts_after_success is original_post_tts_unload
        assert pipeline._mm.load(ModelType.TTS) == "TTS"
        original_mm_load.assert_called_once_with(ModelType.TTS)
        pipeline._mm.unload_tts(force=True)
        original_mm_unload_tts.assert_called_once_with(force=True)
        pipeline._maybe_unload_tts_after_success()
        original_post_tts_unload.assert_called_once_with()


def test_tts_streaming_cli_overrides_env(tmp_path: Path, monkeypatch: Any) -> None:
    """CLI truthy enablement should win when the env alias is explicitly false."""
    ko_dir = tmp_path / "ko"
    en_dir = tmp_path / "en"
    _touch_audio_files(ko_dir, ["audio_0_ko_zero.wav"])
    _touch_audio_files(en_dir, ["audio_0_en_zero.wav"])

    helper_calls: list[str] = []
    tts_call_log: list[tuple[str, str]] = []
    monkeypatch.setenv("MUNGI_TTS_STREAMING", "off")

    def _fake_sentence_helper(
        text: str,
        voice_style: str,
        model_dir: str | None = None,
        output_device: str | None = None,
    ) -> e2e_qwen3_asr_mix.SentenceSynthesisResult:
        """Capture text without changing the helper's deterministic payload."""
        del voice_style, model_dir, output_device
        helper_calls.append(text)
        return e2e_qwen3_asr_mix.SentenceSynthesisResult(
            total_duration_ms=900.0,
            first_chunk_ms=300.0,
            sentence_count=2,
            full_wav_path=None,
        )

    monkeypatch.setattr(
        e2e_qwen3_asr_mix,
        "synthesize_to_speaker_by_sentence",
        _fake_sentence_helper,
    )

    _patch_runtime(
        monkeypatch,
        [
            _fake_turn_result(
                user_text="ko text",
                response_text="First reply. Second reply.",
                total_time_s=0.100,
                llm_time_s=0.060,
                language="ko",
            ),
            _fake_turn_result(
                user_text="en text",
                response_text="First answer. Second answer.",
                total_time_s=0.200,
                llm_time_s=0.070,
                language="en",
            ),
        ],
        tts_call_log=tts_call_log,
    )
    result = e2e_qwen3_asr_mix.main(
        [
            "--ko-dir",
            str(ko_dir),
            "--en-dir",
            str(en_dir),
            "--output-root",
            str(tmp_path / "output_a"),
            "--skip-preflight",
            "--tts-streaming",
        ],
    )

    assert result == 0
    assert len(helper_calls) == 2
    assert tts_call_log == []


def test_cli_llm_max_tokens_sets_env(monkeypatch: Any) -> None:
    """CLI max-token override should validate range and update the env mapping."""
    parser = e2e_qwen3_asr_mix.build_parser()
    env: dict[str, str] = {}
    monkeypatch.setattr(e2e_qwen3_asr_mix.os, "environ", env)
    args = parser.parse_args(["--ko-dir", "ko", "--en-dir", "en", "--llm-max-tokens", "60"])
    e2e_qwen3_asr_mix._apply_llm_max_tokens_override(args=args, parser=parser)
    assert env["MUNGI_LLM_MAX_TOKENS"] == "60"
    for raw_value in ("0", "4097"):
        invalid_args = parser.parse_args(
            ["--ko-dir", "ko", "--en-dir", "en", "--llm-max-tokens", raw_value],
        )
        with pytest.raises(SystemExit):
            e2e_qwen3_asr_mix._apply_llm_max_tokens_override(args=invalid_args, parser=parser)


def test_tts_first_chunk_ms_recorded_on_streaming(tmp_path: Path, monkeypatch: Any) -> None:
    """Streaming mode should serialize the helper's first-chunk metric into JSONL."""
    ko_dir = tmp_path / "ko"
    en_dir = tmp_path / "en"
    output_root = tmp_path / "output"
    _touch_audio_files(ko_dir, ["audio_0_ko_zero.wav"])
    _touch_audio_files(en_dir, ["audio_0_en_zero.wav"])

    _patch_runtime(
        monkeypatch,
        [
            _fake_turn_result(
                user_text="ko text",
                response_text="First reply. Second reply.",
                total_time_s=0.100,
                llm_time_s=0.060,
                language="ko",
            ),
            _fake_turn_result(
                user_text="en text",
                response_text="First answer. Second answer.",
                total_time_s=0.200,
                llm_time_s=0.070,
                language="en",
            ),
        ],
    )
    monkeypatch.setattr(
        e2e_qwen3_asr_mix,
        "synthesize_to_speaker_by_sentence",
        lambda text, voice_style, model_dir=None, output_device=None: (
            e2e_qwen3_asr_mix.SentenceSynthesisResult(
                total_duration_ms=800.0,
                first_chunk_ms=350.0,
                sentence_count=2,
                full_wav_path=None,
            )
        ),
    )

    result = e2e_qwen3_asr_mix.main(
        [
            "--ko-dir",
            str(ko_dir),
            "--en-dir",
            str(en_dir),
            "--output-root",
            str(output_root),
            "--skip-preflight",
            "--tts-streaming",
        ],
    )

    assert result == 0
    run_dir = next(output_root.glob("qwen3_mix_*"))
    records = [
        json.loads(line)
        for line in (run_dir / "rounds.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [record["tts_first_chunk_ms"] for record in records] == [350.0, 350.0]
    assert [record["timings_ms"]["tts_first_chunk_ms"] for record in records] == [350.0, 350.0]


def test_streaming_disabled_preserves_full_waveform_path(tmp_path: Path, monkeypatch: Any) -> None:
    """With streaming disabled, the runner should use the existing full-waveform TTS path."""
    ko_dir = tmp_path / "ko"
    en_dir = tmp_path / "en"
    output_root = tmp_path / "output"
    _touch_audio_files(ko_dir, ["audio_0_ko_zero.wav"])
    _touch_audio_files(en_dir, ["audio_0_en_zero.wav"])

    helper_calls: list[str] = []
    tts_call_log: list[tuple[str, str]] = []
    _patch_runtime(
        monkeypatch,
        [
            _fake_turn_result(
                user_text="ko text",
                response_text="Only one reply.",
                total_time_s=0.100,
                llm_time_s=0.060,
                language="ko",
            ),
            _fake_turn_result(
                user_text="en text",
                response_text="Only one answer.",
                total_time_s=0.200,
                llm_time_s=0.070,
                language="en",
            ),
        ],
        tts_call_log=tts_call_log,
    )

    def _fake_sentence_helper(
        text: str,
        voice_style: str,
        model_dir: str | None = None,
        output_device: str | None = None,
    ) -> e2e_qwen3_asr_mix.SentenceSynthesisResult:
        """Capture disabled-path text without affecting fake synthesis output."""
        del voice_style, model_dir, output_device
        helper_calls.append(text)
        return e2e_qwen3_asr_mix.SentenceSynthesisResult(
            total_duration_ms=800.0,
            first_chunk_ms=350.0,
            sentence_count=1,
            full_wav_path=None,
        )

    monkeypatch.setattr(
        e2e_qwen3_asr_mix,
        "synthesize_to_speaker_by_sentence",
        _fake_sentence_helper,
    )

    result = e2e_qwen3_asr_mix.main(
        [
            "--ko-dir",
            str(ko_dir),
            "--en-dir",
            str(en_dir),
            "--output-root",
            str(output_root),
            "--skip-preflight",
        ],
    )

    assert result == 0
    assert helper_calls == []
    assert tts_call_log == [("ko", "Only one reply."), ("en", "Only one answer.")]
    run_dir = next(output_root.glob("qwen3_mix_*"))
    assert sorted(path.name for path in (run_dir / "response_wavs").glob("*.wav")) == [
        "r01_ko.wav",
        "r02_en.wav",
    ]
