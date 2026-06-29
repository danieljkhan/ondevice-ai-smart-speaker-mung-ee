"""Unit tests for models.stt_runner."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest


class FakeStream:
    """Minimal Sherpa stream stub."""

    def __init__(self, result: object) -> None:
        self.result = result
        self.accept_calls: list[tuple[int, list[float]]] = []

    def accept_waveform(self, sample_rate: int, samples: list[float]) -> None:
        self.accept_calls.append((sample_rate, samples))


class FakeRecognizer:
    """Minimal Sherpa recognizer stub."""

    def __init__(self, result: object) -> None:
        self._stream = FakeStream(result)
        self.decode_calls = 0

    def create_stream(self) -> FakeStream:
        return self._stream

    def decode_stream(self, stream: FakeStream) -> None:
        assert stream is self._stream
        self.decode_calls += 1


def test_resolve_model_size_maps_legacy_size_aliases_to_qwen3_asr() -> None:
    from models.stt_runner import _QWEN3_ASR_NAME, resolve_model_size

    for alias in ("small", "base", "medium", "large", "large-v2", "large-v3", "tiny"):
        assert resolve_model_size(alias) == _QWEN3_ASR_NAME


def test_provider_candidates_include_cpu_fallback_for_cuda() -> None:
    from models.stt_runner import _provider_candidates

    assert _provider_candidates("cuda") == ["cuda", "cpu"]
    assert _provider_candidates("cpu") == ["cpu"]


@pytest.mark.parametrize(
    ("raw_text", "expected"),
    (
        ("  ▁hello   ▁world  ", "hello world"),
        ("language Korean<asr_text>안녕", "안녕"),
        ("language English<asr_text>hello", "hello"),
        ("▁assistant▁language▁Korean<asr_text>▁안녕▁세상", "안녕 세상"),
        ("assistant\nlanguage Korean<asr_text>이걸로", "이걸로"),
        ("Language korean<asr_text>좋아", "좋아"),
        ("language Korean<asr_text>ignored<asr_text>최종", "최종"),
        ("안녕하세요", "안녕하세요"),
        ("the sheep eats", "the sheep eats"),
        ("", ""),
    ),
)
def test_normalize_transcript_text_strips_asr_template_leaks(
    raw_text: str,
    expected: str,
) -> None:
    from models.stt_runner import _normalize_transcript_text

    assert _normalize_transcript_text(raw_text) == expected


def test_run_stt_normalizes_text_and_info(monkeypatch: pytest.MonkeyPatch) -> None:
    from models import stt_runner

    fake_result = SimpleNamespace(
        text="  ▁hello   ▁world  ",
        lang="<|ko|>",
        timestamps=[0.25, 0.75],
        segment_texts=[],
        segment_timestamps=[],
    )
    recognizer = FakeRecognizer(fake_result)
    model = stt_runner.LoadedSttModel(
        recognizer=recognizer,
        backend="sherpa-onnx",
        requested_model_size="small",
        resolved_model_size=stt_runner._QWEN3_ASR_NAME,
        provider="cpu",
        model_path="/opt/mungi/ai_models/sherpa-onnx-qwen3-asr-test",
        language="ko",
    )

    monkeypatch.setattr(
        stt_runner,
        "_read_wav_samples",
        lambda _path: ([0.0] * 16000, 16000, 1.0),
    )

    segments, info = stt_runner.run_stt(model, Path("dummy.wav"), language="ko", beam_size=3)

    assert len(segments) == 1
    assert segments[0].start == 0.25
    assert segments[0].end == 0.75
    assert segments[0].text == "hello world"
    assert info["language"] == "ko"
    assert info["backend"] == "sherpa-onnx"
    assert info["provider"] == "cpu"
    assert info["resolved_model_size"] == stt_runner._QWEN3_ASR_NAME
    assert info["raw_stt_text"] == "  ▁hello   ▁world  "
    assert recognizer.decode_calls == 1
    assert recognizer.create_stream().accept_calls[0][0] == 16000


def test_resolve_model_size_qwen3_aliases() -> None:
    from models.stt_runner import _QWEN3_ASR_NAME, resolve_model_size

    aliases = (
        "qwen3-asr",
        "qwen3",
        "qwen3-asr-0.6b",
        "qwen3-asr-0.6b-int8",
        "sherpa-onnx-qwen3-asr-foo",
    )

    for alias in aliases:
        assert resolve_model_size(alias) == _QWEN3_ASR_NAME


def test_resolve_model_size_unsupported_message_includes_qwen3() -> None:
    from models.stt_runner import _QWEN3_ASR_NAME, resolve_model_size

    with pytest.raises(ValueError, match=_QWEN3_ASR_NAME):
        resolve_model_size("invalid-name")


def test_resolve_qwen3_asr_hotwords_default_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from models.stt_runner import _resolve_qwen3_asr_hotwords

    monkeypatch.delenv("MUNGI_QWEN3_ASR_HOTWORDS", raising=False)

    assert _resolve_qwen3_asr_hotwords(None) == ""


def test_resolve_qwen3_asr_hotwords_env_var_overrides_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from models.stt_runner import _resolve_qwen3_asr_hotwords

    monkeypatch.setenv("MUNGI_QWEN3_ASR_HOTWORDS", "custom1,custom2")

    assert _resolve_qwen3_asr_hotwords(None) == "custom1,custom2"


def test_resolve_qwen3_asr_hotwords_empty_explicit_disables() -> None:
    from models.stt_runner import _resolve_qwen3_asr_hotwords

    assert _resolve_qwen3_asr_hotwords("") == ""


def test_resolve_qwen3_asr_hotwords_explicit_overrides_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from models.stt_runner import _resolve_qwen3_asr_hotwords

    monkeypatch.setenv("MUNGI_QWEN3_ASR_HOTWORDS", "env_value")

    assert _resolve_qwen3_asr_hotwords("explicit") == "explicit"


def test_legacy_qwen3_asr_hotwords_contains_persona_variants() -> None:
    from models.stt_runner import (
        _HOTWORDS_BASELINE,
        _HOTWORDS_EXPLORATORY_TIER,
        _HOTWORDS_REQUIRED_TIER,
        LEGACY_QWEN3_ASR_HOTWORDS_TIER1_DEFAULT,
    )

    expected = (
        "뭉이야",
        "뭉이",
        "한글",
        "추석",
        "송편",
        "단군신화",
        "일제강점기",
        "빙하",
        "자석",
        "화산",
        "지진",
        "무지개",
        "한복",
    )

    assert len(_HOTWORDS_BASELINE) == 2
    assert len(_HOTWORDS_REQUIRED_TIER) == 4
    assert len(_HOTWORDS_EXPLORATORY_TIER) == 7
    assert _HOTWORDS_BASELINE + _HOTWORDS_REQUIRED_TIER + _HOTWORDS_EXPLORATORY_TIER == expected
    assert tuple(LEGACY_QWEN3_ASR_HOTWORDS_TIER1_DEFAULT.split(",")) == expected
    assert len(LEGACY_QWEN3_ASR_HOTWORDS_TIER1_DEFAULT.split(",")) == 13


def test_transcription_segment_helpers() -> None:
    from models.stt_runner import TranscriptionSegment

    segment = TranscriptionSegment(start=0.5, end=1.75, text="hello")

    assert segment.duration_s() == 1.25
    assert segment.to_dict() == {"start": 0.5, "end": 1.75, "text": "hello"}


def test_decode_pcm_samples_rejects_unsupported_width() -> None:
    from models.stt_runner import _decode_pcm_samples

    with pytest.raises(ValueError, match="24-bit"):
        _decode_pcm_samples(b"\x00\x01\x02", 3)


def test_read_wav_samples_downmixes_stereo_audio(tmp_path: Path) -> None:
    import struct
    import wave

    from models.stt_runner import _read_wav_samples

    wav_path = tmp_path / "stereo.wav"
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(2)
        wav_file.setsampwidth(2)
        wav_file.setframerate(8000)
        wav_file.writeframes(struct.pack("<4h", 16384, 16384, -16384, -16384))

    samples, sample_rate, duration = _read_wav_samples(wav_path)

    assert samples == pytest.approx([0.5, -0.5])
    assert sample_rate == 8000
    assert duration == pytest.approx(2 / 8000)


def test_build_segments_uses_segment_timestamps() -> None:
    from models.stt_runner import _build_segments

    result = SimpleNamespace(
        segment_texts=["  first  ", "second"],
        segment_timestamps=[1.0, 3.5],
        timestamps=[],
    )

    segments = _build_segments(result, "ignored", audio_duration=2.5)

    assert len(segments) == 2
    assert segments[0].start == 0.0
    assert segments[0].end == 1.0
    assert segments[0].text == "first"
    assert segments[1].start == 1.0
    assert segments[1].end == 2.5
    assert segments[1].text == "second"
