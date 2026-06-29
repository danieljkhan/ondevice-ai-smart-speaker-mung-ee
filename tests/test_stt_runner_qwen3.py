"""Unit tests for Qwen3-ASR support in models.stt_runner."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def _create_qwen3_bundle(
    root_dir: Path,
    *,
    include_decoder_int8: bool = True,
    include_encoder_fp32: bool = False,
    include_decoder_fp32: bool = False,
    missing_tokenizer_files: tuple[str, ...] = (),
) -> Path:
    bundle_dir = root_dir / "sherpa-onnx-qwen3-asr-test"
    tokenizer_dir = bundle_dir / "tokenizer"

    _touch(bundle_dir / "conv_frontend.onnx")
    _touch(bundle_dir / "encoder.int8.onnx")
    if include_encoder_fp32:
        _touch(bundle_dir / "encoder.onnx")
    if include_decoder_int8:
        _touch(bundle_dir / "decoder.int8.onnx")
    if include_decoder_fp32:
        _touch(bundle_dir / "decoder.onnx")

    for filename in ("vocab.json", "merges.txt", "tokenizer_config.json"):
        if filename not in missing_tokenizer_files:
            _touch(tokenizer_dir / filename)

    return bundle_dir


def _install_fake_sherpa(
    monkeypatch: pytest.MonkeyPatch,
    from_qwen3_asr: MagicMock,
) -> None:
    fake_sherpa_onnx = SimpleNamespace(
        OfflineRecognizer=SimpleNamespace(from_qwen3_asr=from_qwen3_asr)
    )
    monkeypatch.setitem(sys.modules, "sherpa_onnx", fake_sherpa_onnx)


def test_load_qwen3_asr_invokes_from_qwen3_asr_with_correct_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from models import stt_runner

    bundle_dir = _create_qwen3_bundle(tmp_path)
    from_qwen3_asr = MagicMock(return_value=MagicMock())

    _install_fake_sherpa(monkeypatch, from_qwen3_asr)
    monkeypatch.setattr(stt_runner, "_supported_providers", lambda: set())

    model = stt_runner.load_stt_model(model_size="qwen3-asr-0.6b", model_dir=str(tmp_path))

    from_qwen3_asr.assert_called_once_with(
        conv_frontend=str(bundle_dir / "conv_frontend.onnx"),
        encoder=str(bundle_dir / "encoder.int8.onnx"),
        decoder=str(bundle_dir / "decoder.int8.onnx"),
        tokenizer=str(bundle_dir / "tokenizer"),
        provider="cpu",
        num_threads=1,
        sample_rate=16000,
        feature_dim=128,
        decoding_method="greedy_search",
        max_total_len=512,
        max_new_tokens=128,
        hotwords="",
    )
    assert model.resolved_model_size == stt_runner._QWEN3_ASR_NAME
    assert model.model_path == str(bundle_dir)
    assert model.provider == "cpu"


def test_load_qwen3_asr_missing_decoder_raises_filenotfound(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from models import stt_runner

    _create_qwen3_bundle(tmp_path, include_decoder_int8=False)
    from_qwen3_asr = MagicMock()

    _install_fake_sherpa(monkeypatch, from_qwen3_asr)
    monkeypatch.setattr(stt_runner, "_supported_providers", lambda: set())

    with pytest.raises(FileNotFoundError):
        stt_runner.load_stt_model(model_size="qwen3-asr-0.6b", model_dir=str(tmp_path))


def test_load_qwen3_asr_missing_tokenizer_file_raises_filenotfound(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from models import stt_runner

    _create_qwen3_bundle(tmp_path, missing_tokenizer_files=("vocab.json",))
    from_qwen3_asr = MagicMock()

    _install_fake_sherpa(monkeypatch, from_qwen3_asr)
    monkeypatch.setattr(stt_runner, "_supported_providers", lambda: set())

    with pytest.raises(FileNotFoundError, match="vocab\\.json"):
        stt_runner.load_stt_model(model_size="qwen3-asr-0.6b", model_dir=str(tmp_path))


def test_load_qwen3_asr_prefers_int8_when_both_present(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from models import stt_runner

    bundle_dir = _create_qwen3_bundle(
        tmp_path,
        include_encoder_fp32=True,
        include_decoder_fp32=True,
    )
    from_qwen3_asr = MagicMock(return_value=MagicMock())

    _install_fake_sherpa(monkeypatch, from_qwen3_asr)
    monkeypatch.setattr(stt_runner, "_supported_providers", lambda: set())

    stt_runner.load_stt_model(model_size="qwen3-asr", model_dir=str(tmp_path))

    call_kwargs = from_qwen3_asr.call_args.kwargs
    assert call_kwargs["encoder"] == str(bundle_dir / "encoder.int8.onnx")
    assert call_kwargs["decoder"] == str(bundle_dir / "decoder.int8.onnx")


def test_load_qwen3_asr_forwards_explicit_hotwords(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from models import stt_runner

    _create_qwen3_bundle(tmp_path)
    from_qwen3_asr = MagicMock(return_value=MagicMock())

    _install_fake_sherpa(monkeypatch, from_qwen3_asr)
    monkeypatch.setattr(stt_runner, "_supported_providers", lambda: set())

    stt_runner.load_stt_model(
        model_size="qwen3-asr-0.6b",
        model_dir=str(tmp_path),
        qwen3_asr_hotwords="custom_word",
    )

    assert from_qwen3_asr.call_args.kwargs["hotwords"] == "custom_word"


def test_load_qwen3_asr_uses_empty_hotwords_when_arg_omitted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from models import stt_runner

    _create_qwen3_bundle(tmp_path)
    from_qwen3_asr = MagicMock(return_value=MagicMock())

    _install_fake_sherpa(monkeypatch, from_qwen3_asr)
    monkeypatch.setattr(stt_runner, "_supported_providers", lambda: set())
    monkeypatch.delenv("MUNGI_QWEN3_ASR_HOTWORDS", raising=False)

    stt_runner.load_stt_model(model_size="qwen3-asr-0.6b", model_dir=str(tmp_path))

    assert from_qwen3_asr.call_args.kwargs["hotwords"] == ""


def test_load_qwen3_asr_env_var_hotwords_used(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from models import stt_runner

    _create_qwen3_bundle(tmp_path)
    from_qwen3_asr = MagicMock(return_value=MagicMock())

    _install_fake_sherpa(monkeypatch, from_qwen3_asr)
    monkeypatch.setattr(stt_runner, "_supported_providers", lambda: set())
    monkeypatch.setenv("MUNGI_QWEN3_ASR_HOTWORDS", "envword")

    stt_runner.load_stt_model(model_size="qwen3-asr-0.6b", model_dir=str(tmp_path))

    assert from_qwen3_asr.call_args.kwargs["hotwords"] == "envword"


def test_load_qwen3_asr_empty_explicit_disables_hotwords(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from models import stt_runner

    _create_qwen3_bundle(tmp_path)
    from_qwen3_asr = MagicMock(return_value=MagicMock())

    _install_fake_sherpa(monkeypatch, from_qwen3_asr)
    monkeypatch.setattr(stt_runner, "_supported_providers", lambda: set())

    stt_runner.load_stt_model(
        model_size="qwen3-asr-0.6b",
        model_dir=str(tmp_path),
        qwen3_asr_hotwords="",
    )

    assert from_qwen3_asr.call_args.kwargs["hotwords"] == ""


def test_run_stt_rejects_audio_over_300s_for_qwen3(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from models import stt_runner

    model = stt_runner.LoadedSttModel(
        recognizer=MagicMock(),
        backend="sherpa-onnx",
        requested_model_size="qwen3-asr-0.6b",
        resolved_model_size=stt_runner._QWEN3_ASR_NAME,
        provider="cpu",
        model_path="bundle-dir",
        language="ko",
    )

    monkeypatch.setattr(stt_runner, "_read_wav_samples", lambda _path: ([], 16000, 301.0))

    with pytest.raises(ValueError, match="300s|5 min"):
        stt_runner.run_stt(model, Path("dummy.wav"))
