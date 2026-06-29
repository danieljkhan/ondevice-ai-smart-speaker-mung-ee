from __future__ import annotations

from pathlib import Path

import pytest

from scripts.check_no_mojibake import main


def test_clean_utf8_korean_passes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "clean.py"
    path.write_text('text = "\ubb49\uc774"\n', encoding="utf-8")

    assert main([str(path)]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""


def test_mojibake_glyph_detected_fails(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "broken.py"
    path.write_text('text = "\u8438\ub431\uc528"\n', encoding="utf-8")

    assert main([str(path)]) == 1
    captured = capsys.readouterr()
    assert "mojibake" in captured.err
    assert str(path) in captured.err


def test_invalid_utf8_bytes_fails(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "invalid.py"
    path.write_bytes(b"\xff\xfe valid prefix")

    assert main([str(path)]) == 1
    captured = capsys.readouterr()
    assert "invalid UTF-8" in captured.err


def test_binary_file_skipped(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "model.gguf"
    path.write_bytes(b"\xff\xfe\x00binary")

    assert main([str(path)]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""


def test_large_file_skipped(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "large.txt"
    path.write_bytes(b"a" * (5 * 1024 * 1024 + 1))

    assert main([str(path)]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""


def test_multiple_files_one_failing_returns_1(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    clean_path = tmp_path / "clean.py"
    clean_path.write_text('text = "\ubb49\uc774"\n', encoding="utf-8")
    broken_path = tmp_path / "broken.py"
    broken_path.write_text('text = "\u8438\ub431\uc528"\n', encoding="utf-8")

    assert main([str(clean_path), str(broken_path)]) == 1
    captured = capsys.readouterr()
    assert str(broken_path) in captured.err
    assert str(clean_path) not in captured.err


def test_pattern_override_via_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    patterns_file = tmp_path / "patterns.txt"
    patterns_file.write_text("\u51cd\n", encoding="utf-8")
    source_path = tmp_path / "custom.py"
    source_path.write_text('text = "\u51cd"\n', encoding="utf-8")

    assert main(["--patterns-file", str(patterns_file), str(source_path)]) == 1
    captured = capsys.readouterr()
    assert "\u51cd" in captured.err
