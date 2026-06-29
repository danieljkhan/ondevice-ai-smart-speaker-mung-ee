"""Tests for aggregate-only corpus vocabulary scanning."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.scan_corpus_vocabulary_baseline import (
    count_vocabulary_entries,
    iter_conversation_rows,
    main,
    scan_corpus,
)


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_count_vocabulary_entries_uses_distinct_exact_tokens() -> None:
    text = "뭉이야 뭉이야 한글이랑 송편."

    assert count_vocabulary_entries(text) == 2


def test_scan_corpus_distribution_and_recommendation(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "conversation_a.jsonl",
        [
            {"user_text": "한글 공부했어"},
            {"user_text": "뭉이야 한글 추석 송편 단군신화"},
        ],
    )
    _write_jsonl(
        tmp_path / "conversation_b.jsonl",
        [{"raw_stt_text": "뭉이야 한글 추석 송편 단군신화 일제강점기 빙하"}],
    )

    summary = scan_corpus(tmp_path)

    assert summary["files_scanned"] == 2
    assert summary["utterances_scanned"] == 3
    assert summary["max_vocabulary_entries_in_single_utterance"] == 7
    assert summary["distribution"]["1"] == 1
    assert summary["distribution"]["5"] == 1
    assert summary["distribution"]["7"] == 1
    assert summary["count_threshold_validation"] == {
        "current_threshold_6_violations": 1,
        "recommendation": "fraction-only mode recommended",
    }


def test_scan_corpus_recommends_raise_to_7_for_six_only(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "conversation.jsonl",
        [
            {"user_text": "뭉이야 한글 추석 송편 단군신화 일제강점기"},
            {},
        ],
    )

    summary = scan_corpus(tmp_path)

    assert summary["distribution"]["0"] == 1
    assert summary["distribution"]["6"] == 1
    assert summary["count_threshold_validation"] == {
        "current_threshold_6_violations": 1,
        "recommendation": "raise to 7",
    }


def test_scan_corpus_output_is_aggregate_only(tmp_path: Path) -> None:
    sensitive_text = "비밀 이야기는 한글 하나만 있어"
    _write_jsonl(tmp_path / "conversation.jsonl", [{"user_text": sensitive_text}])

    encoded = json.dumps(scan_corpus(tmp_path), ensure_ascii=False)

    assert sensitive_text not in encoded
    assert "비밀" not in encoded


def test_scan_corpus_cli_writes_aggregate_json(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    output_path = tmp_path / "summary.json"
    _write_jsonl(corpus_dir / "conversation.jsonl", [{"user_text": "한글 공부했어"}])

    assert main(["--corpus-dir", corpus_dir.as_posix(), "--output", output_path.as_posix()]) == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["utterances_scanned"] == 1
    assert "한글 공부했어" not in output_path.read_text(encoding="utf-8")


def test_scan_corpus_cli_writes_stdout_when_output_omitted(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_jsonl(tmp_path / "conversation.jsonl", [{"user_text": "한글 공부했어"}])

    assert main(["--corpus-dir", tmp_path.as_posix()]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["utterances_scanned"] == 1
    assert "한글 공부했어" not in json.dumps(payload, ensure_ascii=False)


def test_scan_corpus_iter_rows_skips_blank_and_rejects_non_object(tmp_path: Path) -> None:
    path = tmp_path / "conversation.jsonl"
    path.write_text("\n[]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Expected JSON object"):
        list(iter_conversation_rows(tmp_path))
