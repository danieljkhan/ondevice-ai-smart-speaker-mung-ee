"""Tests for the post-hoc F31-3 wakeword classifier."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.post_hoc_f31_classifier import (
    f31_3_classify,
    iter_round_rows,
    main,
    summarize_rows,
)


def test_f31_3_classify_full_collapse_without_query_wakeword() -> None:
    row = {"raw_stt_text": "뭉이야 뭉이야 뭉이야 뭉이야 뭉이야", "query": "안녕"}

    assert f31_3_classify(row) == ("full_collapse", 5)


def test_f31_3_classify_partial_injection_without_query_wakeword() -> None:
    row = {"raw_stt_text": "뭉이야 뭉이 안녕", "query": "안녕"}

    assert f31_3_classify(row) == ("partial_injection", 2)


def test_f31_3_classify_clean_when_query_has_wakeword() -> None:
    row = {
        "raw_stt_text": "뭉이야 뭉이야 뭉이야 뭉이야 뭉이야",
        "query": "뭉이야 안녕",
    }

    assert f31_3_classify(row) == ("clean", 5)


def test_f31_3_classify_clean_when_text_fields_are_missing() -> None:
    assert f31_3_classify({"query": ""}) == ("clean", 0)


def test_f31_3_summarize_rows_counts_verdicts() -> None:
    rows = [
        {"id": 1, "raw_stt_text": "뭉이야 뭉이야 뭉이야 뭉이야 뭉이야", "query": ""},
        {"id": 2, "raw_stt_text": "뭉이야 뭉이", "query": ""},
        {"id": 3, "raw_stt_text": "안녕", "query": ""},
    ]

    summary = summarize_rows(rows)

    assert summary["rows_scanned"] == 3
    assert summary["counts"] == {"full_collapse": 1, "partial_injection": 1, "clean": 1}
    assert summary["rows"][0] == {
        "index": 1,
        "id": 1,
        "verdict": "full_collapse",
        "wakeword_token_count": 5,
    }


def test_f31_3_cli_writes_summary_json(tmp_path: Path) -> None:
    input_path = tmp_path / "rounds.jsonl"
    output_path = tmp_path / "summary.json"
    input_path.write_text(
        json.dumps({"id": 10, "raw_stt_text": "뭉이야 뭉이", "query": ""}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    assert main(["--input", input_path.as_posix(), "--output", output_path.as_posix()]) == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["counts"]["partial_injection"] == 1


def test_f31_3_cli_writes_stdout_when_output_omitted(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_path = tmp_path / "rounds.jsonl"
    input_path.write_text(
        "\n" + json.dumps({"raw_stt_text": "안녕", "query": ""}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    assert main(["--input", input_path.as_posix()]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["counts"]["clean"] == 1


def test_f31_3_iter_round_rows_rejects_non_object(tmp_path: Path) -> None:
    input_path = tmp_path / "rounds.jsonl"
    input_path.write_text("[]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Expected JSON object"):
        list(iter_round_rows(input_path))
