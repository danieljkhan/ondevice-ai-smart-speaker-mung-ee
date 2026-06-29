"""Tests for the E2E conversation dump generator."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts import generate_e2e_conversation_dump


def _write_rounds(path: Path, rounds: list[dict[str, object]]) -> None:
    """Write synthetic rounds data as JSONL."""
    payload = "\n".join(json.dumps(item, ensure_ascii=False) for item in rounds) + "\n"
    path.write_text(payload, encoding="utf-8")


def test_dump_writes_expected_header(tmp_path: Path) -> None:
    """The dump should include the document header and totals metadata."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    rounds_path = input_dir / "rounds.jsonl"
    _write_rounds(
        rounds_path,
        [
            {
                "round": 1,
                "topics": [
                    {
                        "topic": "balloon",
                        "turns": [
                            {
                                "round_num": 1,
                                "topic": "balloon",
                                "exchange": 1,
                                "user_text": "hi",
                                "assistant_text": "hello",
                                "llm_tokens": 10,
                                "total_time_s": 1.234,
                                "success": True,
                                "language": "en",
                            },
                        ],
                    },
                ],
            },
        ],
    )
    output = generate_e2e_conversation_dump.generate_markdown(
        generate_e2e_conversation_dump._read_rounds(rounds_path),
        rounds_path,
        "test-label",
    )

    assert "# Conversation Script - test-label" in output
    assert f"> Source: `{rounds_path}`" in output
    assert "> Generated: " in output
    assert "> Total rounds: 1" in output
    assert "> Total turns: 1" in output


def test_dump_groups_turns_by_round_and_surfaces_language(tmp_path: Path) -> None:
    """The dump should render round headings, language, and exchanges."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    rounds_path = input_dir / "rounds.jsonl"
    _write_rounds(
        rounds_path,
        [
            {
                "round": 2,
                "topics": [
                    {
                        "topic": "바다",
                        "turns": [
                            {
                                "round_num": 2,
                                "topic": "바다",
                                "exchange": 1,
                                "user_text": "왜 짜?",
                                "assistant_text": "소금이 있어서 그래.",
                                "llm_tokens": 12,
                                "total_time_s": 2.5,
                                "success": True,
                                "language": "ko",
                            },
                            {
                                "round_num": 2,
                                "topic": "바다",
                                "exchange": 2,
                                "user_text": "신기해",
                                "assistant_text": "정말 신기하지!",
                                "llm_tokens": 8,
                                "total_time_s": 1.75,
                                "success": True,
                                "language": "ko",
                            },
                        ],
                    },
                ],
            },
        ],
    )

    output = generate_e2e_conversation_dump.generate_markdown(
        generate_e2e_conversation_dump._read_rounds(rounds_path),
        rounds_path,
        "grouping",
    )

    assert '## Round 2 - "바다" (ko)' in output
    assert "**Turn 1** [ko, llm_tokens=12, total=2.500s]" in output
    assert "- User: 왜 짜?" in output
    assert "- Mung-i: 소금이 있어서 그래." in output
    assert "**Turn 2** [ko, llm_tokens=8, total=1.750s]" in output


def test_dump_marks_failed_turns(tmp_path: Path) -> None:
    """Failed turns should be surfaced inline with the error message."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    rounds_path = input_dir / "rounds.jsonl"
    _write_rounds(
        rounds_path,
        [
            {
                "round": 3,
                "topics": [
                    {
                        "topic": "lunch",
                        "turns": [
                            {
                                "round_num": 3,
                                "topic": "lunch",
                                "exchange": 1,
                                "user_text": "",
                                "assistant_text": "",
                                "llm_tokens": 5,
                                "total_time_s": 0.5,
                                "success": False,
                                "error": "Unsupported character",
                                "language": "en",
                            },
                        ],
                    },
                ],
            },
        ],
    )

    output = generate_e2e_conversation_dump.generate_markdown(
        generate_e2e_conversation_dump._read_rounds(rounds_path),
        rounds_path,
        "failed",
    )

    assert "**FAILED:** Unsupported character" in output


def test_dump_handles_missing_input_gracefully(tmp_path: Path) -> None:
    """The CLI should return exit code 1 when rounds.jsonl is missing."""
    output_path = tmp_path / "out.md"
    original_argv = sys.argv
    sys.argv = [
        "generate_e2e_conversation_dump.py",
        "--input-dir",
        str(tmp_path / "missing"),
        "--output",
        str(output_path),
    ]
    try:
        result = generate_e2e_conversation_dump.main()
    finally:
        sys.argv = original_argv

    assert result == 1
    assert not output_path.exists()


def test_dump_counts_ko_and_en_turns(tmp_path: Path) -> None:
    """Header counts should reflect Korean and English turns."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    rounds_path = input_dir / "rounds.jsonl"
    _write_rounds(
        rounds_path,
        [
            {
                "round": 1,
                "topics": [
                    {
                        "topic": "mix",
                        "turns": [
                            {
                                "round_num": 1,
                                "topic": "mix",
                                "exchange": 1,
                                "user_text": "안녕",
                                "assistant_text": "안녕",
                                "llm_tokens": 4,
                                "total_time_s": 1.0,
                                "success": True,
                                "language": "ko",
                            },
                            {
                                "round_num": 1,
                                "topic": "mix",
                                "exchange": 2,
                                "user_text": "hello",
                                "assistant_text": "hi",
                                "llm_tokens": 4,
                                "total_time_s": 1.0,
                                "success": True,
                                "language": "en",
                            },
                        ],
                    },
                ],
            },
        ],
    )

    output = generate_e2e_conversation_dump.generate_markdown(
        generate_e2e_conversation_dump._read_rounds(rounds_path),
        rounds_path,
        "counts",
    )

    assert "> KO turns: 1, EN turns: 1" in output
