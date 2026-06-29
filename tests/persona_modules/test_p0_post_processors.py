from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from core.pipeline import ConversationPipeline
from scripts import persona_cep_p0_guide_tokens as guide_tokens
from scripts import persona_cep_p0_intent_template as intent_template
from scripts import persona_cep_p0_tokenize as prompt_tokenize


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> Path:
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    return path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_templates(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "synthetic_guide": {
                    "keywords_ko": ["synthetic"],
                    "keywords_en": ["synthetic"],
                    "response_ko": "Synthetic guide response.",
                    "response_en": "Move to a safe place and follow an adult.",
                    "mode": "guide",
                    "priority": 10,
                },
                "synthetic_block": {
                    "keywords_ko": ["blocked"],
                    "keywords_en": ["blocked"],
                    "response_ko": "Blocked response.",
                    "response_en": "Blocked response.",
                    "mode": "block",
                    "priority": 1,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def test_tokenize_csv_schema(tmp_path: Path) -> None:
    rounds = _write_jsonl(
        tmp_path / "rounds.jsonl",
        [
            {"turn_id": 1, "language": "en", "backend": "gemma4_text", "prompt": "alpha"},
            {"turn_id": 2, "language": "ko", "backend": "qwen3_legacy", "prompt": "beta"},
            {"turn_id": 3, "language": "en", "backend": "gemma4_text", "prompt": "gamma"},
        ],
    )
    output = tmp_path / "prompt_tokens.csv"

    assert prompt_tokenize.process_rounds(rounds, output) == 0

    rows = _read_csv(output)
    assert rows[-1]["turn_id"] == "summary"
    assert rows[0].keys() == set(prompt_tokenize.CSV_COLUMNS)
    assert rows[0]["prompt_len_chars"] == "5"
    assert rows[0]["tokens_heuristic"].isdigit()
    assert rows[0]["tokens_tokenizer"] == ""


@pytest.mark.parametrize(
    "text",
    ["", "a", "ab", "abc", "abcd", "hello", "what is this", "x" * 29, "x" * 30, "x" * 31],
)
def test_tokenize_heuristic_matches_pipeline(text: str) -> None:
    assert prompt_tokenize.estimate_tokens(text) == ConversationPipeline._estimate_tokens(text)


def test_tokenize_missing_model_path_fallback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rounds = _write_jsonl(
        tmp_path / "rounds.jsonl",
        [{"turn_id": 1, "language": "en", "backend": "gemma4_text", "prompt": "hello"}],
    )
    output = tmp_path / "prompt_tokens.csv"

    assert prompt_tokenize.process_rounds(rounds, output) == 0

    captured = capsys.readouterr()
    rows = _read_csv(output)
    assert "--gemma-model-path not provided" in captured.err
    assert rows[0]["tokens_heuristic"] == "2"
    assert rows[0]["tokens_tokenizer"] == ""
    assert rows[0]["deviation_pct"] == ""


def test_intent_template_keyword_match(tmp_path: Path) -> None:
    rounds = _write_jsonl(
        tmp_path / "rounds.jsonl",
        [
            {"turn_id": 1, "language": "en", "user_text": "hello"},
            {"turn_id": 2, "language": "en", "user_text": "what is lava"},
            {"turn_id": 3, "language": "en", "user_text": "I feel sad"},
            {"turn_id": 4, "language": "en", "user_text": "I wonder about clouds"},
            {"turn_id": 5, "language": "en", "user_text": "can you help me"},
        ],
    )
    output = tmp_path / "intent_labels_template.csv"

    assert intent_template.main(["--rounds-jsonl", str(rounds), "--output-csv", str(output)]) == 0

    rows = _read_csv(output)
    assert rows[0]["is_greeting"] == "True"
    assert rows[1]["is_fact_query"] == "True"
    assert rows[2]["is_emotional"] == "True"
    assert rows[3]["is_curious"] == "True"
    assert rows[4]["is_help_request"] == "True"
    assert all(row["auto_or_manual"] == "auto" for row in rows)


def test_intent_template_safety_topic_match(tmp_path: Path) -> None:
    rounds = _write_jsonl(
        tmp_path / "rounds.jsonl",
        [{"turn_id": 1, "language": "en", "user_text": "what is a volcano"}],
    )
    output = tmp_path / "intent_labels_template.csv"

    assert intent_template.main(["--rounds-jsonl", str(rounds), "--output-csv", str(output)]) == 0

    rows = _read_csv(output)
    assert rows[0]["safety_topic_match"] == "volcano"


def test_intent_template_missing_user_text(tmp_path: Path) -> None:
    rounds = _write_jsonl(tmp_path / "rounds.jsonl", [{"turn_id": 1, "language": "en"}])
    output = tmp_path / "intent_labels_template.csv"

    assert intent_template.main(["--rounds-jsonl", str(rounds), "--output-csv", str(output)]) == 0

    row = _read_csv(output)[0]
    assert row["notes"] == "missing_user_text"
    assert row["is_fact_query"] == intent_template.NULL
    assert row["is_emotional"] == intent_template.NULL
    assert row["is_greeting"] == intent_template.NULL
    assert row["is_curious"] == intent_template.NULL
    assert row["is_help_request"] == intent_template.NULL


def test_guide_tokens_volcano_floor() -> None:
    templates = json.loads(
        Path("assets/filters/approved_templates.json").read_text(encoding="utf-8")
    )
    response = templates["volcano"]["response_ko"]

    sentence_count, mandatory_count, floor_tokens, max_sentence_tokens = guide_tokens._floor_counts(
        response,
        "ko",
        None,
    )

    assert sentence_count >= 3
    assert mandatory_count >= 3
    assert 29 <= floor_tokens <= 48
    assert max_sentence_tokens > 0


def test_guide_tokens_en_earthquake_floor_v4() -> None:
    templates = json.loads(
        Path("assets/filters/approved_templates.json").read_text(encoding="utf-8")
    )
    response = templates["earthquake"]["response_en"]
    sentences = guide_tokens._split_sentences(response)
    action_sentence = next(
        sentence for sentence in sentences if "get under a strong table" in sentence
    )
    mandatory = [sentence for sentence in sentences if guide_tokens._score(sentence, "en") > 0]

    assert action_sentence in mandatory
    assert guide_tokens._score(action_sentence, "en") >= 3
    assert guide_tokens._floor_counts(response, "en", None)[2] >= guide_tokens._estimate_tokens(
        action_sentence
    )


def test_guide_tokens_csv_schema(tmp_path: Path) -> None:
    templates = _write_templates(tmp_path / "approved_templates.json")
    output = tmp_path / "guide_tokens.csv"

    assert guide_tokens.main(["--templates-json", str(templates), "--output-csv", str(output)]) == 0

    rows = _read_csv(output)
    assert rows[0].keys() == set(guide_tokens.CSV_COLUMNS)
    assert {row["mode"] for row in rows} == {"guide", "summary"}
    assert {"summary_mean", "summary_max"}.issubset({row["topic_id"] for row in rows})


def test_all_three_postprocessors_chain(tmp_path: Path) -> None:
    rounds = _write_jsonl(
        tmp_path / "rounds.jsonl",
        [
            {
                "turn_id": 1,
                "language": "en",
                "backend": "gemma4_text",
                "user_text": "hello",
                "prompt": "p1",
            },
            {
                "turn_id": 2,
                "language": "en",
                "backend": "gemma4_text",
                "user_text": "what is a volcano",
                "prompt": "p2",
            },
            {
                "turn_id": 3,
                "language": "en",
                "backend": "gemma4_text",
                "user_text": "I feel sad",
                "prompt": "p3",
            },
            {
                "turn_id": 4,
                "language": "en",
                "backend": "gemma4_text",
                "user_text": "I wonder why",
                "prompt": "p4",
            },
            {
                "turn_id": 5,
                "language": "en",
                "backend": "gemma4_text",
                "user_text": "can you help me",
                "prompt": "p5",
            },
        ],
    )
    templates = _write_templates(tmp_path / "approved_templates.json")
    prompt_csv = tmp_path / "prompt_tokens.csv"
    intent_csv = tmp_path / "intent_labels_template.csv"
    guide_csv = tmp_path / "guide_tokens.csv"

    assert prompt_tokenize.process_rounds(rounds, prompt_csv) == 0
    assert (
        intent_template.main(["--rounds-jsonl", str(rounds), "--output-csv", str(intent_csv)]) == 0
    )
    assert (
        guide_tokens.main(["--templates-json", str(templates), "--output-csv", str(guide_csv)]) == 0
    )

    assert _read_csv(prompt_csv)[-1]["turn_id"] == "summary"
    assert len(_read_csv(intent_csv)) == 5
    assert any(row["topic_id"] == "synthetic_guide" for row in _read_csv(guide_csv))
