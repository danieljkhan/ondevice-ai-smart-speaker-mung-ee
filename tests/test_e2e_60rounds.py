"""Tests for the scripted text-only E2E runner."""

from __future__ import annotations

from typing import Any


def test_round_turn_schedule_defaults_to_60_rounds() -> None:
    """The default scripted schedule should match the requested 60-round buckets."""
    from scripts.e2e_60rounds import DEFAULT_ROUNDS, turn_count_for_round

    assert DEFAULT_ROUNDS == 60
    assert turn_count_for_round(1) == 3
    assert turn_count_for_round(10) == 3
    assert turn_count_for_round(11) == 4
    assert turn_count_for_round(20) == 4
    assert turn_count_for_round(21) == 5
    assert turn_count_for_round(30) == 5
    assert turn_count_for_round(31) == 6
    assert turn_count_for_round(40) == 6
    assert turn_count_for_round(41) == 7
    assert turn_count_for_round(50) == 7
    assert turn_count_for_round(51) == 8
    assert turn_count_for_round(60) == 8
    assert turn_count_for_round(61) == 8


def test_build_round_messages_extends_seed_messages() -> None:
    """Rounds that need more than three turns should receive generated follow-ups."""
    from scripts.e2e_60rounds import build_round_messages

    messages = build_round_messages(
        {"topic": "\uacf5\ub8e1", "messages": ["\ud558\ub098", "\ub458", "\uc14b"]},
        55,
    )

    assert messages[:3] == ["\ud558\ub098", "\ub458", "\uc14b"]
    assert len(messages) == 8
    assert len(set(messages)) == 8
    assert messages[3].startswith("\ubb49\uc774\uc57c \uacf5\ub8e1")


def test_run_round_uses_round_specific_turn_count() -> None:
    """run_round() should emit the scheduled number of turns for one topic."""
    from scripts.e2e_60rounds import run_round

    class FakeLLM:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def __call__(self, _prompt: str, **_kwargs: object) -> list[dict[str, object]]:
            return [{"choices": [{"text": "\uadf8\ub798, \uac19\uc774 \ub180\uc790"}]}]

        def create_chat_completion(
            self,
            messages: list[dict[str, str]],
            stream: bool = False,
            **kwargs: object,
        ) -> Any:
            self.calls.append({"messages": messages, "stream": stream, **kwargs})
            if stream:
                yield {"choices": [{"delta": {"role": "assistant"}}]}
                yield {
                    "choices": [{"delta": {"content": "\uadf8\ub798, \uac19\uc774 \ub180\uc790"}}]
                }
                yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}
                return
            return {
                "choices": [{"message": {"content": "\uadf8\ub798, \uac19\uc774 \ub180\uc790"}}]
            }

    fake_llm = FakeLLM()
    result = run_round(
        fake_llm,
        21,
        {"topic": "\ub85c\ubd07", "messages": ["\ud558\ub098", "\ub458", "\uc14b"]},
        repeat_penalty=1.9,
    )

    assert result["planned_turns"] == 5
    assert result["total_tokens"] == 5
    assert len(result["topics"]) == 1
    turns = result["topics"][0]["turns"]
    assert len(turns) == 5
    assert turns[0]["assistant"] == "\uadf8\ub798, \uac19\uc774 \ub180\uc790"
    assert turns[-1]["user"].startswith("\ub85c\ubd07")
    assert fake_llm.calls[0]["repeat_penalty"] == 1.9
