"""Tests for deterministic explicit-recall query routing."""

from __future__ import annotations

import time

import pytest

from safety.recall_query_router import RecallQueryMatch, match_recall_query


@pytest.mark.parametrize(
    "text",
    [
        "내 이름 뭐라고 했어?",
        "내 이름 뭐라고 했지",
        "내 이름이 뭐라고 했어",
        "내 이름 기억나?",
        "내 이름이 기억나",
        "나 누구야?",
        "나 누구라고 했어",
        "뭉이야 내 이름 뭐라고 했어?",
        "음 내 이름 기억나?",
    ],
)
def test_recall_query_matches_name(text: str) -> None:
    """Name-recall questions resolve to the name sub_kind."""
    match = match_recall_query(text)

    assert match is not None
    assert isinstance(match, RecallQueryMatch)
    assert match.kind == "recall"
    assert match.sub_kind == "name"
    assert match.matched_patterns


@pytest.mark.parametrize(
    "text",
    [
        "내가 뭐라고 했지?",
        "내가 뭐라고 했어",
        "내가 뭐라고 했는지 기억나?",
        "내가 무슨 말 했는지 기억나?",
        "내가 뭐라고 말했는지 기억나?",
        "지난번 내가 뭐라고 했어",
        "어제 내가 뭐라고 했지",
        "내 말 기억나?",
        "내 말 기억해",
        "아까 내가 한 말 기억나?",
        "아까 내가 한 말 기억해",
        # In-session short-term recall ("what did I JUST say?").
        "방금 뭐라고 했어?",
        "막 뭐라고 했어?",
        "방금 내가 뭐라고 했어?",
        "방금 내가 한 말 뭐야?",
        "방금 뭐라고 말했어?",
    ],
)
def test_recall_query_matches_general(text: str) -> None:
    """Generic recall questions resolve to the general_recall sub_kind."""
    match = match_recall_query(text)

    assert match is not None
    assert match.sub_kind == "general_recall"
    assert match.matched_patterns


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "안녕",
        "뭉이야 안녕",
        "오늘 날씨 어때",
        "블록 놀이 하자",
        # Third-party recall must not be claimed as the child's own statement.
        "엄마가 뭐라고 했어?",
        "아빠가 뭐 좋아한다고 했어",
        "선생님이 뭐라고 했지",
        "친구가 뭐라고 했어",
        "뭉이가 뭐라고 했어",
        "엄마가 방금 뭐라고 했어?",
        # General-knowledge idioms about memory, not recall requests.
        "기억력이 뭐야",
        "기억 상실이 뭐야",
        "기억해줘서 고마워",
        # Store-request imperative, not a recall question.
        "이거 기억해 둬",
        "내 이름 기억해 줘",
        "이거 기억해 놔",
    ],
)
def test_recall_query_rejects_false_positives(text: str) -> None:
    """Non-recall turns and guard cases remain normal conversation."""
    assert match_recall_query(text) is None


@pytest.mark.parametrize(
    "text",
    [
        "내 이름 뭐라고 했어? 그리고 오늘 뭐 할까 또 영어 공부도 하자",
        "내가 제일 좋아하는 과일은 뭐야? 사실 나는 어제 동물원에 갔다 왔는데",
    ],
)
def test_recall_query_requires_whole_turn_anchor(text: str) -> None:
    """A recall phrase buried in a longer turn is not a whole-turn command."""
    assert match_recall_query(text) is None


@pytest.mark.parametrize(
    "text",
    [
        "내 이름 뭐라고 했어" + " " * 20000 + "그리고",
        "내가 제일 좋아하는 과일은 " + "," * 20000 + " 뭐야",
    ],
)
def test_recall_query_is_linear_on_adversarial_padding(text: str) -> None:
    """The recall matcher must not exhibit super-linear backtracking (ReDoS)."""
    start = time.perf_counter()
    result = match_recall_query(text)
    elapsed = time.perf_counter() - start
    assert result is None
    assert elapsed < 0.5
