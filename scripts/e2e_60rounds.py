"""Automated E2E conversation test with 60 scripted rounds.

Runs on Jetson Orin Nano Super. Outputs JSON results per round to stdout.
Each round uses one topic, and the turn count grows by round bucket:
1-10 => 3 turns, 11-20 => 4, 21-30 => 5, 31-40 => 6, 41-50 => 7, 51+ => 8.

Usage:
    cd /opt/mungi-repo
    source .venv/bin/activate
    python scripts/e2e_60rounds.py [--rounds 60] [--start 1] [--topic-pool PATH]
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Topic pool
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOPIC_POOL_PATH = REPO_ROOT / "assets" / "training" / "e2e_topic_pool_v1.json"
TopicData = dict[str, str | list[str]]


def load_topic_pool(path: Path | None = None) -> list[TopicData]:
    """Load a topic pool JSON file."""
    topic_pool_path = path or DEFAULT_TOPIC_POOL_PATH
    data = json.loads(topic_pool_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        msg = f"Topic pool JSON must contain a list: {topic_pool_path}"
        raise ValueError(msg)
    return data


TOPIC_POOL: list[TopicData] = load_topic_pool()


def strip_think(text: str) -> str:
    """Remove Qwen3 think tags."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    text = text.replace("</think>", "")
    text = re.sub(r"^\s*think\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def sanitize_response(text: str) -> str:
    """Remove foreign chars and English words."""
    allowed = re.compile(
        r"[^\uAC00-\uD7A3\u3131-\u3163\u1100-\u11FF"
        r"a-zA-Z0-9"
        r"\s.,!?~\-\u2026:;'\"()]"
    )
    cleaned = allowed.sub("", text)
    cleaned = re.sub(r"[a-zA-Z]{2,}", "", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    cleaned = re.sub(r"^\s*[.,]\s*", "", cleaned)
    return cleaned if cleaned else "안녕, 무슨 이야기 할까?"


SYSTEM_PROMPT = (
    "너는 '뭉이', 어린이의 AI 친구야.\n"
    "말투: 존댓말은 쓰지 말고, 친절하고 따뜻한 반말만 써.\n"
    "모든 문장은 아이에게 친구처럼 다정한 반말로 말해.\n"
    "한국어만 사용. 영어 단어도 쓰지 마.\n"
    "2~3문장으로만 대답해. 매번 다른 표현을 써.\n"
    "아이가 슬프면: '그랬구나, 많이 속상했겠다' 처럼 공감 먼저.\n"
    "아이가 기쁘면: '정말 멋지다!', '대단하다!' 처럼 함께 기뻐해.\n"
    "모르는 건: '잘 모르겠는데 같이 알아볼까?' 라고 해.\n"
    "이모지 금지. 지어낸 사실 금지.\n"
)

STOP_TOKENS = ["<|im_end|>", "<|im_start|>"]
DEFAULT_ROUNDS = 60
TURN_COUNT_SCHEDULE: tuple[tuple[int, int, int], ...] = (
    (1, 10, 3),
    (11, 20, 4),
    (21, 30, 5),
    (31, 40, 6),
    (41, 50, 7),
    (51, 60, 8),
)
EXTRA_MESSAGE_TEMPLATES: tuple[str, ...] = (
    "뭉이야 {topic} 이야기 더 해줘!",
    "{topic}에서 제일 신기한 건 뭐야?",
    "{topic}랑 같이 놀면 뭐가 제일 재밌을까?",
    "{topic} 생각하면 어떤 기분이 들어?",
    "{topic}에 대해 하나만 더 알려줘!",
)
EXTRA_MESSAGE_TEMPLATES_EN: tuple[str, ...] = (
    "Mung-i, tell me more about {topic}!",
    "What's the most amazing thing about {topic}?",
    "What would be the most fun thing to do with {topic}?",
    "How does {topic} make you feel?",
    "Tell me one more thing about {topic}!",
)


def turn_count_for_round(round_num: int) -> int:
    """Return the scripted turn count for a given round number."""
    if round_num < 1:
        msg = "Round numbers must be positive integers."
        raise ValueError(msg)

    for start_round, end_round, turn_count in TURN_COUNT_SCHEDULE:
        if start_round <= round_num <= end_round:
            return turn_count
    return TURN_COUNT_SCHEDULE[-1][2]


def build_round_messages(
    topic_data: TopicData,
    round_num: int,
    language: str = "ko",
) -> list[str]:
    """Build the user-message script for one round."""
    topic_name = str(topic_data["topic"])
    seed_messages = [str(message) for message in topic_data["messages"]]
    planned_turns = turn_count_for_round(round_num)
    if planned_turns <= len(seed_messages):
        return seed_messages[:planned_turns]

    messages = list(seed_messages)
    extra_needed = planned_turns - len(messages)
    extra_templates = EXTRA_MESSAGE_TEMPLATES_EN if language == "en" else EXTRA_MESSAGE_TEMPLATES
    fallback_template = (
        "Tell me one more thing about {topic}!"
        if language == "en"
        else "{topic}에 대해 하나만 더 말해줘!"
    )
    for index in range(extra_needed):
        template = extra_templates[index % len(extra_templates)]
        candidate = template.format(topic=topic_name)
        if candidate in messages:
            candidate = fallback_template.format(topic=topic_name)
        messages.append(candidate)
    return messages


def choose_round_topic(
    pool: list[TopicData],
    *,
    cursor: int,
    rng: random.Random,
) -> tuple[TopicData, int]:
    """Pick the next topic, reshuffling only when the pool wraps."""
    if not pool:
        msg = "Topic pool must not be empty."
        raise ValueError(msg)

    if cursor == 0 or cursor >= len(pool):
        rng.shuffle(pool)
        cursor = 0
    return pool[cursor], cursor + 1


def run_round(
    llm: Any,
    round_num: int,
    topic_data: TopicData,
    *,
    presence_penalty: float = 1.2,
    repeat_penalty: float = 1.5,
) -> dict[str, Any]:
    """Run one E2E test round for a single topic."""
    total_tokens: int = 0
    total_time: float = 0.0
    topic_name = str(topic_data["topic"])
    messages = build_round_messages(topic_data, round_num)
    turn_list: list[dict[str, Any]] = []
    topic_result: dict[str, Any] = {
        "topic": topic_name,
        "turns": turn_list,
    }

    chat_messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    for i, user_msg in enumerate(messages):
        chat_messages.append({"role": "user", "content": user_msg})

        t0 = time.monotonic()
        stream = llm.create_chat_completion(
            messages=chat_messages,
            max_tokens=128,
            stop=STOP_TOKENS,
            stream=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            min_p=0.0,
            presence_penalty=presence_penalty,
            repeat_penalty=repeat_penalty,
        )

        ttft = 0.0
        toks = 0
        parts: list[str] = []
        for chunk in stream:
            delta = chunk["choices"][0].get("delta", {})
            token_text = delta.get("content", "")
            if not token_text:
                continue
            if toks == 0:
                ttft = time.monotonic() - t0
            parts.append(token_text)
            toks += 1
        gen_time = time.monotonic() - t0

        raw_response = "".join(parts)
        cleaned = strip_think(raw_response)
        cleaned = sanitize_response(cleaned)
        chat_messages.append({"role": "assistant", "content": cleaned})

        tps = toks / gen_time if gen_time > 0 else 0
        turn_data = {
            "exchange": i + 1,
            "user": user_msg,
            "assistant": cleaned,
            "tokens": toks,
            "ttft_s": round(ttft, 2),
            "tok_s": round(tps, 1),
            "time_s": round(gen_time, 1),
        }
        turn_list.append(turn_data)
        total_tokens += toks
        total_time += gen_time

    return {
        "round": round_num,
        "planned_turns": len(messages),
        "topics": [topic_result],
        "total_tokens": total_tokens,
        "total_time": total_time,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="E2E 60-round automated test")
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--start", type=int, default=1, help="Starting round number")
    parser.add_argument(
        "--topic-pool",
        type=Path,
        default=None,
        help="Path to topic pool JSON file. Default: V1 built-in pool",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/opt/mungi/ai_models/Qwen3-4B-Q4_K_M.gguf",
        help="Path to the GGUF model file.",
    )
    parser.add_argument(
        "--presence-penalty",
        type=float,
        default=1.2,
        help="Presence penalty for generation (Qwen3.5 recommends 1.5).",
    )
    parser.add_argument(
        "--repeat-penalty",
        type=float,
        default=1.5,
        help="Repeat penalty for generation.",
    )
    args = parser.parse_args()

    from llama_cpp import Llama

    model_path = args.model_path
    sys.stderr.write(f"Loading model: {model_path}\n")
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=-1,
        n_ctx=2048,
        flash_attn=True,
        verbose=False,
    )
    sys.stderr.write("Model loaded. Warming up...\n")
    llm.create_chat_completion(
        messages=[{"role": "user", "content": "warmup"}],
        max_tokens=1,
        chat_template_kwargs={"enable_thinking": False},
    )
    sys.stderr.write("Warmup done.\n")

    pool = load_topic_pool(args.topic_pool) if args.topic_pool else list(TOPIC_POOL)
    rng = random.Random(42)
    topic_cursor = 0

    for round_num in range(args.start, args.start + args.rounds):
        selected_topic, topic_cursor = choose_round_topic(pool, cursor=topic_cursor, rng=rng)
        planned_turns = turn_count_for_round(round_num)

        sys.stderr.write(f"\n=== Round {round_num}/{args.start + args.rounds - 1} ===\n")
        sys.stderr.write(
            f"Topic: {selected_topic['topic']} ({planned_turns} turns)\n",
        )

        result = run_round(
            llm,
            round_num,
            selected_topic,
            presence_penalty=args.presence_penalty,
            repeat_penalty=args.repeat_penalty,
        )

        # Output JSON result for this round (one line)
        print(json.dumps(result, ensure_ascii=False), flush=True)

        sys.stderr.write(
            f"Round {round_num} done: "
            f"{result['total_tokens']} tokens, "
            f"{result['total_time']:.1f}s\n"
        )

    del llm
    sys.stderr.write("\nAll rounds complete.\n")


if __name__ == "__main__":
    main()
