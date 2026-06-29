"""Kindergarten-level multi-turn conversation test for Jetson.

Runs 5 turns of child-friendly dialogue with Qwen3-4B to evaluate
response quality, latency, and persona consistency.

Usage:
    cd /opt/mungi-repo
    python scripts/test_conversation.py
"""

from __future__ import annotations

import re
import time


def strip_think(text: str) -> str:
    """Remove Qwen3 think tags from output."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return re.sub(r"<think>.*", "", text, flags=re.DOTALL).strip()


def main() -> None:
    """Run multi-turn kindergarten conversation test."""
    from llama_cpp import Llama

    model = "/opt/mungi/ai_models/Qwen3-4B-Q4_K_M.gguf"
    print("Loading model...")
    llm = Llama(model_path=model, n_gpu_layers=-1, n_ctx=2048, flash_attn=True, verbose=False)
    print("Model loaded. Warming up flash attention...")
    llm("warmup", max_tokens=1, echo=False)
    print("Warmup done.\n")

    system_prompt = (
        "너는 뭉이야. 10살 미만 아이들의 첫 번째 AI 친구야.\n"
        "반드시 지켜야 할 규칙:\n"
        "- 존댓말은 쓰지 말고, 친절하고 따뜻한 반말만 써.\n"
        "- 한국어만 써. 다른 언어 절대 섞지 마.\n"
        "- 짧고 쉬운 단어만 써. 대답은 2~3문장으로 짧게 해.\n"
        "- 아이가 슬프거나 속상하면 먼저 공감해줘.\n"
        "- 위험하거나 무서운 이야기는 하지 마.\n"
        "- 아이를 칭찬하고 격려해줘.\n"
        "/no_think"
    )

    user_messages = [
        "뭉이야 안녕! 나 오늘 유치원에서 그림 그렸어!",
        "공룡 그렸는데 선생님이 잘 그렸다고 했어!",
        "근데 민지가 내 그림 보고 웃었어... 속상해",
        "뭉이야 내일 소풍 가는데 뭐 가져갈까?",
        "뭉이야 하늘은 왜 파란색이야?",
    ]

    # Build initial prompt with Qwen3 chat template
    history = f"<|im_start|>system\n{system_prompt}<|im_end|>"
    stop_tokens = ["<|im_end|>", "<|im_start|>"]
    total_time = 0.0

    for i, user_msg in enumerate(user_messages):
        print("=" * 50)
        print(f"[턴 {i + 1}]")
        print(f"아이: {user_msg}")

        history += f"\n<|im_start|>user\n{user_msg}<|im_end|>"
        history += "\n<|im_start|>assistant\n"

        t0 = time.monotonic()
        stream = llm(
            history,
            max_tokens=150,
            stop=stop_tokens,
            echo=False,
            stream=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            min_p=0.0,
            presence_penalty=1.5,
        )
        ttft = 0.0
        toks = 0
        parts: list[str] = []
        for chunk in stream:
            if toks == 0:
                ttft = time.monotonic() - t0
            parts.append(chunk["choices"][0]["text"])
            toks += 1
        gen = time.monotonic() - t0
        total_time += gen

        response = strip_think("".join(parts))
        history += response + "<|im_end|>"

        tps = toks / gen if gen > 0 else 0
        print(f"뭉이: {response}")
        print(f"  (TTFT: {ttft:.2f}s | {toks}tok | {gen:.1f}s | {tps:.1f} tok/s)")
        print()

    print("=" * 50)
    print(f"총 {len(user_messages)}턴 대화 시간: {total_time:.1f}s")
    print(f"평균: {total_time / len(user_messages):.1f}s/턴")
    del llm


if __name__ == "__main__":
    main()
