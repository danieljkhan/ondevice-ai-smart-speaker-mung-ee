"""Merge FT v2 + E2E conversation data with quality labels.

Outputs to assets/training/ft_v3_labeled/ directory.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

HONORIFIC_ENDINGS = [
    "요", "습니다", "세요", "해요", "죠", "까요", "네요",
    "거예요", "줄게요", "할게요", "에요", "겠습니다", "나요",
]
BANMAL_ENDINGS = [
    "야", "해", "지", "거야", "할게", "볼까", "했어",
    "인데", "잖아", "일까", "어", "아",
]
HALLUCINATION_PATTERNS = [
    (r"달의 바다", "달-바다 혼동"),
    (r"갑오징어.*오리|오리.*갑오징어", "혼동"),
    (r"메아리.*생물|생물.*메아리", "메아리 환각"),
    (r"산소.*이산화탄소.*짜", "바닷물 오류"),
    (r"무서워.*만들었.*발", "오리발 환각"),
    (r"날개를 펄럭펄럭", "비행기 환각"),
    (r"아폴로.*바다", "아폴로 혼동"),
    (r"큰수달.*소리", "소리 환각"),
    (r"흙을 뿌려서 풍경", "환각"),
    (r"장갑을 끼고 하늘", "조종사 환각"),
    (r"비행기.*엄마.*닿을 때", "비행기 환각"),
]
EVASION_PATTERNS = [
    "모르겠", "같이 알아", "같이 생각", "같이 찾아",
    "엄마아빠한테", "엄마한테 물어", "아빠한테 물어", "잘 모르겠는데",
]
CATEGORIES = {
    "A": "반말_일관성", "B": "앵무새_방지", "C": "짧은_응답",
    "D": "감정_분화", "E": "환각_방지", "F": "한국어_유창성",
    "G": "의미_명확성", "H": "톤_앵커", "I": "존댓말_교정",
}
SYSTEM_PROMPT = (
    "You are 'Mungi(뭉이)', a warm and curious AI friend for children under 10. "
    "Use ONLY informal casual speech (반말). Keep responses to 2-3 sentences. "
    "Answer factually for common knowledge. Never echo the user's input."
)


def label_response(user_text: str, mungi_text: str) -> list[str]:
    """Detect quality issues in a Mungi response."""
    labels: list[str] = []
    text = mungi_text

    # 1. Honorific
    sentences = re.split(r"[.!?~]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    hon = sum(
        1 for s in sentences
        if any(s.rstrip("!?~. ").endswith(e) for e in HONORIFIC_ENDINGS)
    )
    ban = sum(
        1 for s in sentences
        if any(s.rstrip("!?~. ").endswith(e) for e in BANMAL_ENDINGS)
    )
    if hon > 0 and ban == 0:
        labels.append("존댓말_ONLY")
    elif hon > 0 and ban > 0:
        labels.append("반말+존댓말_혼합")

    # 2. Hallucination
    for pattern, _ in HALLUCINATION_PATTERNS:
        if re.search(pattern, text):
            labels.append("환각")
            break

    # 3. Truncation
    if text and text[-1] in "ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎ":
        labels.append("한국어_토큰_잘림")

    # 4. Repetition
    if re.search(r"(.{5,30})\1{2,}", text):
        labels.append("연속_반복")
    else:
        sents = [s.strip() for s in re.split(r"[.!?]+", text) if len(s.strip()) > 5]
        for i in range(len(sents) - 2):
            if sents[i] == sents[i + 1]:
                labels.append("연속_반복")
                break

    # 5. Foreign / Emoji
    eng = re.findall(r"[a-zA-Z]{3,}", text)
    eng = [w for w in eng if w.upper() not in {"OK", "TTS", "LLM", "RAG", "AI"}]
    if eng:
        labels.append("외국어")
    if re.search("[\U0001F300-\U0001F9FF]", text):
        labels.append("이모지")

    # 6. Evasion
    if any(p in text for p in EVASION_PATTERNS):
        labels.append("회피")

    # 7. Echo
    if user_text:
        cu = re.sub(r"[?!.,~\s뭉이야]+", "", user_text)
        cm = re.sub(r"[?!.,~\s]+", "", text)
        if cu and cm and len(cu) > 5 and cu in cm and len(cu) / len(cm) > 0.5:
            labels.append("앵무새")

    return labels


def main() -> int:
    """Merge and label all training data."""
    all_data: list[dict] = []

    # === 1. Process FT v2 categories ===
    ft_total = 0
    ft_labeled = 0
    for cat_id in "ABCDEFGHI":
        filepath = REPO_ROOT / "assets" / "training" / ("cat_%s.jsonl" % cat_id)
        if not filepath.exists():
            continue
        with filepath.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                msgs = item.get("messages", [])
                user_text = ""
                asst_text = ""
                for m in msgs:
                    if m["role"] == "user":
                        user_text = m["content"]
                    elif m["role"] == "assistant":
                        asst_text = m["content"]

                labels = label_response(user_text, asst_text)
                all_data.append({
                    "source": "ft_v2",
                    "category": "cat_%s_%s" % (cat_id, CATEGORIES[cat_id]),
                    "messages": msgs,
                    "labels": labels,
                    "quality": "BAD" if labels else "GOOD",
                })
                ft_total += 1
                if labels:
                    ft_labeled += 1

    print("FT v2: %d total, %d labeled (%.1f%%)" % (ft_total, ft_labeled, ft_labeled / max(ft_total, 1) * 100))

    # === 2. Process E2E conversations ===
    e2e_path = REPO_ROOT / "assets" / "training" / "e2e_labeled_conversations.jsonl"
    e2e_total = 0
    e2e_labeled = 0
    if e2e_path.exists():
        with e2e_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                all_data.append({
                    "source": "e2e_%s" % item.get("section", "")[:30],
                    "category": "e2e_%s" % item.get("topic", ""),
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": item["user"]},
                        {"role": "assistant", "content": item["mungi"]},
                    ],
                    "labels": item.get("labels", []),
                    "quality": "BAD" if item.get("labels") else "GOOD",
                })
                e2e_total += 1
                if item.get("labels"):
                    e2e_labeled += 1

    print("E2E: %d total, %d labeled (%.1f%%)" % (e2e_total, e2e_labeled, e2e_labeled / max(e2e_total, 1) * 100))
    print("Combined: %d total" % len(all_data))

    # === 3. Output ===
    output_dir = REPO_ROOT / "assets" / "training" / "ft_v3_labeled"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3a. All labeled
    with (output_dir / "all_labeled.jsonl").open("w", encoding="utf-8") as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 3b. GOOD only (SFT)
    good_count = 0
    with (output_dir / "sft_good_only.jsonl").open("w", encoding="utf-8") as f:
        for item in all_data:
            if item["quality"] == "GOOD":
                f.write(json.dumps({"messages": item["messages"]}, ensure_ascii=False) + "\n")
                good_count += 1

    # 3c. BAD only
    bad_count = 0
    with (output_dir / "negative_examples.jsonl").open("w", encoding="utf-8") as f:
        for item in all_data:
            if item["quality"] == "BAD":
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                bad_count += 1

    # 3d. DPO pairs
    user_groups: dict[str, dict[str, list]] = defaultdict(lambda: {"good": [], "bad": []})
    for item in all_data:
        for m in item["messages"]:
            if m["role"] == "user":
                user_groups[m["content"]][item["quality"].lower()].append(item)
                break

    dpo_count = 0
    with (output_dir / "dpo_pairs.jsonl").open("w", encoding="utf-8") as f:
        for user_text, group in user_groups.items():
            if group["good"] and group["bad"]:
                good_asst = next(
                    (m["content"] for m in group["good"][0]["messages"] if m["role"] == "assistant"), ""
                )
                bad_asst = next(
                    (m["content"] for m in group["bad"][0]["messages"] if m["role"] == "assistant"), ""
                )
                f.write(json.dumps({
                    "prompt": user_text,
                    "chosen": good_asst,
                    "rejected": bad_asst,
                    "rejected_labels": group["bad"][0]["labels"],
                }, ensure_ascii=False) + "\n")
                dpo_count += 1

    # 3e. Per-label files
    label_fds: dict[str, object] = {}
    label_counts: dict[str, int] = {}
    for item in all_data:
        for l in item.get("labels", []):
            cat = re.sub(r"[+/: ]", "_", l.split(":")[0].strip())
            if cat not in label_fds:
                label_fds[cat] = (output_dir / ("bad_%s.jsonl" % cat)).open("w", encoding="utf-8")
                label_counts[cat] = 0
            label_fds[cat].write(json.dumps(item, ensure_ascii=False) + "\n")
            label_counts[cat] += 1
    for fd in label_fds.values():
        fd.close()

    # === Summary ===
    print("\n=== Output: %s ===" % output_dir)
    print("all_labeled.jsonl:      %6d (전체)" % len(all_data))
    print("sft_good_only.jsonl:    %6d (SFT 학습용)" % good_count)
    print("negative_examples.jsonl:%6d (부정 사례)" % bad_count)
    print("dpo_pairs.jsonl:        %6d (DPO 학습용)" % dpo_count)
    for cat in sorted(label_counts.keys()):
        print("bad_%-20s: %5d" % (cat + ".jsonl", label_counts[cat]))

    # Combined label distribution
    combined: dict[str, int] = {}
    for item in all_data:
        for l in item.get("labels", []):
            cat = l.split(":")[0].strip()
            combined[cat] = combined.get(cat, 0) + 1

    print("\n=== Combined Label Distribution ===")
    for k, v in sorted(combined.items(), key=lambda x: -x[1]):
        print("  %-20s: %5d (%.1f%%)" % (k, v, v / len(all_data) * 100))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
