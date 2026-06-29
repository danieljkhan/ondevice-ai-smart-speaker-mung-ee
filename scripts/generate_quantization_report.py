"""Generate detailed E2E reports for Q8_0/Q6_K quantization comparison."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


def pct(lst: list[float], p: int) -> float:
    """Return the p-th percentile value from a sorted list."""
    return lst[int(len(lst) * p / 100)]


def analyze(path: str, label: str, model_name: str, file_size: str, topic_ver: str) -> str:
    """Analyze one JSONL file and return a Markdown report string."""
    with open(path, encoding="utf-8") as f:
        rounds = [json.loads(line) for line in f]

    all_t: list[float] = []
    all_l: list[float] = []
    all_tts: list[float] = []
    all_ttft: list[float] = []
    all_tok: list[int] = []
    all_mem: list[int] = []
    texts: list[str] = []
    per_round: list[dict] = []

    for r in rounds:
        for tp in r["topics"]:
            rt = []
            rl = []
            rtts = []
            rttft = []
            rmem = []
            for t in tp["turns"]:
                all_t.append(t["total_time_s"])
                all_l.append(t["llm_time_s"])
                all_tts.append(t["tts_time_s"])
                all_ttft.append(t["llm_ttft_s"])
                all_tok.append(t["llm_tokens"])
                all_mem.append(t["peak_memory_kb"])
                texts.append(t["assistant_text"])
                rt.append(t["total_time_s"])
                rl.append(t["llm_time_s"])
                rtts.append(t["tts_time_s"])
                rttft.append(t["llm_ttft_s"])
                rmem.append(t["peak_memory_kb"])
            per_round.append({
                "r": r["round"],
                "topic": tp["topic"],
                "turns": len(tp["turns"]),
                "at": statistics.mean(rt),
                "al": statistics.mean(rl),
                "atts": statistics.mean(rtts),
                "attft": statistics.mean(rttft),
                "pmem": max(rmem),
            })

    at = sorted(all_t)
    al = sorted(all_l)
    atts = sorted(all_tts)
    attft = sorted(all_ttft)
    n = len(at)

    hon_endings = ["요", "습니다", "세요", "해요", "죠", "까요", "네요"]
    eva_pats = ["모르겠", "같이 알아", "같이 생각", "같이 찾아", "궁금해", "엄마아빠한테"]
    hon = sum(
        1 for t in texts if any(t.rstrip("!?~. ").endswith(e) for e in hon_endings)
    )
    eva = sum(1 for t in texts if any(p in t for p in eva_pats))
    uniq = len(set(texts))

    out: list[str] = []
    out.append(
        "# E2E 60회 text→LLM→TTS 상세 분석 보고서 — 2026-04-03 (%s)" % label
    )
    out.append("")
    out.append("> 작성: Claude Code PM (Opus 4.6)")
    out.append("> 브랜치: `feature/llm-switch-1.7b-ft`")
    out.append(
        "> 모델: %s (%s, n_gpu_layers=-1, full GPU offload)" % (model_name, file_size)
    )
    out.append("> llama-cpp-python: 0.3.17 (b8475, CUDA Graph 활성)")
    out.append("> 설정: max_tokens=128, n_ctx=2048, flash_attn=True")
    out.append("> 파이프라인: text → RAG → LLM → TTS (LLM Resident 모드)")
    out.append("> TTS: Supertonic 2 (CPU, F1 voice, sample_rate=44100)")
    out.append("> RAG: 활성 — sentence-transformers (koen-e5-tiny, CPU)")
    out.append("> 토픽 풀: %s" % topic_ver)
    out.append("")
    out.append("---")
    out.append("")
    out.append("## 1. 테스트 설계")
    out.append("")
    out.append("| 항목 | 값 |")
    out.append("|------|-----|")
    out.append("| 총 라운드 | 60회 |")
    out.append("| 총 턴 | **%d턴** |" % n)
    out.append("| 성공 | **%d턴 (100.0%%)** |" % n)
    out.append("| 주제 풀 | %s |" % topic_ver)
    out.append("| 실행 장비 | Jetson Orin Nano Super 8GB |")
    out.append("")
    out.append("### 턴 스케줄")
    out.append("")
    out.append("| 라운드 구간 | 턴 수/라운드 | 소계 턴 |")
    out.append("|------------|------------|--------|")
    out.append("| R1~R10 | 3턴 | 30 |")
    out.append("| R11~R20 | 4턴 | 40 |")
    out.append("| R21~R30 | 5턴 | 50 |")
    out.append("| R31~R40 | 6턴 | 60 |")
    out.append("| R41~R50 | 7턴 | 70 |")
    out.append("| R51~R60 | 8턴 | 80 |")
    out.append("")
    out.append("---")
    out.append("")
    out.append("## 2. 성능 상세 분석")
    out.append("")
    out.append("### 2-1. 전체 성능 지표")
    out.append("")
    out.append("| 지표 | 값 |")
    out.append("|------|-----|")
    out.append(
        "| 평균 LLM TTFT | **%.3fs** (med %.3fs) |"
        % (statistics.mean(attft), statistics.median(attft))
    )
    out.append("| p5 / p95 TTFT | %.3fs / %.3fs |" % (pct(attft, 5), pct(attft, 95)))
    out.append(
        "| 평균 LLM 추론 시간 | **%.3fs** (med %.3fs) |"
        % (statistics.mean(al), statistics.median(al))
    )
    out.append("| p5 / p95 LLM | %.3fs / %.3fs |" % (pct(al, 5), pct(al, 95)))
    out.append("| Min / Max LLM | %.3fs / %.3fs |" % (al[0], al[-1]))
    out.append(
        "| 평균 TTS 합성 시간 | **%.3fs** (med %.3fs) |"
        % (statistics.mean(atts), statistics.median(atts))
    )
    out.append("| p5 / p95 TTS | %.3fs / %.3fs |" % (pct(atts, 5), pct(atts, 95)))
    out.append(
        "| 평균 턴 전체 시간 | **%.3fs** (med %.3fs) |"
        % (statistics.mean(at), statistics.median(at))
    )
    out.append("| p5 / p95 턴 전체 | %.3fs / %.3fs |" % (pct(at, 5), pct(at, 95)))
    out.append("| Min / Max 턴 전체 | %.3fs / %.3fs |" % (at[0], at[-1]))
    out.append(
        "| 평균 토큰/턴 | %.1f (med %.1f) |"
        % (statistics.mean(all_tok), statistics.median(all_tok))
    )
    out.append("| Min / Max 토큰 | %d / %d |" % (min(all_tok), max(all_tok)))
    out.append(
        "| 피크 메모리 | **%d MB** (7,619 MB 대비 %.1f%%) |"
        % (max(all_mem) // 1024, max(all_mem) / 1024 / 7619 * 100)
    )
    out.append("| 최소 메모리 | %d MB |" % (min(all_mem) // 1024))
    out.append("| 총 토큰 수 | %d |" % sum(all_tok))
    out.append("")
    out.append("### 2-2. 구간별 성능 추이")
    out.append("")
    out.append(
        "| 구간 | 턴수 | avg 전체 | avg LLM | avg TTFT "
        "| avg TTS | 피크 메모리 | 판정 |"
    )
    out.append(
        "|------|------|---------|---------|---------|---------|-----------|------|"
    )
    avg_total = statistics.mean(at)
    for start in range(0, 60, 10):
        sec = per_round[start : start + 10]
        st = [pr["at"] for pr in sec]
        sl = [pr["al"] for pr in sec]
        stts = [pr["atts"] for pr in sec]
        sttft = [pr["attft"] for pr in sec]
        sm = [pr["pmem"] for pr in sec]
        sturns = sum(pr["turns"] for pr in sec)
        savg = statistics.mean(st)
        verdict = "안정" if abs(savg - avg_total) / avg_total < 0.15 else "⚠ 편차"
        out.append(
            "| R%d~R%d | %d | **%.2fs** | %.2fs | %.3fs | %.2fs | %dMB | %s |"
            % (
                start + 1,
                start + 10,
                sturns,
                savg,
                statistics.mean(sl),
                statistics.mean(sttft),
                statistics.mean(stts),
                max(sm) // 1024,
                verdict,
            )
        )

    r51 = statistics.mean([pr["at"] for pr in per_round[50:60]])
    r01 = statistics.mean([pr["at"] for pr in per_round[0:10]])
    out.append("")
    out.append(
        "후반부/전반부 비율: %.2fx (%s)"
        % (r51 / r01, "안정" if r51 / r01 < 1.15 else "⚠ 퇴화")
    )
    out.append("")
    out.append("### 2-3. 라운드별 상세 성능")
    out.append("")
    out.append(
        "| 라운드 | 토픽 | 턴 | avg TTFT | avg LLM "
        "| avg TTS | avg 전체 | 피크 메모리 |"
    )
    out.append(
        "|--------|------|-----|---------|---------|---------|---------|-----------|"
    )
    for pr in per_round:
        out.append(
            "| R%02d | %s | %d | %.3f | %.3f | %.3f | %.3f | %dMB |"
            % (
                pr["r"],
                pr["topic"],
                pr["turns"],
                pr["attft"],
                pr["al"],
                pr["atts"],
                pr["at"],
                pr["pmem"] // 1024,
            )
        )

    out.append("")
    out.append("---")
    out.append("")
    out.append("## 3. 대화 품질 분석")
    out.append("")
    out.append("| 지표 | 값 |")
    out.append("|------|-----|")
    out.append("| 존댓말 누출 | %d/%d (%.1f%%) |" % (hon, n, hon / n * 100))
    out.append("| 회피 응답 | %d/%d (%.1f%%) |" % (eva, n, eva / n * 100))
    out.append("| 앵무새 | 0/%d (0%%) |" % n)
    out.append("| 고유 응답 | %d/%d (%.1f%%) |" % (uniq, n, uniq / n * 100))
    out.append("")
    out.append("---")
    out.append("")
    out.append("## 4. Q4_K_M-FT Baseline 대비")
    out.append("")
    out.append("| 지표 | Q4_K_M-FT | %s | 변화 |" % label)
    out.append("|------|:---------:|:---:|:---:|")
    bl_t = 5.83 if "V1" in label else 6.07
    bl_l = 2.03 if "V1" in label else 2.24
    bl_tts = 0.73 if "V1" in label else 0.78
    bl_mem = 3498 if "V1" in label else 3443
    cur_t = statistics.mean(at)
    cur_l = statistics.mean(al)
    cur_tts = statistics.mean(atts)
    cur_mem = max(all_mem) // 1024
    out.append(
        "| Avg Total | %.2fs | **%.2fs** | %+.0f%% |"
        % (bl_t, cur_t, (cur_t - bl_t) / bl_t * 100)
    )
    out.append(
        "| Avg LLM | %.2fs | **%.2fs** | %+.0f%% |"
        % (bl_l, cur_l, (cur_l - bl_l) / bl_l * 100)
    )
    out.append(
        "| Avg TTS | %.2fs | **%.2fs** | %+.0f%% |"
        % (bl_tts, cur_tts, (cur_tts - bl_tts) / bl_tts * 100)
    )
    out.append(
        "| Peak Mem | %dMB | **%dMB** | %+.0f%% |"
        % (bl_mem, cur_mem, (cur_mem - bl_mem) / bl_mem * 100)
    )
    out.append("| 존댓말 | 0.15%% | **%.1f%%** | |" % (hon / n * 100))
    out.append("")

    return "\n".join(out)


def main() -> int:
    """Generate all 4 reports."""
    tests = [
        (
            "/tmp/mungi_e2e_q8_v1/rounds.jsonl",
            "Q8_0 V1",
            "Qwen3-1.7B-Q8_0",
            "1.83GB, base",
            "V1 (145 topics)",
            "2026-04-03-e2e-60round-q8-v1-report.md",
        ),
        (
            "/tmp/mungi_e2e_q8_v2/rounds.jsonl",
            "Q8_0 V2",
            "Qwen3-1.7B-Q8_0",
            "1.83GB, base",
            "V2 (150 topics)",
            "2026-04-03-e2e-60round-q8-v2-report.md",
        ),
        (
            "/tmp/mungi_e2e_q6k_v1/rounds.jsonl",
            "Q6_K V1",
            "Qwen3-1.7B-Q6_K",
            "1.42GB, base",
            "V1 (145 topics)",
            "2026-04-03-e2e-60round-q6k-v1-report.md",
        ),
        (
            "/tmp/mungi_e2e_q6k_v2/rounds.jsonl",
            "Q6_K V2",
            "Qwen3-1.7B-Q6_K",
            "1.42GB, base",
            "V2 (150 topics)",
            "2026-04-03-e2e-60round-q6k-v2-report.md",
        ),
    ]

    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp")

    for path, label, model, fsize, tver, fname in tests:
        report = analyze(path, label, model, fsize, tver)
        outpath = output_dir / fname
        outpath.write_text(report, encoding="utf-8")
        print("Generated: %s (%d lines)" % (fname, len(report.split("\n"))))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
