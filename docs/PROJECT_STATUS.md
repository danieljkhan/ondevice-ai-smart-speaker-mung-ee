# Mungi 프로젝트 현황 및 백로그

> 최종 업데이트: 2026-05-17 KST (Session 46)
> 브랜치: `dev` (PR #108 squash-merged 2026-05-17, `6ba4024`)
> Primary Orchestrator: Claude Code (Opus 4.6)
>
> **이 문서는 Codex, Gemini 등 모든 에이전트가 작업 시작 전 반드시 읽어야 하는 필수 참조 문서입니다.**

---

## 1. 프로젝트 개요

| 항목 | 내용 |
|------|------|
| 비전 | "세상에서 가장 안전한, 우리 아이의 첫 번째 AI 친구" |
| 제품 | 오프라인 엣지 AI 대화 장치 (3~10세 아동 대상) |
| 하드웨어 | Jetson Orin Nano Super 8GB |
| 개발 기간 | 2026.3.1 ~ 5.31 (3개월) |
| 개발 형태 | 1인 개발 + AI 에이전트 (Claude Code PM + Codex Implementer) |
| 현재 단계 | POC 완료 → 품질 안정화 + RAG 검증 |

---

## 2. AI 파이프라인 (현재 활성)

```text
마이크 → [1. Silero VAD] → [2. Qwen3-ASR STT] → [3. Gemma 4 E2B LLM] → [4. Supertonic TTS 2] → 스피커
                (CPU, ~30MB)     (per ADR 0055)        (GPU, ~2.0–2.5GB)       (CPU, ~0.5GB)
                                                  [RAG: FAISS + koen-e5-tiny ONNX, 100-token budget]
```

| 모듈 | 엔진 | 모델 파일 | 상태 |
|------|------|----------|------|
| VAD | Silero VAD | (내장) | 정상 |
| STT | Qwen3-ASR (per ADR 0055 + 2026-04-29 Update; SenseVoice fallback retired) | Qwen3-ASR ONNX bundle | 정상, 별칭 매핑 적용 (ADR 0024 — alias map retained pending Qwen3-ASR verification) |
| LLM (primary) | llama.cpp + Gemma 4 E2B (per ADR 0073) | `gemma-4-E2B-it-Q5_K_M.gguf` | 정상 |
| LLM (fallback) | llama.cpp + Qwen3.5-2B-DPO (env/config override) | `Qwen3.5-2B-DPO.Q6_K.gguf` | 대기 (env/config override 시 활성) |
| TTS | Supertonic TTS 2 (sole TTS engine — Piper retired per ADR 0088) | (내장) | 정상 |
| RAG (대화 메모리) | FAISS + koen-e5-tiny ONNX (CPU/GPU) | `conversation_memory.faiss` + `koen-e5-tiny-onnx/` | 배포 진행 (100-token prompt budget) |
| RAG (위키, 별도) | FAISS + E5-small (CPU) | `wiki_faiss.index` + `e5_small_onnx/` | 배포 완료, E2E 테스트 진행 중 (150-token budget; 대화 메모리 인덱스와 분리) |

---

## 3. 에이전트 R&R (2026-03-26 기준)

```text
┌───────────────────────┬─────────────────────────────────────────┬────────────────────────────────────────┐
│         영역          │               Claude Code               │                 Codex                  │
├───────────────────────┼─────────────────────────────────────────┼────────────────────────────────────────┤
│ 프로젝트 관리         │ 우선순위, 스코프, 스케줄                │ —                                      │
├───────────────────────┼─────────────────────────────────────────┼────────────────────────────────────────┤
│ 설계/아키텍처         │ 기술 결정, 시스템 설계                  │ ADR 초안 작성 (Claude 결정 → Codex 문서)│
├───────────────────────┼─────────────────────────────────────────┼────────────────────────────────────────┤
│ 태스크 기획           │ task spec 작성 (.codex/current-task.md) │ task spec 읽고 실행                    │
├───────────────────────┼─────────────────────────────────────────┼────────────────────────────────────────┤
│ 코드 구현             │ ❌ 금지 (1줄이라도 위임)                │ ✅ 전담 (feature/platform/safety/test) │
├───────────────────────┼─────────────────────────────────────────┼────────────────────────────────────────┤
│ QC 실행               │ 검증 체인 용 (2nd filter)               │ 자체 검증 용 (1st filter)              │
├───────────────────────┼─────────────────────────────────────────┼────────────────────────────────────────┤
│ 코드 리뷰             │ 공식 3라운드 검증 체인                  │ 자체 self-verification 3라운드         │
├───────────────────────┼─────────────────────────────────────────┼────────────────────────────────────────┤
│ 폴리시 루프           │ 규칙 문서용 직접 수행                   │ 코드용 자체 수행 (20 iterations)       │
├───────────────────────┼─────────────────────────────────────────┼────────────────────────────────────────┤
│ 커밋/푸시             │ ✅ 사용자 승인 후 실행                  │ ❌ 금지                                │
├───────────────────────┼─────────────────────────────────────────┼────────────────────────────────────────┤
│ 배포                  │ ✅ 사용자 승인 후 실행                  │ ❌ 금지                                │
├───────────────────────┼─────────────────────────────────────────┼────────────────────────────────────────┤
│ E2E 테스트 보고서     │ 트리거 + 리뷰                           │ generate_e2e_report.py로 자동 생성     │
├───────────────────────┼─────────────────────────────────────────┼────────────────────────────────────────┤
│ 작업일지              │ 직접 작성 (PM 고유 업무)                │ —  (generate_worklog.py는 보조 도구)   │
├───────────────────────┼─────────────────────────────────────────┼────────────────────────────────────────┤
│ 규칙 문서             │ CLAUDE.md, AGENTS.md 직접 작성          │ —                                      │
├───────────────────────┼─────────────────────────────────────────┼────────────────────────────────────────┤
│ Jetson SSH            │ 진단, 모니터링, 배포                    │ —                                      │
├───────────────────────┼─────────────────────────────────────────┼────────────────────────────────────────┤
│ 오케스트레이션 인프라 │ .codex/chat/ 도구 작성 가능             │ —                                      │
└───────────────────────┴─────────────────────────────────────────┴────────────────────────────────────────┘
```

**Gemini CLI**: dormant. 설정 파일 유지, 활성 워크플로우에서 제외.

R&R 상세: `docs/agents/rnr-status-2026-03-26.md`

---

## 4. 최근 완료 작업 (2026-03-21 ~ 03-26)

| 날짜 | 작업 | 담당 | 커밋 | 상세 워크로그 |
|------|------|------|------|--------------|
| 03-21 | Jetson 7턴 실증 테스트 | 사용자 | — | `2026-03-21-live-demo-7turn-test-report.md` |
| 03-22 | 프롬프트 튜닝 + 출력 validator | Codex | — | `2026-03-22-daily-worklog.md` |
| 03-23 | E2E 60라운드 text-TTS 테스트 #2 | Claude | — | `2026-03-23-e2e-text-tts-60round-summary.md` |
| 03-24 | 라이브 음성 테스트 + 5초 응답 목표 분석 | Claude | — | `2026-03-24-live-voice-test-report.md` |
| 03-25 | QLoRA Phase 2 데이터 17,797건 생성 + 검증 | Claude+Codex | `deb6fc0` | `2026-03-25-daily-worklog.md` |
| 03-25 | 위키 RAG 시스템 구축 (43,234 chunks) | Codex | `deb6fc0` | `2026-03-25-daily-worklog.md` |
| 03-25 | ADR 0019~0025 신규 7건 작성 | Claude | `cd55d3f` | `2026-03-25-daily-worklog.md` |
| 03-25 | QLoRA 파인튜닝 실행 + GGUF 변환 + Jetson 배포 | 사용자 | — | — |
| 03-25 | E2E 60라운드 text-TTS 테스트 #3 (RAG baseline) | Claude | — | `2026-03-25-e2e-text-tts-60round-summary.md` |
| 03-26 | **P0 3건 수정** (후반부 퇴화·응답 반복·존댓말 유출) | Codex | `90a9e00` | `2026-03-26-daily-worklog.md` |
| 03-26 | **직통 워크플로우 전환** (relay.py → run-task.py) | Claude | `90a9e00` | `2026-03-26-daily-worklog.md` |
| 03-26 | E2E --rag 플래그 추가 | Codex | `231f674` | `2026-03-26-daily-worklog.md` |
| 03-26 | 검증 훅 설계 갭 수정 + mungidev.sh | Claude+Codex | `b882657` | `2026-03-26-daily-worklog.md` |
| 03-26 | RAG 임베딩 CPU 전용 + E5 모델 Jetson 배포 | Claude | `da764f2` | `2026-03-26-daily-worklog.md` |
| 03-26 | R&R 재정의 + 규칙 문서 전면 업데이트 | Claude | — | `rnr-status-2026-03-26.md` |

---

## 5. E2E 테스트 결과 요약

### 최신: 2026-03-26 P0 수정 후 60라운드 (baseline, RAG 비활성)

| 지표 | 3/25 baseline | **3/26 P0 fix** | 변화 |
|------|-------------|----------------|------|
| 성공률 | 100% | **100%** | 유지 |
| 평균 LLM | 3.78s | **3.22s** | -15% |
| 평균 턴 전체 | 10.32s | **9.76s** | -5% |
| **R51-R60 avg LLM** | **7.23s** | **3.32s** | **-54%** |
| **Late/Early ratio** | **2.6x** | **0.99x** | **퇴화 해소** |
| **느린 턴 (>15s)** | **23건** | **0건** | **해소** |
| **응답 반복** | **11.8%** | **4.2%** | **-64%** |
| **존댓말 유출** | **1.8%** | **0.0%** | **해소** |
| 피크 메모리 | 5,553MB | **5,451MB** | -102MB |

**상세**: `docs/runbooks/weekly/archive/2026-03-25-e2e-text-tts-60round-summary.md`, `2026-03-26-daily-worklog.md`

### 진행 중: RAG 활성 60라운드 E2E (--rag, seed=42)

- 상태: Jetson에서 실행 중 (~26/60 라운드, 15:20 KST 기준)
- 목적: baseline vs RAG A/B 비교

---

## 6. 우선순위 백로그

### P0 — 즉시 (완료)

| # | 과제 | 상태 |
|---|------|------|
| ~~P0-1~~ | 후반부 퇴화 (R54+ LLM 3x) | ✅ 해소 — session_audio RAM 제거 + clear_history |
| ~~P0-2~~ | 응답 반복 (11.8%) | ✅ 4.2%로 개선 — KNOWLEDGE BOUNDARY 다양화 + dedupe |
| ~~P0-3~~ | 존댓말 유출 (1.8%) | ✅ 0%로 해소 — SAFE_FALLBACK 반말 + 할까요 매핑 |

### P1 — 높은 우선순위 (이번 주)

| # | 과제 | 상태 | 관련 파일 |
|---|------|------|----------|
| P1-1 | E2E RAG A/B 비교 테스트 | 🔄 진행 중 | `scripts/e2e_60rounds_text_tts.py --rag` |
| P1-2 | TTFT 악화 원인 조사 (0.64s→1.41s) | 대기 | `core/model_manager.py` |
| P1-3 | 자동화 스크립트 3건 | 대기 | `scripts/generate_e2e_report.py` 등 |
| P1-4 | 컨텍스트 리크 방지 | 대기 (재테스트 결과 0건, 재확인 필요) | `core/pipeline.py` |

### P2 — 중간 우선순위

| # | 과제 | 관련 파일 |
|---|------|----------|
| P2-1 | STT 영어 혼용 오인식 개선 | `models/stt_runner.py` |
| P2-2 | Processing feedback audio (대기음) | `assets/sounds/`, `core/pipeline.py` |
| P2-3 | systemd 서비스 (자동 부팅) | `systemd/` |
| P2-4 | 보호자 웹 대시보드 | `parental/` |
| P2-5 | 디스플레이 표정 + 정전식 터치 (4inch HDMI LCD, 720×720) | `hardware/display.py` (예정) |

---

## 7. Jetson 환경 정보

| 항목 | 값 |
|------|-----|
| 사용자 | `mungi` |
| IP | `172.20.10.9` (Wi-Fi) |
| OS | Ubuntu 22.04, JetPack 6.2, CUDA 12.6, Python 3.10 |
| 런타임 트리 | `/opt/mungi` (보호됨, 직접 수정 금지) |
| 개발 트리 | `/opt/mungi-repo` (git-managed) |
| 모델 경로 | `/opt/mungi/ai_models/` |
| LLM 모델 (primary) | `gemma-4-E2B-it-Q5_K_M.gguf` (per ADR 0073, 2026-04-25 primary swap) |
| LLM 모델 (fallback) | `Qwen3.5-2B-DPO.Q6_K.gguf` (env/config override `MUNGI_LLM_BACKEND=qwen3_legacy` 또는 `"llm_backend": "qwen3_legacy"`) |
| RAG 데이터 | `/opt/mungi/ai_models/rag/` (FAISS 64MB + E5 466MB + chunks 30MB) |
| 설정 경로 | `/var/lib/mungi/config/config.json` |
| 시작 명령 | `mungidev` (개발), `scripts/mungidev.sh` (SSH 비대화식) |
| 오디오 | USB PnP Audio Device (Waveshare, hw:0,0) |

---

## 8. 워크플로우

```text
Claude Code ──(task spec)──→ run-task.py ──→ Codex (GPT-5.4 xhigh)
    │                           │                    │
    │ ← auto-notification ──────┤                    │ 구현 + self-verification
    │                           │← handoff.md ───────┘
    │ 검증 체인 3라운드         │
    │ 사용자 승인 → 커밋/푸시   │
    │                           │→ verification-status.json 자동 리셋
```

모니터링: `tail -f .codex/chat/codex-output.log`

---

## 9. 문서 참조 맵

| 문서 | 위치 | 용도 |
|------|------|------|
| 프로젝트 규칙 (원본) | `CLAUDE.md` | 모든 규칙의 single source of truth |
| Codex 규칙 | `AGENTS.md` | Codex용 규칙 (CLAUDE.md에서 파생) |
| Codex 헌법 | `docs/CODEX_CONSTITUTION.md` | Codex 바인딩 운영 규칙 |
| Codex CLI 설정 | `.codex/config.json` | 모델, 샌드박스, 자율 범위 |
| R&R 현황 | `docs/agents/rnr-status-2026-03-26.md` | 역할 분담 상세 보고서 |
| R&R 정의 | `docs/agents/agent-team-rr.md` | 에이전트 역할 정의 |
| 에이전트 셋업 | `AGENT_TEAM_SETUP.md` | 워크플로우 + 자동화 스크립트 |
| 검증 체인 상세 | `docs/agents/instructions/verification-chain.md` | 3라운드 검증 + 폴리시 루프 |
| 검증 정책 JSON | `.claude/verification-policy.json` | 기계 판독용 검증 정책 |
| 뭉이 페르소나 | `assets/prompts/persona.md` | 캐릭터 설정 |
| 일일 워크로그 | `docs/runbooks/weekly/` | 날짜별 작업 기록 |

---

> 이 문서는 Claude Code(Primary Orchestrator)가 관리합니다. 최신 상태를 반영하기 위해 주요 작업 완료 시 업데이트됩니다.
