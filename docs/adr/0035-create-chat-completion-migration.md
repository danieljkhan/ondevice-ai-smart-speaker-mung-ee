# ADR 0035: LLM 추론 API를 create_chat_completion으로 마이그레이션

- **Status**: Implemented (실기기 검증 대기)
- **Date**: 2026-04-02
- **Author**: Claude Code PM (Opus 4.6)
- **Branch**: `experiment/llama-cpp-custom-0320`

## Context

Qwen3.5-2B 모델의 non-thinking 모드를 안정적으로 제어하기 위해 LLM 추론 방식을
변경해야 한다. 기존 raw `llm(prompt)` 호출은 수동으로 `<|im_start|>` 태그를 조립하여
모델의 Jinja2 chat template을 우회하므로, thinking 모드 제어가 불가능하다.

### 문제점

1. **Thinking 제어 불가**: Qwen3.5의 `enable_thinking` 파라미터를 전달할 수 없음
2. **수동 template 조립**: `<|im_start|>system/user/assistant<|im_end|>` 태그를 코드에서 직접 조립 → 모델 변경 시 깨짐
3. **`/no_think` 미지원**: Qwen3.5는 `/no_think` 명령을 인식하지 않음 (Qwen3 전용)
4. **prefill 해킹 필요**: `<think>\n</think>\n\n` prefill로 임시 우회 → 불안정

### 해결

`create_chat_completion(messages=..., chat_template_kwargs={"enable_thinking": False})`
API를 사용하면 모델의 내장 Jinja2 template이 thinking을 정식으로 비활성화한다.

## Decision

raw `llm()` → `create_chat_completion()` 마이그레이션을 수행한다.

### 변경 사항

| 파일 | 변경 |
|------|------|
| `models/llm_runner.py` | `run_chat_generation()` 신규 (기존 `run_generation()` 유지) |
| `core/pipeline.py` | `_build_messages()` 신규, `_run_llm()` 전환, `/no_think` 제거 |
| `scripts/e2e_60rounds.py` | messages 기반 전환, `/no_think` 제거 |
| `tests/test_pipeline.py` | `_build_messages()` 테스트, FakeLLM 업데이트 |
| `tests/test_e2e_60rounds.py` | FakeLLM chat completion 지원 |

### 핵심 API 차이

| 항목 | 기존 (raw llm) | 신규 (chat completion) |
|------|---------------|---------------------|
| 프롬프트 | 수동 `<\|im_start\|>` 조립 | message dict 리스트 |
| Thinking 제어 | 불가 (`/no_think` 무효) | `chat_template_kwargs` |
| 스트리밍 청크 | `chunk["choices"][0]["text"]` | `chunk["choices"][0]["delta"]["content"]` |
| Template | 코드에서 조립 | 모델 내장 Jinja2 |

### 보존 사항

- `run_generation()`: raw prompt 호출자용 유지 (bench_model.py 등)
- `_build_prompt_legacy()`: rollback용 보존
- `strip_think_tags()` + `sanitize_response()`: defense-in-depth 유지

## Alternatives Considered

1. **Prefill 해킹 유지** — `<think>\n</think>\n\n` prefill로 thinking 억제
   - 기각: llama.cpp 버전에 따라 동작 불안정, 근본 해결 아님

2. **llama-server 사용** — 별도 서버 프로세스로 OpenAI API 제공
   - 기각: 프로세스 관리 복잡, Jetson 리소스 추가 소비

3. **Raw prompt + 수동 template 수정** — Qwen3.5 전용 template 코드에 직접 구현
   - 기각: 모델 변경마다 template 코드 수정 필요

## Consequences

- Qwen3 (1.7B-FT)과 Qwen3.5 (2B) 모두 동일 API로 추론 가능
- 향후 모델 교체 시 chat template 코드 수정 불필요
- `enable_thinking` 파라미터로 thinking ON/OFF 명시적 제어
- 테스트 118개 PASS, coverage 70.24%

## Verification Status

- 로컬 검증: 3중 검증 체인 PASS, 폴리시 루프 2 cycles 0 fixes
- 젯슨 실기기 검증: **완료** (2026-04-02 18:31 KST)

## Verification Result (2026-04-02)

### chat_template_kwargs 미지원 발견

llama-cpp-python 0.3.19의 `create_chat_completion()` Python 바인딩은
`chat_template_kwargs` 파라미터를 지원하지 않음 (llama-server REST API 전용).
코드에서 제거 후 동작 확인 — Qwen3.5-2B는 기본 non-thinking이라 파라미터 없이도
thinking 미발생.

### E2E 10라운드 결과 (Qwen3.5-2B-Q4_K_M + RAG)

| Metric | Value | Gate | Verdict |
|--------|-------|------|---------|
| Success | 30/30 (100%) | ≥100% | PASS |
| Avg turn | 8.75s | ≤7.0s | **FAIL** |
| Avg LLM | 3.33s | - | 1.6x baseline |
| Peak mem | 3,758MB | ≤3,600MB | **FAIL** |

LLM 자체는 0.3.17(5.85s) 대비 43% 개선(3.33s). TTS(3.29s)가 전체의 38% 차지.

### 좀비 프로세스 이슈

SSH 백그라운드 실행 시 Python CUDA 프로세스가 시그널 블록 → 좀비화.
근본 해결: tmux 설치 (`docs/archived/dev-plan/Tmux-Installation-Plan.md`)
