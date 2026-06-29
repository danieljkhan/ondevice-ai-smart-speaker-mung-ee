# ADR 0004: Enforce mandatory verification chain via Claude Code hooks

## Status

Accepted

## Context

CLAUDE.md와 AGENT_TEAM_SETUP.md에 6단계 검증 체인을 명문화했으나,
문서 규정만으로는 PM이 절차를 건너뛸 수 있다는 것이 Sprint 2에서 실증되었다
(PM이 2회 검증 절차를 위반하고 직접 산출물을 리뷰함).

문서적 강제만으로는 불충분하며, 기술적 강제가 필요하다.
특히 본 제품은 아동(10세 미만) 대상 AI 디바이스이므로,
안전 관련 검증 누락은 허용할 수 없다.

## Decision

Claude Code Hook 시스템을 활용하여 검증 원칙을 기술적으로 강제한다.

### 구성

| 파일 | 역할 |
|------|------|
| `.claude/verification-policy.json` | 검증 원칙 정의 (4명 에이전트, 6단계 체인) |
| `.claude/settings.json` | Hook 등록 |
| `.claude/hooks/enforce_verification.py` | PreToolUse hook: git commit/push/merge 차단 |
| `.claude/hooks/reset_verification.py` | PostToolUse hook: 파일 수정 시 검증 상태 초기화 |
| `.claude/verification-status.json` | 런타임 상태 (gitignore, 로컬 전용) |

### 동작 원리

1. **파일 수정 시** (Edit/Write/NotebookEdit): `reset_verification.py`가 실행되어
   `verification-status.json`을 초기화한다 (4명 전원 미완료).
2. **git commit/push/merge 시도 시**: `enforce_verification.py`가 실행되어
   `verification-status.json`을 검사한다.
   - 4명 에이전트 전원 완료 + consensus BLOCK 0건 + verified=true → 허용.
   - 그 외 → exit 2로 차단.

### 안전 설계

- **Fail-closed**: 예외 발생 시 커밋 차단 (fail-open이 아님).
- **Regex 명령어 탐지**: `\bgit\b.*\bcommit\b` 패턴으로 `git -C path commit` 등 우회 방지.
- **JSON 파싱 안전**: `verification-status.json` 손상/부재 시 fail-safe (unverified) 반환.
- **경로 예외**: `.claude/` 내부 파일과 `docs/runbooks/weekly/` 편집은 검증 초기화를 트리거하지 않음.

## Alternatives considered

1. **문서 규정만 유지**: Sprint 2에서 실패 실증됨. 기각.
2. **CI에서 검증 증거 확인**: GitHub Actions에서 검증 보고서 존재 여부를 체크하는 방안.
   장기적으로는 도입 가능하나, 현재는 로컬 에이전트 워크플로가 주요 작업 경로이므로
   Claude Code Hook이 더 직접적.
3. **pre-commit hook (git 레벨)**: Claude Code 외부에서도 적용 가능하지만,
   에이전트 검증 상태를 추적하려면 별도 상태 파일이 필요하며,
   Claude Code Hook이 에이전트 도구 호출과 직접 연동되므로 더 적합.

## Consequences

### 긍정적

- PM이 검증 절차를 건너뛸 수 없게 되어 안전 정책 준수가 기술적으로 보장됨.
- 파일 수정 → 자동 초기화 → 재검증 강제 사이클로 검증 누락 방지.
- Fail-closed 설계로 "세상에서 가장 안전한" 제품 비전에 부합.

### 부정적

- 검증 에이전트 4명을 매번 투입해야 하므로 커밋까지의 시간 증가.
- `verification-status.json` 상태 관리가 수동적 (PM이 에이전트 결과를 반영해야 함).
- Hook이 Claude Code 세션 내에서만 동작하므로, CLI git 직접 사용 시 우회 가능 (의도적 우회는 별도 통제).

## Related documents

- `CLAUDE.md` §8 (Mandatory deliverable verification)
- `AGENT_TEAM_SETUP.md` (Rules 15-18, Operating procedure steps 7-9)
- `docs/agents/agent-team-rr.md` (Mandatory Deliverable Verification section)
- ADR 0003 (QA/QC Agent split — 본 ADR의 선행 결정)
