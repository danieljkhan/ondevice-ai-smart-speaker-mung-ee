# ADR 0003: Split QA/QC Agent into separate QA Agent and QC Agent

## Status

Accepted

## Context

Sprint 2 초반에 PM이 산출물을 직접 리뷰하여 검증 절차를 2회 위반하는 사고가 발생했다.
원인 분석 결과, 기존 단일 QA/QC Agent가 테스트 작성과 실행을 모두 담당하면서
다음 문제가 확인되었다:

1. **역할 혼재**: 테스트를 작성한 에이전트가 자기 테스트를 실행하면 독립적 검증이 보장되지 않음.
2. **병목**: 테스트 작성이 끝나야 실행할 수 있으므로 직렬 대기 발생.
3. **책임 불명확**: 테스트 실패 시 작성 문제인지 실행 환경 문제인지 구분이 어려움.

## Decision

QA/QC Agent 1명을 QA Agent + QC Agent 2명으로 분리한다.

| 에이전트 | 역할 | 쓰기 스코프 |
|---------|------|-----------|
| QA Agent | 테스트 설계 + 테스트 코드 작성 | `tests/` |
| QC Agent | 테스트 실행 + lint/type 검사 + 검증 보고서 작성 | `docs/runbooks/` |

주요 규칙:
- QA Agent는 테스트를 실행하지 않는다 (QC Agent의 역할).
- QC Agent는 테스트를 작성하지 않는다 (QA Agent의 역할).
- 양쪽 모두 프로덕션 코드를 수정하지 않는다.

이에 따라 전체 에이전트 수가 10명에서 11명으로 증가한다.

## Consequences

### 긍정적

- 테스트 작성과 실행의 독립성이 보장되어 검증 신뢰도 향상.
- QA/QC 병렬 준비 가능 (QA가 작성하는 동안 QC가 기존 테스트 실행 가능).
- 역할 경계가 명확하여 위반 감지 용이.

### 부정적

- 에이전트 수 증가에 따른 조율 복잡도 미세 증가.
- PM의 에이전트 관리 부담 증가 (4명 검증 에이전트 동시 투입).

## Related documents

- `CLAUDE.md` §8 (Lane composition, Key rules)
- `AGENT_TEAM_SETUP.md` (QA Agent, QC Agent sections)
- `docs/agents/agent-team-rr.md` (§9, §10)
