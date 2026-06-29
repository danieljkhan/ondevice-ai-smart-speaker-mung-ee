# 뭉이 프로젝트 로드맵 (5-Phase)

- **Date**: 2026-05-27
- **Status**: 사용자 정의 권위 문서
- **Scope**: 전체 프로젝트 phase 정의 및 현재 진행 상태
- **Note**: 이 문서는 phase 명명/매핑의 단일 권위(single source of truth). 개별 plan 문서가 자체 internal phase를 갖더라도, 프로젝트 전체 phase 명칭은 이 문서를 따른다. CLAUDE.md는 의도적으로 이 로드맵을 참조하지 않는다 (rule 문서와 roadmap 분리).

---

## 1. Phase 요약

| Phase | 주제 | 상태 | 권위 plan |
|---|---|---|---|
| **1** | 음성인식 + 음성답변 (음성 파이프라인) | **완성** — 동작 가능 | (다수 ADR; STT/LLM/TTS 영역별) |
| **2** | 터치스크린 UI + 뭉이 표정 표현 | **진행 중** — B-1 완료, B-2 미시작 | `Dev_Plan/2026-05-25-touchscreen-entry-phase1-plan.md` |
| **3** | 학습모드 + 부모모드 | **미착수** | (plan 미작성) |
| **4** | WiFi CSI 센싱 (부모 안전 모니터링 데이터 레이어) | **Plan v1 draft** — Codex r1 대기 | `Dev_Plan/2026-05-26-parental-safety-monitoring-plan.md` |
| **5** | 제품 출시 + 3D 프린터 스마트스피커 케이스 | **미시작** | (plan 미작성) |

---

## 2. Phase 상세

### Phase 1 — 음성인식 + 음성답변

핵심 음성 파이프라인. VAD → STT → LLM → TTS 전 구간 동작.

- **VAD**: Silero VAD
- **STT**: Qwen3-ASR INT8 (sherpa-onnx) — ADR 0055
- **LLM**: Gemma 4 E2B Q5_K_M (primary, llama.cpp GGUF), Qwen3.5-2B-DPO Q6_K (fallback) — ADR 0073
- **TTS**: Supertonic 2 + tobi (KO) / F2 (EN) — ADR 0088, per-language routing
- **검증 산출물**: `artifacts/live-demo-20260524/`, Workstream A 완결

STT fine-tune은 영구 보류 결정 (백업 모델 작업 종료).

### Phase 2 — 터치스크린 UI + 뭉이 표정 표현

H/W: Waveshare 4inch HDMI Capacitive Touch IPS LCD 720×720. WS2812B LED + RP2040 Pico는 폐기 (ADR-NEW-1).

내부 구성:

- **B-1 터치 트리거 대화 진입** — **완료** (PR #139 squash-merge `a541405`, 2026-05-25)
  - sound bank, core infra (`AudioCapture`, `SessionManager`, `EventLog`), main loop 통합
  - `CharacterRenderer` Protocol + `CharacterExpression(NEUTRAL)` enum placeholder
- **B-2 캐릭터 디스플레이 + 뭉이 표정 표현** — 미시작 (다음 작업 후보)
  - `PygameCharacterRenderer` 구현
  - `CharacterExpression` 8 멤버 확장: `idle / listening / thinking / speaking / happy / sad / surprised / concerned`
  - 표정 ↔ 세션 상태 머신 매핑

### Phase 3 — 학습모드 + 부모모드

부모 UI/UX 레이어 + 어린이 학습 모드.

- **부모모드**: PIN UI + 부모 대시보드 + 알림 통합 (Workstream B Plan v4 §4 Phase 3에 architecture skeleton 존재; 본 phase는 이를 정식 deliverable로 격상)
- **학습모드**: 영어 우선 (Workstream B Plan v4 §4 Phase 4에 skeleton 존재; persona module mode-aware registry 확장 필요)
- **데이터 소스**: Phase 4의 WiFi CSI 이벤트 (presence/motion/vital signs)를 부모 대시보드가 소비

Plan 미작성 상태. Phase 4 dependency 해소 후 본격 plan 착수.

### Phase 4 — WiFi CSI 센싱 (데이터 레이어)

부모 안전 모니터링의 **데이터 수집 백엔드**. Phase 3 부모모드가 소비할 이벤트 소스를 만든다 (presence/motion/breathing BPM/heart rate BPM).

- **구현**: `ruvnet/wifi-densepose` Docker (multi-arch arm64) on Jetson + ESP32-S3 × 4 mesh
- **H/W**: ESP32-S3 × 4 (소유/주문 완료) + 18650 5V 배터리 팩
- **상태**: Plan v1 draft (2026-05-26, `1dc984a`). Codex `reviewer` r1 + mutual discussion + 사용자 최종 승인 대기 (Gate 1 미완)
- **Phase 0 verification gates**: 7개 미해소 사실(O1–O7) 명시 — 포트, API catalog, 토큰, 전력, 펌웨어 빌드, MQTT 브로커 등

카메라는 persona 충돌(privacy)로 영구 거부. WiFi CSI가 비-이미징 passive sensing 대안.

### Phase 5 — 제품 출시 + 3D 프린터 스마트스피커 케이스

- **3D 프린팅 enclosure**: 스마트스피커 형태의 케이스 제작
  - Jetson Orin Nano Super 8GB 보드 + Waveshare 4inch LCD + USB 오디오카드 + 마이크/스피커 통합
  - ESP32-S3 노드는 별도 enclosure (방 분산 배치)
- **출시 준비**: 패키징, 사용자 매뉴얼, QA, 보증/지원 절차

미시작. Phase 4 안정화 후 케이스 설계 시작 가능.

---

## 3. Touchscreen Plan v4 internal Phase ↔ 5-Phase 매핑

Plan v4가 자체 internal Phase 1~4를 정의해 두어 명칭 충돌 가능성이 있음. 본 로드맵 기준으로 다음과 같이 매핑:

| Plan v4 internal | 5-Phase 매핑 | 비고 |
|---|---|---|
| Plan §3 Phase 1 (터치 트리거 + core infra) | **Phase 2 B-1** | 완료 (PR #139) |
| Plan §4 Phase 2 (캐릭터 디스플레이) | **Phase 2 B-2** | 다음 deliverable |
| Plan §4 Phase 3 (부모 모드 PIN/대시보드) | **Phase 3** | 부모모드 본체로 격상 |
| Plan §4 Phase 4 (학습 모드 영어) | **Phase 3** | 학습모드 본체로 격상 |

향후 plan 작성 시 명칭 충돌을 피하려면 Plan internal 단계는 "step" 또는 "B-1/B-2/..." 명명을 권장한다 (강제 아님; Plan v4는 그대로 둔다).

---

## 4. 현재 진행 위치 (2026-05-27 기준)

- **Phase 1**: 완성 (유지보수만)
- **Phase 2 B-1**: 완료
- **Phase 2 B-2**: 다음 대기열 후보 (caracter renderer/expression)
- **Phase 4**: Plan v1 draft, Codex r1 대기 — Gate 1 미완
- **Phase 3 / Phase 5**: plan 미작성

**병행 가능성**: Phase 2 B-2 (UI/표정)와 Phase 4 (WiFi 센싱)는 의존성이 없어 병행 가능. Phase 3 (부모모드 UI)는 Phase 4 데이터 레이어가 일부라도 완성된 뒤 시작이 효율적.

---

## 5. Changelog

| Version | Date | 변경 |
|---|---|---|
| v1 | 2026-05-27 | 최초 작성 — 5-phase 정의, Phase 2에 표정 표현 포함, Phase 5에 3D 프린터 케이스 제작 추가, Plan v4 internal Phase 매핑 표 |
