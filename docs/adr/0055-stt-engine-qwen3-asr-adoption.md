# ADR 0055 — STT 엔진 선택: Qwen3-ASR-0.6B INT8 도입, SenseVoice 폴백 유지

## Status

- Accepted (Decision 3 superseded 2026-04-28; Decisions 1-2 active)
- Date: 2026-04-13

## Context

2026-04-12 실음성 E2E pilot에서 기존 Sherpa-ONNX SenseVoice 경로가 한국어 아동 발화 전사
품질 문제를 드러냈다. 같은 날 오후 진행한 Gemma 4 ASR spike는 메모리와 배포 경로 제약으로
운영 가능한 대안이 되지 못했다.

2026-04-13 오후 세션에서 Mungi는 재플래시된 Jetson Orin Nano Super 8GB 환경을 처음부터
복원하면서, STT를 별도 경로로 재설계할 필요가 생겼다. 이 시점에 다음 조건을 만족하는 대안이
필요했다.

1. Jetson 8GB에서 현실적으로 배포 가능해야 한다.
2. 한국어 품질이 SenseVoice 대비 개선되어야 한다.
3. 기존 `core/pipeline.py`와 sequential model manager 구조를 유지해야 한다.
4. 실패 시 즉시 기존 SenseVoice로 되돌릴 수 있어야 한다.

Qwen3-ASR-0.6B는 sherpa-onnx 1.12.34+에서 공식 지원이 추가되었고, 2026-04-13 현재 Jetson
환경에는 sherpa-onnx 1.12.38이 설치되어 있었다. 실제 확인된 API는
`OfflineRecognizer.from_qwen3_asr(...)`였고, 배포 bundle
`sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25`는 `conv_frontend.onnx`,
`encoder.int8.onnx`, `decoder.int8.onnx`, `tokenizer/` 구조를 제공했다.

같은 세션에서 수행한 검증 결과는 다음과 같았다.

- KO 12-WAV 단독 전사 평균 CER: 3.7%
- EN 12-WAV 단독 전사 실질 WER: 약 5%
- hotwords MINIMAL 변형 `뭉이,Moongee`: persona recall KO 5/5, EN 5/5
- Jetson 24-round KO/EN interleaved E2E: 24 rounds complete, ASR CER 0.165132, ASR WER 0.340596

## Decision

Mungi의 새로운 STT 채택 경로로 Qwen3-ASR-0.6B INT8를 도입한다.

세부 결정은 다음과 같다.

1. Jetson 복구 및 신규 배포 경로의 기본 STT 후보는
   `sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25` bundle로 한다.
2. 로더 구현은 sherpa-onnx 1.12.38의
   `OfflineRecognizer.from_qwen3_asr(...)` API를 사용한다.
3. 기존 SenseVoice 경로는 삭제하지 않고 `models/stt_runner.py` 내부에 공존시켜 즉시 폴백이
   가능하도록 유지한다.
4. persona hotwords 기본값은 MINIMAL 변형 `뭉이,Moongee`로 고정한다.
5. 초기 운영 경로는 GPU wheel 전제 없이 **INT8 ONNX + CPU provider** 기준으로 문서화한다.
6. 전역 기본값(`DEFAULT_MODEL_SIZE`)의 강제 전환은 같은 세션에서 하지 않고, 후속 A/B 결과와
   formal verification 이후 별도 PR에서 결정한다.

## Consequences

### Positive

- 한국어 아동 발화 전사 품질이 기존 SenseVoice 대비 유의미하게 개선되는 경로를 확보했다.
- Sherpa-ONNX 기반 offline pipeline을 유지해 기존 VAD, LLM, TTS 인터페이스를 바꾸지 않았다.
- hotwords MINIMAL(`뭉이,Moongee`)만으로 persona recall KO 5/5, EN 5/5를 달성했다.
- 24-round KO/EN interleaved E2E가 끝까지 완료되어 실제 운영 가능성을 확인했다.

### Trade-offs

- 모델 artifact 기준으로 약 600 MB급 STT bundle이 추가되고, resident memory 기준으로는
  SenseVoice 대비 더 큰 ONNX Runtime 오버헤드가 생긴다.
- Qwen3-ASR는 streaming이 아니라 offline recognizer 경로이며, 입력 길이 제한도 5분이다.
- 현재 pip wheel 기준 sherpa-onnx GPU build가 아니므로, 초기 운영은 CPU provider에
  의존한다.
- 24-round E2E에서 평균 `first_sound_ms`가 21.5s, `total_ms`가 22.8s로 아동 UX 목표
  (5s 이하)와 큰 차이가 남아 있다.

### Operational Impact

- 단독 STT 품질은 충분히 개선되었지만, E2E mix에서는 KO CER 16.5%, EN WER 34.1%로
  단독 측정 대비 성능 저하가 존재한다.
- `avg_stt_ms`는 5.9s였지만 `avg_stt_total_ms`는 13.3s였다. 이는 순차 로더 구조에서
  turn마다 발생하는 STT reload 비용이 병목임을 뜻한다.
- 후속 우선순위는 STT resident mode 검토, sherpa-onnx GPU build 평가, E2E accuracy gap
  원인 분석이다.

## Related ADRs

- 0054 — Gemma 4 Extended Pilot (병행 의사결정 스레드, 이 worktree에는 본문 파일이 아직 없음)
- `docs/adr/0049-e2e-voice-pilot-methodology.md`
- `docs/adr/0046-tts-voice-selection-f2.md`

## References

- `Dev_Plan/2026-04-13-Qwen3-ASR-Migration-Plan.md`
- `docs/runbooks/weekly/archive/2026-04-13-daily-worklog.md`
- `docs/runbooks/2026-04-13-qwen3-asr-deployment.md`
- `docs/runbooks/weekly/archive/2026-04-13-qwen3-asr-e2e-mix-report.md`
- https://huggingface.co/Qwen/Qwen3-ASR-0.6B
- https://github.com/QwenLM/Qwen3-ASR
- https://github.com/k2-fsa/sherpa-onnx/issues/3110
- https://huggingface.co/csukuangfj/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25

---

## Update — 2026-04-29

**Effective**: 2026-04-29
**Authority**: User direction 2026-04-28 + `docs/archived/dev-plan/2026-04-28-Plan-v21-v27-Update-Plan-v4.md` (Gate 1 final-approval)
**Decision dispositions**:

- Decision 1 (Adopt Qwen3-ASR-0.6B INT8 as primary STT, Sherpa-ONNX runtime): **ACTIVE**
- Decision 2 (Use `OfflineRecognizer.from_qwen3_asr(...)` API): **ACTIVE**
- Decision 3 (Preserve existing SenseVoice path in `models/stt_runner.py` for immediate rollback): **SUPERSEDED**

**Replacement rationale for Decision 3**: User-directed STT consolidation (2026-04-28: "sensevoice 삭제할 것 사용안함"). Qwen3-ASR is the sole supported STT engine; the SenseVoice fallback path is removed from `models/stt_runner.py`. Rollback is now via version-control history rather than runtime fallback.

**Downstream impact** (per Plan v4 §6 + §7):
- `models/stt_runner.py` SenseVoice code path removed (Phase C — Codex `feature` role).
- Tests dual-path SenseVoice fixtures retired; assertions converted to Qwen3-ASR-only (Phase D — Codex `test` role).
- ADR 0016 SenseVoice memory entry marked historical-only (separate Update).
- ADR 0024 alias map continues to apply pending Qwen3-ASR misrecognition verification (separate Update).

This Update modifies disposition only. The original Decision body above remains immutable per the ADR immutability rule.
