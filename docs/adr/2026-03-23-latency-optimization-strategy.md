# [기술 보고서] Mungi 파이프라인 레이턴시 단축 및 LLM 최적화 방안 (llama.cpp 기반)

**작성일**: 2026-03-23
**작성자**: Gemini CLI (Senior Code Reviewer)
**대상 환경**: Jetson Orin Nano Super (8GB, ARM64) + llama.cpp (GGUF)
**현재 이슈**: 사용자 체감 레이턴시 평균 약 10초 발생

---

## 1. 아키텍처 대전환: "순차 로딩(Sequential)"에서 "전면 상주(Resident)"로

현재의 10초 지연 중 가장 큰 병목인 **I/O 지연(모델 로드/언로드 2.5초)**과 **컨텍스트 재계산(Prompt Eval 2.0초)**을 원천 제거하기 위한 아키텍처 변경입니다.

### 1.1. 8GB 통합 메모리 정적 할당(Static Allocation) 설계
실측 데이터 기반 메모리 점유율 분석 결과, 8GB 내에서 전면 상주가 가능합니다.
- **OS 및 기본 시스템:** ~1.5 GB
- **STT (Sherpa-ONNX CPU):** ~0.3 GB
- **TTS (Supertonic CPU):** ~0.24 GB
- **LLM 가중치 (Qwen3-4B Q4_K_M):** ~2.5 GB (GPU Offload)
- **합계:** **약 4.54 GB**
- **여유 마진:** **약 3.4 GB** (KV 캐시 및 PyTorch 런타임 버퍼로 활용 가능)

### 1.2. KV 캐시 (Context Caching) 유지 및 슬라이딩 윈도우
- **문제점:** 매 턴 LLM을 언로드하여 이전 대화의 '기억(KV 캐시)'이 소실됨. 대화가 길어질수록 수천 토큰을 재계산(Prompt Processing)해야 하므로 지연 시간 증가.
- **해결책:** LLM을 상주시키고 KV 캐시를 메모리에 유지. 단, `llama.cpp`의 **컨텍스트 슬라이딩 윈도우(Context Shift)** 기능을 활용하여 `n_ctx=2048` 한도 초과 시 가장 오래된 대화 내역을 밀어내어 OOM(Out of Memory) 방어.

---

## 2. 파이프라인 병렬화: 문장 단위 스트리밍 (Chunked Streaming)

"듣고 -> 생각하고 -> 말한다"는 기존의 선형 구조를 **"생각나는 대로 바로 말하는"** 파이프라인으로 전환합니다.

### 2.1. LLM 텍스트 스트리밍 추론 (Text Streaming)
- `llama-cpp-python`의 `stream=True` 옵션을 적용하여 토큰이 생성되는 즉시 가져오도록 변경. 기존의 100% 생성 대기 시간(약 3초) 해소.

### 2.2. 문장 부호 기반 TTS 핑퐁 (Sentence-level TTS Streaming)
- 생성된 토큰을 버퍼링하다가 구두점(마침표 `.`, 물음표 `?`, 느낌표 `!`, 쉼표 `,`)을 기준으로 텍스트 청크(Chunk)를 분리하여 즉시 TTS 엔진에 전송.
- **효과:** LLM이 첫 문장을 완성하는 즉시 오디오 출력이 시작되므로, **체감 대기 시간(TTFT to Audio)을 1.5초 이하로 급감**시킬 수 있음.

---

## 3. LLM 엔진(llama.cpp) 성능 극대화 방안

### 3.1. 시스템 프롬프트 캐싱 (Prompt State Caching)
- 불변하는 시스템 프롬프트(페르소나, 안전 가이드라인 등)의 처리 상태를 `llama.cpp`의 Save State 기능으로 캐싱.
- 초기 로딩 및 매 턴 응답 시 프롬프트 재연산을 생략하여 **첫 토큰 생성(TTFT) 속도 대폭 향상**.

### 3.2. I-Matrix (IQ) 양자화 포맷 도입 검토
- 기존 K-Quant(Q4_K_M) 대신 최신 **I-Matrix (예: IQ4_XS, IQ4_NL)** 포맷 적용 테스트.
- 동일 메모리 점유율에서 모델의 Perplexity(PPL) 저하를 방어하여 더 지능적이고 자연스러운 응답 유도.

---

## 4. 하드웨어 및 OS 레벨의 숨은 지연 제거

### 4.1. Jetson MAXN 모드 강제 고정
- 클럭 스케일링으로 인한 지연(스파이크) 방지를 위해 시스템 부팅 시 `nvpmodel -m 0` (최대 전력) 및 `jetson_clocks` 강제 적용.

### 4.2. 오디오 ALSA 버퍼 최소화
- 오디오 출력 단계에서의 버퍼링 지연(0.3~0.5초) 최소화. 재생 라이브러리(`sounddevice` 등)의 `blocksize` 축소 및 `latency='low'` 설정 적용.

---

## 5. 단계별 실행 로드맵 (Action Plan)

현재의 안정적인 코드를 보호하기 위해 다음 3단계로 점진적 도입을 권장합니다.

1. **[Phase 1] 스트리밍 텍스트 기반 TTS 병렬화 (최우선 과제)**
   - 메인 루프에서 LLM `stream=True` 적용 및 문장 단위 TTS 청킹 구현. (레이턴시 2~3초 즉시 절감)
2. **[Phase 2] 전면 상주 모드 (Resident Mode) 브랜치 개발**
   - `ModelManager`의 unload/drop_caches 로직 제거 및 정적 메모리 할당(n_ctx 고정). 60라운드 스트레스 테스트 진행. (I/O 지연 2.5초 완전 제거)
3. **[Phase 3] Prompt Caching 및 모델 최적화**
   - 시스템 프롬프트 상태 캐싱 적용 및 IQ 양자화 포맷(I-Matrix) 테스트.
