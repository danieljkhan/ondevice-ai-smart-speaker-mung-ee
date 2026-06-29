# ADR-0046: TTS Voice Selection — F2 Production + F1 Backup

- **Status**: Accepted
- **Date**: 2026-04-08
- **Context**: Supertonic-2 TTS voice selection for Mungi production

## Problem

뭉이의 프로덕션 TTS 음색을 결정해야 함. Supertonic-2는 10개 프리셋(F1~F5, M1~M5)을 제공.

### 평가 과정

1. **블렌딩 실험** (ADR-0043): F2+M4, F2+F5 등 7종 블렌딩 WAV 생성
   - 결과: 남녀 간 블렌딩(F2+M4)은 포먼트 충돌로 부자연스러움
   - 결론: 블렌딩 불채택, 프리셋 음색 직접 사용

2. **프리셋 청취 평가**: F1~F5 × 한국어/영어 = 10개 WAV 비교
   - F2: 밝고 쾌활, 아동 친화적 톤 → **채택**
   - F1: 명확하고 안정적 → **예비**

3. **영어 발음 교정**: "Mung-i" → TTS가 "멍-아이"로 발음
   - 10개 철자 후보 A/B 테스트 (F1, F2 각각)
   - **"Moong-ee"** 채택 (두 음색 모두 "뭉이"에 가장 근접)

## Decision

| 항목 | 설정 |
|------|------|
| 프로덕션 음색 | **F2** (`model_manager.py`: `tts_voice_style="F2"`) |
| 예비 음색 | F1 (`--voice-style F1`으로 전환) |
| 영어 발음 교정 | `Mung-i → Moong-ee` (`tts_runner.py`: `normalize_tts_text()`) |
| 감정 표현 | 텍스트 레벨 (LLM이 감탄사/의성어 생성) |

### 감정 파라미터 조사 결과

Supertonic-2 synthesize() API:
```
synthesize(text, voice_style, total_steps=5, speed=1.05,
           max_chunk_length=None, silence_duration=0.3, lang='en')
```

- 감정(emotion/expression) 전용 파라미터 없음
- voice_style 내부 임베딩(ttl, dp)에 감정 정보가 암묵적으로 인코딩
- 직접 감정 제어 불가 → 텍스트 레벨로 대응

## Consequences

- 프로덕션 음색이 F1→F2로 변경됨 — 밝고 쾌활한 톤으로 아동 UX 개선
- 영어에서 "뭉이" 발음이 자연스러워짐
- 예비 음색 F1으로 즉시 전환 가능 (CLI 인자 변경만으로)

## Related

- ADR-0043: Supertonic Voice Blending Experiment
- `core/model_manager.py`: ManagerConfig.tts_voice_style
- `models/tts_runner.py`: normalize_tts_text() 발음 치환
