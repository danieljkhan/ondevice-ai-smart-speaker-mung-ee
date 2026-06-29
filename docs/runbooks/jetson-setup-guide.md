# Jetson Orin Nano Super — 뭉이 전체 셋업 가이드

> **기준 장비:** Jetson Orin Nano Super 8GB / JetPack 6.2 / Ubuntu 22.04 / CUDA 12.6
> **기준 코드:** 2026-04-29 `dev` 기준 (Plan v2.1+v2.7 Update Plan v4 user-approved)
> **활성 모델:** Gemma 4 E2B Q5_K_M (LLM primary, per ADR 0073) / Qwen3.5-2B-DPO Q6_K (LLM fallback) / Qwen3-ASR (STT, per ADR 0055 + 2026-04-29 Update) / Supertonic 2 (sole TTS engine, per ADR 0088) / Silero VAD / koen-e5-tiny ONNX (대화 메모리 RAG)

---

## 목차

1. [사전 준비](#1-사전-준비)
2. [시스템 패키지 설치](#2-시스템-패키지-설치)
3. [디렉토리 구조 생성](#3-디렉토리-구조-생성)
4. [Python 가상 환경](#4-python-가상-환경)
5. [공통 패키지 설치](#5-공통-패키지-설치)
6. [Jetson 전용 패키지 설치](#6-jetson-전용-패키지-설치)
7. [AI 모델 다운로드](#7-ai-모델-다운로드)
8. [Git 개발 워크플로 설정](#8-git-개발-워크플로-설정)
9. [셸 헬퍼 등록 (mungidev / mungiup)](#9-셸-헬퍼-등록)
10. [CUDA 메모리 관리 — sudoers 설정](#10-cuda-메모리-관리--sudoers-설정)
11. [환경 검증](#11-환경-검증)
12. [IDE 설치 (선택)](#12-ide-설치-선택)
13. [오프라인 백업](#13-오프라인-백업)
14. [트러블슈팅](#14-트러블슈팅)
15. [전체 체크리스트](#15-전체-체크리스트)

---

## 1. 사전 준비

### 1-1. JetPack 6.2 플래싱

Jetson Orin Nano Super에 JetPack 6.2가 설치되어 있어야 합니다.

- NVIDIA SDK Manager (Ubuntu PC 필요) 또는 SD 카드 이미지로 플래싱
- 플래싱 후 Ubuntu 22.04 초기 설정 (사용자명, 비밀번호, WiFi)
- 참고: https://developer.nvidia.com/embedded/jetpack

### 1-2. 네트워크 연결 확인

```bash
# WiFi 또는 이더넷 연결 후 확인
ping -c 3 google.com
# "3 packets transmitted, 3 received" 나오면 성공
```

### 1-3. SSH 설정 (PC에서 원격 작업 시)

```bash
# Jetson에서 — SSH 서버 확인
sudo systemctl status ssh

# PC에서 — Jetson 접속
ssh <사용자명>@<Jetson_IP>
# 예: ssh mungi@192.168.0.20
```

---

## 2. 시스템 패키지 설치

### 2-1. 패키지 목록 업데이트

```bash
sudo apt update
```

### 2-2. 시스템 패키지 일괄 설치

```bash
sudo apt install -y python3-pip python3-venv git cmake build-essential libasound2-dev portaudio19-dev i2c-tools avahi-daemon git-lfs
```

| 패키지 | 용도 |
|--------|------|
| `python3-pip` | pip 패키지 관리자 |
| `python3-venv` | Python 가상 환경 |
| `git` | 버전 관리 |
| `cmake` | C/C++ 빌드 (llama.cpp 등) |
| `build-essential` | gcc/g++ 컴파일러 |
| `libasound2-dev` | ALSA 오디오 라이브러리 |
| `portaudio19-dev` | sounddevice 의존성 |
| `i2c-tools` | 배터리 BMS 통신 |
| `avahi-daemon` | mungi.local mDNS |
| `git-lfs` | 대용량 파일 Git 지원 |

### 2-3. Git LFS 초기화

```bash
git lfs install
# "Git LFS initialized." → 성공
```

### 2-4. 선택 패키지

```bash
sudo apt install -y tree
sudo pip3 install jetson-stats --break-system-packages
# jtop 명령어로 GPU/메모리/온도 모니터링
```

---

## 3. 디렉토리 구조 생성

### 3-1. 런타임 디렉토리

```bash
# 메인 디렉토리
sudo mkdir -p /opt/mungi
sudo chown -R $(whoami):$(whoami) /opt/mungi

# 하위 구조
mkdir -p /opt/mungi/{ai_models,core,models,safety,hardware,scripts,tests,docs/adr}
```

### 3-2. 런타임 데이터 디렉토리

```bash
# 설정 파일
sudo mkdir -p /var/lib/mungi/config
# 대화 기록
sudo mkdir -p /var/lib/mungi/conversations
# 로그
sudo mkdir -p /var/log/mungi

sudo chown -R $(whoami):$(whoami) /var/lib/mungi
sudo chown -R $(whoami):$(whoami) /var/log/mungi
```

### 3-3. 확인

```bash
tree -L 2 /opt/mungi
ls -la /var/lib/mungi/
```

---

## 4. Python 가상 환경

```bash
cd /opt/mungi
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

> 새 터미널마다 `source /opt/mungi/.venv/bin/activate` 필요 (9단계에서 자동화)

---

## 5. 공통 패키지 설치

```bash
cd /opt/mungi
pip install -r requirements-core.txt
```

`requirements-core.txt` 내용 (PC와 Jetson 공통):

| 패키지 | 용도 |
|--------|------|
| sounddevice, soundfile, numpy | 오디오 I/O |
| supertonic | TTS 엔진 |
| ~~openwakeword~~ | ~~호출어 감지~~ — 폐기 (ADR-NEW-3, 2026-04-29 Update; VAD-driven entry로 대체) |
| flask, flask-httpauth, flask-wtf, flask-limiter, bcrypt, waitress | 보호자 대시보드 |
| structlog | 구조화 로깅 |
| pydantic | 설정 검증 |

설치 실패 시 개별 확인:

```bash
pip install flask-httpauth flask-wtf flask-limiter soundfile supertonic
```

---

## 6. Jetson 전용 패키지 설치

> 이 단계가 가장 까다롭습니다. 일반 pip이 아닌 **NVIDIA Jetson AI Lab 전용 저장소**를 사용합니다.

### 6-1. PyTorch 2.10.0

```bash
pip install torch==2.10.0 --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
```

#### libcudss.so.0 에러 해결 (PyTorch 2.8+ 공통)

```bash
# 1) 누락 라이브러리 설치
pip install nvidia-cudss-cu12

# 2) 충돌 의존성 제거
pip uninstall -y nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cusparse-cu12 nvidia-nvjitlink-cu12

# 3) 라이브러리 경로 등록
export LD_LIBRARY_PATH=/opt/mungi/.venv/lib/python3.10/site-packages/nvidia/cu12/lib:$LD_LIBRARY_PATH
```

#### 확인

```bash
python3 -c "import torch; print(f'PyTorch {torch.__version__} / CUDA: {torch.cuda.is_available()}')"
# "CUDA: True" 필수!
```

### 6-2. STT — Qwen3-ASR (per ADR 0055 + 2026-04-29 Update)

> 2026-04-29 업데이트: Faster-Whisper 및 Sherpa-ONNX SenseVoice STT는 폐기. Qwen3-ASR 단일 경로만 사용 (사용자 결정 2026-04-28). 자세한 근거는 ADR 0055 + 2026-04-29 Update 참조.

```bash
pip install sherpa-onnx>=1.12.0
```

> 런타임은 `models/stt_runner.py`의 `_QWEN3_ASR_BUNDLE_PREFIX = "sherpa-onnx-qwen3-asr-"`로 시작하는 디렉터리를 자동 탐색합니다. 모델 다운로드는 §7-2 참조.

### 6-3. onnxruntime-gpu (VAD/TTS GPU 추론)

```bash
pip install onnxruntime-gpu==1.23.0 --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
```

### 6-4. llama-cpp-python (LLM)

**방법 A — Jetson AI Lab 프리빌드 (권장):**

```bash
pip install llama-cpp-python==0.3.14 --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
```

> 현재 프로젝트의 채택안은 `0.3.14`를 유지하되
> `models/llm_runner.py`에서 `seq_id < 0 -> 0` KV cache 호환 패치를 적용하는 것입니다.
> 이 패치는 llama-cpp-python `0.3.16` `_internals.py` 업스트림 수정과 동일한 동작입니다.

**방법 B — 소스 빌드 (프리빌드 실패 시):**

```bash
# 빌드 스크립트 사용 (기본값 0.3.14, CUDA sm_87)
cd /opt/mungi-repo
bash scripts/build_llama_cpp.sh

# 휠 자체에 업스트림 KV cache fix를 포함하려면 명시적으로 0.3.16 빌드
LLAMA_CPP_VERSION=0.3.16 bash scripts/build_llama_cpp.sh
```

### 6-5. 하드웨어 패키지

> 2026-04-29 업데이트: WS2812B LED + RP2040 Pico 하드웨어 폐기 (사용자 결정 2026-04-28, ADR-NEW-1). `pyserial`은 LED 전용 의존성이었으므로 함께 제거. 디스플레이 런타임용 `pygame` / `python-evdev`는 별도 후속 Plan에서 도입 예정 (현재 시점에는 orphan dep 방지를 위해 미설치).

```bash
pip install smbus2
sudo pip3 install Jetson.GPIO
```

### 6-6. 유틸리티

```bash
pip install huggingface_hub[cli]
```

### 6단계 확인

```bash
python3 -c "
import torch, sherpa_onnx, onnxruntime
from llama_cpp import Llama
print(f'PyTorch {torch.__version__} / CUDA: {torch.cuda.is_available()}')
print(f'onnxruntime-gpu {onnxruntime.__version__}')
print(f'sherpa-onnx {sherpa_onnx.__version__} OK (Qwen3-ASR runtime)')
print('llama-cpp-python OK')
"
```

---

## 7. AI 모델 다운로드

모델 저장 위치: `/opt/mungi/ai_models/`

### 7-1. Silero VAD (~2.3 MB)

```bash
curl -L -o /opt/mungi/ai_models/silero_vad.onnx \
  https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx
```

### 7-2. STT 모델 — Qwen3-ASR ONNX bundle (per ADR 0055)

> 2026-04-29 업데이트: Faster-Whisper Small (`Systran/faster-whisper-small`) 다운로드 단계는 제거. Qwen3-ASR ONNX bundle 다운로드 경로는 ADR 0055 References를 따름.

```bash
cd /opt/mungi/ai_models
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('csukuangfj/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25', local_dir='/opt/mungi/ai_models/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25')"
```

> 런타임은 `models/stt_runner.py:36`의 `_QWEN3_ASR_BUNDLE_PREFIX = "sherpa-onnx-qwen3-asr-"`로 시작하는 디렉터리를 자동으로 픽업합니다. ADR 0055 References의 정확한 리포지토리 핀이 변경되면 이 명령의 식별자도 함께 갱신해 주세요.

### 7-3. LLM primary — Gemma 4 E2B Q5_K_M (~2.0–2.5 GB, per ADR 0073)

> 2026-04-29 업데이트: Qwen3-4B-Q4_K_M (이전 활성 LLM) 은 2026-04-05에 Jetson에서 제거됨 (`docs/runbooks/baseline-stack-and-models.md` "Removed models" 섹션). 활성 LLM은 ADR 0073에 따라 Gemma 4 E2B Q5_K_M으로 전환됨. 런타임 기본 경로는 `models/llm_runner.py:92` `DEFAULT_GEMMA4_TEXT_MODEL_PATH`.

```bash
# Gemma 4 E2B Q5_K_M GGUF — 정확한 리포지토리 핀은 ADR 0073 본문 참조. 파일명은
# models/llm_runner.py 의 DEFAULT_GEMMA4_TEXT_MODEL_PATH 와 일치해야 합니다.
ls /opt/mungi/ai_models/gemma-4-E2B-it-Q5_K_M.gguf || echo "MISSING — ADR 0073의 다운로드 절차 따라 설치"
```

### 7-3b. LLM fallback — Qwen3.5-2B-DPO Q6_K (~2.0 GB)

> 폴백 모델은 env/config override (`MUNGI_LLM_BACKEND=qwen3_legacy` 또는 `"llm_backend": "qwen3_legacy"` in `config.json`) 활성화 시 사용. 런타임 기본 경로는 `models/llm_runner.py:91` `DEFAULT_QWEN3_LEGACY_MODEL_PATH`.

```bash
ls /opt/mungi/ai_models/Qwen3.5-2B-DPO.Q6_K.gguf || echo "MISSING — Selection Report v1 + ADR 0073 의 다운로드 절차 따라 설치"
```

### 7-4. Supertonic TTS 2 (~305 MB)

```bash
cd /opt/mungi/ai_models
git clone https://huggingface.co/Supertone/supertonic-2
```

### 7-5. openWakeWord — 폐기 (2026-04-29 Update)

> openWakeWord는 ADR-NEW-3 (proposed slot 0081) 결정으로 폐기되었습니다. 현재 활성 entry path는 VAD-driven (Silero VAD) 단일 경로입니다. 별도 wake-word 모델을 다운로드하지 마십시오.

### 7-6. 대화 메모리 RAG — koen-e5-tiny ONNX (per ADR 0082)

> Exact HuggingFace repo + files follow ADR 0082 References. The conversation-memory FAISS index (`conversation_memory.faiss`) remains physically separate from the retired wiki RAG index (`wiki_faiss.index`, removed by ADR 0085 / PR 4-B) to preserve the CLAUDE.md §6 source-contamination boundary. The conversation-memory prompt token budget is 100; wiki RAG has no active budget after PR 4-B.

```bash
ls -d /opt/mungi/ai_models/koen-e5-tiny-onnx/ || echo "MISSING — ADR 0082 다운로드 절차 따라 설치 (예정)"
```

### 7-6. TTS fallback policy

Supertonic 2 is the sole TTS engine. No in-process TTS fallback package is installed
per ADR 0088.

### 7단계 확인

```bash
echo "=== AI 모델 현황 ==="
ls -lh /opt/mungi/ai_models/silero_vad.onnx
ls -lh /opt/mungi/ai_models/gemma-4-E2B-it-Q5_K_M.gguf  # primary LLM (per ADR 0073)
ls -lh /opt/mungi/ai_models/Qwen3.5-2B-DPO.Q6_K.gguf  # fallback LLM (env/config override)
ls -d /opt/mungi/ai_models/sherpa-onnx-qwen3-asr-*/  # STT bundle (per ADR 0055 + 2026-04-29 Update)
ls -d /opt/mungi/ai_models/supertonic-2/
ls -d /opt/mungi/ai_models/koen-e5-tiny-onnx/  # 대화 메모리 RAG embedding (per ADR-NEW-4)
du -sh /opt/mungi/ai_models/
```

---

## 8. Git 개발 워크플로 설정

### 8-1. SSH 키 등록

```bash
# Jetson에서 SSH 키 생성
ssh-keygen -t ed25519 -C "mungi-jetson"
cat ~/.ssh/id_ed25519.pub
# 출력된 공개키를 GitHub → Settings → SSH keys에 등록

# 연결 확인
ssh -T git@github.com
# "Hi <username>!" 메시지 확인
```

### 8-2. 개발 리포 클론

```bash
cd /opt
sudo git clone git@github.com:OWNER/ondevice-ai-smart-speaker-mung-ee.git mungi-repo
sudo chown -R $(whoami):$(whoami) /opt/mungi-repo
```

### 8-3. 개발 venv 생성

```bash
cd /opt/mungi-repo
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-core.txt
pip install -r requirements-jetson.txt
```

### 8-4. 운영 모델 정리

| 경로 | 역할 | 규칙 |
|------|------|------|
| `/opt/mungi` | 런타임 (모델, 서비스) | 직접 수정 금지 (cutover 전까지) |
| `/opt/mungi-repo` | Git 개발 트리 | 모든 코드 작업은 여기서 |
| `/opt/mungi/ai_models` | AI 모델 저장소 | 코드에서 이 경로 참조 |
| `/var/lib/mungi/config/` | 런타임 설정 | `config.json` 위치 |
| `/var/lib/mungi/conversations/` | 대화 기록 | 자동 저장 |
| `/var/log/mungi/` | 로그 | structlog 출력 |

---

## 9. 셸 헬퍼 등록

`~/.bashrc`에 다음을 추가:

```bash
cat >> ~/.bashrc << 'BASHRC'

# === 뭉이 개발 환경 ===

mungidev() {
    cd /opt/mungi-repo || return 1
    source .venv/bin/activate
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:/opt/mungi-repo/.venv/lib/python3.10/site-packages/nvidia/cu12/lib:${LD_LIBRARY_PATH:-}"
    export CUDA_HOME=/usr/local/cuda
    echo "mungidev: /opt/mungi-repo 활성화 완료"
}

mungiup() {
    cd /opt/mungi-repo || return 1
    git pull --ff-only
    mungidev
}

BASHRC

source ~/.bashrc
```

### LD_LIBRARY_PATH 필수 경로 (3개)

| 경로 | 이유 |
|------|------|
| `/usr/local/cuda/lib64` | CUDA 기본 라이브러리 |
| `/usr/lib/aarch64-linux-gnu` | 시스템 공유 라이브러리 |
| `.venv/.../nvidia/cu12/lib` | nvidia-cudss-cu12 (PyTorch 의존) |

> 이 경로 중 하나라도 빠지면 `CUDAExecutionProvider` 미검출, `torch.cuda.is_available()=False` 등의 문제 발생.
> 근본 원인 및 해결: ADR 0002 참조.

---

## 10. CUDA 메모리 관리 — sudoers 설정

Jetson 8 GB 통합 메모리에서 커널 페이지 캐시와 CUDA `cudaMalloc`이 메모리를 경쟁합니다.
STT→LLM 전환 시 자동으로 페이지 캐시를 드롭하여 GPU 전체 오프로드를 보장합니다.

### sudoers 규칙 생성

```bash
echo "$(whoami) ALL=(root) NOPASSWD: /usr/bin/tee /proc/sys/vm/drop_caches" | \
  sudo tee /etc/sudoers.d/mungi-drop-caches
sudo chmod 440 /etc/sudoers.d/mungi-drop-caches
```

### 동작 확인

```bash
echo 1 | sudo tee /proc/sys/vm/drop_caches
# 비밀번호 없이 실행되면 성공
```

### 메모리 효과

| 항목 | 캐시 드롭 전 | 캐시 드롭 후 |
|------|-------------|-------------|
| LLM GPU 레이어 | 10 (폴백) | **-1 (전체)** |
| E2E 응답 시간 | 20.4s | **12.5s** |

---

## 11. 환경 검증

### 11-1. Phase 0 검증 스크립트

```bash
cd /opt/mungi-repo
mungidev
python -m scripts.phase0_verify --save reports/phase0-verify.json
```

검증 항목:
- Jetson 하드웨어 감지
- PyTorch CUDA 활성화
- onnxruntime CUDAExecutionProvider
- sherpa-onnx (Qwen3-ASR runtime), llama-cpp-python, supertonic import

### 11-2. 런타임 인벤토리 비교

```bash
cd /opt/mungi-repo
bash scripts/inventory_runtime.sh /opt/mungi /opt/mungi-repo
# reports/ 하위에 비교 보고서 생성
```

### 11-3. E2E 파이프라인 테스트

```bash
cd /opt/mungi-repo
mungidev
python scripts/test_e2e_pipeline.py
```

예상 결과 (참고치 — 모델 교체 후 재측정 필요):
- VAD: RTF ~0.32
- STT: RTF (Qwen3-ASR 측정 필요; Faster-Whisper 시점 ~0.28는 historical)
- LLM: tok/s (Gemma 4 E2B 측정 필요; Qwen3-4B 시점 ~11.4 tok/s는 historical)
- TTS: RTF ~0.67
- **E2E**: 모델 교체 후 재측정 필요

### 11-4. 5턴 대화 테스트

```bash
python scripts/test_conversation.py
```

예상: 평균 **3.0s/턴**, 언어 혼입 0건

---

## 12. IDE 설치 (선택)

### 12-1. VSCode

```bash
curl -L -o /tmp/code.deb \
  "https://code.visualstudio.com/sha/download?build=stable&os=linux-deb-arm64"
sudo dpkg -i /tmp/code.deb
# 의존성 에러 시: sudo apt --fix-broken install -y
```

> snap 설치는 ARM64에서 불가. `.deb` 직접 설치 필수.

### 12-2. Claude Code

```bash
curl -fsSL https://claude.ai/install.sh | bash
echo 'export PATH="$HOME/.claude/bin:$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
claude --version
```

---

## 13. 오프라인 백업

환경 검증 완료 후 실행:

```bash
cd /opt/mungi
source .venv/bin/activate

# 버전 고정
pip freeze > requirements.lock

# wheel 캐시
pip download -r requirements.lock -d scripts/wheelhouse/
```

오프라인 복원:

```bash
pip install --no-index --find-links=scripts/wheelhouse/ -r requirements.lock
```

---

## 14. 트러블슈팅

### "libcudss.so.0: cannot open shared object file"

PyTorch 2.8+ 공통 문제. [6-1 libcudss 해결](#6-1-pytorch-2100) 절차 참조.

### CUDAExecutionProvider 미검출

`LD_LIBRARY_PATH`에 3개 경로 모두 포함되어 있는지 확인.
`mungidev` 실행 후 다시 테스트.

```bash
python3 -c "import onnxruntime; print(onnxruntime.get_available_providers())"
# ['CUDAExecutionProvider', 'CPUExecutionProvider'] 나와야 함
```

### CUDA OOM — LLM 로드 실패

페이지 캐시 드롭 후 재시도:

```bash
echo 1 | sudo tee /proc/sys/vm/drop_caches
free -m  # MemFree >= 4000 MB 확인
```

GNOME 데스크탑이 ~500 MB 점유. 서버 모드 전환 시 추가 확보:

```bash
sudo systemctl set-default multi-user.target
sudo reboot
# GUI 복원: sudo systemctl set-default graphical.target
```

### "snap code는 arm64에서 유효하지 않음"

snap 대신 `.deb` ARM64 파일 직접 설치. [12-1 VSCode](#12-1-vscode) 참조.

### HuggingFace 401 Unauthorized

잘못된 토큰 캐시 삭제:

```bash
rm -rf ~/.cache/huggingface/ ~/.huggingface/token ~/.netrc ~/.git-credentials
```

### "패키지를 찾을 수 없습니다" (apt)

`sudo apt update` 먼저 실행.

---

## 15. 전체 체크리스트

```
[1단계] 사전 준비
 [ ] JetPack 6.2 플래싱 완료
 [ ] 네트워크 연결 확인
 [ ] SSH 설정 (원격 작업 시)

[2단계] 시스템 패키지
 [ ] sudo apt update
 [ ] 시스템 패키지 10개 설치
 [ ] git lfs install
 [ ] jetson-stats 설치 (선택)

[3단계] 디렉토리 구조
 [ ] /opt/mungi/ 및 하위 폴더
 [ ] /var/lib/mungi/{config,conversations}
 [ ] /var/log/mungi/

[4단계] Python 환경
 [ ] /opt/mungi/.venv 생성
 [ ] pip 업그레이드

[5단계] 공통 패키지
 [ ] requirements-core.txt 설치

[6단계] Jetson 전용 패키지
 [ ] PyTorch 2.10.0 — CUDA True 확인
 [ ] libcudss.so.0 해결 + LD_LIBRARY_PATH 설정
 [ ] sherpa-onnx (Qwen3-ASR runtime, per ADR 0055)
 [ ] onnxruntime-gpu 1.23.0
 [ ] llama-cpp-python (프리빌드 또는 소스 빌드)
 [ ] smbus2, Jetson.GPIO  (pyserial은 LED 폐기로 제거됨, 2026-04-29)
 [ ] huggingface_hub

[7단계] AI 모델
 [ ] Silero VAD (~2.3 MB)
 [ ] Qwen3-ASR ONNX bundle (per ADR 0055 + 2026-04-29 Update)
 [ ] Gemma 4 E2B Q5_K_M (~2.0–2.5 GB, primary LLM per ADR 0073)
 [ ] Qwen3.5-2B-DPO Q6_K (~2.0 GB, fallback LLM)
 [ ] Supertonic TTS 2 (~305 MB, sole TTS engine per ADR 0088)
 [ ] koen-e5-tiny ONNX (대화 메모리 RAG, per ADR-NEW-4)

[8단계] Git 개발 워크플로
 [ ] SSH 키 생성 + GitHub 등록
 [ ] /opt/mungi-repo 클론
 [ ] 개발 venv 생성 + 패키지 설치

[9단계] 셸 헬퍼
 [ ] mungidev / mungiup → ~/.bashrc 등록
 [ ] LD_LIBRARY_PATH 3개 경로 포함 확인

[10단계] CUDA 메모리
 [ ] /etc/sudoers.d/mungi-drop-caches 생성
 [ ] 비밀번호 없이 캐시 드롭 확인

[11단계] 검증
 [ ] phase0_verify.py — 전항목 통과
 [ ] E2E 파이프라인 테스트 — 모델 교체 후 재측정 (Faster-Whisper+Qwen3-4B 시점 ~12.5s는 historical)
 [ ] 5턴 대화 테스트 — 모델 교체 후 재측정 (이전 ~3.0s/턴은 historical)

[12단계] IDE (선택)
 [ ] VSCode ARM64 .deb
 [ ] Claude Code

[13단계] 오프라인 백업
 [ ] requirements.lock 생성
 [ ] scripts/wheelhouse/ 백업
```

---

## 참고 문서

| 문서 | 위치 |
|------|------|
| 기존 설치 가이드 v2 (실전편) | `docs/archived/dev-plan/Mungi_Jetson_설치가이드_실전편_v2.md` |
| Jetson alignment plan | `docs/runbooks/jetson-alignment-plan.md` |
| 초기 환경 구축 기록 | `docs/runbooks/2026-03-11-jetson-dev-setup-log.md` |
| GitHub↔Jetson 동기화 | `docs/runbooks/github-jetson-sync.md` |
| LLM 업그레이드 워크로그 | `docs/runbooks/weekly/archive/2026-03-17-llm-model-upgrade-worklog.md` |
| LD_LIBRARY_PATH 해결 | `docs/adr/0002-phase0-baseline.md` |
| 순차 GPU 로딩 전략 | `docs/adr/0009-sequential-gpu-loading.md` |
| ~~Qwen3-4B 모델 결정~~ | ~~`docs/adr/0012-llm-upgrade-qwen3-4b.md`~~ — historical (Qwen3-4B 폐기 2026-04-05; 활성 LLM은 Gemma 4 E2B per ADR 0073) |
| LLM primary swap → Gemma 4 E2B | `docs/adr/0073-llm-primary-gemma4-swap.md` |
| STT 엔진 → Qwen3-ASR (SenseVoice 폐기 2026-04-29) | `docs/adr/0055-stt-engine-qwen3-asr-adoption.md` + 2026-04-29 Update |
| 페이지 캐시 드롭 | `docs/adr/0013-cuda-page-cache-drop.md` |
| llama-cpp-python 소스 빌드 | `scripts/build_llama_cpp.sh` |
| Phase 0 검증 스크립트 | `scripts/phase0_verify.py` |
| 런타임 인벤토리 비교 | `scripts/inventory_runtime.sh` |
