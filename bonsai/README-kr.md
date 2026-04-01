# Bonsai — 로컬에서 돌리는 소형 AI 언어 모델

- [개요](#개요)
- [학습 자료](#학습-자료)
- [Bonsai 란?](#bonsai-란)
- [1-bit 양자화](#1-bit-양자화)
- [모델 크기와 메모리](#모델-크기와-메모리)
- [아키텍처](#아키텍처)
  - [추론 엔진](#추론-엔진)
  - [실행 방식](#실행-방식)
- [Mac 에서의 성능](#mac-에서의-성능)
  - [왜 Mac 이 유리한가](#왜-mac-이-유리한가)
  - [예상 성능](#예상-성능)
- [EC2 CPU 에서의 실행](#ec2-cpu-에서의-실행)
- [설치와 실행](#설치와-실행)
  - [Mac 에서 설치](#mac-에서-설치)
  - [Open WebUI 로 실행](#open-webui-로-실행)
- [Claude Code 와 연결 가능한가?](#claude-code-와-연결-가능한가)
- [로컬 LLM 을 코딩 어시스턴트로 쓰는 방법](#로컬-llm-을-코딩-어시스턴트로-쓰는-방법)
- [클라우드 API vs 로컬 모델 비교](#클라우드-api-vs-로컬-모델-비교)

---

# 개요

Bonsai 는 Prism ML 이 개발한 소형 언어 모델 패밀리이다. **1-bit 양자화**라는 극단적 압축 기술로 모델 크기를 줄여서, 클라우드 서버 없이 **내 노트북에서 직접 AI 언어 모델을 실행**할 수 있다.

큰 나무를 작은 화분에서 키우는 분재(Bonsai)처럼, 큰 AI 모델의 능력을 작은 크기로 압축한다는 의미에서 이 이름을 붙였다.

# 학습 자료

- [Bonsai Demo | github](https://github.com/PrismML-Eng/Bonsai-demo)
- [Bonsai 8B 모델 (GGUF) | HuggingFace](https://huggingface.co/prism-ml/Bonsai-8B-gguf)
- [Bonsai 8B 모델 (MLX) | HuggingFace](https://huggingface.co/prism-ml/Bonsai-8B-mlx-1bit)
- [llama.cpp PrismML Fork | github](https://github.com/PrismML-Eng/llama.cpp)

---

# Bonsai 란?

ChatGPT, Claude 같은 대형 언어 모델(LLM)은 수천억 개의 파라미터를 갖고 있어서 강력한 GPU 클러스터가 있는 클라우드에서만 실행할 수 있다. API 호출 비용이 들고, 데이터가 외부 서버로 전송된다.

Bonsai 는 이 문제를 해결한다:

| 기존 방식 (클라우드 LLM) | Bonsai (로컬 LLM) |
|--------------------------|-------------------|
| 클라우드 서버에서 실행 | **내 컴퓨터에서 실행** |
| 월 비용 발생 | 무료 (전기세만) |
| 데이터가 외부로 전송 | **데이터가 로컬에서만 처리** (프라이버시) |
| 인터넷 필요 | **오프라인에서도 동작** |
| 수백 GB 모델 | **2.5GB 로 실행 가능** |

---

# 1-bit 양자화

AI 모델의 가중치(weight)는 원래 32비트 또는 16비트 부동소수점 숫자이다. **양자화(Quantization)** 는 이 숫자의 비트 수를 줄여 모델 크기를 압축하는 기법이다.

Bonsai 는 **1-bit 양자화**를 사용한다. 가중치를 0 또는 1 로 극단적으로 줄인다.

```
일반 모델:    가중치 하나 = 16비트 (0.3421875 같은 실수)
8-bit 모델:   가중치 하나 = 8비트  (정수로 근사)
4-bit 모델:   가중치 하나 = 4비트  (더 거칠게 근사)
1-bit 모델:   가중치 하나 = 1비트  (0 또는 1)
```

1-bit 으로 줄이면:

- **모델 크기**: 약 1/16 로 줄어든다 (16비트 대비)
- **연산 속도**: 곱셈 대신 덧셈/뺄셈으로 대체 가능 → 빠르다
- **정확도**: 떨어질 수 있다 → Bonsai 는 이 손실을 최소화하는 것이 핵심 기술

---

# 모델 크기와 메모리

Bonsai 는 세 가지 크기로 제공된다.

| 모델 | 파라미터 수 | 메모리 사용량 (8K 컨텍스트) | 메모리 사용량 (32K) | 메모리 사용량 (64K) |
|------|-----------|--------------------------|--------------------|--------------------|
| **Bonsai-8B** | 80억 개 | ~2.5GB | ~5.9GB | ~10.5GB |
| **Bonsai-4B** | 40억 개 | 더 작음 | - | - |
| **Bonsai-1.7B** | 17억 개 | 매우 작음 | - | - |

> 8B 모델이 2.5GB 밖에 안 되는 것은 1-bit 양자화 덕분이다. 일반 16비트 8B 모델은 ~16GB 가 필요하다.

---

# 아키텍처

## 추론 엔진

Bonsai 모델을 실행하려면 추론 엔진(Inference Engine)이 필요하다. 두 가지를 지원한다.

| 엔진 | 포맷 | 플랫폼 | 특징 |
|------|------|--------|------|
| **llama.cpp** | GGUF | Mac (Metal), Linux/Windows (CUDA), CPU | C/C++ 기반. PrismML 의 fork 에 1-bit 전용 커널 포함 |
| **MLX** | MLX | **Mac 전용** (Apple Silicon) | Apple Neural Engine 최적화. Mac 에서 최고 성능 |

## 실행 방식

```
사용자
  │
  ├─ 방법 1: 터미널에서 직접 대화 (scripts/run_llama.sh)
  │
  ├─ 방법 2: REST API 서버 (scripts/start_llama_server.sh → localhost:8080)
  │           OpenAI 호환 API → 다른 도구에서 호출 가능
  │
  └─ 방법 3: 웹 UI (scripts/start_openwebui.sh → localhost:9090)
                     ChatGPT 같은 대화 인터페이스
  │
  ▼
추론 엔진 (llama.cpp 또는 MLX)
  │
  ▼
Bonsai 모델 (HuggingFace 에서 다운로드)
```

---

# Mac 에서의 성능

## 왜 Mac 이 유리한가

일반 PC 는 CPU 메모리(RAM)와 GPU 메모리(VRAM)가 분리되어 있다. 모델을 GPU 로 보내려면 데이터를 복사해야 하고, 이것이 병목이 된다.

Apple Silicon Mac 은 **통합 메모리(Unified Memory)** 구조이다. CPU, GPU, Neural Engine 이 같은 메모리를 공유하므로 복사가 필요 없다.

```
일반 PC:
  CPU 메모리 (RAM) ←── 복사 ──→ GPU 메모리 (VRAM)
  느린 데이터 전송이 병목

Mac (Apple Silicon):
  CPU + GPU + Neural Engine → 통합 메모리 (같은 메모리 공유)
  복사 없음 → 빠름
```

1-bit 모델은 연산이 단순하고 **메모리 대역폭**이 성능을 좌우하는데, Apple Silicon 의 높은 메모리 대역폭이 이 특성과 잘 맞는다.

## 예상 성능

| Mac | 메모리 | Bonsai-8B | Bonsai-1.7B |
|-----|-------|-----------|-------------|
| M1 (8GB) | 8GB | 컨텍스트 제한적, ~15 tok/s | ~40 tok/s |
| M2 Pro (16GB) | 16GB | ~25 tok/s | ~60 tok/s |
| M3 Pro (18GB) | 18GB | ~30 tok/s | ~70 tok/s |
| M4 Pro (24GB) | 24GB | ~35 tok/s | ~80 tok/s |

> 20 tokens/sec 이상이면 실시간 대화가 자연스럽다. 사람이 읽는 속도보다 빠르다.

---

# EC2 CPU 에서의 실행

llama.cpp 는 CPU 만으로도 동작하므로 EC2 에서 GPU 없이 실행할 수 있다.

### 권장 인스턴스

| 모델 | 최소 메모리 | 권장 인스턴스 | 시간당 비용 (us-east-1) |
|------|-----------|-------------|----------------------|
| **Bonsai-1.7B** | 2GB+ | `c6i.xlarge` (4 vCPU, 8GB) | ~$0.17 |
| **Bonsai-4B** | 4GB+ | `c6i.2xlarge` (8 vCPU, 16GB) | ~$0.34 |
| **Bonsai-8B** | 6GB+ | `c6i.4xlarge` (16 vCPU, 32GB) | ~$0.68 |

### 성능 비교

```
EC2 c6i.4xlarge (CPU only):   ~5-10 tok/s   타이핑보다 느림
EC2 g5.xlarge (A10G GPU):     ~50 tok/s     쾌적
MacBook M3 Pro (MLX):         ~30 tok/s     쾌적
MacBook M4 Pro (MLX):         ~35 tok/s     쾌적
```

CPU 에서 동작은 하지만 느리다. **배치 처리나 비동기 API 용도**로는 쓸 만하고, 실시간 대화는 GPU 또는 Mac 을 권장한다.

---

# 설치와 실행

## Mac 에서 설치

```bash
# 클론
git clone https://github.com/PrismML-Eng/Bonsai-demo.git
cd Bonsai-demo

# 모델 선택 (8B, 4B, 1.7B 중 택 1)
export BONSAI_MODEL=8B

# 설치 (한 줄로 끝 — Python 환경, 모델 다운로드, 엔진 빌드 모두 자동)
./setup.sh
```

setup.sh 이 자동으로 수행하는 작업:
- Python 3.11 가상 환경 생성
- HuggingFace 에서 GGUF + MLX 모델 다운로드
- llama.cpp 바이너리 다운로드
- MLX 소스 빌드 (Apple Silicon, 2~5분 소요)

## Open WebUI 로 실행

```bash
# Open WebUI 설치 (최초 1회)
source .venv/bin/activate
uv pip install open-webui

# 실행 (llama-server + MLX server + Open WebUI 모두 자동 시작)
bash scripts/start_openwebui.sh
```

실행되면 세 개의 서버가 올라간다:

| 서버 | 포트 | 용도 |
|------|------|------|
| llama-server (GGUF) | 8080 | OpenAI 호환 API |
| MLX server | 8081 | Apple Silicon 최적 API |
| **Open WebUI** | **9090** | **ChatGPT 같은 웹 대화 인터페이스** |

브라우저에서 **http://localhost:9090** 에 접속하면 된다. 인증 없이 바로 사용 가능하다.

### 기타 실행 방법

```bash
# 터미널에서 바로 대화 (llama.cpp)
bash scripts/run_llama.sh

# 터미널에서 바로 대화 (MLX, Mac 전용)
bash scripts/run_mlx.sh

# API 서버만 시작
bash scripts/start_llama_server.sh
```

---

# Claude Code 와 연결 가능한가?

**직접 연결은 안 된다.** Claude Code 는 Anthropic API 전용이고, Bonsai 는 OpenAI 호환 API 를 제공한다. 프로토콜이 다르다.

```
Claude Code → Anthropic API (Claude 모델 전용)
Bonsai      → OpenAI 호환 API (llama.cpp 서버)

→ 프로토콜이 다르므로 직접 연결 불가
```

---

# 로컬 LLM 을 코딩 어시스턴트로 쓰는 방법

Bonsai 같은 로컬 모델을 코딩에 쓰고 싶다면 OpenAI 호환 API 를 지원하는 도구를 사용해야 한다.

| 도구 | 로컬 모델 지원 | 유형 | 설명 |
|------|--------------|------|------|
| **aider** | O | CLI | `--openai-api-base http://localhost:8080` 로 연결 |
| **Continue.dev** | O | VS Code 확장 | 로컬 llama.cpp 서버 연결 가능 |
| **OpenCode** | O | CLI | OpenAI 호환 엔드포인트 지정 가능 |
| **Cursor** | O | IDE | 로컬 모델 설정 가능 |
| **Claude Code** | X | CLI | Anthropic API 전용 |

### 현실적인 성능 비교

```
코딩 어시스턴트 성능 (대략적 비교):

Claude Opus (수천억 파라미터, 클라우드)  ████████████████████  최상
Claude Sonnet                          ██████████████████    상
GPT-4                                  █████████████████     상
Bonsai 8B (80억 파라미터, 로컬)         ██████                중하
Bonsai 1.7B                            ███                   하
```

8B 모델은 간단한 코드 생성은 가능하지만, 복잡한 리팩토링이나 대규모 코드베이스 이해는 어렵다.

---

# 클라우드 API vs 로컬 모델 비교

| 항목 | 클라우드 API (Claude, GPT) | 로컬 모델 (Bonsai) |
|------|--------------------------|-------------------|
| **성능** | 압도적으로 우수 | 간단한 작업에 적합 |
| **비용** | 사용량에 따라 과금 | 무료 (하드웨어 비용만) |
| **프라이버시** | 데이터가 외부 서버로 전송 | 데이터가 로컬에서만 처리 |
| **인터넷** | 필수 | 불필요 (오프라인 가능) |
| **설치** | 없음 (API 키만 있으면 됨) | 모델 다운로드 + 엔진 설치 필요 |
| **지연 시간** | 네트워크 왕복 시간 포함 | 네트워크 지연 없음 |

로컬 모델이 유리한 경우:
- **인터넷이 안 되는 환경** (비행기, 보안 네트워크)
- **코드를 외부로 보내면 안 되는 보안 요건** (금융, 군사, 의료)
- **API 비용을 절약하고 싶을 때** (대량 처리)
- **실험/학습 용도** (AI 모델이 어떻게 동작하는지 직접 만져보기)

그 외에는 Claude, GPT 등 클라우드 API 를 쓰는 것이 압도적으로 생산성이 높다.
