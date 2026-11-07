# Project03: 공부 집중 도우미 (Study Focus Assistant)

## 개요
책상 위 카메라와 마이크로 공부하는 학생을 지켜보는 Vision-LLM 시청각 멀티모달 시스템입니다.
YOLO로 **휴대폰 사용**과 **자리비움**을, 마이크로 **주변 소음**을 감지하면 Gemma가 상황에 맞는 한마디를 만들어 음성으로 말합니다.
화면에서 `s` 키를 누르고 질문하면, 현재 책상 상황과 집중 통계를 바탕으로 Gemma가 음성으로 대답합니다.

| 이벤트 | 감지 조건 | 입력 | 응답 |
|---|---|---|---|
| `phone` 딴짓 | 휴대폰이 3초 이상 계속 보임 | 카메라 (YOLO) | 다시 집중하라는 한마디 |
| `absent` 자리비움 | 사람이 10초 이상 안 보임 | 카메라 (YOLO) | 자리비움 알림 |
| `noise` 소음 | 소음 기준 초과가 2번 연속 (5초마다 측정) | 마이크 (음량) | 조용한 환경을 권하는 한마디 |
| `question` 질문 | 화면에서 `s` 키 | 마이크 (STT) | 질문에 대한 답변 |

## 발표 자료와 영상
| 파일 | 내용 |
|---|---|
| [`presentation/project03_presentation.pdf`](presentation/project03_presentation.pdf) | 발표 자료 (5쪽) |
| [`presentation/demo.mp4`](presentation/demo.mp4) | 데모 영상 (37초, 자막 포함): 공개 스톡 영상 4개(공부 → 휴대폰 → 빈 책상 → 공부)를 이어 붙인 테스트 영상에 실제 탐지 화면과 Gemma 음성 응답을 합침 |
| [`presentation/intro_film.mp4`](presentation/intro_film.mp4) | 프로젝트 소개 영상 (24초) |

## 구조

```
 ┌──────────── vision.py ────────────┐            ┌──────────────── assistant.py ────────────────┐
 │ USB 카메라 → YOLO11n INT8 엔진     │            │ state.json 감시 (event id가 바뀌면 새 이벤트) │
 │ → focus_tracker (phone / absent)   │ state.json │ 쉬는 동안 소음 측정 (noise)                   │
 │ → s 키 (question)                  │ ─────────▶ │ → 상황 텍스트 + 요청 → Gemma 4 E2B            │
 │ → 화면 (BBox, FPS, 집중 통계)       │            │ → Piper TTS (질문은 Whisper STT 먼저)          │
 └────────────────────────────────────┘            └──────────────────────────────────────────────┘
```

| 파일 | 내용 | 수업 출처 |
|---|---|---|
| `main.py` | 실행 진입점: `assistant.py`와 `vision.py`를 별도 프로세스로 함께 실행, Vision 종료 시 Assistant도 종료 | 06 A-7) 파이프라인 분리 |
| `vision.py` | 카메라 + YOLO 탐지, 이벤트 판정, `state/state.json` 저장, 화면 출력 | 04 F. FPS (EMA), 04 TensorRT INT8, 06 A-2) YOLO → JSON, A-7) 파이프라인 분리 |
| `focus_tracker.py` | 폰 사용·자리비움 판정과 집중 시간 통계 | 새로 설계 |
| `assistant.py` | `state.json` 감시, 소음 이벤트, Gemma 프롬프트, 음성 응답 | 05 D. Prompt Engineering, F. Gemma, 06 A-4)·A-5)·A-7) |
| `voice.py` | STT (Whisper.cpp), TTS (Piper), 소음 측정 (dBFS). 단독 실행 시 소음 측정 모드 | 06 C. STT, TTS |

## 실행 환경
- 기기: NVIDIA Jetson Orin Nano (JetPack 6.2) / Python 3.10
- 레포 루트 `.venv` (수업에서 설치한 패키지 그대로): `ultralytics`, `opencv-python` (JetPack), `numpy==1.23.0`, `llama-cpp-python` (CUDA 빌드)
- 수업에서 설치한 외부 도구: `whisper.cpp` (레포 루트, `build-cpu`), Piper (`.piper_venv`), `arecord`·`aplay` (ALSA)
- 추가 설치 없음. 소음 측정은 표준 라이브러리 `wave`와 `numpy`만 사용합니다.

### 필요한 모델 파일 (git에 올리지 않음)
| 파일 | 만드는 방법 |
|---|---|
| `src/models/YOLO/yolo11n_int8.engine` | `04_DL-Object-Detection.ipynb` H·양자화 섹션 (기기마다 다시 빌드) |
| `src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf` | `05-LLM_and_Gemma.ipynb` F-1) |
| `whisper.cpp/models/ggml-base.bin` | `06` C-1) `download-ggml-model.sh base` |
| `src/models/Piper/ko_KR-kss-medium.onnx` | `06` C-2) `piper.download_voices` |

모든 경로는 `Path(__file__)` 기준으로 레포 루트에서 찾으므로 어느 폴더에서 실행해도 됩니다.

> **폴더 독립 실행 규칙의 예외:** Project01은 모델을 폴더 안에 넣었지만, 이 프로젝트는 레포 루트의 `src/models/`, `whisper.cpp/`, `.piper_venv/`를 사용합니다.
> Gemma gguf는 수 GB이고, TensorRT 엔진은 기기마다 빌드해야 하며, whisper.cpp는 컴파일된 도구, Piper는 별도 가상환경이라 폴더 안에 복사하거나 커밋할 수 없습니다.
> 수업 자료가 모두 레포 루트에 설치하도록 안내하므로 그 위치를 그대로 씁니다.

## 실행 방법

```bash
cd ~/vision-llm
source .venv/bin/activate

# 0) (처음 한 번) 마이크 번호 확인 후 voice.py의 MIC_DEVICE 수정
arecord -l

# 1) (처음 한 번) 소음 기준 정하기: 조용할 때와 떠들 때 값을 보고 assistant.py의 NOISE_THRESHOLD_DB 수정
python3 Project03/voice.py

# 2) 실행 (Assistant와 Vision을 함께 켬)
python3 Project03/main.py
```

디버깅할 때는 두 터미널에서 따로 실행할 수 있습니다: `python3 Project03/assistant.py`, `python3 Project03/vision.py`

| 키 (카메라 창) | 동작 |
|---|---|
| `s` | 음성 질문 ("네, 말씀하세요." 뒤 5초 녹음) |
| `q` | 종료 (`main.py`로 실행했으면 Assistant도 함께 종료) |

## 구현 설명

### 1. 프로세스 분리 (06 A-7)
YOLO와 Gemma를 한 프로그램에서 돌리면, Gemma 추론과 5초 녹음 동안 카메라 화면이 멈춥니다.
그래서 수업처럼 두 프로세스로 나누고 JSON 파일로 연결했습니다. `vision.py`는 임시 파일에 다 쓴 뒤 `os.replace`로 교체하므로, `assistant.py`가 쓰다 만 파일을 읽지 않습니다.

### 2. 이벤트 번호 (`event_id`)
`state.json`은 0.5초마다 갱신됩니다. 그래서 수업 코드처럼 파일 수정 시각만 보면 이벤트와 일반 갱신을 구분할 수 없습니다.
`vision.py`는 이벤트마다 번호를 1씩 올리고, `assistant.py`는 번호가 바뀌었을 때만 응답합니다.
또 10초(`EVENT_MAX_AGE`)보다 오래된 이벤트는 이미 지난 상황이므로 말하지 않습니다.

### 3. 이벤트 판정 (`focus_tracker.py`)
- **시간 기준:** 프레임 수가 아니라 초 단위로 판정하므로, FPS가 변해도 "3초"의 의미가 같습니다.
- **YOLO 놓침 허용:** 폰을 들고 있어도 YOLO가 몇 프레임씩 놓칩니다. 0.5초(`MISS_GRACE_SECONDS`)보다 짧게 끊긴 것은 계속 보인 것으로 칩니다.
- **한 번만 경고:** 폰을 내려놓거나 자리에 돌아올 때까지 같은 경고를 반복하지 않습니다.
- **집중 시간:** 사람이 보이고 폰 구간이 아닌 시간만 더합니다.

### 4. 소음 측정 (`voice.py`)
1초 녹음한 16-bit 샘플의 RMS(제곱 평균의 제곱근)로 평균 음량을 구하고, `20·log10(RMS / 32768)`로 dBFS를 계산합니다.
값은 0 dBFS가 최대이고, 조용할수록 큰 음수입니다.
- **오탐 방지:** 순간적인 소리(기침, 책 덮는 소리)에 반응하지 않도록 2번 연속 기준을 넘어야 이벤트로 봅니다.
- **자기 소리 제외:** 스피커로 말한 직후 3초는 자기 목소리를 재지 않도록 측정을 쉽니다.
- **마이크 공유:** STT와 소음 측정은 같은 프로세스에서 순서대로 마이크를 쓰므로 장치 충돌이 없습니다.

### 5. 프롬프트 (05 D, 06 A-5)
수업에서 배운 형식으로 프롬프트를 나눴습니다.
- **system:** `Instruction / Constraint / Output Format` 형식. 역할은 공부 도우미, 규칙은 "Context에 없는 것은 추측 금지", 출력은 두 문장 이내입니다.
- **user:** `Context`(탐지 물체 + 집중 통계를 한국어 문장으로 바꾼 것)와 이벤트별 요청 문장입니다.
- **클래스 이름:** 한국어로 바꿔 넣어 TTS가 자연스럽게 읽게 했습니다.

## 참고 사항
- **이벤트 손실:** `state.json`에는 마지막 이벤트 하나만 저장됩니다. Gemma가 응답하는 몇 초 사이에 이벤트가 두 번 생기면 앞의 것은 건너뜁니다. 공부 도우미에서는 최신 상황에만 반응하면 되므로 의도한 동작입니다.
- **자리비움 알림:** 자리를 비운 사람에게 말하는 것이므로, 돌아왔을 때 들리도록 짧게 말합니다.
- **질문 타이밍:** `s`를 눌렀을 때 Assistant가 소음을 측정하는 중이면 최대 1초 늦게 시작합니다. "네, 말씀하세요."가 들린 뒤 말해야 합니다.
- **소음 기준값:** 환경마다 다르므로 발표장에서 `python3 Project03/voice.py`로 다시 맞춰야 합니다.
- **생성 파일:** 실행 중 `state/`, `audio/` 폴더가 생깁니다. 둘 다 `.gitignore`에 있습니다.
