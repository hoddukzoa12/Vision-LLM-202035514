# Project01 — 손가락 제스처 화이트보드

## 개요
오른손에 든 **초록 펜의 끝**을 Color Segmentation으로 검출하고 칼만 필터로 궤적을 부드럽게 보정해, 카메라 화면 위에 그림을 그리는 프로그램입니다.
**왼손의 펼친 손가락 개수**(엄지 제외)를 MediaPipe로 인식해 모드를 고릅니다. 신체(왼손 제스처)가 객체(펜)의 동작을 결정하는 구조입니다.

| 왼손 손가락 | 모드 | 동작 |
|:---:|---|---|
| 1개 | `DRAW` | 펜 끝을 따라 선을 그림 |
| 2개 | `STOP` | 그리기 멈춤 (펜을 옮겨도 선이 생기지 않음) |
| 손바닥 펴기 (엄지 제외 4개) | `CLEAR` | 캔버스 전체 지우기 → 이후 `STOP` |

| 파일 | 내용 | 원본 (`02_Computer-Vision.ipynb`) |
|---|---|---|
| `main.py` | 실행 진입점: 모드 판정, 캔버스 그리기, 화면 출력 | |
| `pen_tracker.py` | 초록 펜 끝 검출 + 칼만 필터 보정 (단독 실행 시 펜 추적 테스트) | K. 실시간 객체 검출 [255], L. Kalman Filter [266], USB 카메라 파이프라인 [245] |
| `hand_gesture.py` | 손 검출 + 왼손 손가락 개수 (단독 실행 시 제스처 테스트) | M. MediaPipe 손 검출 [275], 손가락 개수 [277] |
| `models/hand_landmarker.task` | MediaPipe 손 landmark 모델 (폴더만으로 실행되도록 포함) | 모델 다운로드 [273] |

## 실행 환경
- 기기: NVIDIA Jetson Orin / Python 3.10 (레포 루트 `.venv`, `--system-site-packages`)
- 카메라: USB 웹캠 `/dev/video0`, YUYV 640×480 30fps (GStreamer `v4l2src`)
- 의존성
  - `opencv` (`cv2`), `numpy`: JetPack 시스템 패키지
  - `mediapipe`: 수업 안내(노트북 셀 [271])대로 설치
    ```bash
    pip install --no-deps mediapipe
    pip install absl-py flatbuffers sounddevice
    ```
- 펜: 초록색 볼펜 1자루 (화면에 다른 초록 물체가 없어야 함)

## 실행 방법
```bash
source ../.venv/bin/activate   # Project01 폴더에서
python3 main.py
```
- `q`: 종료
- 창을 띄우므로 Jetson에 연결한 모니터에서 실행합니다.
- GStreamer가 없는 환경(Mac 등)에서는 `open_camera()`가 기본 카메라 0번을 640×480으로 엽니다. 이때는 `pip install opencv-python mediapipe numpy`로 만든 별도 가상환경에서 실행합니다 (Jetson에서는 이 명령을 쓰지 않습니다).
- 단계별 확인용: `python3 pen_tracker.py` (펜 추적만), `python3 hand_gesture.py` (제스처만)

## 구현 설명

### 1. 펜 끝 검출 (`pen_tracker.find_pen_tip`)
수업 K장의 초록 공 검출 흐름을 그대로 사용합니다.
1. **전처리:** `GaussianBlur(7×7)` → BGR을 LAB로 변환. LAB은 밝기(L)와 색(A, B)이 분리되어 조명 변화에 덜 민감합니다.
2. **Color Segmentation:** 초록 범위 `[30, 60, 90] ~ [230, 115, 180]`로 `inRange` → binary mask.
3. **Morphology:** 5×5 타원 커널로 Opening 2회(작은 노이즈 제거) → Closing 2회(펜 내부 구멍 메우기).
4. **Contour:** 면적 `MIN_AREA`(150px) 이하인 contour는 노이즈로 버리고, 가장 큰 contour를 펜으로 선택합니다.
5. **펜 끝:** 펜은 길쭉해서 무게중심이 펜 가운데에 생기고, 손으로 쥐면 가려집니다. 그래서 contour 점 중 **y가 가장 작은 점**(`argmin`), 즉 화면에서 가장 위쪽 점을 펜 끝으로 씁니다.

### 2. 칼만 필터 보정 (`create_kalman`, `reset_kalman`, `smooth_tip`)
- 수업 L장의 `cv2.KalmanFilter(4, 2)` 설정을 그대로 사용합니다. 상태는 `[x, y, vx, vy]`, 측정은 `[x, y]`인 등속 모델입니다.
- 매 프레임 `predict()` → `correct(측정값)`으로 검출 위치의 떨림을 줄입니다.
- **직접 구현 버전 대신 `cv2.KalmanFilter`를 쓴 이유:** 상태(`statePost`)를 객체가 들고 있어서, 함수로 나눠도 `mu_t`, `sigma_t`를 전역 변수로 주고받을 필요가 없습니다.
- **초기화:** 펜이 처음 보이거나, 한 프레임에 `MAX_JUMP`(80px)보다 멀리 튀면 그 위치에서 칼만을 다시 시작합니다.
  - `predict()`는 `statePost`에서 출발하므로 `statePre`와 `statePost`를 함께 설정합니다.
  - 초기화하지 않으면 보정 위치가 이전 위치에서 새 위치로 천천히 따라가면서, 실제로 지나가지 않은 구간에 선이 그려집니다.

### 3. 왼손 손가락 개수 (`hand_gesture.py`)
- **손 검출:** MediaPipe `HandLandmarker`(최대 2손)로 21개 landmark를 검출합니다.
- **왼손 라벨:** 화면을 좌우 반전(거울 모드)해서 넣으면 MediaPipe의 Left/Right가 실제 손과 반대로 나옵니다. 그래서 셀 [277]처럼 라벨을 뒤집어 실제 손 기준으로 맞춥니다.
- **손가락 개수:** 노트북은 양손 합계를 세지만, 여기서는 **`GESTURE_HAND`(왼손) 한 손만** 셉니다. 검지~소지마다 관절 세 점(예: 검지 5-6-7)이 이루는 각도를 계산해 160° 이상이면 펼친 것으로 봅니다.
- **엄지를 세지 않는 이유:** 측정해 보니 엄지 1-2-3 각도는 접었을 때도 중앙값 160°(검지 1개 자세), 폈을 때 170° 정도라 임계값으로 나눌 수 없었습니다. 엄지를 세면 검지 1개가 2개로 읽혀 `DRAW`가 `STOP`이 됩니다. 그래서 엄지를 빼고, 손바닥을 편 자세를 4개로 봅니다.

### 4. 모드 판정 (`main.update_mode`)
- 손가락 개수는 프레임마다 흔들릴 수 있습니다. 그래서 **같은 개수가 `GESTURE_HOLD_FRAMES`(5) 프레임 연속**일 때만 모드를 바꿉니다.
- 1, 2, 4가 아닌 개수이거나 왼손이 안 보이면 현재 모드를 유지합니다.

### 5. 캔버스 (`main.py`)
- 프레임과 같은 크기의 검은 캔버스(`np.zeros_like`)를 만듭니다.
- `DRAW` 모드에서 펜이 보이면, 직전 점과 현재 점을 `cv2.line`으로 잇습니다.
- 멈춤 모드이거나 펜이 안 보이면 `prev_point = None`으로 선을 끊습니다.
- **출력:** 캔버스에서 그림이 있는 픽셀(`canvas.any(axis=2)`)만 프레임 위에 덮어씁니다.
  - 펜 검출과 손 검출은 그림을 덮기 **전의** 원본 프레임에서 합니다. 그래서 화면에 그린 선이 검출에 영향을 주지 않습니다.

## 참고 사항
- **처음 실행할 때 확인할 것**
  1. `python3 hand_gesture.py`를 실행하고 **왼손만** 들었을 때 화면에 `Left`가 나와야 합니다. `Right`가 나오면 `hand_gesture.py`의 `GESTURE_HAND`를 `"Right"`로 바꿉니다.
  2. `python3 pen_tracker.py`를 실행해 "Pen Mask" 창에서 펜이 흰색으로 잡히는지 봅니다. 배경이 같이 잡히면 `MIN_AREA`나 초록 범위를 조정합니다.
- **알려진 한계**
  - **엄지 미사용:** 엄지는 세지 않으므로 엄지만 펴거나 엄지+검지를 펴도 1개로 읽힙니다.
  - **작은 초록 물체:** `MIN_AREA`를 150px로 낮춰서, 화면에 150px보다 큰 초록 물체가 있으면 펜으로 잡힐 수 있습니다.
  - **화면 속 초록 물체:** 펜 외에 초록 물체가 화면에 있으면 더 큰 쪽을 펜으로 잡습니다.
  - **펜 방향:** 펜 끝은 가장 위쪽 점으로 정하므로, 펜을 눕히거나 거꾸로 들면 펜 끝 위치가 틀립니다.
  - **칼만 지연:** 칼만 보정 때문에 펜을 빠르게 움직이면 선이 조금 늦게 따라옵니다. `pen_tracker.py`의 `measurementNoiseCov`를 줄이면 반응이 빨라지는 대신 떨림이 늘어납니다.
