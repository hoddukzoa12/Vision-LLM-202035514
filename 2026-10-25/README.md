# Day06 과제 #3 — GPU 가속 (CuPy, PyTorch), IoU·NMS

## 개요
`03_DL-and-GPU.ipynb`, `04_DL-Object-Detection.ipynb`의 미완성 TODO를 독립 실행 가능한 Python 파일로 옮겨 구현한 과제입니다.
- **F. CNN 실습 – MNIST:** 학습한 CNN의 Loss·Accuracy를 그래프로 그리고(16), 층별 Feature Map을 시각화합니다(17).
- **G. GPU 병렬 연산 (CuPy):** 이미지 잔상 효과(4)와 2차원 convolution(5)을 NumPy(CPU)와 CuPy(GPU)로 각각 구현하고 처리 시간을 비교합니다.
- **I. MNIST GPU 가속 학습:** `main_cpu()`를 복사해 `main_gpu()`를 만들고 CPU/GPU 학습 시간을 비교합니다.
- **J. CIFAR-10 GPU 가속 학습:** 5 Epoch 학습·평가 후 모델을 `.pth`로 저장하고 다시 불러와 확인합니다.
- **04 B, C. IoU, NMS:** 두 BBox의 IoU와, Confidence Thresholding + NMS로 중복 BBox를 제거하는 함수를 구현합니다.

| 파일 | 내용 | 원본 셀 |
|---|---|---|
| `mnist_cnn.py` | todo01~03이 함께 쓰는 MNIST 데이터셋, `SimpleCNN`, 학습·평가 함수, `train_model()` | [145]~[237] |
| `todo01_loss_graph.py` | Epoch별 Train/Test Loss 그래프 | [243] |
| `todo02_accuracy_graph.py` | Epoch별 Train/Test Accuracy 그래프 (백분율) | [245] |
| `todo03_feature_maps.py` | Conv1 8개, Conv2 16개, 6개 층별 대표 Feature Map 4개씩 비교 | [196]~[198], [248] |
| `todo04_motion_trail.py` | 잔상 효과 NumPy/CuPy 함수, CPU↔GPU 복사, 처리 시간 비교 | [299]~[324] |
| `todo05_convolution.py` | CuPy 2차원 convolution 함수, 세로/가로 edge CPU·GPU 처리, 결과 비교 | [117]~[122], [330]~[338] |
| `todo06_mnist_gpu.py` | `main_cpu()` / `main_gpu()` MNIST 학습 시간 비교 | [386]~[391] |
| `todo07_cifar10.py` | CIFAR-10 CNN GPU 학습(5 Epoch), 그래프, 모델 저장·불러오기 | [400]~[476] |
| `todo08_iou.py` | (04 노트북) BBox 좌표 변환, IoU 계산, 사과 이미지 3장 IoU | [23], [30]~[42] |
| `todo09_nms.py` | (04 노트북) Confidence Thresholding, NMS, OpenCV NMS와 비교 | [48]~[65] |
| `models/CIFAR10_CNN.pth` | `todo07`이 저장한 CIFAR-10 모델 | [471] |
| `results/` | 실행 결과 그림 (PNG) | |

## 실행 환경
- CUDA GPU 필요: `todo04`, `todo05`는 CuPy, `todo06`, `todo07`은 PyTorch CUDA를 씁니다. `todo01~03`은 GPU가 없으면 CPU로 학습합니다. `todo08`, `todo09`는 CPU만 씁니다.
- 의존성: `torch`, `torchvision`, `cupy`, `numpy`, `opencv-python`, `matplotlib`, `scikit-image`(`todo09`의 `skimage.data.hubble_deep_field()`)

## 실행 방법
```bash
source .venv/bin/activate
python 2026-10-25/todo01_loss_graph.py
python 2026-10-25/todo02_accuracy_graph.py
python 2026-10-25/todo03_feature_maps.py
python 2026-10-25/todo04_motion_trail.py
python 2026-10-25/todo05_convolution.py
python 2026-10-25/todo06_mnist_gpu.py
python 2026-10-25/todo07_cifar10.py
python 2026-10-25/todo08_iou.py
python 2026-10-25/todo09_nms.py
```
- **경로:** 어느 위치에서 실행해도 레포 기준 경로로 이미지와 데이터셋을 읽습니다. 결과 그림은 `2026-10-25/results/`, CIFAR-10 모델은 `2026-10-25/models/`에 저장됩니다.
- **데이터셋:** MNIST, CIFAR-10이 없으면 처음 실행할 때 `src/datasets/`에 내려받습니다.
- **화면 출력:** 창을 띄우지 않으므로 SSH 접속 중에도 실행할 수 있습니다.

## 구현 설명

### 공통: `mnist_cnn.py`
- **공통 모듈로 뺀 이유:** todo01~03은 모두 학습된 모델과 학습 기록이 필요합니다. 노트북 셀 [145]~[237]의 학습 코드를 파일마다 복사하는 대신 한 모듈에 모았습니다.
- **`train_model()`:** 셀 [237]의 Epoch 반복을 함수로 묶은 것입니다. 모델, 테스트 데이터셋, device, Epoch별 기록(dict)을 돌려줍니다.
- **device:** 노트북은 CPU로 고정했지만, 여기서는 `torch.cuda.is_available()`이면 GPU를 씁니다.
- **`show_feature_maps()`:** 셀 [194] 그대로에 `save_path` 인자만 추가해, 그림을 PNG로 저장합니다.

### F-16) 모델 성능 확인: `todo01`, `todo02`
- **x축:** `range(1, len(history) + 1)`로 Epoch 번호를 1부터 셉니다. `plt.xticks()`로 정수 눈금만 표시합니다.
- **Accuracy 단위:** `evaluate()`는 0~1 비율을 돌려주므로, 100을 곱해 y축 단위(%)에 맞췄습니다.

### F-17) 학습된 모델의 Feature Map 확인: `todo03`
1. **이미지 선택:** `test_dataset[0]`(정답 7)을 고르고, `unsqueeze(0)`으로 Batch 차원을 붙여 `[1, 1, 28, 28]`로 만듭니다.
2. **순전파:** `model.eval()`과 `torch.no_grad()` 안에서 `return_features=True`로 순전파합니다. 역전파 없이 층별 출력을 dict로 받습니다.
3. **Conv1·Conv2 시각화:** `show_feature_maps()`로 각 층의 채널 전체(8개, 16개)를 `cmap="viridis"`로 그립니다.
4. **6개 층 비교:** 행은 층 6개(`conv1` → `pool2`), 열은 채널 0~3입니다. 제목에 크기를 적어 28×28 → 14×14 → 7×7로 줄어드는 과정이 보이게 했습니다.

### G-4) CuPy 잔상 효과: `todo04`
- **NumPy (셀 [303]):** 출력 픽셀마다 `input_x = output_x - shift`로 왼쪽 픽셀을 찾아 가중치를 곱해 누적합니다.
  - `input_x < 0`이면 이미지 밖이므로 `break`합니다. 노트북 설명의 "밀린 자리는 0"과 같습니다.
- **CuPy (셀 [305]):** 같은 계산을 이동 거리 단위로 바꿨습니다.
  - `image[:, :width - shift]`와 `output[:, shift:]`는 크기가 같고, 원본을 오른쪽으로 `shift`칸 민 위치에 겹칩니다.
  - `output_region`은 `output`의 view라서, `+=` 하면 `output`에 바로 누적됩니다.
  - Python 반복은 `TRAIL_LENGTH`(32)번뿐이고, 한 번에 이미지 전체(1280×720)를 GPU가 병렬로 계산합니다.
- **CPU↔GPU 복사 (셀 [311], [317]):** 업로드는 `cp.asarray()`, 다운로드는 `cp.asnumpy()`입니다. matplotlib은 NumPy 배열만 그릴 수 있어서 결과를 다시 CPU로 가져옵니다.
- **시간 측정:** GPU 연산은 비동기라서 측정 전후에 `cp.cuda.Stream.null.synchronize()`로 연산이 끝나기를 기다립니다.

### G-5) CuPy Convolution: `todo05`
- **CPU (셀 [332]):** 셀 [117] `convolution2d()`를 그대로 옮겨 세로(`dx`)·가로(`dy`) edge 필터와 편향 -0.2(셀 [120], [122])를 적용합니다.
- **GPU 함수 (셀 [330]):** 잔상 효과의 slicing 누적을 2차원으로 넓혔습니다.
  - CPU 버전은 출력 픽셀(1280×720개)마다 3×3 영역을 잘라 가중합합니다.
  - GPU 버전은 반대로 커널 원소 `(kernel_y, kernel_x)` 9개만 반복합니다.
  - 반복마다 모든 출력 위치에서 그 원소와 곱해질 입력 픽셀을 slicing으로 한 번에 꺼내 `kernel[kernel_y, kernel_x]`를 곱해 출력 전체에 더합니다.
  - 출력 `(y, x)`에 대응하는 입력은 `(y * stride + kernel_y, x * stride + kernel_x)`이므로, slice 간격을 `stride`로 두어 stride가 1이 아니어도 같은 결과가 나옵니다.
  - padding(`cp.pad`)과 출력 크기 공식 `(H + 2P - K) // S + 1`은 CPU 버전과 같습니다.
- **GPU 처리 (셀 [334]):** 변환(`cp.asarray`) → 워밍업(64×64 이미지로 한 번 실행) → 처리(시간 측정) → 변환(`cp.asnumpy`) 순서입니다.
  - CPU 처리와 같은 구조로 맞췄습니다. 시간은 두 쪽 모두 dx + dy convolution 구간만 재고, ReLU는 그 뒤에 적용합니다.
  - ReLU는 셀 [58] `relu()`(`np.maximum`)의 CuPy 버전인 `relu_cp()`(`cp.maximum`)로 GPU에서 적용합니다.
- **비교 (셀 [336]):** 처리 시간과 속도 향상, CPU·GPU 결과의 최대 차이를 출력합니다.

### I-3) MNIST GPU 가속 학습: `todo06`
- **`main_cpu()` (셀 [386]):** 노트북 그대로입니다. `SimpleCNN`, `train_one_epoch()`, `evaluate()`는 같은 코드인 `mnist_cnn.py`에서 가져옵니다.
- **`main_gpu()` (셀 [389]):** `main_cpu()`를 복사해 네 군데만 바꿨습니다.
  1. `torch.cuda.manual_seed(SEED)`로 GPU 난수도 고정합니다.
  2. `device = torch.device("cuda")`로 모델을 GPU에 올립니다. `train_one_epoch()`, `evaluate()` 안의 `images.to(device)`가 Batch도 GPU로 옮기므로 학습 함수는 그대로 씁니다.
  3. 시간 측정 전후에 `torch.cuda.synchronize()`를 넣었습니다. GPU 연산은 비동기라서, 기다리지 않으면 GPU가 일을 끝내기 전에 시간을 재게 됩니다. CuPy의 `cp.cuda.Stream.null.synchronize()`와 같은 역할입니다.
  4. 출력 문구를 `Total GPU training time`으로 바꿨습니다.

### J) CIFAR-10 GPU 가속 학습: `todo07`
- **모델·전처리 (셀 [413]~[453]):** 노트북의 증강(RandomCrop, RandomHorizontalFlip), 정규화, `ConvBlock` × 3 + Global Average Pooling 모델, AdamW, GPU Warm-up을 그대로 옮겼습니다.
- **학습 (셀 [456]):** MNIST와 같은 `train_one_epoch()`, `evaluate()`를 5 Epoch 반복합니다.
  - `train_one_epoch()`은 `model.train()`, `evaluate()`는 `model.eval()`로 바꾸므로, Dropout과 BatchNorm이 학습/평가에 맞게 동작합니다.
  - Epoch마다, 그리고 전체 학습이 끝난 뒤 `torch.cuda.synchronize()`로 GPU 연산을 기다린 다음 시간을 잽니다.
- **모델 저장 (셀 [471]):** 노트북과 같은 dict(모델·optimizer 가중치, 정확도, Epoch, 클래스 이름, 평균·표준편차)를 과제 폴더 안 `models/CIFAR10_CNN.pth`에 저장합니다. 노트북은 `src/models/CIFAR10/`에 저장하지만, 제출물을 과제 폴더에 모으기 위해 위치만 바꿨습니다.
- **불러오기 확인 (셀 [473]~[476]):** 새 `CIFAR10CNN()`에 저장한 가중치를 넣고, 테스트 정확도가 학습 직후와 같은지 확인합니다.

### 04 B) IoU: `todo08`
- **`convert_bbox_coord()`:** `cv2.boundingRect()`의 `(x, y, w, h)`를 `(x1, y1, x2 = x + w, y2 = y + h)`로 바꿉니다.
- **`calculate_iou()`:**
  - 교집합 박스는 좌측 상단이 두 박스 중 큰 값(`max`), 우측 하단이 작은 값(`min`)입니다.
  - 겹치지 않으면 `x2 - x1`이 음수가 되므로 `max(0, ...)`으로 0을 만듭니다.
  - 합집합 = 넓이 A + 넓이 B − 교집합 (겹친 부분이 두 번 더해지므로 한 번 뺌)
  - 합집합이 0이면(넓이 0인 박스 두 개) `ZeroDivisionError` 대신 0을 반환합니다.
- **확인:** 손으로 계산할 수 있는 박스(같은 박스 1.0, 떨어진 박스 0.0, 절반 겹침 1/3, 넓이 0)로 먼저 확인한 뒤 사과 이미지 3장에 적용합니다.

### 04 C) NMS: `todo09`
- **`prune_low_conf()` (셀 [51]):** `zip(boxes, scores)`로 박스와 점수를 함께 꺼내, 점수가 임계값 이상인 것만 남깁니다.
- **`apply_nms()` (셀 [52]):**
  1. `np.argsort(scores)[::-1]`로 Confidence가 높은 순서의 인덱스를 만듭니다.
  2. 가장 높은 박스를 꺼내 결과에 넣습니다.
  3. 남은 박스 중 그 박스와 IoU가 임계값보다 큰 박스(같은 객체로 보이는 박스)를 지웁니다.
  4. 남은 박스가 없을 때까지 2~3을 반복합니다.
- **OpenCV와 비교:** 같은 박스에 `cv2.dnn.NMSBoxes()`(셀 [58])를 적용해, 직접 구현한 결과와 같은 박스가 남는지 출력합니다.

## 실행 결과

### F-16, F-17 (MNIST CNN, 3 Epoch)
```
Epoch 1/3 | Train loss: 0.3141 | Train accuracy: 90.49% | Test loss: 0.0976 | Test accuracy: 96.92%
Epoch 2/3 | Train loss: 0.0930 | Train accuracy: 97.13% | Test loss: 0.0687 | Test accuracy: 97.80%
Epoch 3/3 | Train loss: 0.0665 | Train accuracy: 97.93% | Test loss: 0.0503 | Test accuracy: 98.27%
True label: 7, Predicted label: 7
```

| Loss | Accuracy |
|---|---|
| ![](results/01_loss_graph.png) | ![](results/02_accuracy_graph.png) |

| 그림 | 내용 |
|---|---|
| ![](results/03_layer_comparison.png) | 6개 층 × 채널 0~3 |

Conv1·Conv2 전체 채널: `03_conv1_feature_maps.png`, `03_conv2_feature_maps.png`

**관찰:**
- **Conv 출력:** `conv1`, `conv2` 출력에는 음수가 있습니다. `relu` 뒤에는 0 이상만 남습니다.
- **`relu1 ch2`:** 이 채널은 배경이 크고 획이 음수인 필터라서, ReLU를 거치면 거의 비게 됩니다.
- **Pooling:** 크기를 절반으로 줄이면서 강한 값만 남깁니다.
- **`conv2` 이후:** 가로 획, 대각선 획처럼 숫자의 부분 모양에 반응하는 채널이 나뉩니다.

### G-4 (잔상 효과, 1280×720, `TRAIL_LENGTH = 32`)
```
========== NumPy CPU ==========
Processing time: 10.6360 seconds

========== CuPy GPU ==========
CPU to GPU: 0.2386 seconds
GPU processing: 0.0034 seconds
GPU to CPU: 0.0016 seconds
Total GPU time: 0.2436 seconds

========== Comparison ==========
Compute-only speedup: 3138.15x
Including transfers: 43.67x
Max |CPU - GPU|: 0.00e+00
```

![](results/04_motion_trail_comparison.png)

### G-5 (Convolution, 1280×720, 3×3 edge 필터 2개)
```
========== NumPy CPU ==========
Processing time (dx + dy): 11.2384 seconds

========== CuPy GPU ==========
CPU to GPU: 0.5134 seconds
GPU processing (dx + dy): 0.0027 seconds
GPU to CPU: 0.0041 seconds
Total GPU time: 0.5202 seconds

========== Comparison ==========
Compute-only speedup: 4157.18x
Including transfers: 21.60x
Max |CPU - GPU| (vertical): 1.19e-07
Max |CPU - GPU| (horizontal): 2.38e-07
```

![](results/05_convolution_comparison.png)

### I-3) MNIST CPU vs. GPU 학습 (3 Epoch)
```
=== CPU Training ===
Epoch 1/3 | Train loss: 0.3134 | Train accuracy: 90.53% | Test loss: 0.0957 | Test accuracy: 97.01% | Time: 14.40s
Epoch 2/3 | Train loss: 0.0924 | Train accuracy: 97.12% | Test loss: 0.0693 | Test accuracy: 97.74% | Time: 14.09s
Epoch 3/3 | Train loss: 0.0669 | Train accuracy: 97.92% | Test loss: 0.0499 | Test accuracy: 98.19% | Time: 13.99s
Total CPU training time: 42.48s

=== GPU Training ===
Epoch 1/3 | Train loss: 0.3132 | Train accuracy: 90.54% | Test loss: 0.0963 | Test accuracy: 96.94% | Time: 9.87s
Epoch 2/3 | Train loss: 0.0925 | Train accuracy: 97.14% | Test loss: 0.0697 | Test accuracy: 97.72% | Time: 8.55s
Epoch 3/3 | Train loss: 0.0669 | Train accuracy: 97.90% | Test loss: 0.0492 | Test accuracy: 98.24% | Time: 8.55s
Total GPU training time: 26.97s

=== CPU vs. GPU Comparison ===
CPU training time: 42.48s
GPU training time: 26.97s
Time difference: 15.51s
GPU speedup: 1.57x
```

### J) CIFAR-10 GPU 학습 (5 Epoch)
```
Train data: 50000
Test data: 10000
Device: cuda
Total parameters: 1148874
Epoch 1/5 | Train loss: 1.2857 | Train accuracy: 53.30% | Test loss: 0.9670 | Test accuracy: 65.04% | Time: 31.33s
Epoch 2/5 | Train loss: 0.8851 | Train accuracy: 68.77% | Test loss: 0.8532 | Test accuracy: 70.82% | Time: 31.17s
Epoch 3/5 | Train loss: 0.7120 | Train accuracy: 75.44% | Test loss: 0.7249 | Test accuracy: 75.91% | Time: 31.33s
Epoch 4/5 | Train loss: 0.6250 | Train accuracy: 78.66% | Test loss: 0.6377 | Test accuracy: 78.62% | Time: 31.81s
Epoch 5/5 | Train loss: 0.5552 | Train accuracy: 81.09% | Test loss: 0.5380 | Test accuracy: 81.92% | Time: 30.91s
Total GPU training time: 156.55s
모델 이름: CIFAR10CNN
Epoch 체크포인트: 5
마지막 테스트 정확도: 81.92%
불러온 모델의 테스트 정확도: 81.92%
```

![](results/07_cifar10_training_curves.png)

| 학습 직후 모델 (테스트 0~39번) | 불러온 모델 (테스트 40~79번) |
|---|---|
| ![](results/07_cifar10_predictions.png) | ![](results/07_cifar10_loaded_model_predictions.png) |

빨간 제목은 틀린 예측입니다. 5 Epoch로는 고양이·개처럼 비슷한 클래스를 아직 자주 헷갈립니다.

### 04 B) IoU
```
Same box: 1.0
No overlap: 0.0
Half overlap: 0.3333333333333333
Zero area: 0.0

two_apples1.png IoU: 0.0000
two_apples2.png IoU: 0.3046
two_apples3.png IoU: 0.4934
```

![](results/08_iou_comparison.png)

### 04 C) NMS (BBox 20개, Confidence 임계값 0.5, IoU 임계값 0.2)
```
Before: 20 20
After confidence thresholding: 17 17
After custom NMS: 4 4
After cv2 NMS: 4 4
Same result as cv2.dnn.NMSBoxes(): True
```

![](results/09_nms_comparison.png)

## 참고 사항
- **셀 [313] 워밍업 오류:** 노트북 원본은 `weights_np[:4]`(4개)를 넘깁니다.
  - `motion_trail_cp()`는 템플릿대로 `TRAIL_LENGTH`(32)번 반복하며 `weights[shift]`를 읽으므로, `shift = 4`에서 `IndexError: Index 4 is out of bounds`가 납니다.
  - 그래서 워밍업에는 가중치 전체(32개)를 넘깁니다. 함수 템플릿은 그대로 두었습니다.
- **결과 검증:** CPU와 GPU 결과의 최대 차이를 출력해 두 구현이 같은 값을 내는지 확인합니다.
- **업로드 시간:** 업로드(0.24초)가 GPU 계산(0.003초)보다 깁니다. 셀 순서상 업로드가 CuPy를 처음 쓰는 단계라서, GPU 초기화 시간이 업로드 시간에 함께 측정된 것으로 보입니다.
- **학습 반복:** todo01~03은 각각 독립 실행되도록 매번 모델을 새로 학습합니다. seed를 고정했지만 GPU 연산 특성상 Loss·Accuracy는 실행마다 소수점 둘째 자리 정도 달라질 수 있습니다.
- **잔상 가장자리:** 그림의 왼쪽 가장자리에 생기는 어두운 띠는 이미지 밖(0)을 더했기 때문입니다.
- **Convolution 결과 차이:** CPU와 GPU의 최대 차이는 약 1e-7로, float32에서 곱셈·덧셈 순서가 달라 생기는 반올림 오차 수준입니다. 작은 무작위 이미지로 stride 1~3, padding 0~2 조합에서도 두 함수의 출력 크기와 값이 같은지(오차 1e-6 이하) 확인했습니다.
- **MNIST GPU 속도 향상이 작은 이유:** `SimpleCNN`은 파라미터가 적어 GPU가 한 Batch를 금방 끝냅니다. 대신 Batch마다 DataLoader가 CPU에서 이미지를 준비하고(`num_workers=0`), GPU로 복사하고, `loss.item()`으로 결과를 기다리는 시간이 대부분이라 CuPy 실습처럼 수천 배가 나오지 않습니다.
- **`.pth` 파일:** `models/CIFAR10_CNN.pth`(약 14MB)입니다. 모델 가중치와 optimizer 상태(Adam 계열은 파라미터마다 값 2개를 더 저장)가 함께 들어 있어 모델 크기보다 큽니다.
- **NMS 노트북 셀 수정:** 셀 [54]는 `prune_low_conf()`의 결과가 아닌 원래 `boxes`를 `apply_nms()`에 넘기고, 셀 [55], [63]은 남은 박스에 원래 `scores[i]`를 표시합니다. `todo09`에서는 걸러진 결과(`new_boxes`, `new_scores`)를 이어서 쓰도록 바꿨습니다.
- **NMS 난수 고정:** 노트북은 실행할 때마다 다른 BBox가 생성됩니다. `todo09`는 `random.seed(42)`로 고정해 같은 결과가 나오게 했습니다.
