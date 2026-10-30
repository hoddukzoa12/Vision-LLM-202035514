"""Day06 과제 #3 — CuPy를 활용한 2차원 Convolution CPU vs. GPU

출처: 03_DL-and-GPU.ipynb
    셀 [58]          relu()
    셀 [117]         convolution2d() (NumPy, 출력 픽셀 단위 반복)
    셀 [120], [122]  세로/가로 edge 필터, 편향
    셀 [299], [301]  입력 이미지 (흑백, 1280x720, 0~1 실수)
    셀 [330]         TODO: CuPy 2차원 convolution 함수 (커널 원소 단위 반복)
    셀 [332]         TODO: convolution2d로 CPU 처리
    셀 [334]         TODO: GPU 처리 (변환-워밍업-처리 파이프라인)
    셀 [336]         TODO: CPU/GPU 결과 비교 및 출력
    셀 [338]         이미지 출력

결과 그림은 화면에 띄우지 않고 results/05_convolution_comparison.png로 저장한다.
"""

import time
from pathlib import Path

import cupy as cp
import cv2
import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend("Agg")  # 창을 띄우지 않고 파일로만 저장

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# 셀 [299]
IMAGE_PATH = REPO_ROOT / "src" / "images" / "seagull.jpg"

TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

# 셀 [120] x축 방향 edge (세로 선) 검출 필터
DX_EDGE_FILTER = np.array(
    [
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ],
    dtype=np.float32
)

# 셀 [120] y축 방향 edge (가로 선) 검출 필터
DY_EDGE_FILTER = np.array(
    [
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1]
    ],
    dtype=np.float32
)

# 셀 [122] 너무 약한 특징 (0.2 이하)은 무시
DX_BIAS = -0.2
DY_BIAS = -0.2


# 셀 [58]
def relu(x):
    return np.maximum(0, x)


# 셀 [117]
def convolution2d(image, kernel, bias=0.0, stride=1, padding=0):
    """2차원 이미지에 하나의 CNN 필터를 적용 (NumPy, 출력 픽셀을 하나씩 계산)."""

    kernel_height, kernel_width = kernel.shape

    # 이미지 가장자리에 zero padding 적용
    padded_image = np.pad(
        image,
        ((padding, padding), (padding, padding)),
        mode="constant",
        constant_values=0
    )

    # Convolution 출력 크기 = floor((H + 2P - K) / S) + 1
    output_height = (padded_image.shape[0] - kernel_height) // stride + 1
    output_width = (padded_image.shape[1] - kernel_width) // stride + 1

    output = np.zeros((output_height, output_width), dtype=np.float32)

    # 커널을 움직이며 필터 적용
    for output_y in range(output_height):
        for output_x in range(output_width):
            start_y = output_y * stride
            start_x = output_x * stride

            image_region = padded_image[
                start_y:start_y + kernel_height,
                start_x:start_x + kernel_width
            ]

            weighted_sum = np.sum(image_region * kernel)
            output[output_y, output_x] = weighted_sum + bias

    return output


# 셀 [330]
# TODO: CuPy를 사용하여 2차원 convolution 함수 정의
def convolution2d_cp(image, kernel, bias=0.0, stride=1, padding=0):
    """
    CuPy를 이용한 2차원 convolution.

    커널의 위치는 Python 반복문으로 순차 처리하고,
    각 위치에 해당하는 출력 배열 전체 연산은 GPU에서 처리한다.
    """

    kernel_height, kernel_width = kernel.shape

    # convolution2d()와 같은 zero padding (GPU 배열)
    padded_image = cp.pad(
        image,
        ((padding, padding), (padding, padding)),
        mode="constant",
        constant_values=0
    )

    # 출력 크기 계산도 convolution2d()와 같음
    output_height = (padded_image.shape[0] - kernel_height) // stride + 1
    output_width = (padded_image.shape[1] - kernel_width) // stride + 1

    output = cp.zeros((output_height, output_width), dtype=cp.float32)

    # 커널 원소 (kernel_y, kernel_x) 하나를 고르고,
    # 모든 출력 위치에서 그 원소와 곱해질 입력 픽셀을 한 번에 slicing해 누적한다
    for kernel_y in range(kernel_height):
        for kernel_x in range(kernel_width):
            # 출력 (output_y, output_x)에 대응하는 입력은
            # (output_y * stride + kernel_y, output_x * stride + kernel_x)
            input_region = padded_image[
                kernel_y:kernel_y + stride * (output_height - 1) + 1:stride,
                kernel_x:kernel_x + stride * (output_width - 1) + 1:stride
            ]

            output += input_region * kernel[kernel_y, kernel_x]

    # 편향 적용
    output += bias

    return output


def main():
    # 셀 [301]
    img_gray = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_GRAYSCALE)

    if img_gray is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {IMAGE_PATH}")

    img_gray = cv2.resize(
        img_gray,
        (TARGET_WIDTH, TARGET_HEIGHT),
        interpolation=cv2.INTER_AREA
    )

    input_np = img_gray.astype(np.float32) / 255.0

    # 셀 [332]
    # TODO: 기존에 정의한 convolution2d 함수를 사용하여 convolution을 CPU로 처리
    # TODO: 변수 재사용 가능

    cpu_start = time.perf_counter()

    output_dx_np = convolution2d(input_np, DX_EDGE_FILTER, bias=DX_BIAS, stride=1, padding=1)
    output_dy_np = convolution2d(input_np, DY_EDGE_FILTER, bias=DY_BIAS, stride=1, padding=1)

    cpu_end = time.perf_counter()

    cpu_time = cpu_end - cpu_start
    relu_dx_np = relu(output_dx_np)
    relu_dy_np = relu(output_dy_np)

    # 셀 [334]
    # TODO: convolution을 GPU로 처리 (변환-워밍업-처리 파이프라인 구현)
    # TODO: 변수 재사용 가능

    # 1) 변환: CPU → GPU
    upload_start = time.perf_counter()

    input_cp = cp.asarray(input_np)
    dx_filter_cp = cp.asarray(DX_EDGE_FILTER)
    dy_filter_cp = cp.asarray(DY_EDGE_FILTER)

    cp.cuda.Stream.null.synchronize()

    upload_end = time.perf_counter()

    upload_time = upload_end - upload_start

    # 2) 워밍업: 작은 이미지로 한 번 실행해 GPU 초기화·커널 준비 시간을 측정에서 뺌
    warmup_image = cp.zeros((64, 64), dtype=cp.float32)

    _ = convolution2d_cp(warmup_image, dx_filter_cp, bias=DX_BIAS, stride=1, padding=1)

    cp.cuda.Stream.null.synchronize()

    # 3) 처리: GPU 연산 시간만 측정
    cp.cuda.Stream.null.synchronize()

    gpu_start = time.perf_counter()

    output_dx_cp = convolution2d_cp(input_cp, dx_filter_cp, bias=DX_BIAS, stride=1, padding=1)
    output_dy_cp = convolution2d_cp(input_cp, dy_filter_cp, bias=DY_BIAS, stride=1, padding=1)

    cp.cuda.Stream.null.synchronize()

    gpu_end = time.perf_counter()

    gpu_time = gpu_end - gpu_start

    # 4) ReLU는 GPU에서 적용한 뒤, 결과를 CPU로 복사
    download_start = time.perf_counter()

    relu_dx_cp_np = cp.asnumpy(cp.maximum(0, output_dx_cp))
    relu_dy_cp_np = cp.asnumpy(cp.maximum(0, output_dy_cp))

    download_end = time.perf_counter()

    download_time = download_end - download_start

    # 셀 [336]
    # TODO: CPU/GPU 결과 비교 및 출력
    compute_speedup = cpu_time / gpu_time

    gpu_total_time = upload_time + gpu_time + download_time
    total_speedup = cpu_time / gpu_total_time

    # 두 구현의 결과가 같은지 확인 (float32 덧셈 순서 차이만큼의 오차는 허용)
    max_difference_dx = np.max(np.abs(relu_dx_np - relu_dx_cp_np))
    max_difference_dy = np.max(np.abs(relu_dy_np - relu_dy_cp_np))

    print(f"Input image: {input_np.shape[1]}x{input_np.shape[0]}")
    print()

    print("========== NumPy CPU ==========")
    print(f"Processing time (dx + dy): {cpu_time:.4f} seconds")
    print()

    print("========== CuPy GPU ==========")
    print(f"CPU to GPU: {upload_time:.4f} seconds")
    print(f"GPU processing (dx + dy): {gpu_time:.4f} seconds")
    print(f"ReLU + GPU to CPU: {download_time:.4f} seconds")
    print(f"Total GPU time: {gpu_total_time:.4f} seconds")
    print()

    print("========== Comparison ==========")
    print(f"Compute-only speedup: {compute_speedup:.2f}x")
    print(f"Including transfers: {total_speedup:.2f}x")
    print(f"Max |CPU - GPU| (vertical): {max_difference_dx:.2e}")
    print(f"Max |CPU - GPU| (horizontal): {max_difference_dy:.2e}")

    # 셀 [338]
    RESULTS_DIR.mkdir(exist_ok=True)

    plt.figure(figsize=(18, 8))

    plt.subplot(2, 3, 1), plt.imshow(input_np, cmap="gray"), plt.title("Input Image")
    plt.subplot(2, 3, 2), plt.imshow(relu_dx_np, cmap="gray"), plt.title("Vertical Convolution (CPU)")
    plt.subplot(2, 3, 3), plt.imshow(relu_dx_cp_np, cmap="gray"), plt.title("Vertical Convolution (GPU)")
    plt.subplot(2, 3, 4), plt.imshow(input_np, cmap="gray"), plt.title("Input Image")
    plt.subplot(2, 3, 5), plt.imshow(relu_dy_np, cmap="gray"), plt.title("Horizontal Convolution (CPU)")
    plt.subplot(2, 3, 6), plt.imshow(relu_dy_cp_np, cmap="gray"), plt.title("Horizontal Convolution (GPU)")

    for ax in plt.gcf().axes:
        ax.axis("off")

    plt.savefig(RESULTS_DIR / "05_convolution_comparison.png", bbox_inches="tight")
    print("Saved:", RESULTS_DIR / "05_convolution_comparison.png")


if __name__ == "__main__":
    main()
