"""Day06 과제 #3 — CuPy를 활용한 이미지 잔상(Motion Trail) 효과 CPU vs. GPU

출처: 03_DL-and-GPU.ipynb (G. GPU 병렬 연산 (CuPy), 4) 이미지 잔상 효과 구현)
    셀 [299]  글로벌 변수
    셀 [301]  이미지 불러오기 (흑백, 1280x720, 0~1 실수)
    셀 [303]  TODO: NumPy 잔상 효과 함수 (픽셀 단위 반복)
    셀 [305]  TODO: CuPy 잔상 효과 함수 (이동 거리만 반복, 이미지 전체는 GPU 병렬)
    셀 [307]  가중치 설정
    셀 [309]  NumPy CPU 처리
    셀 [311]  TODO: NumPy 배열을 CuPy 배열로 변환
    셀 [313]  GPU 워밍업
    셀 [315]  CuPy GPU 처리
    셀 [317]  TODO: output_cp를 CPU 메모리로 복사
    셀 [319]~[324]  결과 비교, 출력, 이미지 저장

결과 그림은 화면에 띄우지 않고 results/04_*.png로 저장한다.
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

TRAIL_LENGTH = 32


# 셀 [303]
# TODO: NumPy를 사용하여 잔상 효과 함수 정의
def motion_trail_np(image, weights):
    """
    NumPy를 이용해 이미지 잔상 효과를 생성한다.

    출력 픽셀을 Python 반복문으로 하나씩 계산한다.
    """

    # TODO: 필요한 변수 정의 및 출력 배열 생성
    height, width = image.shape
    output = np.zeros((height, width), dtype=np.float32)

    for output_y in range(height):
        for output_x in range(width):
            pixel_sum = 0.0

            for shift in range(TRAIL_LENGTH):
                # TODO: x값 shift
                input_x = output_x - shift
                if input_x < 0:  # 이미지 왼쪽 밖이면 더할 픽셀이 없음 (0으로 채운 것과 같음)
                    break

                # TODO: shift한 픽셀을 누적
                pixel_sum += image[output_y, input_x] * weights[shift]

            # TODO: 출력 배열에 누적 픽셀 저장
            output[output_y, output_x] = pixel_sum

    return output


# 셀 [305]
# TODO: CuPy를 사용하여 잔상 효과 함수 정의
def motion_trail_cp(image, weights):
    """
    CuPy를 이용해 이미지 잔상 효과를 생성한다.

    이동 거리는 Python에서 순차적으로 반복하지만,
    각 이동 거리의 이미지 전체 계산은 GPU에서 병렬 처리한다.
    """

    # TODO: 필요한 변수 정의 및 출력 배열 생성
    height, width = image.shape
    output = cp.zeros((height, width), dtype=cp.float32)

    for shift in range(TRAIL_LENGTH):
        # TODO: 배열 slicing
        input_region = image[:, :width - shift]  # 오른쪽 끝 shift칸을 뺀 원본
        output_region = output[:, shift:]        # 왼쪽 shift칸을 뺀 출력 (= 오른쪽으로 shift칸 이동한 자리)

        # TODO: GPU가 병렬 연산할 output_region 계산 코드
        output_region += input_region * weights[shift]  # output_region은 output의 view라서 output에 바로 누적됨

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

    # 셀 [307]
    # 이동 거리가 멀어질수록 가중치를 작게 설정
    weights_np = np.linspace(
        1.0,
        0.05,
        TRAIL_LENGTH,
        dtype=np.float32
    )

    # 전체 가중치의 합을 1로 정규화
    weights_np /= np.sum(weights_np)

    # 셀 [309]
    cpu_start = time.perf_counter()

    output_np = motion_trail_np(
        image=input_np,
        weights=weights_np
    )

    cpu_end = time.perf_counter()

    cpu_time = cpu_end - cpu_start

    # 셀 [311]
    # TODO: NumPy 배열을 CuPy 배열로 변환

    upload_start = time.perf_counter()

    input_cp = cp.asarray(input_np)
    weights_cp = cp.asarray(weights_np)

    cp.cuda.Stream.null.synchronize()

    upload_end = time.perf_counter()

    upload_time = upload_end - upload_start

    # 셀 [313] GPU 워밍업
    # 노트북은 weights_np[:4](4개)를 넘기지만, motion_trail_cp는 TRAIL_LENGTH(32)번 반복하며
    # weights[shift]를 읽으므로 shift=4에서 범위를 벗어난다. 그래서 가중치 전체를 넘긴다.
    warmup_image = cp.zeros((64, 64), dtype=cp.float32)
    warmup_weights = cp.asarray(weights_np)

    _ = motion_trail_cp(
        image=warmup_image,
        weights=warmup_weights
    )

    cp.cuda.Stream.null.synchronize()

    # 셀 [315]
    cp.cuda.Stream.null.synchronize()

    gpu_start = time.perf_counter()

    output_cp = motion_trail_cp(
        image=input_cp,
        weights=weights_cp
    )

    cp.cuda.Stream.null.synchronize()

    gpu_end = time.perf_counter()

    gpu_time = gpu_end - gpu_start

    # 셀 [317]
    # TODO: output_cp를 CPU 메모리로 복사

    download_start = time.perf_counter()

    output_cp_np = cp.asnumpy(output_cp)

    download_end = time.perf_counter()

    download_time = download_end - download_start

    # 셀 [319]
    compute_speedup = cpu_time / gpu_time

    gpu_total_time = upload_time + gpu_time + download_time
    total_speedup = cpu_time / gpu_total_time

    # 셀 [321]
    print("========== NumPy CPU ==========")
    print(f"Processing time: {cpu_time:.4f} seconds")
    print()

    print("========== CuPy GPU ==========")
    print(f"CPU to GPU: {upload_time:.4f} seconds")
    print(f"GPU processing: {gpu_time:.4f} seconds")
    print(f"GPU to CPU: {download_time:.4f} seconds")
    print(f"Total GPU time: {gpu_total_time:.4f} seconds")
    print()

    print("========== Comparison ==========")
    print(f"Compute-only speedup: {compute_speedup:.2f}x")
    print(f"Including transfers: {total_speedup:.2f}x")

    # CPU와 GPU 결과가 같은 값인지 확인 (float32 덧셈 순서 차이만큼의 오차는 허용)
    max_difference = np.max(np.abs(output_np - output_cp_np))
    print(f"Max |CPU - GPU|: {max_difference:.2e}")

    # 셀 [323], [324]
    RESULTS_DIR.mkdir(exist_ok=True)

    plt.figure(figsize=(18, 12))

    plt.subplot(1, 3, 1), plt.imshow(input_np, cmap="gray"), plt.title("Original Image")
    plt.subplot(1, 3, 2), plt.imshow(output_np, cmap="gray"), plt.title("Motion Trail (CPU)")
    plt.subplot(1, 3, 3), plt.imshow(output_cp_np, cmap="gray"), plt.title("Motion Trail (GPU)")

    for ax in plt.gcf().axes:
        ax.axis("off")

    plt.savefig(RESULTS_DIR / "04_motion_trail_comparison.png", bbox_inches="tight")

    plt.figure(figsize=(12, 8))
    plt.imshow(output_cp_np, cmap="gray")
    plt.axis("off")
    plt.savefig(RESULTS_DIR / "04_motion_trail_gpu.png", bbox_inches="tight")

    print("Saved:", RESULTS_DIR / "04_motion_trail_comparison.png")
    print("Saved:", RESULTS_DIR / "04_motion_trail_gpu.png")


if __name__ == "__main__":
    main()
