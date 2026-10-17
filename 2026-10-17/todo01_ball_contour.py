"""Day03 과제 #2 — 모폴로지 연산을 적용하여 최종 농구공 Contour 선택 및 출력

출처: 02_Computer-Vision.ipynb (H. Contour Detection)
    셀 [145], [167]  농구공 이미지 불러오기, LAB + CLAHE 강화 binary mask (입력 데이터)
    셀 [188]         TODO: Morphological operation → 외곽선 검출 → 최종 농구공 영역·외곽선 선택
    셀 [191]         최소 외접원 (Minimum Enclosing Circle)
    셀 [194]         무게중심 (Centroid)
    셀 [196]         무게중심 + 최소 외접원 반지름
    셀 [198]         최소 외접원 vs. 무게중심 최종 비교

결과 그림은 화면에 띄우지 않고 results/ 폴더에 PNG로 저장한다.
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend("Agg")  # 창을 띄우지 않고 파일로만 저장 (SSH 접속 중에도 실행 가능)
REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGE_PATH = REPO_ROOT / "src" / "images" / "basketball_crop.jpg"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# 셀 [167]의 LAB 색상 범위
LOWER_BASKETBALL_LAB = np.array([70, 130, 70])
UPPER_BASKETBALL_LAB = np.array([180, 230, 185])

# 셀 [188] Morphological operation 설정 (iteration 비교: results/00_iteration_comparison.png)
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
OPEN_ITERATIONS = 2
CLOSE_ITERATIONS = 3


def load_enhanced_basketball():
    """셀 [167]: LAB 변환 → L 채널 CLAHE 강화 → binary mask 생성."""
    basketball = cv2.imread(str(IMAGE_PATH))
    if basketball is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없음: {IMAGE_PATH}")

    basketball_lab = cv2.cvtColor(basketball, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(basketball_lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_clahe_enhanced = clahe.apply(l)
    basketball_enhanced_lab = cv2.merge((l_clahe_enhanced, a, b))

    mask_enhanced = cv2.inRange(basketball_enhanced_lab, LOWER_BASKETBALL_LAB, UPPER_BASKETBALL_LAB)
    basketball_enhanced_rgb = cv2.cvtColor(basketball_enhanced_lab, cv2.COLOR_LAB2RGB)
    return mask_enhanced, basketball_enhanced_rgb


def apply_morphology(mask, open_iterations, close_iterations):
    """Opening으로 작은 노이즈 제거 → Closing으로 공 내부의 빈 공간을 채운다."""
    mask_opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL, iterations=open_iterations)
    mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, KERNEL, iterations=close_iterations)
    return mask_opened, mask_closed


def find_largest_contour(mask):
    """바깥쪽 외곽선을 모두 찾고, 그중 면적이 가장 큰 것(농구공)을 고른다."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return contours, None
    return contours, max(contours, key=cv2.contourArea)


def save_figure(file_name):
    for ax in plt.gcf().axes:
        ax.axis("off")
    plt.savefig(RESULTS_DIR / file_name, bbox_inches="tight")
    plt.close()
    print(f"저장: {RESULTS_DIR.name}/{file_name}")


def compare_iterations(mask_enhanced):
    """Opening/Closing 횟수 조합별 결과 비교 (OPEN_ITERATIONS, CLOSE_ITERATIONS 선택 근거)."""
    plt.figure(figsize=(16, 16))
    for row, open_iterations in enumerate((1, 2, 3)):
        for col, close_iterations in enumerate((1, 2, 3)):
            _, mask_closed = apply_morphology(mask_enhanced, open_iterations, close_iterations)
            contours, largest_contour = find_largest_contour(mask_closed)

            preview = cv2.cvtColor(mask_closed, cv2.COLOR_GRAY2RGB)
            cv2.drawContours(preview, [largest_contour], -1, (255, 0, 0), 2)

            plt.subplot(3, 3, row * 3 + col + 1)
            plt.imshow(preview)
            plt.title(f"Open x{open_iterations}, Close x{close_iterations} (contours: {len(contours)})")
    save_figure("00_iteration_comparison.png")


def select_ball_contour(mask_enhanced, basketball_enhanced_rgb):
    """셀 [188]: 최종 binary mask 생성 → 외곽선 검출 → 최종 농구공 영역·외곽선 선택."""
    # TODO: Morphological operation을 적용하여 최종 binary mask 생성
    ball_mask_opened, final_ball_mask = apply_morphology(mask_enhanced, OPEN_ITERATIONS, CLOSE_ITERATIONS)

    # TODO: Binary mask로 외곽선 검출
    # TODO: 최종 농구공 영역 및 외곽선 선택
    contours, largest_contour = find_largest_contour(final_ball_mask)
    print(f"검출된 외곽선 {len(contours)}개 중 가장 큰 면적: {cv2.contourArea(largest_contour):.0f}px")

    # 가장 큰 contour만 채운 binary mask = 최종 농구공 영역
    largest_mask = np.zeros_like(final_ball_mask)
    cv2.drawContours(largest_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    final_img = basketball_enhanced_rgb.copy()
    mask_color = np.zeros_like(final_img)
    mask_color[:, :, 0] = largest_mask
    final_img = cv2.addWeighted(final_img, 1.0, mask_color, 0.15, 0)
    cv2.drawContours(final_img, [largest_contour], -1, (255, 0, 0), 2)

    plt.figure(figsize=(20, 5))
    plt.subplot(1, 4, 1), plt.imshow(mask_enhanced, cmap="gray"), plt.title("Enhanced Mask")
    plt.subplot(1, 4, 2), plt.imshow(ball_mask_opened, cmap="gray"), plt.title(f"Opening x{OPEN_ITERATIONS}")
    plt.subplot(1, 4, 3), plt.imshow(final_ball_mask, cmap="gray"), plt.title(f"& Closing x{CLOSE_ITERATIONS}")
    plt.subplot(1, 4, 4), plt.imshow(final_img), plt.title("Largest Contour (Basketball)")
    save_figure("01_ball_contour.png")

    return final_ball_mask, contours


def draw_ball_region(final_ball_mask, contours, basketball_enhanced_rgb):
    """셀 [191]~[198] 공통 부분: 가장 큰 contour를 반투명 빨간색 영역과 외곽선으로 그린다."""
    final_img_enhanced = basketball_enhanced_rgb.copy()

    # 가장 큰 contour
    largest_contour = max(contours, key=cv2.contourArea)

    # 가장 큰 contour만 채운 binary mask 생성
    largest_mask = np.zeros_like(final_ball_mask)
    cv2.drawContours(largest_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    # 가장 큰 contour만 채운 binary mask를 반투명 빨간색으로 overlay
    mask_color = np.zeros_like(final_img_enhanced)
    mask_color[:, :, 0] = largest_mask
    alpha = 0.15
    final_img_enhanced = cv2.addWeighted(final_img_enhanced, 1.0, mask_color, alpha, 0)

    # Contour 그리기 (빨간색 외곽선)
    cv2.drawContours(final_img_enhanced, [largest_contour], -1, (255, 0, 0), 2)
    return final_img_enhanced, largest_contour


def min_enclosing_circle(largest_contour):
    """최소 외접원 중심 및 반지름."""
    (x, y), radius = cv2.minEnclosingCircle(largest_contour)
    return (int(x), int(y)), int(radius)


def centroid(largest_contour, fallback):
    """Moments를 통해 무게중심 (Centroid) 산출: C_x = M10 / M00, C_y = M01 / M00."""
    moments = cv2.moments(largest_contour)
    if moments["m00"] != 0:
        c_x = int(moments["m10"] / moments["m00"])
        c_y = int(moments["m01"] / moments["m00"])
        return (c_x, c_y)
    return fallback


def save_with_original(final_img_enhanced, basketball_enhanced_rgb, title, file_name):
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1), plt.imshow(basketball_enhanced_rgb), plt.title("LAB (CLAHE) Enhanced Image")
    plt.subplot(1, 2, 2), plt.imshow(final_img_enhanced), plt.title(title)
    save_figure(file_name)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    mask_enhanced, basketball_enhanced_rgb = load_enhanced_basketball()

    compare_iterations(mask_enhanced)

    # 셀 [188]
    final_ball_mask, contours = select_ball_contour(mask_enhanced, basketball_enhanced_rgb)

    # 셀 [191]: 최소 외접원 (파란색)
    final_img_enhanced, largest_contour = draw_ball_region(final_ball_mask, contours, basketball_enhanced_rgb)
    center_circle, radius = min_enclosing_circle(largest_contour)
    cv2.circle(final_img_enhanced, center_circle, radius, (0, 0, 255), 3)
    cv2.circle(final_img_enhanced, center_circle, 5, (0, 0, 255), -1)
    save_with_original(final_img_enhanced, basketball_enhanced_rgb,
                       "Minimum Enclosing Circle", "02_min_enclosing_circle.png")
    print(f"최소 외접원: 중심 {center_circle}, 반지름 {radius}px")

    # 셀 [194]: 무게중심 (노란색 점)
    final_img_enhanced, largest_contour = draw_ball_region(final_ball_mask, contours, basketball_enhanced_rgb)
    ball_centroid = centroid(largest_contour, fallback=(0, 0))
    cv2.circle(final_img_enhanced, ball_centroid, 5, (255, 255, 0), -1)
    save_with_original(final_img_enhanced, basketball_enhanced_rgb, "Centroid", "03_centroid.png")
    print(f"무게중심: {ball_centroid}")

    # 셀 [196]: 무게중심 + 최소 외접원 반지름 (노란색 원)
    final_img_enhanced, largest_contour = draw_ball_region(final_ball_mask, contours, basketball_enhanced_rgb)
    center_circle, radius = min_enclosing_circle(largest_contour)
    ball_centroid = centroid(largest_contour, fallback=center_circle)
    cv2.circle(final_img_enhanced, ball_centroid, radius, (255, 255, 0), 3)
    cv2.circle(final_img_enhanced, ball_centroid, 5, (255, 255, 0), -1)
    save_with_original(final_img_enhanced, basketball_enhanced_rgb,
                       "Centroid + Enclosing Radius", "04_centroid_circle.png")

    # 셀 [198]: 최소 외접원 (파란색) vs. 무게중심 기반 원 (노란색)
    final_img_enhanced, largest_contour = draw_ball_region(final_ball_mask, contours, basketball_enhanced_rgb)
    center_circle, radius = min_enclosing_circle(largest_contour)
    cv2.circle(final_img_enhanced, center_circle, radius, (0, 0, 255), 3)
    cv2.circle(final_img_enhanced, center_circle, 5, (0, 0, 255), -1)
    ball_centroid = centroid(largest_contour, fallback=center_circle)
    cv2.circle(final_img_enhanced, ball_centroid, radius, (255, 255, 0), 3)
    cv2.circle(final_img_enhanced, ball_centroid, 5, (255, 255, 0), -1)
    save_with_original(final_img_enhanced, basketball_enhanced_rgb,
                       "Enclosing Circle (blue) vs. Centroid (yellow)", "05_comparison.png")


if __name__ == "__main__":
    main()
