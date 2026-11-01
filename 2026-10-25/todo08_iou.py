"""Day06 과제 #3 — 두 Bounding Box의 IoU (Intersection over Union) 계산

출처: 04_DL-Object-Detection.ipynb (B. Intersection over Union (IoU))
    셀 [23]          draw_bbox()
    셀 [30]          TODO: (x, y, w, h) → (x1, y1, x2, y2) 변환 함수, IoU 계산 함수
    셀 [33]          get_contour(): Color Segmentation으로 빨간/초록 사과 Contour 추출
    셀 [34]~[40]     사과 이미지 3장에서 BBox를 그리고 IoU 계산
    셀 [42]          IoU 비교 출력

결과 그림은 화면에 띄우지 않고 results/08_iou_comparison.png로 저장한다.
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend("Agg")  # 창을 띄우지 않고 파일로만 저장

REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = REPO_ROOT / "src" / "images"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

IMAGE_NAMES = ["two_apples1.png", "two_apples2.png", "two_apples3.png"]

RED_COLOR = (255, 0, 0)    # RGB
GREEN_COLOR = (0, 255, 0)  # RGB


# 셀 [30]
# TODO: 1) BBox 좌표를 (x,y,w,h)에서 (x1,y1,x2,y2)로 변환하는 함수 정의
# TODO: 2) IoU 계산 함수 정의
def convert_bbox_coord(bbox):
    """(x, y, w, h) 형식의 BBox를 (x1, y1, x2, y2) 형식으로 변환한다."""
    x, y, w, h = bbox

    # 우측 하단 좌표 = 좌측 상단 좌표 + 크기
    return x, y, x + w, y + h


def calculate_iou(bbox_a, bbox_b):
    """(x1, y1, x2, y2) 형식의 두 BBox의 IoU(교집합 넓이 / 합집합 넓이)를 반환한다."""
    xa_1, ya_1, xa_2, ya_2 = bbox_a
    xb_1, yb_1, xb_2, yb_2 = bbox_b

    # 교집합 박스: 좌측 상단은 두 박스 중 더 안쪽(큰 값), 우측 하단도 더 안쪽(작은 값)
    intersection_x1 = max(xa_1, xb_1)
    intersection_y1 = max(ya_1, yb_1)
    intersection_x2 = min(xa_2, xb_2)
    intersection_y2 = min(ya_2, yb_2)

    # 겹치지 않으면 x2 - x1이 음수가 되므로 0으로 처리
    intersection_w = max(0, intersection_x2 - intersection_x1)
    intersection_h = max(0, intersection_y2 - intersection_y1)

    area_a = (xa_2 - xa_1) * (ya_2 - ya_1)
    area_b = (xb_2 - xb_1) * (yb_2 - yb_1)

    intersection_area = intersection_w * intersection_h
    # 두 넓이를 더하면 겹친 부분이 두 번 들어가므로 한 번 뺌
    union_area = area_a + area_b - intersection_area

    # TODO: ZeroDivisionError 주의
    # 두 박스 모두 넓이가 0이면 합집합도 0 → 겹친 것이 없으므로 0을 반환
    if union_area == 0:
        return 0.0

    return intersection_area / union_area


# 셀 [23]
def draw_bbox(img, coord1, coord2, color, thickness=3, txt=""):
    _, txt_h = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    cv2.rectangle(img, coord1, coord2, color, thickness)
    cv2.putText(img, txt, (coord1[0], coord1[1] - txt_h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


# 셀 [33]
def get_contour(img_bgr):
    red_lower1 = np.array([0, 100, 50])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 100, 50])
    red_upper2 = np.array([180, 255, 255])
    green_lower = np.array([25, 10, 30])
    green_upper = np.array([100, 255, 255])

    img_blr = cv2.GaussianBlur(img_bgr, (7, 7), 0)
    blr_hsv = cv2.cvtColor(img_blr, cv2.COLOR_BGR2HSV)

    red_mask1 = cv2.inRange(blr_hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(blr_hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    grn_mask = cv2.inRange(blr_hsv, green_lower, green_upper)

    contours_red, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_grn, _ = cv2.findContours(grn_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cntr_red, cntr_grn = None, None

    if len(contours_red) > 0:
        cntr_red = max(contours_red, key=cv2.contourArea)
    if len(contours_grn) > 0:
        cntr_grn = max(contours_grn, key=cv2.contourArea)

    return cntr_red, cntr_grn


def detect_apples_iou(image_path):
    """셀 [34]~[40]: 사과 두 개의 BBox를 그린 RGB 이미지와 두 BBox의 IoU를 반환한다."""
    apples = cv2.imread(str(image_path))

    if apples is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {image_path}")

    apples_rgb = cv2.cvtColor(apples, cv2.COLOR_BGR2RGB)

    cntr_red, cntr_grn = get_contour(apples)

    # cv2.boundingRect()는 (x, y, w, h) 형식
    bbox1 = cv2.boundingRect(cntr_red)
    bbox2 = cv2.boundingRect(cntr_grn)

    # cv2.rectangle()은 (x1, y1), (x2, y2) 좌표가 필요
    x_r1, y_r1, x_r2, y_r2 = convert_bbox_coord(bbox1)
    x_g1, y_g1, x_g2, y_g2 = convert_bbox_coord(bbox2)

    draw_bbox(apples_rgb, (x_r1, y_r1), (x_r2, y_r2), RED_COLOR, txt="Red Apple")
    draw_bbox(apples_rgb, (x_g1, y_g1), (x_g2, y_g2), GREEN_COLOR, txt="Green Apple")

    iou = calculate_iou(convert_bbox_coord(bbox1), convert_bbox_coord(bbox2))

    return apples_rgb, iou


def main():
    # IoU 함수 자체 확인 (값을 손으로 계산할 수 있는 경우)
    print("Same box:", calculate_iou((0, 0, 10, 10), (0, 0, 10, 10)))       # 1.0
    print("No overlap:", calculate_iou((0, 0, 10, 10), (20, 20, 30, 30)))   # 0.0
    print("Half overlap:", calculate_iou((0, 0, 10, 10), (5, 0, 15, 10)))   # 50 / 150 = 0.333...
    print("Zero area:", calculate_iou((0, 0, 0, 0), (0, 0, 0, 0)))          # 0.0 (ZeroDivisionError 방지)
    print()

    results = []

    for image_name in IMAGE_NAMES:
        apples_rgb, iou = detect_apples_iou(IMAGE_DIR / image_name)
        results.append((apples_rgb, iou))
        print(f"{image_name} IoU: {iou:.4f}")

    # 셀 [42]
    RESULTS_DIR.mkdir(exist_ok=True)

    plt.figure(figsize=(16, 6))

    for i, (apples_rgb, iou) in enumerate(results):
        plt.subplot(1, 3, i + 1), plt.imshow(apples_rgb), plt.title(f"IoU: {iou:.4f}")

    for ax in plt.gcf().axes:
        ax.axis("off")

    plt.savefig(RESULTS_DIR / "08_iou_comparison.png", bbox_inches="tight")
    print("Saved:", RESULTS_DIR / "08_iou_comparison.png")


if __name__ == "__main__":
    main()
