"""Day06 과제 #3 — Confidence Thresholding과 NMS (Non-Maximum Suppression)

출처: 04_DL-Object-Detection.ipynb (C. Non-Maximum Suppression (NMS))
    셀 [48]          random_bbox(), random_conf()
    셀 [49]          무작위 BBox 20개 생성
    셀 [51]          TODO: Confidence Thresholding 함수
    셀 [52]          TODO: NMS 함수
    셀 [54], [55]    직접 구현한 NMS 적용
    셀 [58]          OpenCV cv2.dnn.NMSBoxes()로 NMS 적용
    셀 [62]~[65]     두 결과 비교

노트북과 다른 점:
    - 실행할 때마다 같은 BBox가 나오도록 random.seed(SEED)로 난수를 고정했다.
    - 셀 [54]는 prune_low_conf()의 결과가 아닌 원래 boxes를 apply_nms()에 넘기고,
      셀 [55], [63]은 남은 박스에 원래 scores[i]를 표시한다.
      여기서는 걸러진 결과(new_boxes, new_scores)를 이어서 쓰도록 바꿨다.

결과 그림은 화면에 띄우지 않고 results/09_nms_comparison.png로 저장한다.
"""

import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage.data

from todo08_iou import calculate_iou, draw_bbox

plt.switch_backend("Agg")  # 창을 띄우지 않고 파일로만 저장

RESULTS_DIR = Path(__file__).resolve().parent / "results"

SEED = 42
N = 20
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.2


# 셀 [48]
def random_bbox(img, min_w=100, min_h=400, max_w=600, max_h=600):
    img_h, img_w = img.shape[:2]

    w = random.randint(min_w, min(max_w, img_w))
    h = random.randint(min_h, min(max_h, img_h))

    x1 = random.randint(0, img_w - w)
    y1 = random.randint(0, img_h - h)

    return x1, y1, x1 + w, y1 + h


def random_conf():
    return min(1.0, max(0.0, random.gauss(0.6, 0.1)))


def random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


# 셀 [51]
# TODO: Confidence Thresholding 함수 정의
# TODO: Confidence Score가 특정 임계값보다 낮으면 해당 BBox 제거
def prune_low_conf(boxes, scores, conf_threshold=0.5):
    """
    Confidence Thresholding 적용

    Parameters
    ----------
    boxes : list
        Bounding Box 목록
        각 박스는 (x1, y1, x2, y2) 형식

    scores : list
        각 Bounding Box의 Confidence Score

    conf_threshold : float
        Bounding Box를 유지하기 위한 최소 Confidence Score

    Returns
    -------
    pruned_boxes : list
        Confidence Score가 임계값 이상인 Bounding Box 목록

    pruned_scores : list
        Confidence Score가 임계값 이상인 Bounding Box의 Confidence Score 목록
    """

    pruned_boxes = []
    pruned_scores = []

    # 박스와 점수를 같은 순서로 하나씩 꺼내서, 임계값 이상인 것만 남김
    for box, score in zip(boxes, scores):
        if score >= conf_threshold:
            pruned_boxes.append(box)
            pruned_scores.append(score)

    return pruned_boxes, pruned_scores


# 셀 [52]
# TODO: NMS 함수 정의
# TODO: np.sort() 함수를 사용하면 1차원 배열을 정렬
# TODO: np.argsort() 함수를 사용하면 1차원 배열의 정렬된 인덱스를 반환
def apply_nms(boxes, scores, iou_threshold=0.5):
    """
    NMS 적용

    Parameters
    ----------
    boxes : list
        Bounding Box 목록
        각 박스는 (x1, y1, x2, y2) 형식

    scores : list
        각 Bounding Box의 Confidence Score

    iou_threshold : float
        중복 Bounding Box를 제거할 IoU 기준값

    Returns
    -------
    nms_boxes : list
        NMS 적용 후 유지된 Bounding Box 목록

    nms_scores : list
        NMS 적용 후 유지된 Bounding Box의 Confidence Score 목록
    """

    if len(boxes) == 0:
        return [], []

    # np.argsort()는 오름차순 인덱스 → [::-1]로 뒤집어 Confidence가 높은 순서로 만듦
    order = np.argsort(scores)[::-1].tolist()

    nms_boxes = []
    nms_scores = []

    while len(order) > 0:
        # 남은 박스 중 Confidence가 가장 높은 박스를 선택해 결과에 추가
        best_index = order.pop(0)
        nms_boxes.append(boxes[best_index])
        nms_scores.append(scores[best_index])

        # 선택한 박스와 IoU가 임계값보다 큰(같은 객체를 가리키는) 박스는 제거하고 나머지만 남김
        order = [
            index for index in order
            if calculate_iou(boxes[best_index], boxes[index]) <= iou_threshold
        ]

    return nms_boxes, nms_scores


# 셀 [58]
def apply_nms_cv2(boxes, scores, conf_threshold=0.5, iou_threshold=0.5):
    """OpenCV의 cv2.dnn.NMSBoxes()로 Confidence Thresholding과 NMS를 함께 적용한다."""

    if len(boxes) == 0:
        return [], []

    # cv2.dnn.NMSBoxes()는 (x, y, width, height) 형식을 사용
    boxes_xywh = []

    for box in boxes:
        x1, y1, x2, y2 = box

        width = x2 - x1
        height = y2 - y1

        boxes_xywh.append([int(x1), int(y1), int(width), int(height)])

    keep_indices = cv2.dnn.NMSBoxes(
        bboxes=boxes_xywh,
        scores=scores,
        score_threshold=conf_threshold,
        nms_threshold=iou_threshold,
    )

    keep_indices = np.array(keep_indices).reshape(-1).tolist()

    nms_boxes = [boxes[index] for index in keep_indices]
    nms_scores = [scores[index] for index in keep_indices]

    return nms_boxes, nms_scores


def draw_boxes(boxes, scores):
    """허블 이미지 위에 BBox와 Confidence Score를 그려 반환한다."""
    hubble = skimage.data.hubble_deep_field()

    for box, score in zip(boxes, scores):
        draw_bbox(hubble, (box[0], box[1]), (box[2], box[3]), random_color(), txt=f"{score:.2f}")

    return hubble


def main():
    random.seed(SEED)

    # 셀 [49]
    hubble = skimage.data.hubble_deep_field()

    boxes = []
    scores = []

    for _ in range(N):
        boxes.append(random_bbox(hubble))
        scores.append(random_conf())

    # 셀 [54]
    print("Before:", len(boxes), len(scores))

    new_boxes, new_scores = prune_low_conf(boxes, scores, conf_threshold=CONF_THRESHOLD)
    print("After confidence thresholding:", len(new_boxes), len(new_scores))

    new_boxes, new_scores = apply_nms(new_boxes, new_scores, iou_threshold=IOU_THRESHOLD)
    print("After custom NMS:", len(new_boxes), len(new_scores))

    # 셀 [62]
    cv2_new_boxes, cv2_new_scores = apply_nms_cv2(
        boxes,
        scores,
        conf_threshold=CONF_THRESHOLD,
        iou_threshold=IOU_THRESHOLD,
    )
    print("After cv2 NMS:", len(cv2_new_boxes), len(cv2_new_scores))

    # 두 구현이 같은 박스를 남겼는지 확인 (둘 다 Confidence가 높은 순서로 반환)
    print("Same result as cv2.dnn.NMSBoxes():", new_boxes == cv2_new_boxes)

    # 셀 [55], [60], [63], [65]
    RESULTS_DIR.mkdir(exist_ok=True)

    plt.figure(figsize=(24, 8))

    plt.subplot(1, 3, 1), plt.imshow(draw_boxes(boxes, scores)), plt.title(f"Before NMS ({len(boxes)} boxes)")
    plt.subplot(1, 3, 2), plt.imshow(draw_boxes(new_boxes, new_scores)), plt.title("Custom NMS Result")
    plt.subplot(1, 3, 3), plt.imshow(draw_boxes(cv2_new_boxes, cv2_new_scores)), plt.title("CV2 NMS Result")

    for ax in plt.gcf().axes:
        ax.axis("off")

    plt.savefig(RESULTS_DIR / "09_nms_comparison.png", bbox_inches="tight")
    print("Saved:", RESULTS_DIR / "09_nms_comparison.png")


if __name__ == "__main__":
    main()
