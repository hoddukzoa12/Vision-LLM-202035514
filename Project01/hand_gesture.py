"""Project01 2단계 — MediaPipe 손 검출 + 제스처 손의 펼친 손가락 개수.

단독 실행하면 검출된 손의 Left/Right 라벨과 제스처 손의 손가락 개수를 화면에 표시한다.
출처: 02_Computer-Vision.ipynb
    셀 [275]  M. MediaPipe 손 검출, landmark·skeleton 그리기
    셀 [277]  관절 각도 기반 펼친 손가락 개수 계산 (rule-based)
"""

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from pen_tracker import open_camera

MODEL_PATH = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"

# 모드를 고르는 손 (사용자 기준 실제 손).
# hand_gesture.py를 단독 실행해 왼손만 들었을 때 화면에 "Left"가 나오는지 확인하고,
# 반대로 나오면 "Right"로 바꾼다.
GESTURE_HAND = "Left"

CONNECTIONS = vision.HandLandmarksConnections.HAND_CONNECTIONS  # 각 landmark를 잇는 연결선 정보
FINGER_TIPS = (4, 8, 12, 16, 20)  # 손가락 끝 landmark의 index
ANGLE_THRESHOLD = 160             # 손가락 펼쳐짐 여부 판단 각도 임계값

# 각 손가락의 각도를 계산할 landmark index
FINGER_ANGLE_POINTS = (
    (1, 2, 3),      # 엄지: 2번 중심
    (5, 6, 7),      # 검지: 6번 중심
    (9, 10, 11),    # 중지: 10번 중심
    (13, 14, 15),   # 약지: 14번 중심
    (17, 18, 19),   # 소지: 18번 중심
)


def calculate_angle(p1, p2, p3):
    """셀 [277]: p2를 중심으로 p1-p2-p3가 이루는 각도(degree)."""
    vector1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
    vector2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])

    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    cosine = np.dot(vector1, vector2) / (magnitude1 * magnitude2)
    cosine = np.clip(cosine, -1.0, 1.0)

    return np.degrees(np.arccos(cosine))


def create_hand_detector():
    """셀 [275]: 모델 경로와 최대 손 개수(2)로 손 검출 객체 생성."""
    base_option = python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.HandLandmarkerOptions(base_options=base_option, num_hands=2)
    return vision.HandLandmarker.create_from_options(options)


def find_gesture_hand(frame, hand_detector):
    """좌우 반전된 BGR 프레임에서 손을 검출해 (제스처 손 landmark, 모든 손 라벨)을 반환한다.

    좌우 반전된 화면에서는 MediaPipe의 Left/Right가 실제 손과 반대로 나오므로,
    셀 [277]처럼 라벨을 뒤집어 사용자 기준 실제 손으로 맞춘다.
    제스처 손이 없으면 landmark 자리에 None을 반환한다.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hand_detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

    labels = ["Left" if handedness[0].category_name == "Right" else "Right" for handedness in result.handedness]
    for hand, label in zip(result.hand_landmarks, labels):
        if label == GESTURE_HAND:
            return hand, labels
    return None, labels


def count_fingers(hand):
    """셀 [277]: 손가락마다 관절 각도가 ANGLE_THRESHOLD 이상이면 펼친 것으로 센다 (한 손만)."""
    finger_count = 0
    for point1_idx, point2_idx, point3_idx in FINGER_ANGLE_POINTS:
        angle = calculate_angle(hand[point1_idx], hand[point2_idx], hand[point3_idx])
        if angle >= ANGLE_THRESHOLD:
            finger_count += 1
    return finger_count


def draw_hand(frame, hand):
    """셀 [275]: landmark 연결선(skeleton)과 관절 점을 그린다. 손가락 끝은 빨간 점."""
    h, w = frame.shape[:2]
    points = [(int(p.x * w), int(p.y * h)) for p in hand]

    for c in CONNECTIONS:
        cv2.line(frame, points[c.start], points[c.end], (0, 255, 0), 2)

    for i, point in enumerate(points):
        color = (0, 0, 255) if i in FINGER_TIPS else (255, 0, 0)
        cv2.circle(frame, point, 6 if i in FINGER_TIPS else 4, color, -1)


def main():
    cap = open_camera()
    hand_detector = create_hand_detector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # 좌우 반전 (거울 모드)
        hand, labels = find_gesture_hand(frame, hand_detector)

        if hand is not None:
            draw_hand(frame, hand)
            finger_text = f"{GESTURE_HAND} fingers: {count_fingers(hand)}"
        else:
            finger_text = f"{GESTURE_HAND} hand: not found"

        cv2.putText(frame, f"Hands: {' / '.join(labels)}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, finger_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Hand Gesture", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    hand_detector.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
