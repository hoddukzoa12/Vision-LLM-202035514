"""Project01 — 손가락 제스처 화이트보드.

오른손으로 든 초록 펜의 끝을 추적해 화면에 그림을 그리고,
왼손의 펼친 손가락 개수로 모드를 고른다 (1개: 그리기 / 2개: 멈춤 / 손바닥 펴기: 전체 지우기).

구성:
    pen_tracker.py   초록 펜 끝 검출 + 칼만 필터 보정
    hand_gesture.py  MediaPipe 손 검출 + 제스처 손의 손가락 개수
    main.py          모드 판정 + 캔버스 그리기 + 화면 출력 (이 파일)
"""

import cv2
import numpy as np

from hand_gesture import count_fingers, create_hand_detector, draw_hand, find_gesture_hand
from pen_tracker import create_kalman, find_pen_tip, open_camera, reset_kalman, smooth_tip

# 손가락 개수 (엄지 제외) → 모드
MODES = {1: "DRAW", 2: "STOP", 4: "CLEAR"}
GESTURE_HOLD_FRAMES = 5  # 같은 손가락 개수가 이만큼 연속으로 나와야 모드를 바꿈 (흔들림 방지)

INK_COLOR = (255, 0, 255)  # 그림 색 (BGR, 자홍색)
INK_THICKNESS = 5
MAX_JUMP = 80  # 펜 끝이 한 프레임에 이보다 멀리 튀면 오검출로 보고 새 선을 시작 (px)

TEXT_COLOR = (0, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def update_mode(mode, finger_count, candidate, hold_count):
    """같은 손가락 개수가 GESTURE_HOLD_FRAMES 프레임 연속일 때만 모드를 바꾼다.

    candidate는 직전 프레임까지 이어진 손가락 개수, hold_count는 그 연속 프레임 수.
    제스처 손이 없으면 (finger_count is None) 현재 모드를 유지한다.
    반환: (mode, candidate, hold_count)
    """
    if finger_count is None:
        return mode, None, 0

    if finger_count == candidate:
        hold_count += 1
    else:
        candidate = finger_count
        hold_count = 1

    if hold_count >= GESTURE_HOLD_FRAMES and candidate in MODES:
        mode = MODES[candidate]
    return mode, candidate, hold_count


def overlay_canvas(frame, canvas):
    """캔버스에서 그림이 있는 픽셀만 프레임 위에 덮어쓴다."""
    ink = canvas.any(axis=2)  # 세 채널 중 하나라도 0이 아니면 그림이 있는 픽셀
    frame[ink] = canvas[ink]


def draw_status(frame, mode, finger_count, labels, pen_visible):
    """화면 좌측 상단에 모드, 손가락 개수, 검출된 손 라벨, 펜 검출 여부를 표시한다."""
    finger_text = "-" if finger_count is None else str(finger_count)
    lines = (
        f"Mode: {mode}",
        f"Fingers: {finger_text}",
        f"Hands: {' / '.join(labels) if labels else '-'}",
        f"Pen: {'O' if pen_visible else 'X'}",
    )
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (20, 35 + 35 * i), FONT, 0.8, TEXT_COLOR, 2)


def main():
    cap = open_camera()
    hand_detector = create_hand_detector()
    kalman = create_kalman()

    canvas = None
    mode = "STOP"
    candidate, hold_count = None, 0
    prev_tip = None    # 직전 프레임에서 측정한 펜 끝 (칼만 보정 전)
    prev_point = None  # 선을 이을 직전 펜 끝 위치 (칼만 보정 후)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # 좌우 반전 (거울 모드)
        if canvas is None:
            canvas = np.zeros_like(frame)

        # 1. 펜 끝 추적 (칼만 보정)
        tip, _ = find_pen_tip(frame)
        if tip is None:
            prev_tip = None
            point = None
        else:
            # 처음 보였거나 갑자기 멀리 튀면 칼만을 새 위치에서 다시 시작하고, 이전 선과 잇지 않음
            if prev_tip is None or np.hypot(tip[0] - prev_tip[0], tip[1] - prev_tip[1]) > MAX_JUMP:
                reset_kalman(kalman, tip)
                prev_point = None
            prev_tip = tip
            point = smooth_tip(kalman, tip)

        # 2. 제스처 손의 손가락 개수 → 모드
        hand, labels = find_gesture_hand(frame, hand_detector)
        finger_count = count_fingers(hand) if hand is not None else None
        mode, candidate, hold_count = update_mode(mode, finger_count, candidate, hold_count)

        # 3. 캔버스에 그리기
        if mode == "CLEAR":
            canvas[:] = 0
            mode = "STOP"  # 한 번 지운 뒤에는 멈춤 상태로

        if mode == "DRAW" and point is not None:
            if prev_point is not None:
                cv2.line(canvas, prev_point, point, INK_COLOR, INK_THICKNESS)
            prev_point = point
        else:
            prev_point = None  # 멈춤이거나 펜이 안 보이면 선을 끊음

        # 4. 화면 출력: 프레임 + 캔버스 + 손 skeleton + 펜 끝 + 상태
        overlay_canvas(frame, canvas)
        if hand is not None:
            draw_hand(frame, hand)
        if point is not None:
            pen_color = INK_COLOR if mode == "DRAW" else (200, 200, 200)
            cv2.circle(frame, point, 10, pen_color, 2)
        draw_status(frame, mode, finger_count, labels, point is not None)

        cv2.imshow("Finger Gesture Whiteboard", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    hand_detector.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
