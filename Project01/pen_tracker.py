"""Project01 1단계 — 초록 펜 끝 검출 + 칼만 필터 보정.

단독 실행하면 펜 끝(빨간 점)과 칼만 보정 위치(노란 원)를 화면에 표시한다.
출처: 02_Computer-Vision.ipynb
    셀 [245]  USB 카메라 (YUYV) GStreamer 파이프라인
    셀 [255]  K. 실시간 객체 검출 (초록 LAB range, 전처리, 모폴로지)
    셀 [266]  L. OpenCV Kalman Filter 설정
"""

import cv2
import numpy as np

# 셀 [245] USB 카메라 (YUYV) 파이프라인
PIPELINE = (
    "v4l2src device=/dev/video0 ! "
    "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 "
    "max-size-bytes=0 max-size-time=0 ! "
    "appsink drop=true max-buffers=1 sync=false"
)

# 셀 [255] 초록 LAB range
GREEN_LOWER = np.array([30, 60, 90], dtype=np.uint8)
GREEN_UPPER = np.array([230, 115, 180], dtype=np.uint8)
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
MIN_AREA = 500  # 이보다 작은 contour는 노이즈로 보고 버림


def open_camera():
    """Jetson은 셀 [245] GStreamer 파이프라인, GStreamer가 없는 환경(Mac 등)은 기본 카메라 0번을 연다."""
    cap = cv2.VideoCapture(PIPELINE, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


def find_pen_tip(frame):
    """초록 펜 contour의 가장 위쪽 점(펜 끝)과 binary mask를 반환한다. 펜이 없으면 tip은 None."""
    frame_blur = cv2.GaussianBlur(frame, (7, 7), 0)
    frame_lab = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2LAB)
    mask = cv2.inRange(frame_lab, GREEN_LOWER, GREEN_UPPER)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > MIN_AREA]
    if len(contours) == 0:
        return None, mask

    pen = max(contours, key=cv2.contourArea)
    top_idx = pen[:, :, 1].argmin()  # contour 점들 중 y가 가장 작은 (= 가장 위쪽) 점의 index
    x, y = pen[top_idx][0]
    return (int(x), int(y)), mask


def create_kalman():
    """셀 [266] 설정 그대로: 상태 [x, y, vx, vy], 측정 [x, y]."""
    kalman = cv2.KalmanFilter(4, 2)
    kalman.transitionMatrix = np.array(
        [
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    kalman.measurementMatrix = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ],
        dtype=np.float32,
    )
    kalman.processNoiseCov = np.array(
        [
            [1e-2, 0, 0, 0],
            [0, 1e-2, 0, 0],
            [0, 0, 5e-2, 0],
            [0, 0, 0, 5e-2],
        ],
        dtype=np.float32,
    )
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
    kalman.errorCovPost = np.eye(4, dtype=np.float32)
    return kalman


def reset_kalman(kalman, tip):
    """펜이 (다시) 나타났을 때 현재 위치, 속도 0으로 상태를 초기화한다."""
    state = np.array([[tip[0]], [tip[1]], [0.0], [0.0]], dtype=np.float32)
    kalman.statePre = state
    kalman.statePost = state.copy()  # predict()는 statePost에서 출발하므로 함께 설정
    kalman.errorCovPost = np.eye(4, dtype=np.float32)


def smooth_tip(kalman, tip):
    """측정한 펜 끝으로 prediction → update 후, 보정된 위치를 반환한다."""
    kalman.predict()
    corrected = kalman.correct(np.array([[tip[0]], [tip[1]]], dtype=np.float32))
    return (int(corrected[0, 0]), int(corrected[1, 0]))


def main():
    cap = open_camera()
    kalman = create_kalman()
    found = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # 좌우 반전 (거울 모드)
        tip, mask = find_pen_tip(frame)

        if tip is None:
            found = False  # 펜이 사라지면 다음에 나타날 때 다시 초기화
        else:
            if not found:
                reset_kalman(kalman, tip)
                found = True
            smoothed = smooth_tip(kalman, tip)

            cv2.circle(frame, tip, 6, (0, 0, 255), -1)        # 측정한 펜 끝 (빨강)
            cv2.circle(frame, smoothed, 12, (0, 255, 255), 2)  # 칼만 보정 위치 (노랑)

        cv2.imshow("Pen Mask", mask)
        cv2.imshow("Pen Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
