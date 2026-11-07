"""Project03 — 공부 집중 도우미 (Vision 프로세스).

카메라 프레임을 YOLO로 탐지하고, focus_tracker로 이벤트(폰 사용, 자리비움)를 판정해
탐지 결과·통계·이벤트를 state/state.json에 저장한다. assistant.py가 이 파일을 읽어 Gemma와 음성으로 응답한다.

    s  음성 질문 요청 (assistant.py가 녹음을 시작)
    q  종료

출처:
    04_DL-Object-Detection.ipynb  F. FPS (EMA), H/G. INT8 TensorRT 엔진 실시간 탐지
    06-Vision_LLM_Multimodal_Systems.ipynb  A-2) YOLO 결과 → JSON, A-7) 파이프라인 분리 (임시 파일 → os.replace)
    2026-11-07/practice/int8.py  USB 카메라 (YUY2) 파이프라인
"""

import json
import os
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from focus_tracker import FocusTracker

PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent
YOLO_MODEL_PATH = REPO_ROOT / "src" / "models" / "YOLO" / "yolo11n_int8.engine"
STATE_DIR = PROJECT_DIR / "state"
STATE_PATH = STATE_DIR / "state.json"
TEMP_STATE_PATH = STATE_DIR / "state_tmp.json"

PIPELINE = (
    "v4l2src device=/dev/video0 ! "
    "video/x-raw,format=YUY2,width=1280,height=720,framerate=30/1 ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 "
    "max-size-bytes=0 max-size-time=0 ! "
    "appsink drop=true max-buffers=1 sync=false"
)

# 공부 상황과 관련된 COCO 클래스만 탐지 (0 person, 63 laptop, 67 cell phone, 73 book)
TARGET_CLASSES = [0, 63, 67, 73]
CONF_THRESHOLD = 0.4
IOU_THRESHOLD = 0.5

STATE_WRITE_INTERVAL = 0.5  # 이벤트가 없을 때 state.json 저장 주기 (초)
ALERT_SECONDS = 3.0         # 이벤트 문구를 화면에 띄워 두는 시간 (초)

TEXT_COLOR = (0, 255, 0)
ALERT_COLOR = (0, 0, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def result_to_objects(result):
    """06 A-2): YOLO 결과 하나를 [{class, confidence, bbox}, ...] 목록으로 바꾼다."""
    objects = []
    for box in result.boxes:
        class_id = int(box.cls[0].item())
        x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
        objects.append(
            {
                "class": result.names[class_id],
                "confidence": round(float(box.conf[0].item()), 3),
                "bbox": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)},
            }
        )
    return objects


def write_state(state):
    """06 A-7): 임시 파일에 다 쓴 뒤 교체해, assistant.py가 쓰다 만 파일을 읽지 않게 한다."""
    with open(TEMP_STATE_PATH, "w", encoding="utf-8") as file:
        json.dump(state, file, ensure_ascii=False, indent=4)
    os.replace(TEMP_STATE_PATH, STATE_PATH)


def draw_status(frame, stats, fps, alert_text):
    """화면 좌측 상단에 FPS와 집중 통계, 최근 이벤트를 표시한다."""
    lines = (
        f"FPS: {fps:.1f}",
        f"Focus: {stats['focus_seconds'] // 60}m {stats['focus_seconds'] % 60}s",
        f"Phone: {stats['phone_count']}  Away: {stats['absent_count']}",
    )
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (20, 40 + 35 * i), FONT, 0.9, TEXT_COLOR, 2)
    if alert_text:
        cv2.putText(frame, alert_text, (20, 40 + 35 * len(lines)), FONT, 0.9, ALERT_COLOR, 2)


def main():
    if not YOLO_MODEL_PATH.exists():
        raise FileNotFoundError(f"YOLO 엔진이 없습니다: {YOLO_MODEL_PATH} (04 노트북 INT8 변환을 먼저 실행)")
    STATE_DIR.mkdir(exist_ok=True)

    yolo = YOLO(str(YOLO_MODEL_PATH))
    cap = cv2.VideoCapture(PIPELINE, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError("카메라를 열 수 없습니다.")

    tracker = FocusTracker(time.time())
    event_id = 0             # 이벤트마다 1씩 증가. assistant.py는 이 번호로 새 이벤트를 구분
    last_event = None
    last_write = 0.0
    displayed_fps = 0.0
    alert_text = ""
    alert_until = 0.0

    print("s : 음성 질문 / q : 종료")

    while True:
        start_time = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            break
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        result = yolo.predict(
            source=frame,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            classes=TARGET_CLASSES,
            verbose=False,
        )[0]
        objects = result_to_objects(result)

        now = time.time()
        events = tracker.update([obj["class"] for obj in objects], now)
        if key == ord("s"):
            events.append("question")

        for event in events:
            event_id += 1
            last_event = {"id": event_id, "type": event, "time": now}
            alert_until = now + ALERT_SECONDS
            alert_text = {"phone": "PHONE DETECTED", "absent": "AWAY", "question": "LISTENING..."}[event]
            print(f"[이벤트 {event_id}] {event}")

        stats = tracker.summary(now)

        # 이벤트가 생겼으면 바로, 아니면 STATE_WRITE_INTERVAL마다 저장
        if events or now - last_write >= STATE_WRITE_INTERVAL:
            write_state(
                {
                    "time": now,
                    "image_width": frame.shape[1],
                    "image_height": frame.shape[0],
                    "objects": objects,
                    "stats": stats,
                    "last_event": last_event,
                }
            )
            last_write = now

        # 04 F. FPS (EMA)
        current_fps = 1.0 / (time.perf_counter() - start_time)
        displayed_fps = current_fps if displayed_fps == 0 else 0.9 * displayed_fps + 0.1 * current_fps

        output_frame = result.plot()
        draw_status(output_frame, stats, displayed_fps, alert_text if now < alert_until else "")
        cv2.imshow("Study Focus Assistant", output_frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
