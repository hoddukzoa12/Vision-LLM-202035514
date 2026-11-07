"""Project03 — 공부 집중 도우미 실행 진입점.

assistant.py(Gemma + 음성)와 vision.py(카메라 + YOLO)를 별도 프로세스로 함께 실행한다.
카메라 창에서 q를 눌러 vision.py가 끝나면 assistant.py도 종료한다.

두 프로그램을 한 프로세스에 합치지 않는 이유는 06 A-7) 파이프라인 분리와 같다:
Gemma 추론과 음성 녹음이 진행되는 동안에도 카메라 화면과 이벤트 감지는 멈추지 않아야 한다.
"""

import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
ASSISTANT_SCRIPT = PROJECT_DIR / "assistant.py"
VISION_SCRIPT = PROJECT_DIR / "vision.py"
STOP_TIMEOUT = 5  # assistant.py가 종료 요청 후 이 시간(초) 안에 끝나지 않으면 강제 종료


def main():
    # sys.executable: 지금 main.py를 실행한 파이썬(.venv)으로 두 스크립트도 실행
    assistant = subprocess.Popen([sys.executable, str(ASSISTANT_SCRIPT)])
    try:
        subprocess.run([sys.executable, str(VISION_SCRIPT)], check=False)
    finally:
        assistant.terminate()
        try:
            assistant.wait(timeout=STOP_TIMEOUT)
        except subprocess.TimeoutExpired:
            assistant.kill()


if __name__ == "__main__":
    main()
