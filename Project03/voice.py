"""Project03 — 음성 입출력 (STT, TTS, 주변 소음 측정).

    speech_to_text()    마이크 녹음 → Whisper.cpp → 텍스트
    text_to_speech()    텍스트 → Piper → 스피커 재생
    measure_noise_db()  마이크 1초 녹음 → 음량(dBFS)

단독 실행하면 소음 기준값(NOISE_THRESHOLD_DB)을 정하기 위해 1초마다 음량을 출력한다.

출처:
    06-Vision_LLM_Multimodal_Systems.ipynb  C. STT, TTS (arecord, whisper-cli, piper, aplay)
    2026-11-07/practice/def.py  speech_to_text(), text_to_speech() 함수
"""

import math
import os
import signal
import subprocess
import wave
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent
AUDIO_DIR = PROJECT_DIR / "audio"

# arecord -l 의 "card 3: Microphone [USB Microphone]"에서 대괄호 앞 이름(Microphone)으로 지정.
# card 번호는 USB를 꽂는 순서·부팅마다 바뀌지만 이름은 그대로라 같은 마이크를 찾는다.
MIC_DEVICE = "plughw:CARD=Microphone,DEV=0"
RECORD_SECONDS = 5
NOISE_RECORD_SECONDS = 1
RECORD_TIMEOUT_MARGIN = 3  # 녹음 시간보다 이만큼(초) 더 걸리면 장치 문제로 보고 중단

WHISPER_PATH = REPO_ROOT / "whisper.cpp" / "build-cpu" / "bin" / "whisper-cli"
WHISPER_MODEL = REPO_ROOT / "whisper.cpp" / "models" / "ggml-base.bin"
PIPER_PYTHON = REPO_ROOT / ".piper_venv" / "bin" / "python"
PIPER_MODEL = REPO_ROOT / "src" / "models" / "Piper" / "ko_KR-kss-medium.onnx"

QUESTION_AUDIO = AUDIO_DIR / "question.wav"
NOISE_AUDIO = AUDIO_DIR / "noise.wav"
RESPONSE_AUDIO = AUDIO_DIR / "response.wav"

INT16_MAX = 32768   # 16-bit 오디오 샘플의 최대 크기 (S16_LE)
SILENCE_DB = -100.0  # 완전 무음일 때 log(0)을 피하기 위한 값


def record_audio(path, seconds):
    """06 C: arecord로 16kHz, 16-bit, mono WAV를 녹음한다. 실패하거나 멈추면 RuntimeError."""
    AUDIO_DIR.mkdir(exist_ok=True)
    # start_new_session=True: pasuspender와 그 자식 arecord를 한 프로세스 그룹으로 묶어,
    # 멈췄을 때 둘 다 종료할 수 있게 한다 (pasuspender만 죽이면 arecord가 남아 장치를 계속 점유함)
    process = subprocess.Popen(
        [
            "pasuspender", "--",
            "arecord",
            "-D", MIC_DEVICE,
            "-f", "S16_LE",
            "-r", "16000",
            "-c", "1",
            "-d", str(seconds),
            "-q",
            str(path),
        ],
        start_new_session=True,
    )
    try:
        process.wait(timeout=seconds + RECORD_TIMEOUT_MARGIN)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        raise RuntimeError(f"녹음이 {seconds + RECORD_TIMEOUT_MARGIN}초 안에 끝나지 않음 ({MIC_DEVICE})")
    if process.returncode != 0:
        raise RuntimeError(f"녹음 실패 ({MIC_DEVICE}, 종료 코드 {process.returncode})")


def speech_to_text():
    """RECORD_SECONDS초 녹음한 뒤 Whisper.cpp로 한국어 텍스트를 반환한다."""
    record_audio(QUESTION_AUDIO, RECORD_SECONDS)

    result = subprocess.run(
        [
            str(WHISPER_PATH),
            "-m", str(WHISPER_MODEL),
            "-f", str(QUESTION_AUDIO),
            "-l", "ko",
            "--no-timestamps",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    text = result.stdout.strip()

    # 아무 말도 없으면 Whisper가 [BLANK_AUDIO] 같은 표시만 출력한다
    if text.startswith("[") and text.endswith("]"):
        return ""
    return text


def text_to_speech(text):
    """Piper로 음성 파일을 만들고 aplay로 재생한다."""
    AUDIO_DIR.mkdir(exist_ok=True)
    subprocess.run(
        [
            str(PIPER_PYTHON), "-m", "piper",
            "-m", str(PIPER_MODEL),
            "-f", str(RESPONSE_AUDIO),
            "--",
            text,
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(["aplay", "-q", str(RESPONSE_AUDIO)], check=True)


def measure_noise_db():
    """마이크를 1초 녹음해 음량을 dBFS로 반환한다 (0이 최대, 조용할수록 큰 음수)."""
    record_audio(NOISE_AUDIO, NOISE_RECORD_SECONDS)

    with wave.open(str(NOISE_AUDIO), "rb") as wav_file:
        frames = wav_file.readframes(wav_file.getnframes())

    # 바이트 → int16 샘플 배열 → float로 바꿔 제곱해도 넘치지 않게 함
    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
    if samples.size == 0:
        return SILENCE_DB

    # RMS (Root Mean Square): 샘플 제곱의 평균의 제곱근 = 평균적인 소리 크기
    rms = np.sqrt(np.mean(samples ** 2))
    if rms == 0:
        return SILENCE_DB

    # dBFS = 20 · log10(RMS / 최대값)
    return 20 * math.log10(rms / INT16_MAX)


if __name__ == "__main__":
    # 소음 기준값 정하기: 조용할 때와 시끄러울 때 값을 비교해 그 사이로 NOISE_THRESHOLD_DB를 정한다
    print("1초마다 주변 소음을 측정합니다. (Ctrl+C 종료)")
    try:
        while True:
            print(f"{measure_noise_db():6.1f} dBFS")
    except KeyboardInterrupt:
        pass
