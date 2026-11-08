"""Project03 — 공부 집중 도우미 (Assistant 프로세스).

vision.py가 저장하는 state/state.json을 감시하다가 새 이벤트가 생기면 Gemma로 응답을 만들어 음성으로 말한다.
쉬는 동안에는 주변 소음을 측정해, 너무 시끄러운 상태가 이어지면 소음 이벤트로 처리한다.

    phone     폰을 3초 이상 봄        → 다시 집중하라는 한마디
    absent    10초 이상 자리비움      → 자리비움 알림
    noise     소음 기준 초과가 연속    → 조용한 곳을 권하는 한마디
    question  화면에서 s 키          → 음성 질문 녹음 → 현재 책상 상황과 함께 Gemma에 전달 → 음성 답변

출처:
    05-LLM_and_Gemma.ipynb  D. Prompt Engineering (role, content), F. Gemma 로컬 추론 (llama_cpp)
    06-Vision_LLM_Multimodal_Systems.ipynb  A-4) JSON → 자연어, A-5) Instruction/Context/Constraint/Output Format,
                                            A-7) 파이프라인 분리 (JSON 파일 감시), C. STT·TTS
"""

import json
import time
from pathlib import Path

from llama_cpp import Llama

from voice import measure_noise_db, speech_to_text, text_to_speech

PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent
GEMMA_MODEL_PATH = REPO_ROOT / "src" / "models" / "Gemma4" / "google_gemma-4-E2B-it-Q4_K_M.gguf"
STATE_PATH = PROJECT_DIR / "state" / "state.json"

CONTEXT_WINDOW = 2048
MAX_TOKENS = 100
TEMPERATURE = 0.7

POLL_INTERVAL = 0.1        # state.json 확인 주기 (초)
EVENT_MAX_AGE = 10.0       # 이보다 오래된 이벤트는 이미 지난 상황이므로 응답하지 않음 (초)

NOISE_THRESHOLD_DB = -28.0  # 이 값보다 크면 시끄러움. 실측(마이크 입력 24): 조용함 -36~-31, 말소리 -27~-22
                            # 환경마다 다르므로 voice.py 단독 실행으로 현장에서 다시 맞출 것
NOISE_REPEAT = 2            # 연속으로 이만큼 시끄러워야 소음 이벤트
NOISE_CHECK_INTERVAL = 5.0  # 소음 측정 주기 (초)
NOISE_ALERT_COOLDOWN = 60.0  # 소음 경고 후 다음 경고까지 최소 간격 (초)
AFTER_SPEECH_DELAY = 3.0    # 스피커 소리를 소음으로 재지 않도록, 말한 직후 측정을 쉬는 시간 (초)

# COCO 영어 클래스 이름 → 한국어 (Gemma와 TTS가 자연스럽게 말하도록)
CLASS_NAMES_KO = {"person": "사람", "cell phone": "휴대폰", "laptop": "노트북", "book": "책"}

SYSTEM_PROMPT = """
Instruction:
너는 책상 위 카메라와 마이크로 학생을 지켜보는 공부 집중 도우미다.
주어진 상황에 맞춰 학생에게 직접 말하듯 대답하시오.

Constraint:
Context에 없는 물체나 사실은 추측하지 마시오.
다정하지만 짧게 말하시오. 이모지와 특수문자는 쓰지 마시오.

Output Format:
한국어 두 문장 이내.
"""

# 이벤트마다 Gemma에게 줄 요청 문장
EVENT_REQUESTS = {
    "phone": "학생이 공부 중에 휴대폰을 계속 보고 있다. 휴대폰을 내려놓고 다시 집중하도록 한마디 하시오.",
    "absent": (
        "학생이 자리를 비웠고 아직 돌아오지 않았다. '다시 왔다'고 말하지 마시오. "
        "자리로 돌아와 공부를 이어가자고 부르는 짧은 한마디를 하시오."
    ),
    "noise": "주변이 너무 시끄러워 집중하기 어렵다. 조용한 환경을 만들도록 권하는 한마디를 하시오.",
}

# Gemma 응답이 비었을 때 대신 말할 문장
FALLBACK_REPLIES = {
    "phone": "휴대폰은 잠시 내려놓고 다시 집중해 볼까요?",
    "absent": "자리를 비우셨네요. 돌아오면 이어서 공부해요.",
    "noise": "주변이 시끄러워요. 조용한 곳에서 집중해 보세요.",
    "question": "죄송해요, 다시 한 번 말씀해 주세요.",
}
MIC_ERROR_REPLY = "마이크를 사용할 수 없어요. 연결을 확인해 주세요."


def read_state():
    """state.json을 읽어 dict로 반환한다. 아직 없으면 None."""
    if not STATE_PATH.exists():
        return None
    with open(STATE_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def state_to_text(state):
    """06 A-4): 탐지 결과와 집중 통계를 Gemma가 읽기 쉬운 한국어 문장으로 바꾼다."""
    counts = {}
    for obj in state["objects"]:
        name = CLASS_NAMES_KO.get(obj["class"], obj["class"])
        counts[name] = counts.get(name, 0) + 1

    if counts:
        object_text = ", ".join(f"{name} {count}개" for name, count in counts.items())
        sentences = [f"현재 카메라에 보이는 물체: {object_text}."]
    else:
        sentences = ["현재 카메라에 보이는 물체가 없다."]

    # "물체가 없다"만으로는 학생이 없다는 뜻인지 Gemma가 헷갈리므로 사람 유무를 따로 적음
    if "person" in (obj["class"] for obj in state["objects"]):
        sentences.append("학생은 지금 자리에 앉아 있다.")
    else:
        sentences.append("학생은 지금 자리에 없다 (카메라에 사람이 보이지 않는다).")

    stats = state["stats"]
    sentences.append(
        f"공부 시작 후 {stats['elapsed_seconds'] // 60}분이 지났고, "
        f"그중 집중한 시간은 {stats['focus_seconds'] // 60}분 {stats['focus_seconds'] % 60}초다."
    )
    sentences.append(f"휴대폰 사용 {stats['phone_count']}회, 자리비움 {stats['absent_count']}회.")
    return "\n".join(sentences)


def ask_gemma(llm, request, context):
    """05 D, 06 A-5): system에 역할·규칙, user에 Context와 요청을 넣어 Gemma 응답 텍스트를 반환한다."""
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\n요청:\n{request}"},
        ],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )
    return response["choices"][0]["message"]["content"].strip()


def handle_event(llm, event_type, state, extra_context=""):
    """이벤트 하나를 처리해 Gemma 응답을 말한다."""
    context = state_to_text(state) + extra_context

    if event_type == "question":
        text_to_speech("네, 말씀하세요.")
        try:
            question = speech_to_text()
        except RuntimeError as error:
            print(f"[마이크 오류] {error}")
            text_to_speech(MIC_ERROR_REPLY)
            return
        print(f"[질문] {question or '(인식 실패)'}")
        if not question:
            reply = FALLBACK_REPLIES["question"]
        else:
            reply = ask_gemma(llm, f"학생의 질문에 대답하시오: {question}", context)
    else:
        reply = ask_gemma(llm, EVENT_REQUESTS[event_type], context)

    reply = reply or FALLBACK_REPLIES[event_type]
    print(f"[{event_type}] {reply}")
    text_to_speech(reply)


def main():
    if not GEMMA_MODEL_PATH.exists():
        raise FileNotFoundError(f"Gemma 모델이 없습니다: {GEMMA_MODEL_PATH}")

    llm = Llama(
        model_path=str(GEMMA_MODEL_PATH),
        n_gpu_layers=-1,
        n_ctx=CONTEXT_WINDOW,
        n_batch=32,
        n_ubatch=32,
        verbose=False,
    )

    # 시작 전에 이미 있던 이벤트는 처리하지 않도록 현재 번호를 기준으로 삼음
    state = read_state()
    handled_event_id = state["last_event"]["id"] if state and state["last_event"] else 0

    loud_count = 0
    last_noise_check = 0.0
    last_noise_alert = 0.0
    last_speech = 0.0

    print("Vision 이벤트 대기 중... (Ctrl+C 종료)")

    while True:
        state = read_state()
        now = time.time()

        # 1) vision.py 이벤트: 번호가 바뀌었을 때만 새 이벤트로 처리
        event = state["last_event"] if state else None
        if event and event["id"] != handled_event_id:
            handled_event_id = event["id"]
            if now - event["time"] <= EVENT_MAX_AGE:
                handle_event(llm, event["type"], state)
                last_speech = time.time()
            continue

        # 2) 소음: 쉬는 동안 NOISE_CHECK_INTERVAL마다 측정 (말한 직후는 제외)
        if state and now - last_noise_check >= NOISE_CHECK_INTERVAL and now - last_speech >= AFTER_SPEECH_DELAY:
            last_noise_check = now
            try:
                noise_db = measure_noise_db()
            except RuntimeError as error:
                # 마이크 문제로 Assistant 전체가 멈추지 않도록 이번 측정만 건너뜀 (phone/absent 응답은 계속)
                print(f"[마이크 오류] {error}")
                continue
            loud_count = loud_count + 1 if noise_db > NOISE_THRESHOLD_DB else 0
            print(f"[소음] {noise_db:.1f} dBFS ({loud_count}/{NOISE_REPEAT})")

            if loud_count >= NOISE_REPEAT and now - last_noise_alert >= NOISE_ALERT_COOLDOWN:
                loud_count = 0
                last_noise_alert = now
                handle_event(llm, "noise", state, f"\n측정한 주변 소음: {noise_db:.1f} dBFS.")
                last_speech = time.time()

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n종료합니다.")
