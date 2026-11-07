"""Project03 — 집중 이벤트 판정.

매 프레임 YOLO가 찾은 클래스 이름 목록을 받아 두 가지 이벤트를 판정한다.
    phone   휴대폰(cell phone)이 PHONE_SECONDS 이상 계속 보임 → 딴짓
    absent  사람(person)이 ABSENT_SECONDS 이상 안 보임      → 자리비움

YOLO는 같은 물체도 몇 프레임씩 놓칠 수 있으므로, MISS_GRACE_SECONDS보다 짧게 끊긴 것은
계속 보인 것으로 친다. 프레임 수가 아닌 시간(초)으로 판정해 FPS가 변해도 기준이 같다.
"""

PHONE_SECONDS = 3.0        # 폰이 이만큼 계속 보이면 딴짓 이벤트
ABSENT_SECONDS = 10.0      # 사람이 이만큼 안 보이면 자리비움 이벤트
MISS_GRACE_SECONDS = 0.5   # 이보다 짧게 안 보인 것은 YOLO가 놓친 것으로 보고 무시


class FocusTracker:
    """폰 사용·자리비움 이벤트와 집중 통계를 관리한다."""

    def __init__(self, now):
        self.phone_since = None       # 폰이 연속으로 보이기 시작한 시각 (안 보이면 None)
        self.phone_last_seen = None   # 폰을 마지막으로 본 시각
        self.phone_fired = False      # 이번 연속 구간에서 이미 이벤트를 냈는지
        self.person_last_seen = now   # 사람을 마지막으로 본 시각 (시작할 때는 앉아 있다고 가정)
        self.absent_fired = False

        self.start_time = now
        self.focus_seconds = 0.0      # 사람이 있고 폰이 없던 시간의 합
        self.phone_count = 0
        self.absent_count = 0
        self.last_update = now

    def update(self, class_names, now):
        """이번 프레임의 클래스 이름 목록으로 상태를 갱신하고, 새로 생긴 이벤트 이름 목록을 반환한다."""
        events = []
        phone_seen = "cell phone" in class_names
        person_seen = "person" in class_names

        # 1) 폰: 보이면 연속 구간 시작/유지, 잠깐(MISS_GRACE_SECONDS) 넘게 안 보이면 구간 종료
        if phone_seen:
            if self.phone_since is None:
                self.phone_since = now
            self.phone_last_seen = now
        elif self.phone_since is not None and now - self.phone_last_seen > MISS_GRACE_SECONDS:
            self.phone_since = None
            self.phone_fired = False

        if self.phone_since is not None and not self.phone_fired:
            if now - self.phone_since >= PHONE_SECONDS:
                self.phone_fired = True  # 폰을 내려놓을 때까지 같은 경고를 반복하지 않음
                self.phone_count += 1
                events.append("phone")

        # 2) 자리비움: 사람이 다시 보이면 초기화
        if person_seen:
            self.person_last_seen = now
            self.absent_fired = False
        elif not self.absent_fired and now - self.person_last_seen >= ABSENT_SECONDS:
            self.absent_fired = True
            self.absent_count += 1
            events.append("absent")

        # 3) 집중 시간: 직전 갱신 이후 시간을 "사람 있음 + 폰 구간 아님"일 때만 더함
        if person_seen and self.phone_since is None:
            self.focus_seconds += now - self.last_update
        self.last_update = now

        return events

    def summary(self, now):
        """화면 표시와 Gemma 전달에 쓰는 통계 dict를 반환한다."""
        return {
            "elapsed_seconds": round(now - self.start_time),
            "focus_seconds": round(self.focus_seconds),
            "phone_count": self.phone_count,
            "absent_count": self.absent_count,
            "phone_now": self.phone_since is not None,
        }
