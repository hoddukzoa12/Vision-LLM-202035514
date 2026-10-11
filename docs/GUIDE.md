# GUIDE.md — 작업 지침 (사람용)

> **Physical AI의 Vision-LLM 융합 시청각 멀티모달 시스템** (강사: 김규래, 2026/10/10 – 11/8, 토·일 13:00–17:30)
> 학번 202035514 · 김진우
>
> 이 문서는 내가 이 레포에서 작업할 때 지키는 규칙이다. **폴더·커밋·코드 스타일 규칙의 원본**이며, 로컬 AI 에이전트 지침(`AGENTS.md`, `CLAUDE.md`)도 이 문서를 따른다.
> 원본 강의 규정: [강의계획서](https://github.com/kyu-rae-kim/GachonUniv-Vision-LLM/blob/main/README.md) (로컬 사본: `../GachonUniv-Vision-LLM/README.md`)

---

## 1. 평가 구조 — 왜 이 규칙이 필요한가

| 항목 | 비율 | 핵심 |
|---|---:|---|
| 출석 | 10% | 본인 직접 서명 |
| 태도 | 10% | **당일 실습 결과물**을 당일 18:00까지 제출 (회당 0 / 0.5 / 1점) |
| 지필고사 | 30% | 마지막 주, 서술형 10문항 (이론 이해도) |
| 프로젝트 및 과제 | 50% | 제출물 + 발표·질의응답 |

과제·프로젝트 점수는 **이해도 가중치 `w` (0 < w < 1)가 곱해진다.**
결과물이 완벽해도 내 코드를 설명하지 못하면 점수 전체가 비율로 깎인다. → **내가 설명할 수 없는 코드는 커밋하지 않는다.**

---

## 2. 제출 기한

| 대상 | 기한 | 기준 |
|---|---|---|
| 당일 실습 (`NN_*.ipynb`) | **수업 당일 18:00** | commit 시각 |
| 과제 | **다음 수업 시작 전 13:00** | commit 시각 |
| 프로젝트 (3회) | 별도 공지 | commit 시각 |

- 제출 시각은 **GitHub에 기록된 commit 시각**이다. commit만 하고 push를 안 하면 강사가 볼 수 없다 → **commit 후 반드시 push.**
- 기한 초과는 **1회만** 인정된다. 2회 이상이면 반영 비율이 큰 항목만 인정, 나머지는 0점.
- 늦을 사정이 있으면 **사전에 이메일**(kkr.kyurae.kim@gmail.com)로 통보.

### 수업 일정 (토·일 기준 예상 — 공지로 확인할 것)

| Day | 날짜 | Day | 날짜 |
|---|---|---|---|
| Day01 | 10/10 (토) | Day06 | 10/25 (일) |
| Day02 | 10/11 (일) | Day07 | 10/31 (토) |
| Day03 | 10/17 (토) | Day08 | 11/01 (일) |
| Day04 | 10/18 (일) | Day09 | 11/07 (토) |
| Day05 | 10/24 (토) | Day10 | 11/08 (일) |

---

## 3. 폴더 규칙

```
Vision-LLM-202035514/
├── README.md                  # 레포 소개 + 진행 현황 (새 Day·과제·프로젝트마다 갱신)
├── docs/GUIDE.md              # 이 문서 (사람용 지침, 폴더·커밋 규칙 원본)
├── AGENTS.md                  # AI 에이전트 공통 지침   ┐ 로컬 전용
├── CLAUDE.md                  # Claude Code 전용         ┘ (.gitignore, 커밋하지 않음)
├── ruff.toml                  # 이름 규칙 검사 설정 (§5)
├── .github/workflows/lint.yml # push마다 이름·커밋 메시지 규칙 자동 검사 (CI)
├── .githooks/commit-msg       # 커밋할 때 메시지 규칙 검사 (로컬 훅, §4)
├── scripts/check_commit_msg.py # 커밋 메시지 검사 스크립트 (훅·CI 공용)
├── .gitignore
│
├── 00_Initial-Setup.ipynb     # ┐
├── 01_Linux-and-Python.ipynb  # ├ 수업 실습 노트북 (강의 자료 파일명 그대로)
├── 02_....ipynb               # ┘
├── 맹구_개인정보.json          # 실습 노트북이 생성한 결과물 (노트북과 같은 위치)
│
├── src/                       # 강의 제공 자원
│   ├── images/                #   실습용 이미지 (수정·삭제 금지)
│   ├── models/{CIFAR10,MNIST,MediaPipe,YOLO}/
│   └── datasets/calibration/
│
├── 2026-10-10/                # 과제: "과제가 부여된 날짜" 폴더
│   ├── README.md              #   필수
│   ├── todo01_normalize.py    #   노트북 TODO를 분리한 독립 파일 (TODO 하나당 하나)
│   ├── todo02_min_max_scale.py
│   └── section_project.py     #   Section Project
│
└── Project01/                 # 프로젝트: Project01, Project02, Project03
    ├── README.md              #   필수
    ├── main.py                #   독립 실행 가능한 진입점
    └── ...                    #   부가 파일 (최종 프로젝트는 발표자료 .pdf 추가)
```

### 규칙

1. **실습 노트북**은 루트에 강의 자료와 **같은 파일명**(`NN_Topic.ipynb`)으로 둔다. **수업 중 실행한 상태 그대로 올리기만 한다.** 노트북 안의 `TODO` 셀은 채우지 않는다 (강사 안내).
2. **노트북의 `TODO`는 `.py` 파일로 분리해 과제 폴더에서 구현한다.**
   - TODO 하나당 파일 하나: `todoNN_<주제>.py` (예: `todo01_normalize.py`), Section Project는 `section_project.py`.
   - 파일 맨 위 docstring에 **원본 노트북과 셀 번호**를 적는다.
   - 앞 셀에서 정의된 데이터·`import`를 함께 옮겨 **파일 하나만으로 실행**되게 한다.
   - TODO 템플릿의 변수명과 출력 형식("예상 출력")은 그대로 유지한다.
3. **과제 폴더 이름은 과제가 부여된 날짜**, 형식은 `YYYY-MM-DD` (예: `2026-10-10/`).
   - 수업 실습 노트북을 복사해 넣지 않는다. **과제용 독립 파일**을 새로 만든다.
   - `README.md` 필수.
4. **프로젝트 폴더 이름은 `Project01`, `Project02`, `Project03`.**
   - 폴더 안에서 독립 실행 가능해야 한다 (루트의 다른 파일에 의존 금지, 필요하면 폴더 안으로 복사).
   - `README.md` 필수. 최종 프로젝트는 발표 자료 `.pdf`도 포함.
5. **`src/`는 강의 제공 자원 전용.** 기존 파일을 수정·삭제하지 않는다. 학습시킨 모델 가중치는 해당 `src/models/<이름>/`에 저장한다.
6. **커밋하지 않는 것:** 가상환경(`.venv/`), 캐시(`__pycache__/`, `.ipynb_checkpoints/`), OS 파일(`.DS_Store`), Jetson 전용 TensorRT 엔진(`*.engine`, 기기마다 다시 생성), **50MB를 넘는 파일**(GitHub 제한 100MB).
7. 파일·폴더 이름에 공백을 쓰지 않는다 (`_` 또는 `-` 사용).

### 과제 / 프로젝트 `README.md` 템플릿

````markdown
# <과제 또는 프로젝트 제목>

## 개요
무엇을 하는 프로그램인지 2~3문장.

## 실행 환경
- 기기: NVIDIA Jetson Orin (JetPack x.x) / Python 3.10
- 의존성: `numpy`, `opencv-python`, ...  (버전 명시)

## 실행 방법
```bash
source ../.venv/bin/activate
python main.py
```

## 구현 설명
- 설계 의도와 핵심 로직
- 수업 내용 중 어떤 개념을 적용했는지

## 참고 사항
알려진 한계, 입력 파일 위치, 실행 시간 등.
````

---

## 4. 커밋 규칙 (Conventional Commits)

강의 규정: **"update", "fix", "test"처럼 모호한 커밋 메시지는 감점될 수 있다.**
커밋 메시지는 업계 표준인 [Conventional Commits](https://www.conventionalcommits.org/) 형식을 따르고, **영어로 작성**한다.

### 형식

```
<type>(<scope>): <summary>

<optional body — why / how, only when needed>
```

### type — 변경의 종류

| type | 언제 | scope |
|---|---|---|
| `feat` | 새 기능·구현 (TODO·과제·프로젝트 구현) | **필수** |
| `fix` | 버그 수정 | **필수** |
| `refactor` | 동작은 그대로, 구조·가독성 개선 | **필수** |
| `docs` | README, GUIDE 등 문서 | 선택 |
| `chore` | 강의 자료 추가, `.gitignore`, 환경 설정 | 선택 |
| `ci` | GitHub Actions, 검사 스크립트·설정 | 선택 |

### scope — 어느 제출물인지

| scope | 대상 | 예 |
|---|---|---|
| `DayNN` | 수업 실습 노트북 업로드, 강의 자료 | `Day01`, `Day02` |
| `YYYY-MM-DD` | 과제 폴더 | `2026-10-10` |
| `ProjectNN` | 프로젝트 폴더 | `Project01` |

### summary — 무엇을 했는지

- **동사 원형으로 시작**하는 명령형: `add`, `implement`, `fix`, `handle`, `remove`, `rename`, ...
- 소문자로 시작, **끝에 마침표 없음**, 첫 줄 전체 **72자 이하**.
- 두 단어 이상으로 **무엇을** 했는지 드러나게 쓴다.

### 좋은 예 / 나쁜 예

| ✅ 좋은 예 | ❌ 나쁜 예 | 이유 |
|---|---|---|
| `feat(2026-10-10): implement NumPy normalization TODO` | `update` | 형식 없음, 모호 |
| `feat(2026-10-10): add grade analysis script and README` | `feat: add script` | scope 없음 |
| `feat(Project01): show YOLO detections on live feed` | `feat(Day1): add yolo` | scope는 `Day01` |
| `fix(Project01): handle missing camera device` | `fix: fix` | 모호 |
| `chore(Day02): add 02_OpenCV notebook and images` | `Day 02` | 형식 없음 |
| `docs: add course links to README` | `docs: README 수정` | 한글 |
| `ci: add commit message check` | `ci: add check.` | 마침표 |

### 원칙

1. **한 커밋에는 하나의 논리적 변경만** 담는다. 실습·과제·프로젝트를 한 커밋에 섞지 않는다.
2. 마감 직전에 몰아서 하지 말고, **작업 단위마다 커밋**한다. 커밋 기록 자체가 수행 과정의 증거가 된다.
3. 마감 전 마지막 확인: `git status` 깨끗한지 → `git log -1` 시각 확인 → `git push` → GitHub 웹에서 반영 확인.
4. 커밋 메시지에 `Co-Authored-By` 등 AI 표기 트레일러를 **절대 넣지 않는다.**
5. GitHub 웹에서 직접 파일을 수정하지 않는다 (자동 메시지 `Update README.md`가 규칙 위반).

### 자동 검사

같은 검사 스크립트(`scripts/check_commit_msg.py`)를 두 곳에서 실행한다.

| 시점 | 방법 | 역할 |
|---|---|---|
| **커밋할 때 (로컬)** | `.githooks/commit-msg` 훅 | 규칙을 어긴 커밋을 **애초에 만들지 못하게** 막음 |
| **push 후 (CI)** | `.github/workflows/lint.yml` | 이번 push에 포함된 커밋을 다시 검사, 위반 시 ❌ |

- **로컬 훅이 핵심이다.** push된 커밋 메시지는 force push 없이는 고칠 수 없다 (force push는 제출 기록을 바꾸므로 금지).
- 훅은 **clone한 기기마다 1회** 켜야 한다 (Mac, Jetson 각각):
  ```bash
  git config core.hooksPath .githooks
  ```
- 규칙을 정하기 전의 커밋(`Day 01`, `문서: ...` 등)은 검사하지 않는다. CI는 **새로 push된 커밋만** 본다.

---

## 5. 코드 스타일 (PEP 8)

과제 채점의 **"코드 구조 및 가독성"이 40%**(프로젝트 20%)다. Python 공식 스타일 가이드 PEP 8을 따른다.

### 이름 규칙

| 대상 | 표기법 | 예 |
|---|---|---|
| 변수, 함수, 메서드 | `snake_case` | `student_names`, `load_image()` |
| 모듈(`.py` 파일명) | `snake_case` | `grade_analysis.py`, `main.py` |
| 클래스 | `PascalCase` | `StudentRecord`, `YoloDetector` |
| 상수 | `UPPER_SNAKE_CASE` | `MAX_SCORE`, `IMAGE_DIR` |
| 내부용 (외부에서 쓰지 말 것) | `_` 접두사 | `_parse_line()` |

- 폴더 이름(`Project01`, `2026-10-10`)과 실습 노트북 이름(`01_Linux-and-Python.ipynb`)은 **강의 규정·자료를 따르므로 예외**다.
- `l`, `O`, `I` 같은 한 글자 이름은 쓰지 않는다 (숫자 1, 0과 헷갈림). 반복문의 `i`, `j`, `x`는 허용.
- 이름만 보고 내용을 알 수 있게 짓는다: `arr` ❌ → `scores` ✅, `tmp` ❌ → `normalized_scores` ✅

### 그 밖의 형식

- 들여쓰기는 **공백 4칸** (탭 금지).
- `import`는 파일 맨 위, 순서는 표준 라이브러리 → 서드파티(`numpy`, `cv2`) → 내 모듈.
- 함수에는 무엇을 받고 무엇을 돌려주는지 한 줄 docstring을 단다.
- 매직 넘버는 상수로 뺀다: `if score >= 85` → `PASS_SCORE = 85` / `if score >= PASS_SCORE`
- 과제·프로젝트 진입점은 `if __name__ == "__main__":` 블록으로 감싼다.

### 자동 검사 (Ruff)

이름 규칙은 **Ruff**(pep8-naming, 설정: `ruff.toml`)로 `.py`와 `.ipynb` 노트북 셀까지 검사한다.

- **CI에서만 검사한다.** push할 때마다 GitHub Actions(`.github/workflows/lint.yml`)가 검사하고, 위반이 있으면 GitHub 커밋 옆에 ❌가 표시된다.
- **Jetson에는 Ruff를 설치하지 않는다** (수업 안내 외의 패키지는 임의로 설치하지 않는다).
- ❌가 뜨면 GitHub Actions 로그에서 위반 줄을 확인하고, **고친 커밋을 하나 더 push**한다. 제출 시각에는 영향이 없다. 마감 직전에는 push 후 CI 결과까지 확인할 시간을 남겨 둔다.
- 예외 허용: `F` (`import torch.nn.functional as F`), `X`, `X_*` (입력 행렬). 예외를 추가하려면 `ruff.toml`의 `extend-ignore-names`에 넣는다.
- Ruff는 **표기 형식만** 검사한다. `arr`, `tmp`처럼 의미가 불분명한 이름은 직접 점검한다.

---

## 6. 생성형 AI 사용 원칙

AI 사용은 허용되지만 **과도한 의존은 감점**, 이해도는 `w`로 **곱해진다**. 강의 규정은 "설계는 내가, AI는 세부 작업의 보조 수단"이다.

1. **설계는 내가 먼저 한다.** 과제·프로젝트는 접근 방법을 내가 정한 뒤 AI에게 부분 구현이나 리뷰를 맡긴다.
2. **AI가 쓴 코드는 줄 단위로 설명을 받고, 이해한 뒤에만 커밋한다.**
3. AI가 제시한 "발표 때 받을 만한 질문"에 **스스로 답해본다.** 답 못 하는 부분이 있으면 그 코드는 아직 내 것이 아니다.
4. 막히면 AI보다 **수업 중·후 강사에게 질문**하는 것을 우선한다 (강의 권장 사항).
5. **Jetson에 인터넷 브라우저 설치 금지** (위반 시 최종 점수 불이익). Chromium은 삭제 상태를 유지한다.
6. **Jetson에는 수업 자료·강사 안내에 없는 패키지를 임의로 설치하지 않는다.** AI가 설치를 제안해도 따르지 말고 강사에게 먼저 확인한다.

---

## 7. 제출 전 체크리스트

### 당일 실습 (18:00 전)
- [ ] 수업 중 실행한 노트북을 **그대로** 루트에 올림 (TODO 셀은 비워 둠)
- [ ] `chore(DayNN): add NN_Topic practice notebook` 형식으로 commit & push
- [ ] GitHub Actions ✅ 확인 (❌면 고친 커밋 추가)
- [ ] README 진행 현황 표에 Day 행 추가

### 과제 (다음 수업 13:00 전)
- [ ] 노트북의 `TODO`를 `YYYY-MM-DD/` 폴더에 `.py`로 하나씩 분리
- [ ] 각 파일이 단독 실행되고, 예상 출력 형식과 결과가 맞는지 확인
- [ ] `README.md`: 실행 방법, 의존성, 참고 사항
- [ ] 새 터미널에서 README에 적힌 방법대로 실행해 보기
- [ ] 모든 줄을 내가 설명할 수 있는가?
- [ ] commit & push 후 GitHub에서 반영 및 Actions ✅ 확인

### 프로젝트
- [ ] `ProjectNN/` 폴더 안에서 독립 실행 가능
- [ ] `README.md` (+ 최종 프로젝트는 발표 `.pdf`)
- [ ] 시연 시나리오와 예상 질문 준비
- [ ] commit & push 후 GitHub에서 반영 및 Actions ✅ 확인
