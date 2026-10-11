# 202035514 김진우

[![lint](https://github.com/hoddukzoa12/Vision-LLM-202035514/actions/workflows/lint.yml/badge.svg)](https://github.com/hoddukzoa12/Vision-LLM-202035514/actions/workflows/lint.yml)

가천대학교 **「Physical AI의 Vision-LLM 융합 시청각 멀티모달 시스템」** 수업의 실습·과제·프로젝트 제출 레포입니다.

| | |
|---|---|
| 강사 | 김규래 |
| 기간 | 2026/10/10 – 2026/11/08 (토·일 13:00 – 17:30) |
| 실행 환경 | NVIDIA Jetson Orin · Python 3.10 |
| 강의 자료 | [kyu-rae-kim/GachonUniv-Vision-LLM](https://github.com/kyu-rae-kim/GachonUniv-Vision-LLM) · [강의계획서](https://github.com/kyu-rae-kim/GachonUniv-Vision-LLM/blob/main/README.md) |

---

## 진행 현황

| Day | 날짜 | 실습 노트북 | 과제 |
|---|---|---|---|
| Day01 | 10/10 (토) | [`00_Initial-Setup`](00_Initial-Setup.ipynb), [`01_Linux-and-Python`](01_Linux-and-Python.ipynb) | [`2026-10-10/`](2026-10-10/) |

| 프로젝트 | 주제 | 상태 |
|---|---|---|
| Project01 | – | – |
| Project02 | – | – |
| Project03 (최종) | – | – |

---

## 레포 구조

```
├── NN_Topic.ipynb     # 수업 실습 노트북
├── YYYY-MM-DD/        # 과제 (과제가 부여된 날짜)
├── ProjectNN/         # 프로젝트
└── src/               # 강의 제공 자원 (이미지, 모델, 데이터셋)
```

폴더·커밋·코드 스타일 규칙은 [`GUIDE.md`](GUIDE.md)에 정리했습니다.

## 실행 환경 설정 (Jetson)

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install jupyter ipykernel
python -m ipykernel install --user --name=vision_llm --display-name "Vision-LLM (Python3.10)"
```

VS Code에서 커널 `Vision-LLM (Python3.10)`을 선택해 노트북을 실행합니다. 과제·프로젝트별 실행 방법과 의존성은 각 폴더의 `README.md`를 참고하세요.

## 자동 검사

push마다 GitHub Actions에서 두 가지 규칙을 검사합니다.

| 검사 | 규칙 | 로컬 |
|---|---|---|
| 이름 규칙 | PEP 8 `snake_case` / `PascalCase` ([Ruff](https://docs.astral.sh/ruff/)) | CI에서만 검사 (Jetson에 설치하지 않음) |
| 커밋 메시지 | [Conventional Commits](https://www.conventionalcommits.org/) (영어) | 커밋할 때 훅이 자동 검사 |

커밋 메시지 훅은 별도 설치 없이 `python3`만 사용합니다. clone한 기기마다 1회 설정:

```bash
git config core.hooksPath .githooks
```

## 문서

| 파일 | 내용 |
|---|---|
| [`GUIDE.md`](GUIDE.md) | 작업 지침: 제출 기한, 폴더·커밋·코드 스타일 규칙, 제출 전 체크리스트 |

---

<sub>실습 노트북의 강의 내용과 `src/`의 자원은 강사(김규래)에게 저작권이 있으며, 수업 제출 목적으로만 포함되어 있습니다.</sub>
