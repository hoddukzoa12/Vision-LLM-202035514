# Day01 과제 — Python 기초 & NumPy

## 개요
`01_Linux-and-Python.ipynb`의 미완성 TODO 3개를 독립 실행 가능한 Python 파일로 옮겨 구현한 과제입니다.

| 파일 | 내용 | 원본 셀 |
|---|---|---|
| `todo01_normalize.py` | NumPy 연산 함수로 배열 정규화 | [79] |
| `todo02_min_max_scale.py` | 배열을 0.0 ~ 1.0 범위의 `float32` 배열로 변환 | [80] |
| `section_project.py` | 학생 명단·성적 분석 (Section Project) | [87], [89] |

## 실행 환경
- 기기: NVIDIA Jetson Orin / Python 3.10
- 의존성: `numpy` (Jetson 시스템 패키지, 버전: `<python3 -c "import numpy; print(numpy.__version__)" 결과>`)

## 실행 방법
```bash
cd 2026-10-10
source ../.venv/bin/activate
python todo01_normalize.py
python todo02_min_max_scale.py
python section_project.py
```

## 구현 설명
- `todo01_normalize.py`: **최댓값 정규화** `x / max`. `.max()`로 나눠 모든 값을 0~1 사이로 변환 (최댓값 = 1.0).
- `todo02_min_max_scale.py`: **Min-Max 정규화** `(x - min) / (max - min)`. 최솟값 0.0, 최댓값 1.0으로 맞춘 뒤 `.astype(np.float32)`로 변환.
- `section_project.py`:
  1. list comprehension과 `split("_")`으로 `"studentID_name"`에서 이름만 추출
  2. `flatten()`으로 5×3 성적 배열을 1차원(15개)으로 변환
  3. Boolean 배열 `(all_scores >= 85)`의 `sum()`으로 85점 이상 개수 계산
  4. `mean(axis=1)`로 학생별(행) 평균
  5. `argmax()`로 평균이 가장 높은 학생의 인덱스
  6. 1차원 배열의 `argmax()` 인덱스를 과목 수(3)로 나눠 `//` → 학생, `%` → 과목

## 실행 결과
```
[0.2 0.4 0.6 0.8 1. ]      # todo01

[0.   0.25 0.5  0.75 1.  ]                                         # todo02
float32

1. 학생 명단: 짱구, 철수, 맹구, 유리, 훈이                          # section_project
2. 전산 시스템에 입력된 모든 성적: 95점, 82점, 88점, 70점, 65점, 90점, 98점, 100점, 92점, 60점, 75점, 80점, 81점, 88점, 84점
3. 전체 15개 성적 중 85점 이상 개수: 7개
4. 학생별 평균 점수: 짱구 88.33점, 철수 75.00점, 맹구 96.67점, 유리 71.67점, 훈이 84.33점
5. 최고 평균 수석 학생: 맹구 (96.67점)
6. 단일 과목 최고 점수: 100점 (과목: 수학, 학생: 맹구)
```

## 참고 사항
- 원본 셀 [79]은 정규화 방식이 명시되어 있지 않습니다. 셀 [80]이 최솟값 0.0·최댓값 1.0을 맞추는 Min-Max 정규화이므로, 셀 [79]은 그보다 단순한 **최댓값으로 나누는 정규화**로 해석했습니다.
- 4번 평균은 5번 예상 출력(`xx.xx점`)과 맞춰 소수점 둘째 자리까지 표시했습니다.
- 원본 노트북 셀 [89]의 주석 "전체 12개 성적 중"은 실제 데이터(5명 × 3과목 = 15개)와 다릅니다. 출력 문구는 15개 기준입니다.
