"""Day01 Section Project — 학생 명단·성적 분석

출처: 01_Linux-and-Python.ipynb 셀 [89] (Section Project)
"""

import numpy as np


def main():
    # 데이터 (셀 [87])
    raw_students = [
        "zg0505_짱구",
        "cs0822_철수",
        "mg0910_맹구",
        "yr0605_유리",
        "hi0205_훈이"
        ]

    raw_scores = np.array([
        [95, 82, 88],   # 짱구
        [70, 65, 90],   # 철수
        [98, 100, 92],  # 맹구
        [60, 75, 80],   # 유리
        [81, 88, 84],   # 훈이
    ])

    subjects = ["국어", "수학", "과학"]

    # TODO: list comprehension을 활용해 raw_students 리스트에서 이름만 추출한 리스트 생성
    student_names = [student.split("_")[1] for student in raw_students]
    print("1. 학생 명단: ", end="")
    for idx, name in enumerate(student_names):
        sep = ", " if idx < len(student_names) - 1 else "\n\n"
        print(name, end=sep)


    # TODO: 전산 시스템에 입력된 모든 성적들을 1차원 배열로 출력 (총 15개)
    all_scores = raw_scores.flatten()
    print("2. 전산 시스템에 입력된 모든 성적: ", end="")
    for idx, score in enumerate(all_scores):
        sep = ", " if idx < len(all_scores) - 1 else "\n\n"
        print(f"{score}점", end=sep)

    # TODO: 전체 12개 성적 중 85점 이상 개수
    over_85 = (all_scores >= 85).sum()
    print(f"3. 전체 15개 성적 중 85점 이상 개수: {over_85}개", end="\n\n")

    # TODO: 학생별 평균 점수 출력
    student_means = raw_scores.mean(axis=1)
    mean_texts = [f"{name} {mean:.2f}점" for name, mean in zip(student_names, student_means)]
    print("4. 학생별 평균 점수: " + ", ".join(mean_texts), end="\n\n")  # 예상 출력: "4. 학생별 평균 점수: 짱구 00점, 철수 00점, 맹구 00점, 유리 00점, 훈이 00점"

    # TODO: 평균 점수가 가장 높은 학생 선택
    top_student_idx = student_means.argmax()
    top_student_name = student_names[top_student_idx]
    print(f"5. 최고 평균 수석 학생: {top_student_name} ({student_means[top_student_idx]:.2f}점)", end="\n\n")  # 예상 출력: "5. 최고 평균 수석 학생: XX (xx.xx점)"

    # TODO: 전체 점수 기준, 최고 점수와 과목 그리고 해당 학생 계산 (modulo % 사용 추천)
    overall_max_score = all_scores.max()
    overall_max_idx = all_scores.argmax()
    overall_max_subject = subjects[overall_max_idx % len(subjects)]
    overall_max_student = student_names[overall_max_idx // len(subjects)]
    print(f"6. 단일 과목 최고 점수: {overall_max_score}점 (과목: {overall_max_subject}, 학생: {overall_max_student})")  # 예상 출력: "6. 단일 과목 최고 점수: xx점 (과목: XX, 학생: XX)"


if __name__ == "__main__":
    main()
