"""Day01 TODO 2 — 배열을 0.0 ~ 1.0 범위의 float32 배열로 변환

출처: 01_Linux-and-Python.ipynb 셀 [80] (B. 5) NumPy > NumPy 연산 함수)
"""

import numpy as np


def main():
    np_arr = np.array([1, 2, 3, 4, 5])  # 셀 [76]에서 사용한 배열

    # TODO: NumPy 연산 함수를 사용하여 배열을 0.0 ~ 1.0 범위의 float32 배열로 변환
    # TODO: 최종 배열의 최솟값 = 0.0, 최종 배열의 최댓값 = 1.0

    # Min-Max 정규화: (x - 최솟값) / (최댓값 - 최솟값), 결과를 float32로 변환
    min_value = np_arr.min()
    max_value = np_arr.max()
    np_arr_final = ((np_arr - min_value) / (max_value - min_value)).astype(np.float32)
    print(np_arr_final)
    print(np_arr_final.dtype)


if __name__ == "__main__":
    main()
