"""Day01 TODO 1 — NumPy 연산 함수로 배열 정규화

출처: 01_Linux-and-Python.ipynb 셀 [79] (B. 5) NumPy > NumPy 연산 함수)
"""

import numpy as np


def main():
    np_arr = np.array([1, 2, 3, 4, 5])  # 셀 [76]에서 사용한 배열

    # TODO: NumPy 연산 함수를 사용하여 배열을 정규화

    np_arr_normalized = np_arr / np_arr.max()  # max scaling
    print(np_arr_normalized)


if __name__ == "__main__":
    main()
