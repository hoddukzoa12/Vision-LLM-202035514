"""Day06 과제 #3 — CNN 학습/평가 Accuracy 그래프 (백분율)

출처: 03_DL-and-GPU.ipynb (F. CNN 실습 – MNIST, 16) 모델 성능 확인)
    셀 [148]~[237]  MNIST CNN 학습 (mnist_cnn.train_model())
    셀 [245]        TODO: train_accuracy_history, test_accuracy_history 그래프 생성 (백분율)

결과 그림은 화면에 띄우지 않고 results/02_accuracy_graph.png로 저장한다.
"""

import matplotlib.pyplot as plt

from mnist_cnn import RESULTS_DIR, train_model

plt.switch_backend("Agg")  # 창을 띄우지 않고 파일로만 저장


def main():
    _, _, _, history = train_model()
    train_accuracy_history = history["train_accuracy"]
    test_accuracy_history = history["test_accuracy"]

    # x축: Epoch 번호 1, 2, 3
    epoch_numbers = range(1, len(train_accuracy_history) + 1)

    # 기록은 0~1 비율이므로 100을 곱해 백분율로 변환
    train_accuracy_percent = [accuracy * 100 for accuracy in train_accuracy_history]
    test_accuracy_percent = [accuracy * 100 for accuracy in test_accuracy_history]

    # 셀 [245]
    # TODO: plt.plot(x축_데이터, y축_데이터, marker="o", label="데이터_이름")을 사용하여 그래프 생성
    plt.figure(figsize=(8, 5))

    # TODO: train_accuracy_history 그래프 생성 (백분율)
    plt.plot(epoch_numbers, train_accuracy_percent, marker="o", label="Train accuracy")
    # TODO: test_accuracy_history 그래프 생성 (백분율)
    plt.plot(epoch_numbers, test_accuracy_percent, marker="o", label="Test accuracy")

    plt.xticks(epoch_numbers)  # 눈금을 정수 Epoch(1, 2, 3)에만 표시
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("CNN Accuracy")
    plt.grid()
    plt.legend()

    RESULTS_DIR.mkdir(exist_ok=True)
    plt.savefig(RESULTS_DIR / "02_accuracy_graph.png", bbox_inches="tight")
    print("Saved:", RESULTS_DIR / "02_accuracy_graph.png")


if __name__ == "__main__":
    main()
