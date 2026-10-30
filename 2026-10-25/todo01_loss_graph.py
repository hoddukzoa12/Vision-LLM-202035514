"""Day06 과제 #3 — CNN 학습/평가 Loss 그래프

출처: 03_DL-and-GPU.ipynb (F. CNN 실습 – MNIST, 16) 모델 성능 확인)
    셀 [148]~[237]  MNIST CNN 학습 (mnist_cnn.train_model())
    셀 [243]        TODO: train_loss_history, test_loss_history 그래프 생성

결과 그림은 화면에 띄우지 않고 results/01_loss_graph.png로 저장한다.
"""

import matplotlib.pyplot as plt

from mnist_cnn import RESULTS_DIR, train_model

plt.switch_backend("Agg")  # 창을 띄우지 않고 파일로만 저장


def main():
    _, _, _, history = train_model()
    train_loss_history = history["train_loss"]
    test_loss_history = history["test_loss"]

    # x축: Epoch 번호 1, 2, 3 (리스트 index는 0부터라 1을 더함)
    epoch_numbers = range(1, len(train_loss_history) + 1)

    # 셀 [243]
    # TODO: plt.plot(x축_데이터, y축_데이터, marker="o", label="데이터_이름")을 사용하여 그래프 생성
    plt.figure(figsize=(8, 5))

    # TODO: train_loss_history 그래프 생성
    plt.plot(epoch_numbers, train_loss_history, marker="o", label="Train loss")
    # TODO: test_loss_history 그래프 생성
    plt.plot(epoch_numbers, test_loss_history, marker="o", label="Test loss")

    plt.xticks(epoch_numbers)  # 눈금을 정수 Epoch(1, 2, 3)에만 표시
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CNN Loss")
    plt.grid()
    plt.legend()

    RESULTS_DIR.mkdir(exist_ok=True)
    plt.savefig(RESULTS_DIR / "01_loss_graph.png", bbox_inches="tight")
    print("Saved:", RESULTS_DIR / "01_loss_graph.png")


if __name__ == "__main__":
    main()
