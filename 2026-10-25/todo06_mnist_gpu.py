"""Day06 과제 #3 — MNIST 데이터셋 CPU vs. GPU 학습 시간 비교

출처: 03_DL-and-GPU.ipynb (I. MNIST 데이터셋 GPU 가속 학습)
    셀 [386]  CPU 기반 MNIST 학습 (상수, main_cpu())
              SimpleCNN, train_one_epoch(), evaluate()는 같은 코드인 mnist_cnn.py에서 가져온다
    셀 [389]  TODO: GPU 기반 MNIST 학습 코드 작성 (main_gpu())
    셀 [391]  CPU / GPU 학습 시간 비교

main_gpu()는 main_cpu()를 복사한 뒤 다음만 바꿨다.
    1) torch.cuda.manual_seed()로 GPU 난수도 고정
    2) device를 "cuda"로 지정
    3) 시간 측정 전후에 torch.cuda.synchronize()로 GPU 연산이 끝날 때까지 대기
    4) 출력 문구를 "GPU"로 변경
"""

import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from mnist_cnn import DATASET_DIR, SimpleCNN, evaluate, train_one_epoch

# 셀 [386]
SEED = 42
DATA_ROOT = DATASET_DIR

BATCH_SIZE = 64
TEST_BATCH_SIZE = 1000
EPOCHS = 3
LEARNING_RATE = 0.001
NUM_WORKERS = 0


# 셀 [386]
def main_cpu():
    """CPU에서 MNIST를 학습하고 전체 학습 시간(초)을 반환한다."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root=DATA_ROOT,
        train=True,
        transform=transform,
        download=True,
    )

    test_dataset = datasets.MNIST(
        root=DATA_ROOT,
        train=False,
        transform=transform,
        download=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    device = torch.device("cpu")

    model = SimpleCNN()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    train_loss_history = []
    train_accuracy_history = []
    test_loss_history = []
    test_accuracy_history = []

    training_start_time = time.perf_counter()

    for epoch in range(EPOCHS):
        epoch_start_time = time.perf_counter()

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        test_loss, test_accuracy = evaluate(
            model,
            test_loader,
            criterion,
            device,
        )

        train_loss_history.append(train_loss)
        train_accuracy_history.append(train_accuracy)
        test_loss_history.append(test_loss)
        test_accuracy_history.append(test_accuracy)

        epoch_time = time.perf_counter() - epoch_start_time

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train accuracy: "
            f"{train_accuracy * 100:.2f}% | "
            f"Test loss: {test_loss:.4f} | "
            f"Test accuracy: "
            f"{test_accuracy * 100:.2f}% | "
            f"Time: {epoch_time:.2f}s"
        )

    total_training_time = time.perf_counter() - training_start_time
    print(f"Total CPU training time: {total_training_time:.2f}s")

    return total_training_time


# 셀 [389]
# TODO: GPU 기반 MNIST 데이터셋 학습 코드 작성
def main_gpu():
    """CUDA GPU에서 MNIST를 학습하고 전체 학습 시간(초)을 반환한다."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)  # GPU에서 쓰는 난수도 고정

    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root=DATA_ROOT,
        train=True,
        transform=transform,
        download=True,
    )

    test_dataset = datasets.MNIST(
        root=DATA_ROOT,
        train=False,
        transform=transform,
        download=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU를 사용할 수 없습니다.")

    # 모델과 Batch를 GPU로 보냄 (train_one_epoch, evaluate 안의 .to(device)도 GPU로 이동)
    device = torch.device("cuda")

    model = SimpleCNN()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    train_loss_history = []
    train_accuracy_history = []
    test_loss_history = []
    test_accuracy_history = []

    # GPU는 CPU와 비동기로 동작하므로, 남은 GPU 작업을 끝낸 뒤 측정을 시작
    torch.cuda.synchronize()
    training_start_time = time.perf_counter()

    for epoch in range(EPOCHS):
        torch.cuda.synchronize()
        epoch_start_time = time.perf_counter()

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        test_loss, test_accuracy = evaluate(
            model,
            test_loader,
            criterion,
            device,
        )

        train_loss_history.append(train_loss)
        train_accuracy_history.append(train_accuracy)
        test_loss_history.append(test_loss)
        test_accuracy_history.append(test_accuracy)

        # 이번 Epoch의 GPU 연산이 모두 끝난 뒤 시간 측정
        torch.cuda.synchronize()
        epoch_time = time.perf_counter() - epoch_start_time

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train accuracy: "
            f"{train_accuracy * 100:.2f}% | "
            f"Test loss: {test_loss:.4f} | "
            f"Test accuracy: "
            f"{test_accuracy * 100:.2f}% | "
            f"Time: {epoch_time:.2f}s"
        )

    torch.cuda.synchronize()
    total_training_time = time.perf_counter() - training_start_time
    print(f"Total GPU training time: {total_training_time:.2f}s")

    return total_training_time


# 셀 [391]
if __name__ == "__main__":
    print("=== CPU Training ===")
    cpu_training_time = main_cpu()

    print("\n=== GPU Training ===")
    gpu_training_time = main_gpu()

    speedup = cpu_training_time / gpu_training_time
    time_difference = cpu_training_time - gpu_training_time

    print("\n=== CPU vs. GPU Comparison ===")
    print(f"CPU training time: {cpu_training_time:.2f}s")
    print(f"GPU training time: {gpu_training_time:.2f}s")
    print(f"Time difference: {time_difference:.2f}s")
    print(f"GPU speedup: {speedup:.2f}x")
