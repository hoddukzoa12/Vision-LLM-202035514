"""Day06 과제 #3 — CIFAR-10 이미지 분류 GPU 가속 학습 및 모델 저장

출처: 03_DL-and-GPU.ipynb (J. CIFAR-10 이미지 분류 GPU 가속 학습)
    셀 [400]         난수 고정
    셀 [413]         데이터 전처리 (증강 + 정규화)
    셀 [416]         CIFAR-10 데이터셋 다운로드 및 불러오기
    셀 [425]         클래스 이름
    셀 [430]         DataLoader
    셀 [438], [440]  ConvBlock, CIFAR10CNN 모델
    셀 [445]         모델 생성 및 GPU 지정
    셀 [450]         손실 함수, Optimizer
    셀 [453]         GPU Warm-up
    셀 [456]         TODO: CUDA GPU에서 5 Epoch 학습·평가, Epoch별/전체 학습 시간 출력
                     train_one_epoch(), evaluate()는 MNIST와 같은 함수 (mnist_cnn.py)
    셀 [460]         학습된 모델로 테스트 이미지 40장 분류
    셀 [465], [467]  Loss / Accuracy 그래프
    셀 [471]         모델 저장 → 2026-10-25/models/CIFAR10_CNN.pth (노트북은 src/models/CIFAR10/)
    셀 [473]~[476]   저장한 모델 불러오기 및 확인

결과 그림은 화면에 띄우지 않고 results/07_*.png로 저장한다.
"""

import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from mnist_cnn import evaluate, train_one_epoch

plt.switch_backend("Agg")  # 창을 띄우지 않고 파일로만 저장

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "src" / "datasets"
MODEL_PATH = Path(__file__).resolve().parent / "models" / "CIFAR10_CNN.pth"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

SEED = 42
EPOCHS = 5
NUM_PREVIEW_IMAGES = 40

# 셀 [413] CIFAR-10 채널별 평균, 표준편차
CIFAR10_MEAN = (
    0.4914,
    0.4822,
    0.4465,
)

CIFAR10_STD = (
    0.2470,
    0.2435,
    0.2616,
)

# 셀 [425]
CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


# 셀 [438]
class ConvBlock(nn.Module):
    """
    Convolution을 이용해 특징을 추출하고 Max Pooling을 적용하는 재사용 가능한 CNN 블록.

    기본 구조:
    Conv2d → BatchNorm2d → ReLU
    → Conv2d → BatchNorm2d → ReLU
    → MaxPool2d
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        layers = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),
        ]

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# 셀 [440]
class CIFAR10CNN(nn.Module):
    """
    CIFAR-10의 32x32 RGB 이미지를 10개 클래스로 분류하는 CNN 모델.

    ConvBlock 1 → ConvBlock 2 → ConvBlock 3
    → Global Average Pooling → Dropout → Linear
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(in_channels=3, out_channels=64),     # 출력: [Batch, 64, 16, 16]
            ConvBlock(in_channels=64, out_channels=128),   # 출력: [Batch, 128, 8, 8]
            ConvBlock(in_channels=128, out_channels=256),  # 출력: [Batch, 256, 4, 4]
        )

        self.global_average_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))  # 출력: [Batch, 256, 1, 1]

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features=256, out_features=10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_average_pool(x)
        x = torch.flatten(x, start_dim=1)
        logits = self.classifier(x)

        return logits


def save_prediction_grid(model, test_dataset, device, start_index, save_path):
    """셀 [460], [476]: 테스트 이미지 40장을 분류한 결과를 그림으로 저장한다 (틀린 예측은 빨간 제목)."""
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(3, 1, 1)

    plt.figure(figsize=(15, 10))

    model.eval()

    with torch.no_grad():
        for i in range(NUM_PREVIEW_IMAGES):
            image, true_label = test_dataset[start_index + i]
            input_batch = image.unsqueeze(0).to(device)

            logits = model(input_batch)
            probabilities = torch.softmax(logits, dim=1)
            predicted_label = probabilities.argmax(dim=1).item()
            confidence = probabilities[0, predicted_label].item()

            # 정규화를 되돌려야 원래 색으로 보임
            image_denormalized = image * std + mean
            image_rgb = image_denormalized.permute(1, 2, 0).clamp(0, 1)

            title_color = "red" if predicted_label != true_label else "black"

            plt.subplot(4, 10, i + 1)
            plt.imshow(image_rgb)
            plt.title(
                f"True: {CIFAR10_CLASSES[true_label]}\n"
                f"Pred: {CIFAR10_CLASSES[predicted_label]}\n"
                f"Conf: {confidence * 100:.1f}%",
                color=title_color,
                fontsize=8,
            )
            plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print("Saved:", save_path)


def main():
    # 셀 [400]
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # 셀 [413] 증강은 학습 데이터에만, 정규화는 학습/테스트 모두에 적용
    train_transform = transforms.Compose([
        transforms.RandomCrop(size=32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])

    # 셀 [416]
    train_dataset = datasets.CIFAR10(
        root=DATASET_DIR,
        train=True,
        transform=train_transform,
        download=True,
    )

    test_dataset = datasets.CIFAR10(
        root=DATASET_DIR,
        train=False,
        transform=test_transform,
        download=True,
    )

    print("Train data:", len(train_dataset))
    print("Test data:", len(test_dataset))

    # 셀 [430]
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # 셀 [445]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU를 사용할 수 없습니다.")

    device = torch.device("cuda")
    model = CIFAR10CNN().to(device)

    print("Device:", device)
    print("Total parameters:", sum(parameter.numel() for parameter in model.parameters()))

    # 셀 [450]
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
    )

    # 셀 [453] GPU Warm-up: CUDA Context 생성, cuDNN 초기화 시간을 학습 시간 측정에서 제외
    warmup_input = torch.zeros(64, 3, 32, 32, device=device)

    model.eval()

    with torch.no_grad():
        for _ in range(3):
            _ = model(warmup_input)

    torch.cuda.synchronize()

    # 셀 [456]
    # TODO: CUDA GPU 장치에서 CIFAR-10 데이터셋을 학습과 평가
    # TODO: 각 Epoch 및 전체 학습 시간을 출력

    # 학습 기록 저장 공간 준비
    train_loss_history = []
    train_accuracy_history = []
    test_loss_history = []
    test_accuracy_history = []

    # 전체 학습 시간 측정 시작
    torch.cuda.synchronize()
    training_start_time = time.perf_counter()

    # TODO: 5번의 Epoch 동안 학습 및 평가 진행
    for epoch in range(EPOCHS):
        torch.cuda.synchronize()
        epoch_start_time = time.perf_counter()

        # 학습: train_one_epoch() 안에서 model.train()으로 바꾸므로 Dropout, BatchNorm이 학습 모드로 동작
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        # 평가: evaluate() 안에서 model.eval()과 torch.no_grad() 적용
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

    # TODO: 전체 학습 시간 계산 및 출력
    torch.cuda.synchronize()
    total_training_time = time.perf_counter() - training_start_time
    print(f"Total GPU training time: {total_training_time:.2f}s")

    RESULTS_DIR.mkdir(exist_ok=True)

    # 셀 [460]
    save_prediction_grid(model, test_dataset, device, 0, RESULTS_DIR / "07_cifar10_predictions.png")

    # 셀 [465], [467]
    epoch_axis = range(1, EPOCHS + 1)

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epoch_axis, train_loss_history, marker="o", label="Train loss")
    plt.plot(epoch_axis, test_loss_history, marker="o", label="Test loss")
    plt.xticks(epoch_axis)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CNN Loss")
    plt.grid()
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epoch_axis, np.array(train_accuracy_history) * 100, marker="o", label="Train accuracy")
    plt.plot(epoch_axis, np.array(test_accuracy_history) * 100, marker="o", label="Test accuracy")
    plt.xticks(epoch_axis)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("CNN Accuracy")
    plt.grid()
    plt.legend()

    plt.savefig(RESULTS_DIR / "07_cifar10_training_curves.png", bbox_inches="tight")
    plt.close()
    print("Saved:", RESULTS_DIR / "07_cifar10_training_curves.png")

    # 셀 [471] 모델 저장
    save_data = {
        "model_name": "CIFAR10CNN",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "test_accuracy": test_accuracy_history[-1],
        "epochs": EPOCHS,
        "class_names": CIFAR10_CLASSES,
        "mean": CIFAR10_MEAN,
        "std": CIFAR10_STD,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(save_data, MODEL_PATH)
    print("Saved:", MODEL_PATH)

    # 셀 [473] 저장한 모델 불러오기
    loaded_model = CIFAR10CNN()

    state_dict = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    loaded_model.load_state_dict(state_dict["model_state_dict"])
    loaded_model.to(device)
    loaded_model.eval()

    # 셀 [475]
    print("모델 이름:", state_dict["model_name"])
    print("Epoch 체크포인트:", state_dict["epochs"])
    print(f"마지막 테스트 정확도: {state_dict['test_accuracy'] * 100:.2f}%")

    # 불러온 모델이 학습 직후 모델과 같은 정확도를 내는지 확인
    _, loaded_accuracy = evaluate(loaded_model, test_loader, criterion, device)
    print(f"불러온 모델의 테스트 정확도: {loaded_accuracy * 100:.2f}%")

    # 셀 [476] 불러온 모델로 다음 40장 분류
    save_prediction_grid(loaded_model, test_dataset, device, NUM_PREVIEW_IMAGES,
                         RESULTS_DIR / "07_cifar10_loaded_model_predictions.png")


if __name__ == "__main__":
    main()
