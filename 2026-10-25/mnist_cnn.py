"""F. CNN 실습 – MNIST 공통 코드 (todo01~03이 함께 사용)

출처: 03_DL-and-GPU.ipynb (F. CNN 실습 – MNIST 데이터셋)
    셀 [145], [148]  import, seed 고정
    셀 [151], [167]  MNIST 데이터셋, DataLoader
    셀 [174]         SimpleCNN 모델
    셀 [183]         모델 생성, 연산 장치 지정
    셀 [194]         show_feature_maps() (화면 대신 파일로 저장하도록 save_path 추가)
    셀 [224], [227]  손실 함수, optimizer
    셀 [231], [234]  train_one_epoch(), evaluate()
    셀 [237]         Epoch 반복 학습 (함수 train_model()로 묶음)

노트북은 device를 CPU로 고정했지만, 여기서는 GPU가 있으면 GPU를 쓴다.
"""

import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "src" / "datasets"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

SEED = 42
EPOCHS = 3


# 셀 [174]
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x, return_features=False):
        conv1_output = self.conv1(x)
        relu1_output = F.relu(conv1_output)
        pool1_output = self.pool(relu1_output)
        conv2_output = self.conv2(pool1_output)
        relu2_output = F.relu(conv2_output)
        pool2_output = self.pool(relu2_output)
        flattened = torch.flatten(pool2_output, start_dim=1)
        fc1_output = F.relu(self.fc1(flattened))
        logits = self.fc2(fc1_output)

        if return_features:
            features = {
                "conv1": conv1_output,
                "relu1": relu1_output,
                "pool1": pool1_output,
                "conv2": conv2_output,
                "relu2": relu2_output,
                "pool2": pool2_output,
            }
            return logits, features

        return logits


# 셀 [194] (+ save_path: 창을 띄우지 않고 PNG로 저장)
def show_feature_maps(feature_tensor, title, max_maps=8, cmap="gray", save_path=None):
    if feature_tensor.ndim != 4:
        raise ValueError("Feature Tensor는 [Batch, Channel, Height, Width] 형태여야 합니다.")

    feature_maps = feature_tensor[0].detach().cpu()

    channel_count = min(feature_maps.shape[0], max_maps)
    column_count = 4
    row_count = int(np.ceil(channel_count / column_count))

    plt.figure(figsize=(12, 3 * row_count))

    for channel_index in range(channel_count):
        plt.subplot(row_count, column_count, channel_index + 1)
        plt.imshow(feature_maps[channel_index], cmap=cmap)
        plt.title(f"Channel {channel_index}")
        plt.axis("off")
        plt.suptitle(title)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()


# 셀 [148]
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# 셀 [151]
def load_datasets():
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root=DATASET_DIR, train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root=DATASET_DIR, train=False, transform=transform, download=True)
    return train_dataset, test_dataset


# 셀 [231]
def train_one_epoch(model, data_loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    correct_count = 0
    sample_count = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        predictions = logits.argmax(dim=1)
        correct_count += (predictions == labels).sum().item()
        sample_count += batch_size

    average_loss = total_loss / sample_count
    accuracy = correct_count / sample_count

    return average_loss, accuracy


# 셀 [234]
def evaluate(model, data_loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct_count = 0
    sample_count = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            predictions = logits.argmax(dim=1)
            correct_count += (predictions == labels).sum().item()
            sample_count += batch_size

        average_loss = total_loss / sample_count
        accuracy = correct_count / sample_count

        return average_loss, accuracy


def train_model():
    """셀 [148]~[237]을 순서대로 실행해 학습된 모델과 Epoch별 기록을 반환한다.

    반환: (model, test_dataset, device, history)
        history: train_loss / train_accuracy / test_loss / test_accuracy 리스트를 담은 dict
    """
    set_seed(SEED)
    train_dataset, test_dataset = load_datasets()

    # 셀 [167]
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=0)

    # 셀 [183] (노트북은 CPU 고정, 여기서는 GPU가 있으면 GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    model = SimpleCNN().to(device)
    print("Device:", device)

    # 셀 [224], [227]
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 셀 [237]
    history = {"train_loss": [], "train_accuracy": [], "test_loss": [], "test_accuracy": []}
    training_start_time = time.perf_counter()

    for epoch in range(EPOCHS):
        epoch_start_time = time.perf_counter()

        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["test_loss"].append(test_loss)
        history["test_accuracy"].append(test_accuracy)

        epoch_time = time.perf_counter() - epoch_start_time
        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train accuracy: {train_accuracy * 100:.2f}% | "
            f"Test loss: {test_loss:.4f} | "
            f"Test accuracy: {test_accuracy * 100:.2f}% | "
            f"Time: {epoch_time:.2f}s"
        )

    total_training_time = time.perf_counter() - training_start_time
    print(f"Total training time ({device}): {total_training_time:.2f}s")

    return model, test_dataset, device, history
