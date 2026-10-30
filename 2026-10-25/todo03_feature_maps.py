"""Day06 과제 #3 — 학습된 CNN의 층별 Feature Map 시각화

출처: 03_DL-and-GPU.ipynb (F. CNN 실습 – MNIST, 17) 학습된 모델의 Feature Map 확인)
    셀 [148]~[237]  MNIST CNN 학습 (mnist_cnn.train_model())
    셀 [194]        show_feature_maps() (mnist_cnn.py)
    셀 [196]~[198]  이미지 선택, return_features=True 순전파
    셀 [248]        TODO: Conv1 8개 / Conv2 16개 / 6개 Layer별 대표 Feature Map 4개씩 비교

결과 그림은 화면에 띄우지 않고 results/03_*.png로 저장한다.
"""

import matplotlib.pyplot as plt
import torch

from mnist_cnn import RESULTS_DIR, show_feature_maps, train_model

plt.switch_backend("Agg")  # 창을 띄우지 않고 파일로만 저장

MAPS_PER_LAYER = 4  # 6개 Layer 비교에서 층마다 보여줄 Feature Map 개수


def main():
    model, test_dataset, device, _ = train_model()
    RESULTS_DIR.mkdir(exist_ok=True)

    # 셀 [248]
    # TODO: CNN 각 층의 Feature Map 시각화
    # TODO: cmap="viridis" 사용

    # TODO: 이미지 선택
    image, true_label = test_dataset[0]
    input_batch = image.unsqueeze(0).to(device)  # [1, 28, 28] → [1, 1, 28, 28] (Batch 차원 추가)

    # TODO: 역전파(파라미터 수정) 없이 순전파 수행
    model.eval()
    with torch.no_grad():
        logits, features = model(input_batch, return_features=True)
    predicted_label = logits.argmax(dim=1).item()
    print(f"True label: {true_label}, Predicted label: {predicted_label}")

    # TODO: Conv1 Feature Maps 8개 시각화
    show_feature_maps(
        features["conv1"],
        title="Conv1 Feature Maps",
        max_maps=8,
        cmap="viridis",
        save_path=RESULTS_DIR / "03_conv1_feature_maps.png",
    )

    # TODO: Conv2 Feature Maps 16개 시각화
    show_feature_maps(
        features["conv2"],
        title="Conv2 Feature Maps",
        max_maps=16,
        cmap="viridis",
        save_path=RESULTS_DIR / "03_conv2_feature_maps.png",
    )

    # TODO: 6개 Layer별 대표 Feature Map 4개씩 비교
    layer_names = [
        "conv1",
        "relu1",
        "pool1",
        "conv2",
        "relu2",
        "pool2"
    ]

    # 행 = Layer, 열 = 채널 0~3. 같은 열은 같은 채널 번호
    plt.figure(figsize=(12, 3 * len(layer_names)))
    for row_index, layer_name in enumerate(layer_names):
        feature_maps = features[layer_name][0].detach().cpu()  # [Channel, Height, Width]
        for channel_index in range(MAPS_PER_LAYER):
            plt.subplot(len(layer_names), MAPS_PER_LAYER, row_index * MAPS_PER_LAYER + channel_index + 1)
            plt.imshow(feature_maps[channel_index], cmap="viridis")
            height, width = feature_maps.shape[1:]
            plt.title(f"{layer_name} ch{channel_index} ({height}x{width})")
            plt.axis("off")
    plt.suptitle(f"Feature Maps by Layer (label: {true_label})")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "03_layer_comparison.png", bbox_inches="tight")

    print("Saved:", RESULTS_DIR / "03_conv1_feature_maps.png")
    print("Saved:", RESULTS_DIR / "03_conv2_feature_maps.png")
    print("Saved:", RESULTS_DIR / "03_layer_comparison.png")


if __name__ == "__main__":
    main()
