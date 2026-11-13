"""Model zoo. The reference CNN is given; the alternative is part of assignment A6."""
import torch.nn as nn
from torchvision import models


class ReferenceCNN(nn.Module):
    """Small reference CNN: 3 conv blocks + MLP head (for 224x224 input)."""

    def __init__(self, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),   # 112
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),  # 56
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2), # 28
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),   # resolution-agnostic: works for any input size
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(name, num_classes):
    if name == "reference_cnn":
        return ReferenceCNN(num_classes)
    elif name == "resnet18":
        # A6.6: ResNet-18 pretrained on ImageNet (transfer learning: only 3,000 training images),
        # last layer replaced to predict the 5 DR grades. DEFAULT = IMAGENET1K_V1 in torchvision 0.15+.
        resnet_18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        resnet_18.fc = nn.Linear(resnet_18.fc.in_features, num_classes)
        return resnet_18
    raise ValueError(f"unknown model: {name}")
