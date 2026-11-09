"""Model definitions.

- Reference CNN: 3 conv blocks (32/64/128 filters, 3x3) with 2x2 max pooling,
  then fully connected layers 512 -> 256 -> 128 -> num_classes.
- At least one justified alternative (e.g. lightweight ResNet or pretrained model).
"""

import torch.nn as nn


class ReferenceCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


def build_model(name, num_classes=5):
    """Factory: 'reference_cnn', 'resnet18', ..."""
    raise NotImplementedError
