"""
resnet101/model.py
──────────────────
ResNet-101 adapted for 10-class BDD100K object classification.

Architecture:
    Backbone  : ResNet-101 (torchvision) — 100 conv layers + skip connections
    Head      : model.fc replaced — Linear(2048, 10)
    Parameters: ~47.2M  (thesis Table 5.4: 47.156M)
    Input size: 224 × 224

The skip connections make ResNet-101 much deeper than VGG16 while
remaining trainable from scratch thanks to residual learning.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch.nn as nn
import torchvision.models as models

from shared.config import SharedConfig as C


class ResNet101(nn.Module):
    """
    ResNet-101 with the final fully-connected layer replaced.

    Original fc: Linear(2048, 1000)
    Replaced  : Linear(2048, num_classes)
    """

    def __init__(
        self,
        num_classes: int = C.NUM_CLASSES,
        pretrained:  bool = False,
    ) -> None:
        super().__init__()
        weights    = models.ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
        base       = models.resnet101(weights=weights)
        in_features = base.fc.in_features          # 2048

        # Replace the classification head
        base.fc    = nn.Linear(in_features, num_classes)
        self.model = base

    def forward(self, x):
        return self.model(x)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def get_model(pretrained: bool = False) -> ResNet101:
    return ResNet101(num_classes=C.NUM_CLASSES, pretrained=pretrained)
