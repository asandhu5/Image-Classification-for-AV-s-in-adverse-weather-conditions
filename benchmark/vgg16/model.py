"""
vgg16/model.py
──────────────
VGG16 adapted for 10-class BDD100K object classification.

Architecture:
    Backbone  : VGG16 convolutional feature extractor (torchvision)
    Head      : Classifier[6] replaced — Linear(4096, 10) for our 10 classes
    Parameters: ~136.9M  (matches thesis Table 5.4)
    Input size: 224 × 224

Training from scratch (pretrained=False) matches the thesis comparison.
Set pretrained=True to use ImageNet weights for faster convergence.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch.nn as nn
import torchvision.models as models

from shared.config import SharedConfig as C


class VGG16(nn.Module):
    """
    VGG16 with the final fully-connected layer replaced for num_classes outputs.

    The original VGG16 classifier:
        Linear(25088, 4096) → ReLU → Dropout
        Linear(4096,  4096) → ReLU → Dropout
        Linear(4096,  1000)  ← replaced with Linear(4096, num_classes)
    """

    def __init__(
        self,
        num_classes: int = C.NUM_CLASSES,
        pretrained:  bool = False,
    ) -> None:
        super().__init__()
        weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        base    = models.vgg16(weights=weights)

        self.features    = base.features        # Conv stack (14 conv layers)
        self.avgpool     = base.avgpool         # AdaptiveAvgPool2d(7, 7)
        self.classifier  = base.classifier      # 3-layer FC stack

        # Replace the final classification layer
        in_features = self.classifier[-1].in_features   # 4096
        self.classifier[-1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def get_model(pretrained: bool = False) -> VGG16:
    """Factory function used by train.py / evaluate.py / inference.py."""
    return VGG16(num_classes=C.NUM_CLASSES, pretrained=pretrained)
