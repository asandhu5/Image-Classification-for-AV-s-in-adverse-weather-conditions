"""
inceptionv3/model.py
─────────────────────
Inception-V3 adapted for 10-class BDD100K object classification.

Architecture:
    Backbone  : Inception-V3 (torchvision, Szegedy et al. 2016)
    Head      : model.fc replaced — Linear(2048, 10)
                model.AuxLogits.fc replaced — Linear(768, 10)
    Parameters: ~29.6M  (thesis Table 5.4: 29.577M)
    Input size: 299 × 299  (Inception's native resolution)

Training note:
    During training, model(x) returns InceptionOutputs(logits, aux_logits).
    The shared Trainer handles this: loss = main_loss + 0.4 * aux_loss.
    During eval/inference, model.eval() causes model(x) to return logits only.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch.nn as nn
import torchvision.models as models

from shared.config import SharedConfig as C


class InceptionV3(nn.Module):
    """
    Inception-V3 with both the main and auxiliary classification heads
    replaced for num_classes outputs.
    """

    def __init__(
        self,
        num_classes: int = C.NUM_CLASSES,
        pretrained:  bool = False,
    ) -> None:
        super().__init__()
        weights     = models.Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None
        # aux_logits=True during training for auxiliary loss
        self.model  = models.inception_v3(weights=weights, aux_logits=True)

        # Replace main classifier
        in_main     = self.model.fc.in_features           # 2048
        self.model.fc = nn.Linear(in_main, num_classes)

        # Replace auxiliary classifier (used only during training)
        in_aux      = self.model.AuxLogits.fc.in_features  # 768
        self.model.AuxLogits.fc = nn.Linear(in_aux, num_classes)

    def forward(self, x):
        """
        Returns:
            training  : InceptionOutputs(logits, aux_logits)
            eval mode : logits tensor (no aux)
        """
        return self.model(x)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def get_model(pretrained: bool = False) -> InceptionV3:
    return InceptionV3(num_classes=C.NUM_CLASSES, pretrained=pretrained)
