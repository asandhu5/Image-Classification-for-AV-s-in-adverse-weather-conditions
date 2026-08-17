"""
xception/model.py
──────────────────
Xception (Chollet 2017) implemented from scratch in PyTorch.
No external dependencies beyond torch/torchvision.

Architecture (thesis Table 5.4: 22.949M params):
    Entry flow  : 2 regular Conv2d blocks + 3 separable residual blocks
    Middle flow : 8 × identical separable residual blocks
    Exit flow   : 2 separable blocks + GlobalAvgPool → Linear(2048, num_classes)
    Input size  : 299 × 299

Depthwise Separable Convolution:
    DepthwiseConv2d(in, in, 3, groups=in) → PointwiseConv2d(in, out, 1)
    This factorisation reduces parameters significantly versus standard Conv2d.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.config import SharedConfig as C


# ── Building blocks ───────────────────────────────────────────────────────────

class SeparableConv2d(nn.Module):
    """Depthwise + Pointwise convolution."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1,
                 padding: int = 1, bias: bool = False) -> None:
        super().__init__()
        self.depthwise  = nn.Conv2d(in_channels, in_channels, kernel_size,
                                    stride=stride, padding=padding,
                                    groups=in_channels, bias=bias)
        self.pointwise  = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class Block(nn.Module):
    """
    Xception residual block: 3 × (SepConv → BN → ReLU) + shortcut.

    grow_first=True  : channel expansion happens in first layer (entry/middle)
    grow_first=False : channel expansion happens in last layer  (exit)
    """

    def __init__(self, in_filters: int, out_filters: int,
                 reps: int, stride: int = 1,
                 start_with_relu: bool = True,
                 grow_first: bool = True) -> None:
        super().__init__()
        self.skip = None
        if out_filters != in_filters or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_filters, out_filters, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_filters),
            )

        layers  = []
        filters = in_filters
        if grow_first:
            layers += [nn.ReLU(inplace=True)] if start_with_relu else []
            layers += [SeparableConv2d(in_filters, out_filters),
                       nn.BatchNorm2d(out_filters)]
            filters = out_filters

        for _ in range(reps - 1):
            layers += [nn.ReLU(inplace=True),
                       SeparableConv2d(filters, filters),
                       nn.BatchNorm2d(filters)]

        if not grow_first:
            layers += [nn.ReLU(inplace=True),
                       SeparableConv2d(in_filters, out_filters),
                       nn.BatchNorm2d(out_filters)]

        if stride != 1:
            layers.append(nn.MaxPool2d(3, stride=stride, padding=1))

        self.rep = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rep(x)
        skip = self.skip(x) if self.skip is not None else x
        return out + skip


# ── Full Xception ─────────────────────────────────────────────────────────────

class Xception(nn.Module):
    """
    Xception architecture for image classification.

    Faithful to Chollet (2017) with the final Dense layer adapted for
    num_classes outputs.
    """

    def __init__(self, num_classes: int = C.NUM_CLASSES) -> None:
        super().__init__()

        # Entry flow — stem
        self.conv1  = nn.Conv2d(3, 32, 3, stride=2, padding=0, bias=False)
        self.bn1    = nn.BatchNorm2d(32)
        self.conv2  = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2    = nn.BatchNorm2d(64)

        # Entry flow — residual blocks
        self.block1  = Block(64,  128, 2, stride=2, start_with_relu=False, grow_first=True)
        self.block2  = Block(128, 256, 2, stride=2, start_with_relu=True,  grow_first=True)
        self.block3  = Block(256, 728, 2, stride=2, start_with_relu=True,  grow_first=True)

        # Middle flow — 8 identical blocks
        self.block4  = Block(728, 728, 3, stride=1, start_with_relu=True, grow_first=True)
        self.block5  = Block(728, 728, 3, stride=1, start_with_relu=True, grow_first=True)
        self.block6  = Block(728, 728, 3, stride=1, start_with_relu=True, grow_first=True)
        self.block7  = Block(728, 728, 3, stride=1, start_with_relu=True, grow_first=True)
        self.block8  = Block(728, 728, 3, stride=1, start_with_relu=True, grow_first=True)
        self.block9  = Block(728, 728, 3, stride=1, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, 3, stride=1, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, 3, stride=1, start_with_relu=True, grow_first=True)

        # Exit flow
        self.block12 = Block(728, 1024, 2, stride=2, start_with_relu=True, grow_first=False)
        self.conv3   = SeparableConv2d(1024, 1536, 3, padding=1)
        self.bn3     = nn.BatchNorm2d(1536)
        self.conv4   = SeparableConv2d(1536, 2048, 3, padding=1)
        self.bn4     = nn.BatchNorm2d(2048)

        # Classifier
        self.fc = nn.Linear(2048, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stem
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)

        # Entry flow
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        # Middle flow (8 blocks)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)

        # Exit flow
        x = self.block12(x)
        x = F.relu(self.bn3(self.conv3(x)), inplace=True)
        x = F.relu(self.bn4(self.conv4(x)), inplace=True)

        # Global average pool + classification head
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        x = self.fc(x)
        return x

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def get_model() -> Xception:
    return Xception(num_classes=C.NUM_CLASSES)
