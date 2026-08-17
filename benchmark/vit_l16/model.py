"""
vit_l16/model.py
──────────────────
ViT-L/16 Vision Transformer for 10-class BDD100K ROI classification.

Architecture (thesis Table 5.3):
    Attention heads     : 16
    Embedding dimension : 1024
    Encoder layers      : 24
    MLP dim             : 4096
    Patch size          : 16
    Input size          : 224 × 224
    Parameters          : ~273.74M  (thesis Table 5.4)
    Gradient checkpointing: True  (needed for VRAM at batch_size=16)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.models.vit_base import ViT
from shared.config import SharedConfig as C


def get_model() -> ViT:
    return ViT(
        img_size    = 224,
        patch_size  = 16,
        in_channels = C.IN_CHANNELS,
        num_classes = C.NUM_CLASSES,
        embed_dim   = 1024,
        num_heads   = 16,
        num_layers  = 24,
        mlp_dim     = 4096,
        dropout     = 0.1,
        grad_ckpt   = True,
    )
