"""
vit_b16/model.py
──────────────────
ViT-B/16 Vision Transformer for 10-class BDD100K ROI classification.

Architecture (thesis Table 5.3):
    Attention heads     : 12
    Embedding dimension : 768
    Encoder layers      : 12
    MLP dim             : 3072
    Patch size          : 16
    Input size          : 224 × 224
    Parameters          : ~76.41M  (thesis Table 5.4)
    Gradient checkpointing: False  (needed for VRAM at batch_size=16)
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
        embed_dim   = 768,
        num_heads   = 12,
        num_layers  = 12,
        mlp_dim     = 3072,
        dropout     = 0.1,
        grad_ckpt   = False,
    )
