"""
shared/models/vit_base.py
──────────────────────────
Parameterised Vision Transformer base used by all 5 standard ViT variants
(B-16, B-32, L-16, L-32, H-14) AND the custom proposed model.

Differences from the proposed CustomViT in thesis-voc/:
    • Image size and patch size are fully configurable
    • No num_heads adjustment — caller must ensure embed_dim % num_heads == 0
    • Gradient checkpointing flag for large models (L-16, L-32, H-14) that
      would otherwise exceed 24 GB VRAM at batch_size=16

Variants from Table 5.3:
    B-16: heads=12, dim=768,  layers=12, mlp=3072, patch=16
    B-32: heads=12, dim=768,  layers=12, mlp=3072, patch=32
    L-16: heads=16, dim=1024, layers=24, mlp=4096, patch=16
    L-32: heads=16, dim=1024, layers=24, mlp=4096, patch=32
    H-14: heads=16, dim=1280, layers=32, mlp=5120, patch=14
"""
import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt_util


class PatchEmbedding(nn.Module):
    def __init__(self, img_size: int, patch_size: int,
                 in_channels: int = 3, embed_dim: int = 768) -> None:
        super().__init__()
        assert img_size % patch_size == 0
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, mlp_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x): return self.net(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int,
                 mlp_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = nn.MultiheadAttention(embed_dim, num_heads,
                                           dropout=dropout, batch_first=True)
        self.drop  = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff    = FeedForward(embed_dim, mlp_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n      = self.norm1(x)
        a, _   = self.attn(n, n, n)
        x      = x + self.drop(a)
        x      = x + self.ff(self.norm2(x))
        return x


class ViT(nn.Module):
    """
    Standard Vision Transformer.

    Parameters
    ----------
    img_size    : int   — input image resolution (H = W)
    patch_size  : int   — patch size (16, 32, or 14)
    in_channels : int   — 3 for RGB
    num_classes : int   — output classes
    embed_dim   : int   — token embedding dimension (D_model)
    num_heads   : int   — attention heads (must divide embed_dim)
    num_layers  : int   — transformer encoder depth
    mlp_dim     : int   — FFN hidden dimension (D_mlp)
    dropout     : float — dropout probability
    grad_ckpt   : bool  — enable gradient checkpointing (saves VRAM for L/H)
    """

    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 in_channels: int = 3, num_classes: int = 10,
                 embed_dim: int = 768, num_heads: int = 12,
                 num_layers: int = 12, mlp_dim: int = 3072,
                 dropout: float = 0.1, grad_ckpt: bool = False) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"

        self.grad_ckpt   = grad_ckpt
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches      = self.patch_embed.num_patches

        self.cls_token   = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed   = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop    = nn.Dropout(dropout)

        self.encoder     = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm        = nn.LayerNorm(embed_dim)

        # Classification head: CLS token → Linear
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B  = x.shape[0]
        x  = self.patch_embed(x)
        x  = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x  = self.pos_drop(x + self.pos_embed)

        for block in self.encoder:
            if self.grad_ckpt and self.training:
                x = ckpt_util.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        x = self.norm(x)
        return self.head(x[:, 0])

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def parameter_summary(self) -> str:
        t = self.num_parameters
        return f"Total: {t:,}  ({t/1e6:.2f}M)"
