"""
models/vit.py
─────────────
Custom Vision Transformer (ViT) for object ROI classification.

Architecture (thesis Table 5.2):
    img_size   = 224   patch_size = 16  →  196 patches
    embed_dim  = 256   num_heads  = 8   (thesis: 6 — see note below)
    num_layers = 8     mlp_dim    = 512
    dropout    = 0.10  num_classes = 10

Note on num_heads
─────────────────
The thesis specifies 6 attention heads (Table 5.2), but
embed_dim=256 is not divisible by 6 (256 % 6 = 4 ≠ 0), which would
raise a RuntimeError in nn.MultiheadAttention.  The nearest valid choice
is 8 heads (256 / 8 = 32 dims per head), which keeps roughly the same
head granularity as the standard ViT-B configuration.

Pipeline
────────
image (3,224,224)
  → PatchEmbedding           → (B, 196, 256)
  → prepend [CLS] token      → (B, 197, 256)
  → add positional embedding → (B, 197, 256)
  → 8 × TransformerEncoderBlock  → (B, 197, 256)
  → LayerNorm
  → CLS token slice          → (B, 256)
  → MLP head (256→512→10)    → (B, 10)  logits
"""
import math

import torch
import torch.nn as nn

from config import Config


# ──────────────────────────────────────────────────────────────────────────────
# Patch Embedding
# ──────────────────────────────────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """
    Split a 224×224 image into non-overlapping 16×16 patches and linearly
    project each patch to embed_dim.

    Uses a single Conv2d(kernel=patch_size, stride=patch_size) — equivalent to
    the linear projection in the original ViT but more efficient on GPU.
    """

    def __init__(
        self,
        img_size:    int   = Config.IMG_SIZE,
        patch_size:  int   = Config.PATCH_SIZE,
        in_channels: int   = Config.IN_CHANNELS,
        embed_dim:   int   = Config.EMBED_DIM,
    ) -> None:
        super().__init__()
        assert img_size % patch_size == 0, (
            f"img_size ({img_size}) must be divisible by patch_size ({patch_size})"
        )
        self.num_patches = (img_size // patch_size) ** 2   # 196

        # Single conv projects HxW patches to embed_dim in one pass
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, C, H, W)
        x = self.proj(x)       # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)       # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


# ──────────────────────────────────────────────────────────────────────────────
# Feed-Forward Network (FFN) inside encoder block
# ──────────────────────────────────────────────────────────────────────────────

class FeedForward(nn.Module):
    """
    Two-layer MLP used in every Transformer encoder block:
        Linear(embed_dim → mlp_dim) → GELU → Dropout
        → Linear(mlp_dim → embed_dim) → Dropout
    """

    def __init__(
        self,
        embed_dim: int   = Config.EMBED_DIM,
        mlp_dim:   int   = Config.MLP_DIM,
        dropout:   float = Config.DROPOUT,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ──────────────────────────────────────────────────────────────────────────────
# Transformer Encoder Block
# ──────────────────────────────────────────────────────────────────────────────

class TransformerEncoderBlock(nn.Module):
    """
    Pre-norm Transformer encoder block (matches the original ViT paper):

        x  →  LayerNorm  →  Multi-Head Self-Attention  →  + residual  →  x
        x  →  LayerNorm  →  FeedForward                →  + residual  →  x
    """

    def __init__(
        self,
        embed_dim: int   = Config.EMBED_DIM,
        num_heads: int   = Config.NUM_HEADS,
        mlp_dim:   int   = Config.MLP_DIM,
        dropout:   float = Config.DROPOUT,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,    # (B, seq, dim) convention
        )
        self.attn_drop = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff    = FeedForward(embed_dim, mlp_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Multi-head self-attention with pre-norm + residual
        normed   = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + self.attn_drop(attn_out)

        # 2. FFN with pre-norm + residual
        x = x + self.ff(self.norm2(x))
        return x


# ──────────────────────────────────────────────────────────────────────────────
# Custom ViT — full model
# ──────────────────────────────────────────────────────────────────────────────

class CustomViT(nn.Module):
    """
    Lightweight Vision Transformer for 10-class object ROI classification.

    Designed to receive YOLOv8m-detected or ground-truth bounding-box crops
    (resized to 224×224) and output class logits.

    Parameters mirror the thesis Table 5.2 (and hardware constraints of
    RTX 3090 / 24 GB VRAM with batch_size=16).
    """

    def __init__(
        self,
        img_size:    int   = Config.IMG_SIZE,
        patch_size:  int   = Config.PATCH_SIZE,
        in_channels: int   = Config.IN_CHANNELS,
        num_classes: int   = Config.NUM_CLASSES,
        embed_dim:   int   = Config.EMBED_DIM,
        num_heads:   int   = Config.NUM_HEADS,
        num_layers:  int   = Config.NUM_LAYERS,
        mlp_dim:     int   = Config.MLP_DIM,
        dropout:     float = Config.DROPOUT,
    ) -> None:
        super().__init__()

        # ── Patch embedding ───────────────────────────────────────────
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches      = self.patch_embed.num_patches   # 196

        # ── Learnable CLS token + positional embeddings ───────────────
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop  = nn.Dropout(dropout)

        # ── Transformer encoder (8 blocks) ────────────────────────────
        self.encoder = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # ── MLP classification head ────────────────────────────────────
        # CLS token (256) → hidden (512, GELU) → num_classes (10)
        self.mlp_head = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes),
        )

        self._init_weights()

    # ── Weight initialisation ─────────────────────────────────────────

    def _init_weights(self) -> None:
        """Truncated-normal init — standard for ViT."""
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                # Kaiming init for the patch-projection conv
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── Forward pass ──────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 3, 224, 224)  — pre-processed ROI crops

        Returns
        -------
        logits : (B, num_classes)
        """
        B = x.shape[0]

        # 1. Patch embedding → (B, 196, 256)
        x = self.patch_embed(x)

        # 2. Prepend [CLS] token → (B, 197, 256)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # 3. Add positional embeddings → (B, 197, 256)
        x = self.pos_drop(x + self.pos_embed)

        # 4. Transformer encoder
        for block in self.encoder:
            x = block(x)
        x = self.norm(x)

        # 5. CLS token → MLP head → logits
        cls_out = x[:, 0]          # (B, 256)
        logits  = self.mlp_head(cls_out)   # (B, 10)
        return logits

    # ── Convenience properties ────────────────────────────────────────

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_summary(self) -> str:
        total     = self.num_parameters
        trainable = self.num_trainable_parameters
        return (
            f"Total parameters     : {total:,}\n"
            f"Trainable parameters : {trainable:,}\n"
            f"Non-trainable        : {total - trainable:,}"
        )

    # ── Attention-map extraction for interpretability ─────────────────

    @torch.no_grad()
    def get_attention_maps(self, x: torch.Tensor, layer: int = -1) -> torch.Tensor:
        """
        Extract attention weights from a specific encoder layer.

        Returns
        -------
        attn_maps : (B, num_heads, 197, 197)
        """
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed

        layer_idx = layer % len(self.encoder)
        for i, block in enumerate(self.encoder):
            if i < layer_idx:
                x = block(x)
            else:
                # Run just the attention part and capture weights
                normed = block.norm1(x)
                _, attn_weights = block.attn(
                    normed, normed, normed,
                    need_weights=True,
                    average_attn_weights=False,
                )
                return attn_weights   # (B, num_heads, 197, 197)
        raise ValueError(f"Layer {layer} out of range")
