"""
utils/checkpoint.py
───────────────────
Save and load model checkpoints.

Checkpoint format (saved as .pth):
{
    "epoch":       int,
    "model_state": OrderedDict,
    "optim_state": OrderedDict,
    "val_loss":    float,
    "val_acc":     float,
    "config": {
        "embed_dim": ..., "num_heads": ...,  # for reproducibility
    }
}
"""
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from config import Config

logger = logging.getLogger(__name__)


def save_checkpoint(
    model:     nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch:     int,
    val_loss:  float,
    val_acc:   float,
    path:      str,
) -> None:
    """
    Save a full training checkpoint.

    Parameters
    ----------
    model     : CustomViT (state_dict saved)
    optimizer : AdamW (state_dict saved — allows training resumption)
    epoch     : current epoch number
    val_loss  : validation loss at this epoch
    val_acc   : validation accuracy (%) at this epoch
    path      : output .pth file path
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch":       epoch,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "val_loss":    val_loss,
        "val_acc":     val_acc,
        "config": {
            "embed_dim":  Config.EMBED_DIM,
            "num_heads":  Config.NUM_HEADS,
            "num_layers": Config.NUM_LAYERS,
            "mlp_dim":    Config.MLP_DIM,
            "dropout":    Config.DROPOUT,
            "num_classes": Config.NUM_CLASSES,
        },
    }
    torch.save(payload, path)
    logger.debug(f"Saved checkpoint → {path}  [epoch={epoch}, val_acc={val_acc:.2f}%]")


def load_checkpoint(
    path:      str,
    model:     nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device:    str = Config.DEVICE,
) -> dict:
    """
    Load a checkpoint into model (and optionally optimizer).

    Parameters
    ----------
    path      : path to .pth file
    model     : CustomViT instance (weights loaded in-place)
    optimizer : if provided, optimizer state is restored too
    device    : map_location for torch.load

    Returns
    -------
    The raw checkpoint dict (contains epoch, val_loss, val_acc, config).
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    if optimizer is not None and "optim_state" in ckpt:
        optimizer.load_state_dict(ckpt["optim_state"])

    logger.info(
        f"Loaded checkpoint from {path}  "
        f"[epoch={ckpt.get('epoch', '?')}, "
        f"val_acc={ckpt.get('val_acc', 0.0):.2f}%]"
    )
    return ckpt


def load_best_checkpoint(
    model:    nn.Module,
    ckpt_dir: str = str(Config.CHECKPOINT_DIR),
    device:   str = Config.DEVICE,
) -> dict:
    """
    Convenience: load checkpoints/best.pth into model.

    Raises FileNotFoundError if best.pth does not exist.
    """
    best_path = Path(ckpt_dir) / "best.pth"
    return load_checkpoint(str(best_path), model, device=device)


def resume_from_checkpoint(
    path:      str,
    model:     nn.Module,
    optimizer: torch.optim.Optimizer,
    device:    str = Config.DEVICE,
) -> int:
    """
    Load checkpoint and return the epoch to resume from.

    Usage:
        start_epoch = resume_from_checkpoint("checkpoints/epoch_050.pth", model, optimizer)
        for epoch in range(start_epoch, Config.EPOCHS + 1):
            ...
    """
    ckpt = load_checkpoint(path, model, optimizer, device)
    return ckpt.get("epoch", 0) + 1
