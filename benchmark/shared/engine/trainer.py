"""
shared/engine/trainer.py
─────────────────────────
Unified training loop for all benchmark models.

Handles two cases transparently:
  • Standard models  → model(x) returns (B, num_classes) logits
  • InceptionV3      → model(x) returns InceptionOutputs(logits, aux_logits)
                       loss = main_loss + 0.4 * aux_loss  (Szegedy et al.)
"""
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from shared.config import SharedConfig as C
from shared.utils.checkpoint import save_checkpoint

logger = logging.getLogger(__name__)


class Trainer:

    def __init__(self, model: nn.Module, train_loader: DataLoader,
                 val_loader: DataLoader, cfg,
                 device: str = C.DEVICE,
                 output_dir: Optional[str] = None):
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.cfg          = cfg
        self.device       = device
        self.output_dir   = Path(output_dir or cfg.CHECKPOINT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.criterion = nn.CrossEntropyLoss(label_smoothing=C.LABEL_SMOOTH)

        self.optimizer = AdamW(self.model.parameters(),
                               lr=C.LEARNING_RATE, weight_decay=C.WEIGHT_DECAY,
                               betas=(0.9, 0.999))

        warmup = LinearLR(self.optimizer, start_factor=1e-3, end_factor=1.0,
                          total_iters=C.WARMUP_EPOCHS)
        cosine = CosineAnnealingLR(self.optimizer,
                                   T_max=C.EPOCHS - C.WARMUP_EPOCHS,
                                   eta_min=C.LEARNING_RATE * 1e-3)
        self.scheduler = SequentialLR(self.optimizer, [warmup, cosine],
                                      milestones=[C.WARMUP_EPOCHS])

        self.history    = {"train_loss": [], "train_acc": [],
                           "val_loss":   [], "val_acc":   [], "lr": []}
        self.best_val_loss = float("inf")
        self.best_epoch    = 0

    # ── Public ───────────────────────────────────────────────────────

    def train(self, epochs: int = C.EPOCHS) -> Dict[str, list]:
        logger.info(f"[{self.cfg.MODEL_NAME}] Training {epochs} epochs on {self.device}")
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc = self._train_epoch(epoch, epochs)
            va_loss, va_acc = self._val_epoch(epoch, epochs)
            lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step()

            self.history["train_loss"].append(tr_loss)
            self.history["train_acc"].append(tr_acc)
            self.history["val_loss"].append(va_loss)
            self.history["val_acc"].append(va_acc)
            self.history["lr"].append(lr)

            logger.info(f"[{epoch:3d}/{epochs}] "
                        f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.2f}% "
                        f"va_loss={va_loss:.4f} va_acc={va_acc:.2f}% "
                        f"lr={lr:.2e} [{time.time()-t0:.1f}s]")

            if va_loss < self.best_val_loss:
                self.best_val_loss = va_loss
                self.best_epoch    = epoch
                save_checkpoint(self.model, self.optimizer, epoch, va_loss, va_acc,
                                str(self.output_dir / "best.pth"))
                logger.info(f"  ★ New best → saved")

            if epoch % C.SAVE_EVERY == 0:
                save_checkpoint(self.model, self.optimizer, epoch, va_loss, va_acc,
                                str(self.output_dir / f"epoch_{epoch:03d}.pth"))

        logger.info(f"Done. Best val_loss={self.best_val_loss:.4f} at epoch {self.best_epoch}")
        return self.history

    # ── Private ───────────────────────────────────────────────────────

    def _forward_loss(self, images, labels):
        """Handle both standard and InceptionV3 (aux) outputs."""
        out = self.model(images)
        if isinstance(out, tuple):          # InceptionV3 InceptionOutputs
            logits, aux = out[0], out[1]
            if aux is not None:
                loss = self.criterion(logits, labels) + 0.4 * self.criterion(aux, labels)
            else:
                loss = self.criterion(logits, labels)
        else:
            logits = out
            loss   = self.criterion(logits, labels)
        return logits, loss

    def _train_epoch(self, epoch: int, total: int) -> Tuple[float, float]:
        self.model.train()
        running_loss, correct, n = 0.0, 0, 0
        pbar = tqdm(self.train_loader, desc=f"Train [{epoch}/{total}]",
                    leave=False, ncols=100)
        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            logits, loss = self._forward_loss(images, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), C.GRAD_CLIP)
            self.optimizer.step()
            bs            = images.size(0)
            running_loss += loss.item() * bs
            correct      += (logits.argmax(1) == labels).sum().item()
            n            += bs
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        return running_loss / n, 100.0 * correct / n

    @torch.no_grad()
    def _val_epoch(self, epoch: int, total: int) -> Tuple[float, float]:
        self.model.eval()
        running_loss, correct, n = 0.0, 0, 0
        for images, labels in tqdm(self.val_loader, desc=f"Val   [{epoch}/{total}]",
                                   leave=False, ncols=100):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            logits, loss  = self._forward_loss(images, labels)
            bs            = images.size(0)
            running_loss += loss.item() * bs
            correct      += (logits.argmax(1) == labels).sum().item()
            n            += bs
        return running_loss / n, 100.0 * correct / n
