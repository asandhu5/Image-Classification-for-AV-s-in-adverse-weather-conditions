"""
engine/trainer.py
─────────────────
Full training loop for CustomViT.

Hyperparameters (thesis Table 4.2 / 4.3 / 4.4):
    Optimizer    : AdamW  (lr=1e-3, weight_decay=1e-6)
    Activation   : GELU   (baked into model architecture)
    Epochs       : 200
    Batch size   : 16
    Grad clip    : 1.0
    LR schedule  : Linear warmup (10 epochs) → CosineAnnealing (190 epochs)
    Loss         : CrossEntropyLoss with label_smoothing=0.1

Checkpointing:
    checkpoints/best.pth         — lowest val-loss so far
    checkpoints/epoch_NNN.pth   — every SAVE_EVERY epochs
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

from config import Config
from utils.checkpoint import save_checkpoint

logger = logging.getLogger(__name__)


class Trainer:

    def __init__(
        self,
        model:        nn.Module,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        device:       str = Config.DEVICE,
        output_dir:   Optional[str] = None,
    ) -> None:
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device
        self.output_dir   = Path(output_dir or Config.CHECKPOINT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ── Loss ──────────────────────────────────────────────────────
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=Config.LABEL_SMOOTH,
        )

        # ── Optimizer (AdamW — thesis Table 4.4) ─────────────────────
        self.optimizer = AdamW(
            self.model.parameters(),
            lr           = Config.LEARNING_RATE,
            weight_decay = Config.WEIGHT_DECAY,
            betas        = (0.9, 0.999),
            eps          = 1e-8,
        )

        # ── LR Schedule: linear warmup → cosine annealing ─────────────
        warmup = LinearLR(
            self.optimizer,
            start_factor = 1e-3,
            end_factor   = 1.0,
            total_iters  = Config.WARMUP_EPOCHS,
        )
        cosine = CosineAnnealingLR(
            self.optimizer,
            T_max   = Config.EPOCHS - Config.WARMUP_EPOCHS,
            eta_min = Config.LEARNING_RATE * 1e-3,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers = [warmup, cosine],
            milestones = [Config.WARMUP_EPOCHS],
        )

        # ── Tracking ──────────────────────────────────────────────────
        self.history: Dict[str, list] = {
            "train_loss": [], "train_acc": [],
            "val_loss":   [], "val_acc":   [],
            "lr":         [],
        }
        self.best_val_loss = float("inf")
        self.best_epoch    = 0

    # ── Public API ────────────────────────────────────────────────────

    def train(self, epochs: int = Config.EPOCHS) -> Dict[str, list]:
        """
        Run the complete training loop.

        Returns
        -------
        history dict with keys: train_loss, train_acc, val_loss, val_acc, lr
        """
        logger.info(Config.summary())
        logger.info(
            f"Training for {epochs} epochs  |  "
            f"device={self.device}  |  "
            f"params={self.model.num_parameters:,}"
        )

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            train_loss, train_acc = self._train_epoch(epoch, epochs)
            val_loss,   val_acc   = self._val_epoch(epoch, epochs)
            lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(lr)

            elapsed = time.time() - t0
            logger.info(
                f"[{epoch:3d}/{epochs}]  "
                f"train_loss={train_loss:.4f}  train_acc={train_acc:.2f}%  "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.2f}%  "
                f"lr={lr:.2e}  time={elapsed:.1f}s"
            )

            # ── Checkpoint: best model ─────────────────────────────
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch    = epoch
                save_checkpoint(
                    self.model, self.optimizer,
                    epoch, val_loss, val_acc,
                    str(self.output_dir / "best.pth"),
                )
                logger.info(f"  ★ New best val_loss={val_loss:.4f} — checkpoint saved.")

            # ── Checkpoint: periodic ──────────────────────────────
            if epoch % Config.SAVE_EVERY == 0:
                save_checkpoint(
                    self.model, self.optimizer,
                    epoch, val_loss, val_acc,
                    str(self.output_dir / f"epoch_{epoch:03d}.pth"),
                )

        logger.info(
            f"Training complete.  "
            f"Best val_loss={self.best_val_loss:.4f} at epoch {self.best_epoch}."
        )
        return self.history

    # ── Private helpers ───────────────────────────────────────────────

    def _train_epoch(self, epoch: int, total: int) -> Tuple[float, float]:
        self.model.train()
        running_loss = 0.0
        correct      = 0
        n_samples    = 0

        pbar = tqdm(
            self.train_loader,
            desc  = f"Train [{epoch}/{total}]",
            leave = False,
            ncols = 100,
        )
        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(images)
            loss   = self.criterion(logits, labels)
            loss.backward()

            # Gradient norm clipping — stabilises ViT training
            nn.utils.clip_grad_norm_(self.model.parameters(), Config.GRAD_CLIP)
            self.optimizer.step()

            bs            = images.size(0)
            running_loss += loss.item() * bs
            correct      += (logits.argmax(1) == labels).sum().item()
            n_samples    += bs

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return running_loss / n_samples, 100.0 * correct / n_samples

    @torch.no_grad()
    def _val_epoch(self, epoch: int, total: int) -> Tuple[float, float]:
        self.model.eval()
        running_loss = 0.0
        correct      = 0
        n_samples    = 0

        for images, labels in tqdm(
            self.val_loader,
            desc  = f"Val   [{epoch}/{total}]",
            leave = False,
            ncols = 100,
        ):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            logits        = self.model(images)
            loss          = self.criterion(logits, labels)
            bs            = images.size(0)
            running_loss += loss.item() * bs
            correct      += (logits.argmax(1) == labels).sum().item()
            n_samples    += bs

        return running_loss / n_samples, 100.0 * correct / n_samples
