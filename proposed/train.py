"""
train.py
────────
Entry point for training the CustomViT on BDD100K.

Usage
─────
    python train.py                           # full 200-epoch run
    python train.py --epochs 50              # quick test run
    python train.py --resume checkpoints/epoch_050.pth

What this script does
─────────────────────
  1. Build ROI datasets from BDD100K (70 / 10 / 20 split)
  2. Train CustomViT for 200 epochs with AdamW + cosine LR
  3. Save best.pth + periodic checkpoints to ./checkpoints/
  4. Save training history JSON + plots to ./outputs/
  5. Run final evaluation on the test split and print results
"""
import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch

from config import Config
from dataset import get_dataloaders, get_train_transforms, get_val_transforms
from engine.trainer import Trainer
from engine.evaluator import Evaluator
from models.vit import CustomViT
from utils.checkpoint import load_checkpoint
from utils.metrics import (
    compute_confusion_matrix,
    compute_classification_report,
    compute_per_class_metrics,
    print_results_table,
)
from utils.visualization import (
    plot_training_curves,
    plot_confusion_matrix,
    plot_per_class_metrics,
    plot_lr_curve,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CustomViT on BDD100K")
    p.add_argument("--epochs",  type=int, default=Config.EPOCHS,
                   help="Number of training epochs (default: 200)")
    p.add_argument("--resume",  type=str, default=None,
                   help="Resume training from a checkpoint path")
    p.add_argument("--data-root", type=str, default=str(Config.DATA_ROOT),
                   help="Root directory of the BDD100K dataset")
    p.add_argument("--output-dir", type=str, default=str(Config.OUTPUT_DIR),
                   help="Directory for plots and results")
    p.add_argument("--ckpt-dir", type=str, default=str(Config.CHECKPOINT_DIR),
                   help="Directory for saved checkpoints")
    return p.parse_args()


def set_seed(seed: int = Config.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
        handlers = [
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "train.log"),
        ],
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Override paths from CLI args
    data_root  = Path(args.data_root)
    output_dir = Path(args.output_dir)
    ckpt_dir   = Path(args.ckpt_dir)

    setup_logging(Config.LOG_DIR)
    Config.make_dirs()
    set_seed()

    logger = logging.getLogger("train")
    logger.info(Config.summary())

    # ── 1. Datasets ───────────────────────────────────────────────────
    logger.info("Building ROI datasets from BDD100K …")
    train_json    = str(data_root / "labels" / "det_20" / "det_train.json")
    train_img_dir = str(data_root / "images" / "100k" / "train")

    train_loader, val_loader, test_loader = get_dataloaders(
        train_json      = train_json,
        train_img_dir   = train_img_dir,
        train_transform = get_train_transforms(),
        val_transform   = get_val_transforms(),
    )
    logger.info(
        f"Split  →  train={len(train_loader.dataset):,}  "
        f"val={len(val_loader.dataset):,}  "
        f"test={len(test_loader.dataset):,}"
    )

    # Class distribution (useful for diagnosing imbalance)
    dist = train_loader.dataset.class_distribution()
    logger.info("Train class distribution:")
    for cls, cnt in dist.items():
        logger.info(f"  {cls:<16}: {cnt:,}")

    # ── 2. Model ──────────────────────────────────────────────────────
    model = CustomViT()
    logger.info(model.parameter_summary())

    # ── 3. Optionally resume ──────────────────────────────────────────
    start_epoch = 1
    if args.resume:
        from utils.checkpoint import resume_from_checkpoint
        from torch.optim import AdamW
        # Dummy optimizer — will be overwritten by checkpoint
        dummy_opt = AdamW(model.parameters())
        start_epoch = resume_from_checkpoint(args.resume, model, dummy_opt)
        logger.info(f"Resuming from epoch {start_epoch}")

    # ── 4. Train ──────────────────────────────────────────────────────
    trainer = Trainer(model, train_loader, val_loader,
                      device=Config.DEVICE, output_dir=str(ckpt_dir))
    history = trainer.train(epochs=args.epochs)

    # ── 5. Save history ───────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    hist_path = output_dir / "training_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"History saved → {hist_path}")

    # ── 6. Plots ──────────────────────────────────────────────────────
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_training_curves(history, str(plots_dir / "training_curves.png"))
    plot_lr_curve(history,        str(plots_dir / "lr_schedule.png"))
    logger.info("Training plots saved.")

    # ── 7. Test evaluation ────────────────────────────────────────────
    logger.info("Evaluating best checkpoint on test split …")
    load_checkpoint(str(ckpt_dir / "best.pth"), model, device=Config.DEVICE)
    evaluator = Evaluator(model, Config.DEVICE)
    results   = evaluator.evaluate(test_loader, "BDD100K-Test")

    preds  = results["all_preds"]
    labels = results["all_labels"]

    cm     = compute_confusion_matrix(preds, labels)
    report = compute_classification_report(preds, labels)
    pcm    = compute_per_class_metrics(preds, labels)

    logger.info(f"\n{report}")
    print_results_table({"BDD100K-Test": results})

    plot_confusion_matrix(
        cm, title="Confusion Matrix — BDD100K Test",
        save_path=str(plots_dir / "confusion_matrix_bdd100k.png"),
    )
    plot_per_class_metrics(
        pcm,
        save_path=str(plots_dir / "per_class_metrics_bdd100k.png"),
    )

    # ── 8. Save final results JSON ────────────────────────────────────
    final = {
        "test_accuracy": results["accuracy"],
        "inference_s":   results["inference_s"],
        "n_samples":     results["n_samples"],
        "per_class":     pcm,
    }
    with open(output_dir / "test_results.json", "w") as f:
        json.dump(final, f, indent=2)

    logger.info(
        f"\n{'='*50}\n"
        f"  Final test accuracy : {results['accuracy']:.2f}%\n"
        f"  Inference time      : {results['inference_s']:.4f}s\n"
        f"{'='*50}"
    )


if __name__ == "__main__":
    main()
