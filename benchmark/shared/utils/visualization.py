"""
utils/visualization.py
──────────────────────
All plots produced in thesis Chapter 5.

plot_training_curves    → Fig 4.4 / 4.5  (accuracy + loss vs epochs)
plot_confusion_matrix   → Fig 5.17, 5.37, … (heatmap)
plot_per_class_metrics  → per-class precision / recall / F1 bar chart
plot_lr_curve           → learning rate schedule
"""
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe on headless servers
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from shared.config import SharedConfig as Config


# ── Shared style ─────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.size":   11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 120,
})


def _save(fig: plt.Figure, path: Optional[str]) -> None:
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(
    history:   Dict[str, list],
    save_path: Optional[str] = None,
) -> None:
    """
    Reproduce Fig 4.4 (accuracy) and Fig 4.5 (loss) from the thesis
    in a single side-by-side figure.

    Parameters
    ----------
    history : dict with keys train_loss, val_loss, train_acc, val_acc
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(epochs, history["train_acc"], label="Train", linewidth=1.8)
    axes[0].plot(epochs, history["val_acc"],   label="Validation", linewidth=1.8)
    axes[0].set_title("Training & Validation Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 100)

    # Loss
    axes[1].plot(epochs, history["train_loss"], label="Train", linewidth=1.8)
    axes[1].plot(epochs, history["val_loss"],   label="Validation", linewidth=1.8)
    axes[1].set_title("Training & Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("CustomViT — Training Progress", fontsize=14, y=1.01)
    fig.tight_layout()
    _save(fig, save_path)


def plot_confusion_matrix(
    cm:          np.ndarray,
    class_names: List[str]     = Config.CLASS_NAMES,
    normalize:   bool          = True,
    title:       str           = "Confusion Matrix",
    save_path:   Optional[str] = None,
) -> None:
    """
    Plot confusion matrix as a Seaborn heatmap.

    normalize=True  → row-normalised proportions (matches thesis figures)
    normalize=False → raw sample counts
    """
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True).astype(float)
        cm_plot  = np.divide(cm.astype(float), row_sums,
                             out=np.zeros_like(cm, dtype=float),
                             where=row_sums != 0)
        fmt, vmax = ".2f", 1.0
    else:
        cm_plot = cm
        fmt, vmax = "d", None

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm_plot,
        annot       = True,
        fmt         = fmt,
        xticklabels = class_names,
        yticklabels = class_names,
        cmap        = "Blues",
        vmin        = 0,
        vmax        = vmax,
        linewidths  = 0.4,
        linecolor   = "gray",
        ax          = ax,
    )
    ax.set_title(title, pad=14)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    fig.tight_layout()
    _save(fig, save_path)


def plot_per_class_metrics(
    metrics:   Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
) -> None:
    """
    Grouped bar chart: precision / recall / F1 per class.
    """
    classes   = list(metrics.keys())
    precision = [metrics[c]["precision"] for c in classes]
    recall    = [metrics[c]["recall"]    for c in classes]
    f1        = [metrics[c]["f1"]        for c in classes]

    x  = np.arange(len(classes))
    w  = 0.25
    fig, ax = plt.subplots(figsize=(14, 5))

    bars_p = ax.bar(x - w, precision, w, label="Precision", color="#4C72B0", edgecolor="white")
    bars_r = ax.bar(x,     recall,    w, label="Recall",    color="#DD8452", edgecolor="white")
    bars_f = ax.bar(x + w, f1,        w, label="F1-score",  color="#55A868", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Per-class Precision / Recall / F1")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    # Value labels on bars
    for bars in [bars_p, bars_r, bars_f]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.2f}",
                xy     = (bar.get_x() + bar.get_width() / 2, h),
                xytext = (0, 3),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=7,
            )

    fig.tight_layout()
    _save(fig, save_path)


def plot_lr_curve(
    history:   Dict[str, list],
    save_path: Optional[str] = None,
) -> None:
    """Plot the learning-rate schedule over epochs (log scale)."""
    epochs = range(1, len(history["lr"]) + 1)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.semilogy(epochs, history["lr"], linewidth=1.8, color="#4C72B0")
    ax.set_title("Learning Rate Schedule  (warmup → cosine)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, save_path)


def plot_dataset_comparison(
    results:   Dict[str, Dict],
    save_path: Optional[str] = None,
) -> None:
    """
    Horizontal bar chart comparing accuracy across all tested datasets
    (BDD100K, ACDC, CADC, Cityscapes, ONCE).
    """
    datasets  = list(results.keys())
    accuracy  = [results[d]["accuracy"] for d in datasets]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(datasets, accuracy, color="#4C72B0", edgecolor="white")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Model Accuracy Across Datasets")
    ax.bar_label(bars, fmt="%.2f%%", padding=4, fontsize=10)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    _save(fig, save_path)
