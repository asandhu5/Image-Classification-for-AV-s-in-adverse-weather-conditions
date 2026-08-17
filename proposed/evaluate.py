"""
evaluate.py
───────────
Evaluate the trained model on BDD100K and / or cross-datasets.

Usage
─────
    # BDD100K test split only
    python evaluate.py

    # All five datasets (requires data prepared by scripts/prepare_*.py)
    python evaluate.py --all-datasets

    # Specific checkpoint
    python evaluate.py --ckpt checkpoints/epoch_100.pth

Cross-dataset support
─────────────────────
For ACDC / CADC / Cityscapes / ONCE:
    1. Download each dataset.
    2. Run the matching prepare script:
           python scripts/prepare_acdc.py
           python scripts/prepare_cadc.py
           python scripts/prepare_cityscapes.py
           python scripts/prepare_once.py
    3. Then run:  python evaluate.py --all-datasets
"""
import argparse
import json
import logging
from pathlib import Path

from config import Config
from dataset import get_dataloaders, get_val_transforms
from dataset.bdd100k import CrossDatasetROI
from engine.evaluator import Evaluator
from models.vit import CustomViT
from utils.checkpoint import load_best_checkpoint, load_checkpoint
from utils.metrics import (
    compute_confusion_matrix,
    compute_classification_report,
    compute_per_class_metrics,
    print_results_table,
)
from utils.visualization import (
    plot_confusion_matrix,
    plot_per_class_metrics,
    plot_dataset_comparison,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",          default=str(Config.CHECKPOINT_DIR / "best.pth"))
    p.add_argument("--all-datasets",  action="store_true")
    p.add_argument("--output-dir",    default=str(Config.OUTPUT_DIR / "eval"))
    p.add_argument("--data-root",     default=str(Config.DATA_ROOT))
    return p.parse_args()


def evaluate_one(
    evaluator:  Evaluator,
    loader,
    name:       str,
    output_dir: Path,
) -> dict:
    """Run evaluation, produce plots, and return result dict."""
    results = evaluator.evaluate(loader, name)
    preds   = results["all_preds"]
    labels  = results["all_labels"]

    cm      = compute_confusion_matrix(preds, labels)
    report  = compute_classification_report(preds, labels)
    pcm     = compute_per_class_metrics(preds, labels)

    logging.getLogger("evaluate").info(f"\n{report}")

    slug = name.lower().replace(" ", "_")
    plot_confusion_matrix(
        cm, title=f"Confusion Matrix — {name}",
        save_path=str(output_dir / f"cm_{slug}.png"),
    )
    plot_per_class_metrics(
        pcm,
        save_path=str(output_dir / f"pcm_{slug}.png"),
    )
    return {
        "accuracy":    results["accuracy"],
        "inference_s": results["inference_s"],
        "n_samples":   results["n_samples"],
        "per_class":   pcm,
    }


def load_cross_dataset(name: str, transform) -> "DataLoader | None":
    """
    Load a cross-dataset ROI JSON prepared by scripts/prepare_*.py.

    Expected file: data/<name_lower>/rois.json
    """
    from torch.utils.data import DataLoader

    roi_json = Path("data") / name.lower() / "rois.json"
    if not roi_json.exists():
        logging.getLogger("evaluate").warning(
            f"{name}: {roi_json} not found — skipping.  "
            f"Run scripts/prepare_{name.lower()}.py first."
        )
        return None

    with open(roi_json) as f:
        samples = json.load(f)

    ds = CrossDatasetROI(samples, transform)
    return DataLoader(
        ds,
        batch_size  = Config.BATCH_SIZE,
        shuffle     = False,
        num_workers = Config.NUM_WORKERS,
        pin_memory  = Config.PIN_MEMORY,
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s [%(levelname)s] %(message)s",
    )
    Config.make_dirs()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────
    model = CustomViT()
    load_checkpoint(args.ckpt, model, device=Config.DEVICE)
    evaluator = Evaluator(model, Config.DEVICE)

    transform = get_val_transforms()
    all_results: dict = {}

    # ── BDD100K test split ────────────────────────────────────────────
    data_root     = Path(args.data_root)
    train_json    = str(data_root / "labels" / "det_20" / "det_train.json")
    train_img_dir = str(data_root / "images" / "100k" / "train")

    _, _, test_loader = get_dataloaders(
        train_json, train_img_dir,
        transform,   # val_transform used for both (no augmentation)
        transform,
    )
    all_results["BDD100K"] = evaluate_one(evaluator, test_loader, "BDD100K", output_dir)

    # ── Cross-datasets ────────────────────────────────────────────────
    if args.all_datasets:
        for ds_name in ["ACDC", "CADC", "Cityscapes", "ONCE"]:
            loader = load_cross_dataset(ds_name, transform)
            if loader is not None:
                all_results[ds_name] = evaluate_one(
                    evaluator, loader, ds_name, output_dir
                )

    # ── Summary ───────────────────────────────────────────────────────
    print_results_table(all_results)

    if len(all_results) > 1:
        plot_dataset_comparison(
            {k: v for k, v in all_results.items()},
            save_path=str(output_dir / "dataset_comparison.png"),
        )

    summary = {
        k: {"accuracy": v["accuracy"], "inference_s": v["inference_s"]}
        for k, v in all_results.items()
    }
    with open(output_dir / "eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logging.getLogger("evaluate").info(
        f"Summary saved → {output_dir / 'eval_summary.json'}"
    )


if __name__ == "__main__":
    main()
