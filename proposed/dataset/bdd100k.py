"""
dataset/bdd100k.py
──────────────────
Builds a ROI-level PyTorch Dataset from BDD100K detection annotations.

BDD100K JSON structure (one entry per image):
{
  "name": "0000f77c-6257be58.jpg",
  "labels": [
    {
      "category": "car",
      "box2d": {"x1": 528.0, "y1": 375.0, "x2": 700.0, "y2": 432.0}
    }, ...
  ]
}

Each bounding box → one ROISample (image path + box coords + class index).
The dataset returns 224×224 PIL crops, transformed on the fly.

Split logic (thesis: 70 / 10 / 20):
    The 70K BDD100K training images are shuffled once with a fixed seed,
    then split at the image level to avoid data leakage between splits.
"""
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from config import Config


# ──────────────────────────────────────────────────────────────────────────────
# Data container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class ROISample:
    """Lightweight record for a single bounding-box crop."""
    img_path:  str
    x1:        float
    y1:        float
    x2:        float
    y2:        float
    label_idx: int


# ──────────────────────────────────────────────────────────────────────────────
# BDD100K dataset
# ──────────────────────────────────────────────────────────────────────────────

class BDD100KDataset(Dataset):
    """
    ROI-level dataset built from BDD100K detection annotations.

    Each __getitem__ returns:
        (image_tensor [3, 224, 224],  label_idx [int])
    """

    def __init__(
        self,
        samples:   List[ROISample],
        transform=None,
    ) -> None:
        self.samples   = samples
        self.transform = transform

    # ── Class-method builders ─────────────────────────────────────────

    @classmethod
    def build_samples(cls, json_path: str, img_dir: str) -> List[ROISample]:
        """
        Parse one BDD100K detection JSON file and return all valid ROISamples.

        Filtering:
            • category must be in Config.BDD_CLASSES
            • box area  ≥ Config.MIN_BOX_AREA
            • aspect ratio ≤ Config.MAX_ASPECT
            • image file must exist on disk
        """
        samples: List[ROISample] = []
        img_dir = Path(img_dir)

        with open(json_path, "r") as fh:
            data = json.load(fh)

        for item in data:
            img_path = img_dir / item.get("name", "")
            if not img_path.exists():
                continue

            for lbl in item.get("labels", []):
                cat = lbl.get("category", "").lower()
                if cat not in Config.CLASS2IDX:
                    continue
                box = lbl.get("box2d")
                if box is None:
                    continue

                x1 = float(box["x1"])
                y1 = float(box["y1"])
                x2 = float(box["x2"])
                y2 = float(box["y2"])

                w = x2 - x1
                h = y2 - y1
                if w <= 0 or h <= 0:
                    continue
                if w * h < Config.MIN_BOX_AREA:
                    continue
                if max(w, h) / (min(w, h) + 1e-6) > Config.MAX_ASPECT:
                    continue

                samples.append(ROISample(
                    img_path  = str(img_path),
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    label_idx = Config.CLASS2IDX[cat],
                ))

        return samples

    @classmethod
    def from_json(
        cls,
        json_path:  str,
        img_dir:    str,
        split:      str = "train",   # "train" | "val" | "test"
        transform=None,
        seed:       int = Config.SEED,
    ) -> "BDD100KDataset":
        """
        Build one split from a BDD100K JSON.

        Shuffles at the ROI level with a fixed seed so splits are
        deterministic and reproducible across runs.
        """
        all_samples = cls.build_samples(json_path, img_dir)
        rng = random.Random(seed)
        rng.shuffle(all_samples)

        n       = len(all_samples)
        n_train = int(n * Config.TRAIN_SPLIT)
        n_val   = int(n * Config.VAL_SPLIT)

        if split == "train":
            subset = all_samples[:n_train]
        elif split == "val":
            subset = all_samples[n_train: n_train + n_val]
        else:
            subset = all_samples[n_train + n_val:]

        return cls(subset, transform)

    # ── Dataset interface ─────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        s = self.samples[idx]

        # Load image
        try:
            img = Image.open(s.img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (Config.IMG_SIZE, Config.IMG_SIZE), color=(0, 0, 0))

        # Expand box slightly to add context (helps ViT attend to object edges)
        w_img, h_img = img.size
        box_w  = s.x2 - s.x1
        box_h  = s.y2 - s.y1
        pad_x  = box_w * Config.BOX_PADDING
        pad_y  = box_h * Config.BOX_PADDING
        x1 = max(0.0,    s.x1 - pad_x)
        y1 = max(0.0,    s.y1 - pad_y)
        x2 = min(w_img,  s.x2 + pad_x)
        y2 = min(h_img,  s.y2 + pad_y)

        if x2 > x1 and y2 > y1:
            img = img.crop((int(x1), int(y1), int(x2), int(y2)))
        # If crop is degenerate, fall through to full-image resize

        if self.transform is not None:
            img = self.transform(img)

        return img, s.label_idx

    # ── Utility ───────────────────────────────────────────────────────

    def class_distribution(self) -> Dict[str, int]:
        """Return {class_name: count} for this split — useful for weight calc."""
        counts: Dict[str, int] = {n: 0 for n in Config.CLASS_NAMES}
        for s in self.samples:
            counts[Config.IDX2CLASS[s.label_idx]] += 1
        return counts

    def class_weights(self) -> torch.Tensor:
        """
        Inverse-frequency class weights for WeightedRandomSampler or
        weighted CrossEntropyLoss.  Shape: (NUM_CLASSES,).
        """
        dist    = self.class_distribution()
        counts  = torch.tensor(
            [dist[Config.IDX2CLASS[i]] for i in range(Config.NUM_CLASSES)],
            dtype=torch.float,
        )
        counts  = counts.clamp(min=1)
        weights = 1.0 / counts
        return weights / weights.sum() * Config.NUM_CLASSES   # normalise


# ──────────────────────────────────────────────────────────────────────────────
# Generic cross-dataset loader
# ──────────────────────────────────────────────────────────────────────────────

class CrossDatasetROI(Dataset):
    """
    Generic ROI dataset for cross-dataset evaluation:
        ACDC | CADC | Cityscapes | ONCE

    Expects a list of dicts (typically loaded from rois.json generated by
    the scripts/prepare_<dataset>.py scripts):

        [
          {"img_path": str, "x1": float, "y1": float,
           "x2": float,     "y2": float, "label_idx": int},
          ...
        ]
    """

    def __init__(
        self,
        samples:   List[Dict],
        transform=None,
    ) -> None:
        self.samples   = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        s = self.samples[idx]

        try:
            img = Image.open(s["img_path"]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (Config.IMG_SIZE, Config.IMG_SIZE), color=(0, 0, 0))

        w, h = img.size
        x1   = max(0, int(s["x1"]))
        y1   = max(0, int(s["y1"]))
        x2   = min(w, int(s["x2"]))
        y2   = min(h, int(s["y2"]))
        if x2 > x1 and y2 > y1:
            img = img.crop((x1, y1, x2, y2))

        if self.transform is not None:
            img = self.transform(img)

        return img, int(s["label_idx"])


# ──────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ──────────────────────────────────────────────────────────────────────────────

def get_dataloaders(
    train_json:      str,
    train_img_dir:   str,
    train_transform,
    val_transform,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build and return (train_loader, val_loader, test_loader) for BDD100K.

    The 70K BDD100K training annotations are split 70 / 10 / 20 at
    the ROI level.  Both val and test use val_transform (no augmentation).
    """
    train_ds = BDD100KDataset.from_json(
        train_json, train_img_dir, "train", train_transform
    )
    val_ds = BDD100KDataset.from_json(
        train_json, train_img_dir, "val", val_transform
    )
    test_ds = BDD100KDataset.from_json(
        train_json, train_img_dir, "test", val_transform
    )

    def _loader(ds: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size  = Config.BATCH_SIZE,
            shuffle     = shuffle,
            num_workers = Config.NUM_WORKERS,
            pin_memory  = Config.PIN_MEMORY,
            drop_last   = shuffle,
            persistent_workers = Config.NUM_WORKERS > 0,
        )

    return _loader(train_ds, True), _loader(val_ds, False), _loader(test_ds, False)
