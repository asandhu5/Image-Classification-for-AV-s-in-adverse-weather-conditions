"""
shared/dataset/bdd100k.py
──────────────────────────
ROI-level BDD100K dataset identical to thesis-voc, but img_size is
injected via the transform rather than baked into the class, so every
model in the benchmark can use its own native resolution.
"""
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from shared.config import SharedConfig as C


@dataclass(slots=True)
class ROISample:
    img_path:  str
    x1: float; y1: float; x2: float; y2: float
    label_idx: int


class BDD100KDataset(Dataset):

    def __init__(self, samples: List[ROISample], transform=None):
        self.samples   = samples
        self.transform = transform

    @classmethod
    def build_samples(cls, json_path: str, img_dir: str) -> List[ROISample]:
        out = []
        img_dir = Path(img_dir)
        with open(json_path) as f:
            data = json.load(f)
        for item in data:
            img_path = img_dir / item.get("name", "")
            if not img_path.exists():
                continue
            for lbl in item.get("labels", []):
                cat = lbl.get("category", "").lower()
                if cat not in C.CLASS2IDX:
                    continue
                box = lbl.get("box2d")
                if box is None:
                    continue
                x1, y1 = float(box["x1"]), float(box["y1"])
                x2, y2 = float(box["x2"]), float(box["y2"])
                w, h = x2 - x1, y2 - y1
                if w <= 0 or h <= 0 or w * h < C.MIN_BOX_AREA:
                    continue
                if max(w, h) / (min(w, h) + 1e-6) > C.MAX_ASPECT:
                    continue
                out.append(ROISample(str(img_path), x1, y1, x2, y2, C.CLASS2IDX[cat]))
        return out

    @classmethod
    def from_json(cls, json_path: str, img_dir: str, split: str = "train",
                  transform=None, seed: int = C.SEED) -> "BDD100KDataset":
        all_s = cls.build_samples(json_path, img_dir)
        rng   = random.Random(seed)
        rng.shuffle(all_s)
        n       = len(all_s)
        n_train = int(n * C.TRAIN_SPLIT)
        n_val   = int(n * C.VAL_SPLIT)
        if split == "train":
            subset = all_s[:n_train]
        elif split == "val":
            subset = all_s[n_train: n_train + n_val]
        else:
            subset = all_s[n_train + n_val:]
        return cls(subset, transform)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        s = self.samples[idx]
        try:
            img = Image.open(s.img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224))
        w, h   = img.size
        pad_x  = (s.x2 - s.x1) * C.BOX_PADDING
        pad_y  = (s.y2 - s.y1) * C.BOX_PADDING
        x1 = max(0.0, s.x1 - pad_x); y1 = max(0.0, s.y1 - pad_y)
        x2 = min(w,   s.x2 + pad_x); y2 = min(h,   s.y2 + pad_y)
        if x2 > x1 and y2 > y1:
            img = img.crop((int(x1), int(y1), int(x2), int(y2)))
        if self.transform:
            img = self.transform(img)
        return img, s.label_idx

    def class_weights(self) -> torch.Tensor:
        counts = torch.zeros(C.NUM_CLASSES)
        for s in self.samples:
            counts[s.label_idx] += 1
        counts = counts.clamp(min=1)
        w = 1.0 / counts
        return w / w.sum() * C.NUM_CLASSES


def get_dataloaders(train_json: str, img_dir: str,
                    train_transform, val_transform,
                    batch_size: int = C.BATCH_SIZE):
    tr = BDD100KDataset.from_json(train_json, img_dir, "train", train_transform)
    va = BDD100KDataset.from_json(train_json, img_dir, "val",   val_transform)
    te = BDD100KDataset.from_json(train_json, img_dir, "test",  val_transform)

    def loader(ds, shuffle):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=C.NUM_WORKERS, pin_memory=C.PIN_MEMORY,
                          drop_last=shuffle,
                          persistent_workers=C.NUM_WORKERS > 0)

    return loader(tr, True), loader(va, False), loader(te, False)
