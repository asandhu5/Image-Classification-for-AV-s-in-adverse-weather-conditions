import logging, time
from typing import Dict, List
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from shared.config import SharedConfig as C

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, model: nn.Module, device: str = C.DEVICE):
        self.model  = model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, name: str = "Dataset") -> Dict:
        all_preds, all_labels = [], []
        t = 0.0
        for images, labels in tqdm(loader, desc=f"Eval [{name}]", ncols=100):
            images = images.to(self.device, non_blocking=True)
            t0     = time.perf_counter()
            out    = self.model(images)
            t     += time.perf_counter() - t0
            logits = out[0] if isinstance(out, tuple) else out
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_labels.extend(labels.tolist())
        p, l = np.array(all_preds), np.array(all_labels)
        acc  = 100.0 * (p == l).mean()
        logger.info(f"[{name}] acc={acc:.2f}% infer={t:.4f}s n={len(l):,}")
        return {"accuracy": acc, "all_preds": p, "all_labels": l,
                "inference_s": t, "n_samples": len(l)}
