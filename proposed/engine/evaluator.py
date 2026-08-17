"""
engine/evaluator.py
────────────────────
Evaluation engine for test-set and cross-dataset evaluation.

Used for:
    • BDD100K test split (thesis Table 5.3 / 5.4)
    • ACDC, CADC, Cityscapes, ONCE cross-dataset tests (thesis Sections 5.2.2–5.2.6)

Returns raw predictions + labels so that metrics and plots can be
computed externally by utils/metrics.py and utils/visualization.py.
"""
import logging
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Wraps a trained model and evaluates it over any DataLoader.

    Parameters
    ----------
    model  : CustomViT (or any nn.Module with matching output shape)
    device : torch device string
    """

    def __init__(self, model: nn.Module, device: str = Config.DEVICE) -> None:
        self.model  = model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def evaluate(
        self,
        loader:       DataLoader,
        dataset_name: str = "Dataset",
    ) -> Dict:
        """
        Run full evaluation over a DataLoader.

        Returns
        -------
        dict with:
            "accuracy"    : float (%)
            "all_preds"   : np.ndarray of shape (N,)
            "all_labels"  : np.ndarray of shape (N,)
            "inference_s" : total inference time in seconds (GPU only)
            "n_samples"   : total number of evaluated samples
        """
        all_preds:  List[int] = []
        all_labels: List[int] = []
        inference_s = 0.0

        for images, labels in tqdm(loader, desc=f"Eval [{dataset_name}]", ncols=100):
            images = images.to(self.device, non_blocking=True)

            t0     = time.perf_counter()
            logits = self.model(images)
            inference_s += time.perf_counter() - t0

            preds = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

        preds_np  = np.array(all_preds)
        labels_np = np.array(all_labels)
        accuracy  = 100.0 * (preds_np == labels_np).mean()

        logger.info(
            f"[{dataset_name}]  "
            f"Accuracy={accuracy:.2f}%  "
            f"Inference={inference_s:.4f}s  "
            f"Samples={len(all_labels):,}"
        )

        return {
            "accuracy":    accuracy,
            "all_preds":   preds_np,
            "all_labels":  labels_np,
            "inference_s": inference_s,
            "n_samples":   len(all_labels),
        }

    def evaluate_all_datasets(
        self,
        loaders: Dict[str, DataLoader],
    ) -> Dict[str, Dict]:
        """
        Evaluate across multiple datasets in one call.

        Parameters
        ----------
        loaders : {"BDD100K": loader1, "ACDC": loader2, ...}

        Returns
        -------
        {"BDD100K": {...results...}, "ACDC": {...results...}, ...}
        """
        results = {}
        for name, loader in loaders.items():
            results[name] = self.evaluate(loader, name)
        return results
