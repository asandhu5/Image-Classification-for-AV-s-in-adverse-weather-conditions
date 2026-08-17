"""
utils/metrics.py
────────────────
All classification metrics used in thesis Chapter 5.

Functions
─────────
compute_accuracy              → overall accuracy (%)
compute_confusion_matrix      → sklearn confusion matrix (raw or normalised)
compute_classification_report → sklearn classification report (str or dict)
compute_per_class_metrics     → dict of per-class precision / recall / F1
"""
from typing import Dict, Optional, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from config import Config


def compute_accuracy(
    all_preds:  np.ndarray,
    all_labels: np.ndarray,
) -> float:
    """Overall accuracy as a percentage."""
    return 100.0 * accuracy_score(all_labels, all_preds)


def compute_confusion_matrix(
    all_preds:  np.ndarray,
    all_labels: np.ndarray,
    normalize:  Optional[str] = None,   # None | "true" | "pred" | "all"
) -> np.ndarray:
    """
    Compute confusion matrix.

    Parameters
    ----------
    normalize : None     → raw counts
                "true"   → normalised by true class (row-wise, as in thesis)
                "pred"   → normalised by predicted class
                "all"    → normalised by total samples
    """
    return confusion_matrix(
        all_labels,
        all_preds,
        labels    = list(range(Config.NUM_CLASSES)),
        normalize = normalize,
    )


def compute_classification_report(
    all_preds:   np.ndarray,
    all_labels:  np.ndarray,
    output_dict: bool = False,
) -> Union[str, Dict]:
    """
    sklearn classification report: per-class precision, recall, F1, support.

    Parameters
    ----------
    output_dict : if True, returns a dict; otherwise a formatted string.

    Note: labels=list(range(NUM_CLASSES)) is always passed so the report
    shows all 10 classes even when some are absent from the evaluated split.
    """
    return classification_report(
        all_labels,
        all_preds,
        labels       = list(range(Config.NUM_CLASSES)),
        target_names = Config.CLASS_NAMES,
        digits       = 4,
        output_dict  = output_dict,
        zero_division = 0,
    )


def compute_per_class_metrics(
    all_preds:  np.ndarray,
    all_labels: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """
    Per-class precision, recall, F1, and support.

    Returns
    -------
    {
        "Car":    {"precision": 0.92, "recall": 0.91, "f1": 0.91, "support": 1234},
        "Person": {...},
        ...
    }
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels,
        all_preds,
        labels        = list(range(Config.NUM_CLASSES)),
        zero_division = 0,
    )
    return {
        Config.CLASS_NAMES[i]: {
            "precision": float(precision[i]),
            "recall":    float(recall[i]),
            "f1":        float(f1[i]),
            "support":   int(support[i]),
        }
        for i in range(Config.NUM_CLASSES)
    }


def print_results_table(results: Dict[str, Dict]) -> None:
    """
    Pretty-print a multi-dataset comparison table.

    Parameters
    ----------
    results : {"BDD100K": {"accuracy": 91.47, "inference_s": 0.50}, ...}
    """
    print("\n" + "=" * 52)
    print(f"{'Dataset':<20} {'Accuracy (%)':>14} {'Infer (s)':>12}")
    print("-" * 52)
    for ds, r in results.items():
        print(f"{ds:<20} {r['accuracy']:>13.2f}% {r.get('inference_s', 0):>12.4f}")
    print("=" * 52 + "\n")
