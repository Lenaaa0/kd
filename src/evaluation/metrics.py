"""
Metrics computation for model evaluation.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str] | None = None,
) -> Dict[str, Any]:
    """
    Compute a comprehensive set of classification metrics.

    Args:
        y_true: Ground truth labels (integers)
        y_pred: Predicted labels (integers)
        label_names: Optional class names for reporting

    Returns:
        Dict with accuracy, macro_f1, precision, recall, confusion matrix, per-class report
    """
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    macro_precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    if label_names is None:
        labels = sorted(set(y_true) | set(y_pred))
        label_names = [str(l) for l in labels]

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names)))).tolist()

    report_dict = classification_report(
        y_true, y_pred,
        labels=list(range(len(label_names))),
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "confusion_matrix": cm,
        "per_class": {
            label_names[i]: {
                "precision": float(report_dict[label_names[i]]["precision"]),
                "recall": float(report_dict[label_names[i]]["recall"]),
                "f1-score": float(report_dict[label_names[i]]["f1-score"]),
                "support": int(report_dict[label_names[i]]["support"]),
            }
            for i in range(len(label_names))
        },
        "label_names": label_names,
    }
