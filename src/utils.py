from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(p: str | Path) -> Path:
    pp = Path(p)
    pp.mkdir(parents=True, exist_ok=True)
    return pp


def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def metrics_from_predictions(y_true, y_pred, labels: List[str]) -> Dict[str, Any]:
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
        "classification_report": report,
        "labels": labels,
    }

