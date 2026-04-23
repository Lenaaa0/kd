"""
Alternative teachers: Random Forest and Extra Trees for encrypted traffic classification.

These tree-based models are used as "oracle" teachers that achieve >90% accuracy
on rich statistical features, providing better soft labels for knowledge distillation.
They also serve as strong standalone baselines.

Both models operate on the same rich feature space as the DNN teacher.
"""
from __future__ import annotations

import joblib
import numpy as np
from pathlib import Path
from typing import List, Optional

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


class TreeTeacher:
    """
    Base class for tree-based teachers.
    Provides soft-label generation for distillation.
    """

    def __init__(self, model):
        self.model = model

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def soft_labels(self, X: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """
        Generate soft labels (probability distributions) for distillation.
        Optionally apply temperature scaling.
        """
        proba = self.predict_proba(X)
        if temperature != 1.0:
            log_proba = np.log(proba + 1e-10)
            exp_proba = np.exp(log_proba / temperature)
            proba = exp_proba / exp_proba.sum(axis=1, keepdims=True)
        return proba

    def get_config(self) -> dict:
        return {
            "model_type": self.model.__class__.__name__,
            "n_classes": len(self.model.classes_),
        }


class RFTeacher(TreeTeacher):
    """
    Random Forest teacher.
    Known to achieve ~91% accuracy on ISCX VPN-nonVPN flow features.

    Strengths:
    - Handles high-dimensional feature spaces well
    - Robust to feature scaling
    - Provides calibrated probability estimates
    - Fast inference
    """

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int | None = None,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        max_features: str = "sqrt",
        n_jobs: int = -1,
        random_state: int = 42,
        class_weight: str | dict | None = "balanced",
    ):
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            n_jobs=n_jobs,
            random_state=random_state,
            class_weight=class_weight,
        )
        super().__init__(model)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RFTeacher":
        self.model.fit(X, y)
        return self

    def save(self, path: Path | str) -> None:
        joblib.dump({
            "model": self.model,
            "model_type": self.model.__class__.__name__,
        }, path)

    @classmethod
    def load(cls, path: Path | str) -> "RFTeacher":
        data = joblib.load(path)
        inst = object.__new__(cls)
        inst.model = data["model"]
        return inst


class ExtraTreesTeacher(TreeTeacher):
    """
    Extra Trees (Extremely Randomized Trees) teacher.
    Often outperforms RF on noisy datasets due to extra randomization.

    Strengths:
    - Faster than RF (no bootstrapping)
    - Less overfitting on noisy data
    - Better generalization in some cases
    """

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int | None = None,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        max_features: str = "sqrt",
        n_jobs: int = -1,
        random_state: int = 42,
        class_weight: str | dict | None = "balanced",
    ):
        model = ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            n_jobs=n_jobs,
            random_state=random_state,
            class_weight=class_weight,
        )
        super().__init__(model)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ExtraTreesTeacher":
        self.model.fit(X, y)
        return self

    def save(self, path: Path | str) -> None:
        joblib.dump({
            "model": self.model,
            "model_type": self.model.__class__.__name__,
        }, path)

    @classmethod
    def load(cls, path: Path | str) -> "ExtraTreesTeacher":
        data = joblib.load(path)
        inst = object.__new__(cls)
        inst.model = data["model"]
        return inst


def evaluate_tree_teacher(
    teacher,
    X_te: np.ndarray,
    y_te: np.ndarray,
    label_names: List[str],
) -> dict:
    """
    Evaluate a tree-based teacher on test data.
    Returns accuracy, macro F1, per-class metrics, and confusion matrix.
    """
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        accuracy_score,
    )

    pred = teacher.predict(X_te)
    acc = float(accuracy_score(y_te, pred))
    macro_f1 = float(f1_score(y_te, pred, average="macro", zero_division=0))
    macro_precision = float(precision_score(y_te, pred, average="macro", zero_division=0))
    macro_recall = float(recall_score(y_te, pred, average="macro", zero_division=0))

    cm = confusion_matrix(
        y_te, pred, labels=list(range(len(label_names)))
    ).tolist()

    report = classification_report(
        y_te, pred,
        labels=list(range(len(label_names))),
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )

    per_class = {
        label_names[i]: {
            "precision": float(report[label_names[i]]["precision"]),
            "recall": float(report[label_names[i]]["recall"]),
            "f1-score": float(report[label_names[i]]["f1-score"]),
            "support": int(report[label_names[i]]["support"]),
        }
        for i in range(len(label_names))
    }

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "confusion_matrix": cm,
        "per_class": per_class,
        "label_names": label_names,
    }
