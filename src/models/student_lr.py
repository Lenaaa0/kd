"""
LR (Logistic Regression) student model.
Lightweight deployable model using statistical features + CNN logits.

Fixed: now accepts any feature dimension (49 from rich_features, not hardcoded 20).
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class LRStudent:
    """
    LR-based student model with class weight support and flexible input dimension.

    Two variants:
    - pure:    Input = statistical features only
    - distill: Input = statistical features + CNN logits (normalized)

    This is a wrapper around sklearn LogisticRegression that mimics
    a PyTorch-like interface for compatibility with the evaluation pipeline.
    """

    def __init__(
        self,
        num_classes: int,
        mode: str = "pure",
        max_iter: int = 2000,
        C: float = 1.0,
        class_weight: str | dict | None = None,
        solver: str = "lbfgs",
    ):
        """
        Args:
            num_classes: Number of output classes
            mode: "pure" (stats only) or "distill" (stats + CNN logits)
            max_iter: Maximum iterations for sklearn LR
            C: Inverse regularization strength
            class_weight: 'balanced', 'balanced_subsample', dict, or None
            solver: 'lbfgs', 'saga', 'newton-cg', 'sag', 'saga'
        """
        if mode not in {"pure", "distill"}:
            raise ValueError(f"mode must be 'pure' or 'distill', got {mode}")
        self.num_classes = num_classes
        self.mode = mode
        self.max_iter = max_iter
        self.C = C
        self.class_weight = class_weight
        self._input_dim = None

        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            max_iter=max_iter,
            C=C,
            solver=solver,
            multi_class="multinomial",
            class_weight=class_weight,
        )

    @property
    def input_dim(self) -> int:
        """Actual feature dimension after fitting or from config."""
        return self._input_dim or 0

    def fit(
        self,
        X_stats: np.ndarray,
        y: np.ndarray,
        cnn_logits: np.ndarray | None = None,
    ) -> "LRStudent":
        """
        Train the LR student.

        Args:
            X_stats: Statistical features, shape (n, n_features)
            y: Labels, shape (n,)
            cnn_logits: CNN logits, shape (n, num_classes). Required if mode="distill"
        """
        X_aug = self._augment(X_stats, cnn_logits)
        self._input_dim = X_aug.shape[1]
        X_scaled = self.scaler.fit_transform(X_aug)
        self.model.fit(X_scaled, y)
        return self

    def predict(self, X: np.ndarray, cnn_logits: np.ndarray | None = None) -> np.ndarray:
        X_aug = self._augment(X, cnn_logits)
        X_scaled = self.scaler.transform(X_aug)
        return self.model.predict(X_scaled)

    def predict_proba(
        self, X: np.ndarray, cnn_logits: np.ndarray | None = None
    ) -> np.ndarray:
        X_aug = self._augment(X, cnn_logits)
        X_scaled = self.scaler.transform(X_aug)
        return self.model.predict_proba(X_scaled)

    def score(self, X: np.ndarray, y: np.ndarray,
              cnn_logits: np.ndarray | None = None) -> float:
        """Return accuracy score."""
        pred = self.predict(X, cnn_logits)
        return float(np.mean(pred == y))

    def _augment(
        self, X_stats: np.ndarray, cnn_logits: np.ndarray | None
    ) -> np.ndarray:
        if self.mode == "distill":
            if cnn_logits is None:
                raise ValueError(
                    f"mode='distill' requires cnn_logits but got None"
                )
            logits_scaled = cnn_logits / (cnn_logits.max(axis=1, keepdims=True) + 1e-9)
            return np.hstack([X_stats, logits_scaled])
        return X_stats

    def get_config(self) -> dict:
        return {
            "model_type": "lr_student",
            "mode": self.mode,
            "num_classes": self.num_classes,
            "input_dim": self.input_dim,
            "max_iter": self.max_iter,
            "C": self.C,
            "class_weight": self.class_weight,
        }
