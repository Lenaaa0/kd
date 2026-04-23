"""
CNN student trainer: supports both plain training and knowledge distillation.

Distillation uses precomputed soft labels (computed once before training)
to avoid expensive per-batch feature extraction.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ..models import CNNStudent, TransformerTeacher
from ..utils import set_seed
from .base_trainer import BaseTrainer


class TrainerCNN(BaseTrainer):
    """
    Train CNN student either plain or with precomputed knowledge distillation.

    Modes:
    - plain:  Standard CE loss on ground-truth labels
    - distill: Combined KD loss (soft KL + hard CE) using precomputed soft labels
    """

    def __init__(
        self,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_te: np.ndarray,
        y_te: np.ndarray,
        run_dir: str | Path,
        num_classes: int,
        mode: str = "distill",
        teacher_soft_labels: Optional[torch.Tensor] = None,
        distill_T: float = 2.0,
        distill_alpha: float = 0.3,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 80,
        patience: int = 15,
        seed: int = 42,
        device: Optional[str] = None,
    ):
        if mode not in {"plain", "distill"}:
            raise ValueError(f"mode must be 'plain' or 'distill', got {mode}")

        super().__init__(run_dir=run_dir, device=device, seed=seed)
        self.mode = mode
        self.num_classes = num_classes
        self.X_tr = X_tr
        self.y_tr = y_tr
        self.X_te = X_te
        self.y_te = y_te
        self.teacher_soft_labels = teacher_soft_labels
        self.distill_T = distill_T
        self.distill_alpha = distill_alpha
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.patience = patience

        if teacher_soft_labels is not None:
            self.teacher_soft_labels = teacher_soft_labels.to(device)

        self.model = CNNStudent(
            seq_len=X_tr.shape[1],
            n_features=X_tr.shape[2],
            num_classes=num_classes,
        )
        self._save_name = "cnn_baseline.pt" if mode == "plain" else "cnn_student.pt"

        self.opt = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=epochs
        )

        self.train_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_tr).float(),
                torch.from_numpy(y_tr).long(),
                torch.arange(len(y_tr)).long(),
            ),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=(self.device == "cuda"),
        )

        self.X_te_t = torch.from_numpy(X_te).float().to(self.device)
        self.y_te_np = y_te

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        n_samples = 0

        for xb, yb, indices in self.train_loader:
            xb = xb.to(self.device)
            yb = yb.to(self.device)

            self.opt.zero_grad()

            if self.mode == "distill" and self.teacher_soft_labels is not None:
                # teacher_soft_labels stores raw log-probabilities from RF/ET
                # Use T=1 to avoid numerical instability from extreme log-prob values
                logits = self.model(xb)
                p_teacher = F.softmax(
                    self.teacher_soft_labels[indices.to(self.teacher_soft_labels.device)],
                    dim=1,
                )
                q_student = F.log_softmax(logits, dim=1)
                # reduction='none' then mean avoids NaN from batchmean overflow
                soft_loss = F.kl_div(q_student, p_teacher, reduction="none").sum(dim=1).mean()
                hard_loss = F.cross_entropy(logits, yb)
                loss = self.distill_alpha * soft_loss + (1 - self.distill_alpha) * hard_loss
            else:
                logits = self.model(xb)
                loss = F.cross_entropy(logits, yb)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()

            total_loss += float(loss.item()) * xb.size(0)
            n_samples += xb.size(0)

        self.scheduler.step()
        return total_loss / max(n_samples, 1)

    def _eval(self) -> float:
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.X_te_t).cpu().numpy()
        preds = logits.argmax(axis=1)
        return float(accuracy_score(self.y_te_np, preds))

    def _get_model(self) -> nn.Module:
        return self.model

    def save(self) -> Path:
        """Save model checkpoint and training summary."""
        path = self.artifacts_dir / self._save_name
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "best_val_acc": self.best_acc,
                "history": self.history,
                "model_config": self.model.get_config(),
                "train_config": {
                    "mode": self.mode,
                    "distill_T": self.distill_T,
                    "distill_alpha": self.distill_alpha,
                    "batch_size": self.batch_size,
                    "lr": self.lr,
                    "weight_decay": self.weight_decay,
                },
            },
            path,
        )
        return path


def quick_train_cnn(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    num_classes: int,
    run_dir: str | Path,
    mode: str = "distill",
    teacher_state_dict: Optional[dict] = None,
    distill_T: float = 2.0,
    distill_alpha: float = 0.3,
    epochs: int = 80,
    patience: int = 15,
    seed: int = 42,
    device: Optional[str] = None,
) -> Tuple[CNNStudent, dict]:
    set_seed(seed)

    teacher = None
    if mode == "distill" and teacher_state_dict is not None:
        teacher = TransformerTeacher(
            seq_len=teacher_state_dict.get("seq_len", X_tr.shape[1]),
            n_features=teacher_state_dict.get("n_features", X_tr.shape[2]),
            num_classes=teacher_state_dict["num_classes"],
        )
        teacher.load_state_dict(teacher_state_dict)
        teacher.eval()

    trainer = TrainerCNN(
        X_tr=X_tr, y_tr=y_tr, X_te=X_te, y_te=y_te,
        run_dir=run_dir, num_classes=num_classes,
        mode=mode, teacher_model=teacher,
        distill_T=distill_T, distill_alpha=distill_alpha,
        epochs=epochs, patience=patience, seed=seed, device=device,
    )

    summary = trainer.train(epochs=epochs, patience=patience)
    trainer.save()
    return trainer.model, summary
