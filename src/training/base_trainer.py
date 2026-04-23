"""
Base trainer with early stopping and checkpoint management.
All training scripts inherit from this to avoid duplicated logic.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset


class BaseTrainer:
    """
    Common training utilities: device setup, early stopping, checkpointing, logging.
    Subclasses must implement `_train_epoch` and `_eval`.
    """

    def __init__(
        self,
        run_dir: str | Path,
        device: Optional[str] = None,
        seed: int = 42,
    ):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir = self.run_dir / "artifacts"
        self.artifacts_dir.mkdir(exist_ok=True)

        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.seed = seed
        self._set_seed(seed)

        self.history: list[dict] = []
        self.best_acc = 0.0
        self.best_state: Optional[dict] = None

    @staticmethod
    def _set_seed(seed: int) -> None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        if hasattr(torch.backends.mps, 'manual_seed'):
            torch.backends.mps.manual_seed(seed)

    # ── Subclass hooks ────────────────────────────────────────────
    def _train_epoch(self, epoch: int) -> float:
        raise NotImplementedError

    def _eval(self) -> float:
        raise NotImplementedError

    def _get_model(self) -> torch.nn.Module:
        raise NotImplementedError

    def _save_checkpoint(self, state: Dict[str, Any], filename: str) -> Path:
        path = self.artifacts_dir / filename
        torch.save(state, path)
        return path

    # ── Core training loop ───────────────────────────────────────
    def train(
        self,
        epochs: int = 100,
        patience: int = 15,
        print_every: int = 5,
    ) -> Dict[str, Any]:
        """
        Training loop with early stopping.

        Args:
            epochs: Maximum number of epochs
            patience: Stop if val accuracy doesn't improve for this many epochs
            print_every: Print progress every N epochs

        Returns:
            Training summary dict
        """
        model = self._get_model()
        model.to(self.device)

        patience_counter = 0
        t0 = time.time()

        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(epoch)
            val_acc = self._eval()

            self.history.append({
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_acc": float(val_acc),
            })

            # Checkpoint on best
            improved = val_acc > self.best_acc
            if improved:
                self.best_acc = val_acc
                self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            # Print progress
            if epoch == 1 or epoch % print_every == 0 or improved:
                status = "best" if improved else ""
                print(
                    f"  epoch={epoch:>3d}/{epochs}  "
                    f"loss={train_loss:.4f}  "
                    f"val_acc={val_acc:.4f} {status}"
                )

            # Early stopping
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch} (patience={patience})")
                break

        elapsed = time.time() - t0

        # Restore best model weights
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
            model.to(self.device)

        summary = {
            "best_val_acc": float(self.best_acc),
            "total_epochs": len(self.history),
            "elapsed_seconds": round(elapsed, 1),
            "history": self.history,
        }
        self._save_summary(summary)

        print(
            f"  Training done: best_val_acc={self.best_acc:.4f} "
            f"in {len(self.history)} epochs ({elapsed:.0f}s)"
        )
        return summary

    def _save_summary(self, summary: Dict[str, Any]) -> None:
        with open(self.run_dir / "training_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    # ── Utility ─────────────────────────────────────────────────
    @staticmethod
    def _accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
        preds = logits.argmax(dim=1)
        return float(accuracy_score(targets.cpu().numpy(), preds.cpu().numpy()))
