"""
Enhanced DNN Teacher model for encrypted traffic classification.
Upgraded with residual connections, GELU activation, and focal loss support.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Residual block: Linear -> BatchNorm -> GELU -> Dropout -> Linear -> BatchNorm.
    Skip connection bridges the two Linear layers.
    """

    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.activation(self.net(x))


class DNNTeacher(nn.Module):
    """
    Enhanced DNN teacher with residual blocks for better training.
    Architecture: Input -> Linear+Bn+ReLU -> ResidualBlocks -> Head
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        hidden_sizes: List[int] = None,
        dropout: float = 0.3,
        use_batch_norm: bool = True,
        residual_depth: int = 2,
        use_gelu: bool = True,
    ):
        """
        Args:
            in_dim: Input feature dimension
            num_classes: Number of classification classes
            hidden_sizes: List of hidden layer sizes, e.g. [256, 128, 64]
            dropout: Dropout rate
            use_batch_norm: Whether to use BatchNorm
            residual_depth: Number of residual blocks in the main body
            use_gelu: Use GELU instead of ReLU
        """
        super().__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes

        if hidden_sizes is None:
            hidden_sizes = [512, 256, 128]

        act_fn = nn.GELU() if use_gelu else nn.ReLU()

        layers: List[nn.Module] = []
        last_dim = in_dim

        for hs in hidden_sizes[:-1]:
            layers.append(nn.Linear(last_dim, hs))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hs))
            layers.append(act_fn)
            layers.append(nn.Dropout(dropout))
            last_dim = hs

        for _ in range(residual_depth):
            layers.append(ResidualBlock(last_dim, dropout))

        layers.append(nn.Linear(last_dim, hidden_sizes[-1]))
        layers.append(nn.BatchNorm1d(hidden_sizes[-1]))
        layers.append(act_fn)
        layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_sizes[-1], num_classes))
        self.net = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def get_config(self) -> dict:
        return {
            "model_type": "dnn_teacher_enhanced",
            "in_dim": self.in_dim,
            "num_classes": self.num_classes,
        }


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    gamma: focusing parameter (higher = more focus on hard examples)
    alpha: class balancing weights
    """

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_classes = logits.size(1)
        log_pt = F.log_softmax(logits, dim=-1)
        pt = log_pt.exp()

        if self.label_smoothing > 0:
            with torch.no_grad():
                true_dist = torch.zeros_like(log_pt)
                true_dist.fill_(self.label_smoothing / n_classes)
                true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing + self.label_smoothing / n_classes)
            ce = -(true_dist * log_pt).sum(dim=-1)
        else:
            ce = F.nll_loss(log_pt, targets, reduction="none")

        pt_selector = pt.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - pt_selector) ** self.gamma

        loss = focal_weight * ce

        if self.alpha is not None:
            alpha_t = self.alpha[targets].to(logits.device)
            loss = alpha_t * loss

        return loss.mean()
