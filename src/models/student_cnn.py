"""
CNN student model for packet sequence classification.
Lightweight 1D CNN designed to be deployed on edge devices.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNStudent(nn.Module):
    """
    Lightweight 1D CNN student.
    Architecture: Conv1D x3 + BatchNorm + ReLU + AdaptivePool + Dropout + FC
    """

    def __init__(
        self,
        seq_len: int = 100,
        n_features: int = 3,
        num_classes: int = 2,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.num_classes = num_classes

        # Three-stage Conv1D feature extractor
        self.conv1 = nn.Conv1d(n_features, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(256)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, n_features) -> (B, n_features, seq_len)
        x = x.permute(0, 2, 1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, 2)  # seq/2

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, 2)  # seq/4

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1)  # (B, 256)

        x = self.dropout(x)
        return self.fc(x)

    def get_config(self) -> dict:
        return {
            "model_type": "cnn_student",
            "seq_len": self.seq_len,
            "n_features": self.n_features,
            "num_classes": self.num_classes,
        }
