"""
Transformer teacher model for packet sequence classification.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(1, max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]  # (B, seq, d_model)
        return self.dropout(x)


class TransformerTeacher(nn.Module):
    """
    Transformer-based teacher for packet sequence classification.
    Processes (batch, seq_len, 3) input with multi-head self-attention.
    """

    def __init__(
        self,
        seq_len: int = 100,
        n_features: int = 3,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        num_classes: int = 2,
        dropout: float = 0.1,
        dim_feedforward: Optional[int] = None,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.d_model = d_model
        self.num_classes = num_classes

        if dim_feedforward is None:
            dim_feedforward = d_model * 4

        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=seq_len + 2, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, n_features)
        batch_size = x.shape[0]
        x = self.input_proj(x)  # (B, seq, d_model)
        x = self.pos_enc(x)  # (B, seq, d_model)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 1+seq, d_model)

        x = self.transformer(x)  # (B, 1+seq, d_model)
        cls_out = self.norm(x[:, 0, :])  # (B, d_model)
        return self.head(cls_out)

    def get_config(self) -> dict:
        return {
            "model_type": "transformer_teacher",
            "seq_len": self.seq_len,
            "n_features": self.n_features,
            "d_model": self.d_model,
            "num_classes": self.num_classes,
        }
