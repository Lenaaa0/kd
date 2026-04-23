from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerClassifier(nn.Module):
    """Transformer 分类器，处理包序列数据"""
    def __init__(
        self,
        seq_len: int = 100,
        n_features: int = 3,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        num_classes: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.d_model = d_model
        
        # 输入投影
        self.input_proj = nn.Linear(n_features, d_model)
        
        # 位置编码
        self.pos_encoder = nn.Parameter(torch.zeros(1, seq_len, d_model))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 分类头
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        
        # 初始化
        nn.init.normal_(self.pos_encoder, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        
    def forward(self, x):
        # x: (batch, seq_len, n_features)
        batch_size = x.shape[0]
        
        # 投影到 d_model 维度
        x = self.input_proj(x)  # (batch, seq, d_model)
        
        # 添加位置编码
        x = x + self.pos_encoder
        
        # 添加 CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, 1+seq, d_model)
        
        # Transformer
        x = self.transformer(x)
        
        # 取 CLS token
        cls_output = self.norm(x[:, 0, :])
        
        return self.head(cls_output)


class MLPTeacher(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, hidden_sizes: List[int], dropout: float):
        super().__init__()
        layers: List[nn.Module] = []
        last = in_dim
        for hs in hidden_sizes:
            layers.append(nn.Linear(last, hs))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            last = hs
        layers.append(nn.Linear(last, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TabularTransformerTeacher(nn.Module):
    """
    一个面向纯数值表格特征的轻量 Transformer 教师模型。
    思路：把每个标量特征映射到 token embedding，拼成序列后用 TransformerEncoder，
    取 [CLS] token 做分类。
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.num_features = int(num_features)
        self.num_classes = int(num_classes)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)
        self.dropout = float(dropout)
        self.ff_mult = int(ff_mult)

        # 每个 feature 一个可学习线性映射：x_j -> token_j (d_model)
        self.feat_proj = nn.Linear(1, self.d_model)
        # 特征位置编码（learnable）
        self.feat_pos = nn.Parameter(torch.zeros(1, self.num_features, self.d_model))
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_model * self.ff_mult,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=self.n_layers)
        self.norm = nn.LayerNorm(self.d_model)
        self.head = nn.Linear(self.d_model, self.num_classes)

        self._init_params()

    def _init_params(self) -> None:
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.feat_pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F)
        b, f = x.shape
        if f != self.num_features:
            raise ValueError(f"Expected {self.num_features} features, got {f}.")

        # (B, F, 1) -> (B, F, d)
        tok = self.feat_proj(x.unsqueeze(-1))
        tok = tok + self.feat_pos

        cls = self.cls_token.expand(b, -1, -1)  # (B, 1, d)
        seq = torch.cat([cls, tok], dim=1)  # (B, 1+F, d)
        z = self.encoder(seq)
        cls_z = self.norm(z[:, 0, :])
        return self.head(cls_z)

