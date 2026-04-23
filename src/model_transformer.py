"""
Transformer 教师模型
处理包序列数据
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model: int, max_len: int = 200, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class PacketTransformerTeacher(nn.Module):
    """
    处理包序列的 Transformer 教师模型
    """
    def __init__(
        self,
        seq_len: int = 100,
        n_features: int = 3,
        num_classes: int = 2,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
        ff_mult: int = 4,
    ):
        super().__init__()
        
        self.seq_len = seq_len
        self.n_features = n_features
        self.d_model = d_model
        
        # 输入嵌入：把 3 维特征映射到 d_model 维
        self.input_proj = nn.Linear(n_features, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len=seq_len, dropout=dropout)
        
        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )
        
        # 特征输出层（用于蒸馏）
        self.feature_proj = nn.Linear(d_model, d_model)
    
    def forward(self, x, return_features=False):
        """
        x: (batch, seq_len, n_features)
        """
        # 输入投影 + 位置编码
        x = self.input_proj(x)  # (batch, seq_len, d_model)
        x = self.pos_encoding(x)
        
        # Transformer 编码
        x = self.transformer(x)  # (batch, seq_len, d_model)
        
        # 池化：取最后一个 token 或平均池化
        x = x.mean(dim=1)  # (batch, d_model)
        
        # 分类
        logits = self.classifier(x)
        
        if return_features:
            features = self.feature_proj(x)
            return logits, features
        
        return logits
    
    def get_embedding(self, x):
        """获取嵌入向量"""
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return x


def create_transformer_teacher(
    seq_len: int = 100,
    n_features: int = 3,
    num_classes: int = 2,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 3,
    dropout: float = 0.1,
    ff_mult: int = 4,
) -> PacketTransformerTeacher:
    """工厂函数"""
    return PacketTransformerTeacher(
        seq_len=seq_len,
        n_features=n_features,
        num_classes=num_classes,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
        ff_mult=ff_mult,
    )


if __name__ == "__main__":
    # 测试
    model = create_transformer_teacher(seq_len=100, num_classes=2)
    x = torch.randn(32, 100, 3)
    out = model(x)
    print(f"输入: {x.shape}")
    print(f"输出: {out.shape}")
    
    out, features = model(x, return_features=True)
    print(f"特征: {features.shape}")
