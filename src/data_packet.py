"""
包序列数据加载模块
用于 Transformer 教师训练
支持二分类(VPN/NonVPN)或多分类(service/app 等)
"""

from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from joblib import dump, load


def load_packet_sequences(
    pkl_path: str,
    test_size: float = 0.2,
    seed: int = 42,
    max_packets: int = 100,
    normalize: bool = True,
    label_encoder: Optional[LabelEncoder] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """
    加载包序列数据
    
    Args:
        normalize: 是否标准化
        label_encoder: 若提供则用其转换标签；否则根据 label 列自动二分类或多分类
    
    Returns:
        X_train, y_train, X_test, y_test, feature_names
        y 为整数 0,1,...,num_classes-1
    """
    df = pd.read_pickle(pkl_path)
    
    n_samples = len(df)
    X = np.zeros((n_samples, max_packets, 3), dtype=np.float32)
    
    for i, row in df.iterrows():
        X[i, :, 0] = row["packet_lengths"][:max_packets]
        X[i, :, 1] = row["directions"][:max_packets]
        X[i, :, 2] = row["inter_arrival_times"][:max_packets]
    
    # 标签：二分类或多分类
    raw_labels = df["label"].astype(str).values
    if label_encoder is not None:
        y = label_encoder.transform(raw_labels).astype(np.int32)
    else:
        uniq = pd.unique(raw_labels)
        if len(uniq) == 2 and set(uniq) <= {"VPN", "NonVPN"}:
            y = (raw_labels == "VPN").astype(np.int32)
        else:
            le = LabelEncoder()
            y = le.fit_transform(raw_labels).astype(np.int32)
            # 可保存 le 供后续预测时用
            dump(le, Path(pkl_path).parent / "label_encoder.pkl")
    
    # 划分训练/测试
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    
    # 标准化：每个特征维度单独标准化
    if normalize:
        scaler = StandardScaler()
        
        # reshape: (n, seq, 3) -> (n*seq, 3) -> 标准化 -> reshape 回
        n_tr, seq_len, n_feat = X_tr.shape
        n_te = X_te.shape[0]
        
        X_tr_flat = X_tr.reshape(-1, n_feat)
        X_te_flat = X_te.reshape(-1, n_feat)
        
        scaler.fit(X_tr_flat)
        X_tr_scaled = scaler.transform(X_tr_flat).reshape(n_tr, seq_len, n_feat)
        X_te_scaled = scaler.transform(X_te_flat).reshape(n_te, seq_len, n_feat)
        
        return X_tr_scaled, y_tr, X_te_scaled, y_te, ["packet_length", "direction", "inter_time"]
    else:
        return X_tr, y_tr, X_te, y_te, ["packet_length", "direction", "inter_time"]


def extract_flow_statistics(X: np.ndarray) -> np.ndarray:
    """
    从包序列提取流级统计特征
    用于 EBM 学生训练
    
    X: (n, seq_len, 3)
    Returns: (n, n_features)
    """
    n = X.shape[0]
    features = []
    
    # 包长特征
    pkt_len = X[:, :, 0]  # (n, seq)
    features.append(pkt_len.mean(axis=1))
    features.append(pkt_len.std(axis=1))
    features.append(pkt_len.max(axis=1))
    features.append(pkt_len.min(axis=1))
    features.append(np.percentile(pkt_len, 25, axis=1))
    features.append(np.percentile(pkt_len, 75, axis=1))
    
    # 方向特征
    direction = X[:, :, 1]  # (n, seq)
    features.append(direction.mean(axis=1))  # 比例
    features.append((direction == 1).sum(axis=1))  # 出向包数
    features.append((direction == 0).sum(axis=1))  # 入向包数
    
    # 时间间隔特征
    inter_time = X[:, :, 2]  # (n, seq)
    features.append(inter_time.mean(axis=1))
    features.append(inter_time.std(axis=1))
    features.append(inter_time.max(axis=1))
    features.append((inter_time > 0).sum(axis=1))  # 非零间隔数
    
    # 流长度
    features.append((pkt_len > 0).sum(axis=1))  # 实际包数
    
    return np.column_stack(features).astype(np.float32)


if __name__ == "__main__":
    # 测试
    X_tr, y_tr, X_te, y_te, feat_names = load_packet_sequences(
        "data/packet_sequences/packet_sequences.pkl",
        test_size=0.2,
        seed=42,
    )
    print(f"训练集: {X_tr.shape}, 标签分布: {np.bincount(y_tr)}")
    print(f"测试集: {X_te.shape}, 标签分布: {np.bincount(y_te)}")
    
    # 测试统计特征提取
    stats = extract_flow_statistics(X_tr)
    print(f"统计特征: {stats.shape}")
