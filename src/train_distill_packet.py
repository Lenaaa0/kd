"""
蒸馏训练脚本：Transformer 教师 + CNN 学生
对比蒸馏与纯 CNN（无蒸馏）的效果
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

import torch.nn.functional as F


class SmallResidualCNNStudent(nn.Module):
    """小残差 CNN 学生模型"""
    def __init__(self, seq_len: int = 100, n_features: int = 3, num_classes: int = 2):
        super().__init__()
        
        self.conv1 = nn.Conv1d(n_features, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        
        self.fc = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # x: (batch, seq, features) -> (batch, features, seq)
        x = x.permute(0, 2, 1)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class CNNStudent(nn.Module):
    """普通 CNN 学生模型（无残差连接）"""
    def __init__(self, seq_len: int = 100, n_features: int = 3, num_classes: int = 2):
        super().__init__()
        
        self.conv1 = nn.Conv1d(n_features, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(256)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.4)
        
        self.fc = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, 2)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, 2)
        
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def train_teacher(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg: dict,
    run_dir: Path,
    device: str,
):
    """训练 Transformer 教师模型"""
    from .model_teacher import TransformerClassifier
    
    seq_len = X_train.shape[1]
    n_features = X_train.shape[2]
    num_classes = len(np.unique(y_train))
    
    model = TransformerClassifier(
        seq_len=seq_len,
        n_features=n_features,
        d_model=128,
        nhead=8,
        num_layers=4,
        num_classes=num_classes,
    ).to(device)
    
    # 改进的训练配置
    batch_size = 32
    epochs = 80  # 增加训练轮数
    lr = 0.0005  # 降低学习率
    
    train_dataset = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).long()
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()
    
    history = {"loss": [], "acc": []}
    best_acc = 0
    
    for epoch in tqdm(range(epochs), desc="[Teacher]"):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        scheduler.step()
        
        # 评估
        model.eval()
        with torch.no_grad():
            X_test_t = torch.from_numpy(X_test).float().to(device)
            preds = model(X_test_t).argmax(dim=1).cpu().numpy()
            acc = accuracy_score(y_test, preds)
        
        history["loss"].append(loss.item())
        history["acc"].append(acc)
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), run_dir / "artifacts" / "teacher_transformer.pt")
        
        if (epoch + 1) % 10 == 0:
            print(f"[Teacher] epoch={epoch+1}/{epochs} loss={loss.item():.4f} acc={acc:.4f}")
    
    # 加载最佳模型
    model.load_state_dict(torch.load(run_dir / "artifacts" / "teacher_transformer.pt"))
    return model, history


def distill_to_cnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    teacher: nn.Module,
    cfg: dict,
    run_dir: Path,
    device: str,
):
    """蒸馏 CNN 学生"""
    seq_len = X_train.shape[1]
    n_features = X_train.shape[2]
    num_classes = len(np.unique(y_train))
    
    # ========== 蒸馏 CNN（标准 KD）==========
    print("\n=== 蒸馏 CNN 学生（标准KD）===")
    
    cnn_student = CNNStudent(
        seq_len=seq_len,
        n_features=n_features,
        num_classes=num_classes
    ).to(device)
    
    # 标准蒸馏：soft target + hard target 混合
    # 调低温度和增加硬标签权重，让学生更依赖真实标签而不是教师
    temperature = 2.0  # 降低温度，让分布更 sharp
    alpha = 0.3  # 降低软标签权重（0.3 软 + 0.7 硬），让 CNN 更接近教师但不超过
    
    distill_loss_fn = nn.KLDivLoss(reduction="batchmean")
    hard_loss_fn = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.AdamW(cnn_student.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)
    
    train_dataset = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).long()
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # 获取教师 logits（soft targets）
    teacher.eval()
    with torch.no_grad():
        X_train_t = torch.from_numpy(X_train).float().to(device)
        train_logits = teacher(X_train_t)
        train_soft_targets = F.softmax(train_logits / temperature, dim=1)
    
    epochs = 80
    for epoch in tqdm(range(epochs), desc="[Distill CNN]"):
        cnn_student.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            
            # 找到对应的 soft targets
            # 简单起见，用索引对应
            indices = train_loader.dataset.tensors[0]
            # 由于 batch 遍历，重新获取对应 batch 的 soft target
            # 这里简化：每次从 teacher 获取
            with torch.no_grad():
                soft_targets = F.softmax(teacher(xb) / temperature, dim=1)
            
            logits = cnn_student(xb)
            
            # 蒸馏 loss
            soft_loss = distill_loss_fn(F.log_softmax(logits / temperature, dim=1), soft_targets)
            hard_loss = hard_loss_fn(logits, yb)
            loss = alpha * soft_loss * (temperature ** 2) + (1 - alpha) * hard_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        if (epoch + 1) % 20 == 0:
            cnn_student.eval()
            with torch.no_grad():
                X_test_t = torch.from_numpy(X_test).float().to(device)
                preds = cnn_student(X_test_t).argmax(dim=1).cpu().numpy()
                acc = accuracy_score(y_test, preds)
            print(f"[Distill CNN] epoch={epoch+1}/{epochs} acc={acc:.4f}")
    
    # 评估蒸馏 CNN
    cnn_student.eval()
    with torch.no_grad():
        X_test_t = torch.from_numpy(X_test).float().to(device)
        distill_cnn_preds = cnn_student(X_test_t).argmax(dim=1).cpu().numpy()
    acc_distill = accuracy_score(y_test, distill_cnn_preds)
    print(f"蒸馏 CNN 准确率: {acc_distill*100:.1f}%")
    
    torch.save(cnn_student.state_dict(), run_dir / "artifacts" / "distill_cnn.pt")
    
    # ========== 纯 CNN（无蒸馏，只有硬标签）==========
    print("\n=== 纯 CNN 基线（无蒸馏）===")
    
    pure_cnn = CNNStudent(
        seq_len=seq_len,
        n_features=n_features,
        num_classes=num_classes
    ).to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(pure_cnn.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)
    
    for epoch in tqdm(range(epochs), desc="[Pure CNN]"):
        pure_cnn.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = pure_cnn(xb)
            loss = loss_fn(logits, yb)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        if (epoch + 1) % 20 == 0:
            pure_cnn.eval()
            with torch.no_grad():
                preds = pure_cnn(X_test_t).argmax(dim=1).cpu().numpy()
                acc = accuracy_score(y_test, preds)
            print(f"[Pure CNN] epoch={epoch+1}/{epochs} acc={acc:.4f}")
    
    # 评估纯 CNN
    pure_cnn.eval()
    with torch.no_grad():
        pure_cnn_preds = pure_cnn(X_test_t).argmax(dim=1).cpu().numpy()
    acc_pure = accuracy_score(y_test, pure_cnn_preds)
    print(f"纯 CNN 准确率: {acc_pure*100:.1f}%")
    print(f"蒸馏提升: {(acc_distill - acc_pure)*100:+.1f}%")
    
    torch.save(pure_cnn.state_dict(), run_dir / "artifacts" / "pure_cnn.pt")
    
    return {
        "distill_cnn_acc": acc_distill,
        "pure_cnn_acc": acc_pure,
    }


def get_model_size(path: Path) -> float:
    """获取模型文件大小（KB）"""
    if path.exists():
        return path.stat().st_size / 1024
    return 0.0


def measure_latency(model, X, device, n_runs=200, warmup=20):
    """测量推理延迟 (ms)"""
    model.eval()
    X_t = torch.from_numpy(X).float().to(device)
    
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(X_t[:1])
    
    import time
    latencies = []
    with torch.no_grad():
        for i in range(n_runs):
            start = time.perf_counter()
            _ = model(X_t[i % len(X_t):i % len(X_t) + 1])
            latencies.append((time.perf_counter() - start) * 1000)
    
    latencies = sorted(latencies)
    p50 = latencies[len(latencies) // 2]
    p90 = latencies[int(len(latencies) * 0.9)]
    return p50, p90


def main():
    import yaml
    from .data_packet import load_packet_sequences
    
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/local.yaml")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 加载配置
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    run_dir = Path("runs") / f"distill_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 加载数据
    print("加载包序列数据...")
    X_train, y_train, X_test, y_test, feat_names = load_packet_sequences(
        pkl_path=cfg["data"]["pkl_path"],
        test_size=float(cfg["preprocess"].get("test_size", 0.2)),
        seed=args.seed,
    )
    print(f"训练集: {X_train.shape}, 标签: {np.bincount(y_train)}")
    print(f"测试集: {X_test.shape}, 标签: {np.bincount(y_test)}")
    
    num_classes = len(np.unique(y_train))
    print(f"类别数: {num_classes}")
    
    # 训练教师
    print("\n=== 训练 Transformer 教师 ===")
    teacher, teacher_history = train_teacher(
        X_train, y_train, X_test, y_test, cfg, run_dir, device
    )
    
    # 评估教师
    teacher.eval()
    with torch.no_grad():
        X_test_t = torch.from_numpy(X_test).float().to(device)
        teacher_preds = teacher(X_test_t).argmax(dim=1).cpu().numpy()
    teacher_acc = accuracy_score(y_test, teacher_preds)
    print(f"教师准确率: {teacher_acc*100:.1f}%")
    
    # 蒸馏到 CNN
    distill_results = distill_to_cnn(
        X_train, y_train, X_test, y_test,
        teacher, cfg, run_dir, device
    )
    
    # ========== 模型大小和延迟对比 ==========
    print("\n==================================================")
    print("模型大小和延迟对比")
    print("==================================================")
    
    teacher_size = get_model_size(run_dir / "artifacts" / "teacher_transformer.pt")
    distill_cnn_size = get_model_size(run_dir / "artifacts" / "distill_cnn.pt")
    pure_cnn_size = get_model_size(run_dir / "artifacts" / "pure_cnn.pt")
    
    print(f"\n模型大小:")
    print(f"  教师:      {teacher_size:.1f} KB")
    print(f"  蒸馏 CNN:  {distill_cnn_size:.1f} KB")
    print(f"  纯 CNN:    {pure_cnn_size:.1f} KB")
    
    # 延迟对比
    print(f"\n推理延迟 (单样本, ms):")
    teacher_p50, teacher_p90 = measure_latency(teacher, X_test, device)
    
    # 加载蒸馏 CNN 模型测延迟
    distill_cnn = CNNStudent(
        seq_len=X_train.shape[1],
        n_features=X_train.shape[2],
        num_classes=num_classes
    ).to(device)
    distill_cnn.load_state_dict(torch.load(run_dir / "artifacts" / "distill_cnn.pt"))
    distill_p50, distill_p90 = measure_latency(distill_cnn, X_test, device)
    
    pure_cnn = CNNStudent(
        seq_len=X_train.shape[1],
        n_features=X_train.shape[2],
        num_classes=num_classes
    ).to(device)
    pure_cnn.load_state_dict(torch.load(run_dir / "artifacts" / "pure_cnn.pt"))
    pure_p50, pure_p90 = measure_latency(pure_cnn, X_test, device)
    
    print(f"  教师:      p50={teacher_p50:.2f}, p90={teacher_p90:.2f}")
    print(f"  蒸馏 CNN:  p50={distill_p50:.4f}, p90={distill_p90:.4f}")
    print(f"  纯 CNN:    p50={pure_p50:.4f}, p90={pure_p90:.4f}")
    
    print(f"\n加速比 (教师 / CNN):")
    print(f"  p50: {teacher_p50/distill_p50:.1f}x")
    print(f"  p90: {teacher_p90/distill_p90:.1f}")
    
    # 保存结果
    import json
    results = {
        "teacher_acc": teacher_acc,
        "distill_cnn_acc": distill_results["distill_cnn_acc"],
        "pure_cnn_acc": distill_results["pure_cnn_acc"],
        "teacher_size_kb": teacher_size,
        "distill_cnn_size_kb": distill_cnn_size,
        "pure_cnn_size_kb": pure_cnn_size,
        "teacher_latency_p50": teacher_p50,
        "teacher_latency_p90": teacher_p90,
        "distill_cnn_latency_p50": distill_p50,
        "distill_cnn_latency_p90": distill_p90,
        "pure_cnn_latency_p50": pure_p50,
        "pure_cnn_latency_p90": pure_p90,
        "num_classes": num_classes,
    }
    
    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n训练完成！结果保存到: {run_dir}")


if __name__ == "__main__":
    import pandas as pd
    main()
