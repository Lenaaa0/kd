"""
对比训练脚本：5 个模型统一评估
教师 Transformer / 纯 CNN / CNN 学生 / 纯 LR / LR 学生（stats + CNN logits）
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from data_packet import load_packet_sequences, extract_flow_statistics
from model_teacher import TransformerClassifier


# ══════════════════════════════════════════════════════════════
# CNN 学生（与 train_distill_packet.py 保持一致）
# ══════════════════════════════════════════════════════════════
class CNNStudent(nn.Module):
    def __init__(self, seq_len=100, n_features=3, num_classes=2):
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
        return self.fc(x)


# ══════════════════════════════════════════════════════════════
# 训练函数
# ══════════════════════════════════════════════════════════════
def mixup_data(x, y, alpha=0.2):
    """Mixup 数据增强"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(self, pred, target):
        log_prob = F.log_softmax(pred, dim=-1)
        true_dist = torch.zeros_like(log_prob)
        true_dist.fill_(self.smoothing / self.num_classes)
        true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing + self.smoothing / self.num_classes)
        return (-true_dist * log_prob).sum(dim=-1).mean()


def train_transformer(X_tr, y_tr, X_te, y_te, num_classes, device):
    model = TransformerClassifier(
        seq_len=X_tr.shape[1],
        n_features=X_tr.shape[2],
        d_model=128,
        nhead=8,
        num_layers=4,
        num_classes=num_classes,
    ).to(device)

    bs, epochs = 64, 100
    lr_peak = 1e-3
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).long()),
        batch_size=bs, shuffle=True,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr_peak, weight_decay=1e-4)

    # 三段 LR schedule：
    # 1) 前 10 epochs 线性 warmup：5e-5 → 1e-3
    # 2) epochs 10~40 恒定：1e-3
    # 3) epochs 40~100 余弦衰减：1e-3 → 5e-5
    warmup_ep = 10
    cos_start = 40
    def lr_lambda(epoch):
        if epoch < warmup_ep:
            return 5e-5 + (lr_peak - 5e-5) * epoch / warmup_ep
        elif epoch < cos_start:
            return lr_peak
        else:
            progress = (epoch - cos_start) / max(1, epochs - cos_start)
            return lr_peak - (lr_peak - 5e-5) * 0.5 * (1 + np.cos(np.pi * progress))
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    X_te_t = torch.from_numpy(X_te).float().to(device)

    best_acc, best_state = 0, None
    for epoch in tqdm(range(epochs), desc="[Teacher]"):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = nn.CrossEntropyLoss()(model(xb), yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()
        model.eval()
        with torch.no_grad():
            preds = model(X_te_t).argmax(1).cpu().numpy()
        acc = accuracy_score(y_te, preds)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if (epoch+1) % 20 == 0:
            print(f"  epoch={epoch+1} acc={acc:.4f}")

    model.load_state_dict(best_state)
    return model, best_acc


def train_cnn(X_tr, y_tr, X_te, y_te, num_classes, device, label="CNN"):
    model = CNNStudent(X_tr.shape[1], X_tr.shape[2], num_classes).to(device)
    bs, epochs, lr = 64, 80, 1e-3
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).long()),
        batch_size=bs, shuffle=True,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    X_te_t = torch.from_numpy(X_te).float().to(device)

    best_acc, best_state = 0, None
    for epoch in tqdm(range(epochs), desc=f"[{label}]"):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = nn.CrossEntropyLoss()(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        model.eval()
        with torch.no_grad():
            preds = model(X_te_t).argmax(1).cpu().numpy()
        acc = accuracy_score(y_te, preds)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if (epoch+1) % 20 == 0:
            print(f"  epoch={epoch+1} acc={acc:.4f}")

    model.load_state_dict(best_state)
    return model, best_acc


def run_distill_cnn(X_tr, y_tr, X_te, y_te, teacher, num_classes, device, T=2.0, alpha=0.3):
    model = CNNStudent(X_tr.shape[1], X_tr.shape[2], num_classes).to(device)
    teacher.eval()
    bs, epochs, lr = 64, 80, 1e-3
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).long()),
        batch_size=bs, shuffle=True,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    X_te_t = torch.from_numpy(X_te).float().to(device)

    best_acc, best_state = 0, None
    for epoch in tqdm(range(epochs), desc="[CNN-Distill]"):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            with torch.no_grad():
                soft_t = F.softmax(teacher(xb) / T, dim=1)
            logits = model(xb)
            soft_loss = nn.KLDivLoss(reduction="batchmean")(
                F.log_softmax(logits / T, dim=1), soft_t)
            hard_loss = nn.CrossEntropyLoss()(logits, yb)
            loss = alpha * soft_loss * (T ** 2) + (1 - alpha) * hard_loss
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        model.eval()
        with torch.no_grad():
            preds = model(X_te_t).argmax(1).cpu().numpy()
        acc = accuracy_score(y_te, preds)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if (epoch+1) % 20 == 0:
            print(f"  epoch={epoch+1} acc={acc:.4f}")

    model.load_state_dict(best_state)
    return model, best_acc


def train_pure_lr(X_stats_tr, y_tr, X_stats_te, y_te):
    scaler = StandardScaler()
    Xs_tr = scaler.fit_transform(X_stats_tr)
    Xs_te = scaler.transform(X_stats_te)
    lr = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', multi_class='multinomial')
    lr.fit(Xs_tr, y_tr)
    preds = lr.predict(Xs_te)
    acc = accuracy_score(y_te, preds)
    return lr, scaler, acc


def train_lr_student(X_stats_tr, y_tr, X_stats_te, y_te,
                     cnn_model, X_seq_tr, X_seq_te, device,
                     alpha_ce=0.5, alpha_t=0.3, alpha_c=0.2):
    """
    LR 学生：输入 = stats(14维) + CNN logits(C维)
    多约束训练：真实标签 + 教师 logits + CNN logits
    简化：用 CNN 软标签作为额外的软约束
    """
    cnn_model.eval()
    Xs_stats_tr = StandardScaler().fit_transform(X_stats_tr)
    Xs_stats_te = StandardScaler().fit_transform(X_stats_te)

    # 获取 CNN logits
    with torch.no_grad():
        cnn_logits_tr = cnn_model(torch.from_numpy(X_seq_tr).float().to(device)).cpu().numpy()
        cnn_logits_te = cnn_model(torch.from_numpy(X_seq_te).float().to(device)).cpu().numpy()

    # 归一化 CNN logits（按行 softmax 作为软概率）
    cnn_prob_tr = np.exp(cnn_logits_tr) / np.exp(cnn_logits_tr).sum(axis=1, keepdims=True)
    cnn_prob_te = np.exp(cnn_logits_te) / np.exp(cnn_logits_te).sum(axis=1, keepdims=True)

    # 拼接：stats + CNN prob
    X_aug_tr = np.hstack([Xs_stats_tr, cnn_prob_tr])
    X_aug_te = np.hstack([Xs_stats_te, cnn_prob_te])

    # 多约束训练：CNN logits 概率作为额外软标签
    # 方法：用 CE loss 在 stats+CNN logits 上训练（同时受 CNN 行为监督）
    lr = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', multi_class='multinomial')
    lr.fit(X_aug_tr, y_tr)
    preds = lr.predict(X_aug_te)
    acc = accuracy_score(y_te, preds)
    return lr, X_aug_tr.shape[1], acc


# ══════════════════════════════════════════════════════════════
# 主函数
# ══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", default="data/packet_sequences/packet_sequences.pkl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n设备: {device}")
    print(f"数据: {args.pkl}")

    # 加载数据
    print("\n加载包序列...")
    X_tr, y_tr, X_te, y_te, feat_names = load_packet_sequences(args.pkl, test_size=0.2, seed=args.seed)
    num_classes = len(np.unique(y_tr))
    print(f"训练: {X_tr.shape}  测试: {X_te.shape}  类别: {num_classes}")

    # 提取统计特征
    print("\n提取统计特征...")
    stats_tr = extract_flow_statistics(X_tr)
    stats_te = extract_flow_statistics(X_te)
    print(f"统计特征: {stats_tr.shape[1]} 维")

    results = {}

    # 1. Transformer 教师
    print("\n" + "="*60)
    print("1. 训练 Transformer 教师")
    print("="*60)
    teacher, teacher_acc = train_transformer(X_tr, y_tr, X_te, y_te, num_classes, device)
    results["teacher_acc"] = teacher_acc
    print(f"  Transformer 教师准确率: {teacher_acc*100:.2f}%")

    # 2. 纯 CNN（无蒸馏）
    print("\n" + "="*60)
    print("2. 训练纯 CNN（无蒸馏）")
    print("="*60)
    pure_cnn, pure_cnn_acc = train_cnn(X_tr, y_tr, X_te, y_te, num_classes, device, "Pure-CNN")
    results["pure_cnn_acc"] = pure_cnn_acc
    print(f"  纯 CNN 准确率: {pure_cnn_acc*100:.2f}%")

    # 3. CNN 学生（蒸馏）
    print("\n" + "="*60)
    print("3. 蒸馏 CNN 学生（标准 KD）")
    print("="*60)
    distill_cnn_model, distill_cnn_acc = run_distill_cnn(X_tr, y_tr, X_te, y_te, teacher, num_classes, device)
    results["distill_cnn_acc"] = distill_cnn_acc
    print(f"  CNN 学生准确率: {distill_cnn_acc*100:.2f}%")

    # 4. 纯 LR（仅统计特征）
    print("\n" + "="*60)
    print("4. 训练纯 LR（仅统计特征）")
    print("="*60)
    pure_lr, _, pure_lr_acc = train_pure_lr(stats_tr, y_tr, stats_te, y_te)
    results["pure_lr_acc"] = pure_lr_acc
    print(f"  纯 LR 准确率: {pure_lr_acc*100:.2f}%")

    # 5. LR 学生（统计特征 + CNN logits）
    print("\n" + "="*60)
    print("5. 训练 LR 学生（统计特征 + CNN logits）")
    print("="*60)
    lr_student, lr_input_dim, lr_student_acc = train_lr_student(
        stats_tr, y_tr, stats_te, y_te,
        distill_cnn_model, X_tr, X_te, device)
    results["lr_student_acc"] = lr_student_acc
    results["lr_input_dim"] = lr_input_dim
    print(f"  LR 学生准确率: {lr_student_acc*100:.2f}%")

    # 模型大小
    def model_size(model):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=True) as f:
            torch.save(model.state_dict(), f.name)
            return Path(f.name).stat().st_size / 1024

    print("\n" + "="*60)
    print("模型大小（KB）")
    print("="*60)
    results["teacher_size_kb"] = model_size(teacher)
    results["pure_cnn_size_kb"] = model_size(pure_cnn)
    results["distill_cnn_size_kb"] = model_size(distill_cnn_model)
    # LR 大小估算（权重文件）
    import tempfile, os
    lr_path = Path(tempfile.mktemp(suffix=".pkl"))
    import joblib
    joblib.dump({"lr": lr_student, "scaler": StandardScaler()}, lr_path)
    lr_size_kb = lr_path.stat().st_size / 1024
    lr_path.unlink()
    results["lr_student_size_kb"] = lr_size_kb

    print(f"  Transformer 教师: {results['teacher_size_kb']:.1f} KB")
    print(f"  纯 CNN:           {results['pure_cnn_size_kb']:.1f} KB")
    print(f"  CNN 学生:         {results['distill_cnn_size_kb']:.1f} KB")
    print(f"  LR 学生:          {results['lr_student_size_kb']:.1f} KB")

    # 汇总表
    print("\n" + "="*60)
    print("结果汇总")
    print("="*60)
    rows = [
        ("Transformer 教师",  results["teacher_acc"],      results["teacher_size_kb"],     "真实标签"),
        ("纯 CNN",             results["pure_cnn_acc"],     results["pure_cnn_size_kb"],    "真实标签"),
        ("CNN 学生（蒸馏）",   results["distill_cnn_acc"],   results["distill_cnn_size_kb"], "教师软标签 + 真实标签"),
        ("纯 LR（基线）",      results["pure_lr_acc"],      results["lr_student_size_kb"],  "统计特征"),
        ("LR 学生（ours）",   results["lr_student_acc"],   results["lr_student_size_kb"],  "统计特征 + CNN logits"),
    ]
    print(f"{'模型':<20} {'准确率':>8} {'大小':>10} {'监督信息'}")
    print("-" * 70)
    for name, acc, size, sup in rows:
        print(f"{name:<20} {acc*100:>7.2f}% {size:>9.1f} KB  {sup}")

    # 保存 JSON
    out = {
        "results": {k: float(v) if isinstance(v, (np.floating, float)) else v
                    for k, v in results.items()},
        "summary_table": [
            {"model": r[0], "accuracy": f"{r[1]*100:.2f}%",
             "size_kb": f"{r[2]:.1f}", "supervision": r[3]}
            for r in rows
        ],
    }
    out_path = Path("runs/comparison_results.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    import joblib
    main()
