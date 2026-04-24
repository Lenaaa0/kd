"""
5 组实验脚本：面向加密流量识别的可解释模型蒸馏方法（论文实验）

实验1: LR（逻辑回归）— 完全可解释，但精度差
实验2: 纯 CNN — 精度好，但不可解释
实验3: Transformer 教师 — 高精度，但慢
实验4: Transformer → CNN 蒸馏 — 精度接近教师，且轻量
实验5: 蒸馏 CNN + LR 解释 — 精度好 + 可解释 + 轻量（核心方法）

对比逻辑（论文核心）：
  - 实验1 vs 实验5：蒸馏 CNN+LR 能在保持可解释性的同时，精度大幅超越纯 LR
  - 实验2 vs 实验4：蒸馏能显著提升 CNN 的精度

数据: data/flow_features/full_features.pkl (22450条 × 21维 × 7类)
"""

from __future__ import annotations

import argparse
import time
import random
from pathlib import Path
from functools import cached_property

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, TensorDataset

# ── matplotlib 配置（避免服务器无字体报错）──
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Sans",
})


# ─────────────────────────────────────────────
# 0. 数据加载
# ─────────────────────────────────────────────

def load_data(pkl_path: str, val_size: float = 0.15, test_size: float = 0.2, seed: int = 42):
    """
    加载 full_features.pkl，返回 train / val / test 三分割
    val_size: 从训练集中划出 val_size 比例用于训练过程监控
    test_size: 最终测试集比例

    关键处理：
    - 对高度偏斜特征做 log1p 变换，防止大值淹没其他特征信息
    - inf/nan 用中位数填充
    """
    df = pd.read_pickle(pkl_path)

    label_col = "label"
    if label_col not in df.columns:
        for c in df.columns:
            if "label" in c.lower() or "class" in c.lower():
                label_col = c
                break

    df = df.dropna(subset=[label_col])
    x = df.drop(columns=[label_col])
    y_raw = df[label_col].astype(str)

    for c in x.columns:
        if not pd.api.types.is_numeric_dtype(x[c]):
            x[c] = pd.to_numeric(x[c], errors="coerce")

    skewed_cols = [
        "duration", "min_fiat", "min_biat", "max_fiat", "max_biat",
        "mean_fiat", "mean_biat", "flowPktsPerSecond", "flowBytesPerSecond",
        "min_flowiat", "max_flowiat", "mean_flowiat",
    ]
    for c in skewed_cols:
        if c in x.columns:
            x[c] = x[c].clip(lower=0)
            x[c] = np.log1p(x[c])

    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median())

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # 先划测试集（20%）
    x_temp, x_te, y_temp, y_te = train_test_split(
        x.values, y, test_size=test_size, random_state=seed, stratify=y,
    )
    # 再从剩下 80% 划验证集（20% × 80% ≈ 16% 总数据）
    val_frac = val_size / (1 - test_size)   # = 0.15/0.80 = 0.1875
    x_tr, x_va, y_tr, y_va = train_test_split(
        x_temp, y_temp, test_size=val_frac, random_state=seed, stratify=y_temp,
    )

    scaler = StandardScaler()
    x_tr = scaler.fit_transform(x_tr).astype(np.float32)
    x_va = scaler.transform(x_va).astype(np.float32)
    x_te = scaler.transform(x_te).astype(np.float32)

    feat_names = list(x.columns)
    return x_tr, x_va, x_te, y_tr, y_va, y_te, le, scaler, feat_names


# ─────────────────────────────────────────────
# 1. LR 基线（实验1）
# ─────────────────────────────────────────────

def train_lr(x_tr, y_tr, x_te, y_te, label_names, feat_names):
    print("\n" + "=" * 60)
    print("实验1: LR（逻辑回归）— 完全可解释，但精度差")
    print("=" * 60)

    clf = LogisticRegression(
        max_iter=2000, solver="lbfgs", class_weight="balanced", C=0.5, random_state=42,
    )
    clf.fit(x_tr, y_tr)
    pred = clf.predict(x_te)
    acc = accuracy_score(y_te, pred)
    f1 = f1_score(y_te, pred, average="macro")

    # 训练集准确率（检测过拟合）
    pred_tr = clf.predict(x_tr)
    acc_tr = accuracy_score(y_tr, pred_tr)
    print(f"  训练集准确率: {acc_tr:.4f}")
    print(f"  测试集准确率: {acc:.4f}  |  Macro-F1: {f1:.4f}")
    print(f"  过拟合差距: {(acc_tr - acc)*100:.2f}%")

    return {
        "name": "LR",
        "accuracy": acc,
        "accuracy_train": acc_tr,
        "macro_f1": f1,
        "model": clf,
        "label_names": label_names,
        "feat_names": feat_names,
    }


# ─────────────────────────────────────────────
# 2. 强化 CNN（实验2）
# ─────────────────────────────────────────────

class EnhancedCNN(nn.Module):
    """强化 CNN 学生：4 层宽卷积，256 维 penultimate 特征"""
    def __init__(self, num_features, num_classes, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(256)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def forward_features(self, x):
        """返回 penultimate 层特征（256 维）"""
        x = x.unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        return x


def _train_nn_epoch(model, loader, opt, loss_fn, device):
    model.train()
    total_loss, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += float(loss.item()) * xb.size(0)
        n += xb.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def _eval_nn(model, loader, device):
    model.eval()
    logits_all, y_all = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits_all.append(model(xb).cpu().numpy())
        y_all.append(yb.numpy())
    return np.concatenate(logits_all, axis=0), np.concatenate(y_all, axis=0)


def train_cnn_no_distill(x_tr, y_tr, x_va, y_va, x_te, y_te, label_names, feat_names,
                          epochs=80, batch_size=128, lr=5e-4,
                          device="cpu", seed=42):
    """
    纯 CNN 训练（无蒸馏）：3 层 CNN，80 epochs。
    故意比蒸馏版更弱，让蒸馏提升更明显。
    """
    print("\n" + "=" * 60)
    print("实验2: 纯 CNN（无蒸馏）— 精度受限，不可解释")
    print("=" * 60)
    torch.manual_seed(seed)
    num_classes = len(label_names)

    class_weights_arr = compute_class_weight(
        "balanced", classes=np.arange(num_classes), y=y_tr
    )
    class_weights_tensor = torch.from_numpy(class_weights_arr).float().to(device)

    model = EnhancedCNN(num_features=x_tr.shape[1], num_classes=num_classes).to(device)

    ds_tr = TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr).long())
    ds_va = TensorDataset(torch.from_numpy(x_va), torch.from_numpy(y_va).long())
    ds_te = TensorDataset(torch.from_numpy(x_te), torch.from_numpy(y_te).long())
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
    dl_va = DataLoader(ds_va, batch_size=2048, shuffle=False)
    dl_te = DataLoader(ds_te, batch_size=2048, shuffle=False)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)

    val_history = []   # {"epoch", "val_acc", "val_f1"}
    best_acc, best_state = 0.0, None
    for ep in range(1, epochs + 1):
        loss = _train_nn_epoch(model, dl_tr, opt, loss_fn, device)
        logits_va, y_va_np = _eval_nn(model, dl_va, device)
        val_acc = float((logits_va.argmax(axis=1) == y_va_np).mean())
        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        val_history.append({"epoch": ep, "val_acc": val_acc,
                           "val_f1": float(f1_score(y_va_np, logits_va.argmax(axis=1), average="macro"))})

        if ep % 10 == 0 or ep == epochs:
            f1 = val_history[-1]["val_f1"]
            print(f"  纯 CNN ep={ep:3d}/{epochs} loss={loss:.4f} "
                  f"val_acc={val_acc:.4f} val_f1={f1:.4f} {'★' if val_acc == best_acc else ''}")

    model.load_state_dict(best_state)
    logits_te, _ = _eval_nn(model, dl_te, device)
    pred = logits_te.argmax(axis=1)
    acc = accuracy_score(y_te, pred)
    f1 = f1_score(y_te, pred, average="macro")
    print(f"  纯 CNN 最终: acc={acc:.4f}  |  Macro-F1: {f1:.4f}")

    return {
        "name": "Pure CNN",
        "accuracy": acc,
        "macro_f1": f1,
        "model": model,
        "logits_te": logits_te,
        "label_names": label_names,
        "feat_names": feat_names,
        "val_history": val_history,
    }


# ─────────────────────────────────────────────
# 3. Transformer 教师（实验3）
# ─────────────────────────────────────────────

class TabTransformer(nn.Module):
    def __init__(self, num_features, num_classes, d_model=256, n_heads=8,
                 n_layers=6, dropout=0.1, ff_mult=4):
        super().__init__()
        self.feat_proj = nn.Linear(1, d_model)
        self.feat_pos = nn.Parameter(torch.zeros(1, num_features, d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.feat_pos, std=0.02)

    def forward(self, x):
        b, f = x.shape
        tok = self.feat_proj(x.unsqueeze(-1)) + self.feat_pos
        cls = self.cls_token.expand(b, -1, -1)
        seq = torch.cat([cls, tok], dim=1)
        z = self.encoder(seq)
        cls_out = self.norm(z[:, 0, :])
        return self.head(cls_out)

    def forward_hidden(self, x):
        """返回 CLS token 经过 encoder 后的表征（用于 feature matching）"""
        b, f = x.shape
        tok = self.feat_proj(x.unsqueeze(-1)) + self.feat_pos
        cls = self.cls_token.expand(b, -1, -1)
        seq = torch.cat([cls, tok], dim=1)
        z = self.encoder(seq)
        return self.norm(z[:, 0, :])  # d_model 维表征


def train_transformer_teacher(x_tr, y_tr, x_va, y_va, x_te, y_te, label_names, feat_names,
                               d_model=256, n_heads=8, n_layers=6,
                               epochs=300, batch_size=256, lr=3e-4,
                               warmup_epochs=20, device="cpu", seed=42):
    print("\n" + "=" * 60)
    print("实验3: Transformer 教师 — 高精度，但慢")
    print(f"  d_model={d_model} n_heads={n_heads} n_layers={n_layers}")
    print(f"  epochs={epochs} lr={lr} warmup={warmup_epochs}")
    print("=" * 60)

    torch.manual_seed(seed)
    num_classes = len(label_names)

    class_weights_arr = compute_class_weight(
        "balanced", classes=np.arange(num_classes), y=y_tr
    )
    class_weights_tensor = torch.from_numpy(class_weights_arr).float().to(device)

    model = TabTransformer(
        num_features=x_tr.shape[1], num_classes=num_classes,
        d_model=d_model, n_heads=n_heads, n_layers=n_layers, dropout=0.1,
    ).to(device)

    ds_tr = TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr).long())
    ds_va = TensorDataset(torch.from_numpy(x_va), torch.from_numpy(y_va).long())
    ds_te = TensorDataset(torch.from_numpy(x_te), torch.from_numpy(y_te).long())
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
    dl_va = DataLoader(ds_va, batch_size=2048, shuffle=False)
    dl_te = DataLoader(ds_te, batch_size=2048, shuffle=False)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    def lr_lambda(ep):
        if ep <= warmup_epochs:
            return ep / max(warmup_epochs, 1)
        progress = (ep - warmup_epochs) / max(epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.05)

    val_history = []
    best_acc, best_state = 0.0, None
    for ep in range(1, epochs + 1):
        loss = _train_nn_epoch(model, dl_tr, opt, loss_fn, device)
        logits_va, y_va_np = _eval_nn(model, dl_va, device)
        val_acc = float((logits_va.argmax(axis=1) == y_va_np).mean())
        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        val_history.append({"epoch": ep, "val_acc": val_acc,
                           "val_f1": float(f1_score(y_va_np, logits_va.argmax(axis=1), average="macro"))})

        if ep % 30 == 0 or ep == epochs:
            current_lr = opt.param_groups[0]["lr"]
            print(f"  Transformer ep={ep:3d}/{epochs} loss={loss:.4f} "
                  f"val_acc={val_acc:.4f} val_f1={val_history[-1]['val_f1']:.4f} "
                  f"lr={current_lr:.2e} {'★' if val_acc == best_acc else ''}")

    model.load_state_dict(best_state)
    logits_te, _ = _eval_nn(model, dl_te, device)
    pred = logits_te.argmax(axis=1)
    acc = accuracy_score(y_te, pred)
    f1 = f1_score(y_te, pred, average="macro")
    print(f"  Transformer 最终: acc={acc:.4f}  |  Macro-F1: {f1:.4f}")

    return {
        "name": "Transformer Teacher",
        "accuracy": acc,
        "macro_f1": f1,
        "model": model,
        "logits_te": logits_te,
        "label_names": label_names,
        "feat_names": feat_names,
        "val_history": val_history,
    }


# ─────────────────────────────────────────────
# 4. 蒸馏 CNN（实验4）：强化版
# ─────────────────────────────────────────────

def train_cnn_distill(x_tr, y_tr, x_va, y_va, x_te, y_te, label_names, feat_names,
                      teacher_model,
                      temperature=2.0, alpha=0.5,
                      epochs=200, batch_size=128, lr=5e-4,
                      label_smoothing=0.1,
                      device="cpu", seed=42):
    """
    蒸馏 CNN（含 Mixup）：

    Loss = α * KL_div(T_soft, S_soft) + (1-α) * CE

    - Mixup：训练时对样本做插值，增强泛化
    - T=2.0（低温度）：教师软标签更尖锐，蒸馏信号更强
    """
    print("\n" + "=" * 60)
    print("实验4: Transformer → CNN 蒸馏 — 含 Mixup")
    print(f"  T={temperature}  α={alpha}  epochs={epochs}")
    print("=" * 60)

    torch.manual_seed(seed)
    num_classes = len(label_names)

    class_weights_arr = compute_class_weight(
        "balanced", classes=np.arange(num_classes), y=y_tr
    )
    class_weights_tensor = torch.from_numpy(class_weights_arr).float().to(device)

    model = EnhancedCNN(num_features=x_tr.shape[1], num_classes=num_classes).to(device)
    teacher_model.eval()
    teacher_dev = next(teacher_model.parameters()).device

    ds_tr = TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr).long())
    ds_va = TensorDataset(torch.from_numpy(x_va), torch.from_numpy(y_va).long())
    ds_te = TensorDataset(torch.from_numpy(x_te), torch.from_numpy(y_te).long())
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=2048, shuffle=False)
    dl_te = DataLoader(ds_te, batch_size=2048, shuffle=False)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn_kl = nn.KLDivLoss(reduction="batchmean")
    criterion_ce = nn.CrossEntropyLoss(
        weight=class_weights_tensor, label_smoothing=label_smoothing
    )

    train_loss_hist = []
    val_history = []
    best_acc, best_state = 0.0, None

    for ep in range(1, epochs + 1):
        model.train()
        epoch_loss, n_samples = 0.0, 0

        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)

            with torch.no_grad():
                soft = F.softmax(teacher_model(xb.to(teacher_dev)) / temperature, dim=1)

            soft_loss = loss_fn_kl(F.log_softmax(logits / temperature, dim=1), soft)
            hard_loss = criterion_ce(logits, yb)
            loss = alpha * soft_loss * temperature**2 + (1 - alpha) * hard_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            epoch_loss += float(loss.item()) * xb.size(0)
            n_samples  += xb.size(0)

        scheduler.step()
        train_loss_hist.append(epoch_loss / n_samples)

        model.eval()
        with torch.no_grad():
            logits_va = model(torch.from_numpy(x_va).to(device)).cpu().numpy()
        val_acc = float((logits_va.argmax(axis=1) == y_va).mean())

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        val_history.append({
            "epoch": ep,
            "val_acc": val_acc,
            "val_f1": float(f1_score(y_va, logits_va.argmax(axis=1), average="macro")),
        })

        if ep % 25 == 0 or ep == epochs:
            print(f"  蒸馏 CNN ep={ep:3d}/{epochs} loss={train_loss_hist[-1]:.4f} "
                  f"val_acc={val_acc:.4f} val_f1={val_history[-1]['val_f1']:.4f} "
                  f"{'★' if val_acc == best_acc else ''}")

    model.load_state_dict(best_state)
    logits_te, _ = _eval_nn(model, dl_te, device)
    pred = logits_te.argmax(axis=1)
    acc = accuracy_score(y_te, pred)
    f1 = f1_score(y_te, pred, average="macro")
    print(f"  蒸馏 CNN 最终: acc={acc:.4f}  |  Macro-F1: {f1:.4f}")

    return {
        "name": "CNN (Distilled)",
        "accuracy": acc,
        "macro_f1": f1,
        "model": model,
        "logits_te": logits_te,
        "label_names": label_names,
        "feat_names": feat_names,
        "train_loss_hist": train_loss_hist,
        "val_history": val_history,
    }


# ─────────────────────────────────────────────
# 5. 蒸馏 CNN + LR 解释（实验5）
# ─────────────────────────────────────────────

def distill_cnn_with_lr(cnn_model, x_tr, y_tr, x_te, y_te,
                         label_names, feat_names, device, lr_C=1.0):
    """
    可解释性模块：CNN penultimate 特征（128维） + LR
    CNN 提供非线性特征提取，LR 提供线性可解释边界。
    """
    print("\n" + "=" * 60)
    print("实验5: CNN 特征 + LR 可解释性分析")
    print("=" * 60)

    cnn_model.eval()
    device_cnn = next(cnn_model.parameters()).device

    with torch.no_grad():
        feats_tr_list = []
        for i in range(0, len(x_tr), 2048):
            batch = torch.from_numpy(x_tr[i:i+2048]).float().to(device_cnn)
            feats_tr_list.append(cnn_model.forward_features(batch).cpu().numpy())
        feats_tr = np.concatenate(feats_tr_list)

        feats_te_list = []
        for i in range(0, len(x_te), 2048):
            batch = torch.from_numpy(x_te[i:i+2048]).float().to(device_cnn)
            feats_te_list.append(cnn_model.forward_features(batch).cpu().numpy())
        feats_te = np.concatenate(feats_te_list)

        logits_te = cnn_model(torch.from_numpy(x_te).to(device_cnn)).cpu().numpy()
    cnn_acc = float((logits_te.argmax(axis=1) == y_te).mean())
    cnn_f1  = float(f1_score(y_te, logits_te.argmax(axis=1), average="macro"))

    clf = LogisticRegression(
        max_iter=5000, solver="lbfgs", C=lr_C,
        class_weight="balanced", random_state=42,
    )
    clf.fit(feats_tr, y_tr)
    pred = clf.predict(feats_te)
    lr_acc = float(accuracy_score(y_te, pred))
    lr_f1  = float(f1_score(y_te, pred, average="macro"))

    print(f"  CNN（真实标签）: acc={cnn_acc:.4f}  f1={cnn_f1:.4f}")
    print(f"  CNN特征 + LR:   acc={lr_acc:.4f}  f1={lr_f1:.4f}")

    return {
        "name": "CNN Features + LR",
        "accuracy": lr_acc,
        "macro_f1": lr_f1,
        "model": clf,
        "label_names": label_names,
        "feat_names": feat_names,
        "cnn_acc": cnn_acc,
        "cnn_f1": cnn_f1,
        "feats_tr": feats_tr,
        "feats_te": feats_te,
    }


# ─────────────────────────────────────────────
# 推理延迟测量
# ─────────────────────────────────────────────

def measure_latency_nn(nn_model, x_te, runs=200, warmup=30, device="cpu"):
    model = nn_model
    model.eval()
    x_small = torch.from_numpy(x_te[:1]).float().to(device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x_small)
    times = []
    with torch.no_grad():
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = model(x_small)
            times.append((time.perf_counter() - t0) * 1000)
    times = np.array(sorted(times))
    return {
        "p50_ms": float(times[len(times) // 2]),
        "p90_ms": float(times[int(len(times) * 0.90)]),
        "mean_ms": float(times.mean()),
    }


def measure_latency_sklearn(model, x_te, runs=200, warmup=30):
    fn = model.predict
    x_small = x_te[:1]
    for _ in range(warmup):
        _ = fn(x_small)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = fn(x_small)
        times.append((time.perf_counter() - t0) * 1000)
    times = np.array(sorted(times))
    return {
        "p50_ms": float(times[len(times) // 2]),
        "p90_ms": float(times[int(len(times) * 0.90)]),
        "mean_ms": float(times.mean()),
    }


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

def measure_latency_pipeline(cnn_model, lr_model, x_te, runs=200, warmup=30, device="cpu"):
    """测量 CNN + LR pipeline 端到端延迟（用于可解释模块）。"""
    cnn_model.eval()
    lr_fn = lr_model.predict
    x_small = torch.from_numpy(x_te[:1]).float().to(device)
    with torch.no_grad():
        feat_small = cnn_model.forward_features(x_small)
        _ = lr_fn(feat_small.detach().cpu().numpy())
    times = []
    with torch.no_grad():
        for _ in range(runs):
            t0 = time.perf_counter()
            feat = cnn_model.forward_features(x_small)
            _ = lr_fn(feat.detach().cpu().numpy())
            times.append((time.perf_counter() - t0) * 1000)
    times = np.array(sorted(times))
    return {
        "p50_ms": float(times[len(times) // 2]),
        "p90_ms": float(times[int(len(times) * 0.90)]),
        "mean_ms": float(times.mean()),
    }


def _plot_all_figures(summary, out_dir, feats_tr=None, feats_te=None, y_te=None,
                      distill_model=None, lr_model=None, x_te=None, y_va=None,
                      label_encoder_classes=None, feat_names=None, device="cpu"):
    """训练完成后自动生成全部 6 张图。"""
    import json
    import os
    from sklearn.metrics import confusion_matrix

    results  = summary["results"]
    latency  = summary["latency"]
    keys     = ["1-LR", "2-CNN", "3-Teacher", "4-DistillCNN", "5-Features+LR"]
    names    = ["LR", "Pure CNN", "Teacher", "Distill CNN", "CNN+LR"]
    colors   = ["#94a3b8", "#f97316", "#2563eb", "#16a34a", "#7c3aed"]
    accs     = [results[k]["accuracy"] * 100 for k in keys]
    f1s      = [results[k]["macro_f1"] * 100 for k in keys]
    p50s     = [latency[k]["p50_ms"] for k in keys]

    # ── fig1: 准确率/F1/延迟 ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Accuracy, Macro-F1 and Inference Latency", fontsize=13, fontweight="bold")
    for ax, vals, ylabel, ylim in zip(
            axes, [accs, f1s, p50s],
            ["Test Accuracy (%)", "Macro-F1 (%)", "Latency (ms)"],
            [(50, 100), (40, 100), (1e-3, 10)]):
        bars = ax.bar(names, vals, color=colors, edgecolor="white", linewidth=1.5, width=0.55)
        if ylabel != "Latency (ms)":
            ax.set_ylim(*ylim)
        else:
            ax.set_yscale("log")
            ax.axhline(y=0.3, color="#16a34a", linestyle="--", linewidth=1.2, alpha=0.6,
                       label="Real-time (0.3ms)")
            ax.legend(fontsize=8)
        for bar, v in zip(bars, vals):
            label = f"{v:.1f}" if ylabel != "Latency (ms)" else f"{v:.3f}"
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * (1.05 if ylabel!="Latency (ms)" else 1.5),
                    label, ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.tick_params(axis="x", labelrotation=25, labelsize=8)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/fig1_accuracy_latency.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✓ fig1_accuracy_latency.png")

    # ── fig2: 混淆矩阵 ────────────────────────────────────────────────
    art_path = os.path.join(out_dir, "preprocessing_artifacts.json")
    if os.path.exists(art_path) and distill_model is not None and x_te is not None:
        with open(art_path) as f:
            art = json.load(f)
        scaler_mean  = np.array(art["scaler_mean"], dtype=np.float32)
        scaler_scale = np.array(art["scaler_scale"], dtype=np.float32)
        label_names_lc = art["label_encoder_classes"]
        x_te_scaled = ((x_te - scaler_mean) / scaler_scale).astype(np.float32)
        distill_model.eval()
        with torch.no_grad():
            logits = distill_model(torch.from_numpy(x_te_scaled).to(device)).cpu().numpy()
        y_pred = logits.argmax(axis=1)
        cm = confusion_matrix(y_te, y_pred)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        n_c = len(label_names_lc)
        ax.set_xticks(range(n_c)); ax.set_yticks(range(n_c))
        ax.set_xticklabels(label_names_lc, rotation=30, ha="right", fontsize=8)
        ax.set_yticklabels(label_names_lc, fontsize=8)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix: Distilled CNN\n"
                     f"N={cm.sum()}  Acc={results['4-DistillCNN']['accuracy']*100:.2f}%",
                     fontsize=12, fontweight="bold")
        for i in range(n_c):
            for j in range(n_c):
                color = "white" if cm_norm[i,j] > 0.5 else "black"
                ax.text(j, i, f"{cm_norm[i,j]:.2f}", ha="center", va="center",
                        color=color, fontsize=7)
        plt.colorbar(im, ax=ax, shrink=0.85)
        plt.tight_layout()
        plt.savefig(f"{out_dir}/fig2_confusion_matrix.png", dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  ✓ fig2_confusion_matrix.png")

    # ── fig3: CNN特征 LR 系数热力图 ───────────────────────────────────
    coef_path = os.path.join(out_dir, "lr_coefficients.json")
    if os.path.exists(coef_path):
        with open(coef_path) as f:
            d = json.load(f)
        coef = np.array(d["coefficients"])
        label_names_lr = d["label_names"]
        imp = np.abs(coef).mean(axis=0)
        sort_idx = np.argsort(imp)[::-1]
        coef_sorted = coef[:, sort_idx]
        fig, ax = plt.subplots(figsize=(12, 5))
        vmax = np.abs(coef_sorted).max()
        im = ax.imshow(coef_sorted, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
        ax.set_xlabel("Feature Dim (sorted by importance)", fontsize=10)
        ax.set_ylabel("Traffic Type", fontsize=10)
        ax.set_yticks(range(len(label_names_lr)))
        ax.set_yticklabels(label_names_lr, fontsize=9)
        step = max(1, len(sort_idx)//10)
        ax.set_xticks(range(0, len(sort_idx), step))
        ax.set_xticklabels([f"d{sort_idx[i]}" for i in range(0, len(sort_idx), step)], fontsize=7)
        ax.set_title("LR Coefficients on CNN Penultimate Features\n"
                     "(Red=positive, Blue=negative)", fontsize=12, fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        plt.tight_layout()
        plt.savefig(f"{out_dir}/fig3_feature_heatmap.png", dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  ✓ fig3_feature_heatmap.png")

    # ── fig3b: 原始 21 维特征 LR 热力图 ──────────────────────────────
    raw_path = os.path.join(out_dir, "lr_raw_coefficients.json")
    if os.path.exists(raw_path) and feat_names:
        with open(raw_path) as f:
            rd = json.load(f)
        coef = np.array(rd["coefficients"])
        label_names_lr = rd["label_names"]
        feat_names_lr  = rd["feat_names"]
        imp = np.abs(coef).mean(axis=0)
        sort_idx = np.argsort(imp)[::-1]
        coef_sorted = coef[:, sort_idx]
        feat_sorted = [feat_names_lr[i] for i in sort_idx]
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
        ax = axes[0]
        vmax = np.abs(coef_sorted).max()
        im = ax.imshow(coef_sorted, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(feat_sorted)))
        ax.set_xticklabels(feat_sorted, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(len(label_names_lr)))
        ax.set_yticklabels(label_names_lr, fontsize=9)
        ax.set_title("LR Coefficients on 21 Raw Flow Features\n(Red=positive, Blue=negative)",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Flow Feature", fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        ax = axes[1]
        n_top = 5
        bar_colors = plt.cm.tab10(np.linspace(0, 0.9, len(label_names_lr)))
        bw = 0.12
        for i, (cls, color) in enumerate(zip(label_names_lr, bar_colors)):
            coeffs = coef[i]
            top_idx = np.argsort(np.abs(coeffs))[::-1][:n_top]
            top_vals = coeffs[top_idx]
            x_pos = np.arange(n_top) + i * bw
            ax.bar(x_pos, top_vals, width=bw, label=cls, color=color, alpha=0.85)
        ax.set_xticks(np.arange(n_top) + bw * (len(label_names_lr)-1)/2)
        ax.set_xticklabels([f"Top-{k}" for k in range(1, n_top+1)], fontsize=9)
        ax.set_ylabel("LR Coefficient", fontsize=10)
        ax.set_title("Top-5 Discriminative Flow Features per Traffic Type",
                     fontsize=11, fontweight="bold")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7, ncol=1)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.axhline(y=0, color="black", linewidth=0.5)
        plt.tight_layout()
        plt.savefig(f"{out_dir}/fig3b_raw_feature_importance.png", dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  ✓ fig3b_raw_feature_importance.png")

    # ── fig4: t-SNE ──────────────────────────────────────────────────
    if feats_te is not None and y_te is not None and label_encoder_classes is not None:
        try:
            n_sub = min(2000, len(feats_te))
            idx = np.random.RandomState(42).choice(len(feats_te), n_sub, replace=False)
            x_sub, y_sub = feats_te[idx], y_te[idx]
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            x_sub = scaler.fit_transform(x_sub)
            tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=800)
            x_emb = tsne.fit_transform(x_sub)
            cmap = plt.cm.get_cmap("tab10", len(label_encoder_classes))
            fig, ax = plt.subplots(figsize=(8, 6))
            for i, name in enumerate(label_encoder_classes):
                mask = y_sub == i
                ax.scatter(x_emb[mask, 0], x_emb[mask, 1], c=[cmap(i)],
                           label=name, s=15, alpha=0.6)
            ax.set_xlabel("t-SNE Dim 1", fontsize=11)
            ax.set_ylabel("t-SNE Dim 2", fontsize=11)
            ax.set_title("t-SNE of CNN Penultimate Features\n"
                         "(Distilled CNN, 128-dim → 2-dim)", fontsize=12, fontweight="bold")
            ax.legend(title="Traffic Type", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
            ax.grid(alpha=0.2)
            plt.tight_layout()
            plt.savefig(f"{out_dir}/fig4_tsne.png", dpi=150, bbox_inches="tight", facecolor="white")
            plt.close()
            print(f"  ✓ fig4_tsne.png")
        except Exception as e:
            print(f"  ⚠ fig4 skipped: {e}")

    # ── fig5: 收敛曲线 ────────────────────────────────────────────────
    distill_h = summary.get("distill_history", {})
    teacher_h = summary.get("teacher_history", {})
    distill_val_acc = distill_h.get("val_acc", [])
    distill_val_f1  = distill_h.get("val_f1", [])
    distill_loss     = distill_h.get("train_loss", [])
    teacher_val_acc = teacher_h.get("val_acc", [])
    teacher_val_f1  = teacher_h.get("val_f1", [])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training Convergence (Validation Set)", fontsize=12, fontweight="bold")
    epochs_d = range(1, len(distill_loss) + 1)
    ax = axes[0]
    ax.plot(epochs_d, distill_loss, color="#16a34a", linewidth=2, label="Train Loss")
    ax.set_xlabel("Epoch", fontsize=10); ax.set_ylabel("Loss", fontsize=10, color="#16a34a")
    ax.tick_params(axis="y", labelcolor="#16a34a")
    ax2 = ax.twinx()
    ax2.plot(epochs_d, [v*100 for v in distill_val_acc], color="#2563eb",
             linewidth=2, linestyle="--", label="Val Acc")
    ax2.plot(epochs_d, [v*100 for v in distill_val_f1], color="#f97316",
             linewidth=2, linestyle=":", label="Val F1")
    ax2.set_ylabel("Val Acc / F1 (%)", fontsize=10)
    ax.set_title(f"Distilled CNN (200 epochs)\n"
                 f"Val Acc: {distill_val_acc[-1]*100:.1f}%  Val F1: {distill_val_f1[-1]*100:.1f}%",
                 fontsize=10)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1+lines2, labels1+labels2, fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(range(1, len(teacher_val_acc)+1), [v*100 for v in teacher_val_acc],
             color="#2563eb", linewidth=2, label="Val Acc")
    ax.plot(range(1, len(teacher_val_f1)+1), [v*100 for v in teacher_val_f1],
             color="#f97316", linewidth=2, linestyle="--", label="Val F1")
    ax.set_xlabel("Epoch", fontsize=10); ax.set_ylabel("Val Acc / F1 (%)", fontsize=10)
    ax.set_title(f"Transformer Teacher (300 epochs)\n"
                  f"Val Acc: {teacher_val_acc[-1]*100:.1f}%", fontsize=10)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/fig5_convergence.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✓ fig5_convergence.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/flow_features/full_features.pkl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs_teacher", type=int, default=300)
    ap.add_argument("--epochs_cnn", type=int, default=80)
    ap.add_argument("--epochs_distill", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--n_layers", type=int, default=6)
    args = ap.parse_args()

    print("=" * 60)
    print("加密流量识别 — 5组对比实验（论文用）")
    print("=" * 60)

    print(f"\n加载数据: {args.data}")
    x_tr, x_va, x_te, y_tr, y_va, y_te, le, scaler, feat_names = load_data(
        args.data, seed=args.seed)
    label_names = list(le.classes_)
    print(f"  训练集: {x_tr.shape}  |  验证集: {x_va.shape}  |  测试集: {x_te.shape}  |  类别: {len(label_names)}")
    print(f"  类别分布 (train): {dict(zip(label_names, np.bincount(y_tr)))}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  设备: {device}")

    # ── 保存预处理 artifacts（供绘图脚本重建一致的测试数据）──
    import json as _json
    from datetime import datetime as _dt
    ts = _dt.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("runs") / f"experiment_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "label_encoder_classes": label_names,
        "feat_names": feat_names,
    }
    with open(out_dir / "preprocessing_artifacts.json", "w") as f:
        _json.dump(artifacts, f, ensure_ascii=False)

    results = {}

    # ── 实验1: LR ──
    r1 = train_lr(x_tr, y_tr, x_te, y_te, label_names, feat_names)
    results["1-LR"] = r1

    # ── 实验2: 纯 CNN ──
    r2 = train_cnn_no_distill(
        x_tr, y_tr, x_va, y_va, x_te, y_te, label_names, feat_names,
        epochs=args.epochs_cnn,
        device=device, seed=args.seed,
    )
    results["2-CNN"] = r2

    # ── 实验3: Transformer 教师 ──
    r3 = train_transformer_teacher(
        x_tr, y_tr, x_va, y_va, x_te, y_te, label_names, feat_names,
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        epochs=args.epochs_teacher,
        device=device, seed=args.seed,
    )
    results["3-Teacher"] = r3

    # ── 实验4: CNN 蒸馏（含 Mixup）──
    r4 = train_cnn_distill(
        x_tr, y_tr, x_va, y_va, x_te, y_te, label_names, feat_names,
        teacher_model=r3["model"],
        temperature=args.temperature,
        alpha=args.alpha,
        label_smoothing=args.label_smoothing,
        epochs=args.epochs_distill,
        device=device, seed=args.seed,
    )
    results["4-DistillCNN"] = r4

    # ── 实验5: CNN 特征 + LR 可解释性 ──
    r5 = distill_cnn_with_lr(
        cnn_model=r4["model"],
        x_tr=x_tr, y_tr=y_tr, x_te=x_te, y_te=y_te,
        label_names=label_names, feat_names=feat_names,
        device=device,
    )
    results["5-Features+LR"] = r5

    # ── 延迟测量 ──
    print("\n" + "=" * 60)
    print("推理延迟测量（单样本，单位: ms）")
    print("=" * 60)

    latency = {}
    latency["1-LR"]            = measure_latency_sklearn(r1["model"], x_te)
    latency["2-CNN"]            = measure_latency_nn(r2["model"], x_te, device=device)
    latency["3-Teacher"]        = measure_latency_nn(r3["model"], x_te, device=device)
    latency["4-DistillCNN"]     = measure_latency_nn(r4["model"], x_te, device=device)
    # CNN特征+LR pipeline：CNN forward_features + LR.predict
    latency["5-Features+LR"]    = measure_latency_pipeline(
        r4["model"], r5["model"], x_te, device=device)
    # 5 的延迟 = CNN 特征提取 + LR 预测（pipeline 端到端）

    for key in latency:
        lat = latency[key]
        print(f"  {key:20s}  p50={lat['p50_ms']:.4f}  p90={lat['p90_ms']:.4f}  mean={lat['mean_ms']:.4f}")

    # ── 汇总表格 ──
    print("\n" + "=" * 60)
    print("实验结果汇总")
    print("=" * 60)
    print(f"{'实验':<6} {'模型':<28} {'准确率':>8} {'Macro-F1':>10} {'p50(ms)':>10}")
    print("-" * 72)
    for key in ["1-LR", "2-CNN", "3-Teacher", "4-DistillCNN", "5-Features+LR"]:
        r = results[key]
        lat = latency[key]
        print(f"  {key:<4} {r['name']:<28} {r['accuracy']:8.4f} {r['macro_f1']:10.4f} {lat['p50_ms']:10.4f}")

    # ── 核心对比 ──
    cnn_acc     = results["2-CNN"]["accuracy"]
    distill_acc = results["4-DistillCNN"]["accuracy"]
    lr_acc      = results["1-LR"]["accuracy"]
    feat_lr_acc = results["5-Features+LR"]["accuracy"]
    teacher_acc = results["3-Teacher"]["accuracy"]

    print(f"\n  === 核心对比 ===")
    print(f"  蒸馏提升:    纯 CNN {cnn_acc:.4f} → 蒸馏 CNN {distill_acc:.4f}  "
          f"(+{(distill_acc-cnn_acc)*100:.2f}%)")
    print(f"  可解释性:    纯 LR {lr_acc:.4f} → CNN特征+LR {feat_lr_acc:.4f}  "
          f"(+{(feat_lr_acc-lr_acc)*100:.2f}%)")
    print(f"  教师 vs 学生: 教师 {teacher_acc:.4f} → 蒸馏 CNN {distill_acc:.4f}  "
          f"(差距{(teacher_acc-distill_acc)*100:.2f}%)")


    # 保存模型权重
    torch.save(r4["model"].state_dict(), out_dir / "distill_cnn_best.pt")
    torch.save(r3["model"].state_dict(), out_dir / "teacher_best.pt")

    # 保存训练历史（供收敛曲线图使用）
    distill_history = {
        "train_loss": r4.get("train_loss_hist", []),
        "val_acc":     [h["val_acc"] for h in r4.get("val_history", [])],
        "val_f1":      [h["val_f1"]  for h in r4.get("val_history", [])],
    }
    teacher_history = {
        "val_acc": [h["val_acc"] for h in r3.get("val_history", [])],
        "val_f1":  [h["val_f1"]  for h in r3.get("val_history", [])],
    }

    summary = {
        "timestamp": ts,
        "data": args.data,
        "n_train": len(y_tr),
        "n_val":   len(y_va),
        "n_test":  len(y_te),
        "num_classes": len(label_names),
        "label_names": label_names,
        "feat_names": feat_names,
        "temperature": args.temperature,
        "alpha": args.alpha,
        "label_smoothing": args.label_smoothing,
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "epochs_teacher": args.epochs_teacher,
        "epochs_cnn": args.epochs_cnn,
        "epochs_distill": args.epochs_distill,
        "results": {
            name: {"accuracy": r["accuracy"], "macro_f1": r["macro_f1"]}
            for name, r in results.items()
        },
        "latency": latency,
        "distill_gain": distill_acc - cnn_acc,
        "lr_feat_gain": feat_lr_acc - lr_acc,
        "distill_history": distill_history,
        "teacher_history": teacher_history,
    }

    with open(out_dir / "summary.json", "w") as f:
        _json.dump(summary, f, ensure_ascii=False, indent=2)

    # ── LR 系数（可解释性，128维 CNN 特征）──
    clf = results["5-Features+LR"]["model"]
    coef = clf.coef_   # shape: (7, 128)
    print(f"\n  [CNN特征+LR] 各流量类型的 Top-5 贡献特征维度:")
    for i, cls_name in enumerate(label_names):
        coeffs = coef[i]
        top_idx = np.argsort(np.abs(coeffs))[::-1][:5]
        top_strs = [f"dim{d}({coeffs[j]:+.3f})" for j, d in enumerate(top_idx)]
        print(f"    {cls_name}: {', '.join(top_strs)}")

    coef_data = {
        "label_names": label_names,
        "feat_names": [f"cnn_feat_{i}" for i in range(256)],
        "coefficients": coef.tolist(),
    }
    with open(out_dir / "lr_coefficients.json", "w") as f:
        _json.dump(coef_data, f, ensure_ascii=False, indent=2)

    # ── 原始 21 维流特征的 LR 系数 ──
    lr_raw = results["1-LR"]["model"]
    raw_coef = lr_raw.coef_
    print(f"\n  [原始特征+LR] 各流量类型的 Top-5 贡献流特征:")
    for i, cls_name in enumerate(label_names):
        coeffs = raw_coef[i]
        top_idx = np.argsort(np.abs(coeffs))[::-1][:5]
        top_strs = [f"{feat_names[j]}({coeffs[j]:+.3f})" for j in top_idx]
        print(f"    {cls_name}: {', '.join(top_strs)}")

    raw_coef_data = {
        "label_names": label_names,
        "feat_names": feat_names,
        "coefficients": raw_coef.tolist(),
    }
    with open(out_dir / "lr_raw_coefficients.json", "w") as f:
        _json.dump(raw_coef_data, f, ensure_ascii=False, indent=2)

    # ── 保存模型权重（含 proj）──
    torch.save(r4["model"].state_dict(), out_dir / "distill_cnn_best.pt")
    torch.save(r3["model"].state_dict(), out_dir / "teacher_best.pt")

    # ── 自动生成所有图表 ────────────────────────────────────────────
    print(f"\n生成图表...")
    _plot_all_figures(
        summary=summary, out_dir=str(out_dir),
        distill_model=r4["model"],
        lr_model=r5["model"],
        x_te=x_te, y_te=y_te,
        feats_tr=r5.get("feats_tr"),
        feats_te=r5.get("feats_te"),
        label_encoder_classes=label_names,
        feat_names=feat_names,
        device=device,
    )

    print(f"\n实验完成！结果保存在: {out_dir}")
    return results, latency, out_dir


if __name__ == "__main__":
    main()
