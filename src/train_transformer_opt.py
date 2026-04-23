"""
最优纯 Transformer 教师训练脚本
基于 TransECA-Net (2025) 论文发现：
  - 纯 Transformer 在 ISCX 数据集上达 95.30% accuracy
  - 关键配置：d_model=256, n_heads=4-8, n_layers=4, dropout=0.3, GELU, norm_first
  - 训练策略：AdamW + warmup + cosine, label_smoothing, class_weights
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import set_seed


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ─── 模型 ─────────────────────────────────────────────────────────────────────
class SinusoidalPosEmb(nn.Module):
    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class TransformerTeacher(nn.Module):
    """
    纯 Transformer 教师模型
    - 输入: (batch, seq_len, 3) = [packet_length, direction, inter_arrival_time]
    - 输入投影: 3 -> d_model
    - n_layers 个 TransformerEncoderLayer (norm_first, GELU)
    - 平均池化 + 分类头
    """
    def __init__(
        self,
        seq_len: int = 100,
        n_features: int = 3,
        num_classes: int = 6,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.3,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.d_model = d_model

        self.input_proj = nn.Linear(n_features, d_model)

        self.pos_emb = SinusoidalPosEmb(d_model, max_len=seq_len + 10)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,          # 残差在 norm 之前（更稳定）
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        # x: (B, seq, 3)
        x = self.input_proj(x)           # (B, seq, d_model)
        x = self.pos_emb(x)
        x = self.dropout(x)
        x = self.transformer(x)          # (B, seq, d_model)
        x = self.norm(x)
        x = x.mean(dim=1)                # (B, d_model)
        return self.head(x)


# ─── 数据 ─────────────────────────────────────────────────────────────────────
def load_packet_data(pkl_path: str, seq_len: int = 100, min_packets: int = 10, seed: int = 42):
    import pandas as pd

    df = pd.read_pickle(pkl_path)
    df = df[df["num_packets"] >= min_packets].reset_index(drop=True)
    print(f"  有效样本: {len(df)} (≥ {min_packets} 包)")

    le = LabelEncoder()
    y = le.fit_transform(df["label"].values)

    n = len(df)
    X = np.zeros((n, seq_len, 3), dtype=np.float32)
    for i, (_, row) in enumerate(df.iterrows()):
        m = min(len(row["packet_lengths"]), seq_len)
        X[i, :m, 0] = row["packet_lengths"][:m]
        X[i, :m, 1] = row["directions"][:m]
        X[i, :m, 2] = row["inter_arrival_times"][:m]

    # 每个通道独立标准化
    for ch in range(3):
        mean, std = X[:, :, ch].mean(), X[:, :, ch].std() + 1e-9
        X[:, :, ch] = (X[:, :, ch] - mean) / std

    # 60/20/20 划分
    can_strat = (np.bincount(y) >= 3).all()
    kw = dict(test_size=0.2, random_state=seed)
    if can_strat:
        kw["stratify"] = y
    X_tv, X_te, y_tv, y_te = train_test_split(X, y, **kw)

    kw2 = dict(test_size=0.25, random_state=seed)
    if can_strat:
        kw2["stratify"] = y_tv
    X_tr, X_va, y_tr, y_va = train_test_split(X_tv, y_tv, **kw2)

    print(f"  划分: train={len(X_tr)} val={len(X_va)} test={len(X_te)}")
    print(f"  类别: {list(le.classes_)}")
    for i, name in enumerate(le.classes_):
        print(f"    {name:<15} train={np.sum(y_tr==i):>5}  val={np.sum(y_va==i):>5}  test={np.sum(y_te==i):>5}")

    return dict(X_tr=X_tr, X_va=X_va, X_te=X_te, y_tr=y_tr, y_va=y_va, y_te=y_te, le=le)


# ─── 训练工具 ────────────────────────────────────────────────────────────────
class LabelSmoothingLoss(nn.Module):
    def __init__(self, n_classes: int, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        self.n_classes = n_classes

    def forward(self, pred, target):
        log_prob = F.log_softmax(pred, dim=-1)
        true_dist = torch.zeros_like(log_prob)
        true_dist.fill_(self.smoothing / self.n_classes)
        true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing + self.smoothing / self.n_classes)
        return (-true_dist * log_prob).sum(dim=-1).mean()


def cosine_warmup_lr(opt, warmup_steps: int, total_steps: int, min_lr: float = 1e-6):
    def step(epoch):
        if epoch < warmup_steps:
            return float(epoch) / max(1, warmup_steps)
        progress = (epoch - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=step)


def train_epoch(model, loader, loss_fn, opt, device, grad_clip: float = 1.0):
    model.train()
    total_loss, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        total_loss += float(loss.item()) * xb.size(0)
        n += xb.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        preds.append(model(xb).argmax(1).cpu().numpy())
        labels.append(yb.numpy())
    return np.concatenate(preds), np.concatenate(labels)


# ─── 主训练 ────────────────────────────────────────────────────────────────
def train(
    data,
    num_classes,
    label_names,
    run_dir: str,
    # 模型
    d_model: int = 256,
    n_heads: int = 8,
    n_layers: int = 4,
    dropout: float = 0.3,
    ff_mult: int = 4,
    # 训练
    batch_size: int = 128,
    epochs: int = 200,
    patience: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    warmup_epochs: int = 10,
    label_smoothing: float = 0.1,
    # 数据
    seed: int = 42,
    force: bool = False,
):
    ckpt = Path(run_dir) / "artifacts" / "transformer_teacher.pt"
    if ckpt.exists() and not force:
        print("[Transformer] checkpoint 存在，跳过")
        return

    set_seed(seed)
    device = get_device()
    print(f"Device: {device}")

    # class weights
    class_counts = np.bincount(data["y_tr"])
    total = len(data["y_tr"])
    weights = total / (num_classes * class_counts + 1e-8)
    weights = weights / weights.sum() * num_classes
    class_weights = torch.from_numpy(weights).float().to(device)
    print(f"  Class weights: {[f'{w:.2f}' for w in weights]}")

    model = TransformerTeacher(
        seq_len=data["X_tr"].shape[1],
        n_features=3,
        num_classes=num_classes,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
        ff_mult=ff_mult,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  模型参数: {n_params:,}")

    # data loaders
    ds_tr = TensorDataset(torch.from_numpy(data["X_tr"].astype(np.float32)), torch.from_numpy(data["y_tr"]).long())
    ds_va = TensorDataset(torch.from_numpy(data["X_va"].astype(np.float32)), torch.from_numpy(data["y_va"]).long())
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=(device == "cuda"))
    dl_va = DataLoader(ds_va, batch_size=2048, shuffle=False, num_workers=0)

    loss_fn = LabelSmoothingLoss(num_classes, smoothing=label_smoothing)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = cosine_warmup_lr(opt, warmup_epochs, epochs)

    best_f1, best_acc, best_state, wait, history = 0.0, 0.0, None, 0, []
    t0 = time.time()

    for epoch in tqdm(range(epochs), desc="[Transformer]"):
        loss_tr = train_epoch(model, dl_tr, loss_fn, opt, device)
        scheduler.step()

        preds_va, y_va = evaluate(model, dl_va, device)
        acc_va = accuracy_score(y_va, preds_va)
        f1_va = f1_score(y_va, preds_va, average="macro", zero_division=0)

        improved = f1_va > best_f1
        if improved:
            best_f1, best_acc = f1_va, acc_va
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        history.append({"epoch": epoch + 1, "loss": loss_tr, "acc": acc_va, "f1": f1_va})

        if (epoch + 1) % 20 == 0 or improved:
            print(
                f"  ep={epoch+1:3d}  loss={loss_tr:.4f}  "
                f"acc={acc_va:.4f}  f1={f1_va:.4f}  lr={opt.param_groups[0]['lr']:.2e} {'*best*' if improved else ''}"
            )

        if wait >= patience:
            print(f"  Early stop at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    elapsed = time.time() - t0
    print(f"  Best: val_acc={best_acc:.4f}  val_f1={best_f1:.4f}  ({elapsed:.0f}s)")

    # 保存
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    (Path(run_dir) / "artifacts").mkdir(exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "num_classes": num_classes,
            "seq_len": data["X_tr"].shape[1],
            "n_features": 3,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "dropout": dropout,
            "ff_mult": ff_mult,
            "best_val_f1": float(best_f1),
            "best_val_acc": float(best_acc),
            "label_names": label_names,
        },
        ckpt,
    )

    # 测试评估
    ds_te = TensorDataset(torch.from_numpy(data["X_te"].astype(np.float32)), torch.from_numpy(data["y_te"]).long())
    dl_te = DataLoader(ds_te, batch_size=2048, shuffle=False, num_workers=0)
    preds_te, y_te = evaluate(model, dl_te, device)
    acc_te = accuracy_score(y_te, preds_te)
    f1_te = f1_score(y_te, preds_te, average="macro", zero_division=0)
    report = classification_report(y_te, preds_te, labels=list(range(len(label_names))),
                                 target_names=label_names, output_dict=True, zero_division=0)

    print(f"\n[Test Set]")
    print(f"  Accuracy: {acc_te*100:.2f}%")
    print(f"  Macro F1: {f1_te:.4f}")
    print()
    for name in label_names:
        m = report[name]
        print(f"  {name:<15} P={m['precision']*100:6.2f}%  R={m['recall']*100:6.2f}%  F1={m['f1-score']:6.4f}  (n={int(m['support'])})")

    results = {
        "accuracy": float(acc_te),
        "macro_f1": float(f1_te),
        "per_class": {
            name: {"precision": float(report[name]["precision"]),
                   "recall": float(report[name]["recall"]),
                   "f1": float(report[name]["f1-score"]),
                   "support": int(report[name]["support"])}
            for name in label_names
        },
        "best_val_acc": float(best_acc),
        "best_val_f1": float(best_f1),
        "epochs": len(history),
        "elapsed": round(elapsed, 1),
        "n_params": n_params,
        "history": history,
    }

    with open(Path(run_dir) / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(Path(run_dir) / "training_summary.json", "w") as f:
        json.dump({"best_val_acc": float(best_acc), "best_val_f1": float(best_f1),
                   "total_epochs": len(history), "elapsed": round(elapsed, 1)}, f, indent=2)

    return model, best_acc, best_f1


# ─── 主入口 ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--flows", default="data/packet_sequences/all_flows.pkl")
    ap.add_argument("--run_dir", default="runs/transformer_opt")
    ap.add_argument("--seq_len", type=int, default=100)
    ap.add_argument("--min_packets", type=int, default=10)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--ff_mult", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--warmup_epochs", type=int, default=10)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    print(f"\n{'='*60}")
    print(f"纯 Transformer 教师训练")
    print(f"  数据: {args.flows}  (seq_len={args.seq_len}, min_packets={args.min_packets})")
    print(f"  模型: d_model={args.d_model}, n_heads={args.n_heads}, n_layers={args.n_layers}")
    print(f"  训练: batch={args.batch_size}, lr={args.lr}, wd={args.weight_decay}")
    print(f"  正则: dropout={args.dropout}, label_smoothing={args.label_smoothing}")
    print(f"{'='*60}\n")

    data = load_packet_data(args.flows, seq_len=args.seq_len, min_packets=args.min_packets, seed=args.seed)
    num_classes = len(data["le"].classes_)
    label_names = list(data["le"].classes_)

    train(
        data=data,
        num_classes=num_classes,
        label_names=label_names,
        run_dir=args.run_dir,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        ff_mult=args.ff_mult,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        label_smoothing=args.label_smoothing,
        seed=args.seed,
        force=args.force,
    )
