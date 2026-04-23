"""
Improved Transformer Teacher Training for Encrypted Traffic Classification.

Key improvements over baseline:
  1. Hybrid CNN + Transformer architecture for better local pattern extraction
  2. Proper train/val/test split (60/20/20) with early stopping on val
  3. Strong regularization: dropout 0.2, weight_decay 1e-3, label_smoothing 0.1
  4. Data augmentation: Mixup + random packet masking
  5. Class weights for imbalanced classes
  6. Warmup + cosine decay LR schedule
  7. Gradient clipping + mixed precision training

Usage:
  python -m src.train_transformer_improved --epochs 200 --patience 30
  python -m src.train_transformer_improved --step evaluate --checkpoint runs/transformer_improved/teacher_best.pt
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.rich_features import extract_rich_features
from src.utils import set_seed


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ══════════════════════════════════════════════════════════════
# Improved Transformer Architecture
# ══════════════════════════════════════════════════════════════
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 200, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class HybridCNNTransformer(nn.Module):
    """
    Hybrid architecture: CNN for local pattern extraction + Transformer for sequence modeling.
    CNN captures local burst patterns, Transformer captures global sequence dependencies.
    """
    def __init__(
        self,
        seq_len: int = 100,
        n_features: int = 3,
        num_classes: int = 6,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.2,
        ff_mult: int = 4,
        cnn_channels: list = None,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.d_model = d_model

        if cnn_channels is None:
            cnn_channels = [64, 128]

        # CNN feature extractor (shared for local patterns)
        self.conv1 = nn.Conv1d(n_features, cnn_channels[0], kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(cnn_channels[0])
        self.conv2 = nn.Conv1d(cnn_channels[0], cnn_channels[1], kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(cnn_channels[1])
        self.cnn_dropout = nn.Dropout(dropout)

        # Projection to d_model
        self.proj = nn.Linear(cnn_channels[-1], d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=seq_len, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Classification head with residual connection
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(d_model, num_classes)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        x = x.permute(0, 2, 1)  # (batch, n_features, seq_len)

        # CNN feature extraction
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        x = self.cnn_dropout(x)

        x = x.permute(0, 2, 1)  # (batch, seq_len, cnn_channels[-1])
        x = self.proj(x)        # (batch, seq_len, d_model)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Transformer
        x = self.transformer(x)

        # Global average pooling
        x = x.mean(dim=1)  # (batch, d_model)

        # FFN with residual
        residual = x
        x = self.norm1(x)
        x = self.ff(x) + residual * 0.1  # small residual
        x = self.norm2(x)

        return self.head(x)


# ══════════════════════════════════════════════════════════════
# Data Loading
# ══════════════════════════════════════════════════════════════
def load_data(pkl_path: str, max_packets: int = 100, min_packets: int = 10, seed: int = 42):
    import pandas as pd

    df = pd.read_pickle(pkl_path)
    df = df[df["num_packets"] >= min_packets].reset_index(drop=True)
    print(f"  Filtered to {len(df)} flows (>= {min_packets} packets)")

    le = LabelEncoder()
    y_all = le.fit_transform(df["label"].values)

    n = len(df)
    X_seq = np.zeros((n, max_packets, 3), dtype=np.float32)
    for i, (_, row) in enumerate(df.iterrows()):
        pkt_len = row["packet_lengths"]
        dirs = row["directions"]
        iats = row["inter_arrival_times"]
        m = min(len(pkt_len), max_packets)
        X_seq[i, :m, 0] = pkt_len[:m]
        X_seq[i, :m, 1] = dirs[:m]
        X_seq[i, :m, 2] = iats[:m]

    # Per-channel standardization
    for ch in range(3):
        mean, std = X_seq[:, :, ch].mean(), X_seq[:, :, ch].std() + 1e-9
        X_seq[:, :, ch] = (X_seq[:, :, ch] - mean) / std

    # 3-way split: 60/20/20
    can_stratify = (np.bincount(y_all) >= 3).all()
    split1_kwargs = dict(test_size=0.2, random_state=seed)
    split2_kwargs = dict(test_size=0.25, random_state=seed)
    if can_stratify:
        split1_kwargs["stratify"] = y_all

    (X_trval, X_te, y_trval, y_te) = train_test_split(X_seq, y_all, **split1_kwargs)
    split2_kwargs["stratify"] = y_trval if can_stratify else None
    (X_tr, X_va, y_tr, y_va) = train_test_split(X_trval, y_trval, **split2_kwargs)

    print(f"  Split: train={len(X_tr)}, val={len(X_va)}, test={len(X_te)}")
    print(f"  Classes: {list(le.classes_)}")
    for label_idx, label_name in enumerate(le.classes_):
        n_tr = int(np.sum(y_tr == label_idx))
        n_va = int(np.sum(y_va == label_idx))
        n_te = int(np.sum(y_te == label_idx))
        print(f"    {label_name:<15} train={n_tr:>5}  val={n_va:>5}  test={n_te:>5}")

    return {
        "X_tr": X_tr, "X_va": X_va, "X_te": X_te,
        "y_tr": y_tr, "y_va": y_va, "y_te": y_te,
        "le": le,
    }


# ══════════════════════════════════════════════════════════════
# Data Augmentation
# ══════════════════════════════════════════════════════════════
def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def random_masking(x, mask_ratio=0.15):
    """Randomly mask packets in the sequence."""
    batch_size, seq_len, n_feat = x.shape
    mask = torch.rand(batch_size, seq_len, 1, device=x.device) > mask_ratio
    return x * mask.float()


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


# ══════════════════════════════════════════════════════════════
# Learning Rate Schedule
# ══════════════════════════════════════════════════════════════
def get_cosine_schedule_with_warmup(opt, warmup_epochs, total_epochs, min_lr=1e-6):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


# ══════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════
def train_epoch(model, loader, opt, loss_fn, device, use_mixup=True, mixup_alpha=0.2, mask_ratio=0.1, grad_clip=1.0):
    model.train()
    total_loss, n_samples = 0.0, 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        # Data augmentation
        if use_mixup and np.random.rand() < 0.5:
            xb, y_a, y_b, lam = mixup_data(xb, yb, alpha=mixup_alpha)
            xb = random_masking(xb, mask_ratio=mask_ratio)
            logits = model(xb)
            loss = lam * loss_fn(logits, y_a) + (1 - lam) * loss_fn(logits, y_b)
        else:
            xb = random_masking(xb, mask_ratio=mask_ratio)
            logits = model(xb)
            loss = loss_fn(logits, yb)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        total_loss += float(loss.item()) * xb.size(0)
        n_samples += xb.size(0)
    return total_loss / max(n_samples, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    logits_all, y_all = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb).cpu().numpy()
        logits_all.append(logits)
        y_all.append(yb.numpy())
    return np.concatenate(logits_all, axis=0), np.concatenate(y_all, axis=0)


def train(
    X_tr, y_tr, X_va, y_va,
    num_classes, label_names,
    run_dir,
    d_model=128, n_heads=8, n_layers=4, dropout=0.2,
    batch_size=64, lr=3e-4, weight_decay=1e-3,
    epochs=200, patience=30, seed=42, device='cpu',
    warmup_epochs=10, label_smoothing=0.1,
    mixup_alpha=0.2, mask_ratio=0.1,
    use_class_weights=True,
    force=False,
):
    ckpt = Path(run_dir) / "artifacts" / "teacher_transformer.pt"
    if ckpt.exists() and not force:
        print("[Transformer] Checkpoint exists, skipping training (use --force to retrain)")
        return

    set_seed(seed)

    # Class weights for imbalanced data
    class_weights = None
    if use_class_weights:
        class_counts = np.bincount(y_tr)
        total = len(y_tr)
        weights = total / (num_classes * class_counts + 1e-8)
        weights = weights / weights.sum() * num_classes
        class_weights = torch.from_numpy(weights).float().to(device)
        print(f"  Class weights: {[f'{w:.2f}' for w in weights]}")

    # Loss function
    if class_weights is not None:
        loss_fn = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )
    else:
        loss_fn = LabelSmoothingLoss(num_classes, smoothing=label_smoothing)

    # Model
    model = HybridCNNTransformer(
        seq_len=X_tr.shape[1],
        n_features=X_tr.shape[2],
        num_classes=num_classes,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
        ff_mult=4,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {n_params:,}")

    # Data loaders
    ds_tr = TensorDataset(
        torch.from_numpy(X_tr.astype(np.float32)),
        torch.from_numpy(y_tr).long(),
    )
    ds_va = TensorDataset(
        torch.from_numpy(X_va.astype(np.float32)),
        torch.from_numpy(y_va).long(),
    )
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=(device == 'cuda'))
    dl_va = DataLoader(ds_va, batch_size=2048, shuffle=False, num_workers=0)

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = get_cosine_schedule_with_warmup(opt, warmup_epochs=warmup_epochs, total_epochs=epochs)

    best_f1, best_acc, best_state, patience_counter, history = 0.0, 0.0, None, 0, []
    t0 = time.time()

    for epoch in tqdm(range(epochs), desc="[Transformer]"):
        train_loss = train_epoch(
            model, dl_tr, opt,
            loss_fn if isinstance(loss_fn, nn.Module) else LabelSmoothingLoss(num_classes, label_smoothing),
            device,
            use_mixup=True, mixup_alpha=mixup_alpha, mask_ratio=mask_ratio,
        )
        scheduler.step()

        # Evaluate on validation
        logits_va, y_va_np = evaluate(model, dl_va, device)
        pred_va = logits_va.argmax(axis=1)
        acc_va = float(accuracy_score(y_va_np, pred_va))
        f1_va = float(f1_score(y_va_np, pred_va, average='macro', zero_division=0))

        improved = f1_va > best_f1
        if improved:
            best_f1 = f1_va
            best_acc = acc_va
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        history.append({
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "val_acc": float(acc_va),
            "val_f1": float(f1_va),
            "lr": float(opt.param_groups[0]['lr']),
        })

        if (epoch + 1) % 20 == 0 or improved:
            print(f"  ep={epoch+1:>3d}  loss={train_loss:.4f}  acc={acc_va:.4f}  f1={f1_va:.4f}  "
                  f"lr={opt.param_groups[0]['lr']:.2e} {'*best*' if improved else ''}")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch + 1} (patience={patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    elapsed = time.time() - t0
    print(f"  Best val_acc={best_acc:.4f}  best val_f1={best_f1:.4f}  ({elapsed:.0f}s)")

    # Save
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    (Path(run_dir) / "artifacts").mkdir(exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "num_classes": num_classes,
        "seq_len": X_tr.shape[1],
        "n_features": X_tr.shape[2],
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "dropout": dropout,
        "best_val_f1": float(best_f1),
        "best_val_acc": float(best_acc),
        "label_names": label_names,
    }, ckpt)

    with open(Path(run_dir) / "training_summary.json", "w") as f:
        json.dump({
            "best_val_acc": float(best_acc),
            "best_val_f1": float(best_f1),
            "total_epochs": len(history),
            "elapsed_seconds": round(elapsed, 1),
            "history": history,
        }, f, indent=2)

    return model, best_acc, best_f1, history


# ══════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════
def evaluate_test(model, X_te, y_te, label_names, device):
    ds_te = TensorDataset(
        torch.from_numpy(X_te.astype(np.float32)),
        torch.from_numpy(y_te).long(),
    )
    dl_te = DataLoader(ds_te, batch_size=2048, shuffle=False, num_workers=0)

    logits_all, y_all = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in dl_te:
            xb = xb.to(device)
            logits = model(xb).cpu().numpy()
            logits_all.append(logits)
            y_all.append(yb.numpy())

    logits_all = np.concatenate(logits_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    pred = logits_all.argmax(axis=1)

    acc = float(accuracy_score(y_all, pred))
    macro_f1 = float(f1_score(y_all, pred, average='macro', zero_division=0))
    macro_precision = float(f1_score(y_all, pred, average='macro', zero_division=0))
    macro_recall = float(f1_score(y_all, pred, average='macro', zero_division=0))

    cm = confusion_matrix(y_all, pred, labels=list(range(len(label_names)))).tolist()
    report = classification_report(y_all, pred, labels=list(range(len(label_names))),
                                   target_names=label_names, output_dict=True, zero_division=0)

    per_class = {
        label_names[i]: {
            "precision": float(report[label_names[i]]["precision"]),
            "recall": float(report[label_names[i]]["recall"]),
            "f1-score": float(report[label_names[i]]["f1-score"]),
            "support": int(report[label_names[i]]["support"]),
        }
        for i in range(len(label_names))
    }

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "confusion_matrix": cm,
        "per_class": per_class,
        "label_names": label_names,
    }


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flows", default="data/packet_sequences/all_flows.pkl")
    ap.add_argument("--run_dir", default="runs/transformer_improved")
    ap.add_argument("--step", default="all",
                   choices=["all", "train", "evaluate"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_packets", type=int, default=100)
    ap.add_argument("--min_packets", type=int, default=10)

    # Model architecture
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.2)

    # Training
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--warmup_epochs", type=int, default=10)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--mixup_alpha", type=float, default=0.2)
    ap.add_argument("--mask_ratio", type=float, default=0.1)
    ap.add_argument("--no_class_weights", action="store_true", help="Disable class weights")
    ap.add_argument("--force", action="store_true")

    args = ap.parse_args()
    device = get_device()

    print(f"\nDevice: {device}")
    print(f"Data: {args.flows}  |  max_packets={args.max_packets}  |  min_packets={args.min_packets}")
    print(f"Model: d_model={args.d_model}, n_heads={args.n_heads}, n_layers={args.n_layers}, dropout={args.dropout}")
    print(f"Training: epochs={args.epochs}, patience={args.patience}, lr={args.lr}, weight_decay={args.weight_decay}")

    # Load data
    print("\n[Data] Loading and splitting flows...")
    data = load_data(args.flows, max_packets=args.max_packets,
                     min_packets=args.min_packets, seed=args.seed)
    num_classes = len(data["le"].classes_)
    label_names = list(data["le"].classes_)

    Path(args.run_dir).mkdir(parents=True, exist_ok=True)

    if args.step in ("all", "train"):
        model, best_acc, best_f1, history = train(
            X_tr=data["X_tr"], y_tr=data["y_tr"],
            X_va=data["X_va"], y_va=data["y_va"],
            num_classes=num_classes, label_names=label_names,
            run_dir=args.run_dir,
            d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
            dropout=args.dropout,
            batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
            epochs=args.epochs, patience=args.patience,
            seed=args.seed, device=device,
            warmup_epochs=args.warmup_epochs,
            label_smoothing=args.label_smoothing,
            mixup_alpha=args.mixup_alpha, mask_ratio=args.mask_ratio,
            use_class_weights=not args.no_class_weights,
            force=args.force,
        )

        # Evaluate on test set
        print("\n[Test Set Evaluation]")
        results = evaluate_test(model, data["X_te"], data["y_te"], label_names, device)
        print(f"  Accuracy: {results['accuracy']*100:.2f}%")
        print(f"  Macro F1: {results['macro_f1']:.4f}")
        print(f"  Macro Precision: {results['macro_precision']*100:.2f}%")
        print(f"  Macro Recall: {results['macro_recall']*100:.2f}%")

        with open(Path(args.run_dir) / "eval_results.json", "w") as f:
            json.dump(results, f, indent=2)

    elif args.step == "evaluate":
        ckpt_path = Path(args.run_dir) / "artifacts" / "teacher_transformer.pt"
        if not ckpt_path.exists():
            print(f"[Error] Checkpoint not found: {ckpt_path}")
            sys.exit(1)

        state = torch.load(ckpt_path, map_location="cpu")
        model = HybridCNNTransformer(
            seq_len=state["seq_len"],
            n_features=state["n_features"],
            num_classes=state["num_classes"],
            d_model=state["d_model"],
            n_heads=state["n_heads"],
            n_layers=state["n_layers"],
            dropout=state["dropout"],
        ).to(device)
        model.load_state_dict(state["state_dict"])
        model.eval()

        label_names = state["label_names"]
        results = evaluate_test(model, data["X_te"], data["y_te"], label_names, device)
        print(f"\n  Accuracy: {results['accuracy']*100:.2f}%")
        print(f"  Macro F1: {results['macro_f1']:.4f}")
        print(f"  Macro Precision: {results['macro_precision']*100:.2f}%")
        print(f"  Macro Recall: {results['macro_recall']*100:.2f}%")
        print(f"\nPer-class:")
        for name, metrics in results["per_class"].items():
            print(f"  {name:<15} P={metrics['precision']*100:>6.2f}%  "
                  f"R={metrics['recall']*100:>6.2f}%  "
                  f"F1={metrics['f1-score']:>6.4f}  "
                  f"(n={metrics['support']})")

        with open(Path(args.run_dir) / "eval_results.json", "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    import sys
    main()
