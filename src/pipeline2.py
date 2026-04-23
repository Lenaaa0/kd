"""
Unified pipeline: single data source (24,763 flow samples from ISCXVPN2016 PCAPs).

Architecture:
  - Teacher:  DNN trained on rich statistical features (49 dims) extracted from packet flows
  - CNN Student: 1D CNN trained on raw packet sequences (100 x 3)
  - LR Student:  Logistic Regression on the same rich features

Both teacher and student see the same flows, just in different representations.
Teacher provides soft labels for CNN distillation.

Usage:
  python -m src.pipeline2 --step all
  python -m src.pipeline2 --step teacher
  python -m src.pipeline2 --step cnn_baseline
  python -m src.pipeline2 --step cnn
  python -m src.pipeline2 --step evaluate
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.rich_features import extract_rich_features
from src.evaluation import compute_metrics, measure_latency
from src.models import CNNStudent, LRStudent
from src.models.teacher_dnn import DNNTeacher
from src.training import TrainerCNN
from src.utils import set_seed


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ─── Label Smoothing Loss ──────────────────────────────────────────────────────
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1, class_weights=None):
        super().__init__()
        self.smoothing = smoothing
        self.class_weights = class_weights

    def forward(self, pred, target):
        n_classes = pred.size(1)
        log_prob = F.log_softmax(pred, dim=-1)
        true_dist = torch.zeros_like(log_prob)
        true_dist.fill_(self.smoothing / n_classes)
        true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing + self.smoothing / n_classes)
        loss = (-true_dist * log_prob).sum(dim=-1)
        if self.class_weights is not None:
            loss = loss * self.class_weights[target].to(pred.device)
        return loss.mean()


# ─── Data Loading ─────────────────────────────────────────────────────────────
def load_flows(pcap_path: str, max_packets: int = 100, seed: int = 42):
    """
    Load flow data, split into train/test, extract features.
    Returns raw sequences and rich features for both splits.
    """
    import pandas as pd

    df = pd.read_pickle(pcap_path)

    # Filter flows with enough packets (at least 10)
    min_packets = 10
    df = df[df["num_packets"] >= min_packets].reset_index(drop=True)
    print(f"  Filtered to {len(df)} flows (>= {min_packets} packets)")

    # Encode labels
    le = LabelEncoder()
    y_all = le.fit_transform(df["label"].values)

    # ── Raw packet sequences (for CNN) ──────────────────────────────────────
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

    # ── Train/test split ───────────────────────────────────────────────────
    # Check if stratification is possible
    class_counts = np.bincount(y_all)
    can_stratify = (class_counts >= 2).all()

    split_kwargs = dict(test_size=0.2, random_state=seed)
    if can_stratify:
        split_kwargs["stratify"] = y_all

    (X_seq_tr, X_seq_te,
     y_tr, y_te) = train_test_split(X_seq, y_all, **split_kwargs)

    # ── Rich features (for Teacher and LR) ─────────────────────────────────
    print("  Extracting rich features for teacher/LR...")
    feat_tr = extract_rich_features(X_seq_tr, max_packets=max_packets)
    feat_te = extract_rich_features(X_seq_te, max_packets=max_packets)
    feat_tr = np.nan_to_num(feat_tr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    feat_te = np.nan_to_num(feat_te, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # Standardize features
    scaler = StandardScaler()
    feat_tr = scaler.fit_transform(feat_tr).astype(np.float32)
    feat_te = scaler.transform(feat_te).astype(np.float32)

    print(f"  Classes: {list(le.classes_)}")
    print(f"  Train: {len(y_tr)}, Test: {len(y_te)}")
    print(f"  Rich feature dim: {feat_tr.shape[1]}")
    print(f"  Sequence shape: {X_seq_tr.shape}")

    return {
        "feat_tr": feat_tr, "feat_te": feat_te,
        "seq_tr": X_seq_tr, "seq_te": X_seq_te,
        "y_tr": y_tr, "y_te": y_te,
        "le": le, "scaler": scaler,
    }


# ─── Teacher Training ─────────────────────────────────────────────────────────
def train_teacher(feat_tr, y_tr, feat_te, y_te, num_classes, run_dir,
                  hidden_sizes, dropout, label_smoothing,
                  batch_size, lr, epochs, seed, device, force,
                  use_class_weights=True):
    ckpt = Path(run_dir) / "artifacts" / "teacher.pt"
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    (Path(run_dir) / "artifacts").mkdir(exist_ok=True)
    if ckpt.exists() and not force:
        print("[Teacher] Checkpoint exists, skipping")
        return

    set_seed(seed)
    in_dim = feat_tr.shape[1]
    model = DNNTeacher(in_dim=in_dim, num_classes=num_classes,
                      hidden_sizes=hidden_sizes, dropout=dropout).to(device)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(feat_tr), torch.from_numpy(y_tr).long()),
        batch_size=batch_size, shuffle=True, num_workers=0,
        pin_memory=(device == "cuda"),
    )
    X_te_t = torch.from_numpy(feat_te).float().to(device)
    # Class weights for imbalanced data
    class_weights = None
    if use_class_weights:
        class_counts = np.bincount(y_tr)
        total = len(y_tr)
        weights = total / (num_classes * class_counts + 1e-8)
        weights = weights / weights.sum() * num_classes  # Normalize
        class_weights = torch.from_numpy(weights).float().to(device)
        print(f"  Class weights: {[f'{w:.2f}' for w in weights]}")

    loss_fn = LabelSmoothingCrossEntropy(smoothing=label_smoothing, class_weights=class_weights)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    best_acc, best_state, history = 0.0, None, []
    t0 = time.time()

    for epoch in tqdm(range(epochs), desc="[Teacher]"):
        model.train()
        total_loss, n_samples = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += float(loss.item()) * xb.size(0)
            n_samples += xb.size(0)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            preds = model(X_te_t).argmax(1).cpu().numpy()
        acc = accuracy_score(y_te, preds)
        improved = acc > best_acc
        if improved:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        history.append({"epoch": epoch + 1, "loss": total_loss / max(n_samples, 1), "acc": float(acc)})
        if (epoch + 1) % 50 == 0 or improved:
            print(f"  ep={epoch+1:>4d}  loss={total_loss/max(n_samples,1):.4f}  acc={acc:.4f} {'*best*' if improved else ''}")

    model.load_state_dict(best_state)
    elapsed = time.time() - t0
    torch.save({
        "state_dict": model.state_dict(),
        "num_classes": num_classes,
        "in_dim": in_dim,
        "hidden_sizes": hidden_sizes,
        "dropout": dropout,
        "best_val_acc": best_acc,
    }, ckpt)
    with open(Path(run_dir) / "training_summary.json", "w") as f:
        json.dump({"best_val_acc": float(best_acc), "epochs": len(history), "elapsed": round(elapsed, 1)}, f, indent=2)
    print(f"[Teacher] Done. best_val_acc={best_acc:.4f} ({elapsed:.0f}s)")


# ─── CNN Training ─────────────────────────────────────────────────────────────
def train_cnn(seq_tr, y_tr, seq_te, y_te,
              teacher_ckpt, feature_scaler, run_dir, num_classes,
              distill_T, distill_alpha, batch_size, lr,
              epochs, patience, seed, device, force, mode):
    assert mode in {"distill", "plain"}
    ckpt_name = "cnn_baseline.pt" if mode == "plain" else "cnn_student.pt"
    ckpt = Path(run_dir) / "artifacts" / ckpt_name
    if ckpt.exists() and not force:
        print(f"[CNN-{mode}] Checkpoint exists, skipping")
        return

    teacher_model = None
    if mode == "distill":
        class TeacherFromSeq(nn.Module):
            def __init__(self, t_ckpt, scaler):
                super().__init__()
                self.dnn = DNNTeacher(
                    in_dim=t_ckpt["in_dim"], num_classes=t_ckpt["num_classes"],
                    hidden_sizes=t_ckpt["hidden_sizes"], dropout=t_ckpt["dropout"],
                )
                self.dnn.load_state_dict(t_ckpt["state_dict"])
                self.dnn.eval()
                self.scaler = scaler

            @torch.no_grad()
            def forward(self, x_seq):
                if hasattr(x_seq, "numpy"):
                    x_seq = x_seq.cpu().numpy()
                feat = extract_rich_features(x_seq, max_packets=x_seq.shape[1])
                feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
                feat_s = self.scaler.transform(feat.astype(np.float64))
                return self.dnn(torch.from_numpy(feat_s).float())

        teacher_model = TeacherFromSeq(teacher_ckpt, feature_scaler).to(device)
        print(f"[CNN-Distill] Distilling from DNN teacher (T={distill_T}, alpha={distill_alpha})")
    else:
        print(f"[CNN-Baseline] Plain training (no teacher)")

    trainer = TrainerCNN(
        X_tr=seq_tr, y_tr=y_tr,
        X_te=seq_te, y_te=y_te,
        run_dir=run_dir, num_classes=num_classes,
        mode="distill" if mode == "distill" else "plain",
        teacher_model=teacher_model,
        distill_T=distill_T, distill_alpha=distill_alpha,
        batch_size=batch_size, lr=lr,
        epochs=epochs, patience=patience,
        seed=seed, device=device,
    )
    summary = trainer.train(epochs=epochs, patience=patience)
    trainer.save()
    print(f"[CNN-{'Baseline' if mode == 'plain' else 'Student'}] Done. best_val_acc={summary['best_val_acc']:.4f}")


# ─── LR Training ─────────────────────────────────────────────────────────────
def train_lr(feat_tr, y_tr, feat_te, y_te, num_classes, run_dir, force):
    import joblib
    ckpt = Path(run_dir) / "artifacts" / "lr_pure.pkl"
    if ckpt.exists() and not force:
        print("[LR] Checkpoint exists, skipping")
        return
    print("[LR] Training...")
    lr_model = LRStudent(num_classes=num_classes, mode="pure")
    lr_model.fit(feat_tr, y_tr)
    pred = lr_model.predict(feat_te)
    acc = accuracy_score(y_te, pred)
    print(f"  LR acc: {acc:.4f}")
    joblib.dump(lr_model, ckpt)


# ─── Evaluation ───────────────────────────────────────────────────────────────
def evaluate(feat_te, y_te, le,
             seq_te, teacher_ckpt, cnn_state, cnn_base_state,
             run_dir, num_classes, device):
    import joblib
    label_names = list(le.classes_)
    results = {}

    # DNN Teacher
    print("[Eval] DNN Teacher...")
    teacher = DNNTeacher(
        in_dim=teacher_ckpt["in_dim"], num_classes=teacher_ckpt["num_classes"],
        hidden_sizes=teacher_ckpt["hidden_sizes"], dropout=teacher_ckpt["dropout"],
    ).to(device)
    teacher.load_state_dict(teacher_ckpt["state_dict"])
    teacher.eval()
    with torch.no_grad():
        pred_t = teacher(torch.from_numpy(feat_te).float().to(device)).argmax(1).cpu().numpy()
    results["teacher_dnn"] = compute_metrics(y_te, pred_t, label_names)

    # CNN models
    print("[Eval] CNN models...")
    for name, state in [("cnn_baseline", cnn_base_state), ("cnn_student", cnn_state)]:
        if state is None:
            continue
        cnn = CNNStudent(
            seq_len=state["model_config"]["seq_len"],
            n_features=state["model_config"]["n_features"],
            num_classes=state["model_config"]["num_classes"],
        ).to(device)
        cnn.load_state_dict(state["state_dict"])
        cnn.eval()
        with torch.no_grad():
            logits = cnn(torch.from_numpy(seq_te).float().to(device)).cpu().numpy()
        pred = logits.argmax(axis=1)
        results[name] = compute_metrics(y_te, pred, label_names)

    # Pure LR
    lr_ckpt = Path(run_dir) / "artifacts" / "lr_pure.pkl"
    if lr_ckpt.exists():
        print("[Eval] Pure LR...")
        lr_model = joblib.load(lr_ckpt)
        pred_lr = lr_model.predict(feat_te)
        results["lr_pure"] = compute_metrics(y_te, pred_lr, label_names)

    # Latency
    print("[Eval] Latency...")
    latency = {}
    latency["teacher_dnn"] = measure_latency(
        lambda x: teacher(torch.from_numpy(x).float().to(device)).argmax(1).cpu().numpy(),
        feat_te[:1],
    )
    if lr_ckpt.exists():
        latency["lr_pure"] = measure_latency(lambda x: lr_model.predict(x), feat_te[:1])

    for name, state in [("cnn_student", cnn_state), ("cnn_baseline", cnn_base_state)]:
        if state is None:
            continue
        m = CNNStudent(
            seq_len=state["model_config"]["seq_len"],
            n_features=state["model_config"]["n_features"],
            num_classes=state["model_config"]["num_classes"],
        ).to(device)
        m.load_state_dict(state["state_dict"])
        m.eval()
        latency[name] = measure_latency(
            lambda x: m(torch.from_numpy(x).float().to(device)).argmax(1).cpu().numpy(),
            seq_te[:1],
        )
        del m

    # Save
    with open(Path(run_dir) / "metrics.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(Path(run_dir) / "latency.json", "w") as f:
        json.dump(latency, f, ensure_ascii=False, indent=2)

    # Print summary
    print("\n" + "=" * 72)
    print(f"  {'Model':<18} {'Accuracy':>10} {'F1-Score':>10} {'Latency':>12}")
    print("-" * 72)
    for name, res in results.items():
        lat = latency.get(name, {}).get("p50_ms", 0)
        print(f"  {name:<18} {res['accuracy']*100:>9.2f}% {res['macro_f1']:>10.4f} {lat:>10.2f} ms")
    print("=" * 72)
    print("\nAll models evaluated on the same test set (flow-level).")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flows", default="data/packet_sequences/all_flows.pkl")
    ap.add_argument("--run_dir", default="runs/exp03")
    ap.add_argument("--step", default="all",
                    choices=["all", "teacher", "cnn", "cnn_baseline", "lr", "evaluate"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_packets", type=int, default=100)
    ap.add_argument("--epochs_teacher", type=int, default=600)
    ap.add_argument("--epochs_cnn", type=int, default=300)
    ap.add_argument("--distill_T", type=float, default=4.0)
    ap.add_argument("--distill_alpha", type=float, default=0.5)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--teacher_hidden", type=str, default="256,128,64")
    ap.add_argument("--teacher_dropout", type=float, default=0.3)
    ap.add_argument("--teacher_lr", type=float, default=1e-3)
    ap.add_argument("--teacher_batch_size", type=int, default=128)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "artifacts").mkdir(exist_ok=True)
    device = get_device()
    print(f"\nDevice: {device}  |  Step: {args.step}")
    print(f"Flow data: {args.flows}")

    # ── Load and prepare data ────────────────────────────────────────────────
    print("\n[Data] Loading flows...")
    data = load_flows(args.flows, max_packets=args.max_packets, seed=args.seed)
    num_classes = len(data["le"].classes_)

    # Save scaler and label encoder
    import joblib
    joblib.dump(data["scaler"], Path(run_dir) / "artifacts" / "feature_scaler.pkl")
    joblib.dump(data["le"], Path(run_dir) / "artifacts" / "label_encoder.pkl")

    teacher_path = Path(run_dir) / "artifacts" / "teacher.pt"
    cnn_path = Path(run_dir) / "artifacts" / "cnn_student.pt"
    cnn_base_path = Path(run_dir) / "artifacts" / "cnn_baseline.pt"

    # ── Steps ───────────────────────────────────────────────────────────────
    if args.step in ("all", "teacher"):
        train_teacher(
            data["feat_tr"], data["y_tr"], data["feat_te"], data["y_te"],
            num_classes, run_dir,
            hidden_sizes=[int(x) for x in args.teacher_hidden.split(",")],
            dropout=args.teacher_dropout, label_smoothing=args.label_smoothing,
            batch_size=args.teacher_batch_size, lr=args.teacher_lr,
            epochs=args.epochs_teacher, seed=args.seed, device=device, force=args.force,
            use_class_weights=True,
        )

    if args.step in ("all", "cnn"):
        if not teacher_path.exists():
            print("[Error] teacher.pt not found. Run --step teacher first.")
            sys.exit(1)
        t_ckpt = torch.load(teacher_path, map_location="cpu")
        train_cnn(
            data["seq_tr"], data["y_tr"], data["seq_te"], data["y_te"],
            t_ckpt, data["scaler"], run_dir, num_classes,
            distill_T=args.distill_T, distill_alpha=args.distill_alpha,
            batch_size=64, lr=1e-3,
            epochs=args.epochs_cnn, patience=args.patience,
            seed=args.seed, device=device, force=args.force, mode="distill",
        )

    if args.step == "cnn_baseline":
        train_cnn(
            data["seq_tr"], data["y_tr"], data["seq_te"], data["y_te"],
            None, data["scaler"], run_dir, num_classes,
            distill_T=args.distill_T, distill_alpha=args.distill_alpha,
            batch_size=64, lr=1e-3,
            epochs=args.epochs_cnn, patience=args.patience,
            seed=args.seed, device=device, force=args.force, mode="plain",
        )

    if args.step in ("all", "lr"):
        train_lr(data["feat_tr"], data["y_tr"], data["feat_te"], data["y_te"],
                 num_classes, run_dir, force=args.force)

    if args.step in ("all", "evaluate"):
        if not teacher_path.exists():
            print("[Error] Checkpoints not found."); sys.exit(1)
        t_ckpt = torch.load(teacher_path, map_location="cpu")
        cnn_ckpt = torch.load(cnn_path, map_location="cpu") if cnn_path.exists() else None
        cnn_base_ckpt = torch.load(cnn_base_path, map_location="cpu") if cnn_base_path.exists() else None
        evaluate(
            data["feat_te"], data["y_te"], data["le"],
            data["seq_te"], t_ckpt, cnn_ckpt, cnn_base_ckpt,
            run_dir, num_classes, device,
        )


if __name__ == "__main__":
    main()
