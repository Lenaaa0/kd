"""
Main training pipeline for dual-student knowledge distillation.
Architecture:
  DNN Teacher (statistical features) -> CNN Student (packet sequences) -> LR Student (stats + CNN logits)

Usage:
    # Full pipeline
    python -m src.pipeline --data data/packet_sequences/packet_sequences.pkl --run_dir runs/exp01 --step all

    # Step by step
    python -m src.pipeline --step teacher --force
    python -m src.pipeline --step cnn --force
    python -m src.pipeline --step cnn_baseline --force
    python -m src.pipeline --step lr --force
    python -m src.pipeline --step evaluate
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
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import extract_flow_statistics, load_packet_sequences
from src.data.rich_features import extract_rich_features
from src.evaluation import compute_metrics, measure_latency
from src.models import CNNStudent, LRStudent
from src.models.teacher_dnn import DNNTeacher
from src.training import TrainerCNN
from src.utils import set_seed


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ══════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════
def load_data(data_path: str, seed: int = 42, run_dir: Path = None):
    """Load packet sequences and extract features."""
    X_seq_tr, y_tr, X_seq_te, y_te, le = load_packet_sequences(
        data_path, test_size=0.2, seed=seed
    )
    # Rich features for DNN teacher
    X_feat_tr = extract_rich_features(X_seq_tr)
    X_feat_te = extract_rich_features(X_seq_te)
    # Simple stats for LR student
    X_stats_tr = extract_flow_statistics(X_seq_tr)
    X_stats_te = extract_flow_statistics(X_seq_te)
    # Replace NaN/Inf
    for arr in [X_feat_tr, X_feat_te, X_stats_tr, X_stats_te]:
        arr[np.isnan(arr) | np.isinf(arr)] = 0.0
    # Standardize
    scaler = StandardScaler()
    X_feat_tr_s = scaler.fit_transform(X_feat_tr)
    X_feat_te_s = scaler.transform(X_feat_te)
    # Save scaler
    if run_dir is not None:
        import joblib
        joblib.dump(scaler, run_dir / "artifacts" / "feature_scaler.pkl")
    print(f"  Sequences: train={X_seq_tr.shape}, test={X_seq_te.shape}")
    print(f"  Rich features: {X_feat_tr_s.shape[1]} dims")
    print(f"  Stats features: {X_stats_tr.shape[1]} dims")
    print(f"  Classes ({len(le.classes_)}): {list(le.classes_)}")
    print(f"  Train labels: {np.bincount(y_tr)}")
    print(f"  Test  labels: {np.bincount(y_te)}")
    return {
        "seq_tr": X_seq_tr, "seq_te": X_seq_te,
        "feat_tr": X_feat_tr_s.astype(np.float32),
        "feat_te": X_feat_te_s.astype(np.float32),
        "stats_tr": X_stats_tr.astype(np.float32),
        "stats_te": X_stats_te.astype(np.float32),
        "y_tr": y_tr, "y_te": y_te,
        "label_encoder": le,
    }


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1, weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight

    def forward(self, pred, target):
        n_classes = pred.size(1)
        log_prob = F.log_softmax(pred, dim=-1)
        true_dist = torch.zeros_like(log_prob)
        true_dist.fill_(self.smoothing / n_classes)
        true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing + self.smoothing / n_classes)
        loss = (-true_dist * log_prob).sum(dim=-1)
        if self.weight is not None:
            loss = loss * self.weight[target].to(pred.device)
        return loss.mean()


# ══════════════════════════════════════════════════════════════════
# Step 1: Train DNN Teacher
# ══════════════════════════════════════════════════════════════════
def step_train_teacher(
    data, run_dir, num_classes,
    hidden_sizes=None, dropout=0.3, label_smoothing=0.1,
    batch_size=64, lr=1e-3, weight_decay=1e-4,
    epochs=300, seed=42, device="cpu", force=False,
):
    ckpt_path = run_dir / "artifacts" / "teacher.pt"
    if ckpt_path.exists() and not force:
        print(f"[Teacher] Checkpoint exists, skipping (use --force to retrain)")
        return

    set_seed(seed)
    X_tr = data["feat_tr"]
    y_tr = data["y_tr"]
    X_te = data["feat_te"]
    y_te = data["y_te"]

    in_dim = X_tr.shape[1]
    if hidden_sizes is None:
        hidden_sizes = [512, 256, 128]

    model = DNNTeacher(in_dim=in_dim, num_classes=num_classes,
                      hidden_sizes=hidden_sizes, dropout=dropout).to(device)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr).long()),
        batch_size=batch_size, shuffle=True, num_workers=0,
        pin_memory=(device == "cuda"),
    )
    X_te_t = torch.from_numpy(X_te).float().to(device)

    class_counts = np.bincount(y_tr, minlength=num_classes).astype(float)
    class_weights = 1.0 / (class_counts + 1)
    class_weights = class_weights / class_weights.sum() * num_classes
    print(f"[Teacher] Class weights: {dict(zip(range(num_classes), class_weights.round(2)))}")

    loss_fn = LabelSmoothingCrossEntropy(
        smoothing=label_smoothing,
        weight=torch.from_numpy(class_weights).float(),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    best_acc, best_state = 0.0, None
    history = []
    t0 = time.time()

    for epoch in tqdm(range(epochs), desc="[Teacher]"):
        model.train()
        total_loss, n_samples = 0.0, 0
        for xb, yb in train_loader:
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
        history.append({"epoch": epoch + 1, "train_loss": float(total_loss / max(n_samples, 1)), "val_acc": float(acc)})
        if (epoch + 1) % 20 == 0 or improved:
            print(f"  ep={epoch+1:>3d}  loss={total_loss/max(n_samples,1):.4f}  acc={acc:.4f} {'best' if improved else ''}")

    model.load_state_dict(best_state)
    elapsed = time.time() - t0

    torch.save({
        "state_dict": model.state_dict(),
        "num_classes": num_classes,
        "in_dim": in_dim,
        "hidden_sizes": hidden_sizes,
        "dropout": dropout,
        "best_val_acc": best_acc,
        "history": history,
    }, ckpt_path)

    with open(run_dir / "training_summary.json", "w") as f:
        json.dump({"best_val_acc": float(best_acc), "total_epochs": len(history),
                   "elapsed_seconds": round(elapsed, 1), "history": history}, f, indent=2)
    print(f"[Teacher] Done. best_val_acc={best_acc:.4f} in {len(history)} epochs ({elapsed:.0f}s)")


# ══════════════════════════════════════════════════════════════════
# Step 2: Train CNN Student (distill OR plain)
# ══════════════════════════════════════════════════════════════════
def step_train_cnn(
    data, teacher_ckpt, run_dir, num_classes,
    distill_T=2.0, distill_alpha=0.3,
    batch_size=64, lr=1e-3,
    epochs=80, patience=15,
    seed=42, device="cpu", force=False, mode="distill",
):
    assert mode in {"distill", "plain"}, f"mode must be 'distill' or 'plain', got {mode}"

    ckpt_name = "cnn_baseline.pt" if mode == "plain" else "cnn_student.pt"
    ckpt_path = run_dir / "artifacts" / ckpt_name
    if ckpt_path.exists() and not force:
        label = "Baseline" if mode == "plain" else "Student"
        print(f"[CNN-{label}] Checkpoint exists, skipping")
        return

    teacher_model = None
    if mode == "distill":
        import joblib
        scaler = joblib.load(run_dir / "artifacts" / "feature_scaler.pkl")

        class TeacherFromSequence(nn.Module):
            def __init__(self, t_ckpt, sc):
                super().__init__()
                self.dnn = DNNTeacher(
                    in_dim=t_ckpt["in_dim"], num_classes=t_ckpt["num_classes"],
                    hidden_sizes=t_ckpt["hidden_sizes"], dropout=t_ckpt["dropout"],
                )
                self.dnn.load_state_dict(t_ckpt["state_dict"])
                self.dnn.eval()
                self.scaler = sc

            @torch.no_grad()
            def forward(self, x_seq):
                if hasattr(x_seq, "numpy"):
                    x_seq = x_seq.cpu().numpy()
                feat = extract_rich_features(x_seq)
                feat[np.isnan(feat) | np.isinf(feat)] = 0.0
                feat_s = torch.from_numpy(self.scaler.transform(feat)).float()
                return self.dnn(feat_s)

        teacher_model = TeacherFromSequence(teacher_ckpt, scaler).to(device)
        print(f"[CNN-Student] Distilling from DNN teacher (T={distill_T}, alpha={distill_alpha})")
    else:
        print(f"[CNN-Baseline] Plain training (no teacher)")

    trainer = TrainerCNN(
        X_tr=data["seq_tr"], y_tr=data["y_tr"],
        X_te=data["seq_te"], y_te=data["y_te"],
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
    label = "Baseline" if mode == "plain" else "Student"
    print(f"[CNN-{label}] Done. best_val_acc={summary['best_val_acc']:.4f}")


# ══════════════════════════════════════════════════════════════════
# Step 3: Train LR Student
# ══════════════════════════════════════════════════════════════════
def step_train_lr(
    data, cnn_state_dict, run_dir, num_classes,
    seed=42, device="cpu", force=False,
):
    import joblib
    pure_ckpt = run_dir / "artifacts" / "lr_pure.pkl"
    distill_ckpt = run_dir / "artifacts" / "lr_distill.pkl"

    # Pure LR baseline
    if not pure_ckpt.exists() or force:
        print("[LR-Student] Training pure LR (stats only)...")
        lr_pure = LRStudent(num_classes=num_classes, mode="pure")
        lr_pure.fit(data["stats_tr"], data["y_tr"])
        joblib.dump(lr_pure, pure_ckpt)
        pred = lr_pure.predict(data["stats_te"])
        print(f"  Pure LR acc: {accuracy_score(data['y_te'], pred):.4f}")

    # Distilled LR
    if not distill_ckpt.exists() or force:
        print("[LR-Student] Training LR with CNN logits...")
        cnn = CNNStudent(
            seq_len=cnn_state_dict["model_config"]["seq_len"],
            n_features=cnn_state_dict["model_config"]["n_features"],
            num_classes=cnn_state_dict["model_config"]["num_classes"],
        )
        cnn.load_state_dict(cnn_state_dict["state_dict"])
        cnn.eval().to(device)
        with torch.no_grad():
            logits_tr = cnn(torch.from_numpy(data["seq_tr"]).float().to(device)).cpu().numpy()
            logits_te = cnn(torch.from_numpy(data["seq_te"]).float().to(device)).cpu().numpy()
        lr_distill = LRStudent(num_classes=num_classes, mode="distill")
        lr_distill.fit(data["stats_tr"], data["y_tr"], cnn_logits=logits_tr)
        joblib.dump(lr_distill, distill_ckpt)
        pred = lr_distill.predict(data["stats_te"], cnn_logits=logits_te)
        print(f"  Distilled LR acc: {accuracy_score(data['y_te'], pred):.4f}")


# ══════════════════════════════════════════════════════════════════
# Step 4: Evaluate
# ══════════════════════════════════════════════════════════════════
def step_evaluate(data, teacher_ckpt, cnn_state_dict, run_dir, num_classes, device="cpu"):
    import joblib

    results = {}
    label_names = list(data["label_encoder"].classes_)

    # DNN Teacher
    print("[Evaluate] DNN Teacher...")
    teacher = DNNTeacher(
        in_dim=teacher_ckpt["in_dim"], num_classes=teacher_ckpt["num_classes"],
        hidden_sizes=teacher_ckpt["hidden_sizes"], dropout=teacher_ckpt["dropout"],
    ).to(device)
    teacher.load_state_dict(teacher_ckpt["state_dict"])
    teacher.eval()
    X_te_t = torch.from_numpy(data["feat_te"]).float().to(device)
    with torch.no_grad():
        pred_t = teacher(X_te_t).argmax(1).cpu().numpy()
    results["teacher_dnn"] = compute_metrics(data["y_te"], pred_t, label_names)

    # CNN Baseline
    cnn_base_path = run_dir / "artifacts" / "cnn_baseline.pt"
    if cnn_base_path.exists():
        print("[Evaluate] CNN Baseline...")
        cnn_base = CNNStudent(
            seq_len=cnn_state_dict["model_config"]["seq_len"],
            n_features=cnn_state_dict["model_config"]["n_features"],
            num_classes=cnn_state_dict["model_config"]["num_classes"],
        ).to(device)
        ck = torch.load(cnn_base_path, map_location="cpu")
        cnn_base.load_state_dict(ck["state_dict"])
        cnn_base.eval()
        X_seq_t = torch.from_numpy(data["seq_te"]).float().to(device)
        with torch.no_grad():
            pred_base = cnn_base(X_seq_t).argmax(1).cpu().numpy()
        results["cnn_baseline"] = compute_metrics(data["y_te"], pred_base, label_names)
    else:
        print("[Evaluate] CNN Baseline: not found (skip)")

    # CNN Student (distilled)
    print("[Evaluate] CNN Student...")
    cnn = CNNStudent(
        seq_len=cnn_state_dict["model_config"]["seq_len"],
        n_features=cnn_state_dict["model_config"]["n_features"],
        num_classes=cnn_state_dict["model_config"]["num_classes"],
    ).to(device)
    cnn.load_state_dict(cnn_state_dict["state_dict"])
    cnn.eval()
    X_seq_t = torch.from_numpy(data["seq_te"]).float().to(device)
    with torch.no_grad():
        logits_cnn = cnn(X_seq_t).cpu().numpy()
    pred_cnn = logits_cnn.argmax(axis=1)
    results["cnn_student"] = compute_metrics(data["y_te"], pred_cnn, label_names)

    # Pure LR
    lr_pure_path = run_dir / "artifacts" / "lr_pure.pkl"
    if lr_pure_path.exists():
        print("[Evaluate] Pure LR...")
        lr_pure = joblib.load(lr_pure_path)
        pred_lr = lr_pure.predict(data["stats_te"])
        results["lr_pure"] = compute_metrics(data["y_te"], pred_lr, label_names)

    # Distilled LR
    lr_dist_path = run_dir / "artifacts" / "lr_distill.pkl"
    if lr_dist_path.exists():
        print("[Evaluate] Distilled LR...")
        lr_dist = joblib.load(lr_dist_path)
        with torch.no_grad():
            logits_cnn_te = cnn(X_seq_t).cpu().numpy()
        pred_lr_d = lr_dist.predict(data["stats_te"], cnn_logits=logits_cnn_te)
        results["lr_distill"] = compute_metrics(data["y_te"], pred_lr_d, label_names)

    # Latency
    print("[Evaluate] Latency...")
    latency = {}

    def _dnn_pred(x):
        return teacher(torch.from_numpy(x).float().to(device)).argmax(1).cpu().numpy()

    def _cnn_pred(x):
        return cnn(torch.from_numpy(x).float().to(device)).argmax(1).cpu().numpy()

    latency["teacher_dnn"] = measure_latency(_dnn_pred, data["feat_te"][:1])
    latency["cnn_student"] = measure_latency(_cnn_pred, data["seq_te"][:1])

    if cnn_base_path.exists():
        def _cnn_base_pred(x):
            return cnn_base(torch.from_numpy(x).float().to(device)).argmax(1).cpu().numpy()
        latency["cnn_baseline"] = measure_latency(_cnn_base_pred, data["seq_te"][:1])

    if lr_pure_path.exists():
        latency["lr_pure"] = measure_latency(lambda x: lr_pure.predict(x), data["stats_te"][:1])

    if lr_dist_path.exists():
        def _lr_dist_pred(x):
            logits = cnn(torch.from_numpy(data["seq_te"][:1]).float().to(device)).detach().cpu().numpy()
            return lr_dist.predict(x, cnn_logits=logits)
        latency["lr_distill"] = measure_latency(_lr_dist_pred, data["stats_te"][:1])

    # Save
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(run_dir / "latency.json", "w") as f:
        json.dump(latency, f, ensure_ascii=False, indent=2)

    # Print table
    print("\n" + "=" * 70)
    print(f"{'Model':<18} {'Accuracy':>10} {'F1':>8} {'Latency(p50)':>15}")
    print("-" * 70)
    for name, res in results.items():
        lat = latency.get(name, {}).get("p50_ms", 0)
        print(f"{name:<18} {res['accuracy']*100:>9.2f}% {res['macro_f1']:>8.4f} {lat:>13.2f} ms")
    print("=" * 70)
    print(f"\nResults saved to {run_dir}")


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="Dual-student knowledge distillation pipeline")
    ap.add_argument("--data", default="data/packet_sequences/packet_sequences.pkl")
    ap.add_argument("--run_dir", default="runs/exp01")
    ap.add_argument("--step", default="all",
                    choices=["all", "teacher", "cnn", "cnn_baseline", "lr", "evaluate"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs_teacher", type=int, default=600)
    ap.add_argument("--epochs_cnn", type=int, default=120)
    ap.add_argument("--distill_T", type=float, default=3.0)
    ap.add_argument("--distill_alpha", type=float, default=0.5)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--teacher_hidden", type=str, default="512,256,128")
    ap.add_argument("--teacher_dropout", type=float, default=0.2)
    ap.add_argument("--teacher_lr", type=float, default=5e-4)
    ap.add_argument("--teacher_batch_size", type=int, default=32)
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "artifacts").mkdir(exist_ok=True)
    device = get_device()

    print(f"\nDevice: {device}")
    print(f"Data: {args.data}")
    print(f"Run dir: {run_dir}")
    print(f"Step: {args.step}")

    data = load_data(args.data, seed=args.seed, run_dir=run_dir)
    num_classes = len(data["label_encoder"].classes_)
    import joblib
    joblib.dump(data["label_encoder"], run_dir / "artifacts" / "label_encoder.pkl")

    teacher_path = run_dir / "artifacts" / "teacher.pt"
    cnn_path = run_dir / "artifacts" / "cnn_student.pt"

    if args.step in ("all", "teacher"):
        hidden_sizes = [int(x) for x in args.teacher_hidden.split(",")]
        step_train_teacher(
            data, run_dir, num_classes,
            hidden_sizes=hidden_sizes,
            dropout=args.teacher_dropout,
            label_smoothing=args.label_smoothing,
            batch_size=args.teacher_batch_size,
            lr=args.teacher_lr,
            epochs=args.epochs_teacher,
            seed=args.seed, device=device, force=args.force,
        )

    if args.step in ("all", "cnn"):
        if not teacher_path.exists():
            print("[Error] teacher.pt not found."); sys.exit(1)
        t_ckpt = torch.load(teacher_path, map_location="cpu")
        step_train_cnn(
            data, t_ckpt, run_dir, num_classes,
            distill_T=args.distill_T, distill_alpha=args.distill_alpha,
            epochs=args.epochs_cnn, patience=args.patience,
            seed=args.seed, device=device, force=args.force, mode="distill",
        )

    if args.step == "cnn_baseline":
        step_train_cnn(
            data, None, run_dir, num_classes,
            distill_T=args.distill_T, distill_alpha=args.distill_alpha,
            epochs=args.epochs_cnn, patience=args.patience,
            seed=args.seed, device=device, force=args.force, mode="plain",
        )

    if args.step in ("all", "lr"):
        if not cnn_path.exists():
            print("[Error] cnn_student.pt not found."); sys.exit(1)
        c_ckpt = torch.load(cnn_path, map_location="cpu")
        step_train_lr(data, c_ckpt, run_dir, num_classes, seed=args.seed, device=device, force=args.force)

    if args.step in ("all", "evaluate"):
        if not teacher_path.exists() or not cnn_path.exists():
            print("[Error] Checkpoints not found."); sys.exit(1)
        t_ckpt = torch.load(teacher_path, map_location="cpu")
        c_ckpt = torch.load(cnn_path, map_location="cpu")
        step_evaluate(data, t_ckpt, c_ckpt, run_dir, num_classes, device=device)


if __name__ == "__main__":
    main()
