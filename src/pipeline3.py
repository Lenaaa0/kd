"""
Unified Pipeline v3 (exp06): Full optimization for encrypted traffic classification.

Improvements over exp03:
  - Proper train/val/test split (60/20/20) with early stopping on val
  - RF/ExtraTrees teachers (known to achieve >90% on ISCX VPN-nonVPN)
  - Enhanced DNN teacher with residual blocks + GELU + focal loss
  - Fixed LR: uses all 49 rich features (not hardcoded 20)
  - LR with class weights for minority classes
  - Better distillation: soft labels from tree teachers instead of weak DNN
  - Class-weight-aware Focal Loss for DNN

Teachers:
  1. RF Teacher (Random Forest, class_weight='balanced')
  2. ExtraTrees Teacher (Extremely Randomized Trees)
  3. DNN Teacher (enhanced, residual blocks, focal loss)

Students:
  1. CNN Student (distilled from best teacher)
  2. LR Student (distilled from teacher + class weights)

Data:
  - Source: data/packet_sequences/all_flows.pkl (24,763 flows)
  - Filter: >=10 packets per flow -> ~13,750 samples
  - Split: 60% train / 20% val / 20% test
  - Teachers use 49-dim rich features
  - CNN uses raw packet sequences (100 x 3)

Usage:
  python -m src.pipeline3 --step all
  python -m src.pipeline3 --step rf_teacher
  python -m src.pipeline3 --step et_teacher
  python -m src.pipeline3 --step dnn_teacher
  python -m src.pipeline3 --step cnn_baseline
  python -m src.pipeline3 --step cnn_distill
  python -m src.pipeline3 --step lr
  python -m src.pipeline3 --step evaluate
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

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
from src.models.teacher_dnn import DNNTeacher, FocalLoss
from src.models.teacher_tree import (
    ExtraTreesTeacher, RFTeacher, evaluate_tree_teacher
)
from src.training import TrainerCNN
from src.utils import set_seed


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ─── Data Loading ─────────────────────────────────────────────────────────────
def load_flows(pcap_path: str, max_packets: int = 100,
               min_packets: int = 10, seed: int = 42):
    """
    Load flow data, split into train/val/test (60/20/20).
    Extracts both raw sequences and rich features.
    """
    import pandas as pd

    df = pd.read_pickle(pcap_path)
    df = df[df["num_packets"] >= min_packets].reset_index(drop=True)
    print(f"  Filtered to {len(df)} flows (>= {min_packets} packets)")

    le = LabelEncoder()
    y_all = le.fit_transform(df["label"].values)

    # Raw packet sequences (for CNN)
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
    split2_kwargs = dict(test_size=0.25, random_state=seed)  # 0.25 of 80 = 20%
    if can_stratify:
        split1_kwargs["stratify"] = y_all

    (X_seq_trval, X_seq_te,
     y_trval, y_te) = train_test_split(X_seq, y_all, **split1_kwargs)

    split2_kwargs["stratify"] = y_trval if can_stratify else None
    (X_seq_tr, X_seq_va,
     y_tr, y_va) = train_test_split(X_seq_trval, y_trval, **split2_kwargs)

    # Rich features
    print("  Extracting rich features...")
    feat_tr = extract_rich_features(X_seq_tr, max_packets=max_packets)
    feat_va = extract_rich_features(X_seq_va, max_packets=max_packets)
    feat_te = extract_rich_features(X_seq_te, max_packets=max_packets)

    feat_tr = np.nan_to_num(feat_tr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    feat_va = np.nan_to_num(feat_va, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    feat_te = np.nan_to_num(feat_te, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    scaler = StandardScaler()
    feat_tr = scaler.fit_transform(feat_tr).astype(np.float32)
    feat_va = scaler.transform(feat_va).astype(np.float32)
    feat_te = scaler.transform(feat_te).astype(np.float32)

    # Class distribution
    print(f"  Classes: {list(le.classes_)}")
    for label_idx, label_name in enumerate(le.classes_):
        n_tr = int(np.sum(y_tr == label_idx))
        n_va = int(np.sum(y_va == label_idx))
        n_te = int(np.sum(y_te == label_idx))
        print(f"    {label_name:<15} train={n_tr:>5}  val={n_va:>5}  test={n_te:>5}")

    return {
        "feat_tr": feat_tr, "feat_va": feat_va, "feat_te": feat_te,
        "seq_tr": X_seq_tr, "seq_va": X_seq_va, "seq_te": X_seq_te,
        "y_tr": y_tr, "y_va": y_va, "y_te": y_te,
        "le": le, "scaler": scaler,
    }


# ─── Teacher Training ────────────────────────────────────────────────────────
def train_rf_teacher(feat_tr, y_tr, feat_va, y_va, num_classes,
                      run_dir, seed, force):
    """Train Random Forest teacher."""
    ckpt = Path(run_dir) / "artifacts" / "rf_teacher.pkl"
    if ckpt.exists() and not force:
        print("[RF-Teacher] Checkpoint exists, skipping")
        return

    print("[RF-Teacher] Training Random Forest (n_estimators=500, balanced)...")
    t0 = time.time()
    model = RFTeacher(
        n_estimators=500,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=seed,
    )
    model.fit(feat_tr, y_tr)

    pred = model.predict(feat_va)
    acc = float(accuracy_score(y_va, pred))
    f1 = float(f1_score(y_va, pred, average="macro", zero_division=0))
    print(f"  RF Teacher: val_acc={acc:.4f}  val_f1={f1:.4f}")

    model.save(ckpt)
    elapsed = time.time() - t0
    with open(Path(run_dir) / "training_summary_rf.json", "w") as f:
        json.dump({"val_acc": float(acc), "val_f1": float(f1), "elapsed": round(elapsed, 1)}, f, indent=2)
    print(f"[RF-Teacher] Done ({elapsed:.0f}s)")


def train_et_teacher(feat_tr, y_tr, feat_va, y_va, num_classes,
                      run_dir, seed, force):
    """Train Extra Trees teacher."""
    ckpt = Path(run_dir) / "artifacts" / "et_teacher.pkl"
    if ckpt.exists() and not force:
        print("[ET-Teacher] Checkpoint exists, skipping")
        return

    print("[ET-Teacher] Training Extra Trees (n_estimators=500, balanced)...")
    t0 = time.time()
    model = ExtraTreesTeacher(
        n_estimators=500,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=seed,
    )
    model.fit(feat_tr, y_tr)

    pred = model.predict(feat_va)
    acc = float(accuracy_score(y_va, pred))
    f1 = float(f1_score(y_va, pred, average="macro", zero_division=0))
    print(f"  ET Teacher: val_acc={acc:.4f}  val_f1={f1:.4f}")

    model.save(ckpt)
    elapsed = time.time() - t0
    with open(Path(run_dir) / "training_summary_et.json", "w") as f:
        json.dump({"val_acc": float(acc), "val_f1": float(f1), "elapsed": round(elapsed, 1)}, f, indent=2)
    print(f"[ET-Teacher] Done ({elapsed:.0f}s)")


def train_dnn_teacher(feat_tr, y_tr, feat_va, y_va, num_classes,
                      run_dir, hidden_sizes, dropout, label_smoothing,
                      batch_size, lr, epochs, patience,
                      seed, device, force, use_focal,
                      gamma, use_class_weights):
    """Train Enhanced DNN teacher with residual blocks and focal loss."""
    ckpt = Path(run_dir) / "artifacts" / "dnn_teacher.pt"
    if ckpt.exists() and not force:
        print("[DNN-Teacher] Checkpoint exists, skipping")
        return

    set_seed(seed)
    in_dim = feat_tr.shape[1]
    model = DNNTeacher(
        in_dim=in_dim, num_classes=num_classes,
        hidden_sizes=hidden_sizes, dropout=dropout,
        residual_depth=3, use_gelu=True,
    ).to(device)

    # Class weights
    class_weights_tensor = None
    if use_class_weights:
        class_counts = np.bincount(y_tr)
        total = len(y_tr)
        weights = total / (num_classes * class_counts + 1e-8)
        weights = weights / weights.sum() * num_classes
        class_weights_tensor = torch.from_numpy(weights).float().to(device)
        print(f"  Class weights: {[f'{w:.2f}' for w in weights]}")

    # Loss
    if use_focal:
        loss_fn = FocalLoss(
            gamma=gamma,
            alpha=class_weights_tensor,
            label_smoothing=label_smoothing,
        )
        print(f"  Using Focal Loss (gamma={gamma}, label_smoothing={label_smoothing})")
    else:
        loss_fn = nn.CrossEntropyLoss(
            weight=class_weights_tensor,
            label_smoothing=label_smoothing,
        )

    loader = DataLoader(
        TensorDataset(torch.from_numpy(feat_tr), torch.from_numpy(y_tr).long()),
        batch_size=batch_size, shuffle=True, num_workers=0,
        pin_memory=(device == "cuda"),
    )
    X_va_t = torch.from_numpy(feat_va).float().to(device)
    y_va_np = y_va

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    best_f1, best_acc, best_state, patience_counter, history = 0.0, 0.0, None, 0, []
    t0 = time.time()

    for epoch in tqdm(range(epochs), desc="[DNN-Teacher]"):
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
            preds = model(X_va_t).argmax(1).cpu().numpy()
        acc = accuracy_score(y_va_np, preds)
        f1 = f1_score(y_va_np, preds, average="macro", zero_division=0)
        improved = f1 > best_f1
        if improved:
            best_f1 = f1
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        history.append({
            "epoch": epoch + 1,
            "loss": total_loss / max(n_samples, 1),
            "acc": float(acc),
            "f1": float(f1),
        })
        if (epoch + 1) % 50 == 0 or improved:
            print(f"  ep={epoch+1:>4d}  loss={total_loss/max(n_samples,1):.4f}  "
                  f"acc={acc:.4f}  f1={f1:.4f} {'*best*' if improved else ''}")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch + 1} (patience={patience})")
            break

    model.load_state_dict(best_state)
    elapsed = time.time() - t0
    torch.save({
        "state_dict": model.state_dict(),
        "num_classes": num_classes,
        "in_dim": in_dim,
        "hidden_sizes": hidden_sizes,
        "dropout": dropout,
        "best_val_f1": float(best_f1),
        "best_val_acc": float(best_acc),
    }, ckpt)

    summary = {
        "best_val_f1": float(best_f1),
        "best_val_acc": float(best_acc),
        "epochs": len(history),
        "elapsed": round(elapsed, 1),
    }
    with open(Path(run_dir) / "training_summary_dnn.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[DNN-Teacher] Done. best_val_f1={best_f1:.4f} ({elapsed:.0f}s)")


def _compute_teacher_soft_labels(seq_data, teacher_ckpt, feature_scaler, num_classes, device):
    """
    Precompute teacher soft labels for the entire training set ONCE.
    This avoids expensive per-batch feature extraction during training.
    """
    import joblib

    print("  Precomputing teacher soft labels...")
    t0 = time.time()

    is_dict = isinstance(teacher_ckpt, dict)
    # Tree teacher: dict with model_type="RandomForestClassifier"/"ExtraTreesClassifier", or raw sklearn object
    is_tree = (is_dict and teacher_ckpt.get("model_type") in ("RandomForestClassifier", "ExtraTreesClassifier")) or \
              (not is_dict and hasattr(teacher_ckpt, "predict_proba") and not hasattr(teacher_ckpt, "state_dict"))
    # DNN teacher: dict with "state_dict" key
    is_dnn = is_dict and "state_dict" in teacher_ckpt
    if is_dict:
        teacher_model = teacher_ckpt["model"]
    else:
        teacher_model = teacher_ckpt

    n = len(seq_data)

    if is_tree:
        feat = extract_rich_features(seq_data, max_packets=seq_data.shape[1])
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        feat = feature_scaler.transform(feat)
        proba = teacher_model.predict_proba(feat)
        # Clamp and renormalize to prevent 0 probabilities
        eps = 1e-6
        proba = np.clip(proba, eps, 1.0 - eps)
        proba = proba / proba.sum(axis=1, keepdims=True)
        # Store as log-probabilities for KL computation
        logits = np.log(proba)
        soft_labels = torch.from_numpy(logits).float()
        print(f"  Tree teacher soft labels (log-probs) computed in {time.time()-t0:.1f}s")
    elif is_dnn:
        feat = extract_rich_features(seq_data, max_packets=seq_data.shape[1])
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        feat = feature_scaler.transform(feat)
        feat_t = torch.from_numpy(feat).float()
        batch_size = 256
        all_logits = []
        dnn_model = DNNTeacher(
            in_dim=teacher_ckpt["in_dim"],
            num_classes=teacher_ckpt["num_classes"],
            hidden_sizes=teacher_ckpt["hidden_sizes"],
            dropout=teacher_ckpt["dropout"],
        ).to(device)
        dnn_model.load_state_dict(teacher_ckpt["state_dict"])
        dnn_model.eval()
        for i in range(0, n, batch_size):
            batch = feat_t[i:i+batch_size].to(device)
            with torch.no_grad():
                logits = dnn_model(batch)
            all_logits.append(logits.cpu())
        soft_labels = torch.cat(all_logits, dim=0)
        del dnn_model
        print(f"  DNN teacher soft labels computed in {time.time()-t0:.1f}s")

    return soft_labels


def train_cnn(seq_tr, y_tr, seq_va, y_va, num_classes,
              teacher_ckpt, feature_scaler, run_dir,
              distill_T, distill_alpha, batch_size, lr,
              epochs, patience, seed, device, force, mode):
    """Train CNN student (plain or distilled) with precomputed soft labels."""
    assert mode in {"distill", "plain"}
    ckpt_name = "cnn_baseline.pt" if mode == "plain" else "cnn_student.pt"
    ckpt = Path(run_dir) / "artifacts" / ckpt_name
    if ckpt.exists() and not force:
        print(f"[CNN-{mode}] Checkpoint exists, skipping")
        return

    teacher_soft = None
    if mode == "distill" and teacher_ckpt is not None:
        import joblib
        if isinstance(teacher_ckpt, dict):
            teacher_name = teacher_ckpt.get("model_type", "Teacher")
        else:
            teacher_name = teacher_ckpt.__class__.__name__
        print(f"[CNN-Distill] Distilling from {teacher_name} (T={distill_T}, alpha={distill_alpha})")
        teacher_soft = _compute_teacher_soft_labels(
            seq_tr, teacher_ckpt, feature_scaler, num_classes, device
        )
    else:
        print(f"[CNN-Baseline] Plain training (no teacher)")

    trainer = TrainerCNN(
        X_tr=seq_tr, y_tr=y_tr,
        X_te=seq_va, y_te=y_va,
        run_dir=run_dir, num_classes=num_classes,
        mode="distill" if mode == "distill" else "plain",
        teacher_soft_labels=teacher_soft,
        distill_T=distill_T, distill_alpha=distill_alpha,
        batch_size=batch_size, lr=lr,
        epochs=epochs, patience=patience,
        seed=seed, device=device,
    )
    summary = trainer.train(epochs=epochs, patience=patience)
    trainer.save()
    print(f"[CNN-{'Baseline' if mode == 'plain' else 'Student'}] Done. "
          f"best_val_acc={summary['best_val_acc']:.4f}")


# ─── LR Training ───────────────────────────────────────────────────────────
def train_lr(feat_tr, y_tr, feat_va, y_va, num_classes,
             run_dir, force, use_class_weights):
    """Train LR student with class weights."""
    ckpt = Path(run_dir) / "artifacts" / "lr_balanced.pkl"
    if ckpt.exists() and not force:
        print("[LR] Checkpoint exists, skipping")
        return

    cw = "balanced" if use_class_weights else None
    print(f"[LR] Training (class_weight={cw})...")
    lr_model = LRStudent(
        num_classes=num_classes, mode="pure",
        class_weight=cw, C=1.0, solver="lbfgs",
    )
    lr_model.fit(feat_tr, y_tr)

    pred = lr_model.predict(feat_va)
    acc = float(accuracy_score(y_va, pred))
    f1 = float(f1_score(y_va, pred, average="macro", zero_division=0))
    print(f"  LR: val_acc={acc:.4f}  val_f1={f1:.4f}")

    import joblib
    joblib.dump(lr_model, ckpt)
    with open(Path(run_dir) / "training_summary_lr.json", "w") as f:
        json.dump({"val_acc": float(acc), "val_f1": float(f1), "class_weight": cw}, f, indent=2)


# ─── Evaluation ─────────────────────────────────────────────────────────────
def evaluate(data, run_dir, num_classes, device):
    """Evaluate all models on the held-out test set."""
    import joblib

    label_names = list(data["le"].classes_)
    results = {}
    latency = {}

    # RF Teacher
    rf_ckpt = Path(run_dir) / "artifacts" / "rf_teacher.pkl"
    if rf_ckpt.exists():
        print("[Eval] RF Teacher...")
        import joblib
        rf_data = joblib.load(rf_ckpt)
        rf_model = rf_data["model"] if isinstance(rf_data, dict) else rf_data
        pred = rf_model.predict(data["feat_te"])
        results["rf_teacher"] = compute_metrics(data["y_te"], pred, label_names)
        results["rf_teacher"]["model_type"] = rf_data.get("model_type", "RandomForest") if isinstance(rf_data, dict) else "RandomForest"
        latency["rf_teacher"] = measure_latency(lambda x: rf_model.predict(x), data["feat_te"][:1])

    # ET Teacher
    et_ckpt = Path(run_dir) / "artifacts" / "et_teacher.pkl"
    if et_ckpt.exists():
        print("[Eval] ET Teacher...")
        et_data = joblib.load(et_ckpt)
        et_model = et_data["model"] if isinstance(et_data, dict) else et_data
        pred = et_model.predict(data["feat_te"])
        results["et_teacher"] = compute_metrics(data["y_te"], pred, label_names)
        results["et_teacher"]["model_type"] = et_data.get("model_type", "ExtraTrees") if isinstance(et_data, dict) else "ExtraTrees"
        latency["et_teacher"] = measure_latency(
            lambda x: et_model.predict(x), data["feat_te"][:1]
        )

    # DNN Teacher
    dnn_ckpt = Path(run_dir) / "artifacts" / "dnn_teacher.pt"
    if dnn_ckpt.exists():
        print("[Eval] DNN Teacher...")
        dnn_state = torch.load(dnn_ckpt, map_location="cpu")
        try:
            dnn = DNNTeacher(
                in_dim=dnn_state["in_dim"],
                num_classes=dnn_state["num_classes"],
                hidden_sizes=dnn_state["hidden_sizes"],
                dropout=dnn_state["dropout"],
            ).to(device)
            dnn.load_state_dict(dnn_state["state_dict"])
            dnn.eval()
            with torch.no_grad():
                pred = dnn(torch.from_numpy(data["feat_te"]).float().to(device)).argmax(1).cpu().numpy()
            results["dnn_teacher"] = compute_metrics(data["y_te"], pred, label_names)
            results["dnn_teacher"]["model_type"] = "DNN"
            latency["dnn_teacher"] = measure_latency(
                lambda x: dnn(torch.from_numpy(x).float().to(device)).argmax(1).cpu().numpy(),
                data["feat_te"][:1],
            )
            del dnn
        except Exception as e:
            print(f"  DNN Teacher skipped: {e}")

    # CNN models
    scaler = joblib.load(Path(run_dir) / "artifacts" / "feature_scaler.pkl")
    for name, fname in [("cnn_baseline", "cnn_baseline.pt"),
                         ("cnn_student", "cnn_student.pt")]:
        fpath = Path(run_dir) / "artifacts" / fname
        if not fpath.exists():
            continue
        print(f"[Eval] {name}...")
        state = torch.load(fpath, map_location="cpu")
        cnn = CNNStudent(
            seq_len=state["model_config"]["seq_len"],
            n_features=state["model_config"]["n_features"],
            num_classes=state["model_config"]["num_classes"],
        ).to(device)
        cnn.load_state_dict(state["state_dict"])
        cnn.eval()
        with torch.no_grad():
            logits = cnn(torch.from_numpy(data["seq_te"]).float().to(device)).cpu().numpy()
        pred = logits.argmax(axis=1)
        results[name] = compute_metrics(data["y_te"], pred, label_names)
        # Capture cnn in closure for timing
        _cnn = cnn
        latency[name] = measure_latency(
            lambda x, _m=_cnn: _m(torch.from_numpy(x).float().to(device)).argmax(1).cpu().numpy(),
            data["seq_te"][:1],
        )
        del cnn

    # LR models
    lr_ckpt = Path(run_dir) / "artifacts" / "lr_balanced.pkl"
    if lr_ckpt.exists():
        print("[Eval] LR...")
        lr_model = joblib.load(lr_ckpt)
        pred = lr_model.predict(data["feat_te"])
        results["lr_balanced"] = compute_metrics(data["y_te"], pred, label_names)
        latency["lr_balanced"] = measure_latency(
            lambda x: lr_model.predict(x), data["feat_te"][:1]
        )

    # Save
    with open(Path(run_dir) / "metrics.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(Path(run_dir) / "latency.json", "w") as f:
        json.dump(latency, f, ensure_ascii=False, indent=2)

    # Print summary table
    print("\n" + "=" * 90)
    print(f"  {'Model':<20} {'Accuracy':>10} {'F1-Score':>10} {'Precision':>10} {'Recall':>10} {'Latency':>12}")
    print("-" * 90)
    for name in _model_order():
        if name not in results:
            continue
        res = results[name]
        lat = latency.get(name, {}).get("p50_ms", 0)
        print(f"  {name:<20} {res['accuracy']*100:>9.2f}% {res['macro_f1']:>10.4f} "
              f"{res['macro_precision']*100:>9.2f}% {res['macro_recall']*100:>9.2f}% {lat:>10.2f} ms")
    print("=" * 90)
    print("\nTest set evaluation (20% of data, never seen during training/val).")
    print("Teachers: RF / ExtraTrees / Enhanced DNN")
    print("Students: CNN (baseline / distilled) / LR (balanced)")


def _model_order():
    """Preferred display order for results."""
    return [
        "rf_teacher", "et_teacher", "dnn_teacher",
        "cnn_student", "cnn_baseline", "lr_balanced",
    ]


# ─── Main ───────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flows", default="data/packet_sequences/all_flows.pkl")
    ap.add_argument("--run_dir", default="runs/exp06")
    ap.add_argument("--step", default="all",
                   choices=["all", "rf_teacher", "et_teacher", "dnn_teacher",
                            "cnn_baseline", "cnn_distill", "lr", "evaluate"])
    ap.add_argument("--teacher", default="rf",
                   choices=["rf", "et", "dnn"],
                   help="Teacher model for distillation")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_packets", type=int, default=100)
    ap.add_argument("--min_packets", type=int, default=10)
    ap.add_argument("--epochs_dnn", type=int, default=800)
    ap.add_argument("--epochs_cnn", type=int, default=300)
    ap.add_argument("--patience_dnn", type=int, default=50)
    ap.add_argument("--patience_cnn", type=int, default=30)
    ap.add_argument("--distill_T", type=float, default=4.0)
    ap.add_argument("--distill_alpha", type=float, default=0.5)
    ap.add_argument("--dnn_hidden", type=str, default="512,256,128")
    ap.add_argument("--dnn_dropout", type=float, default=0.3)
    ap.add_argument("--dnn_lr", type=float, default=3e-4)
    ap.add_argument("--dnn_batch_size", type=int, default=128)
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--use_focal", action="store_true", help="Use focal loss instead of CE")
    ap.add_argument("--focal_gamma", type=float, default=2.0)
    ap.add_argument("--use_class_weights", action="store_true", help="Apply class weights")
    ap.add_argument("--cnn_lr", type=float, default=1e-3)
    ap.add_argument("--cnn_batch_size", type=int, default=64)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "artifacts").mkdir(exist_ok=True)
    device = get_device()
    hidden_sizes = [int(x) for x in args.dnn_hidden.split(",")]

    print(f"\nDevice: {device}  |  Step: {args.step}")
    print(f"Flow data: {args.flows}  |  min_packets={args.min_packets}")

    # Load data
    print("\n[Data] Loading and splitting flows...")
    data = load_flows(args.flows, max_packets=args.max_packets,
                      min_packets=args.min_packets, seed=args.seed)
    num_classes = len(data["le"].classes_)

    # Save preprocessors
    import joblib
    joblib.dump(data["scaler"], Path(run_dir) / "artifacts" / "feature_scaler.pkl")
    joblib.dump(data["le"], Path(run_dir) / "artifacts" / "label_encoder.pkl")

    # Paths
    rf_path = Path(run_dir) / "artifacts" / "rf_teacher.pkl"
    et_path = Path(run_dir) / "artifacts" / "et_teacher.pkl"
    dnn_path = Path(run_dir) / "artifacts" / "dnn_teacher.pt"

    def load_teacher_ckpt():
        """Load the selected teacher checkpoint."""
        if args.teacher == "rf":
            path = rf_path
        elif args.teacher == "et":
            path = et_path
        else:
            path = dnn_path

        if not path.exists():
            print(f"[Error] Teacher checkpoint not found: {path}")
            sys.exit(1)

        if path.suffix == ".pkl":
            import joblib as jl
            return jl.load(path)
        else:
            return torch.load(path, map_location="cpu")

    # ── Steps ──────────────────────────────────────────────────────────────
    if args.step in ("all", "rf_teacher"):
        train_rf_teacher(
            data["feat_tr"], data["y_tr"], data["feat_va"], data["y_va"],
            num_classes, run_dir, seed=args.seed, force=args.force,
        )

    if args.step in ("all", "et_teacher"):
        train_et_teacher(
            data["feat_tr"], data["y_tr"], data["feat_va"], data["y_va"],
            num_classes, run_dir, seed=args.seed, force=args.force,
        )

    if args.step in ("all", "dnn_teacher"):
        train_dnn_teacher(
            data["feat_tr"], data["y_tr"], data["feat_va"], data["y_va"],
            num_classes, run_dir,
            hidden_sizes=hidden_sizes,
            dropout=args.dnn_dropout,
            label_smoothing=args.label_smoothing,
            batch_size=args.dnn_batch_size, lr=args.dnn_lr,
            epochs=args.epochs_dnn, patience=args.patience_dnn,
            seed=args.seed, device=device, force=args.force,
            use_focal=args.use_focal,
            gamma=args.focal_gamma,
            use_class_weights=args.use_class_weights,
        )

    if args.step == "cnn_baseline":
        train_cnn(
            data["seq_tr"], data["y_tr"], data["seq_va"], data["y_va"],
            num_classes, None, None, run_dir,
            distill_T=args.distill_T, distill_alpha=args.distill_alpha,
            batch_size=args.cnn_batch_size, lr=args.cnn_lr,
            epochs=args.epochs_cnn, patience=args.patience_cnn,
            seed=args.seed, device=device, force=args.force, mode="plain",
        )

    if args.step in ("all", "cnn_distill"):
        if not any(p.exists() for p in [rf_path, et_path, dnn_path]):
            print("[Error] No teacher found. Run --step rf_teacher/et_teacher/dnn_teacher first.")
            sys.exit(1)
        t_ckpt = load_teacher_ckpt()
        scaler = joblib.load(Path(run_dir) / "artifacts" / "feature_scaler.pkl")
        train_cnn(
            data["seq_tr"], data["y_tr"], data["seq_va"], data["y_va"],
            num_classes, t_ckpt, scaler, run_dir,
            distill_T=args.distill_T, distill_alpha=args.distill_alpha,
            batch_size=args.cnn_batch_size, lr=args.cnn_lr,
            epochs=args.epochs_cnn, patience=args.patience_cnn,
            seed=args.seed, device=device, force=args.force, mode="distill",
        )

    if args.step in ("all", "lr"):
        train_lr(
            data["feat_tr"], data["y_tr"], data["feat_va"], data["y_va"],
            num_classes, run_dir, force=args.force,
            use_class_weights=True,
        )

    if args.step in ("all", "evaluate"):
        evaluate(data, run_dir, num_classes, device)


if __name__ == "__main__":
    main()
