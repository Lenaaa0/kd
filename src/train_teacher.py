from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from joblib import dump
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .config import load_config, timestamp_run_dir
from .data import fit_or_load_scaler, load_flow_csv, to_numeric_matrix
from .model_teacher import MLPTeacher, TabularTransformerTeacher
from .utils import save_json, set_seed


def _train_epoch(model, loader, opt, loss_fn, device):
    model.train()
    total = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        opt.step()
        total += float(loss.item()) * xb.size(0)
        n += xb.size(0)
    return total / max(n, 1)


@torch.no_grad()
def _eval(model, loader, device):
    model.eval()
    logits_all = []
    y_all = []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb).cpu().numpy()
        logits_all.append(logits)
        y_all.append(yb.numpy())
    return np.concatenate(logits_all, axis=0), np.concatenate(y_all, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    run_dir = timestamp_run_dir(cfg["output"]["runs_dir"])
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)

    x_tr_df, y_tr_s, x_te_df, y_te_s = load_flow_csv(
        train_csv=cfg["data"]["train_csv"],
        test_csv=cfg["data"].get("test_csv"),
        label_col=cfg["data"]["label_col"],
        drop_cols=list(cfg["data"].get("drop_cols", [])),
        label_map=dict(cfg["data"].get("label_map", {})),
        max_rows=cfg["preprocess"].get("max_rows"),
        test_size=float(cfg["preprocess"].get("test_size", 0.2)),
        seed=seed,
    )

    x_tr, feat_names = to_numeric_matrix(x_tr_df)
    x_te, _ = to_numeric_matrix(x_te_df)

    le = LabelEncoder()
    y_tr = le.fit_transform(y_tr_s.astype(str))
    y_te = le.transform(y_te_s.astype(str))
    label_names = list(le.classes_)

    scaler_path = str(run_dir / "artifacts" / "scaler.joblib")
    x_tr, x_te, _ = fit_or_load_scaler(
        x_train=x_tr,
        x_test=x_te,
        standardize=bool(cfg["preprocess"].get("standardize", True)),
        scaler_path=scaler_path,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_type = str(cfg["teacher"].get("model_type", "mlp")).lower()
    if model_type == "mlp":
        model = MLPTeacher(
            in_dim=x_tr.shape[1],
            num_classes=len(label_names),
            hidden_sizes=list(cfg["teacher"].get("hidden_sizes", [256, 128])),
            dropout=float(cfg["teacher"].get("dropout", 0.2)),
        ).to(device)
        teacher_meta = {
            "model_type": "mlp",
            "in_dim": x_tr.shape[1],
            "hidden_sizes": list(cfg["teacher"].get("hidden_sizes", [256, 128])),
            "dropout": float(cfg["teacher"].get("dropout", 0.2)),
        }
    elif model_type in {"tab_transformer", "tabular_transformer", "transformer"}:
        model = TabularTransformerTeacher(
            num_features=x_tr.shape[1],
            num_classes=len(label_names),
            d_model=int(cfg["teacher"].get("d_model", 64)),
            n_heads=int(cfg["teacher"].get("n_heads", 4)),
            n_layers=int(cfg["teacher"].get("n_layers", 2)),
            dropout=float(cfg["teacher"].get("dropout", 0.1)),
            ff_mult=int(cfg["teacher"].get("ff_mult", 4)),
        ).to(device)
        teacher_meta = {
            "model_type": "tab_transformer",
            "num_features": x_tr.shape[1],
            "d_model": int(cfg["teacher"].get("d_model", 64)),
            "n_heads": int(cfg["teacher"].get("n_heads", 4)),
            "n_layers": int(cfg["teacher"].get("n_layers", 2)),
            "dropout": float(cfg["teacher"].get("dropout", 0.1)),
            "ff_mult": int(cfg["teacher"].get("ff_mult", 4)),
        }
    else:
        raise ValueError(f"Unknown teacher.model_type: {model_type}")

    ds_tr = TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr).long())
    ds_te = TensorDataset(torch.from_numpy(x_te), torch.from_numpy(y_te).long())
    dl_tr = DataLoader(ds_tr, batch_size=int(cfg["teacher"].get("batch_size", 512)), shuffle=True)
    dl_te = DataLoader(ds_te, batch_size=2048, shuffle=False)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["teacher"].get("lr", 1e-3)),
        weight_decay=float(cfg["teacher"].get("weight_decay", 1e-5)),
    )
    loss_fn = nn.CrossEntropyLoss()

    epochs = int(cfg["teacher"].get("epochs", 20))
    history = []
    for ep in range(1, epochs + 1):
        train_loss = _train_epoch(model, dl_tr, opt, loss_fn, device)
        logits_te, y_te_np = _eval(model, dl_te, device)
        pred_te = logits_te.argmax(axis=1)
        acc_te = float((pred_te == y_te_np).mean())
        history.append({"epoch": ep, "train_loss": train_loss, "test_acc": acc_te})
        tqdm.write(f"[teacher] epoch={ep}/{epochs} train_loss={train_loss:.4f} test_acc={acc_te:.4f}")

    teacher_path = run_dir / "teacher.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "num_classes": len(label_names),
            **teacher_meta,
            "label_names": label_names,
            "feature_names": feat_names,
        },
        teacher_path,
    )
    dump(le, run_dir / "artifacts" / "label_encoder.joblib")

    save_json(
        {
            "run_dir": str(run_dir),
            "teacher_path": str(teacher_path),
            "history": history,
            "label_names": label_names,
            "num_features": len(feat_names),
        },
        run_dir / "teacher_train_summary.json",
    )

    print(str(run_dir))


if __name__ == "__main__":
    main()

