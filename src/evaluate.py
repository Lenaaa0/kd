from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from joblib import dump, load
from sklearn.preprocessing import LabelEncoder

from .config import load_config
from .data import fit_or_load_scaler, load_flow_csv, to_numeric_matrix
from .model_teacher import MLPTeacher, TabularTransformerTeacher
from .utils import ensure_dir, metrics_from_predictions, save_json, set_seed


def _latest_run_dir(runs_dir: str | Path) -> Path:
    p = Path(runs_dir)
    if not p.exists():
        raise FileNotFoundError(f"runs_dir not found: {p}")
    sub = [d for d in p.iterdir() if d.is_dir()]
    if not sub:
        raise FileNotFoundError(f"No runs found under: {p}")
    return sorted(sub, key=lambda d: d.name)[-1]


@torch.no_grad()
def _build_teacher(ckpt: Dict[str, Any], device: str) -> torch.nn.Module:
    model_type = str(ckpt.get("model_type", "mlp")).lower()
    if model_type == "mlp":
        model = MLPTeacher(
            in_dim=int(ckpt["in_dim"]),
            num_classes=int(ckpt["num_classes"]),
            hidden_sizes=list(ckpt["hidden_sizes"]),
            dropout=float(ckpt["dropout"]),
        ).to(device)
    elif model_type in {"tab_transformer", "tabular_transformer", "transformer"}:
        model = TabularTransformerTeacher(
            num_features=int(ckpt["num_features"]),
            num_classes=int(ckpt["num_classes"]),
            d_model=int(ckpt["d_model"]),
            n_heads=int(ckpt["n_heads"]),
            n_layers=int(ckpt["n_layers"]),
            dropout=float(ckpt["dropout"]),
            ff_mult=int(ckpt["ff_mult"]),
        ).to(device)
    else:
        raise ValueError(f"Unknown teacher model_type in checkpoint: {model_type}")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def _teacher_predict(model: torch.nn.Module, x: np.ndarray, device: str, batch_size: int = 4096) -> np.ndarray:
    logits = []
    for i in range(0, x.shape[0], batch_size):
        xb = torch.from_numpy(x[i : i + batch_size]).to(device)
        lg = model(xb).cpu().numpy()
        logits.append(lg)
    return np.concatenate(logits, axis=0)


def _latency_ms(fn, x: np.ndarray, runs: int, warmup: int = 20) -> Dict[str, float]:
    # 使用单样本多次测量，更接近在线判决
    idx = 0
    sample = x[idx : idx + 1]
    for _ in range(warmup):
        fn(sample)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(sample)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    arr = np.array(times, dtype=np.float64)
    return {
        "runs": float(runs),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p99_ms": float(np.percentile(arr, 99)),
        "mean_ms": float(arr.mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run_dir", default=None, help="可选：指定 runs 子目录；默认取最新一次")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    run_dir = Path(args.run_dir) if args.run_dir else _latest_run_dir(cfg["output"]["runs_dir"])
    artifacts_dir = run_dir / "artifacts"

    teacher_path = run_dir / "teacher.pt"
    if not teacher_path.exists():
        raise FileNotFoundError(f"teacher.pt not found under {run_dir}")
    teacher_ckpt = torch.load(teacher_path, map_location="cpu")
    label_names = list(teacher_ckpt["label_names"])
    feat_names_teacher = list(teacher_ckpt["feature_names"])

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

    x_te, feat_names = to_numeric_matrix(x_te_df)
    if feat_names != feat_names_teacher:
        raise ValueError("Feature columns mismatch; please keep the same schema as training.")

    le_path = artifacts_dir / "label_encoder.joblib"
    le: LabelEncoder = load(le_path)
    y_te = le.transform(y_te_s.astype(str))

    scaler_path = str(artifacts_dir / "scaler.joblib")
    x_te, _, _ = fit_or_load_scaler(
        x_train=x_te,
        x_test=x_te,
        standardize=bool(cfg["preprocess"].get("standardize", True)),
        scaler_path=scaler_path,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher_model = _build_teacher(teacher_ckpt, device=device)
    logits_te = _teacher_predict(teacher_model, x_te, device=device)
    pred_teacher = logits_te.argmax(axis=1)
    metrics_teacher = metrics_from_predictions(y_te, pred_teacher, labels=list(range(len(label_names))))

    student_path = run_dir / "student_ebm.pkl"
    metrics_student = None
    if student_path.exists():
        student = load(student_path)
        pred_student = student.predict(x_te)
        metrics_student = metrics_from_predictions(y_te, pred_student, labels=list(range(len(label_names))))

    student_plain_path = run_dir / "student_ebm_plain.pkl"
    metrics_student_plain = None
    if student_plain_path.exists():
        student_plain = load(student_plain_path)
        pred_student_plain = student_plain.predict(x_te)
        metrics_student_plain = metrics_from_predictions(y_te, pred_student_plain, labels=list(range(len(label_names))))

    # 延迟：教师/学生
    latency_runs = int(cfg["eval"].get("latency_runs", 200))
    latency = {}
    latency["teacher"] = _latency_ms(
        fn=lambda xx: _teacher_predict(teacher_model, xx.astype(np.float32), device=device, batch_size=1),
        x=x_te,
        runs=latency_runs,
    )
    if student_path.exists():
        latency["student_ebm"] = _latency_ms(
            fn=lambda xx: student.predict(xx),
            x=x_te,
            runs=latency_runs,
        )
    if student_plain_path.exists():
        latency["student_ebm_plain"] = _latency_ms(
            fn=lambda xx: student_plain.predict(xx),
            x=x_te,
            runs=latency_runs,
        )

    # 解释导出（学生）
    explain_dir = ensure_dir(run_dir / "explanations")
    if student_path.exists():
        k = int(cfg["eval"].get("explain_samples", 20))
        k = min(k, x_te.shape[0])
        idxs = np.linspace(0, max(k - 1, 0), num=k, dtype=int)
        local = student.explain_local(x_te[idxs], y_te[idxs])
        # local.data(i) 每个样本的 feature贡献；写成轻量 json 便于论文作图
        topk = int(cfg["eval"].get("topk_features", 15))
        rows = []
        for i in range(k):
            names = local.data(i)["names"]
            scores = local.data(i)["scores"]
            pairs = sorted(zip(names, scores), key=lambda t: abs(float(t[1])), reverse=True)[:topk]
            rows.append(
                {
                    "index": int(idxs[i]),
                    "true": int(y_te[idxs[i]]),
                    "pred": int(student.predict(x_te[idxs[i] : idxs[i] + 1])[0]),
                    "top_contrib": [{"feature": str(n), "contrib": float(s)} for n, s in pairs],
                }
            )
        save_json({"samples": rows, "label_names": label_names, "feature_names": feat_names}, explain_dir / "student_local_explanations.json")

    save_json(
        {
            "run_dir": str(run_dir),
            "label_names": label_names,
            "teacher": metrics_teacher,
            "student_ebm": metrics_student,
            "student_ebm_plain": metrics_student_plain,
        },
        run_dir / "metrics.json",
    )
    save_json({"run_dir": str(run_dir), "latency": latency}, run_dir / "latency.json")

    print(str(run_dir))


if __name__ == "__main__":
    main()

