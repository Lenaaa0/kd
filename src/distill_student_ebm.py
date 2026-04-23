from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from interpret.glassbox import ExplainableBoostingClassifier
from joblib import dump, load
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from .config import load_config, timestamp_run_dir
from .data import fit_or_load_scaler, load_flow_csv, to_numeric_matrix
from .model_teacher import MLPTeacher, TabularTransformerTeacher
from .utils import save_json, set_seed


def _soft_targets_from_teacher(
    model: torch.nn.Module,
    x: np.ndarray,
    temperature: float,
    device: str,
    batch_size: int = 4096,
) -> np.ndarray:
    model.eval()
    probs = []
    t = float(temperature)
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            xb = torch.from_numpy(x[i : i + batch_size]).to(device)
            logits = model(xb) / t
            p = torch.softmax(logits, dim=1).cpu().numpy()
            probs.append(p)
    return np.concatenate(probs, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--teacher_run_dir", default=None, help="可选：指定包含 teacher.pt 的 runs 子目录")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    # 运行目录：如果用户指定了 teacher_run_dir，就复用同一目录；否则新建
    run_dir = Path(args.teacher_run_dir) if args.teacher_run_dir else timestamp_run_dir(cfg["output"]["runs_dir"])
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)

    teacher_path = run_dir / "teacher.pt"
    if not teacher_path.exists():
        raise FileNotFoundError(
            f"teacher.pt not found under {run_dir}. "
            f"Please run train_teacher first or pass --teacher_run_dir."
        )

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

    x_tr, feat_names = to_numeric_matrix(x_tr_df)
    x_te, _ = to_numeric_matrix(x_te_df)

    # 特征名必须一致（否则解释意义会变）；不一致就直接报错提醒用户
    if feat_names != feat_names_teacher:
        raise ValueError(
            "Feature columns mismatch between current data and teacher checkpoint.\n"
            f"- teacher features: {len(feat_names_teacher)}\n"
            f"- current features: {len(feat_names)}\n"
            "Make sure you use the same CSV schema and drop_cols."
        )

    le_path = run_dir / "artifacts" / "label_encoder.joblib"
    if le_path.exists():
        le: LabelEncoder = load(le_path)
    else:
        le = LabelEncoder()
        le.fit([str(x) for x in label_names])
        dump(le, le_path)

    y_tr = le.transform(y_tr_s.astype(str))
    y_te = le.transform(y_te_s.astype(str))

    scaler_path = str(run_dir / "artifacts" / "scaler.joblib")
    x_tr, x_te, _ = fit_or_load_scaler(
        x_train=x_tr,
        x_test=x_te,
        standardize=bool(cfg["preprocess"].get("standardize", True)),
        scaler_path=scaler_path,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_type = str(teacher_ckpt.get("model_type", "mlp")).lower()
    if model_type == "mlp":
        teacher = MLPTeacher(
            in_dim=int(teacher_ckpt["in_dim"]),
            num_classes=int(teacher_ckpt["num_classes"]),
            hidden_sizes=list(teacher_ckpt["hidden_sizes"]),
            dropout=float(teacher_ckpt["dropout"]),
        ).to(device)
    elif model_type in {"tab_transformer", "tabular_transformer", "transformer"}:
        teacher = TabularTransformerTeacher(
            num_features=int(teacher_ckpt["num_features"]),
            num_classes=int(teacher_ckpt["num_classes"]),
            d_model=int(teacher_ckpt["d_model"]),
            n_heads=int(teacher_ckpt["n_heads"]),
            n_layers=int(teacher_ckpt["n_layers"]),
            dropout=float(teacher_ckpt["dropout"]),
            ff_mult=int(teacher_ckpt["ff_mult"]),
        ).to(device)
    else:
        raise ValueError(f"Unknown teacher model_type in checkpoint: {model_type}")
    teacher.load_state_dict(teacher_ckpt["state_dict"])

    T = float(cfg["distill"].get("temperature", 3.0))
    alpha = float(cfg["distill"].get("alpha", 0.5))
    alpha = min(max(alpha, 0.0), 1.0)

    # 教师软目标
    p_teacher = _soft_targets_from_teacher(teacher, x_tr, temperature=T, device=device)
    # 软标签 -> “软类别”通过加权采样近似：把每个样本复制一次并赋权（EBM 不直接支持 soft-label）
    # 实现方式：为每个样本生成一个“蒸馏标签” = 教师最大概率类；并把样本权重设为：
    # w = (1-alpha)*1 + alpha*p_teacher[max_class]
    # 这样至少能把“教师置信度”迁移进来，同时保留真标签监督。
    y_teacher_hard = p_teacher.argmax(axis=1)
    w = (1.0 - alpha) * np.ones_like(y_tr, dtype=np.float32) + alpha * p_teacher[np.arange(p_teacher.shape[0]), y_teacher_hard].astype(
        np.float32
    )

    # 最终训练标签：真实标签（保证不偏离任务），权重融合教师置信度（蒸馏信号）
    max_bins = int(cfg["distill"].get("max_bins", 256))
    student = ExplainableBoostingClassifier(random_state=seed, max_bins=max_bins)
    student.fit(x_tr, y_tr, sample_weight=w)

    out_path = run_dir / "student_ebm.pkl"
    dump(student, out_path)

    save_json(
        {
            "run_dir": str(run_dir),
            "teacher_path": str(teacher_path),
            "student_path": str(out_path),
            "temperature": T,
            "alpha": alpha,
            "max_bins": max_bins,
        },
        run_dir / "distill_summary.json",
    )

    print(str(run_dir))


if __name__ == "__main__":
    main()

