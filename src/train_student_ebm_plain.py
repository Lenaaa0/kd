from __future__ import annotations

import argparse
from pathlib import Path

import torch
from interpret.glassbox import ExplainableBoostingClassifier
from joblib import dump, load
from sklearn.preprocessing import LabelEncoder

from .config import load_config, timestamp_run_dir
from .data import fit_or_load_scaler, load_flow_csv, to_numeric_matrix
from .utils import save_json, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run_dir", default=None, help="可选：指定 runs 子目录；不传则新建")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    run_dir = Path(args.run_dir) if args.run_dir else timestamp_run_dir(cfg["output"]["runs_dir"])
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

    le_path = run_dir / "artifacts" / "label_encoder.joblib"
    if le_path.exists():
        le: LabelEncoder = load(le_path)
    else:
        le = LabelEncoder()
        le.fit(y_tr_s.astype(str))
        dump(le, le_path)

    y_tr = le.transform(y_tr_s.astype(str))
    _ = le.transform(y_te_s.astype(str))  # 触发类集合一致性检查

    scaler_path = str(run_dir / "artifacts" / "scaler.joblib")
    x_tr, x_te, _ = fit_or_load_scaler(
        x_train=x_tr,
        x_test=x_te,
        standardize=bool(cfg["preprocess"].get("standardize", True)),
        scaler_path=scaler_path,
    )

    max_bins = int(cfg["distill"].get("max_bins", 256))
    student_plain = ExplainableBoostingClassifier(random_state=seed, max_bins=max_bins)
    student_plain.fit(x_tr, y_tr)

    out_path = run_dir / "student_ebm_plain.pkl"
    dump(student_plain, out_path)

    save_json(
        {
            "run_dir": str(run_dir),
            "student_plain_path": str(out_path),
            "max_bins": max_bins,
            "num_features": len(feat_names),
        },
        run_dir / "student_plain_summary.json",
    )

    print(str(run_dir))


if __name__ == "__main__":
    main()

