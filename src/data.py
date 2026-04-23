from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class DatasetArtifacts:
    feature_names: List[str]
    label_names: List[str]
    scaler_path: Optional[str]


def _read_csv(path: str | Path, max_rows: Optional[int]) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p}")
    return pd.read_csv(p, nrows=max_rows)


def _apply_label_map(y: pd.Series, label_map: Dict[str, str]) -> pd.Series:
    if not label_map:
        return y
    return y.astype(str).map(lambda v: label_map.get(v, v))


def load_flow_csv(
    train_csv: str,
    test_csv: Optional[str],
    label_col: str,
    drop_cols: List[str],
    label_map: Dict[str, str],
    max_rows: Optional[int],
    test_size: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    df_train = _read_csv(train_csv, max_rows=max_rows)

    if label_col not in df_train.columns:
        raise ValueError(f"label_col '{label_col}' not found in train CSV columns.")

    if drop_cols:
        df_train = df_train.drop(columns=[c for c in drop_cols if c in df_train.columns])

    # 丢弃标签为空的样本，避免后续分层切分报 NaN 错误
    df_train = df_train.dropna(subset=[label_col])

    y_train = _apply_label_map(df_train[label_col], label_map=label_map)
    x_train = df_train.drop(columns=[label_col])

    if test_csv and Path(test_csv).exists():
        df_test = _read_csv(test_csv, max_rows=max_rows)
        if label_col not in df_test.columns:
            raise ValueError(f"label_col '{label_col}' not found in test CSV columns.")
        if drop_cols:
            df_test = df_test.drop(columns=[c for c in drop_cols if c in df_test.columns])
        df_test = df_test.dropna(subset=[label_col])
        y_test = _apply_label_map(df_test[label_col], label_map=label_map)
        x_test = df_test.drop(columns=[label_col])
        return x_train, y_train, x_test, y_test

    x_tr, x_te, y_tr, y_te = train_test_split(
        x_train,
        y_train,
        test_size=test_size,
        random_state=seed,
        stratify=y_train if y_train.nunique() > 1 else None,
    )
    return x_tr, y_tr, x_te, y_te


def to_numeric_matrix(x: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    # 丢弃明显非数值列；其余尝试转成数值（不可转的置为 NaN，再用 0 填）
    x2 = x.copy()
    for c in x2.columns:
        if not pd.api.types.is_numeric_dtype(x2[c]):
            x2[c] = pd.to_numeric(x2[c], errors="coerce")
    x2 = x2.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return x2.to_numpy(dtype=np.float32), list(x2.columns)


def fit_or_load_scaler(
    x_train: np.ndarray,
    x_test: np.ndarray,
    standardize: bool,
    scaler_path: Optional[str],
) -> Tuple[np.ndarray, np.ndarray, Optional[str]]:
    if not standardize:
        return x_train, x_test, None

    if scaler_path and Path(scaler_path).exists():
        scaler: StandardScaler = load(scaler_path)
        return scaler.transform(x_train), scaler.transform(x_test), scaler_path

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train_s = scaler.transform(x_train)
    x_test_s = scaler.transform(x_test)
    if scaler_path:
        dump(scaler, scaler_path)
        return x_train_s, x_test_s, scaler_path
    return x_train_s, x_test_s, None

