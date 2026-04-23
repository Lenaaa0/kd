"""
Latency measurement for model inference.
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np


def measure_latency(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X_sample: np.ndarray,
    n_warmup: int = 20,
    n_runs: int = 200,
) -> dict:
    """
    Measure inference latency of a predict function.

    Args:
        predict_fn: Model.predict or equivalent. Takes (1, ...) array, returns (1,) array.
        X_sample: Single sample for timing, shape (1, ...) matching model input.
        n_warmup: Number of warmup runs before measurement.
        n_runs: Number of timed runs.

    Returns:
        Dict with p50, p90, p99, mean, std in milliseconds.
    """
    # Warmup
    for _ in range(n_warmup):
        predict_fn(X_sample)

    times_ms = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        predict_fn(X_sample)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    arr = np.array(times_ms, dtype=np.float64)
    return {
        "n_warmup": n_warmup,
        "n_runs": n_runs,
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p99_ms": float(np.percentile(arr, 99)),
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std()),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
    }
