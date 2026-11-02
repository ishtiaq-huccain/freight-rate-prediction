"""Regression metrics in dollars.

MAE is the primary metric (the official metric is not disclosed; MAE is in
dollars and robust to the ~1.5% corrupted labels). MAPE is safe because every
rate is positive.
"""

from __future__ import annotations

import numpy as np


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if len(y_true) == 0:
        return {
            "n": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "median_ae": np.nan,
            "mape_pct": np.nan,
            "bias": np.nan,
        }
    err = y_pred - y_true
    return {
        "n": int(len(y_true)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "median_ae": float(np.median(np.abs(err))),
        "mape_pct": float(np.mean(np.abs(err) / y_true) * 100),
        "bias": float(np.mean(err)),
    }
