"""Naive reference models. Each predicts distance x a rate per mile."""

from __future__ import annotations

import numpy as np
import pandas as pd


class MedianRatePerMile:
    """One global median rate per mile."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> MedianRatePerMile:
        self.rate_per_mile_ = float(np.median(np.asarray(y) / X["distance"].to_numpy()))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X["distance"].to_numpy() * self.rate_per_mile_


class GroupedMedianRatePerMile:
    """Median rate per mile per (equipment, distance bin).

    Bin edges are quantiles of the training distances. The two drivers found in
    the EDA (distance curve, equipment level) are captured with no other
    features. An empty cell falls back to the bin median, then the global median.
    """

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins

    def _bins(self, X: pd.DataFrame) -> np.ndarray:
        return np.digitize(X["distance"].to_numpy(), self.bin_edges_)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> GroupedMedianRatePerMile:
        self.bin_edges_ = np.quantile(X["distance"], np.linspace(0, 1, self.n_bins + 1)[1:-1])
        cells = pd.DataFrame(
            {
                "equipment": X["equipment"].to_numpy(),
                "bin": self._bins(X),
                "rpm": np.asarray(y, dtype=float) / X["distance"].to_numpy(),
            }
        )
        self.cell_rpm_ = cells.groupby(["equipment", "bin"])["rpm"].median()
        self.bin_rpm_ = cells.groupby("bin")["rpm"].median()
        self.global_rpm_ = float(cells["rpm"].median())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        bins = self._bins(X)
        keys = pd.MultiIndex.from_arrays([X["equipment"].to_numpy(), bins])
        rpm = self.cell_rpm_.reindex(keys).to_numpy()
        rpm = np.where(np.isnan(rpm), self.bin_rpm_.reindex(bins).to_numpy(), rpm)
        rpm = np.where(np.isnan(rpm), self.global_rpm_, rpm)
        return X["distance"].to_numpy() * rpm
