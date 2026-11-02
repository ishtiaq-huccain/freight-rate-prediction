"""Single feature pipeline shared by training, evaluation and inference.

``FeatureBuilder`` turns a *cleaned* load frame (output of ``LoadCleaner``)
into a numeric matrix. It learns nothing from data. Every feature is a
function of the row itself plus fixed constants, so a row's features don't
depend on which other rows are in the batch, or on whether it's a training,
validation or December-chart row. That rules out train/serve skew.

Features are selected by *group name*, so feature sets for ablations are just
lists of names in the config.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from freight_rate.data import schema

# Fixed origin for the time index. A constant, not learned from data.
TIME_ORIGIN = pd.Timestamp("2025-01-01")

_EQUIPMENT_COLUMNS = {e: "eq_" + e.lower().replace(" ", "_") for e in schema.EQUIPMENT_TYPES}

FEATURE_GROUPS: dict[str, list[str]] = {
    "log_distance": ["log_distance"],
    "distance": ["distance"],
    "equipment": list(_EQUIPMENT_COLUMNS.values()),
    "weight": ["weight_k"],
    "weight_missing": ["weight_missing"],
    "market_index": ["market_index"],
    "market_index_missing": ["market_index_missing"],
    "quote_signal": ["quote_signal"],
    "coords": ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"],
    "time_index": ["days_since_start"],
    "day_of_week": ["day_of_week"],
}


def _compute(X: pd.DataFrame) -> pd.DataFrame:
    unknown = set(X["equipment"]) - set(_EQUIPMENT_COLUMNS)
    if unknown:
        raise ValueError(f"unknown equipment types: {sorted(unknown)}")
    dates = pd.to_datetime(X[schema.DATE])
    out = pd.DataFrame(index=X.index)
    out["log_distance"] = np.log(X["distance"].astype(float))
    out["distance"] = X["distance"].astype(float)
    for equipment, col in _EQUIPMENT_COLUMNS.items():
        out[col] = (X["equipment"] == equipment).astype(float)
    out["weight_k"] = X["weight"].astype(float) / 1000.0
    out["weight_missing"] = X["weight_missing"].astype(float)
    out["market_index"] = X["market_index"].astype(float)
    out["market_index_missing"] = X["market_index_missing"].astype(float)
    out["quote_signal"] = X["quote_signal"].astype(float)
    for col in FEATURE_GROUPS["coords"]:
        out[col] = X[col].astype(float)
    out["days_since_start"] = (dates - TIME_ORIGIN).dt.days.astype(float)
    out["day_of_week"] = dates.dt.dayofweek.astype(float)
    return out


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Builds the selected feature groups from a cleaned load frame."""

    def __init__(self, groups: Sequence[str] = ("log_distance", "equipment")):
        self.groups = groups

    def fit(self, X: pd.DataFrame, y=None) -> FeatureBuilder:
        unknown = [g for g in self.groups if g not in FEATURE_GROUPS]
        if unknown:
            raise ValueError(f"unknown feature groups {unknown}; known: {sorted(FEATURE_GROUPS)}")
        if len(set(self.groups)) != len(self.groups):
            raise ValueError("feature groups must be unique")
        self.feature_names_ = [c for g in self.groups for c in FEATURE_GROUPS[g]]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        features = _compute(X)[self.feature_names_]
        bad = features.columns[~np.isfinite(features.to_numpy()).all(axis=0)]
        if len(bad):
            raise ValueError(f"non-finite feature values in {list(bad)}; was the frame cleaned?")
        return features

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(self.feature_names_, dtype=object)
