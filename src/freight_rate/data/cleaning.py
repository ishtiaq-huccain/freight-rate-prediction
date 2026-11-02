"""Leakage-safe cleaning, fitted on the training window only.

Two components, both following the scikit-learn fit/transform convention:

* ``LoadCleaner`` repairs input features. It never looks at the target and
  never drops rows, so every load that needs a prediction gets one.
* ``LabelAnomalyFilter`` identifies corrupted training labels. It is only used
  to choose which *training* rows to fit on, and to label test rows as clean or
  anomalous when reporting metrics. It is never used to drop rows at inference.

Decisions and evidence: notebooks/01_eda.ipynb and the README (section 4).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from freight_rate.data import schema


def daily_market_index(*frames: pd.DataFrame) -> pd.Series:
    """Mean market_index per date across the given feature frames.

    market_index is a market-wide daily value (within-day std ~0.025 vs ~0.17
    across days), so the mean over a day's loads is a reliable estimate for any
    load on that day. Uses features only, never labels.
    """
    parts = [f[[schema.DATE, "market_index"]] for f in frames if "market_index" in f]
    if not parts:
        return pd.Series(dtype=float, name="market_index")
    stacked = pd.concat(parts)
    return stacked.groupby(schema.DATE)["market_index"].mean().dropna().sort_index()


def city_coordinates(*frames: pd.DataFrame) -> pd.DataFrame:
    """One (lat, lon) per city, from pickup and delivery columns of the given frames.

    Each city has exactly one coordinate pair in the provided data, so taking
    the first occurrence is exact.
    """
    parts = []
    for frame in frames:
        for side in ("pickup", "delivery"):
            cols = [side, f"{side}_lat", f"{side}_lon"]
            if set(cols) <= set(frame.columns):
                parts.append(frame[cols].set_axis(["city", "lat", "lon"], axis=1))
    if not parts:
        return pd.DataFrame(columns=["lat", "lon"])
    return pd.concat(parts).dropna().drop_duplicates("city").set_index("city")


def _lookup_nearest(dates: pd.Series, table: pd.Series) -> pd.Series:
    """Value from ``table`` (indexed by date) at the nearest available date."""
    if table.empty:
        return pd.Series(np.nan, index=dates.index)
    order = np.argsort(dates.to_numpy())
    looked_up = pd.merge_asof(
        pd.DataFrame({schema.DATE: dates.to_numpy()[order], "_row": order}),
        table.rename("value").rename_axis(schema.DATE).reset_index(),
        on=schema.DATE,
        direction="nearest",
    )
    out = np.empty(len(dates))
    out[looked_up["_row"].to_numpy()] = looked_up["value"].to_numpy()
    return pd.Series(out, index=dates.index)


class LoadCleaner(BaseEstimator, TransformerMixin):
    """Repairs feature values. Learned statistics come from ``fit`` data only.

    * weight: negative values are sign errors -> absolute value; missing ->
      training median, with a ``weight_missing`` flag.
    * market_index: missing -> mean of the same day's other loads in the frame
      being transformed (features only); if that day has no values, the nearest
      date in the training table; finally the training median. A
      ``market_index_missing`` flag records the original gap.
    * coordinates: if absent (the December chart file has none), looked up
      from the training city table.
    * quote_signal: if absent or missing, the training median.
    """

    def fit(self, X: pd.DataFrame, y=None) -> LoadCleaner:
        self.weight_median_ = float(X["weight"].abs().median())
        self.market_index_median_ = float(X["market_index"].median())
        self.daily_market_index_ = daily_market_index(X)
        self.quote_signal_median_ = float(X["quote_signal"].median())
        self.city_coordinates_ = city_coordinates(X)
        return self

    def _fill_coordinates(self, out: pd.DataFrame) -> None:
        """Fill absent coordinates (e.g. the December chart file) from the city table."""
        for side in ("pickup", "delivery"):
            for axis in ("lat", "lon"):
                col = f"{side}_{axis}"
                if col not in out:
                    out[col] = np.nan
                gap = out[col].isna()
                if gap.any():
                    out.loc[gap, col] = out.loc[gap, side].map(self.city_coordinates_[axis])
                if out[col].isna().any():
                    unknown = sorted(set(out.loc[out[col].isna(), side]))
                    raise ValueError(f"no coordinates available for cities: {unknown}")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        self._fill_coordinates(out)

        if "quote_signal" not in out:
            out["quote_signal"] = np.nan
        out["quote_signal"] = out["quote_signal"].fillna(self.quote_signal_median_)

        weight = out["weight"].abs()
        out["weight_missing"] = weight.isna().astype(int)
        out["weight"] = weight.fillna(self.weight_median_)

        if "market_index" not in out:
            out["market_index"] = np.nan
        missing = out["market_index"].isna()
        out["market_index_missing"] = missing.astype(int)
        if missing.any():
            same_day = out.groupby(schema.DATE)["market_index"].transform("mean")
            filled = out["market_index"].fillna(same_day)
            still = filled.isna()
            if still.any():
                nearest = _lookup_nearest(out.loc[still, schema.DATE], self.daily_market_index_)
                filled.loc[still] = nearest
            out["market_index"] = filled.fillna(self.market_index_median_)
        return out


class LabelAnomalyFilter(BaseEstimator):
    """Flags labels whose rate per mile is far from comparable training loads.

    Reference = median rate per mile per (equipment, distance bin), with bin
    edges at training-distance quantiles. Medians are robust, so fitting on all
    training rows (including the anomalies) is safe.
    """

    def __init__(self, low_ratio: float = 0.5, high_ratio: float = 2.0, n_bins: int = 10):
        self.low_ratio = low_ratio
        self.high_ratio = high_ratio
        self.n_bins = n_bins

    def _bins(self, distance: pd.Series) -> np.ndarray:
        return np.digitize(distance.to_numpy(), self.bin_edges_)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> LabelAnomalyFilter:
        rpm = np.asarray(y, dtype=float) / X["distance"].to_numpy()
        self.bin_edges_ = np.quantile(X["distance"], np.linspace(0, 1, self.n_bins + 1)[1:-1])
        cells = pd.DataFrame(
            {"equipment": X["equipment"].to_numpy(), "bin": self._bins(X["distance"]), "rpm": rpm}
        )
        self.reference_ = cells.groupby(["equipment", "bin"])["rpm"].median()
        self.bin_reference_ = cells.groupby("bin")["rpm"].median()
        return self

    def ratio(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Rate per mile divided by the reference for the load's cell."""
        bins = self._bins(X["distance"])
        keys = pd.MultiIndex.from_arrays([X["equipment"].to_numpy(), bins])
        expected = self.reference_.reindex(keys).to_numpy()
        fallback = self.bin_reference_.reindex(bins).to_numpy()
        expected = np.where(np.isnan(expected), fallback, expected)
        rpm = np.asarray(y, dtype=float) / X["distance"].to_numpy()
        return pd.Series(rpm / expected, index=X.index)

    def is_anomaly(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        r = self.ratio(X, y)
        return (r < self.low_ratio) | (r > self.high_ratio)
