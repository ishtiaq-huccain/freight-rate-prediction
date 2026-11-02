"""Data-quality diagnostics that span more than one column.

quote_signal regimes
--------------------
Within a calendar month, quote_signal is an almost exact linear encoding of
the true rate per mile: rank correlation about 0.96, R^2 about 0.99 on clean
rows. The direction flips between months: positive in Jan-Mar, Jun and Sep,
inverted in Apr-May, Jul and Oct, and no relationship from 2 Aug onwards.
Pooled over the year the correlation is about 0.07, which hides all of this.

The regime of a period can be read *without labels*: correlate quote_signal
with the rate per mile expected from distance and equipment alone. On
labelled data this matches the labelled regime in 93% of weeks. In
validation (Nov-Dec) it's about 0 on every day, the same as August, so
quote_signal carries no price information there and the final model doesn't
use it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from freight_rate.data import schema
from freight_rate.models.baselines import GroupedMedianRatePerMile

# |corr| below this means "no regime"; regime periods sit near 0.9.
NO_REGIME_THRESHOLD = 0.3


def _corr(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def quote_signal_regimes(
    reference: pd.DataFrame, frame: pd.DataFrame, freq: str = "M"
) -> pd.DataFrame:
    """Per-period quote_signal regime of ``frame``.

    ``reference``: labelled loads, used only to estimate the rate per mile
    expected from distance and equipment.

    Returns one row per period:
      label_free_corr  corr(quote_signal, expected rate per mile). Needs no labels.
      labelled_corr    corr(quote_signal, actual rate per mile), if frame has labels.
      regime           "positive", "inverted" or "none" (from label_free_corr).
    """
    expected = GroupedMedianRatePerMile().fit(reference, reference[schema.TARGET])
    df = frame.assign(
        expected_rpm=expected.predict(frame) / frame["distance"].to_numpy(),
        period=frame[schema.DATE].dt.to_period(freq),
    )
    has_labels = schema.TARGET in frame
    if has_labels:
        df["rpm"] = df[schema.TARGET] / df["distance"]

    rows = []
    for period, g in df.groupby("period"):
        row = {
            "period": str(period),
            "loads": len(g),
            "label_free_corr": _corr(g["quote_signal"], g["expected_rpm"]),
        }
        if has_labels:
            row["labelled_corr"] = _corr(g["quote_signal"], g["rpm"])
        rows.append(row)
    out = pd.DataFrame(rows)
    out["regime"] = np.select(
        [
            out["label_free_corr"] > NO_REGIME_THRESHOLD,
            out["label_free_corr"] < -NO_REGIME_THRESHOLD,
        ],
        ["positive", "inverted"],
        "none",
    )
    return out
