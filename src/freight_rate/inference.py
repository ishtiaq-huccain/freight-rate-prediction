"""Inference: produce the two submission files from a fitted model.

The same ``predict_rows`` used in cross-validation (cleaner -> feature
pipeline -> model) produces every prediction, so validation and December rows
go through exactly the code path that was evaluated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from freight_rate.data import schema
from freight_rate.data.cleaning import daily_market_index
from freight_rate.evaluation.cross_validation import FittedFold, predict_rows

EXPECTED_VALIDATION_ROWS = 12_000
DECEMBER_DATES = pd.date_range("2025-12-01", "2025-12-31", freq="D")


class SubmissionError(ValueError):
    """Raised when an output would be rejected by score.py."""


# ----------------------------------------------------------------------------- validation file


def predict_validation(
    model: FittedFold, validation: pd.DataFrame, template: pd.DataFrame
) -> pd.DataFrame:
    """Fill the template's predicted_rate, keeping the template's row order."""
    pred = pd.Series(predict_rows(model, validation), index=validation[schema.ID].to_numpy())
    if pred.index.duplicated().any():
        raise SubmissionError("validation contains duplicate load_id values")
    missing = set(template[schema.ID]) - set(pred.index)
    if missing:
        raise SubmissionError(f"{len(missing)} template load_ids have no validation row")
    out = template[[schema.ID]].copy()
    out["predicted_rate"] = pred.reindex(out[schema.ID]).to_numpy()
    check_validation_predictions(out)
    return out


def check_validation_predictions(frame: pd.DataFrame) -> None:
    """The same acceptance rules as score.py's validate_predictions."""
    if list(frame.columns) != schema.TEMPLATE_COLUMNS:
        raise SubmissionError("columns must be exactly load_id,predicted_rate")
    if len(frame) != EXPECTED_VALIDATION_ROWS:
        raise SubmissionError(f"expected {EXPECTED_VALIDATION_ROWS} rows, got {len(frame)}")
    if frame[schema.ID].isna().any() or frame[schema.ID].duplicated().any():
        raise SubmissionError("missing or duplicate load_id values")
    expected_ids = {f"TE-{i:06d}" for i in range(1, EXPECTED_VALIDATION_ROWS + 1)}
    if set(frame[schema.ID].astype(str)) != expected_ids:
        raise SubmissionError("load_ids do not match the validation set")
    _check_rates(frame["predicted_rate"], "validation predictions")


def _check_rates(values: pd.Series, label: str) -> None:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any() or not np.isfinite(numeric).all():
        raise SubmissionError(f"{label}: missing or non-finite predicted_rate")
    if (numeric <= 0).any():
        raise SubmissionError(f"{label}: non-positive predicted_rate")


# ----------------------------------------------------------------------------- December chart


def attach_market_index(frame: pd.DataFrame, *reference: pd.DataFrame) -> pd.DataFrame:
    """Add market_index to a frame that lacks it (the December chart inputs).

    Uses the mean market_index of the loads in ``reference`` on the same date.
    These are feature values from validation.csv, not labels. market_index is a
    market-wide daily value, so this is what any real load on that date would
    carry.
    """
    daily = daily_market_index(*reference)
    out = frame.copy()
    out["market_index"] = out[schema.DATE].map(daily)
    missing = out.loc[out["market_index"].isna(), schema.DATE]
    if len(missing):
        raise SubmissionError(
            f"no reference market_index for dates {sorted(missing.dt.date.unique())}"
        )
    return out


def predict_december(
    model: FittedFold, december: pd.DataFrame, validation: pd.DataFrame
) -> pd.DataFrame:
    """Fill predicted_rate in the December chart frame, keeping its seven columns."""
    inputs = attach_market_index(december.drop(columns="predicted_rate"), validation)
    out = december.copy()
    out["predicted_rate"] = predict_rows(model, inputs)
    check_december_predictions(out)
    return out


def check_december_predictions(frame: pd.DataFrame) -> None:
    """The same acceptance rules as score.py's validate_december."""
    if list(frame.columns) != schema.DECEMBER_COLUMNS:
        raise SubmissionError("December file must keep its seven columns in order")
    dates = pd.to_datetime(frame[schema.DATE], errors="coerce")
    if dates.isna().any() or dates.duplicated().any() or set(dates) != set(DECEMBER_DATES):
        raise SubmissionError("December file needs exactly one row per day of December 2025")
    fixed = {"pickup": "Lexington", "delivery": "Fort Wayne", "equipment": "Dry Van"}
    for col, value in fixed.items():
        if not frame[col].eq(value).all():
            raise SubmissionError(f"December {col} must be {value}")
    if (
        not np.isclose(frame["distance"], 360.0).all()
        or not np.isclose(frame["weight"], 32_000.0).all()
    ):
        raise SubmissionError("December distance/weight must be 360 / 32000")
    _check_rates(frame["predicted_rate"], "December predictions")


def write_december(frame: pd.DataFrame, path) -> None:
    """Write with dates as YYYY-MM-DD, matching the original file."""
    out = frame.copy()
    out[schema.DATE] = pd.to_datetime(out[schema.DATE]).dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False)
