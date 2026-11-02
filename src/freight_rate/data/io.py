"""Loading and structural validation of the provided CSV files.

The loaders check structure only (columns, IDs, parseable dates, valid
categories). They do not clean values: cleaning decisions belong to the feature
pipeline so they are applied identically at training and inference time.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from freight_rate.data import schema


class DataValidationError(ValueError):
    """Raised when an input file breaks its expected contract."""


def _read(path: Path) -> pd.DataFrame:
    if not Path(path).is_file():
        raise FileNotFoundError(f"input file not found: {path}")
    return pd.read_csv(path)


def _check_columns(frame: pd.DataFrame, expected: list[str], label: str) -> None:
    if list(frame.columns) != expected:
        raise DataValidationError(
            f"{label}: expected columns {expected}, got {list(frame.columns)}"
        )


def _check_loads(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    nulls = frame[schema.REQUIRED_NON_NULL].isna().sum()
    if nulls.any():
        raise DataValidationError(f"{label}: missing values in {nulls[nulls > 0].to_dict()}")
    if frame[schema.ID].duplicated().any():
        raise DataValidationError(f"{label}: duplicate {schema.ID} values")

    unknown = set(frame["equipment"]) - set(schema.EQUIPMENT_TYPES)
    if unknown:
        raise DataValidationError(f"{label}: unknown equipment types {sorted(unknown)}")

    frame = frame.copy()
    frame[schema.DATE] = pd.to_datetime(frame[schema.DATE], format="%Y-%m-%d", errors="coerce")
    if frame[schema.DATE].isna().any():
        raise DataValidationError(f"{label}: unparseable dates")
    if (frame["distance"] <= 0).any():
        raise DataValidationError(f"{label}: non-positive distance values")
    return frame


def load_train(path: Path) -> pd.DataFrame:
    frame = _read(path)
    _check_columns(frame, schema.TRAIN_COLUMNS, "train")
    frame = _check_loads(frame, "train")
    target = frame[schema.TARGET]
    if target.isna().any() or (target <= 0).any():
        raise DataValidationError("train: target must be present and positive")
    return frame


def load_validation(path: Path) -> pd.DataFrame:
    frame = _read(path)
    _check_columns(frame, schema.VALIDATION_COLUMNS, "validation")
    return _check_loads(frame, "validation")


def load_template(path: Path) -> pd.DataFrame:
    frame = _read(path)
    _check_columns(frame, schema.TEMPLATE_COLUMNS, "template")
    return frame


def load_december_inputs(path: Path) -> pd.DataFrame:
    frame = _read(path)
    _check_columns(frame, schema.DECEMBER_COLUMNS, "december inputs")
    frame = frame.copy()
    frame[schema.DATE] = pd.to_datetime(frame[schema.DATE], format="%Y-%m-%d", errors="coerce")
    if frame[schema.DATE].isna().any():
        raise DataValidationError("december inputs: unparseable dates")
    return frame
