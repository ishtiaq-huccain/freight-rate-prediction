"""Train/test splits that mimic how the model is used.

Deployment: fit on Jan-Oct 2025, predict Nov-Dec 2025 (1-2 months ahead), with
8 validation cities never seen in training. So:

* ``temporal_masks``: rolling-origin folds. Training data strictly precedes a
  two-month test window.
* ``city_holdout_masks``: on top of a temporal fold, hide some training cities
  and score test loads with exactly one endpoint in a hidden city (no
  validation load has both endpoints unseen).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from freight_rate.config import FoldSpec
from freight_rate.data import schema


@dataclass(frozen=True)
class FoldMasks:
    train: np.ndarray
    test: np.ndarray


def temporal_masks(dates: pd.Series, fold: FoldSpec) -> FoldMasks:
    """Boolean masks for one rolling-origin fold."""
    dates = pd.to_datetime(dates)
    train = (dates <= fold.train_end).to_numpy()
    test = ((dates >= fold.test_start) & (dates <= fold.test_end)).to_numpy()
    if not train.any() or not test.any():
        raise ValueError(f"fold {fold.name}: empty train or test window")
    return FoldMasks(train=train, test=test)


@dataclass(frozen=True)
class CityHoldout:
    cities: tuple[str, ...]
    train: np.ndarray  # training window minus every load touching a hidden city
    test: np.ndarray  # test loads with exactly one endpoint in a hidden city


def city_holdout_masks(
    frame: pd.DataFrame, masks: FoldMasks, n_cities: int, rng: np.random.Generator
) -> CityHoldout:
    """Hide ``n_cities`` random training cities from a temporal fold."""
    train_rows = frame.loc[masks.train]
    candidates = sorted(set(train_rows["pickup"]) | set(train_rows["delivery"]))
    if n_cities >= len(candidates):
        raise ValueError("n_cities must be smaller than the number of training cities")
    hidden = tuple(sorted(rng.choice(candidates, size=n_cities, replace=False)))

    pickup_hidden = frame["pickup"].isin(hidden).to_numpy()
    delivery_hidden = frame["delivery"].isin(hidden).to_numpy()
    touches = pickup_hidden | delivery_hidden
    exactly_one = pickup_hidden ^ delivery_hidden
    return CityHoldout(cities=hidden, train=masks.train & ~touches, test=masks.test & exactly_one)


def describe_fold(frame: pd.DataFrame, fold: FoldSpec, masks: FoldMasks) -> dict:
    dates = frame[schema.DATE]
    return {
        "fold": fold.name,
        "role": fold.role,
        "train_rows": int(masks.train.sum()),
        "train_dates": f"{dates[masks.train].min().date()} to {dates[masks.train].max().date()}",
        "test_rows": int(masks.test.sum()),
        "test_dates": f"{dates[masks.test].min().date()} to {dates[masks.test].max().date()}",
    }
