"""Leakage-safe evaluation harness.

For every fold, everything learned (cleaning statistics, anomaly reference,
model) is fitted on that fold's training window only. The target is removed
from the frames the model sees. Test rows are never dropped: anomaly flags on
test rows are used only to split the reported metrics into "all" and "clean".

A model is any object built by ``model_factory()`` with ``fit(X, y)`` and
``predict(X)``, where ``X`` is a cleaned DataFrame of input columns.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from freight_rate.config import CleaningConfig, ValidationConfig
from freight_rate.data import schema
from freight_rate.data.cleaning import LabelAnomalyFilter, LoadCleaner
from freight_rate.evaluation.metrics import regression_metrics
from freight_rate.validation.splits import city_holdout_masks, temporal_masks

log = logging.getLogger(__name__)

# load_id is assigned in date order (a time proxy), so it is never a feature.
NON_FEATURE_COLUMNS = [schema.ID, schema.TARGET]


@dataclass
class FittedFold:
    cleaner: LoadCleaner
    anomaly_filter: LabelAnomalyFilter
    model: object
    n_train_used: int
    n_train_dropped: int


def model_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    """Columns a model may see: everything except the ID and the target."""
    return frame.drop(columns=[c for c in NON_FEATURE_COLUMNS if c in frame])


def fit_on_window(
    train: pd.DataFrame,
    model_factory: Callable[[], object],
    cleaning: CleaningConfig,
    drop_anomalies: bool = True,
) -> FittedFold:
    """Fit cleaner, anomaly filter and model on one training window."""
    y = train[schema.TARGET]
    cleaner = LoadCleaner().fit(train)
    anomaly_filter = LabelAnomalyFilter(
        cleaning.anomaly_low_ratio, cleaning.anomaly_high_ratio, cleaning.anomaly_distance_bins
    ).fit(train, y)
    keep = (
        ~anomaly_filter.is_anomaly(train, y)
        if drop_anomalies
        else pd.Series(True, index=train.index)
    )
    model = model_factory()
    model.fit(model_inputs(cleaner.transform(train[keep])), y[keep])
    return FittedFold(cleaner, anomaly_filter, model, int(keep.sum()), int((~keep).sum()))


def predict_rows(fitted: FittedFold, frame: pd.DataFrame) -> np.ndarray:
    pred = np.asarray(
        fitted.model.predict(model_inputs(fitted.cleaner.transform(frame))), dtype=float
    )
    if pred.shape != (len(frame),):
        raise ValueError(f"model returned {pred.shape} predictions for {len(frame)} rows")
    return pred


def _segment_metrics(test: pd.DataFrame, pred: np.ndarray, clean: np.ndarray) -> list[dict]:
    y = test[schema.TARGET].to_numpy()
    month_index = test[schema.DATE].dt.to_period("M")
    first_month = month_index.min()
    horizon = np.asarray([(m - first_month).n + 1 for m in month_index])
    segments = {"all": np.ones(len(test), bool), "clean": clean}
    for h in np.unique(horizon):
        segments[f"clean_month_{h}"] = clean & (horizon == h)
    return [{"segment": name, **regression_metrics(y[m], pred[m])} for name, m in segments.items()]


def _shuffled(
    frame: pd.DataFrame, columns: tuple[str, ...], rng: np.random.Generator
) -> pd.DataFrame:
    if not columns:
        return frame
    return frame.assign(**{c: rng.permutation(frame[c].to_numpy()) for c in columns})


def run_cv(
    frame: pd.DataFrame,
    model_factory: Callable[[], object],
    validation: ValidationConfig,
    cleaning: CleaningConfig,
    seed: int = 42,
    roles: tuple[str, ...] = ("selection",),
    city_check: bool = True,
    drop_anomalies: bool = True,
    shuffle_test: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate a model on the configured folds.

    ``shuffle_test`` lists input columns to randomly permute within each test
    window before predicting. This keeps their distribution but removes their
    information. It's used to score models under Nov-Dec conditions, where
    quote_signal carries no price information (see freight_rate.data.quality).

    Returns ``(metrics, predictions)``. ``metrics`` has one row per fold x
    scenario x segment. ``predictions`` has one row per temporal test load.
    Only folds whose role is in ``roles`` are run, so the final holdout stays
    untouched during model selection.
    """
    rng = np.random.default_rng(seed)
    metric_rows, prediction_frames = [], []

    for fold in validation.folds:
        if fold.role not in roles:
            continue
        masks = temporal_masks(frame[schema.DATE], fold)
        train, test = frame.loc[masks.train], frame.loc[masks.test]

        fitted = fit_on_window(train, model_factory, cleaning, drop_anomalies)
        pred = predict_rows(fitted, _shuffled(test, shuffle_test, rng))
        clean = ~fitted.anomaly_filter.is_anomaly(test, test[schema.TARGET]).to_numpy()
        log.info(
            "%s: trained on %d rows (%d anomalies dropped), tested on %d",
            fold.name,
            fitted.n_train_used,
            fitted.n_train_dropped,
            len(test),
        )
        for row in _segment_metrics(test, pred, clean):
            metric_rows.append(
                {"fold": fold.name, "role": fold.role, "scenario": "temporal", **row}
            )
        prediction_frames.append(
            test[[schema.ID, schema.DATE, schema.TARGET]].assign(
                fold=fold.name, predicted=pred, clean=clean
            )
        )

        if not city_check:
            continue
        pred_by_id = pd.Series(pred, index=test.index)
        for repeat in range(validation.city_holdout.n_repeats):
            holdout = city_holdout_masks(frame, masks, validation.city_holdout.n_cities, rng)
            unseen_test = frame.loc[holdout.test]
            fitted_wo = fit_on_window(
                frame.loc[holdout.train], model_factory, cleaning, drop_anomalies
            )
            pred_unseen = predict_rows(fitted_wo, _shuffled(unseen_test, shuffle_test, rng))
            clean_u = ~fitted.anomaly_filter.is_anomaly(
                unseen_test, unseen_test[schema.TARGET]
            ).to_numpy()
            y_u = unseen_test[schema.TARGET].to_numpy()
            base = {
                "fold": fold.name,
                "role": fold.role,
                "repeat": repeat,
                "cities": ", ".join(holdout.cities),
            }
            # Same loads scored by the model that saw those cities: isolates the cold-start cost.
            seen_pred = pred_by_id.loc[unseen_test.index].to_numpy()
            metric_rows.append(
                {
                    **base,
                    "scenario": "cities_unseen",
                    "segment": "clean",
                    **regression_metrics(y_u[clean_u], pred_unseen[clean_u]),
                }
            )
            metric_rows.append(
                {
                    **base,
                    "scenario": "cities_seen",
                    "segment": "clean",
                    **regression_metrics(y_u[clean_u], seen_pred[clean_u]),
                }
            )

    return pd.DataFrame(metric_rows), pd.concat(prediction_frames, ignore_index=True)


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    """Mean metrics per scenario x segment across folds and repeats."""
    cols = ["mae", "rmse", "median_ae", "mape_pct", "bias"]
    return metrics.groupby(["scenario", "segment"], sort=False)[cols].mean().round(2)
