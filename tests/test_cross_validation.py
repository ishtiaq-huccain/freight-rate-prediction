"""End-to-end leakage checks on the evaluation harness, using a spy model."""

from __future__ import annotations

import numpy as np
import pytest

from freight_rate.data import io, schema
from freight_rate.evaluation.cross_validation import run_cv, summarize
from freight_rate.evaluation.metrics import regression_metrics
from freight_rate.models.baselines import MedianRatePerMile


class SpyModel(MedianRatePerMile):
    """Records what the harness passes to fit/predict."""

    calls: list = []

    def fit(self, X, y):
        SpyModel.calls.append(("fit", X.copy(), y.copy()))
        return super().fit(X, y)

    def predict(self, X):
        SpyModel.calls.append(("predict", X.copy(), None))
        return super().predict(X)


@pytest.fixture(scope="module")
def train(cfg):
    return io.load_train(cfg.paths.train)


@pytest.fixture(scope="module")
def spy_run(cfg, train):
    SpyModel.calls = []
    metrics, preds = run_cv(
        train, SpyModel, cfg.validation, cfg.cleaning, seed=cfg.seed, roles=("selection",)
    )
    return metrics, preds, list(SpyModel.calls)


def test_model_never_sees_target_or_id(spy_run):
    for _, X, _ in spy_run[2]:
        assert schema.TARGET not in X.columns
        assert schema.ID not in X.columns


def test_training_data_precedes_test_window(cfg, spy_run):
    calls = spy_run[2]
    fold_ends = {f.name: f.train_end for f in cfg.validation.folds}
    fits = [X for kind, X, _ in calls if kind == "fit"]
    assert max(X[schema.DATE].max() for X in fits) <= max(fold_ends.values())
    # The first fit of each fold is the full temporal fit; check against that fold's end.
    temporal_fits = fits[:: 1 + cfg.validation.city_holdout.n_repeats]
    for X, fold in zip(
        temporal_fits, [f for f in cfg.validation.folds if f.role == "selection"], strict=True
    ):
        assert X[schema.DATE].max() <= fold.train_end


def test_final_holdout_untouched_during_selection(cfg, spy_run):
    final = next(f for f in cfg.validation.folds if f.role == "final")
    for _, X, _ in spy_run[2]:
        assert X[schema.DATE].max() < final.test_start
    assert "holdout" not in set(spy_run[0]["fold"])


def test_every_test_row_gets_a_prediction(cfg, train, spy_run):
    preds = spy_run[1]
    for fold in cfg.validation.folds:
        if fold.role != "selection":
            continue
        expected = train[(train.date >= fold.test_start) & (train.date <= fold.test_end)]
        got = preds[preds.fold == fold.name]
        assert len(got) == len(expected)
        assert set(got[schema.ID]) == set(expected[schema.ID])
        assert np.isfinite(got["predicted"]).all()


def test_anomalies_dropped_from_training_only(spy_run):
    fits = [y for kind, _, y in spy_run[2] if kind == "fit"]
    rpm = fits[0] / spy_run[2][0][1]["distance"]
    assert rpm.max() < 6 and rpm.min() > 1  # no 0.3x / 4x labels in the training fit
    preds = spy_run[1]
    assert (~preds["clean"]).sum() > 0  # anomalous test rows are kept and scored


def test_metrics_table_shape(spy_run):
    metrics = spy_run[0]
    assert {"temporal", "cities_unseen", "cities_seen"} <= set(metrics["scenario"])
    summary = summarize(metrics)
    assert summary.loc[("temporal", "all"), "mae"] > summary.loc[("temporal", "clean"), "mae"]


def test_regression_metrics_values():
    m = regression_metrics([100, 200], [110, 180])
    assert m["mae"] == 15 and m["bias"] == -5
    assert m["rmse"] == pytest.approx(np.sqrt((100 + 400) / 2))
    assert m["mape_pct"] == pytest.approx((10 / 100 + 20 / 200) / 2 * 100)


def test_regression_metrics_shape_mismatch():
    with pytest.raises(ValueError):
        regression_metrics([1, 2], [1])


def test_shuffle_test_permutes_only_named_columns(cfg, train):
    """With shuffle_test, the model sees the same values in a different order."""
    SpyModel.calls = []
    run_cv(
        train,
        SpyModel,
        cfg.validation,
        cfg.cleaning,
        seed=cfg.seed,
        city_check=False,
        shuffle_test=("quote_signal",),
    )
    predict_X = [X for kind, X, _ in SpyModel.calls if kind == "predict"][0]
    original = train.loc[predict_X.index]
    assert not np.array_equal(
        predict_X["quote_signal"].to_numpy(), original["quote_signal"].to_numpy()
    )
    assert sorted(predict_X["quote_signal"]) == sorted(original["quote_signal"])
    assert np.array_equal(predict_X["distance"].to_numpy(), original["distance"].to_numpy())
