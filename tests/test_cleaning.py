from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from freight_rate.data import io, schema
from freight_rate.data.cleaning import LabelAnomalyFilter, LoadCleaner, daily_market_index


def _frame():
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01"] * 3 + ["2025-01-02"] * 2 + ["2025-01-05"]),
            "equipment": "Dry Van",
            "distance": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
            "weight": [10_000.0, -20_000.0, np.nan, 30_000.0, 40_000.0, 50_000.0],
            "market_index": [1.0, 1.2, np.nan, 0.8, np.nan, np.nan],
            "quote_signal": [2.0, 2.2, 1.8, 2.0, np.nan, 2.4],
            "pickup": ["A", "B", "A", "C", "B", "A"],
            "delivery": ["B", "C", "C", "A", "A", "B"],
            "pickup_lat": [30.0, 31.0, 30.0, 32.0, 31.0, 30.0],
            "pickup_lon": [-90.0, -91.0, -90.0, -92.0, -91.0, -90.0],
            "delivery_lat": [31.0, 32.0, 32.0, 30.0, 30.0, 31.0],
            "delivery_lon": [-91.0, -92.0, -92.0, -90.0, -90.0, -91.0],
        }
    )


def test_quote_signal_filled_with_training_median():
    out = LoadCleaner().fit(_frame()).transform(_frame())
    assert out.loc[4, "quote_signal"] == pytest.approx(2.0)  # median of 2.0, 2.2, 1.8, 2.0, 2.4


def test_coordinates_filled_from_city_table_when_absent():
    cleaner = LoadCleaner().fit(_frame())
    sparse = _frame().drop(columns=["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"])
    out = cleaner.transform(sparse)
    pd.testing.assert_frame_equal(
        out[["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]],
        _frame()[["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]],
    )


def test_unknown_city_without_coordinates_raises():
    cleaner = LoadCleaner().fit(_frame())
    sparse = _frame().drop(columns=["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"])
    sparse.loc[0, "pickup"] = "Atlantis"
    with pytest.raises(ValueError, match="Atlantis"):
        cleaner.transform(sparse)


# ----------------------------------------------------------------------------- LoadCleaner


def test_weight_abs_and_median_fill():
    out = LoadCleaner().fit(_frame()).transform(_frame())
    assert (out["weight"] > 0).all()
    assert out.loc[1, "weight"] == 20_000.0
    # median of |weights| = median(10k, 20k, 30k, 40k, 50k) = 30k
    assert out.loc[2, "weight"] == 30_000.0
    assert out["weight_missing"].tolist() == [0, 0, 1, 0, 0, 0]


def test_market_index_same_day_then_nearest_day():
    out = LoadCleaner().fit(_frame()).transform(_frame())
    assert out.loc[2, "market_index"] == pytest.approx(1.1)  # same-day mean of 1.0 and 1.2
    assert out.loc[4, "market_index"] == pytest.approx(0.8)  # same-day value
    # 2025-01-05 has no value that day -> nearest training date (2025-01-02 -> 0.8)
    assert out.loc[5, "market_index"] == pytest.approx(0.8)
    assert out["market_index_missing"].tolist() == [0, 0, 1, 0, 1, 1]


def test_market_index_column_absent_is_filled():
    cleaner = LoadCleaner().fit(_frame())
    december_like = _frame().drop(columns="market_index")
    out = cleaner.transform(december_like)
    assert out["market_index"].notna().all()
    assert (out["market_index_missing"] == 1).all()


def test_cleaner_never_drops_rows_or_mutates_input():
    frame = _frame()
    before = frame.copy()
    out = LoadCleaner().fit(frame).transform(frame)
    assert len(out) == len(frame) and out.index.equals(frame.index)
    pd.testing.assert_frame_equal(frame, before)


def test_cleaner_state_depends_only_on_fit_data():
    train = _frame()
    test = _frame().assign(weight=[1e9] * 6, market_index=[99.0] * 6)
    cleaner = LoadCleaner().fit(train)
    state = (
        cleaner.weight_median_,
        cleaner.market_index_median_,
        cleaner.daily_market_index_.copy(),
    )
    cleaner.transform(test)
    assert cleaner.weight_median_ == state[0]
    assert cleaner.market_index_median_ == state[1]
    pd.testing.assert_series_equal(cleaner.daily_market_index_, state[2])


def test_cleaner_ignores_target(cfg):
    train = io.load_train(cfg.paths.train).head(2_000)
    shuffled = train.assign(
        **{schema.TARGET: train[schema.TARGET].sample(frac=1, random_state=0).to_numpy()}
    )
    a = LoadCleaner().fit(train).transform(train)
    b = LoadCleaner().fit(shuffled).transform(shuffled)
    features = [c for c in a.columns if c != schema.TARGET]
    pd.testing.assert_frame_equal(a[features], b[features])


def test_cleaned_real_data_has_no_gaps(cfg):
    train = io.load_train(cfg.paths.train)
    val = io.load_validation(cfg.paths.validation)
    cleaner = LoadCleaner().fit(train)
    for frame in (train, val):
        out = cleaner.transform(frame)
        assert out[["weight", "market_index"]].notna().all().all()
        assert (out["weight"] > 0).all()
        assert len(out) == len(frame)


def test_daily_market_index_uses_all_frames():
    a = _frame().iloc[:3]
    b = _frame().iloc[3:]
    table = daily_market_index(a, b)
    assert table.loc["2025-01-01"] == pytest.approx(1.1)
    assert table.loc["2025-01-02"] == pytest.approx(0.8)
    assert "2025-01-05" not in table.index.strftime("%Y-%m-%d")


# ----------------------------------------------------------------------------- LabelAnomalyFilter


def _priced(n=500, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {"equipment": rng.choice(["Dry Van", "Reefer"], n), "distance": rng.uniform(100, 3000, n)}
    )
    y = pd.Series(X["distance"] * 2.0 * rng.uniform(0.9, 1.1, n))
    return X, y


def test_anomaly_filter_flags_planted_labels_only():
    X, y = _priced()
    y.iloc[0] *= 4
    y.iloc[1] *= 0.25
    flags = LabelAnomalyFilter().fit(X, y).is_anomaly(X, y)
    assert flags.iloc[0] and flags.iloc[1]
    assert not flags.iloc[2:].any()


def test_anomaly_filter_reference_comes_from_fit_data_only():
    X, y = _priced()
    filt = LabelAnomalyFilter().fit(X, y)
    reference = filt.reference_.copy()
    X_test, y_test = _priced(seed=1)
    filt.is_anomaly(X_test, y_test * 10)  # scoring different data must not refit
    pd.testing.assert_series_equal(filt.reference_, reference)


def test_anomaly_filter_handles_unseen_equipment_cell():
    X, y = _priced()
    filt = LabelAnomalyFilter().fit(X, y)
    X_new = pd.DataFrame({"equipment": ["Flatbed"], "distance": [1000.0]})
    ratio = filt.ratio(X_new, pd.Series([2000.0]))
    assert np.isfinite(ratio).all()


def test_anomaly_filter_on_real_training_data(cfg):
    train = io.load_train(cfg.paths.train)
    flags = (
        LabelAnomalyFilter()
        .fit(train, train[schema.TARGET])
        .is_anomaly(train, train[schema.TARGET])
    )
    # EDA found 677 (1.41%) with a decile reference; bins differ slightly here
    assert 0.012 < flags.mean() < 0.016
