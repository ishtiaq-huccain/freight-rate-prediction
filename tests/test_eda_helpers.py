from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from freight_rate.analysis import eda


def test_haversine_known_distance():
    # New York (40.7128, -74.0060) to Los Angeles (34.0522, -118.2437) is ~2,445 miles
    miles = eda.haversine_miles(40.7128, -74.0060, 34.0522, -118.2437)
    assert miles == pytest.approx(2445, rel=0.01)


def test_haversine_zero_for_same_point():
    assert eda.haversine_miles(35.0, -90.0, 35.0, -90.0) == pytest.approx(0.0)


def _synthetic_train(n=200, seed=0):
    rng = np.random.default_rng(seed)
    distance = rng.uniform(100, 3000, n)
    frame = pd.DataFrame(
        {
            "load_id": [f"TR-{i:06d}" for i in range(n)],
            "pickup": "A",
            "delivery": "B",
            "pickup_lat": 35.0,
            "pickup_lon": -90.0,
            "delivery_lat": 36.0,
            "delivery_lon": -85.0,
            "distance": distance,
            "equipment": "Dry Van",
            "weight": 30_000.0,
            "date": pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 300, n), "D"),
            "market_index": 1.0,
            "quote_signal": 2.0,
            "posted_rate": distance * 2.0 * rng.uniform(0.95, 1.05, n),
        }
    )
    frame.loc[0, "posted_rate"] *= 4.0  # inflated label
    frame.loc[1, "posted_rate"] *= 0.25  # deflated label
    return frame


def test_reference_columns_flag_only_planted_anomalies():
    df = eda.add_reference_columns(_synthetic_train())
    assert df.loc[0, "label_anomaly"] == "high"
    assert df.loc[1, "label_anomaly"] == "low"
    assert (df.loc[2:, "label_anomaly"] == "normal").all()


def test_reference_columns_do_not_modify_input():
    train = _synthetic_train()
    before = train.copy()
    eda.add_reference_columns(train)
    pd.testing.assert_frame_equal(train, before)


def test_outlier_impact_reports_every_variant_and_subset():
    df = eda.add_reference_columns(_synthetic_train(n=600))
    table = eda.outlier_impact(df)
    assert len(table) == 6
    assert set(table["training"]) == {"train on all rows", "train excluding anomalies"}
    # The anomalous subset can be empty in synthetic data; the other two never are.
    populated = table[table["evaluated on"] != "anomalous rows"]
    assert populated[["MAE", "RMSE"]].notna().all().all()
    assert (populated["MAE"] > 0).all()
