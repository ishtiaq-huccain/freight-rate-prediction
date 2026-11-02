from __future__ import annotations

import pandas as pd

from freight_rate.evaluation.error_analysis import segment_errors, worst_lanes


def _data():
    loads = pd.DataFrame(
        {
            "load_id": ["a", "b", "c", "d"],
            "date": pd.to_datetime(["2025-05-01", "2025-05-02", "2025-06-01", "2025-06-02"]),
            "posted_rate": [100.0, 200.0, 300.0, 400.0],
            "distance": [100.0, 600.0, 1200.0, 3000.0],
            "equipment": ["Dry Van", "Reefer", "Dry Van", "Flatbed"],
            "weight": [10_000.0, -25_000.0, 35_000.0, 47_500.0],
            "pickup": "X",
            "delivery": "Y",
        }
    )
    oof = loads[["load_id", "date", "posted_rate"]].assign(
        predicted=[110.0, 180.0, 300.0, 1000.0], clean=[True, True, True, False], fold="f"
    )
    return oof, loads


def test_segment_errors_excludes_anomalies_and_computes_bias():
    oof, loads = _data()
    tables = segment_errors(oof, loads)
    month = tables["month"]
    assert month["loads"].sum() == 3  # the anomalous row is excluded
    assert month.loc["2025-05", "MAE $"] == 15.0 and month.loc["2025-05", "bias $"] == -5.0
    assert set(tables) == {"month", "equipment", "distance band", "weight band"}
    assert tables["weight band"]["loads"].sum() == 3  # negative weight banded by its absolute value


def test_worst_lanes_respects_minimum():
    oof, loads = _data()
    assert worst_lanes(oof, loads, min_loads=10).empty
    assert len(worst_lanes(oof, loads, min_loads=1)) == 1
