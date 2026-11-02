"""quote_signal regime detection and the guard that the final model doesn't use it."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from freight_rate.data import io
from freight_rate.data.quality import quote_signal_regimes
from freight_rate.models.registry import resolve_features, resolve_spec


def _synthetic(sign: int, n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Loads whose quote_signal encodes rate per mile with the given sign (0 = noise)."""
    rng = np.random.default_rng(seed)
    distance = rng.uniform(100, 3000, n)
    rpm = 3.0 - 0.0004 * distance + rng.normal(0, 0.02, n)
    if sign > 0:
        qs = rpm
    elif sign < 0:
        qs = 4.1 - rpm
    else:
        qs = rng.normal(2.05, 0.22, n)
    return pd.DataFrame(
        {
            "date": pd.Timestamp("2025-01-15"),
            "equipment": "Dry Van",
            "distance": distance,
            "quote_signal": qs,
            "posted_rate": rpm * distance,
        }
    )


@pytest.mark.parametrize("sign, expected", [(1, "positive"), (-1, "inverted"), (0, "none")])
def test_regime_detected_without_labels(sign, expected):
    reference = _synthetic(1, seed=1)
    frame = _synthetic(sign).drop(columns="posted_rate")
    out = quote_signal_regimes(reference, frame)
    assert out["regime"].tolist() == [expected]
    assert "labelled_corr" not in out


def test_real_data_regimes(cfg):
    train = io.load_train(cfg.paths.train)
    val = io.load_validation(cfg.paths.validation)
    train_regimes = quote_signal_regimes(train, train).set_index("period")["regime"]
    expected = {
        "2025-01": "positive",
        "2025-02": "positive",
        "2025-03": "positive",
        "2025-04": "inverted",
        "2025-05": "inverted",
        "2025-06": "positive",
        "2025-07": "inverted",
        "2025-08": "none",
        "2025-09": "positive",
        "2025-10": "inverted",
    }
    assert train_regimes.to_dict() == expected
    assert set(quote_signal_regimes(train, val)["regime"]) == {"none"}


def test_final_model_does_not_use_quote_signal(cfg):
    spec = resolve_spec(cfg.raw["final_model"], cfg.raw["models"])
    for member in spec["members"]:
        groups = resolve_features(resolve_spec(member, cfg.raw["models"]), cfg.raw["feature_sets"])
        assert "quote_signal" not in groups, member


def test_feature_edits_accumulate_along_extends_chain():
    models = {
        "a": {"type": "hgb", "features": "base"},
        "b": {"extends": "a", "features_drop": ["weight_missing"]},
        "c": {"extends": "b", "features_drop": ["quote_signal"]},
        "d": {"extends": "c", "features": "final"},
    }
    assert resolve_spec("c", models)["features_drop"] == ["weight_missing", "quote_signal"]
    assert "features_drop" not in resolve_spec("d", models)  # a new feature set starts afresh
