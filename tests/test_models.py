from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from freight_rate.data import io, schema
from freight_rate.data.cleaning import LoadCleaner
from freight_rate.evaluation.cross_validation import model_inputs
from freight_rate.models.baselines import GroupedMedianRatePerMile
from freight_rate.models.registry import make_factory


@pytest.fixture(scope="module")
def sample(cfg):
    train = io.load_train(cfg.paths.train).sample(3_000, random_state=0)
    val = io.load_validation(cfg.paths.validation).head(500)
    december = io.load_december_inputs(cfg.paths.december_inputs)
    cleaner = LoadCleaner().fit(train)
    return (
        model_inputs(cleaner.transform(train)),
        train[schema.TARGET],
        {
            "validation": model_inputs(cleaner.transform(val)),
            "december": model_inputs(cleaner.transform(december.drop(columns="predicted_rate"))),
        },
    )


@pytest.mark.parametrize("name", ["global_median", "grouped_median", "ridge", "hgb"])
def test_configured_models_fit_and_predict(cfg, sample, name):
    X, y, targets = sample
    model = make_factory(name, cfg.raw, cfg.seed)().fit(X, y)
    for label, frame in targets.items():
        pred = np.asarray(model.predict(frame))
        assert pred.shape == (len(frame),), label
        assert np.isfinite(pred).all() and (pred > 0).all(), label


def test_models_are_deterministic(cfg, sample):
    X, y, targets = sample
    a = make_factory("hgb", cfg.raw, cfg.seed)().fit(X, y).predict(targets["validation"])
    b = make_factory("hgb", cfg.raw, cfg.seed)().fit(X, y).predict(targets["validation"])
    np.testing.assert_array_equal(a, b)


def test_unknown_model_rejected(cfg):
    with pytest.raises(ValueError, match="unknown model"):
        make_factory("does_not_exist", cfg.raw)


def test_unknown_feature_set_rejected(cfg):
    raw = {**cfg.raw, "models": {"m": {"type": "ridge", "features": "nope"}}}
    with pytest.raises(ValueError, match="feature set"):
        make_factory("m", raw)


def test_grouped_median_uses_cell_medians():
    X = pd.DataFrame(
        {"equipment": ["Dry Van"] * 4 + ["Reefer"] * 4, "distance": [100, 100, 1000, 1000] * 2}
    ).astype({"distance": float})
    y = pd.Series([200, 220, 1500, 1700, 300, 320, 2000, 2200], dtype=float)
    model = GroupedMedianRatePerMile(n_bins=2).fit(X, y)
    pred = model.predict(
        pd.DataFrame({"equipment": ["Dry Van", "Reefer"], "distance": [1000.0, 100.0]})
    )
    np.testing.assert_allclose(pred, [1000 * 1.6, 100 * 3.1])


def test_grouped_median_falls_back_for_unseen_cell():
    X = pd.DataFrame({"equipment": ["Dry Van"] * 4, "distance": [100.0, 200.0, 300.0, 400.0]})
    model = GroupedMedianRatePerMile(n_bins=2).fit(X, pd.Series([200.0, 400.0, 600.0, 800.0]))
    pred = model.predict(pd.DataFrame({"equipment": ["Flatbed"], "distance": [100.0]}))
    assert np.isfinite(pred).all()


# ----------------------------------------------------------------------------- Phase 4 registry


def test_extends_merges_params_one_level():
    from freight_rate.models.registry import resolve_spec

    models = {
        "a": {"type": "hgb", "features": "base", "params": {"max_iter": 10, "learning_rate": 0.1}},
        "b": {"extends": "a", "params": {"max_iter": 20}},
    }
    spec = resolve_spec("b", models)
    assert spec["params"] == {"max_iter": 20, "learning_rate": 0.1}
    assert spec["type"] == "hgb" and "extends" not in spec


def test_circular_extends_rejected():
    from freight_rate.models.registry import resolve_spec

    with pytest.raises(ValueError, match="circular"):
        resolve_spec("a", {"a": {"extends": "b"}, "b": {"extends": "a"}})


def test_features_add_and_drop(cfg):
    from freight_rate.models.registry import resolve_features

    sets = cfg.raw["feature_sets"]
    groups = resolve_features(
        {"features": "base", "features_drop": ["quote_signal"], "features_add": ["time_index"]},
        sets,
    )
    assert "quote_signal" not in groups and groups[-1] == "time_index"
    with pytest.raises(ValueError, match="features_drop"):
        resolve_features({"features": "base", "features_drop": ["not_there"]}, sets)


def test_recency_weights_halve_every_half_life():
    from freight_rate.models.registry import RecencyWeighted

    class Capture:
        def fit(self, X, y, sample_weight=None):
            self.w = sample_weight
            return self

    X = pd.DataFrame({"date": pd.to_datetime(["2025-01-01", "2025-01-31", "2025-03-02"])})
    inner = Capture()
    RecencyWeighted(inner, half_life_days=30, weight_param="sample_weight").fit(X, [1, 2, 3])
    np.testing.assert_allclose(inner.w, [0.25, 0.5, 1.0])


def test_blend_is_weighted_geometric_mean():
    from freight_rate.models.registry import Blend

    class Const:
        def __init__(self, v):
            self.v = v

        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.full(len(X), self.v)

    pred = Blend([Const(100.0), Const(400.0)], [0.5, 0.5]).fit(None, None).predict([0, 1])
    np.testing.assert_allclose(pred, [200.0, 200.0])


def test_final_model_configured_and_trains(cfg, sample):
    from freight_rate.models.registry import training_options

    assert cfg.raw["final_model"] == "final"
    assert training_options("final", cfg.raw)["drop_anomalies"] is True
    X, y, targets = sample
    model = make_factory("final", cfg.raw, cfg.seed)().fit(X, y)
    for frame in targets.values():
        pred = model.predict(frame)
        assert np.isfinite(pred).all() and (pred > 0).all()


def test_experiment_config_cannot_redefine_base_models(cfg, tmp_path):
    from freight_rate.pipelines.experiment import merge_experiment_config

    path = tmp_path / "x.yaml"
    path.write_text("models:\n  hgb: {type: hgb, features: base}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="redefines"):
        merge_experiment_config(cfg.raw, path)


def test_phase4_experiment_config_resolves(cfg):
    from freight_rate.config import PROJECT_ROOT
    from freight_rate.pipelines.experiment import merge_experiment_config

    raw = merge_experiment_config(cfg.raw, PROJECT_ROOT / "configs/experiments/phase4.yaml")
    for name in raw["models"]:
        make_factory(name, raw, cfg.seed)  # every variant must build
