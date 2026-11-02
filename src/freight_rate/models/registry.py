"""Builds model factories from the ``models`` / ``feature_sets`` config sections.

Every learned model is a scikit-learn Pipeline whose first step is
``FeatureBuilder``, so the exact same feature code runs in cross-validation,
final training and inference. Learned models predict log(rate) and are
transformed back with exp. Rates span ~$60 to ~$7k and errors scale with
distance, so a log target makes errors roughly proportional. The back-
transformed prediction estimates the median rate, which suits MAE.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from freight_rate.data import schema
from freight_rate.features.build import FeatureBuilder
from freight_rate.models.baselines import GroupedMedianRatePerMile, MedianRatePerMile

ModelFactory = Callable[[], object]


def _log_target(regressor) -> TransformedTargetRegressor:
    return TransformedTargetRegressor(regressor=regressor, func=np.log, inverse_func=np.exp)


def _ridge(groups: list[str], alpha: float = 1.0, spline_knots: int = 6) -> object:
    # Rate per mile falls non-linearly with distance: a cubic spline on
    # log(distance) captures the curve; every other feature is linear.
    if "log_distance" in groups:
        prep = ColumnTransformer(
            [
                (
                    "distance_spline",
                    SplineTransformer(n_knots=spline_knots, degree=3),
                    ["log_distance"],
                )
            ],
            remainder=StandardScaler(),
        )
    else:
        prep = StandardScaler()
    return _log_target(
        Pipeline(
            [("features", FeatureBuilder(groups)), ("prep", prep), ("ridge", Ridge(alpha=alpha))]
        )
    )


def _hgb(groups: list[str], seed: int, params: dict[str, Any] | None = None) -> object:
    return _log_target(
        Pipeline(
            [
                ("features", FeatureBuilder(groups)),
                ("hgb", HistGradientBoostingRegressor(random_state=seed, **(params or {}))),
            ]
        )
    )


class RecencyWeighted:
    """Wraps a model so recent training loads weigh more.

    weight = 0.5 ** (age_days / half_life_days), where age is measured from the
    latest training date. Weights come from the training rows' own dates, so
    nothing from the test window is used.
    """

    def __init__(self, estimator, half_life_days: float, weight_param: str):
        self.estimator = estimator
        self.half_life_days = half_life_days
        self.weight_param = weight_param

    def fit(self, X, y):
        dates = pd.to_datetime(X[schema.DATE])
        age = (dates.max() - dates).dt.days.to_numpy(dtype=float)
        weights = 0.5 ** (age / self.half_life_days)
        self.estimator.fit(X, y, **{self.weight_param: weights})
        return self

    def predict(self, X):
        return self.estimator.predict(X)


# Feature edits accumulate along an ``extends`` chain, so each variant states only
# its own change ("drop X relative to my parent").
_ACCUMULATING_KEYS = ("features_drop", "features_add")


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if key in _ACCUMULATING_KEYS:
            out[key] = list(dict.fromkeys([*out.get(key, []), *value]))
        elif isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = {**out[key], **value}
        else:
            out[key] = value
    if "features" in override:
        # A child that picks a new feature set starts its feature edits afresh.
        for key in _ACCUMULATING_KEYS:
            if key not in override:
                out.pop(key, None)
    return out


def resolve_spec(name: str, models: dict[str, Any], _seen: tuple[str, ...] = ()) -> dict[str, Any]:
    """Model spec with ``extends`` inheritance applied.

    Dict values merge one level deep; ``features_drop`` / ``features_add``
    accumulate along the chain; other values are replaced by the child.
    """
    if name not in models:
        raise ValueError(f"unknown model '{name}'; configured: {sorted(models)}")
    if name in _seen:
        raise ValueError(f"circular 'extends' chain: {' -> '.join((*_seen, name))}")
    spec = dict(models[name])
    parent = spec.pop("extends", None)
    if parent is None:
        return spec
    return _merge(resolve_spec(parent, models, (*_seen, name)), spec)


def resolve_features(spec: dict[str, Any], feature_sets: dict[str, list[str]]) -> list[str]:
    features = spec.get("features")
    if isinstance(features, str):
        if features not in feature_sets:
            raise ValueError(f"unknown feature set '{features}'")
        groups = list(feature_sets[features])
    elif isinstance(features, list):
        groups = list(features)
    else:
        raise ValueError("model spec needs 'features': a feature-set name or a list of groups")
    missing = [g for g in spec.get("features_drop", []) if g not in groups]
    if missing:
        raise ValueError(f"features_drop names groups not in the set: {missing}")
    groups = [g for g in groups if g not in spec.get("features_drop", [])]
    groups += [g for g in spec.get("features_add", []) if g not in groups]
    return groups


def training_options(name: str, raw_config: dict[str, Any]) -> dict[str, Any]:
    """Harness-level options for a model (currently: whether to drop label anomalies)."""
    spec = resolve_spec(name, raw_config.get("models", {}))
    return {"drop_anomalies": bool(spec.get("drop_anomalies", True))}


def make_factory(name: str, raw_config: dict[str, Any], seed: int = 42) -> ModelFactory:
    """Zero-argument factory for the named model in the config."""
    spec = resolve_spec(name, raw_config.get("models", {}))
    kind = spec.pop("type")
    feature_sets = raw_config.get("feature_sets", {})
    half_life = spec.get("recency_half_life_days")

    def weighted(build: Callable[[], object], step: str) -> ModelFactory:
        if half_life is None:
            return build
        return lambda: RecencyWeighted(build(), float(half_life), f"{step}__sample_weight")

    if kind == "global_median":
        return MedianRatePerMile
    if kind == "grouped_median":
        n_bins = int(spec.get("n_bins", 10))
        return lambda: GroupedMedianRatePerMile(n_bins=n_bins)
    if kind == "ridge":
        groups = resolve_features(spec, feature_sets)
        alpha, knots = float(spec.get("alpha", 1.0)), int(spec.get("spline_knots", 6))
        return weighted(lambda: _ridge(groups, alpha, knots), "ridge")
    if kind == "hgb":
        groups = resolve_features(spec, feature_sets)
        params = dict(spec.get("params", {}))
        return weighted(lambda: _hgb(groups, seed, params), "hgb")
    if kind == "blend":
        members = list(spec.get("members", []))
        weights = list(spec.get("weights", [1.0] * len(members)))
        if len(members) < 2 or len(weights) != len(members):
            raise ValueError(f"model '{name}': blend needs >= 2 members and one weight each")
        factories = [make_factory(m, raw_config, seed) for m in members]
        return lambda: Blend([f() for f in factories], weights)
    raise ValueError(f"model '{name}': unknown type '{kind}'")


class Blend:
    """Weighted geometric mean of member predictions (an average in log space,
    matching the log-target members)."""

    def __init__(self, members: list, weights: list[float]):
        self.members = members
        self.weights = np.asarray(weights, dtype=float) / np.sum(weights)

    def fit(self, X, y):
        for member in self.members:
            member.fit(X, y)
        return self

    def predict(self, X):
        logs = np.stack([np.log(np.asarray(m.predict(X), dtype=float)) for m in self.members])
        return np.exp(self.weights @ logs)
