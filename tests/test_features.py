"""Feature pipeline: schema consistency, row independence, no train/serve skew."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from freight_rate.data import io, schema
from freight_rate.data.cleaning import LoadCleaner
from freight_rate.features.build import FEATURE_GROUPS, FeatureBuilder

ALL_GROUPS = list(FEATURE_GROUPS)


@pytest.fixture(scope="module")
def frames(cfg):
    train = io.load_train(cfg.paths.train)
    val = io.load_validation(cfg.paths.validation)
    december = io.load_december_inputs(cfg.paths.december_inputs)
    cleaner = LoadCleaner().fit(train)
    return {
        "train": cleaner.transform(train),
        "validation": cleaner.transform(val),
        "december": cleaner.transform(december),
    }


def test_same_columns_for_train_validation_and_december(frames):
    builder = FeatureBuilder(ALL_GROUPS).fit(frames["train"])
    outputs = {k: builder.transform(v) for k, v in frames.items()}
    expected = [c for g in ALL_GROUPS for c in FEATURE_GROUPS[g]]
    for name, out in outputs.items():
        assert list(out.columns) == expected, name
        assert np.isfinite(out.to_numpy()).all(), name
        assert len(out) == len(frames[name])


def test_features_are_row_independent(frames):
    """A row's features must not depend on which other rows are in the batch."""
    builder = FeatureBuilder(ALL_GROUPS).fit(frames["train"])
    full = builder.transform(frames["validation"])
    subset = frames["validation"].sample(50, random_state=0)
    pd.testing.assert_frame_equal(builder.transform(subset), full.loc[subset.index])


def test_builder_is_stateless_and_ignores_target(frames):
    train = frames["train"]
    a = FeatureBuilder(ALL_GROUPS).fit(train).transform(train)
    b = (
        FeatureBuilder(ALL_GROUPS)
        .fit(train.head(10), train[schema.TARGET].head(10))
        .transform(train)
    )
    pd.testing.assert_frame_equal(a, b)
    assert schema.TARGET not in a.columns and schema.ID not in a.columns


def test_december_rows_use_fixed_lane_values(frames):
    out = (
        FeatureBuilder(["log_distance", "equipment", "weight", "coords", "time_index"])
        .fit(frames["train"])
        .transform(frames["december"])
    )
    assert np.allclose(out["log_distance"], np.log(360.0))
    assert (out["eq_dry_van"] == 1).all() and (out["eq_reefer"] == 0).all()
    assert np.allclose(out["weight_k"], 32.0)
    lexington = frames["train"].loc[frames["train"].pickup == "Lexington", "pickup_lat"].iloc[0]
    assert np.allclose(out["pickup_lat"], lexington)
    assert (
        out["days_since_start"].is_monotonic_increasing and out["days_since_start"].nunique() == 31
    )


def test_feature_values(frames):
    row = frames["train"].iloc[[0]]
    out = FeatureBuilder(ALL_GROUPS).fit(row).transform(row).iloc[0]
    src = row.iloc[0]
    assert out["log_distance"] == pytest.approx(np.log(src.distance))
    assert out["weight_k"] == pytest.approx(src.weight / 1000)
    assert out["days_since_start"] == (src.date - pd.Timestamp("2025-01-01")).days
    assert out["eq_" + src.equipment.lower().replace(" ", "_")] == 1


def test_unknown_group_rejected(frames):
    with pytest.raises(ValueError, match="unknown feature groups"):
        FeatureBuilder(["log_distance", "lane_magic"]).fit(frames["train"])


def test_unknown_equipment_rejected(frames):
    bad = frames["validation"].head(3).copy()
    bad.loc[bad.index[0], "equipment"] = "Tanker"
    with pytest.raises(ValueError, match="equipment"):
        FeatureBuilder(["equipment"]).fit(bad).transform(bad)


def test_uncleaned_frame_rejected(cfg):
    raw = io.load_train(cfg.paths.train)
    with pytest.raises((ValueError, KeyError)):
        FeatureBuilder(["weight", "weight_missing"]).fit(raw).transform(raw)
