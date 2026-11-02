from __future__ import annotations

import numpy as np
import pytest

from freight_rate.data import io, schema
from freight_rate.validation.splits import city_holdout_masks, temporal_masks


@pytest.fixture(scope="module")
def train(cfg):
    return io.load_train(cfg.paths.train)


def test_temporal_folds_train_strictly_before_test(cfg, train):
    dates = train[schema.DATE]
    for fold in cfg.validation.folds:
        m = temporal_masks(dates, fold)
        assert dates[m.train].max() < dates[m.test].min()
        assert not (m.train & m.test).any()
        assert dates[m.train].max() <= fold.train_end
        assert dates[m.test].min() >= fold.test_start and dates[m.test].max() <= fold.test_end


def test_temporal_folds_expected_windows(cfg, train):
    dates = train[schema.DATE]
    months = {
        f.name: sorted(dates[temporal_masks(dates, f).test].dt.month.unique())
        for f in cfg.validation.folds
    }
    assert months == {"fold_1": [5, 6], "fold_2": [7, 8], "holdout": [9, 10]}


def test_final_holdout_is_the_latest_window(cfg):
    final = [f for f in cfg.validation.folds if f.role == "final"]
    assert len(final) == 1
    assert final[0].test_end == max(f.test_end for f in cfg.validation.folds)


def test_city_holdout_removes_hidden_cities_from_training(cfg, train):
    fold = cfg.validation.folds[-1]
    masks = temporal_masks(train[schema.DATE], fold)
    h = city_holdout_masks(train, masks, n_cities=8, rng=np.random.default_rng(0))
    assert len(h.cities) == 8

    tr, te = train[h.train], train[h.test]
    assert not tr["pickup"].isin(h.cities).any()
    assert not tr["delivery"].isin(h.cities).any()
    # Every scored test load has exactly one endpoint in a hidden city
    hidden_endpoints = te["pickup"].isin(h.cities).astype(int) + te["delivery"].isin(
        h.cities
    ).astype(int)
    assert (hidden_endpoints == 1).all()
    # Still a temporal split
    assert not (h.train & ~masks.train).any() and not (h.test & ~masks.test).any()
    assert len(te) > 500


def test_city_holdout_is_reproducible(cfg, train):
    masks = temporal_masks(train[schema.DATE], cfg.validation.folds[0])
    a = city_holdout_masks(train, masks, 8, np.random.default_rng(42))
    b = city_holdout_masks(train, masks, 8, np.random.default_rng(42))
    assert a.cities == b.cities and (a.train == b.train).all()


def test_city_holdout_rejects_too_many_cities(cfg, train):
    masks = temporal_masks(train[schema.DATE], cfg.validation.folds[0])
    with pytest.raises(ValueError):
        city_holdout_masks(train, masks, 500, np.random.default_rng(0))
