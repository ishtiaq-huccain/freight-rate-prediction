from __future__ import annotations

import pandas as pd
import pytest
import yaml

from freight_rate.config import DEFAULT_CONFIG, CleaningConfig, FoldSpec, load_config


def test_default_config_folds(cfg):
    folds = cfg.validation.folds
    assert [f.name for f in folds] == ["fold_1", "fold_2", "holdout"]
    assert [f.role for f in folds] == ["selection", "selection", "final"]
    # Each test window is two months, matching the Nov-Dec horizon
    for f in folds:
        months = pd.period_range(f.test_start, f.test_end, freq="M")
        assert len(months) == 2


def test_fold_requires_train_before_test():
    with pytest.raises(ValueError, match="train_end < test_start"):
        FoldSpec(
            "bad",
            "selection",
            pd.Timestamp("2025-05-01"),
            pd.Timestamp("2025-05-01"),
            pd.Timestamp("2025-06-30"),
        )


def test_fold_role_is_validated():
    with pytest.raises(ValueError, match="role"):
        FoldSpec(
            "bad",
            "tuning",
            pd.Timestamp("2025-04-30"),
            pd.Timestamp("2025-05-01"),
            pd.Timestamp("2025-06-30"),
        )


def test_cleaning_ratio_bounds_validated():
    with pytest.raises(ValueError):
        CleaningConfig(anomaly_low_ratio=1.5)


def test_duplicate_fold_names_rejected(tmp_path):
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    raw["validation"]["folds"][1]["name"] = "fold_1"
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        load_config(path)
