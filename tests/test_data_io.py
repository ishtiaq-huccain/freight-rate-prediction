from __future__ import annotations

import pandas as pd
import pytest

from freight_rate.data import io, schema


def test_train_loads_with_expected_contract(cfg):
    train = io.load_train(cfg.paths.train)
    assert list(train.columns) == schema.TRAIN_COLUMNS
    assert len(train) == 48_000
    assert pd.api.types.is_datetime64_any_dtype(train[schema.DATE])
    assert (train[schema.TARGET] > 0).all()


def test_validation_and_template_ids_align(cfg):
    val = io.load_validation(cfg.paths.validation)
    template = io.load_template(cfg.paths.validation_template)
    assert len(val) == 12_000
    assert val[schema.ID].tolist() == template[schema.ID].tolist()
    assert template["predicted_rate"].isna().all()


def test_december_inputs_cover_every_day(cfg):
    december = io.load_december_inputs(cfg.paths.december_inputs)
    expected = pd.date_range("2025-12-01", "2025-12-31", freq="D")
    assert december[schema.DATE].tolist() == list(expected)


def _write(tmp_path, frame):
    path = tmp_path / "x.csv"
    frame.to_csv(path, index=False)
    return path


def test_rejects_wrong_columns(tmp_path, cfg):
    frame = pd.read_csv(cfg.paths.train, nrows=5).drop(columns="quote_signal")
    with pytest.raises(io.DataValidationError, match="expected columns"):
        io.load_train(_write(tmp_path, frame))


def test_rejects_duplicate_ids(tmp_path, cfg):
    frame = pd.read_csv(cfg.paths.validation, nrows=5)
    frame.loc[1, schema.ID] = frame.loc[0, schema.ID]
    with pytest.raises(io.DataValidationError, match="duplicate"):
        io.load_validation(_write(tmp_path, frame))


def test_rejects_unknown_equipment(tmp_path, cfg):
    frame = pd.read_csv(cfg.paths.validation, nrows=5)
    frame.loc[0, "equipment"] = "Tanker"
    with pytest.raises(io.DataValidationError, match="equipment"):
        io.load_validation(_write(tmp_path, frame))


def test_rejects_bad_dates(tmp_path, cfg):
    frame = pd.read_csv(cfg.paths.validation, nrows=5)
    frame.loc[0, schema.DATE] = "2025-13-01"
    with pytest.raises(io.DataValidationError, match="dates"):
        io.load_validation(_write(tmp_path, frame))


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        io.load_train(tmp_path / "nope.csv")
