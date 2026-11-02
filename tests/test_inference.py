"""Output contracts and end-to-end scorer compatibility."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from freight_rate.config import DEFAULT_CONFIG, PROJECT_ROOT, load_config
from freight_rate.data import io
from freight_rate.inference import (
    SubmissionError,
    attach_market_index,
    check_december_predictions,
    check_validation_predictions,
)


def _valid_submission():
    ids = [f"TE-{i:06d}" for i in range(1, 12_001)]
    return pd.DataFrame({"load_id": ids, "predicted_rate": 1000.0})


def _valid_december():
    return pd.DataFrame(
        {
            "pickup": "Lexington",
            "delivery": "Fort Wayne",
            "distance": 360,
            "equipment": "Dry Van",
            "weight": 32000,
            "date": pd.date_range("2025-12-01", "2025-12-31").strftime("%Y-%m-%d"),
            "predicted_rate": 800.0,
        }
    )


# ----------------------------------------------------------------------------- contract checks


def test_valid_submission_passes():
    check_validation_predictions(_valid_submission())
    check_december_predictions(_valid_december())


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda f: f.rename(columns={"predicted_rate": "rate"}),
        lambda f: f[["predicted_rate", "load_id"]],
        lambda f: f.iloc[:-1],
        lambda f: f.assign(load_id=f.load_id.where(f.index != 5, "TE-000001")),
        lambda f: f.assign(load_id=f.load_id.where(f.index != 5, "XX-000001")),
        lambda f: f.assign(predicted_rate=f.predicted_rate.where(f.index != 3, np.nan)),
        lambda f: f.assign(predicted_rate=f.predicted_rate.where(f.index != 3, np.inf)),
        lambda f: f.assign(predicted_rate=f.predicted_rate.where(f.index != 3, 0.0)),
    ],
    ids=[
        "column-name",
        "column-order",
        "row-count",
        "duplicate-id",
        "wrong-id",
        "nan",
        "inf",
        "zero",
    ],
)
def test_bad_submissions_rejected(corrupt):
    with pytest.raises(SubmissionError):
        check_validation_predictions(corrupt(_valid_submission()))


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda f: f.drop(columns="weight"),
        lambda f: f.iloc[:-1],
        lambda f: f.assign(pickup="Lexington, KY"),
        lambda f: f.assign(distance=361),
        lambda f: f.assign(predicted_rate=-1.0),
    ],
    ids=["columns", "missing-day", "pickup", "distance", "negative"],
)
def test_bad_december_rejected(corrupt):
    with pytest.raises(SubmissionError):
        check_december_predictions(corrupt(_valid_december()))


def test_attach_market_index_uses_same_day_reference():
    ref = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-12-01", "2025-12-01", "2025-12-02"]),
            "market_index": [0.8, 1.0, 1.2],
        }
    )
    frame = pd.DataFrame({"date": pd.to_datetime(["2025-12-01", "2025-12-02"])})
    out = attach_market_index(frame, ref)
    np.testing.assert_allclose(out["market_index"], [0.9, 1.2])
    with pytest.raises(SubmissionError, match="no reference"):
        attach_market_index(pd.DataFrame({"date": pd.to_datetime(["2025-12-03"])}), ref)


def test_every_december_date_has_validation_market_data(cfg):
    december = io.load_december_inputs(cfg.paths.december_inputs)
    validation = io.load_validation(cfg.paths.validation)
    out = attach_market_index(december.drop(columns="predicted_rate"), validation)
    assert out["market_index"].between(0.5, 1.5).all()


# ----------------------------------------------------------------------------- end to end


@pytest.fixture(scope="module")
def pipeline_outputs(tmp_path_factory):
    """Run train -> predict into a temp directory, with inputs from the real data."""
    from freight_rate.pipelines.predict import predict
    from freight_rate.pipelines.train import train

    tmp = tmp_path_factory.mktemp("run")
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    raw["paths"].update(
        {
            "validation_predictions": str(tmp / "validation_predictions.csv"),
            "december_predictions": str(tmp / "december_chart_inputs.csv"),
            "model_artifact": str(tmp / "model.joblib"),
            "artifacts_dir": str(tmp),
        }
    )
    config_path = tmp / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    train(config_path)
    predict(config_path)
    return tmp, load_config(config_path)


def test_pipeline_writes_model_metadata(pipeline_outputs):
    tmp, _ = pipeline_outputs
    meta = yaml.safe_load((tmp / "model.json").read_text(encoding="utf-8"))
    assert meta["model_name"] == "final"
    assert meta["train_date_range"] == ["2025-01-01", "2025-10-31"]
    assert meta["train_rows_used"] + meta["train_rows_excluded_as_corrupted"] == 48_000


def test_outputs_accepted_by_provided_scorer(pipeline_outputs):
    tmp, cfg = pipeline_outputs
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "score.py"),
            "--predictions",
            str(cfg.paths.validation_predictions),
            "--december-predictions",
            str(cfg.paths.december_predictions),
            "--output-dir",
            str(tmp / "scorer_results"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Validated 12,000 final predictions." in result.stdout
    assert (tmp / "scorer_results" / "candidate_december.png").is_file()


def test_validation_predictions_preserve_template_order(pipeline_outputs, cfg):
    _, run_cfg = pipeline_outputs
    out = pd.read_csv(run_cfg.paths.validation_predictions)
    template = io.load_template(cfg.paths.validation_template)
    assert out["load_id"].tolist() == template["load_id"].tolist()


def test_december_file_keeps_original_inputs(pipeline_outputs, cfg):
    _, run_cfg = pipeline_outputs
    out = pd.read_csv(run_cfg.paths.december_predictions)
    original = pd.read_csv(cfg.paths.december_inputs)
    cols = ["pickup", "delivery", "distance", "equipment", "weight", "date"]
    pd.testing.assert_frame_equal(out[cols], original[cols])


def test_submitted_files_are_reproducible(pipeline_outputs, cfg):
    """A fresh run reproduces the committed submission files."""
    _, run_cfg = pipeline_outputs
    for fresh, committed in [
        (run_cfg.paths.validation_predictions, cfg.paths.validation_predictions),
        (run_cfg.paths.december_predictions, cfg.paths.december_predictions),
    ]:
        if not committed.is_file():
            pytest.skip(f"{committed} not generated yet")
        a, b = pd.read_csv(fresh), pd.read_csv(committed)
        assert list(a.columns) == list(b.columns)
        np.testing.assert_allclose(a["predicted_rate"], b["predicted_rate"], rtol=1e-9)
