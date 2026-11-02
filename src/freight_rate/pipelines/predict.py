"""Generate both submission files from the saved model.

    python -m freight_rate.pipelines.predict

Writes:
  validation_predictions.csv        load_id,predicted_rate for all 12,000 loads
  data/december_chart_inputs.csv    predicted_rate filled for every December day
Both are checked against the scorer's acceptance rules before writing.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib

from freight_rate.config import DEFAULT_CONFIG, load_config
from freight_rate.data import io
from freight_rate.inference import predict_december, predict_validation, write_december

log = logging.getLogger(__name__)


def predict(config_path: Path = DEFAULT_CONFIG) -> tuple[Path, Path]:
    cfg = load_config(config_path)
    if not cfg.paths.model_artifact.is_file():
        raise FileNotFoundError(
            f"{cfg.paths.model_artifact} not found; "
            "run `python -m freight_rate.pipelines.train` first"
        )
    model = joblib.load(cfg.paths.model_artifact)
    validation = io.load_validation(cfg.paths.validation)
    template = io.load_template(cfg.paths.validation_template)
    december = io.load_december_inputs(cfg.paths.december_inputs)

    val_out = predict_validation(model, validation, template)
    val_out.to_csv(cfg.paths.validation_predictions, index=False)
    log.info(
        "wrote %s: %d rows, predicted_rate %.2f-%.2f (median %.2f)",
        cfg.paths.validation_predictions,
        len(val_out),
        val_out.predicted_rate.min(),
        val_out.predicted_rate.max(),
        val_out.predicted_rate.median(),
    )

    dec_out = predict_december(model, december, validation)
    write_december(dec_out, cfg.paths.december_predictions)
    log.info(
        "wrote %s: 31 days, predicted_rate %.2f-%.2f",
        cfg.paths.december_predictions,
        dec_out.predicted_rate.min(),
        dec_out.predicted_rate.max(),
    )
    return cfg.paths.validation_predictions, cfg.paths.december_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Write validation and December predictions.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    predict(args.config)


if __name__ == "__main__":
    main()
