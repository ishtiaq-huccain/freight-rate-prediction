"""Train the selected model on all labelled data and save it.

    python -m freight_rate.pipelines.train [--model final]

Uses the same ``fit_on_window`` as cross-validation, so cleaning, anomaly
filtering and features are identical to what was evaluated. Writes the fitted
model to ``paths.model_artifact`` and a JSON metadata file next to it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from freight_rate import __version__
from freight_rate.config import DEFAULT_CONFIG, load_config
from freight_rate.data import io, schema
from freight_rate.evaluation.cross_validation import fit_on_window
from freight_rate.models.registry import make_factory, resolve_spec, training_options

log = logging.getLogger(__name__)


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def train(config_path: Path = DEFAULT_CONFIG, model_name: str | None = None) -> Path:
    cfg = load_config(config_path)
    model_name = model_name or cfg.raw["final_model"]
    train_frame = io.load_train(cfg.paths.train)

    fitted = fit_on_window(
        train_frame,
        make_factory(model_name, cfg.raw, cfg.seed),
        cfg.cleaning,
        drop_anomalies=training_options(model_name, cfg.raw)["drop_anomalies"],
    )
    log.info(
        "trained '%s' on %d rows (%d corrupted labels excluded)",
        model_name,
        fitted.n_train_used,
        fitted.n_train_dropped,
    )

    artifact = cfg.paths.model_artifact
    artifact.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted, artifact)

    metadata = {
        "model_name": model_name,
        "model_spec": resolve_spec(model_name, cfg.raw["models"]),
        "trained_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "train_rows_used": fitted.n_train_used,
        "train_rows_excluded_as_corrupted": fitted.n_train_dropped,
        "train_date_range": [
            str(train_frame[schema.DATE].min().date()),
            str(train_frame[schema.DATE].max().date()),
        ],
        "train_data_sha256": _sha256(cfg.paths.train),
        "seed": cfg.seed,
        "versions": {
            "freight_rate": __version__,
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    artifact.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )
    log.info("saved %s", artifact)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the selected model on all labelled data.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", help="model name from the config (default: final_model)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    train(args.config, args.model)


if __name__ == "__main__":
    main()
