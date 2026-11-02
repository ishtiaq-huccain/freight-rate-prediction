"""Deployment-like test of quote_signal: does the model do better without it?

    python -m freight_rate.pipelines.quote_signal_check

Nov-Dec has no quote_signal regime (see freight_rate.data.quality). The only
labelled period like that is 2-31 August. This script trains models with and
without quote_signal on data ending before August and scores them on 2-31
August:
  * train Jan-Jul: training ends in an inverted month, as the final model's
    training ends in October
  * train Jan-Jun: a gap before the test period, closer to December
Writes reports/experiments/quote_signal_august_check.{csv,md}.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from freight_rate.config import DEFAULT_CONFIG, PROJECT_ROOT, load_config
from freight_rate.data import io, schema
from freight_rate.data.quality import quote_signal_regimes
from freight_rate.evaluation.cross_validation import fit_on_window, predict_rows
from freight_rate.evaluation.metrics import regression_metrics
from freight_rate.models.registry import make_factory
from freight_rate.pipelines.experiment import REPORT_DIR, merge_experiment_config, to_markdown

log = logging.getLogger(__name__)

TEST_START, TEST_END = pd.Timestamp("2025-08-02"), pd.Timestamp("2025-08-31")
TRAIN_ENDS = {
    "train Jan-Jul": pd.Timestamp("2025-07-31"),
    "train Jan-Jun": pd.Timestamp("2025-06-30"),
}
PAIRS = [("blend_70", "blend_70_noqs"), ("hgb_r3", "hgb_r3_noqs"), ("ridge_r3", "ridge_r3_noqs")]
EXTRA = PROJECT_ROOT / "configs" / "experiments" / "phase4.yaml"


def run(config_path: Path = DEFAULT_CONFIG) -> pd.DataFrame:
    cfg = load_config(config_path)
    raw = merge_experiment_config(cfg.raw, EXTRA)
    train = io.load_train(cfg.paths.train)
    test = train[(train[schema.DATE] >= TEST_START) & (train[schema.DATE] <= TEST_END)]
    regimes = quote_signal_regimes(train, test)
    if set(regimes["regime"]) != {"none"}:
        raise RuntimeError(f"test period is expected to have no quote_signal regime: {regimes}")

    rows = []
    for setup, train_end in TRAIN_ENDS.items():
        window = train[train[schema.DATE] <= train_end]
        y = test[schema.TARGET].to_numpy()
        clean = None
        for name in ["grouped_median"] + [m for pair in PAIRS for m in pair]:
            fitted = fit_on_window(window, make_factory(name, raw, cfg.seed), cfg.cleaning)
            if clean is None:
                clean = ~fitted.anomaly_filter.is_anomaly(test, test[schema.TARGET]).to_numpy()
            pred = predict_rows(fitted, test)
            m_clean = regression_metrics(y[clean], pred[clean])
            rows.append(
                {
                    "setup": setup,
                    "model": name,
                    "quote_signal": "no"
                    if name.endswith("_noqs") or name == "grouped_median"
                    else "yes",
                    "MAE clean": m_clean["mae"],
                    "MAE all": regression_metrics(y, pred)["mae"],
                    "MAPE% clean": m_clean["mape_pct"],
                    "bias clean": m_clean["bias"],
                }
            )
            log.info("%s %s: MAE clean %.2f", setup, name, m_clean["mae"])
    result = pd.DataFrame(rows).round(2)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(REPORT_DIR / "quote_signal_august_check.csv", index=False)
    md = [
        "# quote_signal check: test 2-31 Aug 2025 (no regime, like Nov-Dec)",
        "",
        to_markdown(result.set_index("setup")),
        "",
        "Label-free regime of the test period:",
        "",
        to_markdown(regimes.round(3).set_index("period")),
    ]
    (REPORT_DIR / "quote_signal_august_check.md").write_text("\n".join(md), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score models with/without quote_signal on Aug 2-31."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    with pd.option_context("display.width", 200):
        print(run(args.config).to_string(index=False))


if __name__ == "__main__":
    main()
