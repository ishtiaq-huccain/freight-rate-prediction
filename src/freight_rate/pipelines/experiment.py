"""Compare configured models on the validation folds.

    python -m freight_rate.pipelines.experiment --name phase3_baselines \
        --models global_median grouped_median ridge hgb

Writes to ``reports/experiments/``:
  <name>_metrics.csv   every fold x scenario x segment metric row
  <name>_summary.csv   one row per model (the comparison table)
  <name>_summary.md    the same table, plus the config used
Out-of-fold predictions go to ``artifacts/experiments/`` (not tracked).
Only "selection" folds run unless ``--include-final`` is given.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd
import yaml

from freight_rate.config import DEFAULT_CONFIG, PROJECT_ROOT, load_config
from freight_rate.data import io
from freight_rate.evaluation.cross_validation import run_cv
from freight_rate.models.registry import make_factory, training_options

log = logging.getLogger(__name__)
REPORT_DIR = PROJECT_ROOT / "reports" / "experiments"


def summarize_model(metrics: pd.DataFrame) -> dict:
    t = metrics[metrics.scenario == "temporal"].set_index(["fold", "segment"])
    seg = lambda s: t.xs(s, level="segment")  # noqa: E731
    unseen = metrics[metrics.scenario == "cities_unseen"]["mae"].mean()
    seen = metrics[metrics.scenario == "cities_seen"]["mae"].mean()
    row = {
        "MAE all": seg("all")["mae"].mean(),
        "MAE clean": seg("clean")["mae"].mean(),
        "RMSE clean": seg("clean")["rmse"].mean(),
        "MAPE% clean": seg("clean")["mape_pct"].mean(),
        "bias clean": seg("clean")["bias"].mean(),
        "MAE clean m1": seg("clean_month_1")["mae"].mean(),
        "MAE clean m2": seg("clean_month_2")["mae"].mean(),
    }
    for fold, value in seg("clean")["mae"].items():
        row[f"MAE clean {fold}"] = value
    row["MAE unseen city"] = unseen
    row["MAE same loads, city seen"] = seen
    row["unseen-city cost %"] = (unseen / seen - 1) * 100 if seen else float("nan")
    return row


def merge_experiment_config(raw: dict, extra_path: Path | None) -> dict:
    """Add ``models`` / ``feature_sets`` from an experiment YAML to the base config."""
    if extra_path is None:
        return raw
    extra = yaml.safe_load(Path(extra_path).read_text(encoding="utf-8")) or {}
    unknown = set(extra) - {"models", "feature_sets"}
    if unknown:
        raise ValueError(f"experiment config may only define models/feature_sets, got {unknown}")
    merged = dict(raw)
    for key in ("models", "feature_sets"):
        clash = set(raw.get(key, {})) & set(extra.get(key, {}))
        if clash:
            raise ValueError(f"experiment config redefines {key}: {sorted(clash)}")
        merged[key] = {**raw.get(key, {}), **extra.get(key, {})}
    return merged


def run(
    names: list[str],
    name: str,
    config_path: Path,
    include_final: bool,
    city_check: bool,
    extra_config: Path | None = None,
    shuffle_test: tuple[str, ...] = (),
) -> pd.DataFrame:
    cfg = load_config(config_path)
    raw = merge_experiment_config(cfg.raw, extra_config)
    train = io.load_train(cfg.paths.train)
    roles = ("selection", "final") if include_final else ("selection",)

    all_metrics, summaries = [], []
    pred_dir = cfg.paths.artifacts_dir / "experiments"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for model_name in names:
        factory = make_factory(model_name, raw, cfg.seed)
        options = training_options(model_name, raw)
        start = time.perf_counter()
        metrics, preds = run_cv(
            train,
            factory,
            cfg.validation,
            cfg.cleaning,
            cfg.seed,
            roles=roles,
            city_check=city_check,
            drop_anomalies=options["drop_anomalies"],
            shuffle_test=shuffle_test,
        )
        seconds = time.perf_counter() - start
        n_fits = len(set(metrics["fold"])) * (
            1 + (cfg.validation.city_holdout.n_repeats if city_check else 0)
        )
        log.info("%s done in %.1fs (%d fits)", model_name, seconds, n_fits)

        all_metrics.append(metrics.assign(model=model_name))
        preds.assign(model=model_name).to_csv(
            pred_dir / f"{name}_{model_name}_oof.csv", index=False
        )
        summaries.append(
            {"model": model_name, **summarize_model(metrics), "sec per fit": seconds / n_fits}
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.concat(all_metrics, ignore_index=True)
    summary = pd.DataFrame(summaries).set_index("model").round(2)
    metrics_df.to_csv(REPORT_DIR / f"{name}_metrics.csv", index=False)
    summary.to_csv(REPORT_DIR / f"{name}_summary.csv")

    used = {m: raw["models"][m] for m in names}
    md = [
        f"# Experiment: {name}",
        "",
        f"Folds: {', '.join(r for r in roles)} | seed {cfg.seed} | "
        f"city check: {'on' if city_check else 'off'}"
        + (f" | test columns shuffled: {', '.join(shuffle_test)}" if shuffle_test else ""),
        "",
        "MAE/RMSE/bias in USD, mean over folds. 'clean' excludes test rows flagged as "
        "corrupted labels. m1/m2 = first/second month of each 2-month test window.",
        "",
        to_markdown(summary.T),
        "",
        "## Model configs",
        "",
        "```json",
        json.dumps(
            {"models": used, "feature_sets": raw.get("feature_sets", {})}, indent=2, default=str
        ),
        "```",
    ]
    (REPORT_DIR / f"{name}_summary.md").write_text("\n".join(md), encoding="utf-8")
    return summary


def to_markdown(frame: pd.DataFrame) -> str:
    """Minimal markdown table (avoids an extra dependency)."""
    header = "| " + " | ".join([frame.index.name or ""] + [str(c) for c in frame.columns]) + " |"
    rule = "|" + "---|" * (len(frame.columns) + 1)
    body = [
        "| "
        + " | ".join([str(idx)] + [f"{v:,.2f}" if isinstance(v, float) else str(v) for v in row])
        + " |"
        for idx, row in zip(frame.index, frame.to_numpy(), strict=True)
    ]
    return "\n".join([header, rule, *body])


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare models on validation folds.")
    parser.add_argument("--name", required=True, help="experiment name (output file prefix)")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--extra-config", type=Path, help="YAML with extra models/feature_sets")
    parser.add_argument("--include-final", action="store_true", help="also run the final holdout")
    parser.add_argument("--no-city-check", action="store_true")
    parser.add_argument(
        "--shuffle-test",
        nargs="*",
        default=[],
        help="input columns to permute within each test window (e.g. quote_signal)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    summary = run(
        args.models,
        args.name,
        args.config,
        args.include_final,
        not args.no_city_check,
        args.extra_config,
        tuple(args.shuffle_test),
    )
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(summary.T)


if __name__ == "__main__":
    main()
