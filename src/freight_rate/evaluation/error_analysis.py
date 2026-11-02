"""Error breakdown of out-of-fold predictions by segment."""

from __future__ import annotations

import pandas as pd

from freight_rate.data import schema

DISTANCE_BANDS = [0, 250, 500, 1000, 1500, 2000, 2500, 4000]
WEIGHT_BANDS = [0, 20_000, 30_000, 40_000, 47_499, 50_000]


def segment_errors(
    oof: pd.DataFrame, loads: pd.DataFrame, clean_only: bool = True
) -> dict[str, pd.DataFrame]:
    """MAE, bias and MAPE per segment.

    ``oof``: columns load_id, predicted, clean, fold (from ``run_cv``).
    ``loads``: the raw training frame, for segment attributes.
    """
    df = oof.merge(
        loads[
            [
                schema.ID,
                schema.DATE,
                schema.TARGET,
                "distance",
                "equipment",
                "weight",
                "pickup",
                "delivery",
            ]
        ],
        on=schema.ID,
        suffixes=("_oof", ""),
    )
    if clean_only:
        df = df[df["clean"]]
    df["err"] = df["predicted"] - df[schema.TARGET]
    df["abs_err"] = df["err"].abs()
    df["ape"] = df["abs_err"] / df[schema.TARGET] * 100
    df["distance band"] = pd.cut(df["distance"], DISTANCE_BANDS)
    df["weight band"] = pd.cut(
        df["weight"].abs(),
        WEIGHT_BANDS,
        labels=["<20k", "20-30k", "30-40k", "40k-cap", "cap 47.5k"],
    )
    df["month"] = df[schema.DATE].dt.to_period("M").astype(str)

    def table(key: str) -> pd.DataFrame:
        g = df.groupby(key, observed=True)
        return pd.DataFrame(
            {
                "loads": g.size(),
                "MAE $": g["abs_err"].mean(),
                "bias $": g["err"].mean(),
                "MAPE %": g["ape"].mean(),
                "mean rate $": g[schema.TARGET].mean(),
            }
        ).round(2)

    return {k: table(k) for k in ["month", "equipment", "distance band", "weight band"]}


def worst_lanes(
    oof: pd.DataFrame, loads: pd.DataFrame, min_loads: int = 15, n: int = 10
) -> pd.DataFrame:
    df = oof[oof["clean"]].merge(
        loads[[schema.ID, "pickup", "delivery", schema.TARGET]], on=schema.ID, suffixes=("_oof", "")
    )
    df["ape"] = (df["predicted"] - df[schema.TARGET]).abs() / df[schema.TARGET] * 100
    g = df.groupby(["pickup", "delivery"]).agg(loads=("ape", "size"), mape=("ape", "mean"))
    return g[g.loads >= min_loads].sort_values("mape", ascending=False).head(n).round(2)
