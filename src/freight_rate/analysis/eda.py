"""Reusable helpers for the EDA notebook (``notebooks/01_eda.ipynb``).

The notebook holds the narrative and the step-by-step analysis. This module
holds only logic that is either reused or too long to read comfortably in a
cell: geographic distance, the reference "expected rate per mile", the
outlier-impact baseline, and the report figures.

"Expected rate per mile" is an EDA-only reference: the median rate per mile in
the same equipment x distance-decile cell. It makes anomalies visible; it is
not a model and is not used for training.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from freight_rate.analysis import plotting as P
from freight_rate.data import schema

EARTH_RADIUS_MILES = 3958.8
WEIGHT_CAP = 47_500.0
# Rate-per-mile ratio bounds used to *describe* label anomalies in the EDA.
LOW_RATIO, HIGH_RATIO = 0.5, 2.0


def haversine_miles(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in miles."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_MILES * np.arcsin(np.sqrt(a))


def add_reference_columns(train: pd.DataFrame) -> pd.DataFrame:
    """Add EDA helper columns: month, rpm, expected rpm, ratio, anomaly flag, circuity."""
    df = train.copy()
    df["month"] = df[schema.DATE].dt.month
    df["rpm"] = df[schema.TARGET] / df["distance"]
    df["dist_decile"] = pd.qcut(df["distance"], 10, labels=False)
    df["expected_rpm"] = df.groupby(["equipment", "dist_decile"])["rpm"].transform("median")
    df["rpm_ratio"] = df["rpm"] / df["expected_rpm"]
    df["label_anomaly"] = np.select(
        [df["rpm_ratio"] < LOW_RATIO, df["rpm_ratio"] > HIGH_RATIO], ["low", "high"], "normal"
    )
    df["haversine"] = haversine_miles(
        df.pickup_lat, df.pickup_lon, df.delivery_lat, df.delivery_lon
    )
    df["circuity"] = df["distance"] / df["haversine"]
    return df


def outlier_impact(df: pd.DataFrame, train_months=(1, 8), test_months=(9, 10)) -> pd.DataFrame:
    """Measure how label anomalies affect a simple temporal-holdout baseline.

    Baseline: predicted rate = median rate per mile of the (equipment, distance
    decile) cell in the training window x distance. Anomaly flags in the
    training window use training-window statistics only. Returns one row per
    (training variant, evaluation subset) with MAE, RMSE, median AE and MAPE.
    """
    train = df[df.month.between(*train_months)].copy()
    test = df[df.month.between(*test_months)].copy()
    edges = np.quantile(train.distance, np.linspace(0, 1, 11))[1:-1]
    for frame in (train, test):
        frame["bin"] = np.digitize(frame.distance, edges)

    train_ratio = train["rpm"] / train.groupby(["equipment", "bin"])["rpm"].transform("median")
    keep = train_ratio.between(LOW_RATIO, HIGH_RATIO)
    test_ratio = test["rpm"] / test.groupby(["equipment", "bin"])["rpm"].transform("median")
    test_clean = test_ratio.between(LOW_RATIO, HIGH_RATIO)
    subsets = {
        "all holdout rows": test_clean | True,
        "clean rows": test_clean,
        "anomalous rows": ~test_clean,
    }

    rows = []
    for variant, rows_used in {
        "train on all rows": train,
        "train excluding anomalies": train[keep],
    }.items():
        table = rows_used.groupby(["equipment", "bin"])["rpm"].median().rename("rpm_hat")
        pred = test.join(table, on=["equipment", "bin"])["rpm_hat"] * test["distance"]
        for subset, mask in subsets.items():
            err = (pred - test[schema.TARGET])[mask]
            y = test.loc[mask, schema.TARGET]
            rows.append(
                {
                    "training": variant,
                    "evaluated on": subset,
                    "rows": int(mask.sum()),
                    "MAE": err.abs().mean(),
                    "RMSE": np.sqrt((err**2).mean()),
                    "median AE": err.abs().median(),
                    "MAPE %": (err.abs() / y).mean() * 100,
                    "total abs error": err.abs().sum(),
                }
            )
    return pd.DataFrame(rows).round(2)


# --------------------------------------------------------------------------- figures
# Each function returns a Figure so the notebook can display it inline and save it.


def fig_target_vs_distance(df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].hist(df[schema.TARGET], bins=80, color=P.PRIMARY, edgecolor="white", linewidth=0.4)
    axes[0].set_title("Posted rate distribution")
    axes[0].set_xlabel("posted_rate ($)")
    axes[0].set_ylabel("loads")
    axes[1].hexbin(
        df.distance, df.rpm, gridsize=60, bins="log", cmap="Blues", mincnt=1, yscale="log"
    )
    axes[1].set_title("Rate per mile vs distance (log scale)")
    axes[1].set_xlabel("distance (miles)")
    axes[1].set_ylabel("posted_rate / distance ($/mile)")
    axes[1].grid(False)
    return fig


def fig_anomaly_clusters(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.hist(np.log2(df.rpm_ratio), bins=200, color=P.PRIMARY, edgecolor="none")
    ax.set_yscale("log")
    for x in (np.log2(LOW_RATIO), np.log2(HIGH_RATIO)):
        ax.axvline(x, color=P.ACCENT, linewidth=1.2, linestyle="--")
    ax.set_title("Rate per mile relative to equipment x distance median (log2)")
    ax.set_xlabel("log2(rate per mile / expected)   dashed lines: 0.5x and 2x")
    ax.set_ylabel("loads (log scale)")
    return fig


def fig_market_over_time(df: pd.DataFrame, val: pd.DataFrame) -> plt.Figure:
    """Two stacked panels sharing the time axis (never a dual y-axis)."""
    clean = df[df.label_anomaly == "normal"]
    weekly_rpm = clean.groupby(schema.DATE)["rpm_ratio"].median().resample("W").mean()
    mi_train = df.groupby(schema.DATE)["market_index"].mean().resample("W").mean()
    mi_val = val.groupby(schema.DATE)["market_index"].mean().resample("W").mean()
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True)
    axes[0].plot(weekly_rpm.index, weekly_rpm.values, color=P.PRIMARY)
    axes[0].set_title("Weekly median relative rate per mile (training, clean labels)")
    axes[0].set_ylabel("ratio to expected")
    axes[1].plot(mi_train.index, mi_train.values, color=P.PRIMARY, label="training")
    axes[1].plot(mi_val.index, mi_val.values, color=P.ACCENT, label="validation (unlabeled)")
    axes[1].set_title("Weekly mean market_index")
    axes[1].set_ylabel("market_index")
    axes[1].legend(loc="upper right")
    return fig


def fig_rpm_by_equipment_distance(df: pd.DataFrame) -> plt.Figure:
    clean = df[df.label_anomaly == "normal"]
    bins = pd.cut(clean.distance, [0, 150, 300, 500, 750, 1000, 1500, 2000, 2500, 3500])
    table = clean.groupby([bins, "equipment"], observed=True)["rpm"].median().unstack()
    centers = [interval.mid for interval in table.index]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for equipment in ["Dry Van", "Reefer", "Flatbed"]:
        ax.plot(
            centers,
            table[equipment],
            marker="o",
            markersize=5,
            color=P.EQUIPMENT_COLORS[equipment],
            label=equipment,
        )
        ax.annotate(
            equipment,
            (centers[-1], table[equipment].iloc[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            color=P.TEXT_SECONDARY,
        )
    ax.set_title("Median rate per mile by distance and equipment (clean labels)")
    ax.set_xlabel("distance (miles, bin midpoint)")
    ax.set_ylabel("$/mile")
    ax.legend(loc="upper right")
    return fig


def fig_weight_distribution(train: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.hist(train.weight.dropna(), bins=120, color=P.PRIMARY, edgecolor="none")
    ax.set_yscale("log")
    ax.set_title("Weight distribution (training): negative values and a 47,500 lb cap")
    ax.set_xlabel("weight (lb)")
    ax.set_ylabel("loads (log scale)")
    return fig


def fig_distance_vs_haversine(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.2, 5))
    ax.scatter(df.haversine, df.distance, s=3, alpha=0.25, color=P.PRIMARY, linewidths=0)
    odd = df[(df.circuity > 1.5) | (df.circuity < 1.0)]
    ax.scatter(
        odd.haversine, odd.distance, s=14, color=P.ACCENT, label="distance / great-circle > 1.5"
    )
    ax.set_title("Stated vs great-circle distance")
    ax.set_xlabel("great-circle distance from coordinates (miles)")
    ax.set_ylabel("stated distance (miles)")
    ax.grid(axis="both")
    ax.legend(loc="upper left")
    return fig


def fig_city_coverage(train: pd.DataFrame, val: pd.DataFrame) -> plt.Figure:
    cols = ["city", "lat", "lon"]
    cities = pd.concat(
        [
            val[["pickup", "pickup_lat", "pickup_lon"]].set_axis(cols, axis=1),
            val[["delivery", "delivery_lat", "delivery_lon"]].set_axis(cols, axis=1),
        ]
    ).drop_duplicates("city")
    known = set(train.pickup) | set(train.delivery)
    seen, unseen = cities[cities.city.isin(known)], cities[~cities.city.isin(known)]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(seen.lon, seen.lat, s=22, color=P.PRIMARY, label=f"in training ({len(seen)})")
    ax.scatter(
        unseen.lon,
        unseen.lat,
        s=60,
        marker="D",
        color=P.ACCENT,
        edgecolor="white",
        linewidth=1.5,
        label=f"validation only ({len(unseen)})",
    )
    for _, row in unseen.iterrows():
        ax.annotate(
            row.city,
            (row.lon, row.lat),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8.5,
            color=P.TEXT_SECONDARY,
        )
    ax.set_title("City coordinates as given in the data")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.grid(axis="both")
    ax.legend(loc="lower left")
    return fig
