"""Figures made specifically for the report."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

from freight_rate.analysis import plotting as P
from freight_rate.config import FoldSpec

TRAIN_START = pd.Timestamp("2025-01-01")


def validation_timeline(folds: tuple[FoldSpec, ...], path: Path) -> Path:
    """Gantt-style view of each fold's training and test windows plus deployment."""
    P.apply_style()
    rows = [
        (
            f"{f.name} ({'model selection' if f.role == 'selection' else 'final check'})",
            TRAIN_START,
            f.train_end,
            f.test_start,
            f.test_end,
        )
        for f in folds
    ]
    rows.append(
        (
            "Deployment",
            TRAIN_START,
            pd.Timestamp("2025-10-31"),
            pd.Timestamp("2025-11-01"),
            pd.Timestamp("2025-12-31"),
        )
    )

    fig, ax = plt.subplots(figsize=(10, 2.9))
    one_day = timedelta(days=1)
    for i, (label, train_start, train_end, test_start, test_end) in enumerate(reversed(rows)):
        y = i
        ax.barh(
            y,
            (train_end + one_day - train_start).days,
            left=mdates.date2num(train_start),
            height=0.55,
            color=P.SERIES[0],
            edgecolor="white",
            linewidth=2,
        )
        ax.barh(
            y,
            (test_end + one_day - test_start).days,
            left=mdates.date2num(test_start),
            height=0.55,
            color=P.SERIES[1],
            edgecolor="white",
            linewidth=2,
        )
        ax.text(
            mdates.date2num(TRAIN_START - timedelta(days=4)),
            y,
            label,
            ha="right",
            va="center",
            color=P.TEXT,
            fontsize=9,
        )
    ax.set_yticks([])
    ax.set_xlim(mdates.date2num(TRAIN_START), mdates.date2num(pd.Timestamp("2026-01-01")))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.grid(axis="x", color=P.GRID)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.legend(
        handles=[
            Patch(color=P.SERIES[0], label="training window"),
            Patch(color=P.SERIES[1], label="test window (2 months)"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
    )
    ax.set_title("Rolling-origin validation mirrors the Nov-Dec forecast", pad=10)
    return P.save(fig, path)


def quote_signal_regime_chart(regimes: pd.DataFrame, path: Path) -> Path:
    """Monthly label-free quote_signal correlation, coloured by regime.

    ``regimes``: output of freight_rate.data.quality.quote_signal_regimes for
    training and validation months concatenated, with a ``split`` column.
    """
    P.apply_style()
    colors = {"positive": P.SERIES[0], "inverted": P.SERIES[1], "none": P.MUTED}
    fig, ax = plt.subplots(figsize=(10, 3.2))
    labels = [pd.Period(p).strftime("%b") for p in regimes["period"]]
    ax.bar(
        range(len(regimes)),
        regimes["label_free_corr"],
        color=[colors[r] for r in regimes["regime"]],
        width=0.7,
        edgecolor="white",
        linewidth=2,
    )
    ax.axhline(0, color=P.MUTED, linewidth=0.8)
    first_val = int((regimes["split"] == "validation").to_numpy().argmax())
    ax.axvspan(first_val - 0.5, len(regimes) - 0.5, color=P.GRID, alpha=0.6, zorder=0)
    ax.text(
        first_val + 0.5,
        0.95,
        "validation\n(to predict)",
        ha="center",
        va="top",
        fontsize=8.5,
        color=P.TEXT_SECONDARY,
    )
    ax.set_xticks(range(len(regimes)), labels)
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("corr(quote_signal,\nexpected rate per mile)")
    pairs = zip(regimes["label_free_corr"], regimes["regime"], strict=True)
    for i, (value, regime) in enumerate(pairs):
        if regime == "none":  # bars near zero are invisible, so label them
            ax.text(
                i,
                0.06,
                f"no link\n({value:+.2f})",
                ha="center",
                va="bottom",
                fontsize=8,
                color=P.TEXT_SECONDARY,
            )
    ax.legend(
        handles=[Patch(color=colors[k], label=k) for k in ("positive", "inverted")],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
    )
    ax.set_title("quote_signal's link to price flips by month and is absent in Nov-Dec")
    return P.save(fig, path)
