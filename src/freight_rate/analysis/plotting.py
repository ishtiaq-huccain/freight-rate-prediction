"""Shared matplotlib styling for report figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

# Validated categorical slots 1-3 (safe for all-pairs use) and neutral inks.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
PRIMARY = SERIES[0]
ACCENT = SERIES[1]
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
MUTED = "#a3a29c"

EQUIPMENT_COLORS = {"Dry Van": SERIES[0], "Reefer": SERIES[1], "Flatbed": SERIES[2]}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.edgecolor": MUTED,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "text.color": TEXT,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
        }
    )


def save(fig: plt.Figure, path: Path, close: bool = True) -> Path:
    """Save a figure as PNG. Pass ``close=False`` in notebooks to keep it displayed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    if close:
        plt.close(fig)
    return path
