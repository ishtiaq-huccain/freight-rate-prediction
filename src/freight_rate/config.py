"""Project configuration loaded from YAML."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


@dataclass(frozen=True)
class Paths:
    train: Path
    validation: Path
    validation_template: Path
    december_inputs: Path
    validation_predictions: Path
    december_predictions: Path
    model_artifact: Path
    eda_dir: Path
    artifacts_dir: Path


@dataclass(frozen=True)
class CleaningConfig:
    anomaly_low_ratio: float = 0.5
    anomaly_high_ratio: float = 2.0
    anomaly_distance_bins: int = 10

    def __post_init__(self):
        if not 0 < self.anomaly_low_ratio < 1 < self.anomaly_high_ratio:
            raise ValueError("cleaning: require 0 < anomaly_low_ratio < 1 < anomaly_high_ratio")
        if self.anomaly_distance_bins < 1:
            raise ValueError("cleaning: anomaly_distance_bins must be >= 1")


@dataclass(frozen=True)
class FoldSpec:
    name: str
    role: str
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def __post_init__(self):
        if self.role not in {"selection", "final"}:
            raise ValueError(f"fold {self.name}: role must be 'selection' or 'final'")
        if not self.train_end < self.test_start <= self.test_end:
            raise ValueError(f"fold {self.name}: require train_end < test_start <= test_end")


@dataclass(frozen=True)
class CityHoldoutConfig:
    n_cities: int = 8
    n_repeats: int = 3


@dataclass(frozen=True)
class ValidationConfig:
    folds: tuple[FoldSpec, ...]
    city_holdout: CityHoldoutConfig


@dataclass(frozen=True)
class Config:
    paths: Paths
    seed: int
    cleaning: CleaningConfig
    validation: ValidationConfig
    raw: dict[str, Any]


def _parse_validation(raw: dict[str, Any]) -> ValidationConfig:
    folds = tuple(
        FoldSpec(
            name=str(f["name"]),
            role=str(f["role"]),
            train_end=pd.Timestamp(f["train_end"]),
            test_start=pd.Timestamp(f["test_start"]),
            test_end=pd.Timestamp(f["test_end"]),
        )
        for f in raw.get("folds", [])
    )
    if not folds:
        raise ValueError("validation: at least one fold is required")
    if len({f.name for f in folds}) != len(folds):
        raise ValueError("validation: fold names must be unique")
    return ValidationConfig(
        folds=folds, city_holdout=CityHoldoutConfig(**raw.get("city_holdout", {}))
    )


def load_config(path: str | Path = DEFAULT_CONFIG, root: Path = PROJECT_ROOT) -> Config:
    """Load a YAML config. Relative paths are resolved against ``root``."""
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    try:
        paths = Paths(**{key: (root / value).resolve() for key, value in raw["paths"].items()})
        cleaning = CleaningConfig(**raw.get("cleaning", {}))
    except TypeError as exc:
        raise ValueError(f"invalid config {path}: {exc}") from exc
    return Config(
        paths=paths,
        seed=int(raw.get("seed", 42)),
        cleaning=cleaning,
        validation=_parse_validation(raw.get("validation", {})),
        raw=raw,
    )
