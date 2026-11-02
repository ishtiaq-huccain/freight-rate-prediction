"""Column contracts for the provided datasets."""

from __future__ import annotations

ID = "load_id"
TARGET = "posted_rate"
DATE = "date"

FEATURE_COLUMNS = [
    "pickup",
    "delivery",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "equipment",
    "weight",
    "date",
    "market_index",
    "quote_signal",
]

TRAIN_COLUMNS = [ID, *FEATURE_COLUMNS, TARGET]
VALIDATION_COLUMNS = [ID, *FEATURE_COLUMNS]
TEMPLATE_COLUMNS = [ID, "predicted_rate"]
DECEMBER_COLUMNS = [
    "pickup",
    "delivery",
    "distance",
    "equipment",
    "weight",
    "date",
    "predicted_rate",
]

EQUIPMENT_TYPES = ("Dry Van", "Flatbed", "Reefer")

# Columns that must never be missing in a raw file. Weight and market_index are
# known to contain gaps and are handled by the feature pipeline.
REQUIRED_NON_NULL = [ID, "pickup", "delivery", "distance", "equipment", "date"]
