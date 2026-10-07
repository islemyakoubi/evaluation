"""Input checks and deterministic, target-free freight feature engineering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
TARGET = "posted_rate"
CATEGORICAL_FEATURES = ["pickup", "delivery", "route", "equipment"]
CORE_FEATURES = CATEGORICAL_FEATURES + [
    "distance", "weight", "weight_invalid", "month", "day_of_month",
    "day_of_week", "day_of_year",
]
RICH_FEATURES = CORE_FEATURES + [
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
    "market_index", "quote_signal",
]
STABLE_FEATURES = [f for f in RICH_FEATURES if f != "quote_signal"] + [
    "year_sin", "year_cos", "inverse_distance", "log_distance",
]
CYCLIC_FEATURES = [f for f in STABLE_FEATURES if f not in ("month", "day_of_year")]
OPTIONAL_FEATURES = {"pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "market_index"}
STABLE_CORE_FEATURES = [f for f in STABLE_FEATURES if f not in OPTIONAL_FEATURES]
CYCLIC_CORE_FEATURES = [f for f in CYCLIC_FEATURES if f not in OPTIONAL_FEATURES]
DECEMBER_COLUMNS = ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]


def load_raw_data(root: Path = ROOT) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Keep the supplied data immutable and preserve the template's ID order."""
    return tuple(pd.read_csv(root / filename) for filename in (
        "train-test.csv", "validation.csv", "december-chart-inputs.csv",
        "validation-predictions-template.csv",
    ))


def add_candidate_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive features from one row; no target or learned statistics are used."""
    required = {"pickup", "delivery", "equipment", "distance", "weight", "date"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing input columns: {sorted(missing)}")
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], format="%Y-%m-%d", errors="raise")
    if result["date"].isna().any():
        raise ValueError("Dates must not be missing.")
    distance = pd.to_numeric(result["distance"], errors="raise")
    if not np.isfinite(distance).all() or (distance <= 0).any():
        raise ValueError("Distance must be finite and strictly positive.")
    result["distance"] = distance
    weight = pd.to_numeric(result["weight"], errors="raise")
    if np.isinf(weight).any():
        raise ValueError("Weight cannot be infinite.")
    invalid = weight.isna() | (weight <= 0)
    if "weight_invalid" in result:
        invalid |= result["weight_invalid"].astype(bool)
    result["weight_invalid"] = invalid.astype(int)
    # Similar magnitudes to positive weights support a sign-error assumption.
    # Retain the magnitude, record the anomaly, and leave true missing values NaN.
    result["weight"] = weight.abs().replace(0, np.nan)
    for column in ("pickup", "delivery", "equipment"):
        result[column] = result[column].fillna("__MISSING__").astype(str)
    result["route"] = result["pickup"] + "__TO__" + result["delivery"]
    result["month"] = result["date"].dt.month
    result["day_of_month"] = result["date"].dt.day
    result["day_of_week"] = result["date"].dt.dayofweek
    result["day_of_year"] = result["date"].dt.dayofyear
    phase = 2 * np.pi * (result["day_of_year"] - 1) / 365.25
    result["year_sin"] = np.sin(phase)
    result["year_cos"] = np.cos(phase)
    result["inverse_distance"] = 1000 / result["distance"]
    result["log_distance"] = np.log(result["distance"])
    for column in OPTIONAL_FEATURES | {"quote_signal"}:
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="raise")
            if np.isinf(result[column]).any():
                raise ValueError(f"{column} cannot be infinite.")
    return result


def validate_sources(train: pd.DataFrame, validation: pd.DataFrame, template: pd.DataFrame) -> None:
    for label, frame in (("Development", train), ("Final validation", validation), ("Template", template)):
        if "load_id" not in frame or frame["load_id"].isna().any() or frame["load_id"].duplicated().any():
            raise ValueError(f"{label} must have unique, non-missing load IDs.")
    if TARGET not in train or TARGET in validation:
        raise ValueError("The target must exist only in development data.")
    y = pd.to_numeric(train[TARGET], errors="raise")
    if not np.isfinite(y).all() or (y <= 0).any():
        raise ValueError("Development targets must be finite and positive.")
    if list(template.columns) != ["load_id", "predicted_rate"]:
        raise ValueError("The prediction template has an unexpected schema.")
    if set(template["load_id"]) != set(validation["load_id"]):
        raise ValueError("Template IDs do not exactly match final-validation IDs.")
    if set(train["load_id"]) & set(validation["load_id"]):
        raise ValueError("Development and final-validation IDs overlap.")
    if pd.to_datetime(train["date"]).max() >= pd.to_datetime(validation["date"]).min():
        raise ValueError("Final validation must follow the development period.")


def audit_data(train: pd.DataFrame, validation: pd.DataFrame) -> dict:
    cities = set(train["pickup"]) | set(train["delivery"])
    audit = {}
    for label, frame in (("development", train), ("final_validation", validation)):
        unknown = ~frame["pickup"].isin(cities) | ~frame["delivery"].isin(cities)
        audit[label] = {
            "rows": len(frame), "columns": len(frame.columns),
            "date_start": str(frame["date"].min()), "date_end": str(frame["date"].max()),
            "duplicate_rows": int(frame.duplicated().sum()),
            "duplicate_ids": int(frame["load_id"].duplicated().sum()),
            "missing_values": {key: int(value) for key, value in frame.isna().sum().items() if value},
            "negative_weights": int((frame["weight"] < 0).sum()),
            "unseen_city_rows": int(unknown.sum()),
            "unseen_cities": sorted((set(frame["pickup"]) | set(frame["delivery"])) - cities),
        }
    date = pd.to_datetime(train["date"])
    correlations = []
    for month, group in train.groupby(date.dt.strftime("%Y-%m")):
        correlations.append({
            "month": month,
            "quote_rate_correlation": float(group["quote_signal"].corr(group[TARGET])),
            "quote_distance_correlation": float(group["quote_signal"].corr(group["distance"])),
        })
    audit["monthly_quote_correlations"] = correlations
    audit["target_distance_correlation"] = float(train["distance"].corr(train[TARGET]))
    audit["target_quantiles"] = {str(q): float(value) for q, value in train[TARGET].quantile([0, .01, .5, .95, .99, 1]).items()}
    return audit
