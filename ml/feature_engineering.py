"""Deterministic feature engineering from the ACTUAL STATS19 columns present.

Every feature is created only when its source columns exist; otherwise it is skipped and the
reason is recorded (nothing is invented). The same function is used for training data and for
single API requests, so training and serving cannot drift apart.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .feature_config import (
    ENGINEERED_FEATURES, GROUP_MAPS, MODEL_CATEGORICAL, MODEL_NUMERIC,
)
from .preprocessing import clean_codes


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce").astype("float32")


def engineer_features(df: pd.DataFrame):
    """MUTATES df (adds columns). Returns (df, report)."""
    created, skipped = [], {}

    # ---- time features ------------------------------------------------------------
    hour = None
    if "hour" in df.columns:
        h = _num(df, "hour")
        hour = h.where(h.between(0, 23)).astype("float32")
        df["hour"] = hour
    month = weekday = None
    if "date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["date"]):
        month = df["date"].dt.month.astype("float32")
        weekday = df["date"].dt.dayofweek.astype("float32")
    if "month" in df.columns:
        m = _num(df, "month")
        month = m if month is None else month.fillna(m)
    if "day_of_week" in df.columns:        # STATS19: 1=Sunday .. 7=Saturday
        code = _num(df, "day_of_week")
        wk = ((code - 2) % 7).where(code.between(1, 7)).astype("float32")
        weekday = wk if weekday is None else weekday.fillna(wk)
    elif "weekday" in df.columns and weekday is None:
        weekday = _num(df, "weekday")

    if hour is not None:
        df["hour_sin"] = np.sin(2 * np.pi * hour / 24).astype("float32")
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24).astype("float32")
        df["is_night"] = ((hour >= 21) | (hour < 6)).astype("float32").where(hour.notna())
        created += ["hour", "hour_sin", "hour_cos", "is_night"]
    else:
        for f in ("hour", "hour_sin", "hour_cos", "is_night", "is_rush_hour"):
            skipped[f] = "source column 'hour' (from collision time) not available"
    if month is not None:
        df["month"] = month.astype("float32")
        m = df["month"].to_numpy()
        df["season"] = np.select(
            [np.isin(m, [12, 1, 2]), np.isin(m, [3, 4, 5]), np.isin(m, [6, 7, 8]), np.isin(m, [9, 10, 11])],
            ["winter", "spring", "summer", "autumn"], default="unknown")
        created += ["month", "season"]
    else:
        skipped["month"] = skipped["season"] = "no date/month available"
    if weekday is not None:
        df["weekday"] = weekday.astype("float32")
        df["is_weekend"] = (weekday >= 5).astype("float32").where(weekday.notna())
        created += ["weekday", "is_weekend"]
    else:
        skipped["weekday"] = skipped["is_weekend"] = "no date/day_of_week available"
    if hour is not None and weekday is not None:
        rush = (hour.between(7, 9) | hour.between(16, 18)) & (weekday < 5)
        df["is_rush_hour"] = rush.astype("float32").where(hour.notna() & weekday.notna())
        created.append("is_rush_hour")
    elif "is_rush_hour" not in skipped:
        skipped["is_rush_hour"] = "needs both hour and weekday"

    # ---- speed / vehicles -----------------------------------------------------------
    if "speed_limit" in df.columns:
        sp = _num(df, "speed_limit").to_numpy()
        df["speed_band"] = np.select(
            [sp <= 20, sp == 30, (sp >= 40) & (sp <= 50), sp == 60, sp == 70],
            ["20_or_less", "30", "40_50", "60", "70"], default="unknown")
        created.append("speed_band")
    else:
        skipped["speed_band"] = "speed_limit not available"
    if "number_of_vehicles" in df.columns:
        nv = _num(df, "number_of_vehicles")
        df["multi_vehicle"] = (nv >= 2).astype("float32").where(nv.notna())
        created.append("multi_vehicle")
    else:
        skipped["multi_vehicle"] = "number_of_vehicles not available"

    # ---- code groupings ---------------------------------------------------------------
    for src, (new, mapping) in GROUP_MAPS.items():
        if src in df.columns:
            df[new] = df[src].astype(str).map(mapping).fillna("unknown")
            created.append(new)
        else:
            skipped[new] = f"source column '{src}' not available"

    report = {
        "created": sorted(set(created)), "skipped": skipped,
        "reasons": {k: ENGINEERED_FEATURES[k] for k in sorted(set(created)) if k in ENGINEERED_FEATURES},
    }
    return df, report


def build_features(df: pd.DataFrame, num_vehicles_cap: int = 10):
    """clean STATS19 codes + engineer features. MUTATES df. Returns (df, report)."""
    df = clean_codes(df, num_vehicles_cap)
    return engineer_features(df)


def resolve_feature_lists(df: pd.DataFrame):
    """Model feature lists restricted to columns that actually exist (never invents columns)."""
    numeric = [c for c in MODEL_NUMERIC if c in df.columns]
    categorical = [c for c in MODEL_CATEGORICAL if c in df.columns]
    excluded_missing = {c: "not present in the loaded data / could not be engineered"
                        for c in MODEL_NUMERIC + MODEL_CATEGORICAL if c not in df.columns}
    return numeric, categorical, excluded_missing
