"""STATS19 code cleaning, sklearn preprocessing pipeline and leakage-safe splitting."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import Settings
from .feature_config import CATEGORICAL_RAW_INVALID, SPEED_LIMIT_VALID


def clean_codes(df: pd.DataFrame, num_vehicles_cap: int = 10) -> pd.DataFrame:
    """Handle STATS19 special codes. MUTATES df columns and returns it.

    * coded categorical columns -> string codes ("1","6",...) with "unknown" for every
      DfT missing / unknown / unallocated code. Unknown is kept as its own category because
      "unknown" itself can be informative; it is NOT silently treated as a number.
    * speed_limit -> float, anything other than 20/30/40/50/60/70 becomes NaN (imputed later).
    * number_of_vehicles -> float in [1, cap]; <1 becomes NaN (outlier winsorising at the cap).
    """
    for col, invalid in CATEGORICAL_RAW_INVALID.items():
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        s = s.where(~s.isin(list(invalid)))
        s = s.where(s == s.round())
        vals = s.fillna(-999).astype("int64").astype(str)
        df[col] = vals.where(s.notna(), "unknown")
    if "speed_limit" in df.columns:
        s = pd.to_numeric(df["speed_limit"], errors="coerce")
        df["speed_limit"] = s.where(s.isin(list(SPEED_LIMIT_VALID))).astype("float32")
    if "number_of_vehicles" in df.columns:
        s = pd.to_numeric(df["number_of_vehicles"], errors="coerce")
        df["number_of_vehicles"] = s.where(s >= 1).clip(upper=num_vehicles_cap).astype("float32")
    return df


def build_preprocessor(numeric: list[str], categorical: list[str], scale_numeric: bool) -> ColumnTransformer:
    """Numeric: median imputation (+ missing-indicator columns), optional standard scaling.
    Categorical: constant 'unknown' imputation, one-hot (unseen -> all zeros, rare levels pooled).

    Scaling is applied ONLY for the linear model (scale_numeric=True); tree ensembles are
    scale-invariant, so scaling would add cost without benefit.
    """
    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale_numeric:
        num_steps.append(("scale", StandardScaler()))
    cat_steps = [
        ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                 dtype=np.float32, min_frequency=0.001)),
    ]
    return ColumnTransformer(
        [("num", Pipeline(num_steps), numeric), ("cat", Pipeline(cat_steps), categorical)],
        remainder="drop", verbose_feature_names_out=True)


def assign_splits(y: np.ndarray, dates: pd.Series | None, cfg: Settings) -> np.ndarray:
    """Return an array of 'train'/'val'/'test' labels.

    random:   stratified on the target, random_state = cfg.random_seed (default 42).
    temporal: ordered by collision date; earliest rows train, then validation, latest test
              (use when you want to simulate predicting future collisions).
    """
    n = len(y)
    labels = np.empty(n, dtype=object)
    if cfg.split_strategy == "temporal":
        if dates is None or dates.isna().all():
            raise ValueError("Temporal split needs a parsed 'date' column.")
        order = np.argsort(dates.to_numpy(dtype="datetime64[ns]"), kind="stable")
        n_tr, n_va = int(round(n * cfg.train_frac)), int(round(n * cfg.val_frac))
        labels[order[:n_tr]] = "train"
        labels[order[n_tr:n_tr + n_va]] = "val"
        labels[order[n_tr + n_va:]] = "test"
        return labels
    idx = np.arange(n)
    hold = cfg.val_frac + cfg.test_frac
    tr, tmp = train_test_split(idx, test_size=hold, stratify=y, random_state=cfg.random_seed)
    va, te = train_test_split(tmp, test_size=cfg.test_frac / hold, stratify=y[tmp], random_state=cfg.random_seed)
    labels[tr], labels[va], labels[te] = "train", "val", "test"
    return labels


def describe_preprocessing(cfg: Settings, load_report: dict, drops: dict, feature_report: dict,
                           numeric: list, categorical: list, split_report: dict) -> dict:
    return {
        "preprocessing_version": __import__("ml").PREPROCESSING_VERSION,
        "dataset_years_used": load_report["years_used"],
        "columns_read_from_csv": "only SOURCE_COLUMNS (ml/feature_config.py); all other columns never loaded",
        "dtype_strategy": "all text read as str per chunk, then float32 / datetime64 / str; chunk size %d" % cfg.chunk_size,
        "row_cleaning": drops,
        "special_code_handling": {c: sorted(v) for c, v in CATEGORICAL_RAW_INVALID.items()},
        "special_code_policy": "DfT missing/unknown codes -> category 'unknown' (kept, not imputed); speed_limit outside {20..70} -> NaN -> median imputation + missing indicator",
        "numeric_features": numeric, "categorical_features": categorical,
        "numeric_pipeline": "SimpleImputer(median, add_indicator=True) -> StandardScaler (logistic regression ONLY)",
        "categorical_pipeline": "SimpleImputer(constant 'unknown') -> OneHotEncoder(handle_unknown='ignore', min_frequency=0.001)",
        "high_cardinality_geography_decision": "latitude/longitude/LSOA/local-authority/police-force NOT used as features (location memorisation, reporting-practice bias, thousands of levels); coordinates rounded to 2 dp are kept only for the historical map.",
        "outlier_policy": f"number_of_vehicles winsorised at {cfg.num_vehicles_cap}; no rows removed for outliers",
        "feature_engineering": feature_report,
        "split": split_report,
    }
