"""Leakage audit, run before every training. Writes reports/leakage_report.json."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from .config import Settings
from .feature_config import (
    ALLOWED_FEATURES, EXCLUDED_FEATURES, ID_COL, LEAKAGE_NAME_PATTERNS, TARGET,
)
from .utils import write_json


class LeakageError(RuntimeError):
    pass


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    if x.dtype.kind in "fiu" and x.nunique() > 20:
        x = pd.qcut(x, q=10, duplicates="drop")
    ct = pd.crosstab(x.astype(str), y)
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return 0.0
    chi2 = chi2_contingency(ct, correction=False)[0]
    n = ct.to_numpy().sum()
    return float(np.sqrt(chi2 / (n * (min(ct.shape) - 1))))


def audit_features(features: list[str], df: pd.DataFrame | None = None, y_col: str = "target",
                   train_mask: np.ndarray | None = None, loaded_columns: list[str] | None = None,
                   cfg: Settings | None = None, v_threshold: float = 0.5, write: bool = True) -> dict:
    """Run all leakage checks. Raises LeakageError on any hard failure; returns the report."""
    problems = []
    in_excluded = [f for f in features if f in EXCLUDED_FEATURES]
    not_allowed = [f for f in features if f not in ALLOWED_FEATURES]
    pattern_hits = [f for f in features if any(p in f.lower() for p in LEAKAGE_NAME_PATTERNS)]
    if TARGET in features or y_col in features:
        problems.append("target column is among the features")
    if in_excluded:
        problems.append(f"excluded features used: {in_excluded}")
    if not_allowed:
        problems.append(f"features not on the whitelist: {not_allowed}")
    if pattern_hits:
        problems.append(f"feature names match leakage patterns {LEAKAGE_NAME_PATTERNS}: {pattern_hits}")

    assoc, id_dups = {}, None
    if df is not None:
        if ID_COL in df.columns:
            id_dups = int(df[ID_COL].duplicated().sum())
            if id_dups:
                problems.append(f"{id_dups} collision_index values appear more than once (possible train/test overlap)")
        d = df if train_mask is None else df.loc[train_mask]
        if len(d) > 200_000:
            d = d.sample(200_000, random_state=42)
        for f in features:
            if f in d.columns:
                assoc[f] = round(cramers_v(d[f], d[y_col]), 4)
        suspicious = {f: v for f, v in assoc.items() if v >= v_threshold}
        if suspicious:
            problems.append(f"features almost determine the target (Cramer's V >= {v_threshold}): {suspicious}")
    report = {
        "passed": not problems, "problems": problems,
        "features_used": list(features),
        "features_excluded": EXCLUDED_FEATURES,
        "excluded_columns_found_in_loaded_data": sorted(c for c in (loaded_columns or []) if c in EXCLUDED_FEATURES),
        "checks": {
            "target_not_in_features": TARGET not in features,
            "no_excluded_feature_used": not in_excluded,
            "all_features_whitelisted": not not_allowed,
            "no_leakage_name_patterns": not pattern_hits,
            "duplicate_collision_index_count": id_dups,
            "single_feature_cramers_v_threshold": v_threshold,
        },
        "single_feature_cramers_v_with_target": assoc,
        "note": "Cramer's V is computed on training rows only. A high value is a red flag, a low value is not proof of absence of leakage.",
    }
    if write and cfg is not None:
        write_json(cfg.reports_dir / "leakage_report.json", report)
    if problems:
        raise LeakageError("LEAKAGE AUDIT FAILED: " + " | ".join(problems))
    return report
