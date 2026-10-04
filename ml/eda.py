"""Phase 5: analytical EDA. Uses TRAIN + VALIDATION rows only so the test set stays unseen.
Writes reports/eda_summary.json (consumed by notebooks and, later, the insights page)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from .config import Settings
from .feature_config import CLASS_NAMES, MODEL_CATEGORICAL
from .utils import log, write_json

DISCRETE_NUMERIC = ["speed_limit", "number_of_vehicles", "hour", "weekday", "month",
                    "is_weekend", "is_rush_hour", "is_night", "multi_vehicle"]


def severity_table(d: pd.DataFrame, col: str, max_levels: int = 30) -> dict:
    ct = pd.crosstab(d[col], d["severity"]).reindex(columns=CLASS_NAMES, fill_value=0)
    ct["n"] = ct.sum(axis=1)
    ct = ct.sort_values("n", ascending=False).head(max_levels)
    try:
        ct = ct.sort_index() if col in DISCRETE_NUMERIC else ct
    except TypeError:
        pass
    out = {}
    for lvl, row in ct.iterrows():
        n = int(row["n"])
        out[str(lvl)] = {"n": n, **{f"pct_{c}": round(100 * float(row[c]) / n, 3) for c in CLASS_NAMES}}
    return out


def association(d: pd.DataFrame, col: str) -> dict:
    ct = pd.crosstab(d[col].astype(str), d["severity"])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return {"cramers_v": 0.0, "chi2_p": 1.0, "levels": int(ct.shape[0])}
    chi2, p, _, _ = chi2_contingency(ct, correction=False)
    v = float(np.sqrt(chi2 / (ct.to_numpy().sum() * (min(ct.shape) - 1))))
    return {"cramers_v": round(v, 4), "chi2_p": float(p), "levels": int(ct.shape[0]),
            "note": "With very large N p-values are ~0 for tiny effects; read Cramer's V (effect size)."}


def run_eda(df: pd.DataFrame, cfg: Settings) -> dict:
    d = df[df["split"] != "test"]
    log(f"EDA on {len(d):,} rows (train+val only; test rows excluded)")
    cat_cols = [c for c in MODEL_CATEGORICAL + DISCRETE_NUMERIC if c in d.columns]
    summary = {"rows_analysed": int(len(d)), "excluded_from_eda": "test split",
               "target_distribution": d["severity"].value_counts().reindex(CLASS_NAMES).fillna(0).astype(int).to_dict()}
    summary["numeric_distributions"] = {
        c: {k: (None if pd.isna(v) else float(v)) for k, v in d[c].describe().items()}
        for c in ("speed_limit", "number_of_vehicles", "hour") if c in d.columns}
    summary["categorical_distributions_pct"] = {
        c: (100 * d[c].value_counts(normalize=True)).round(3).head(15).to_dict()
        for c in MODEL_CATEGORICAL if c in d.columns}
    summary["severity_by_feature"] = {c: severity_table(d, c) for c in cat_cols}
    summary["association_with_severity"] = dict(sorted(
        ((c, association(d, c)) for c in cat_cols), key=lambda kv: -kv[1]["cramers_v"]))
    if "date" in d.columns and d["date"].notna().any():
        ym = d.assign(ym=d["date"].dt.to_period("M").astype(str))
        t = pd.crosstab(ym["ym"], ym["severity"]).reindex(columns=CLASS_NAMES, fill_value=0)
        summary["monthly_counts"] = {k: {c: int(v) for c, v in row.items()} for k, row in t.iterrows()}
    if "collision_year" in d.columns:
        t = pd.crosstab(d["collision_year"], d["severity"]).reindex(columns=CLASS_NAMES, fill_value=0)
        summary["yearly_counts"] = {str(k): {c: int(v) for c, v in row.items()} for k, row in t.iterrows()}
    head = {}
    for c, lab in (("is_night", "pct_night"), ("is_weekend", "pct_weekend"), ("is_rush_hour", "pct_rush_hour"),
                   ("multi_vehicle", "pct_multi_vehicle")):
        if c in d.columns:
            head[lab] = round(100 * float(d[c].mean()), 2)
    if "urban_or_rural_area" in d.columns:
        head["pct_rural"] = round(100 * float((d["urban_or_rural_area"] == "2").mean()), 2)
    summary["headline_stats"] = head
    write_json(cfg.reports_dir / "eda_summary.json", summary)
    log("wrote reports/eda_summary.json")
    return summary
