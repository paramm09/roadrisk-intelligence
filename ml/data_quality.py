"""Schema inspection, target validation and the data report (missing values, duplicates, outliers, ...)."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Settings
from .data_loader import SchemaError, map_columns, read_header
from .feature_config import (
    ALLOWED_FEATURES, CATEGORICAL_RAW_INVALID, CODE_TO_INT, EXCLUDED_FEATURES, ID_COL, SOURCE_COLUMNS,
    SPEED_LIMIT_VALID, TARGET, TARGET_CODE_TO_LABEL, CLASS_NAMES,
)


# --------------------------------------------------------------------------- schema inspection
def inspect_file(path: Path, chunksize: int = 100_000, unique_cap: int = 500) -> dict:
    """Stream a raw CSV (all columns) and summarise it without loading it whole."""
    raw_cols, enc = read_header(path)
    mapping = map_columns(raw_cols)
    first = pd.read_csv(path, nrows=2000, encoding=enc)
    dtypes = {c: str(t) for c, t in first.dtypes.items()}
    n_rows = 0
    missing = Counter()
    uniques = {c: set() for c in raw_cols}
    samples = {c: [] for c in raw_cols}
    id_seen, dup_ids = set(), 0
    row_hashes, dup_rows = set(), 0
    target_raw = Counter()
    id_raw = next((r for r, c in mapping.items() if c == ID_COL), None)
    tgt_raw = next((r for r, c in mapping.items() if c == TARGET), None)
    for chunk in pd.read_csv(path, dtype=str, chunksize=chunksize, encoding=enc):
        n_rows += len(chunk)
        for c in raw_cols:
            col = chunk[c]
            missing[c] += int(col.isna().sum())
            if len(uniques[c]) < unique_cap:
                uniques[c].update(col.dropna().unique()[: unique_cap])
            if len(samples[c]) < 5:
                for v in col.dropna().unique()[:5]:
                    if v not in samples[c] and len(samples[c]) < 5:
                        samples[c].append(v)
        if id_raw:
            for v in chunk[id_raw].astype(str).tolist():
                if v in id_seen:
                    dup_ids += 1
                else:
                    id_seen.add(v)
        h = pd.util.hash_pandas_object(chunk, index=False).to_numpy()
        for v in h.tolist():
            if v in row_hashes:
                dup_rows += 1
            else:
                row_hashes.add(v)
        if tgt_raw:
            target_raw.update(chunk[tgt_raw].fillna("<missing>").tolist())
    cols = []
    for c in raw_cols:
        cols.append({
            "column": c, "canonical_name": mapping.get(c, c), "dtype_inferred": dtypes.get(c),
            "n_missing": missing[c], "pct_missing": round(100 * missing[c] / max(n_rows, 1), 3),
            "n_unique": (len(uniques[c]) if len(uniques[c]) < unique_cap else f">={unique_cap}"),
            "sample_values": samples[c],
            "used_by_pipeline": mapping.get(c, c) in SOURCE_COLUMNS,
        })
    return {
        "file": path.name, "encoding": enc, "rows": n_rows, "n_columns": len(raw_cols),
        "duplicate_full_rows": dup_rows, "duplicate_index_values": dup_ids,
        "target_column_raw_name": tgt_raw, "target_raw_values": dict(target_raw), "columns": cols,
    }


# --------------------------------------------------------------------------- target
def validate_target(df: pd.DataFrame):
    """Verify the raw target coding against reality. Returns (clean_df, report).

    Hard fail: no target column, or more than 1% of rows have an unexpected / missing code.
    Sanity check: Slight(3) should be the most frequent class and Fatal(1) the rarest; if not,
    the DfT coding may have changed and the mapping must be re-checked by hand.
    """
    if TARGET not in df.columns:
        raise SchemaError(f"Target column '{TARGET}' not found after column normalisation.")
    raw = df[TARGET]
    n = len(df)
    n_missing = int(raw.isna().sum())
    vals = raw.dropna()
    observed = {int(k): int(v) for k, v in vals.value_counts().items() if float(k).is_integer()}
    unexpected = {k: v for k, v in observed.items() if k not in TARGET_CODE_TO_LABEL}
    n_bad = n_missing + sum(unexpected.values())
    if n and n_bad / n > 0.01:
        raise SchemaError(
            f"Target validation failed: {n_bad:,}/{n:,} rows ({100*n_bad/n:.1f}%) are missing or use codes "
            f"outside {sorted(TARGET_CODE_TO_LABEL)}: {unexpected}. Check the DfT data guide for your year.")
    keep = raw.isin(list(TARGET_CODE_TO_LABEL))
    out = df.loc[keep].copy()
    out["target"] = out[TARGET].astype("int64").map(CODE_TO_INT).astype("int8")
    out["severity"] = out["target"].map(dict(enumerate(CLASS_NAMES)))
    counts = out["severity"].value_counts().reindex(CLASS_NAMES).fillna(0).astype(int)
    pct = (100 * counts / max(len(out), 1)).round(3)
    code_counts = {c: observed.get(c, 0) for c in (1, 2, 3)}
    sane = code_counts[3] >= code_counts[2] >= code_counts[1]
    rep = {
        "raw_target_column": TARGET,
        "raw_values_observed": observed,
        "mapping_used": {str(k): v for k, v in TARGET_CODE_TO_LABEL.items()},
        "model_class_order": CLASS_NAMES,
        "n_rows_before": n, "n_missing_target": n_missing, "n_unexpected_codes": sum(unexpected.values()),
        "unexpected_codes": unexpected, "n_rows_after": int(len(out)),
        "class_counts": counts.to_dict(), "class_percentages": pct.to_dict(),
        "imbalance_ratio_majority_to_minority": round(float(counts.max() / max(counts.min(), 1)), 2),
        "frequency_sanity_check_passed": bool(sane),
        "sanity_check_note": ("Expected frequency order Slight > Serious > Fatal."
                              if sane else "WARNING: unexpected class frequency order; verify the coding in the DfT guide."),
    }
    return out, rep


# --------------------------------------------------------------------------- data report
def _iqr_outliers(s: pd.Series) -> dict:
    s = s.dropna()
    if s.empty:
        return {}
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n = int(((s < lo) | (s > hi)).sum())
    return {"q1": float(q1), "q3": float(q3), "lower_fence": float(lo), "upper_fence": float(hi),
            "n_outliers": n, "pct_outliers": round(100 * n / len(s), 3)}


def build_data_report(raw: pd.DataFrame, load_report: dict, target_report: dict, cfg: Settings) -> dict:
    """Report on the loaded (pre-cleaning) data. Computed on the already-reduced frame."""
    n = len(raw)
    cols = []
    for c in raw.columns:
        s = raw[c]
        cols.append({"column": c, "dtype": str(s.dtype), "n_missing": int(s.isna().sum()),
                     "pct_missing": round(100 * float(s.isna().mean()), 3), "n_unique": int(s.nunique(dropna=True))})
    special = {}
    for c, invalid in CATEGORICAL_RAW_INVALID.items():
        if c in raw.columns:
            vc = raw[c].value_counts(dropna=False)
            special[c] = {"special_code_counts": {str(int(k)): int(v) for k, v in vc.items()
                                                  if pd.notna(k) and int(k) in invalid},
                          "observed_codes": {str(int(k)): int(v) for k, v in vc.items() if pd.notna(k)},
                          "pct_unknown_or_missing": round(100 * float(
                              (raw[c].isna() | raw[c].isin(list(invalid))).mean()), 3)}
    stats, outliers = {}, {}
    for c in ("speed_limit", "number_of_vehicles", "hour"):
        if c in raw.columns:
            d = raw[c].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99])
            stats[c] = {k: (None if pd.isna(v) else float(v)) for k, v in d.items()}
            outliers[c] = _iqr_outliers(raw[c])
    suspicious = {}
    if "speed_limit" in raw.columns:
        suspicious["speed_limit_not_in_{20..70}"] = int((raw["speed_limit"].notna() & ~raw["speed_limit"].isin(list(SPEED_LIMIT_VALID))).sum())
    if "number_of_vehicles" in raw.columns:
        suspicious["number_of_vehicles_lt_1"] = int((raw["number_of_vehicles"] < 1).sum())
        suspicious["number_of_vehicles_gt_cap"] = int((raw["number_of_vehicles"] > cfg.num_vehicles_cap).sum())
    if "date" in raw.columns:
        suspicious["unparseable_dates"] = int(raw["date"].isna().sum())
    suspicious["hour_out_of_range_set_to_missing"] = int(sum(f.get("hour_out_of_range", 0) for f in load_report["files"]))
    if {"latitude", "longitude"} <= set(raw.columns):
        la, lo = raw["latitude"], raw["longitude"]
        suspicious["coordinates_missing"] = int((la.isna() | lo.isna()).sum())
        suspicious["coordinates_outside_uk_box"] = int((la.notna() & lo.notna() & ~(la.between(49, 61) & lo.between(-9, 2.5))).sum())
    present = set(raw.columns)
    cand = [f for f in ALLOWED_FEATURES]
    return {
        "dataset": {
            "name": "UK DfT STATS19 Road Safety Open Data - collision file(s)",
            "licence": "Open Government Licence v3.0 (attribution required: Contains public sector information licensed under the OGL v3.0). Confirm on the DfT page for your download date.",
            "years_requested": load_report["years_requested"], "years_used": load_report["years_used"],
            "years_missing": load_report["years_missing"], "files": load_report["files"],
        },
        "n_rows_loaded": n, "n_columns_loaded": int(raw.shape[1]),
        "duplicates": {"duplicate_full_rows": int(raw.duplicated().sum()),
                       "duplicate_collision_index": int(raw[ID_COL].duplicated().sum()) if ID_COL in raw else None},
        "columns": cols, "special_code_analysis": special, "numeric_statistics": stats,
        "outlier_analysis_iqr": outliers,
        "outlier_handling": f"number_of_vehicles capped at {cfg.num_vehicles_cap} (winsorised); speed_limit outside {sorted(SPEED_LIMIT_VALID)} set to missing; not removing rows.",
        "suspicious_values": suspicious, "target_distribution": target_report,
        "candidate_features_raw_columns_present": sorted(c for c in present if c in ALLOWED_FEATURES),
        "model_features_planned": cand,
        "excluded_features": EXCLUDED_FEATURES,
        "columns_loaded_but_not_features": sorted(c for c in present if c not in ALLOWED_FEATURES),
    }


def render_data_report_md(r: dict) -> str:
    d, t = r["dataset"], r["target_distribution"]
    L = ["# STATS19 data report", "",
         f"- **Source:** {d['name']}", f"- **Licence:** {d['licence']}",
         f"- **Years requested:** {d['years_requested']}", f"- **Years used:** {d['years_used']}",
         f"- **Years missing:** {d['years_missing'] or 'none'}",
         f"- **Rows loaded:** {r['n_rows_loaded']:,}   **Columns loaded:** {r['n_columns_loaded']}",
         f"- **Duplicate full rows:** {r['duplicates']['duplicate_full_rows']:,}   "
         f"**Duplicate collision_index:** {r['duplicates']['duplicate_collision_index']}", "",
         "## Target", "", "| Class | Count | % |", "|---|---:|---:|"]
    for k in t["model_class_order"]:
        L.append(f"| {k} | {t['class_counts'][k]:,} | {t['class_percentages'][k]:.2f} |")
    L += ["", f"Imbalance ratio (majority:minority) = {t['imbalance_ratio_majority_to_minority']}. {t['sanity_check_note']}", "",
          "## Missing values (loaded columns)", "", "| Column | Missing | % |", "|---|---:|---:|"]
    for c in sorted(r["columns"], key=lambda x: -x["pct_missing"])[:25]:
        L.append(f"| {c['column']} | {c['n_missing']:,} | {c['pct_missing']} |")
    L += ["", "## Unknown / special codes (% of rows)", "", "| Column | % unknown |", "|---|---:|"]
    for c, v in r["special_code_analysis"].items():
        L.append(f"| {c} | {v['pct_unknown_or_missing']} |")
    L += ["", "## Suspicious values", ""] + [f"- {k}: {v:,}" for k, v in r["suspicious_values"].items()]
    L += ["", "## Outliers (IQR rule)", ""] + [f"- {k}: {v.get('n_outliers')} ({v.get('pct_outliers')}%)" for k, v in r["outlier_analysis_iqr"].items()]
    L += ["", f"Handling: {r['outlier_handling']}", "", "## Excluded features", "", "| Feature | Reason |", "|---|---|"]
    for k, v in r["excluded_features"].items():
        L.append(f"| {k} | {v} |")
    return "\n".join(L) + "\n"
