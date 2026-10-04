"""Phase 2-4: raw STATS19 -> validated, cleaned, feature-engineered, split dataset."""
from __future__ import annotations

import shutil

import numpy as np
import pandas as pd

from .config import Settings
from .data_loader import load_years
from .data_quality import build_data_report, render_data_report_md, validate_target
from .feature_config import CLASS_NAMES, ID_COL, YEAR_COL
from .feature_engineering import build_features, resolve_feature_lists
from .preprocessing import assign_splits, describe_preprocessing
from .utils import free_memory, log, now_iso, save_frame, section, write_json


# Feature-engineering report metadata: where each engineered feature comes from and how.
ENGINEERING_SOURCES = {
    "hour": "time (STATS19)", "hour_sin": "hour", "hour_cos": "hour",
    "month": "date (STATS19)", "season": "month", "weekday": "date / day_of_week (STATS19)",
    "is_weekend": "weekday", "is_rush_hour": "hour + weekday", "is_night": "hour",
    "speed_band": "speed_limit", "multi_vehicle": "number_of_vehicles",
    "weather_group": "weather_conditions", "surface_group": "road_surface_conditions",
    "lighting_group": "light_conditions", "junction_group": "junction_detail",
    "road_type_group": "road_type",
}
ENGINEERING_TRANSFORMATIONS = {
    "hour": "extract hour-of-day from recorded time; out-of-range -> missing",
    "hour_sin": "sin(2*pi*hour/24) cyclical encoding",
    "hour_cos": "cos(2*pi*hour/24) cyclical encoding",
    "month": "extract month (1-12) from date",
    "season": "meteorological season from month",
    "weekday": "ISO weekday 0=Mon..6=Sun from date (fallback: STATS19 day_of_week)",
    "is_weekend": "weekday in {5,6}",
    "is_rush_hour": "weekday 07-09 or 16-18",
    "is_night": "hour >= 21 or hour < 6",
    "speed_band": "speed_limit grouped: <=20 / 30 / 40-50 / 60 / 70 / unknown",
    "multi_vehicle": "number_of_vehicles >= 2",
    "weather_group": "weather codes collapsed (fine/rain/snow/high_wind/fog/other/unknown)",
    "surface_group": "surface codes collapsed (dry/wet/snow_ice/contaminated/unknown)",
    "lighting_group": "light codes collapsed (daylight/dark_lit/dark_unlit/unknown)",
    "junction_group": "junction detail collapsed (no_junction/roundabout/t_junction/...)",
    "road_type_group": "road type collapsed (roundabout/one_way/dual/single/unknown)",
}


def _split_report(out: pd.DataFrame, cfg: Settings) -> dict:
    rep = {"strategy": cfg.split_strategy, "random_seed": cfg.random_seed,
           "fractions": {"train": cfg.train_frac, "val": cfg.val_frac, "test": cfg.test_frac}, "splits": {}}
    for name in ("train", "val", "test"):
        s = out[out["split"] == name]
        counts = s["severity"].value_counts().reindex(CLASS_NAMES).fillna(0).astype(int)
        entry = {"rows": int(len(s)), "class_counts": counts.to_dict(),
                 "class_percentages": (100 * counts / max(len(s), 1)).round(3).to_dict()}
        if "date" in s.columns and s["date"].notna().any():
            entry["date_min"], entry["date_max"] = str(s["date"].min().date()), str(s["date"].max().date())
        rep["splits"][name] = entry
    rep["note"] = ("Test rows are stored in the processed file but are never read by train_models.py "
                   "(they are dropped on load) and are only used by evaluate_models.py.")
    return rep


def run_prepare(cfg: Settings, strict: bool = False) -> dict:
    cfg.ensure_dirs()
    section(f"PREPARE DATA  (years requested: {cfg.years}, seed {cfg.random_seed}, split {cfg.split_strategy})")
    raw, load_report = load_years(cfg, strict=strict)
    n_loaded = len(raw)

    section("Validate target")
    raw, target_report = validate_target(raw)
    log(f"target classes: {target_report['class_counts']}  ({target_report['sanity_check_note']})")
    if not target_report["frequency_sanity_check_passed"]:
        log("WARNING: " + target_report["sanity_check_note"])
    write_json(cfg.reports_dir / "target_distribution.json", target_report)

    section("Data report (missing values, duplicates, outliers, special codes)")
    data_report = build_data_report(raw, load_report, target_report, cfg)
    write_json(cfg.reports_dir / "data_report.json", data_report)
    (cfg.reports_dir / "data_report.md").write_text(render_data_report_md(data_report), encoding="utf-8")
    n_dup_idx = int(raw[ID_COL].duplicated().sum())
    raw = raw.drop_duplicates(subset=ID_COL, keep="first")
    log(f"dropped {n_dup_idx:,} duplicate collision_index rows; rows now {len(raw):,}")
    drops = {"rows_loaded": n_loaded, "dropped_invalid_or_missing_target": target_report["n_rows_before"] - target_report["n_rows_after"],
             "dropped_duplicate_collision_index": n_dup_idx, "rows_final": int(len(raw))}
    free_memory()

    section("Clean codes + engineer features")
    df, feat_report = build_features(raw, cfg.num_vehicles_cap)
    numeric, categorical, missing_feats = resolve_feature_lists(df)
    feat_report["model_features_unavailable"] = missing_feats
    for k, v in feat_report["skipped"].items():
        log(f"  feature skipped: {k} -> {v}")
    log(f"  model features: {len(numeric)} numeric + {len(categorical)} categorical = {len(numeric) + len(categorical)}")

    section("Split (leakage-safe)")
    labels = assign_splits(df["target"].to_numpy(), df["date"] if "date" in df.columns else None, cfg)
    keep = [c for c in [ID_COL, YEAR_COL, "date", "target", "severity"] if c in df.columns] + numeric + categorical
    out = df[keep].copy()
    if {"latitude", "longitude"} <= set(df.columns):
        out["lat_r"] = df["latitude"].round(2).astype("float32")
        out["lon_r"] = df["longitude"].round(2).astype("float32")
    out[YEAR_COL] = out[YEAR_COL].astype("int16")
    out["split"] = labels
    del df, raw
    free_memory()
    split_report = _split_report(out, cfg)
    for k, v in split_report["splits"].items():
        log(f"  {k:5s} rows={v['rows']:,}  {v['class_counts']}")

    path = save_frame(out, cfg.processed_base)
    log(f"saved processed data -> {path}  ({out.memory_usage(deep=True).sum() / 1e6:.0f} MB in RAM)")
    meta = {
        "created_at": now_iso(), "years_requested": load_report["years_requested"],
        "years_used": load_report["years_used"], "years_missing": load_report["years_missing"],
        "n_rows": int(len(out)), "numeric_features": numeric, "categorical_features": categorical,
        "split": split_report, "feature_engineering": feat_report, "load_report": load_report,
        "row_cleaning": drops, "settings": cfg.snapshot(), "processed_file": str(path.name),
    }
    write_json(cfg.processed_meta_path, meta)
    write_json(cfg.reports_dir / "preprocessing_report.json",
               describe_preprocessing(cfg, load_report, drops, feat_report, numeric, categorical, split_report))

    # --- spec-named reports (all values computed above, never hand-written) --------------
    write_json(cfg.reports_dir / "data_preparation_report.json", {
        "raw_rows_per_year": {str(f["year"]): int(f["rows_loaded"]) for f in load_report["files"]},
        "combined_rows": int(load_report["total_rows_loaded"]),
        "removed_rows": {
            "invalid_or_missing_target": int(drops["dropped_invalid_or_missing_target"]),
            "duplicate_collision_index": int(drops["dropped_duplicate_collision_index"]),
            "total": int(drops["dropped_invalid_or_missing_target"] + drops["dropped_duplicate_collision_index"]),
        },
        "duplicates": data_report["duplicates"],
        "missing_values": {c["column"]: {"n_missing": c["n_missing"], "pct_missing": c["pct_missing"]}
                           for c in data_report["columns"]},
        "invalid_values": {
            "special_code_policy": "DfT -1/9/99/unallocated codes -> category 'unknown'; "
                                   "speed_limit outside 20-70 -> missing (median-imputed later)",
            "special_code_counts": {c: v["special_code_counts"]
                                    for c, v in data_report["special_code_analysis"].items()},
            "suspicious_values": data_report["suspicious_values"],
            "unparseable_dates": data_report["suspicious_values"].get("unparseable_dates", 0),
        },
        "final_rows": int(len(out)),
        "final_columns": list(out.columns),
        "processed_file": str(path.name),
        "years_used": load_report["years_used"],
        "settings": cfg.snapshot(),
    })
    write_json(cfg.reports_dir / "feature_engineering_report.json", {
        "engineered_features": [
            {"original_feature": ENGINEERING_SOURCES.get(name, "(model feature from STATS19)"),
             "engineered_feature": name,
             "transformation": ENGINEERING_TRANSFORMATIONS.get(name, "cleaned code / derived feature"),
             "justification": feat_report["reasons"].get(name, "Documented in ml/feature_config.py ENGINEERED_FEATURES."),
             "available_in_data": name in feat_report["created"]}
            for name in ENGINEERING_SOURCES],
        "model_features_numeric": numeric,
        "model_features_categorical": categorical,
        "model_features_unavailable": feat_report.get("model_features_unavailable", {}),
        "not_features_deliberately": "See reports/leakage_report.json and ml/feature_config.py EXCLUDED_FEATURES "
                                     "(target, casualty outcomes, identifiers, raw geography, ...).",
    })
    # Spec filename: keep an identically-named copy of the processed dataset so downstream
    # tooling finds data/processed/roadrisk_processed.parquet (same bytes, no re-computation).
    if path.suffix == ".parquet":
        alias = cfg.processed_dir / "roadrisk_processed.parquet"
        if alias.resolve() != path.resolve():
            shutil.copyfile(path, alias)
            log(f"also wrote {alias.name} (identical copy of {path.name})")
    log("PREPARE COMPLETE")
    return meta
