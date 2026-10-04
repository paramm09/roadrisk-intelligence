"""Inspect the ACTUAL raw files and write reports/schema_report.json (columns, dtypes, samples,
missing %, unique counts, rows, duplicates, raw target values, cross-year column differences)."""
from _common import parser, run, settings_from

from ml.data_loader import discover_raw_files
from ml.data_quality import inspect_file
from ml.feature_config import ALLOWED_FEATURES, CATEGORICAL_RAW_INVALID, EXCLUDED_FEATURES, SOURCE_COLUMNS
from ml.utils import log, now_iso, section, write_json


def main():
    args = parser(__doc__).parse_args()
    cfg = settings_from(args)
    cfg.ensure_dirs()
    section(f"INSPECT DATA (years requested: {cfg.years})")
    found, missing = discover_raw_files(cfg)
    if not found:
        raise FileNotFoundError(f"No STATS19 collision CSVs for {cfg.years} in {cfg.raw_dir}. Run scripts/download_data.py first.")
    if missing:
        log(f"WARNING: years not found: {missing}")
    files = {}
    for year, path in found.items():
        log(f"inspecting {path.name} ...")
        rep = inspect_file(path, cfg.chunk_size)
        files[str(year)] = rep
        log(f"  rows={rep['rows']:,} cols={rep['n_columns']} dup_rows={rep['duplicate_full_rows']} "
            f"dup_index={rep['duplicate_index_values']} raw target values={rep['target_raw_values']}")
    col_sets = {y: {c["canonical_name"] for c in r["columns"]} for y, r in files.items()}
    union = set().union(*col_sets.values())
    inter = set.intersection(*col_sets.values())
    present = union
    availability = {
        "source_columns_found_in_all_years": sorted(SOURCE_COLUMNS & inter),
        "source_columns_missing_in_some_years": sorted((SOURCE_COLUMNS & union) - inter),
        "source_columns_not_found_at_all": sorted(SOURCE_COLUMNS - union),
        "model_features_that_need_a_missing_source": {},
        "excluded_columns_present_in_files": {c: EXCLUDED_FEATURES[c] for c in sorted(present) if c in EXCLUDED_FEATURES},
        "columns_in_files_ignored_by_pipeline": sorted(c for c in union if c not in SOURCE_COLUMNS and c not in EXCLUDED_FEATURES),
    }
    for c in CATEGORICAL_RAW_INVALID:
        if c not in union:
            availability["model_features_that_need_a_missing_source"][c] = "column absent from the downloaded files -> feature skipped"
    rows_per_year = {y: r["rows"] for y, r in files.items()}
    target_totals: dict = {}
    for r in files.values():
        for code, n in (r.get("target_raw_values") or {}).items():
            target_totals[code] = target_totals.get(code, 0) + int(n)
    total_rows = sum(rows_per_year.values())
    write_json(cfg.reports_dir / "schema_report.json", {
        "created_at": now_iso(), "years_requested": cfg.years, "years_found": sorted(found), "years_missing": missing,
        "row_count_per_year": rows_per_year,
        "total_row_count": total_rows,
        "duplicate_count": {
            "full_rows": sum(int(r["duplicate_full_rows"]) for r in files.values()),
            "index_values": sum(int(r["duplicate_index_values"]) for r in files.values()),
        },
        "target_column": "collision_severity (canonical: accident_severity)",
        "target_values": dict(sorted(target_totals.items())),
        "target_distribution": {
            "mapping_expected": {"1": "Fatal", "2": "Serious", "3": "Slight"},
            "counts_by_code": dict(sorted(target_totals.items())),
            "percentages_by_code": {k: round(100 * v / total_rows, 3) for k, v in sorted(target_totals.items())} if total_rows else {},
            "note": "Aggregated across all inspected years; see reports/target_distribution.json for the validated, label-mapped version.",
        },
        "column_differences_between_years": {"in_all_years": sorted(inter), "not_in_all_years": sorted(union - inter)},
        "feature_availability": availability, "files": files})
    log("wrote reports/schema_report.json")
    if availability["source_columns_not_found_at_all"]:
        log(f"NOTE: expected columns not found: {availability['source_columns_not_found_at_all']} (dependent features will be skipped)")
    return 0


if __name__ == "__main__":
    run(main)
