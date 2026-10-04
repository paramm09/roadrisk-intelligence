"""RAM-efficient STATS19 loading: only required columns, chunked, compact dtypes."""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Settings
from .feature_config import (
    ID_COL, NUMERIC_SOURCE_COLUMNS, SOURCE_COLUMNS, TARGET, YEAR_COL, normalise_column_name,
)
from .utils import free_memory, log


class SchemaError(RuntimeError):
    pass


def discover_raw_files(cfg: Settings):
    """Find collision CSVs in data/raw for the configured years.

    A file matches if its name contains the year and 'collision' (or legacy 'accident').
    Returns ({year: path}, [missing_years]).
    """
    found, missing = {}, []
    files = sorted(Path(cfg.raw_dir).glob("*.csv"))
    for year in cfg.years:
        cands = [f for f in files
                 if re.search(rf"(?<!\d){year}(?!\d)", f.name)
                 and ("collision" in f.name.lower() or "accident" in f.name.lower())]
        if not cands:
            missing.append(year)
            continue
        if len(cands) > 1:
            cands.sort(key=lambda p: ("provisional" in p.name.lower(), len(p.name)))
            log(f"WARNING: several files match {year}: {[c.name for c in cands]}; using {cands[0].name}")
        found[year] = cands[0]
    return found, missing


def read_header(path: Path):
    """Return (raw_columns, encoding)."""
    for enc in ("utf-8-sig", "latin-1"):
        try:
            cols = list(pd.read_csv(path, nrows=0, encoding=enc).columns)
            return cols, enc
        except UnicodeDecodeError:
            continue
    raise SchemaError(f"Cannot decode header of {path}")


def map_columns(raw_cols):
    """raw name -> canonical name (first raw column wins if two collapse to one name)."""
    mapping, seen = {}, set()
    for c in raw_cols:
        n = normalise_column_name(c)
        if n in seen:
            continue
        seen.add(n)
        mapping[c] = n
    return mapping


def _parse_dates(s: pd.Series) -> pd.Series:
    d = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    if len(d) and d.isna().mean() > 0.5:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = pd.to_datetime(s, errors="coerce", dayfirst=True)
    return d


def _reduce_chunk(chunk: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Convert a raw all-string chunk to compact numeric / datetime columns."""
    for col in NUMERIC_SOURCE_COLUMNS:
        if col in chunk.columns:
            orig = chunk[col]
            num = pd.to_numeric(orig, errors="coerce")
            bad = int((num.isna() & orig.notna()).sum())
            if bad:
                stats["non_numeric_coerced"][col] = stats["non_numeric_coerced"].get(col, 0) + bad
            chunk[col] = num.astype("float32")
    if "date" in chunk.columns:
        chunk["date"] = _parse_dates(chunk["date"])
    if "time" in chunk.columns:
        hour = pd.to_numeric(chunk["time"].astype(str).str.slice(0, 2), errors="coerce")
        out_of_range = int((hour.notna() & ~hour.between(0, 23)).sum())
        if out_of_range:
            stats["hour_out_of_range"] = stats.get("hour_out_of_range", 0) + out_of_range
        chunk["hour"] = hour.where(hour.between(0, 23)).astype("float32")
        chunk.drop(columns=["time"], inplace=True)
    if ID_COL in chunk.columns:
        chunk[ID_COL] = chunk[ID_COL].astype(str)
    return chunk


def load_year(path: Path, year: int, cfg: Settings):
    raw_cols, enc = read_header(path)
    mapping = map_columns(raw_cols)
    usecols = [raw for raw, canon in mapping.items() if canon in SOURCE_COLUMNS]
    canon_found = {mapping[r] for r in usecols}
    if TARGET not in canon_found:
        raise SchemaError(
            f"{path.name}: no severity column found (looked for 'collision_severity'/'accident_severity'). "
            f"Columns present: {raw_cols[:15]}...")
    missing_cols = sorted(SOURCE_COLUMNS - canon_found)
    stats = {"non_numeric_coerced": {}}
    parts, n_rows = [], 0
    reader = pd.read_csv(path, usecols=usecols, dtype=str, chunksize=cfg.chunk_size, encoding=enc)
    for chunk in reader:
        if cfg.max_rows_per_year is not None and n_rows >= cfg.max_rows_per_year:
            break
        if cfg.max_rows_per_year is not None:
            chunk = chunk.iloc[: cfg.max_rows_per_year - n_rows]
        chunk = chunk.rename(columns=mapping)
        parts.append(_reduce_chunk(chunk, stats))
        n_rows += len(chunk)
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=list(canon_found))
    del parts
    free_memory()
    if YEAR_COL not in df.columns or df[YEAR_COL].isna().all():
        df[YEAR_COL] = np.float32(year)
    else:
        df[YEAR_COL] = df[YEAR_COL].fillna(year).astype("float32")
    # a code column that is mostly text means a *labelled* export, which this pipeline does not support
    for col, n_bad in stats["non_numeric_coerced"].items():
        if col != "date" and n_rows and n_bad / n_rows > 0.5:
            raise SchemaError(
                f"{path.name}: column '{col}' is mostly non-numeric ({n_bad}/{n_rows}). "
                "This looks like a labelled export; download the coded (numeric) DfT CSV instead.")
    info = {
        "year": year, "file": path.name, "size_mb": round(path.stat().st_size / 1e6, 2),
        "encoding": enc, "rows_loaded": int(n_rows), "columns_in_file": len(raw_cols),
        "columns_loaded": sorted(canon_found), "wanted_columns_missing": missing_cols,
        "non_numeric_coerced": stats["non_numeric_coerced"],
        "hour_out_of_range": stats.get("hour_out_of_range", 0),
        "row_limit_applied": cfg.max_rows_per_year,
    }
    return df, info


def load_years(cfg: Settings, strict: bool = False):
    """Load all configured years that exist in data/raw. Missing years are reported loudly."""
    found, missing = discover_raw_files(cfg)
    if missing:
        msg = (f"Years requested but not found in {cfg.raw_dir}: {missing}. "
               "Run scripts/download_data.py or place the files manually (see README).")
        if strict or not found:
            raise FileNotFoundError(msg)
        log("WARNING: " + msg + "  Continuing WITHOUT them (recorded in the reports).")
    frames, infos = [], []
    for year, path in found.items():
        log(f"Loading {year}: {path.name}")
        df, info = load_year(path, year, cfg)
        log(f"  rows={info['rows_loaded']:,}  cols={len(info['columns_loaded'])}")
        frames.append(df)
        infos.append(info)
    out = pd.concat(frames, ignore_index=True)
    del frames
    free_memory()
    if not cfg.allow_test_data and out[ID_COL].astype(str).str.startswith("TEST").any():
        raise RuntimeError("Synthetic TEST fixture rows detected in data/raw. Test data must never be "
                           "used for real training. Remove the file from data/raw.")
    report = {
        "years_requested": cfg.years, "years_used": sorted(found), "years_missing": missing,
        "files": infos, "total_rows_loaded": int(len(out)),
        "memory_mb_loaded_frame": round(out.memory_usage(deep=True).sum() / 1e6, 1),
    }
    return out, report
