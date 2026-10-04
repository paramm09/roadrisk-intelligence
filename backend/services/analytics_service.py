"""Aggregated analytics over the REAL processed STATS19 dataset.

- The processed parquet is loaded ONCE (only the columns the API needs).
- Aggregations are computed from that DataFrame and memoised per filter set;
  no CSV is ever read per request and no value is ever hardcoded.
- Filters map to the dashboard's interactive filters.

All outputs are descriptive summaries of historical reported collisions.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..config import BackendConfig
from .model_service import ModelService

SEVERITY_ORDER = ["Slight", "Serious", "Fatal"]
MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
WEEKDAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _norm_key(k):
    """Group keys: NaN/None -> None; integer-valued numbers -> int; anything else -> str."""
    if k is None:
        return None
    try:
        if pd.isna(k):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(k, bool):
        return str(k)
    if isinstance(k, (int,)):
        return int(k)
    if isinstance(k, (float,)) or type(k).__name__.startswith(("float", "int")):
        f = float(k)
        return int(f) if f == int(f) else str(k)
    return str(k)

ANALYTICS_COLUMNS = [
    "collision_year", "severity", "speed_limit", "number_of_vehicles", "hour", "weekday", "month",
    "road_type", "junction_detail", "junction_control", "light_conditions", "weather_conditions",
    "road_surface_conditions", "urban_or_rural_area", "season", "lat_r", "lon_r", "split",
]


@dataclass(frozen=True)
class Filters:
    year: int | None = None
    month: int | None = None
    severity: str | None = None
    road_type: int | str | None = None
    weather: int | str | None = None
    lighting: int | str | None = None
    speed_limit: int | None = None
    urban_rural: int | str | None = None
    hour: int | None = None
    weekday: int | None = None

    DIMENSIONS = (
        "year", "month", "severity", "road_type", "weather",
        "lighting", "speed_limit", "urban_rural", "hour", "weekday",
    )

    def key(self) -> tuple:
        return (self.year, self.month, self.severity, self.road_type, self.weather,
                self.lighting, self.speed_limit, self.urban_rural, self.hour, self.weekday)

    def exclude(self, *names: str) -> "Filters":
        """Copy without the named dimensions. Tableau-style cross-filtering: the chart for
        dimension D keeps showing EVERY category of D (so the user can pick another one),
        while everything else is filtered by D."""
        values = {n: getattr(self, n) for n in self.DIMENSIONS}
        for n in names:
            values[n] = None
        return Filters(**values)

    @property
    def is_empty(self) -> bool:
        return all(v is None for v in self.key())


def apply_filters(df: pd.DataFrame, f: Filters) -> pd.DataFrame:
    out = df
    if f.year is not None:
        out = out[out["collision_year"] == f.year]
    if f.month is not None:
        out = out[out["month"].fillna(-1).astype("int64") == f.month]
    if f.severity:
        out = out[out["severity"] == f.severity]
    if f.road_type is not None:
        out = out[out["road_type"].astype("string") == str(f.road_type)]
    if f.weather is not None:
        out = out[out["weather_conditions"].astype("string") == str(f.weather)]
    if f.lighting is not None:
        out = out[out["light_conditions"].astype("string") == str(f.lighting)]
    if f.speed_limit is not None:
        out = out[out["speed_limit"].fillna(-1).astype("int64") == f.speed_limit]
    if f.urban_rural is not None:
        out = out[out["urban_or_rural_area"].astype("string") == str(f.urban_rural)]
    if f.hour is not None:
        out = out[out["hour"].fillna(-1).astype("int64") == f.hour]
    if f.weekday is not None:
        out = out[out["weekday"].fillna(-1).astype("int64") == f.weekday]
    return out


class AnalyticsService:
    def __init__(self, cfg: BackendConfig, model_service: ModelService):
        self.cfg = cfg
        self.model = model_service
        self._lock = threading.Lock()
        self._df: pd.DataFrame | None = None
        self._labels: dict = {}
        self._cache: dict[tuple, object] = {}
        self._error: str | None = None

    # ------------------------------------------------------------------ lifecycle
    def load(self) -> bool:
        with self._lock:
            try:
                path = self.cfg.processed_path
                if not path.exists():
                    raise FileNotFoundError(
                        f"Processed dataset not found: {path}. Run: python scripts/prepare_data.py")
                if path.suffix == ".parquet":
                    import pyarrow.parquet as pq
                    available = set(pq.read_schema(path).names)
                    cols = [c for c in ANALYTICS_COLUMNS if c in available]
                    self._df = pq.read_table(path, columns=cols).to_pandas()
                else:
                    self._df = pd.read_pickle(path)
                    self._df = self._df[[c for c in ANALYTICS_COLUMNS if c in self._df.columns]]
                fmeta_path = self.cfg.artifacts_dir / "feature_metadata.json"
                if fmeta_path.exists():
                    with open(fmeta_path, encoding="utf-8") as fh:
                        self._labels = json.load(fh).get("labels", {})
                self._error = None
                return True
            except Exception as e:                       # noqa: BLE001 - reported via /health
                self._error = f"{type(e).__name__}: {e}"
                self._df = None
                return False

    @property
    def loaded(self) -> bool:
        return self._df is not None

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def data(self) -> pd.DataFrame:
        if self._df is None:
            raise RuntimeError(self._error or "Analytics data not loaded.")
        return self._df

    # ------------------------------------------------------------------ helpers
    def _cached(self, key: tuple, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    def _label(self, col: str, key) -> str:
        if key is None or (isinstance(key, float) and pd.isna(key)):
            return "Unknown"
        table = self._labels.get(col, {})
        return table.get(str(key), str(key))

    def _breakdown(self, df: pd.DataFrame, col: str, label_fn=None, order=None) -> list[dict]:
        """Severity breakdown per value of `col` (real counts from the data)."""
        if df.empty:
            return []
        sub = df[[col, "severity"]].copy()
        counts = (sub.groupby([col, "severity"], observed=True, dropna=False)
                  .size().unstack(fill_value=0))
        for s in SEVERITY_ORDER:
            if s not in counts.columns:
                counts[s] = 0
        counts["__total"] = counts[SEVERITY_ORDER].sum(axis=1)
        flat = counts.reset_index()
        norm_keys = [_norm_key(v) for v in flat[col]]
        if order is not None:
            rank = {k: i for i, k in enumerate(order)}
            idx = sorted(range(len(flat)), key=lambda i: rank.get(norm_keys[i], len(rank)))
        else:
            idx = sorted(range(len(flat)), key=lambda i: (norm_keys[i] is None, str(norm_keys[i])))
        label_fn = label_fn or (lambda k: self._label(col, k))
        items = []
        for i in idx:
            row = flat.iloc[i]
            k = norm_keys[i]
            total = int(row["__total"])
            slight, serious, fatal = (int(row[s]) for s in SEVERITY_ORDER)
            items.append({
                "key": k,
                "label": label_fn(k),
                "total": total,
                "slight": slight, "serious": serious, "fatal": fatal,
                "slight_pct": round(100 * slight / total, 2) if total else 0.0,
                "serious_pct": round(100 * serious / total, 2) if total else 0.0,
                "fatal_pct": round(100 * fatal / total, 2) if total else 0.0,
            })
        return items

    # ------------------------------------------------------------------ endpoints
    def severity_distribution(self, f: Filters) -> dict:
        def build():
            df = apply_filters(self.data, f)
            total = len(df)
            counts = df["severity"].value_counts()
            items = [{
                "severity": s,
                "count": int(counts.get(s, 0)),
                "percentage": round(100 * float(counts.get(s, 0)) / total, 2) if total else 0.0,
            } for s in SEVERITY_ORDER]
            return {"total": total, "items": items}
        return self._cached(("severity", f.key()), build)

    def summary(self) -> dict:
        return self._cached(("summary",),
                            lambda: self._summary_payload(self.data, filtered=False))

    def _summary_payload(self, df: pd.DataFrame, filtered: bool) -> dict:
        """Headline numbers for `df` (the whole dataset, or a cross-filtered subset)."""
        total = len(df)
        counts = df["severity"].value_counts()
        years = sorted(int(y) for y in df["collision_year"].dropna().unique())
        sev = [{"severity": s, "count": int(counts.get(s, 0)),
                "percentage": round(100 * float(counts.get(s, 0)) / total, 2) if total else 0.0}
               for s in SEVERITY_ORDER]
        out = {
            "total_collisions": total,
            "severity_distribution": sev,
            "years": years,
            "dataset_period": f"{min(years)}-{max(years)}" if years else None,
            "source": "UK Department for Transport STATS19 collision data (police-reported)",
            "filtered": filtered,
        }
        if self.model.loaded:
            meta = self.model.model_metadata
            mm = {}
            mm_path = self.cfg.artifacts_dir / "model_metrics.json"
            if mm_path.exists():
                with open(mm_path, encoding="utf-8") as fh:
                    mm = json.load(fh)
            test = mm.get("test") or {}
            out["model"] = {
                "name": meta.get("model_name"),
                "training_date": meta.get("training_date"),
                "macro_f1": test.get("macro_f1"),
                "balanced_accuracy": test.get("balanced_accuracy"),
                "accuracy": test.get("accuracy"),
                "macro_f1_label": "test macro F1",
            }
        return out

    # -------------------------------------------------- cross-filtered dashboard
    def filter_labels(self) -> dict:
        """Category label per filter value, computed once from the full dataset.
        The frontend uses these for the active-filter chips (e.g. month 5 -> 'May')."""
        def build():
            e = Filters()
            return {
                "year": {str(i["key"]): i["label"] for i in self.by_year(e)["items"]},
                "month": {str(i["key"]): i["label"] for i in self.by_month(e)["items"]},
                "severity": {i["severity"]: i["severity"] for i in self.severity_distribution(e)["items"]},
                "road_type": {str(i["key"]): i["label"] for i in self.by_road_type(e)["items"]},
                "weather": {str(i["key"]): i["label"] for i in self.by_weather(e)["items"]},
                "lighting": {str(i["key"]): i["label"] for i in self.by_lighting(e)["items"]},
                "speed_limit": {str(i["key"]): i["label"] for i in self.by_speed_limit(e)["items"]},
                "urban_rural": {str(i["key"]): i["label"] for i in self.by_urban_rural(e)["items"]},
                "hour": {str(i["key"]): i["label"] for i in self.by_hour(e)["items"]},
                "weekday": {str(i["key"]): i["label"] for i in self.by_weekday(e)["items"]},
            }
        return self._cached(("filter_labels",), build)

    def dashboard(self, f: Filters) -> dict:
        """ONE request for the whole cross-filtered dashboard (Tableau-style).

        - `summary` (KPI cards) uses ALL active filters.
        - Every series excludes ONLY its own dimension, so the chart the user clicks
          keeps showing all of its categories (the UI emphasises the selected one)
          while being filtered by every other active selection.
        - Everything is computed from the real processed STATS19 DataFrame; nothing
          is mocked and nothing is re-read from disk.
        """
        def build():
            return {
                "summary": self._summary_payload(apply_filters(self.data, f), filtered=not f.is_empty),
                "severity": self.severity_distribution(f.exclude("severity")),
                "series": {
                    "by-year": self.by_year(f.exclude("year")),
                    "by-month": self.by_month(f.exclude("month")),
                    "by-road-type": self.by_road_type(f.exclude("road_type")),
                    "by-weather": self.by_weather(f.exclude("weather")),
                    "by-lighting": self.by_lighting(f.exclude("lighting")),
                    "by-speed-limit": self.by_speed_limit(f.exclude("speed_limit")),
                    "by-hour": self.by_hour(f.exclude("hour")),
                    "by-weekday": self.by_weekday(f.exclude("weekday")),
                    "by-urban-rural": self.by_urban_rural(f.exclude("urban_rural")),
                },
                "labels": self.filter_labels(),
            }
        return self._cached(("dashboard", f.key()), build)

    def by_year(self, f: Filters) -> dict:
        return self._cached(("by_year", f.key()),
                            lambda: {"items": self._breakdown(
                                apply_filters(self.data, f), "collision_year",
                                label_fn=lambda k: str(int(k)),
                                order=sorted(self.data["collision_year"].dropna().unique()))})

    def by_road_type(self, f: Filters) -> dict:
        return self._cached(("by_road_type", f.key()),
                            lambda: {"items": self._breakdown(
                                apply_filters(self.data, f), "road_type",
                                label_fn=lambda k: self._label("road_type", k))})

    def by_weather(self, f: Filters) -> dict:
        return self._cached(("by_weather", f.key()),
                            lambda: {"items": self._breakdown(
                                apply_filters(self.data, f), "weather_conditions",
                                label_fn=lambda k: self._label("weather_conditions", k))})

    def by_lighting(self, f: Filters) -> dict:
        return self._cached(("by_lighting", f.key()),
                            lambda: {"items": self._breakdown(
                                apply_filters(self.data, f), "light_conditions",
                                label_fn=lambda k: self._label("light_conditions", k))})

    def by_speed_limit(self, f: Filters) -> dict:
        def lab(k):
            return "Unknown" if k is None else f"{int(k)} mph"
        return self._cached(("by_speed_limit", f.key()),
                            lambda: {"items": self._breakdown(
                                apply_filters(self.data, f), "speed_limit", label_fn=lab,
                                order=[20, 30, 40, 50, 60, 70, None])})

    def by_hour(self, f: Filters) -> dict:
        def build():
            df = apply_filters(self.data, f)
            items = self._breakdown(df, "hour",
                                    label_fn=lambda k: "Unknown" if k is None else f"{int(k):02d}:00",
                                    order=range(24))
            known = {it["key"] for it in items}
            items += [{"key": h, "label": f"{h:02d}:00", "total": 0,
                       "slight": 0, "serious": 0, "fatal": 0,
                       "slight_pct": 0.0, "serious_pct": 0.0, "fatal_pct": 0.0}
                      for h in range(24) if h not in known]
            items.sort(key=lambda it: (it["key"] is None, it["key"] if it["key"] is not None else 0))
            return {"items": items}
        return self._cached(("by_hour", f.key()), build)

    def by_weekday(self, f: Filters) -> dict:
        def build():
            df = apply_filters(self.data, f)
            items = self._breakdown(df, "weekday",
                                    label_fn=lambda k: "Unknown" if k is None else WEEKDAY_LABELS[int(k)],
                                    order=range(7))
            known = {it["key"] for it in items}
            items += [{"key": d, "label": WEEKDAY_LABELS[d], "total": 0,
                       "slight": 0, "serious": 0, "fatal": 0,
                       "slight_pct": 0.0, "serious_pct": 0.0, "fatal_pct": 0.0}
                      for d in range(7) if d not in known]
            items.sort(key=lambda it: (it["key"] is None, it["key"] if it["key"] is not None else 0))
            return {"items": items}
        return self._cached(("by_weekday", f.key()), build)

    def by_month(self, f: Filters) -> dict:
        def build():
            df = apply_filters(self.data, f)
            items = self._breakdown(df, "month",
                                    label_fn=lambda k: "Unknown" if k is None else MONTH_LABELS[int(k) - 1],
                                    order=range(1, 13))
            known = {it["key"] for it in items}
            items += [{"key": m, "label": MONTH_LABELS[m - 1], "total": 0,
                       "slight": 0, "serious": 0, "fatal": 0,
                       "slight_pct": 0.0, "serious_pct": 0.0, "fatal_pct": 0.0}
                      for m in range(1, 13) if m not in known]
            items.sort(key=lambda it: (it["key"] is None, it["key"] if it["key"] is not None else 0))
            return {"items": items}
        return self._cached(("by_month", f.key()), build)

    def by_urban_rural(self, f: Filters) -> dict:
        return self._cached(("by_urban_rural", f.key()),
                            lambda: {"items": self._breakdown(
                                apply_filters(self.data, f), "urban_or_rural_area",
                                label_fn=lambda k: self._label("urban_or_rural_area", k))})

    def by_junction(self, f: Filters) -> dict:
        return self._cached(("by_junction", f.key()),
                            lambda: {"items": self._breakdown(
                                apply_filters(self.data, f), "junction_detail",
                                label_fn=lambda k: self._label("junction_detail", k))})

    # ------------------------------------------------------------------ map
    def map_points(self, f: Filters, max_cells: int = 15000) -> dict:
        def build():
            df = apply_filters(self.data, f)
            pts = df.dropna(subset=["lat_r", "lon_r"])
            if pts.empty:
                return {"points": [], "total_cells": 0, "truncated": False,
                        "note": "No coordinates available for this filter."}
            g = (pts.groupby(["lat_r", "lon_r", "severity"], observed=True)
                 .size().unstack(fill_value=0))
            for s in SEVERITY_ORDER:
                if s not in g.columns:
                    g[s] = 0
            g["__total"] = g[SEVERITY_ORDER].sum(axis=1)
            g = g.sort_values("__total", ascending=False)
            truncated = len(g) > max_cells
            g = g.head(max_cells)
            points = [{"lat": float(lat), "lon": float(lon),
                       "total": int(r["__total"]),
                       "slight": int(r["Slight"]), "serious": int(r["Serious"]),
                       "fatal": int(r["Fatal"])}
                      for (lat, lon), r in g.iterrows()]
            return {"points": points, "total_cells": int(len(g)),
                    "truncated": truncated,
                    "note": ("Aggregated 0.01-degree grid cells of HISTORICAL police-reported "
                             "collisions. This is not a prediction of future collisions.")}
        return self._cached(("map", f.key()), build)

    # ------------------------------------------------------------------ insights
    def insights(self) -> dict:
        """Descriptive findings computed live from the real data (no causal claims)."""
        def build():
            df = self.data
            total = len(df)
            findings = []

            def fatal_share(sub):
                n = len(sub)
                return (n, 100 * (sub["severity"] == "Fatal").mean() if n else 0.0,
                        100 * (sub["severity"] != "Slight").mean() if n else 0.0)

            # 1. urban vs rural
            if "urban_or_rural_area" in df:
                for code, name in (("1", "Urban"), ("2", "Rural")):
                    sub = df[df["urban_or_rural_area"].astype("string") == code]
                    n, fp, sp = fatal_share(sub)
                    findings.append({
                        "id": f"urban_rural_{name.lower()}",
                        "category": "Road characteristics",
                        "title": f"{name} collisions",
                        "stat": f"{n:,} collisions ({100 * n / total:.1f}%); "
                                f"Fatal {fp:.2f}%, Fatal-or-Serious {sp:.1f}%",
                        "numbers": {"count": int(n), "fatal_pct": round(fp, 2),
                                    "serious_or_fatal_pct": round(sp, 1)},
                    })
            # 2. speed limit bands
            if "speed_limit" in df:
                sp = df.dropna(subset=["speed_limit"]).copy()
                if not sp.empty:
                    sp["band"] = sp["speed_limit"].fillna(-1).astype(int)
                    grp = sp.groupby("band").agg(
                        n=("severity", "size"),
                        fatal=("severity", lambda s: 100 * (s == "Fatal").mean()))
                    grp = grp[grp["n"] >= 100].sort_values("fatal", ascending=False)
                    if len(grp):
                        top = grp.index[0]
                        findings.append({
                            "id": "speed_fatal", "category": "Road characteristics",
                            "title": "Highest fatal share by speed limit",
                            "stat": f"{top} mph roads: {grp.loc[top, 'fatal']:.2f}% of their "
                                    f"{int(grp.loc[top, 'n']):,} collisions were Fatal "
                                    f"(lowest: {grp.index[-1]} mph at {grp['fatal'].iloc[-1]:.2f}%)",
                            "numbers": {"band_mph": int(top),
                                        "fatal_pct": round(float(grp.loc[top, "fatal"]), 2)},
                        })
            # 3. lighting
            if "light_conditions" in df:
                dark = df[df["light_conditions"].astype("string") != "1"]
                light = df[df["light_conditions"].astype("string") == "1"]
                for sub, name in ((dark, "Hours of darkness"), (light, "Daylight")):
                    n, fp, _ = fatal_share(sub)
                    if n:
                        findings.append({
                            "id": f"light_{name.split()[0].lower()}",
                            "category": "Environmental conditions",
                            "title": f"{name}",
                            "stat": f"{n:,} collisions ({100 * n / total:.1f}%); Fatal share {fp:.2f}%",
                            "numbers": {"count": int(n), "fatal_pct": round(fp, 2)},
                        })
            # 4. weather: fine vs adverse
            if "weather_conditions" in df:
                adverse = df[df["weather_conditions"].astype("string").isin(["2", "3", "4", "5", "6", "7"])]
                n, fp, sp = fatal_share(adverse)
                if n:
                    findings.append({
                        "id": "adverse_weather", "category": "Environmental conditions",
                        "title": "Adverse weather (rain, snow, wind, fog)",
                        "stat": f"{n:,} collisions ({100 * n / total:.1f}%); Fatal share {fp:.2f}%, "
                                f"Fatal-or-Serious {sp:.1f}%",
                        "numbers": {"count": int(n), "fatal_pct": round(fp, 2)},
                    })
            # 5. road surface wet vs dry
            if "road_surface_conditions" in df:
                wet = df[df["road_surface_conditions"].astype("string").isin(["2", "3", "4", "5", "6", "7"])]
                n, fp, sp = fatal_share(wet)
                if n:
                    findings.append({
                        "id": "wet_surface", "category": "Environmental conditions",
                        "title": "Wet / compromised road surface",
                        "stat": f"{n:,} collisions ({100 * n / total:.1f}%); Fatal share {fp:.2f}%, "
                                f"Fatal-or-Serious {sp:.1f}%",
                        "numbers": {"count": int(n), "fatal_pct": round(fp, 2)},
                    })
            # 6. temporal: peak hour + weekend share
            if "hour" in df:
                hour_counts = df["hour"].value_counts()
                if len(hour_counts):
                    peak = int(hour_counts.idxmax())
                    findings.append({
                        "id": "peak_hour", "category": "Temporal patterns",
                        "title": "Busiest hour of the day",
                        "stat": f"{peak:02d}:00 with {int(hour_counts.max()):,} collisions "
                                f"in {min(df['collision_year'].astype(int))}-{max(df['collision_year'].astype(int))}",
                        "numbers": {"hour": peak, "count": int(hour_counts.max())},
                    })
            if "weekday" in df:
                wknd = df[df["weekday"].isin([5, 6])]
                n, fp, _ = fatal_share(wknd)
                if n:
                    findings.append({
                        "id": "weekend", "category": "Temporal patterns",
                        "title": "Weekend share (Fri-Sun weeks: Sat/Sun)",
                        "stat": f"{n:,} collisions fell on Sat/Sun ({100 * n / total:.1f}%); "
                                f"Fatal share {fp:.2f}%",
                        "numbers": {"count": int(n), "fatal_pct": round(fp, 2)},
                    })
            # 7. year trend of fatal share
            if "collision_year" in df:
                yr = df.groupby(df["collision_year"].astype(int)).agg(
                    n=("severity", "size"),
                    fatal_pct=("severity", lambda s: 100 * (s == "Fatal").mean()),
                    sf_pct=("severity", lambda s: 100 * (s != "Slight").mean()))
                if len(yr) > 1:
                    findings.append({
                        "id": "year_trend", "category": "Temporal patterns",
                        "title": "Yearly severity mix",
                        "stat": "; ".join(f"{y}: Fatal {r['fatal_pct']:.2f}%, "
                                          f"Serious-or-Fatal {r['sf_pct']:.1f}% (n={int(r['n']):,})"
                                          for y, r in yr.iterrows()),
                        "numbers": {str(int(y)): {"fatal_pct": round(float(r["fatal_pct"]), 2),
                                                  "count": int(r["n"])} for y, r in yr.iterrows()},
                    })
            # 8. junctions
            if "junction_detail" in df:
                j = df.groupby(df["junction_detail"].astype("string")).agg(
                    n=("severity", "size"),
                    fatal_pct=("severity", lambda s: 100 * (s == "Fatal").mean()))
                j = j[j["n"] >= 500]
                if len(j):
                    top = j["fatal_pct"].idxmax()
                    label = self._label("junction_detail", top)
                    findings.append({
                        "id": "junction", "category": "Road characteristics",
                        "title": "Highest fatal share by junction type",
                        "stat": f"{label}: {j.loc[top, 'fatal_pct']:.2f}% Fatal "
                                f"(n={int(j.loc[top, 'n']):,})",
                        "numbers": {"junction": label,
                                    "fatal_pct": round(float(j.loc[top, "fatal_pct"]), 2)},
                    })
            return {
                "findings": findings,
                "rows_analysed": int(total),
                "note": ("DATA FINDING: descriptive statistics of historical police-reported "
                         "collisions. Associations only - no causal claims. Fatal shares are "
                         "not adjusted for exposure (traffic volume)."),
            }
        return self._cached(("insights",), build)
