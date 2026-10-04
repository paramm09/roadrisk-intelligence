"""Central configuration. Every value can be overridden with an environment variable
(ROADRISK_*) or, for scripts, a command-line flag. Nothing here is a result.

Examples (PowerShell):  $env:ROADRISK_YEARS = "2020-2024"
Examples (cmd.exe):     set ROADRISK_YEARS=2020-2024
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_YEARS_SPEC = "2021-2025"

# Official DfT road-safety data URL pattern. NOTE: this could not be verified from the
# development sandbox (network blocked). If a download fails, edit this template, set
# ROADRISK_URL_TEMPLATE, or pass --url YEAR=URL to scripts/download_data.py, or download
# manually (see README, "Manual download fallback").
DEFAULT_URL_TEMPLATE = (
    "https://data.dft.gov.uk/road-accidents-safety-data/"
    "dft-road-casualty-statistics-collision-{year}.csv"
)


def parse_years(spec) -> list[int]:
    """'2021-2025' -> [2021..2025];  '2019,2021,2023' -> [2019, 2021, 2023]."""
    if isinstance(spec, (list, tuple)):
        return sorted({int(y) for y in spec})
    spec = str(spec).strip()
    if "," in spec:
        return sorted({int(x) for x in spec.split(",") if x.strip()})
    if "-" in spec:
        a, b = spec.split("-", 1)
        a, b = int(a), int(b)
        if b < a:
            raise ValueError(f"Invalid year range: {spec}")
        return list(range(a, b + 1))
    return [int(spec)]


def _env(name: str, default, cast=str):
    return field(default_factory=lambda: cast(os.environ.get(name, default)))


def _env_bool(v) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_optional_int(v):
    v = str(v).strip()
    return int(v) if v not in {"", "none", "None"} else None


@dataclass
class Settings:
    # ---- paths -----------------------------------------------------------------
    root: Path = field(default_factory=lambda: Path(
        os.environ.get("ROADRISK_ROOT") or Path(__file__).resolve().parents[1]))

    # ---- dataset period (configurable) ------------------------------------------
    years: list = field(default_factory=lambda: parse_years(
        os.environ.get("ROADRISK_YEARS", DEFAULT_YEARS_SPEC)))
    url_template: str = _env("ROADRISK_URL_TEMPLATE", DEFAULT_URL_TEMPLATE)
    url_overrides: dict = field(default_factory=lambda: json.loads(
        os.environ.get("ROADRISK_URL_OVERRIDES", "{}")))  # {"2025": "https://..."}

    # ---- RAM controls -------------------------------------------------------------
    chunk_size: int = _env("ROADRISK_CHUNK_SIZE", 100_000, int)
    max_rows_per_year: object = field(default_factory=lambda: _env_optional_int(
        os.environ.get("ROADRISK_MAX_ROWS", "")))  # development only; recorded in reports
    num_vehicles_cap: int = _env("ROADRISK_VEHICLE_CAP", 10, int)

    # ---- reproducibility / splitting ---------------------------------------------
    random_seed: int = _env("ROADRISK_SEED", 42, int)
    split_strategy: str = _env("ROADRISK_SPLIT", "random", str)   # random | temporal
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15

    # ---- imbalance ------------------------------------------------------------------
    imbalance_strategy: str = _env("ROADRISK_IMBALANCE", "CLASS_WEIGHT", str)  # NONE|CLASS_WEIGHT|SMOTE
    smote_ratio: float = _env("ROADRISK_SMOTE_RATIO", 0.25, float)
    run_balancing_comparison: bool = field(default_factory=lambda: _env_bool(
        os.environ.get("ROADRISK_BALANCING_COMPARISON", "1")))

    # ---- tuning -----------------------------------------------------------------------
    search_n_iter: int = _env("ROADRISK_SEARCH_ITER", 8, int)
    cv_folds: int = _env("ROADRISK_CV_FOLDS", 3, int)
    n_jobs: int = _env("ROADRISK_N_JOBS", 1, int)
    tuning_max_rows: int = _env("ROADRISK_TUNING_ROWS", 100_000, int)
    refit_on_train_val: bool = field(default_factory=lambda: _env_bool(
        os.environ.get("ROADRISK_REFIT_TRAIN_VAL", "1")))

    # ---- explainability ---------------------------------------------------------------
    permutation_sample: int = _env("ROADRISK_PERM_SAMPLE", 20_000, int)
    permutation_repeats: int = _env("ROADRISK_PERM_REPEATS", 3, int)

    # ---- safety -------------------------------------------------------------------------
    allow_test_data: bool = field(default_factory=lambda: _env_bool(
        os.environ.get("ROADRISK_ALLOW_TEST_DATA", "0")))  # tests only

    # ---- derived paths ------------------------------------------------------------------
    @property
    def raw_dir(self) -> Path: return self.root / "data" / "raw"
    @property
    def processed_dir(self) -> Path: return self.root / "data" / "processed"
    @property
    def sample_dir(self) -> Path: return self.root / "data" / "sample"
    @property
    def reports_dir(self) -> Path: return self.root / "reports"
    @property
    def artifacts_dir(self) -> Path: return self.root / "ml" / "artifacts"
    @property
    def processed_base(self) -> Path: return self.processed_dir / "stats19_model_data"
    @property
    def processed_meta_path(self) -> Path: return self.processed_dir / "processed_meta.json"

    def url_for(self, year: int) -> str:
        return self.url_overrides.get(str(year)) or self.url_template.format(year=year)

    def ensure_dirs(self) -> None:
        for d in (self.raw_dir, self.processed_dir, self.reports_dir, self.artifacts_dir):
            d.mkdir(parents=True, exist_ok=True)

    def validate(self) -> None:
        if abs(self.train_frac + self.val_frac + self.test_frac - 1.0) > 1e-9:
            raise ValueError("train/val/test fractions must sum to 1.")
        if self.split_strategy not in {"random", "temporal"}:
            raise ValueError("ROADRISK_SPLIT must be 'random' or 'temporal'.")
        if self.imbalance_strategy not in {"NONE", "CLASS_WEIGHT", "SMOTE"}:
            raise ValueError("ROADRISK_IMBALANCE must be NONE, CLASS_WEIGHT or SMOTE.")

    def snapshot(self) -> dict:
        """JSON-safe copy of the settings that affect results (saved in metadata)."""
        d = {k: getattr(self, k) for k in (
            "years", "random_seed", "split_strategy", "train_frac", "val_frac", "test_frac",
            "imbalance_strategy", "smote_ratio", "search_n_iter", "cv_folds", "n_jobs",
            "tuning_max_rows", "refit_on_train_val", "chunk_size", "max_rows_per_year",
            "num_vehicles_cap")}
        return d


def get_settings() -> Settings:
    s = Settings()
    s.validate()
    return s
