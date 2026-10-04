"""Backend configuration: paths into the ML project + CORS. No secrets, no data.

Environment overrides (mainly for tests):
    ROADRISK_ROOT          project root (defaults to the folder containing backend/)
    ROADRISK_ARTIFACTS_DIR trained artifacts dir (default <root>/ml/artifacts)
    ROADRISK_PROCESSED     processed dataset path (default <root>/data/processed/stats19_model_data.parquet)
    ROADRISK_CORS_ORIGINS  comma-separated origins (default localhost:3000 variants)
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:          # so `import ml.*` works from uvicorn/pytest
    sys.path.insert(0, str(PROJECT_ROOT))


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default) or default


@dataclass(frozen=True)
class BackendConfig:
    root: Path = field(default_factory=lambda: Path(_env("ROADRISK_ROOT", str(PROJECT_ROOT))))
    artifacts_override: Path | None = field(default_factory=lambda: (
        Path(os.environ["ROADRISK_ARTIFACTS_DIR"]) if os.environ.get("ROADRISK_ARTIFACTS_DIR") else None))
    processed_override: Path | None = field(default_factory=lambda: (
        Path(os.environ["ROADRISK_PROCESSED"]) if os.environ.get("ROADRISK_PROCESSED") else None))
    cors_origins: tuple[str, ...] = field(default_factory=lambda: tuple(
        o.strip() for o in _env(
            "ROADRISK_CORS_ORIGINS",
            "http://localhost:3000,http://127.0.0.1:3000").split(",") if o.strip()))

    @property
    def artifacts_dir(self) -> Path:
        return self.artifacts_override or (self.root / "ml" / "artifacts")

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def processed_path(self) -> Path:
        if self.processed_override:
            return self.processed_override
        p = self.root / "data" / "processed"
        for name in ("roadrisk_processed.parquet", "stats19_model_data.parquet",
                     "stats19_model_data.pkl", "roadrisk_processed.pkl"):
            cand = p / name
            if cand.exists():
                return cand
        return p / "roadrisk_processed.parquet"        # default (missing -> clear error)


def get_config() -> BackendConfig:
    return BackendConfig()
