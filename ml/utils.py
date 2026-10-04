"""Small shared helpers: logging, JSON writing, frame I/O, memory."""
from __future__ import annotations

import datetime as dt
import gc
import json
import math
import time
from pathlib import Path

import numpy as np


def log(msg: str) -> None:
    print(f"[roadrisk] {msg}", flush=True)


def section(title: str) -> None:
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72, flush=True)


def free_memory() -> None:
    gc.collect()


def now_iso() -> str:
    return dt.datetime.now().replace(microsecond=0).isoformat()


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.t0


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_clean(v) for v in (sorted(o) if isinstance(o, set) else o)]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (Path,)):
        return str(o)
    if isinstance(o, (dt.date, dt.datetime)):
        return o.isoformat()
    return o


def write_json(path: Path, obj) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_clean(obj), f, indent=2, ensure_ascii=True)
    return path


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_frame(df, base: Path) -> Path:
    """Save as parquet if pyarrow is installed, else pickle."""
    base = Path(base)
    base.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow  # noqa: F401
        p = base.with_suffix(".parquet")
        df.to_parquet(p, index=False)
        return p
    except ImportError:
        p = base.with_suffix(".pkl")
        df.to_pickle(p)
        return p


def load_frame(base: Path):
    import pandas as pd
    base = Path(base)
    pq, pk = base.with_suffix(".parquet"), base.with_suffix(".pkl")
    if pq.exists():
        return pd.read_parquet(pq)
    if pk.exists():
        return pd.read_pickle(pk)
    raise FileNotFoundError(f"No processed data at {pq} (or .pkl). Run scripts/prepare_data.py first.")
