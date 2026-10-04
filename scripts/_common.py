"""Shared bootstrap for the CLI scripts (adds the project root to sys.path, builds Settings)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.config import get_settings, parse_years  # noqa: E402


def parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--years", help="e.g. 2021-2025 or 2019,2021,2023 (overrides ROADRISK_YEARS)")
    return p


def settings_from(args):
    cfg = get_settings()
    if getattr(args, "years", None):
        cfg.years = parse_years(args.years)
    return cfg


def run(main):
    """Run main(); print a clean error and exit non-zero instead of a stack trace for expected failures."""
    try:
        sys.exit(main() or 0)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"\n[roadrisk] ERROR: {e}", file=sys.stderr)
        sys.exit(1)
