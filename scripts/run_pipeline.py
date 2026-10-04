"""Run the full ML pipeline sequentially: inspect -> prepare -> EDA -> train -> evaluate.

    python scripts/run_pipeline.py

Configuration is via environment variables (e.g. ROADRISK_YEARS=2021,2022) exactly as for
the individual scripts; extra CLI flags are intentionally NOT forwarded because not every
step accepts every flag. Stops at the first failing stage (non-zero exit code) so a broken
stage can never be silently skipped.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

STEPS = (
    ("inspect data", "scripts/inspect_data.py"),
    ("prepare data", "scripts/prepare_data.py"),
    ("EDA", "scripts/run_eda.py"),
    ("train models", "scripts/train_models.py"),
    ("evaluate final model", "scripts/evaluate_models.py"),
)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    for label, rel in STEPS:
        print(f"\n{'=' * 70}\nPIPELINE STEP: {label}  ({rel})\n{'=' * 70}", flush=True)
        result = subprocess.run([sys.executable, str(root / rel)], cwd=root)
        if result.returncode != 0:
            print(f"\nERROR: step '{label}' failed with exit code {result.returncode}. "
                  f"Pipeline stopped; later steps were NOT run.", flush=True)
            return result.returncode
    print("\nPIPELINE COMPLETE: all steps finished successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
