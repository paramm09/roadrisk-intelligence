"""Central registry of the REAL trained model artifacts.

Built exclusively from what the training pipeline produced:
- ml/artifacts/*.joblib            (the fitted pipelines themselves)
- ml/artifacts/model_metrics.json  (validation metrics for every candidate)
- reports/model_comparison.json    (display names + artifact paths + selection rule)
- reports/final_evaluation.json    (test metrics - final model only)
- ml/artifacts/model_metadata.json (which model the pipeline selected)

Rules:
- An entry only appears if its artifact file exists on disk (no phantom models).
- Every metric is read from the reports - nothing is invented here.
- The auto-selected model maps to final_model.joblib (the refit of the winner),
  so "Auto" and manually selecting that same model run the identical artifact.
"""
from __future__ import annotations

import json
from pathlib import Path

FINAL_ARTIFACT = "final_model.joblib"

# Mirrors ml/train.py MODEL_FILES. Only used when reports/model_comparison.json
# is unavailable; entries are still filtered by files that actually exist.
FALLBACK_ARTIFACTS = {
    "baseline": "baseline_model.joblib",
    "logistic_regression": "logistic_regression.joblib",
    "random_forest": "random_forest.joblib",
    "gradient_boosting": "gradient_boosting.joblib",
}

VAL_METRICS = ("accuracy", "macro_f1", "balanced_accuracy", "fatal_recall")
TEST_METRICS = ("accuracy", "macro_f1", "balanced_accuracy", "fatal_recall")

# Model ids that mean "use the pipeline's selected best-validation model".
AUTO_IDS = {"", "auto"}


class ModelNotAvailableError(Exception):
    """The caller explicitly requested a model that does not exist / is not loaded."""


def _read_json(path: Path) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _artifact_filename(recorded: str | None) -> str | None:
    """reports store e.g. 'ml\\artifacts\\random_forest.joblib' (platform-specific)."""
    if not recorded:
        return None
    return Path(str(recorded).replace("\\", "/")).name or None


def _slim(metrics: dict | None, keys: tuple[str, ...]) -> dict:
    out = {}
    for k in keys:
        v = (metrics or {}).get(k)
        if isinstance(v, (int, float)):
            out[k] = float(v)
    return out


def build_registry(cfg, model_metadata: dict) -> list[dict]:
    """Return the list of genuinely available trained models (artifact files on disk)."""
    artifacts_dir = cfg.artifacts_dir
    reports_dir = cfg.reports_dir

    comparison = _read_json(reports_dir / "model_comparison.json")
    candidates = comparison.get("models") or {}
    metrics = _read_json(artifacts_dir / "model_metrics.json")
    validation = metrics.get("validation") or {}
    final_eval = _read_json(reports_dir / "final_evaluation.json")
    test = metrics.get("test") or final_eval.get("test_metrics") or None

    selected_key = model_metadata.get("model_key")
    default_name = model_metadata.get("model_name")
    selection_rule = (comparison.get("selection") or model_metadata.get("selection_rationale") or {}).get("rule")

    keys = list(dict.fromkeys(
        [k for k in list(candidates.keys()) + list(validation.keys()) if k]
        + ([selected_key] if selected_key else [])
    ))

    entries: list[dict] = []
    for key in keys:
        cand = candidates.get(key) or {}
        val_src = validation.get(key) or cand.get("validation") or {}
        name = (cand.get("display_name")
                or val_src.get("display_name")
                or key.replace("_", " ").title())
        is_default = key == selected_key
        # The auto-selected model's production artifact is the refit final model.
        artifact = FINAL_ARTIFACT if is_default else (
            _artifact_filename(cand.get("artifact")) or FALLBACK_ARTIFACTS.get(key))
        if not artifact or not (artifacts_dir / artifact).exists():
            continue  # registry only contains artifacts that really exist
        val = _slim(val_src, VAL_METRICS)
        # Test metrics are computed once, for the auto-selected final model only.
        test_owner = (test or {}).get("model") or (default_name if test else None)
        test_metrics = _slim(test, TEST_METRICS) if test and test_owner == name else None
        entries.append({
            "id": key,
            "name": name,
            "artifact": artifact,
            "available": True,
            "is_default": is_default,
            "validation_metrics": val or None,
            "test_metrics": test_metrics or None,
        })
    return entries


def resolve(registry: list[dict], model_id: str | None, model_metadata: dict) -> tuple[str | None, str]:
    """Map a requested model id -> (artifact file, display name).

    None / "" / "auto" -> the automatically selected model (final refit artifact).
    Raises ModelNotAvailableError for any id that is not in the registry.
    """
    default_entry = next((e for e in registry if e["is_default"]), None)
    if model_id is None or str(model_id).strip().lower() in AUTO_IDS:
        if default_entry:
            return default_entry["artifact"], default_entry["name"]
        return FINAL_ARTIFACT, model_metadata.get("model_name") or "final model"

    wanted = str(model_id).strip()
    entry = next((e for e in registry if e["id"] == wanted), None)
    if entry is None or not entry.get("available"):
        available = ", ".join(e["id"] for e in registry) or "none"
        raise ModelNotAvailableError(
            f"Requested model is not available: '{wanted}'. Available models: {available}. "
            "See GET /api/models.")
    return entry["artifact"], entry["name"]
