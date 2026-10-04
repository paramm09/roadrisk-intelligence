"""Loads the REAL trained artifacts once at startup. Never trains, never fabricates.

If artifacts are missing the API stays up but reports model_loaded=false and the
prediction endpoints answer 503 with instructions to run the training scripts.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from ..config import BackendConfig
from ml.predict import ArtifactsNotFoundError, Predictor
from . import model_registry


def _read_json(path: Path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


class ModelService:
    """Singleton-style loader: one Predictor, one set of reports, loaded once."""

    def __init__(self, cfg: BackendConfig):
        self.cfg = cfg
        self._lock = threading.Lock()
        self._predictor: Predictor | None = None
        self._error: str | None = None
        self._feature_metadata: dict | None = None
        self._model_metadata: dict | None = None
        self._registry: list[dict] = []

    # ------------------------------------------------------------------ lifecycle
    def load(self) -> bool:
        with self._lock:
            if self._predictor is not None:
                return True
            try:
                self._predictor = Predictor(artifacts_dir=self.cfg.artifacts_dir)
                self._feature_metadata = _read_json(self.cfg.artifacts_dir / "feature_metadata.json")
                self._model_metadata = _read_json(self.cfg.artifacts_dir / "model_metadata.json")
                self._registry = model_registry.build_registry(self.cfg, self.model_metadata)
                self._error = None
                return True
            except ArtifactsNotFoundError as e:
                self._error = str(e)
                return False

    @property
    def loaded(self) -> bool:
        return self._predictor is not None

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def predictor(self) -> Predictor:
        if self._predictor is None:
            raise ArtifactsNotFoundError(
                self._error or "Model artifacts are not loaded. Run: python scripts/train_models.py")
        return self._predictor

    # ------------------------------------------------------------------ read-only data
    @property
    def feature_metadata(self) -> dict:
        return self._feature_metadata or {}

    @property
    def model_metadata(self) -> dict:
        return self._model_metadata or {}

    def info(self) -> dict:
        """GET /api/model/info - model identity + the exact accepted input fields."""
        base = self.predictor.info()
        fmeta = self.feature_metadata
        return {
            **base,
            "artifacts_dir": str(self.cfg.artifacts_dir),
            "input_fields": fmeta.get("input_fields", []),
            "labels": fmeta.get("labels", {}),
            "value_options": fmeta.get("value_options", {}),
            "feature_families": fmeta.get("feature_families", {}),
            "years_used": self.model_metadata.get("years_used", []),
            "dataset_period": self.model_metadata.get("dataset_period"),
            "selection_rationale": self.model_metadata.get("selection_rationale"),
            "probability_note": self.model_metadata.get("probability_note"),
        }

    # ------------------------------------------------------------------ model registry
    @property
    def registry(self) -> list[dict]:
        """Real trained models whose artifacts exist on disk (see model_registry)."""
        if not self._registry and self.loaded:
            self._registry = model_registry.build_registry(self.cfg, self.model_metadata)
        return self._registry

    def list_models(self) -> dict:
        """GET /api/models - genuinely available trained models + their real metrics."""
        selection = (self.model_metadata.get("selection_rationale") or {})
        return {
            "models": self.registry,
            "default_model": self.model_metadata.get("model_key"),
            "default_model_name": self.model_metadata.get("model_name"),
            "selection_rule": selection.get("rule"),
            "note": ("'auto' (null) uses the model selected by the training pipeline on validation "
                     "metrics, served from final_model.joblib (refit on train+validation). Manually "
                     "selected models load their own trained artifact with its own fitted "
                     "preprocessing. Test metrics exist only for the auto-selected final model "
                     "(single evaluation on the untouched test split)."),
        }

    def resolve_model(self, model_id: str | None) -> tuple[str | None, str]:
        """(artifact file, display name) for a requested model id; 'auto'/None -> the
        pipeline's selected model. Raises ModelNotAvailableError for unknown ids."""
        return model_registry.resolve(self.registry, model_id, self.model_metadata)

    def metrics(self) -> dict:
        """GET /api/model/metrics - validation (all models) + final test metrics, as recorded."""
        model_metrics = _read_json(self.cfg.artifacts_dir / "model_metrics.json") or {}
        final_eval = _read_json(self.cfg.reports_dir / "final_evaluation.json") or None
        comparison = _read_json(self.cfg.reports_dir / "model_comparison.json") or None
        test = model_metrics.get("test") or (final_eval or {}).get("test_metrics")
        validation = model_metrics.get("validation") or (
            {k: {"display_name": v.get("display_name"), **v.get("validation", {})}
             for k, v in (comparison or {}).get("models", {}).items()})
        return {
            "model": self.model_metadata.get("model_name"),
            "validation": validation,
            "test": test,
            "selection": (comparison or {}).get("selection")
            or self.model_metadata.get("selection_rationale"),
            "error_analysis": (final_eval or {}).get("error_analysis"),
            "note": "Test metrics are from the single final evaluation on the untouched test split.",
        }

    def feature_importance(self):
        """GET /api/model/feature-importance - reports/feature_importance.json (real, computed)."""
        imp = _read_json(self.cfg.reports_dir / "feature_importance.json")
        if imp is None:
            raise FileNotFoundError(
                "reports/feature_importance.json not found. Run: python scripts/evaluate_models.py")
        return imp
