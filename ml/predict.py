"""Load the REAL trained artifacts once and serve predictions. No model, no prediction:
if artifacts are missing a clear error is raised (never a fabricated result)."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import Settings, get_settings
from .explain import local_factors
from .feature_config import CLASS_NAMES
from .feature_engineering import build_features
from .utils import read_json

DISCLAIMER = ("Model output is an analytical prediction based on historical reported collision data and should "
              "not be treated as a definitive determination of collision severity or causation.")
REQUIRED_ARTIFACTS = ("final_model.joblib", "feature_metadata.json", "model_metadata.json")
FINAL_ARTIFACT = "final_model.joblib"


class ArtifactsNotFoundError(FileNotFoundError):
    pass


class Predictor:
    def __init__(self, cfg: Settings | None = None, artifacts_dir: Path | None = None):
        cfg = cfg or get_settings()
        self.dir = Path(artifacts_dir) if artifacts_dir else cfg.artifacts_dir
        missing = [f for f in REQUIRED_ARTIFACTS if not (self.dir / f).exists()]
        if missing:
            raise ArtifactsNotFoundError(
                f"Model artifacts missing in {self.dir}: {missing}. Train first: python scripts/train_models.py")
        self.pipeline = joblib.load(self.dir / FINAL_ARTIFACT)
        self.fmeta = read_json(self.dir / "feature_metadata.json")
        self.meta = read_json(self.dir / "model_metadata.json")
        self.features = self.fmeta["features"]
        self.spec = {f["name"]: f for f in self.fmeta["input_fields"]}
        self.cap = self.fmeta["num_vehicles_cap"]
        self._extra_pipelines: dict[str, object] = {}   # lazily loaded, once per artifact

    # ------------------------------------------------------------------ artifacts
    def pipeline_for(self, artifact: str | None = None):
        """Return a REAL fitted pipeline artifact by file name (None -> the final model).

        Every candidate artifact saved by ml/train.py is a complete sklearn Pipeline
        ('prep' + 'clf') fitted on the training split, so each model predicts with the
        exact preprocessing it was trained with. Artifacts are loaded once and cached;
        nothing is ever re-trained here.
        """
        if artifact is None or artifact == FINAL_ARTIFACT:
            return self.pipeline
        pipe = self._extra_pipelines.get(artifact)
        if pipe is None:
            path = self.dir / artifact
            if not path.exists():
                raise ArtifactsNotFoundError(f"Trained model artifact not found: {path}")
            pipe = joblib.load(path)
            self._extra_pipelines[artifact] = pipe
        return pipe

    # ------------------------------------------------------------------ validation
    def validate(self, record: dict) -> dict:
        rec = dict(record)
        if "vehicle_count" in rec and "number_of_vehicles" not in rec:
            rec["number_of_vehicles"] = rec.pop("vehicle_count")
        errors, clean = [], {}
        for name, f in self.spec.items():
            v = rec.get(name)
            if v is None or (isinstance(v, str) and v.strip() == ""):
                if f["required"]:
                    errors.append(f"'{name}' is required")
                continue
            if f["type"] in ("int",):
                try:
                    iv = int(v)
                    if float(v) != iv:
                        raise ValueError
                except (TypeError, ValueError):
                    errors.append(f"'{name}' must be an integer")
                    continue
                if "min" in f and not (f["min"] <= iv <= f["max"]):
                    errors.append(f"'{name}' must be between {f['min']} and {f['max']}")
                    continue
                if "options" in f and f["options"] and iv not in f["options"]:
                    errors.append(f"'{name}' must be one of {f['options']}")
                    continue
                clean[name] = iv
            else:
                sv = str(v).strip()
                try:
                    sv = str(int(float(sv)))
                except ValueError:
                    pass
                allowed = {o["value"] for o in f.get("options", [])}
                if allowed and sv not in allowed:
                    errors.append(f"'{name}' has unknown code '{v}'; allowed: {sorted(allowed)}")
                    continue
                clean[name] = sv
        extra = sorted(set(rec) - set(self.spec) - {"vehicle_count"})
        if extra:
            errors.append(f"unknown fields: {extra}")
        if errors:
            raise ValueError("; ".join(errors))
        return clean

    def _frame(self, cleaned: list[dict]) -> pd.DataFrame:
        raw = pd.DataFrame(cleaned)
        for col in ("month",):          # optional numeric inputs absent -> NaN (median-imputed)
            if col not in raw:
                raw[col] = np.nan
        df, _ = build_features(raw, self.cap)
        for c in self.features:
            if c not in df.columns:
                df[c] = np.nan if c in self.fmeta["numeric_features"] else "unknown"
        return df[self.features]

    # ------------------------------------------------------------------ prediction
    def predict(self, record: dict, top_n: int = 5, artifact: str | None = None,
                model_label: str | None = None) -> dict:
        return self.predict_batch([record], top_n=top_n, artifact=artifact, model_label=model_label)[0]

    def predict_batch(self, records: list[dict], top_n: int = 5, artifact: str | None = None,
                      model_label: str | None = None) -> list[dict]:
        pipe = self.pipeline_for(artifact)
        cleaned = [self.validate(r) for r in records]
        X = self._frame(cleaned)
        proba = pipe.predict_proba(X)
        classes = list(pipe.classes_)
        out = []
        for i in range(len(X)):
            full = np.zeros(3)
            for j, c in enumerate(classes):
                full[int(c)] = proba[i, j]
            k = int(np.argmax(full))
            factors = local_factors(pipe, X.iloc[[i]], self.fmeta["reference_values"], k, top_n)
            out.append({
                "prediction": CLASS_NAMES[k],
                "probabilities": {CLASS_NAMES[c]: float(full[c]) for c in range(3)},
                "confidence": float(full[k]),
                "confidence_note": "Highest model probability for this input; not a certainty.",
                "top_factors": factors,
                "factors_note": "Associations learned from historical data (occlusion vs typical conditions); not causes.",
                "model": model_label or self.meta["model_name"], "disclaimer": DISCLAIMER,
            })
        return out

    def info(self) -> dict:
        m = self.meta
        return {k: m.get(k) for k in ("model_name", "training_date", "dataset_period", "target", "class_names",
                                      "training_rows", "validation_rows", "test_rows", "random_seed",
                                      "best_hyperparameters", "preprocessing_version", "imbalance_strategy",
                                      "n_features_raw", "n_features_after_encoding")} | {"features": self.features}
