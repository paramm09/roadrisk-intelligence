"""Metrics and the FINAL test-set evaluation (the only place the test split is read)."""
from __future__ import annotations

import hashlib
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score,
    log_loss, precision_recall_fscore_support, precision_score, recall_score, roc_auc_score,
)

from .config import Settings
from .explain import family_permutation_importance, native_importance
from .feature_config import CLASS_NAMES, FATAL_INT
from .utils import free_memory, load_frame, log, now_iso, read_json, section, write_json

LABELS = [0, 1, 2]


def compute_metrics(y_true, y_pred, proba=None) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    m = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)),
        "weighted_precision": float(precision_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
        "confusion_matrix_labels": CLASS_NAMES,
        "class_wise": {CLASS_NAMES[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                       for i in range(3)},
        "fatal_recall": float(r[FATAL_INT]), "fatal_precision": float(p[FATAL_INT]), "fatal_f1": float(f[FATAL_INT]),
    }
    if proba is not None:
        proba = np.asarray(proba)
        try:
            m["log_loss"] = float(log_loss(y_true, proba, labels=LABELS))
        except Exception:
            m["log_loss"] = None
        try:
            m["roc_auc_ovr_macro"] = float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=LABELS))
        except Exception:
            m["roc_auc_ovr_macro"] = None
        try:
            m["fatal_average_precision"] = float(average_precision_score(y_true == FATAL_INT, proba[:, FATAL_INT]))
            m["fatal_prevalence"] = float(np.mean(y_true == FATAL_INT))
        except Exception:
            m["fatal_average_precision"] = None
    return m


def time_inference(pipe, X: pd.DataFrame, n: int = 5000, repeats: int = 3) -> float:
    """Best-of-N wall time to predict_proba, in milliseconds per 1000 rows."""
    X = X.iloc[:n]
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        pipe.predict_proba(X)
        best = min(best, time.perf_counter() - t0)
    return best / max(len(X), 1) * 1000.0 * 1000.0


def ids_hash(ids) -> str:
    h = hashlib.sha256()
    for i in sorted(map(str, ids)):
        h.update(i.encode())
    return h.hexdigest()


def _calibration_table(y_true, proba) -> dict:
    out = {}
    for i, c in enumerate(CLASS_NAMES):
        out[c] = {"mean_predicted_probability": float(np.mean(proba[:, i])), "observed_frequency": float(np.mean(y_true == i))}
    return out


def _segment_analysis(X: pd.DataFrame, y, pred, proba, min_n: int = 100) -> dict:
    out = {}
    for col in ("lighting_group", "road_type_group", "speed_band", "urban_or_rural_area", "weather_group", "junction_group"):
        if col not in X.columns:
            continue
        rows = []
        for lvl, idx in X.groupby(col).indices.items():
            if len(idx) < min_n:
                continue
            yt, yp = y[idx], pred[idx]
            nf = int((yt == FATAL_INT).sum())
            rows.append({"level": str(lvl), "n": int(len(idx)), "accuracy": float((yt == yp).mean()),
                         "n_fatal": nf, "fatal_recall": float(((yp == FATAL_INT) & (yt == FATAL_INT)).sum() / nf) if nf else None,
                         "mean_predicted_fatal_probability": float(proba[idx, FATAL_INT].mean()),
                         "observed_fatal_rate": float(nf / len(idx))})
        out[col] = sorted(rows, key=lambda r: -r["n"])
    return out


def run_final_evaluation(cfg: Settings) -> dict:
    section("FINAL EVALUATION ON THE UNTOUCHED TEST SET")
    art = cfg.artifacts_dir
    for f in ("final_model.joblib", "model_metadata.json", "feature_metadata.json"):
        if not (art / f).exists():
            raise FileNotFoundError(f"{art / f} missing. Run scripts/train_models.py first.")
    meta = read_json(art / "model_metadata.json")
    fmeta = read_json(art / "feature_metadata.json")
    pipe = joblib.load(art / "final_model.joblib")
    df = load_frame(cfg.processed_base)
    test = df[df["split"] == "test"].reset_index(drop=True)
    del df
    free_memory()
    if meta.get("test_ids_sha256") and ids_hash(test["collision_index"]) != meta["test_ids_sha256"]:
        raise RuntimeError("Test split changed since training (id hash mismatch). Re-run prepare + train.")
    log(f"test rows: {len(test):,}  (hash verified: test split identical to the one set aside at training time)")
    features = meta["features"]
    X, y = test[features], test["target"].to_numpy()
    pred = pipe.predict(X)
    proba = pipe.predict_proba(X)
    metrics = compute_metrics(y, pred, proba)
    fatal_rows = y == FATAL_INT
    pred_when_fatal = pd.Series(pred[fatal_rows]).map(dict(enumerate(CLASS_NAMES))).value_counts().to_dict()
    errors = {
        "fatal_collisions_in_test": int(fatal_rows.sum()),
        "fatal_collisions_predicted_as": {k: int(v) for k, v in pred_when_fatal.items()},
        "fatal_recall": metrics["fatal_recall"], "fatal_precision": metrics["fatal_precision"],
        "poor_fatal_performance_flag": bool(metrics["fatal_f1"] < 0.30),
        "fatal_flag_rule": "flag set when Fatal F1 < 0.30",
        "predicted_fatal_total": int((pred == FATAL_INT).sum()),
        "segment_performance": _segment_analysis(X.reset_index(drop=True), y, pred, proba),
        "probability_calibration": _calibration_table(y, proba),
        "calibration_note": "Mean predicted probability vs observed frequency. With class_weight='balanced' predicted probabilities are inflated for minority classes and must not be read as real-world frequencies.",
    }
    log(f"accuracy={metrics['accuracy']:.4f}  macro_f1={metrics['macro_f1']:.4f}  "
        f"balanced_acc={metrics['balanced_accuracy']:.4f}  fatal_recall={metrics['fatal_recall']:.4f}")
    if errors["poor_fatal_performance_flag"]:
        log("NOTE: Fatal-class performance is poor (Fatal F1 < 0.30). This is reported, not hidden.")

    section("Explainability on a test subsample")
    n = min(cfg.permutation_sample, len(X))
    sub = np.random.default_rng(cfg.random_seed).choice(len(X), size=n, replace=False)
    perm = family_permutation_importance(pipe, X.iloc[sub].reset_index(drop=True), y[sub], cfg.permutation_repeats, cfg.random_seed)
    native = native_importance(pipe, fmeta["numeric_features"], fmeta["categorical_features"])
    importance = {
        "model": meta["model_name"], "computed_on": f"{n} random test rows", "scoring": "macro F1 drop (and Fatal recall drop)",
        "permutation_importance_by_family": perm, "native_importance": native,
        "caveat": "Importance reflects associations in historical reported collisions, not causes.",
    }
    write_json(cfg.reports_dir / "feature_importance.json", importance)
    final = {"model": meta["model_name"], "evaluated_at": now_iso(), "test_metrics": metrics, "error_analysis": errors,
             "test_set_used_for_nothing_else": True}
    write_json(cfg.reports_dir / "final_evaluation.json", final)
    write_json(cfg.reports_dir / "failure_analysis.json", {
        "model": meta["model_name"], "evaluated_at": final["evaluated_at"],
        "confusion_matrix": metrics.get("confusion_matrix"),
        "per_class": metrics.get("class_wise"),
        "error_analysis": errors,
        "note": ("Where the model is wrong on the untouched test set. Fatal-class performance "
                 "is reported as measured - poor numbers are flagged, never hidden.")})

    meta["test_metrics"] = {k: v for k, v in metrics.items()}
    meta["test_evaluated_at"] = final["evaluated_at"]
    write_json(art / "model_metadata.json", meta)
    mm_path = art / "model_metrics.json"
    mm = read_json(mm_path) if mm_path.exists() else {}
    mm["test"] = {"model": meta["model_name"], **metrics}
    write_json(mm_path, mm)
    (cfg.reports_dir / "model_report.md").write_text(_render_md(meta, metrics, errors, perm), encoding="utf-8")
    log("wrote reports/final_evaluation.json, reports/feature_importance.json, reports/model_report.md")
    return final


def _render_md(meta, m, errors, perm) -> str:
    L = [f"# Final model evaluation - {meta['model_name']}", "",
         f"Test rows: {m['n']:,}. Dataset period: {meta['dataset_period']}.", "",
         "| Metric | Value |", "|---|---:|"]
    for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "balanced_accuracy", "fatal_recall"):
        L.append(f"| {k} | {m[k]:.4f} |")
    L += ["", "| Class | Precision | Recall | F1 | Support |", "|---|---:|---:|---:|---:|"]
    for c, v in m["class_wise"].items():
        L.append(f"| {c} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | {v['support']:,} |")
    L += ["", f"Fatal collisions were predicted as: {errors['fatal_collisions_predicted_as']}.",
          f"Poor Fatal performance flag: {errors['poor_fatal_performance_flag']}.", "",
          "## Permutation importance (families, macro-F1 drop)", ""]
    L += [f"- {r['family']}: {r['macro_f1_drop_mean']:.4f}" for r in perm]
    return "\n".join(L) + "\n"
