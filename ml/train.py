"""Phases 6-8: leakage audit, balancing analysis, baseline + 3 models, tuning, selection, final fit.

The TEST split is dropped from memory immediately after loading and is never used here.
"""
from __future__ import annotations

import platform
import sys
from collections import Counter
from math import prod

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight

from . import PREPROCESSING_VERSION
from .config import Settings
from .evaluate import compute_metrics, ids_hash, time_inference
from .feature_config import (
    CATEGORICAL_RAW_INVALID, CLASS_NAMES, DAY_OF_WEEK_LABELS, ENGINEERED_FEATURES, FEATURE_FAMILIES,
    ID_COL, LABELS, SPEED_LIMIT_VALID, TARGET,
)
from .leakage import audit_features
from .preprocessing import build_preprocessor
from .utils import Timer, free_memory, load_frame, log, now_iso, read_json, section, write_json

MODEL_FILES = {
    "baseline": "baseline_model.joblib", "logistic_regression": "logistic_regression.joblib",
    "random_forest": "random_forest.joblib", "gradient_boosting": "gradient_boosting.joblib",
}
DISPLAY = {"baseline": "Baseline (most frequent class)", "logistic_regression": "Logistic Regression",
           "random_forest": "Random Forest", "gradient_boosting": "Gradient Boosting (HistGradientBoosting)"}

# Deliberately small, RAM/CPU-safe search spaces (RandomizedSearchCV samples n_iter of them).
SEARCH_SPACES = {
    "logistic_regression": {"clf__C": [0.01, 0.1, 1.0, 10.0]},
    "random_forest": {"clf__n_estimators": [60, 100, 150], "clf__max_depth": [10, 14, 18],
                      "clf__min_samples_leaf": [5, 10, 20], "clf__max_features": ["sqrt", 0.3]},
    "gradient_boosting": {"clf__learning_rate": [0.03, 0.06, 0.1], "clf__max_iter": [100, 200, 300],
                         "clf__max_leaf_nodes": [15, 31, 63], "clf__l2_regularization": [0.0, 0.1, 1.0],
                         "clf__min_samples_leaf": [20, 50, 100]},
}


def _smote_strategy(ratio: float):
    def f(y):
        counts = Counter(y)
        maj = max(counts.values())
        return {c: int(maj * ratio) for c, n in counts.items() if n < maj * ratio}
    return f


def build_pipeline(name: str, numeric, categorical, cfg: Settings):
    """prep -> [SMOTE] -> classifier. Resampling lives INSIDE the pipeline, so it only ever
    touches the rows passed to fit() (training folds) and never validation/test data."""
    cw = "balanced" if cfg.imbalance_strategy == "CLASS_WEIGHT" else None
    seed = cfg.random_seed
    if name == "baseline":
        clf = DummyClassifier(strategy="most_frequent")
    elif name == "logistic_regression":
        clf = LogisticRegression(max_iter=300, C=1.0, class_weight=cw, random_state=seed)
    elif name == "random_forest":
        clf = RandomForestClassifier(n_estimators=100, max_depth=14, min_samples_leaf=10, max_features="sqrt",
                                     class_weight=("balanced_subsample" if cw else None),
                                     n_jobs=cfg.n_jobs, random_state=seed)
    elif name == "gradient_boosting":
        clf = HistGradientBoostingClassifier(learning_rate=0.06, max_iter=200, max_leaf_nodes=31, class_weight=cw,
                                             random_state=seed)
    else:
        raise ValueError(name)
    prep = build_preprocessor(numeric, categorical, scale_numeric=(name == "logistic_regression"))
    if cfg.imbalance_strategy == "SMOTE" and name != "baseline":
        try:
            from imblearn.over_sampling import SMOTE
            from imblearn.pipeline import Pipeline as ImbPipeline
        except ImportError as e:
            raise RuntimeError("ROADRISK_IMBALANCE=SMOTE needs imbalanced-learn (pip install imbalanced-learn).") from e
        return ImbPipeline([("prep", prep), ("smote", SMOTE(sampling_strategy=_smote_strategy(cfg.smote_ratio),
                                                            random_state=seed)), ("clf", clf)])
    return Pipeline([("prep", prep), ("clf", clf)])


def _stratified_subsample(X, y, n, seed):
    if len(X) <= n:
        return X, y
    Xs, _, ys, _ = train_test_split(X, y, train_size=n, stratify=y, random_state=seed)
    return Xs, ys


def _tune(name, pipe, X, y, cfg):
    space = SEARCH_SPACES[name]
    size = prod(len(v) for v in space.values())
    n_iter = min(cfg.search_n_iter, size)
    Xs, ys = _stratified_subsample(X, y, cfg.tuning_max_rows, cfg.random_seed)
    cv = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.random_seed)
    search = RandomizedSearchCV(pipe, space, n_iter=n_iter, scoring="f1_macro", cv=cv, n_jobs=cfg.n_jobs,
                                random_state=cfg.random_seed, refit=False, error_score="raise")
    with Timer() as t:
        search.fit(Xs, ys)
    res = search.cv_results_
    cands = [{"params": {k.replace("clf__", ""): v for k, v in p.items()}, "mean_cv_macro_f1": float(m), "std_cv_macro_f1": float(s)}
             for p, m, s in zip(res["params"], res["mean_test_score"], res["std_test_score"])]
    info = {
        "method": "RandomizedSearchCV", "scoring": "f1_macro", "cv_folds": cfg.cv_folds, "n_jobs": cfg.n_jobs,
        "parameter_distributions": {k.replace("clf__", ""): v for k, v in space.items()},
        "search_space_size": size, "n_candidates_evaluated": n_iter,
        "tuning_rows_used": int(len(Xs)), "tuning_rows_available": int(len(X)),
        "tuning_subsample_note": ("tuning ran on a stratified subsample of the training split to bound runtime/RAM; "
                                  "the selected parameters are then refit on the FULL training split") if len(Xs) < len(X) else "full training split used",
        "best_params": {k.replace("clf__", ""): v for k, v in search.best_params_.items()},
        "best_cv_macro_f1": float(search.best_score_), "search_seconds": t.seconds, "candidates": cands,
    }
    return search.best_params_, info


def _balancing_report(y_train, y_val, cfg, X_train, X_val, numeric, categorical) -> dict:
    counts = np.bincount(y_train, minlength=3)
    cw = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=y_train)
    rep = {
        "strategy_selected": cfg.imbalance_strategy,
        "train_class_counts": dict(zip(CLASS_NAMES, map(int, counts))),
        "train_class_percentages": dict(zip(CLASS_NAMES, (100 * counts / counts.sum()).round(3).tolist())),
        "imbalance_ratio_majority_to_minority": round(float(counts.max() / max(counts.min(), 1)), 2),
        "balanced_class_weights_would_be": dict(zip(CLASS_NAMES, cw.round(3).tolist())),
        "resampling_applied_to": "training data only (inside the sklearn/imblearn Pipeline); validation and test are never resampled",
        "smote_ratio": cfg.smote_ratio if cfg.imbalance_strategy == "SMOTE" else None,
        "smote_caveat": "SMOTE interpolates one-hot/binary columns, producing non-physical fractional values; class weighting is the default for that reason." ,
    }
    if cfg.run_balancing_comparison:
        log("Balancing comparison (logistic regression, tuning subsample): NONE vs CLASS_WEIGHT")
        Xs, ys = _stratified_subsample(X_train, y_train, min(cfg.tuning_max_rows, 50_000), cfg.random_seed)
        comp = {}
        for label, cwv in (("NONE", None), ("CLASS_WEIGHT", "balanced")):
            p = Pipeline([("prep", build_preprocessor(numeric, categorical, True)),
                          ("clf", LogisticRegression(max_iter=300, class_weight=cwv, random_state=cfg.random_seed))])
            p.fit(Xs, ys)
            m = compute_metrics(y_val, p.predict(X_val))
            comp[label] = {k: m[k] for k in ("accuracy", "macro_f1", "balanced_accuracy", "fatal_recall", "fatal_precision")}
        rep["comparison_logistic_regression_validation"] = comp
        rep["comparison_rows_used_for_fit"] = int(len(Xs))
        rep["comparison_note"] = "Same model, same data; only class weighting differs. Evaluated on the validation split."
    return rep


def select_model(cands: dict) -> tuple[str, dict]:
    """Selection is NOT 'highest accuracy'.

    1. Only real models (not the baseline) that beat the baseline's macro-F1 are eligible
       (if none do, all are considered and this is reported).
    2. score = 0.6 * macro-F1 + 0.4 * balanced accuracy (both punish ignoring minority classes).
    3. Candidates within 0.01 of the best score are treated as tied; the tie is broken in favour
       of the lower inference time, then the smaller file (cheaper to deploy), then the more
       interpretable model.
    4. Fatal-class recall / F1 is reported for every candidate and flagged when poor.
    """
    base_f1 = cands["baseline"]["validation"]["macro_f1"]
    real = {k: v for k, v in cands.items() if k != "baseline"}
    eligible = {k: v for k, v in real.items() if v["validation"]["macro_f1"] > base_f1}
    note_elig = None
    if not eligible:
        eligible, note_elig = real, "No model beat the baseline macro-F1; selecting among all candidates anyway."
    for k, v in eligible.items():
        val = v["validation"]
        v["selection_score"] = 0.6 * val["macro_f1"] + 0.4 * val["balanced_accuracy"]
    best = max(v["selection_score"] for v in eligible.values())
    tied = {k: v for k, v in eligible.items() if best - v["selection_score"] <= 0.01}
    interp = {"logistic_regression": 0, "random_forest": 1, "gradient_boosting": 2}
    winner = sorted(tied, key=lambda k: (tied[k]["inference_ms_per_1000_rows"], tied[k]["model_size_mb"], interp[k]))[0]
    top_by_score = max(eligible, key=lambda k: eligible[k]["selection_score"])
    rationale = {
        "rule": "score = 0.6*macro_F1 + 0.4*balanced_accuracy on VALIDATION; ties within 0.01 -> faster inference, then smaller model, then more interpretable",
        "baseline_macro_f1": base_f1, "eligibility_note": note_elig,
        "scores": {k: v["selection_score"] for k, v in eligible.items()},
        "tied_within_0.01": sorted(tied), "highest_score_model": top_by_score, "selected": winner,
        "highest_accuracy_model_for_reference": max(real, key=lambda k: real[k]["validation"]["accuracy"]),
        "fatal_class_warnings": {k: (f"Fatal recall {v['validation']['fatal_recall']:.3f}, F1 {v['validation']['fatal_f1']:.3f} "
                                     f"- {'POOR' if v['validation']['fatal_f1'] < 0.30 else 'moderate'}") for k, v in real.items()},
    }
    return winner, rationale


def _feature_metadata(train: pd.DataFrame, numeric, categorical, cfg) -> dict:
    ref = {c: train[c].mode().iloc[0] for c in categorical}
    ref.update({c: float(train[c].median()) for c in numeric})
    opts = {}
    for c in CATEGORICAL_RAW_INVALID:
        if c in categorical:
            vc = train[c].value_counts()
            opts[c] = [{"value": v, "label": LABELS.get(c, {}).get(v, v), "count": int(n)} for v, n in vc.items() if v != "unknown"]
    speeds = sorted(int(v) for v in train["speed_limit"].dropna().unique()) if "speed_limit" in train else []
    spec = []
    def add(name, typ, required, **kw):
        spec.append({"name": name, "type": typ, "required": required, **kw})
    if "speed_limit" in numeric: add("speed_limit", "int", True, options=speeds)
    for c in ("road_type", "junction_detail", "light_conditions", "weather_conditions", "road_surface_conditions", "urban_or_rural_area"):
        if c in opts: add(c, "code", True, options=opts[c])
    if "number_of_vehicles" in numeric: add("number_of_vehicles", "int", True, min=1, max=cfg.num_vehicles_cap, aliases=["vehicle_count"])
    if "hour" in numeric: add("hour", "int", True, min=0, max=23)
    if "weekday" in numeric: add("day_of_week", "int", True, min=1, max=7, note="STATS19 coding: 1=Sunday ... 7=Saturday",
                                 labels=DAY_OF_WEEK_LABELS)
    if "month" in numeric: add("month", "int", False, min=1, max=12)
    for c in ("junction_control", "first_road_class", "trunk_road_flag", "special_conditions_at_site", "carriageway_hazards",
              "pedestrian_crossing_human_control", "pedestrian_crossing_physical_facilities"):
        if c in opts: add(c, "code", False, options=opts[c])
    return {
        "numeric_features": numeric, "categorical_features": categorical, "features": numeric + categorical,
        "target": TARGET, "class_names": CLASS_NAMES, "reference_values": ref, "value_options": opts,
        "input_fields": spec, "feature_families": {k: [c for c in v if c in numeric + categorical] for k, v in FEATURE_FAMILIES.items()},
        "engineered_feature_reasons": {k: v for k, v in ENGINEERED_FEATURES.items() if k in numeric + categorical},
        "labels": LABELS, "num_vehicles_cap": cfg.num_vehicles_cap,
    }


def run_training(cfg: Settings) -> dict:
    cfg.ensure_dirs()
    section("TRAIN MODELS")
    if not (cfg.processed_base.with_suffix(".parquet").exists() or cfg.processed_base.with_suffix(".pkl").exists()):
        raise FileNotFoundError("No processed data. Run scripts/prepare_data.py first.")
    pmeta = read_json(cfg.processed_meta_path)
    df = load_frame(cfg.processed_base)
    numeric, categorical = pmeta["numeric_features"], pmeta["categorical_features"]
    features = numeric + categorical

    section("Leakage audit")
    split = df["split"].to_numpy()
    audit_features(features, df, "target", train_mask=(split == "train"), cfg=cfg,
                   loaded_columns=[c for c in pmeta["load_report"]["files"][0]["columns_loaded"]])
    log(f"leakage audit PASSED for {len(features)} features -> reports/leakage_report.json")

    test_ids_hash = ids_hash(df.loc[split == "test", ID_COL])
    n_test = int((split == "test").sum())
    df = df[df["split"] != "test"].reset_index(drop=True)      # test rows leave memory here
    free_memory()
    tr, va = df[df["split"] == "train"], df[df["split"] == "val"]
    X_tr, y_tr = tr[features], tr["target"].to_numpy()
    X_va, y_va = va[features], va["target"].to_numpy()
    if set(np.unique(y_tr)) != {0, 1, 2}:
        raise RuntimeError("Training split does not contain all three classes.")
    log(f"train rows {len(X_tr):,}  validation rows {len(X_va):,}  (test {n_test:,} set aside, untouched)")

    section("Class imbalance analysis")
    bal = _balancing_report(y_tr, y_va, cfg, X_tr, X_va, numeric, categorical)
    write_json(cfg.reports_dir / "balancing_report.json", bal)
    log(f"train class counts {bal['train_class_counts']}  strategy={cfg.imbalance_strategy}")
    if "comparison_logistic_regression_validation" in bal:
        log(f"  comparison: {bal['comparison_logistic_regression_validation']}")

    cands, tuned_params = {}, {}
    for name in ("baseline", "logistic_regression", "random_forest", "gradient_boosting"):
        section(f"Model: {DISPLAY[name]}")
        pipe = build_pipeline(name, numeric, categorical, cfg)
        search_info = None
        if name != "baseline":
            log(f"tuning on up to {cfg.tuning_max_rows:,} rows, {cfg.search_n_iter} candidates x {cfg.cv_folds}-fold CV (n_jobs={cfg.n_jobs}) ...")
            best, search_info = _tune(name, pipe, X_tr, y_tr, cfg)
            pipe.set_params(**best)
            tuned_params[name] = search_info["best_params"]
            log(f"  best params {search_info['best_params']}  cv macro-F1 {search_info['best_cv_macro_f1']:.4f}  ({search_info['search_seconds']:.0f}s)")
        log("fitting on the full training split ...")
        with Timer() as t:
            pipe.fit(X_tr, y_tr)
        pred, proba = pipe.predict(X_va), pipe.predict_proba(X_va)
        val = compute_metrics(y_va, pred, proba)
        path = cfg.artifacts_dir / MODEL_FILES[name]
        joblib.dump(pipe, path, compress=3)
        cands[name] = {
            "display_name": DISPLAY[name], "training_seconds": t.seconds, "validation": val,
            "inference_ms_per_1000_rows": time_inference(pipe, X_va), "model_size_mb": round(path.stat().st_size / 1e6, 3),
            "hyperparameter_search": search_info, "artifact": str(path.relative_to(cfg.root)),
            "params": {k: v for k, v in pipe.get_params().items() if k.startswith("clf__") and isinstance(v, (int, float, str, bool, type(None)))},
        }
        log(f"  val: acc={val['accuracy']:.4f} macroF1={val['macro_f1']:.4f} balAcc={val['balanced_accuracy']:.4f} "
            f"fatalRecall={val['fatal_recall']:.4f}  train={t.seconds:.0f}s  size={cands[name]['model_size_mb']}MB")
        del pipe, pred, proba
        free_memory()

    section("Model selection")
    winner, rationale = select_model(cands)
    log(f"selected: {DISPLAY[winner]}")
    for k, v in rationale["fatal_class_warnings"].items():
        log(f"  Fatal class [{k}]: {v}")
    write_json(cfg.reports_dir / "model_comparison.json", {
        "created_at": now_iso(), "models": cands, "selection": rationale,
        "imbalance_strategy": cfg.imbalance_strategy, "settings": cfg.snapshot(),
        "note": "All metrics are VALIDATION metrics. The test set is evaluated once, for the final model only (evaluate_models.py)."})
    write_json(cfg.reports_dir / "hyperparameter_results.json", {
        "created_at": now_iso(),
        "search_method": f"RandomizedSearchCV(cv={cfg.cv_folds}, scoring=f1_macro, n_iter={cfg.search_n_iter}, n_jobs={cfg.n_jobs})",
        "rows_used": min(cfg.tuning_max_rows, len(X_tr)),
        "models": {k: {"best_params": tuned_params.get(k),
                        "search": cands[k].get("hyperparameter_search")}
                   for k, v in cands.items() if k != "baseline"},
        "baseline": "DummyClassifier (most frequent) - no hyperparameters.",
        "note": "CV metrics are mean validation-fold macro F1 on a stratified subsample of the training split."})
    write_json(cfg.reports_dir / "model_selection.json", {
        "created_at": now_iso(), "selected": winner, "display_name": DISPLAY[winner],
        "selection_rationale": rationale,
        "validation_metrics_all_models": {k: v["validation"] for k, v in cands.items()},
        "selection_score_rule": rationale["rule"],
        "imbalance_strategy": cfg.imbalance_strategy,
        "note": "Selection used VALIDATION metrics only; the test set stayed untouched."})

    section("Final model fit")
    final = build_pipeline(winner, numeric, categorical, cfg)
    final.set_params(**{f"clf__{k}": v for k, v in tuned_params[winner].items()})
    if cfg.refit_on_train_val:
        X_fit, y_fit, fit_note = pd.concat([X_tr, X_va], ignore_index=True), np.concatenate([y_tr, y_va]), "train+validation"
    else:
        X_fit, y_fit, fit_note = X_tr, y_tr, "train only"
    log(f"refitting {DISPLAY[winner]} on {fit_note}: {len(X_fit):,} rows")
    with Timer() as t:
        final.fit(X_fit, y_fit)
    joblib.dump(final, cfg.artifacts_dir / "final_model.joblib", compress=3)
    joblib.dump(final.named_steps["prep"], cfg.artifacts_dir / "preprocessing_pipeline.joblib", compress=3)

    fmeta = _feature_metadata(tr, numeric, categorical, cfg)
    write_json(cfg.artifacts_dir / "feature_metadata.json", fmeta)
    period = f"{min(pmeta['years_used'])}-{max(pmeta['years_used'])}"
    meta = {
        "model_name": DISPLAY[winner], "model_key": winner, "training_date": now_iso(),
        "dataset_period": period, "years_used": pmeta["years_used"], "years_requested": pmeta["years_requested"],
        "years_missing": pmeta["years_missing"], "features": features, "n_features_raw": len(features),
        "n_features_after_encoding": int(len(final.named_steps["prep"].get_feature_names_out())),
        "target": TARGET, "class_names": CLASS_NAMES,
        "training_rows": int(len(X_fit)), "training_rows_composition": fit_note,
        "validation_rows": int(len(X_va)), "test_rows": n_test, "test_ids_sha256": test_ids_hash,
        "validation_metrics_of_selected_model_before_refit": cands[winner]["validation"],
        "random_seed": cfg.random_seed, "best_hyperparameters": tuned_params[winner],
        "preprocessing_version": PREPROCESSING_VERSION, "imbalance_strategy": cfg.imbalance_strategy,
        "split_strategy": cfg.split_strategy, "final_fit_seconds": t.seconds,
        "selection_rationale": rationale,
        "probability_note": ("Probabilities come from the trained model but are NOT calibrated, especially with "
                             "class weighting; read them as relative model scores, not real-world frequencies."),
        "environment": {"python": sys.version.split()[0], "sklearn": sklearn.__version__, "pandas": pd.__version__,
                        "numpy": np.__version__, "platform": platform.platform()},
        "settings": cfg.snapshot(),
    }
    write_json(cfg.artifacts_dir / "model_metadata.json", meta)
    write_json(cfg.artifacts_dir / "model_metrics.json", {
        "validation": {k: {"display_name": v["display_name"], "training_seconds": v["training_seconds"], **v["validation"]}
                       for k, v in cands.items()}, "test": None,
        "note": "'test' is filled by evaluate_models.py for the final model only."})
    log("TRAINING COMPLETE. Artifacts in ml/artifacts/. Next: python scripts/evaluate_models.py")
    return {"selected": winner, "metadata": meta, "comparison": cands}
