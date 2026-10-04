"""Explainability without SHAP (low RAM): family-level permutation importance, native
importances / coefficients, and per-prediction occlusion factors.

IMPORTANT: all of these describe ASSOCIATIONS the model learned from historical reported
collisions. They do not prove causation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score

from .feature_config import FATAL_INT, FEATURE_FAMILIES


def active_families(columns) -> dict:
    cols = set(columns)
    fams = {k: [c for c in v if c in cols] for k, v in FEATURE_FAMILIES.items()}
    return {k: v for k, v in fams.items() if v}


def family_permutation_importance(pipeline, X: pd.DataFrame, y: np.ndarray, n_repeats: int = 3, seed: int = 42) -> list[dict]:
    """Drop in macro-F1 / Fatal-recall when a whole feature family is shuffled."""
    rng = np.random.default_rng(seed)
    fams = active_families(X.columns)
    pred = pipeline.predict(X)
    base_f1 = f1_score(y, pred, average="macro", labels=[0, 1, 2], zero_division=0)
    base_rec = recall_score(y == FATAL_INT, pred == FATAL_INT, zero_division=0)
    rows = []
    for fam, cols in fams.items():
        d_f1, d_rec = [], []
        for _ in range(n_repeats):
            idx = rng.permutation(len(X))
            Xp = X.copy()
            for c in cols:
                Xp[c] = X[c].to_numpy()[idx]
            p = pipeline.predict(Xp)
            d_f1.append(base_f1 - f1_score(y, p, average="macro", labels=[0, 1, 2], zero_division=0))
            d_rec.append(base_rec - recall_score(y == FATAL_INT, p == FATAL_INT, zero_division=0))
        rows.append({"family": fam, "columns": cols,
                     "macro_f1_drop_mean": float(np.mean(d_f1)), "macro_f1_drop_std": float(np.std(d_f1)),
                     "fatal_recall_drop_mean": float(np.mean(d_rec))})
    rows.sort(key=lambda r: -r["macro_f1_drop_mean"])
    return rows


def _origin_column(transformed_name: str, numeric: list[str], categorical: list[str]) -> str:
    n = transformed_name.split("__", 1)[1] if "__" in transformed_name else transformed_name
    if n.startswith("missingindicator_"):
        return n[len("missingindicator_"):]
    best = ""
    for c in numeric + categorical:
        if (n == c or n.startswith(c + "_")) and len(c) > len(best):
            best = c
    return best or n


def native_importance(pipeline, numeric: list[str], categorical: list[str], top: int = 15) -> dict:
    prep, clf = pipeline.named_steps["prep"], pipeline.named_steps["clf"]
    names = list(prep.get_feature_names_out())
    out = {"model_type": type(clf).__name__}
    if hasattr(clf, "feature_importances_"):
        imp = np.asarray(clf.feature_importances_)
        agg = {}
        for n, v in zip(names, imp):
            o = _origin_column(n, numeric, categorical)
            agg[o] = agg.get(o, 0.0) + float(v)
        out["impurity_importance_by_column"] = dict(sorted(agg.items(), key=lambda kv: -kv[1])[:top * 2])
        out["note"] = "Mean decrease in impurity; biased toward high-cardinality features. Prefer permutation importance."
    elif hasattr(clf, "coef_"):
        coef = np.asarray(clf.coef_)
        out["coefficients_note"] = "Standardised numeric inputs; coefficients are per transformed feature, log-odds scale."
        out["coefficients_by_class"] = {}
        for i, cls in enumerate(clf.classes_):
            order = np.argsort(-np.abs(coef[i]))[:top]
            out["coefficients_by_class"][str(int(cls))] = [{"feature": names[j], "coef": float(coef[i, j])} for j in order]
    else:
        out["note"] = "No native importance available for this estimator; see permutation importance."
    return out


def local_factors(pipeline, row: pd.DataFrame, reference: dict, class_int: int, top_n: int = 5) -> list[dict]:
    """Occlusion: replace each feature family by its typical (training mode/median) value and
    measure how much the predicted-class probability falls. Positive effect = this factor's actual
    value pushed the probability for the predicted class UP relative to typical conditions."""
    classes = list(pipeline.classes_)
    ci = classes.index(class_int)
    base = float(pipeline.predict_proba(row)[0, ci])
    fams = active_families(row.columns)
    variants, names = [], []
    for fam, cols in fams.items():
        v = row.copy()
        for c in cols:
            if c in reference:
                v[c] = reference[c]
        variants.append(v)
        names.append(fam)
    probs = pipeline.predict_proba(pd.concat(variants, ignore_index=True))[:, ci]
    effects = base - probs
    order = np.argsort(-np.abs(effects))[:top_n]
    return [{"factor": names[i],
             "input": {c: (None if pd.isna(row[c].iloc[0]) else (row[c].iloc[0].item() if hasattr(row[c].iloc[0], "item") else row[c].iloc[0]))
                       for c in fams[names[i]]},
             "effect_on_predicted_class_probability": float(effects[i]),
             "direction": "raises" if effects[i] > 0 else "lowers"} for i in order]
