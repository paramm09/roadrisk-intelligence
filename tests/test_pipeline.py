"""Integration: prepare -> train -> evaluate on the SYNTHETIC fixture. Order matters: the 'test set
untouched' check runs before evaluation. Metric VALUES here are meaningless (fake data)."""
import json

from ml.evaluate import run_final_evaluation
from ml.utils import load_frame, read_json


def test_reports_and_artifacts_written(env):
    for f in ("data_report.json", "data_report.md", "target_distribution.json", "preprocessing_report.json",
              "leakage_report.json", "balancing_report.json", "model_comparison.json"):
        assert (env.reports_dir / f).exists(), f
    for f in ("final_model.joblib", "preprocessing_pipeline.joblib", "model_metadata.json", "feature_metadata.json",
              "baseline_model.joblib", "logistic_regression.joblib", "random_forest.joblib", "gradient_boosting.joblib"):
        assert (env.artifacts_dir / f).exists(), f
    assert read_json(env.reports_dir / "leakage_report.json")["passed"] is True


def test_four_models_compared_with_tuning_recorded(env):
    comp = read_json(env.reports_dir / "model_comparison.json")
    assert set(comp["models"]) == {"baseline", "logistic_regression", "random_forest", "gradient_boosting"}
    for k, m in comp["models"].items():
        assert m["training_seconds"] >= 0 and "macro_f1" in m["validation"] and len(m["validation"]["confusion_matrix"]) == 3
        if k != "baseline":
            s = m["hyperparameter_search"]
            assert s["method"] == "RandomizedSearchCV" and s["cv_folds"] == 2 and s["best_params"] and s["parameter_distributions"]
    assert comp["selection"]["selected"] in {"logistic_regression", "random_forest", "gradient_boosting"}
    assert "fatal_class_warnings" in comp["selection"]


def test_balancing_report_has_comparison_and_never_resamples_eval(env):
    b = read_json(env.reports_dir / "balancing_report.json")
    assert b["strategy_selected"] == "CLASS_WEIGHT" and "validation and test are never resampled" in b["resampling_applied_to"]
    assert {"NONE", "CLASS_WEIGHT"} == set(b["comparison_logistic_regression_validation"])


def test_split_is_disjoint_and_test_untouched_by_training(env):
    df = load_frame(env.processed_base)
    assert not df["collision_index"].duplicated().any()
    meta = read_json(env.artifacts_dir / "model_metadata.json")
    n = df["split"].value_counts()
    assert meta["test_rows"] == n["test"] and meta["validation_rows"] == n["val"]
    assert meta["training_rows"] == n["train"] + n["val"]          # refit on train+val, never test
    assert "test_metrics" not in meta                              # nothing evaluated on test yet
    assert read_json(env.artifacts_dir / "model_metrics.json")["test"] is None


def test_metadata_fields(env):
    m = read_json(env.artifacts_dir / "model_metadata.json")
    for k in ("model_name", "training_date", "dataset_period", "features", "target", "class_names", "training_rows",
              "test_rows", "random_seed", "best_hyperparameters", "preprocessing_version", "imbalance_strategy"):
        assert k in m, k
    assert m["random_seed"] == 42 and m["class_names"] == ["Slight", "Serious", "Fatal"] and m["dataset_period"] == "2099-2099"
    assert "number_of_casualties" not in m["features"] and "latitude" not in m["features"]


def test_zz_final_evaluation_runs_once_on_test_set(env):
    final = run_final_evaluation(env)
    meta = read_json(env.artifacts_dir / "model_metadata.json")
    assert final["test_metrics"]["n"] == meta["test_rows"] == meta["test_metrics"]["n"]
    assert (env.reports_dir / "final_evaluation.json").exists() and (env.reports_dir / "feature_importance.json").exists()
    fi = json.loads((env.reports_dir / "feature_importance.json").read_text())
    assert fi["permutation_importance_by_family"] and fi["caveat"]
    assert "probability_calibration" in final["error_analysis"] and "segment_performance" in final["error_analysis"]
