import pandas as pd
import pytest

from ml.feature_config import ALLOWED_FEATURES, EXCLUDED_FEATURES, FEATURE_FAMILIES, MODEL_CATEGORICAL, MODEL_NUMERIC
from ml.leakage import LeakageError, audit_features


def test_real_feature_list_passes():
    rep = audit_features(ALLOWED_FEATURES, write=False)
    assert rep["passed"] and rep["checks"]["no_excluded_feature_used"]


@pytest.mark.parametrize("bad", ["number_of_casualties", "casualty_severity", "accident_severity",
                                 "did_police_officer_attend_scene_of_accident", "latitude", "lsoa_of_accident_location"])
def test_excluded_or_leaky_feature_is_caught(bad):
    with pytest.raises(LeakageError):
        audit_features(ALLOWED_FEATURES + [bad], write=False)


def test_unlisted_feature_blocked_by_whitelist():
    with pytest.raises(LeakageError, match="whitelist"):
        audit_features(["speed_limit", "some_new_column"], write=False)


def test_target_derived_feature_detected_by_association():
    sev = ["Slight"] * 800 + ["Serious"] * 150 + ["Fatal"] * 50
    df = pd.DataFrame({"speed_band": sev, "severity": sev, "target": [0] * 800 + [1] * 150 + [2] * 50})
    with pytest.raises(LeakageError, match="Cramer"):
        audit_features(["speed_band"], df, y_col="severity", write=False)


def test_duplicate_ids_across_rows_detected():
    df = pd.DataFrame({"collision_index": ["a", "a", "b"], "speed_band": ["30", "30", "60"], "severity": ["Slight", "Slight", "Fatal"]})
    with pytest.raises(LeakageError, match="more than once"):
        audit_features(["speed_band"], df, y_col="severity", write=False)


def test_every_excluded_feature_has_a_reason_and_families_cover_features_once():
    assert all(isinstance(v, str) and len(v) > 10 for v in EXCLUDED_FEATURES.values())
    cols = [c for v in FEATURE_FAMILIES.values() for c in v]
    assert sorted(cols) == sorted(MODEL_NUMERIC + MODEL_CATEGORICAL)
