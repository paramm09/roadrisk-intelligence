import numpy as np
import pytest

from ml.predict import ArtifactsNotFoundError, Predictor

VALID = {"speed_limit": 60, "road_type": 6, "junction_detail": 3, "light_conditions": 4, "weather_conditions": 2,
         "road_surface_conditions": 2, "urban_or_rural_area": 2, "vehicle_count": 2, "hour": 21, "day_of_week": 5}


@pytest.fixture(scope="module")
def predictor(env):
    return Predictor(env)


def test_model_loads_and_info(predictor):
    info = predictor.info()
    assert info["model_name"] and info["features"] and info["class_names"] == ["Slight", "Serious", "Fatal"]


def test_prediction_shape_and_probabilities(predictor):
    r = predictor.predict(VALID)
    assert r["prediction"] in {"Slight", "Serious", "Fatal"}
    assert list(r["probabilities"]) == ["Slight", "Serious", "Fatal"]
    assert abs(sum(r["probabilities"].values()) - 1) < 1e-6
    assert r["confidence"] == max(r["probabilities"].values())
    assert 1 <= len(r["top_factors"]) <= 5 and all("factor" in f for f in r["top_factors"])
    assert "not be treated as a definitive" in r["disclaimer"]


def test_probabilities_come_from_the_trained_model(predictor):
    r = predictor.predict(VALID)
    X = predictor._frame([predictor.validate(VALID)])
    direct = predictor.pipeline.predict_proba(X)[0]
    assert np.allclose([r["probabilities"][c] for c in ("Slight", "Serious", "Fatal")], direct)


def test_deterministic_and_batch(predictor):
    a, b = predictor.predict(VALID), predictor.predict(VALID)
    assert a["probabilities"] == b["probabilities"]
    other = dict(VALID, speed_limit=30, light_conditions=1, hour=9)
    out = predictor.predict_batch([VALID, other])
    assert len(out) == 2 and out[0]["probabilities"] != out[1]["probabilities"]


@pytest.mark.parametrize("patch", [{"hour": 25}, {"day_of_week": 0}, {"speed_limit": 65}, {"road_type": 777},
                                   {"vehicle_count": 0}, {"hour": "abc"}, {"bogus_field": 1}])
def test_invalid_input_rejected(predictor, patch):
    with pytest.raises(ValueError):
        predictor.predict({**VALID, **patch})


def test_missing_required_field_rejected(predictor):
    bad = dict(VALID)
    del bad["weather_conditions"]
    with pytest.raises(ValueError, match="required"):
        predictor.predict(bad)


def test_missing_artifacts_raise_clear_error(tmp_path):
    with pytest.raises(ArtifactsNotFoundError, match="train_models"):
        Predictor(artifacts_dir=tmp_path)
