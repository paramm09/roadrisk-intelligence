"""API tests: health, model info/metrics, analytics, prediction (valid + invalid).

Uses the SAME synthetic, test-only fixture as the rest of the suite (see
tests/conftest.py): it trains a real pipeline in a temp folder, so the API
tests exercise the genuine code path (artifacts -> Predictor -> HTTP) without
ever touching production data or fabricating results. The temp folder is
outside the production paths, so nothing here can reach a real deployment.
"""
import pytest
from fastapi.testclient import TestClient

from backend.config import BackendConfig
from backend.main import create_app

VALID = {"speed_limit": 60, "road_type": 6, "junction_detail": 3, "light_conditions": 4,
         "weather_conditions": 2, "road_surface_conditions": 2, "urban_or_rural_area": 2,
         "number_of_vehicles": 2, "hour": 21, "day_of_week": 5}


@pytest.fixture(scope="module")
def client(env):
    """Full app wired to the temp training environment (real artifact + processed data)."""
    base = env.processed_base
    processed = base.with_suffix(".parquet") if base.with_suffix(".parquet").exists() else base.with_suffix(".pkl")
    cfg = BackendConfig(root=env.root, artifacts_override=env.artifacts_dir,
                        processed_override=processed, cors_origins=("http://test",))
    app = create_app(cfg)
    with TestClient(app) as c:
        yield c


# ------------------------------------------------------------------ health / system
def test_health_reports_loaded_model_and_data(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["analytics_loaded"] is True
    assert body["model_error"] is None and body["analytics_error"] is None


def test_health_without_artifacts_is_honest(tmp_path):
    cfg = BackendConfig(root=tmp_path, artifacts_override=tmp_path / "nope",
                        processed_override=tmp_path / "missing.parquet")
    with TestClient(create_app(cfg)) as c:
        body = c.get("/health").json()
        assert body["model_loaded"] is False and body["analytics_loaded"] is False
        assert "train_models" in (body["model_error"] or "")
        assert c.post("/api/predict", json=VALID).status_code == 503
        assert c.get("/api/analytics/summary").status_code == 503


# ------------------------------------------------------------------ model endpoints
def test_model_info(client):
    body = client.get("/api/model/info").json()
    assert body["model_name"] and body["class_names"] == ["Slight", "Serious", "Fatal"]
    assert body["features"] and body["target"]
    names = [f["name"] for f in body["input_fields"]]
    assert "speed_limit" in names and "number_of_vehicles" in names
    # leakage guard: no post-outcome field may be an accepted input
    for forbidden in ("number_of_casualties", "casualty_severity", "collision_severity",
                      "accident_severity", "did_police_officer_attend_scene_of_accident"):
        assert forbidden not in names


def test_model_metrics(client):
    body = client.get("/api/model/metrics").json()
    assert body["validation"], "validation metrics must exist after training"
    for model_key in ("baseline", "logistic_regression", "random_forest", "gradient_boosting"):
        m = body["validation"][model_key]
        for metric in ("accuracy", "macro_f1", "balanced_accuracy", "fatal_recall"):
            assert 0.0 <= m[metric] <= 1.0
    assert body["selection"]["selected"] in body["validation"]


def test_models_endpoint_lists_real_artifacts(client):
    r = client.get("/api/models")
    assert r.status_code == 200, r.text
    body = r.json()
    models = body["models"]
    assert models, "GET /api/models must list the trained models"
    assert body["default_model"] in {m["id"] for m in models}
    artifacts_dir = client.app.state.model_service.cfg.artifacts_dir
    for m in models:
        assert m["available"] is True
        assert (artifacts_dir / m["artifact"]).exists(), f"missing artifact for {m['id']}"
        val = m["validation_metrics"] or {}
        for metric in ("macro_f1", "balanced_accuracy"):
            assert 0.0 <= val[metric] <= 1.0, f"{m['id']} missing real {metric}"
        # test metrics may be null (evaluation not run), but when present they must be real
        if m["test_metrics"]:
            assert 0.0 <= m["test_metrics"]["macro_f1"] <= 1.0
    defaults = [m for m in models if m["is_default"]]
    assert len(defaults) == 1 and defaults[0]["id"] == body["default_model"]
    assert defaults[0]["artifact"] == "final_model.joblib"


def test_predict_with_selected_model_uses_that_artifact(client):
    models = client.get("/api/models").json()["models"]
    entry = next((m for m in models if m["artifact"] != "final_model.joblib"), None)
    if entry is None:
        pytest.skip("only the final artifact is present in this environment")
    r = client.post("/api/predict", json={**VALID, "model": entry["id"]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == entry["name"]
    # the probabilities must come from THAT artifact's pipeline, not the default one
    svc = client.app.state.prediction_service
    pred = svc.model.predictor
    pipe = pred.pipeline_for(entry["artifact"])
    assert pipe is not pred.pipeline
    direct = pipe.predict_proba(pred._frame([pred.validate(VALID)]))[0]
    assert [body["probabilities"][c] for c in ("Slight", "Serious", "Fatal")] == pytest.approx(list(direct))


def test_predict_auto_still_works(client):
    for payload in (VALID, {**VALID, "model": "auto"}):
        r = client.post("/api/predict", json=payload)
        assert r.status_code == 200, r.text
        assert r.json()["model"]  # name of the model actually used


def test_predict_unknown_model_is_400_not_a_silent_fallback(client):
    r = client.post("/api/predict", json={**VALID, "model": "does_not_exist"})
    assert r.status_code == 400, r.text
    detail = r.json()["detail"].lower()
    assert "not available" in detail and "does_not_exist" in detail
    # batch fails the same way
    r = client.post("/api/predict/batch", json={"records": [VALID, {**VALID, "model": "does_not_exist"}]})
    assert r.status_code == 400, r.text


def test_feature_importance_endpoint(client, env):
    r = client.get("/api/model/feature-importance")
    if r.status_code == 404:
        # standalone run (test_pipeline not executed yet): perform the one real evaluation
        from ml.evaluate import run_final_evaluation
        run_final_evaluation(env)
        r = client.get("/api/model/feature-importance")
    assert r.status_code == 200, r.text
    body = r.json()
    assert "permutation_importance_by_family" in body or "native_importance" in body


# ------------------------------------------------------------------ analytics
@pytest.mark.parametrize("endpoint", [
    "/api/analytics/summary", "/api/analytics/severity-distribution", "/api/analytics/by-year",
    "/api/analytics/by-road-type", "/api/analytics/by-weather", "/api/analytics/by-lighting",
    "/api/analytics/by-speed-limit", "/api/analytics/by-hour", "/api/analytics/by-weekday",
    "/api/analytics/by-month", "/api/analytics/by-urban-rural", "/api/analytics/by-junction",
    "/api/analytics/map-points", "/api/insights",
])
def test_analytics_endpoints(client, endpoint):
    r = client.get(endpoint)
    assert r.status_code == 200, f"{endpoint}: {r.text[:200]}"
    assert isinstance(r.json(), dict)


def test_summary_matches_real_processed_data(client):
    import pandas as pd
    body = client.get("/api/analytics/summary").json()
    df = client.app.state.analytics_service.data
    assert body["total_collisions"] == len(df) == int(df.shape[0])
    total = body["total_collisions"]
    assert sum(s["count"] for s in body["severity_distribution"]) == total
    for s in body["severity_distribution"]:
        assert abs(s["percentage"] - 100 * s["count"] / total) < 0.1
    assert body["years"] == sorted(df["collision_year"].unique().tolist())


def test_severity_distribution_counts_are_real(client):
    body = client.get("/api/analytics/severity-distribution").json()
    df = client.app.state.analytics_service.data
    real = df["severity"].value_counts().to_dict()
    for item in body["items"]:
        assert item["count"] == int(real.get(item["severity"], 0))


def test_analytics_filters(client):
    df = client.app.state.analytics_service.data
    year = int(df["collision_year"].iloc[0])
    body = client.get(f"/api/analytics/by-year?year={year}").json()
    assert len(body["items"]) == 1 and body["items"][0]["key"] == year
    assert body["items"][0]["total"] == int((df["collision_year"] == year).sum())
    # invalid severity value -> 422
    assert client.get("/api/analytics/by-year?severity=Bogus").status_code == 422


def test_hourly_series_covers_0_to_23(client):
    items = client.get("/api/analytics/by-hour").json()["items"]
    assert [it["key"] for it in items] == list(range(24))
    df = client.app.state.analytics_service.data
    assert sum(it["total"] for it in items) == len(df)


# ------------------------------------------------------------------ cross-filtered dashboard
def test_dashboard_endpoint_returns_everything_in_one_response(client):
    import pandas as pd  # noqa: F401
    r = client.get("/api/analytics/dashboard")
    assert r.status_code == 200, r.text
    body = r.json()
    for key in ("by-year", "by-month", "by-road-type", "by-weather", "by-lighting",
                "by-speed-limit", "by-hour", "by-weekday", "by-urban-rural"):
        assert body["series"][key]["items"], f"{key} must contain real rows"
    assert body["severity"]["items"]
    assert body["summary"]["filtered"] is False
    assert body["summary"]["total_collisions"] == len(client.app.state.analytics_service.data)
    # chip labels come from the real aggregation
    assert body["labels"]["month"]["5"] == "May"
    assert "Slight" in body["labels"]["severity"]


def test_dashboard_month_filters_everything_else(client):
    df = client.app.state.analytics_service.data
    jan = df[df["month"] == 1]
    body = client.get("/api/analytics/dashboard?month=1").json()
    # KPI summary = January only
    assert body["summary"]["total_collisions"] == len(jan)
    assert body["summary"]["filtered"] is True
    # every other chart = January only
    for key in ("by-road-type", "by-weather", "by-lighting", "by-speed-limit",
                "by-hour", "by-weekday", "by-year", "by-urban-rural"):
        got = sum(i["total"] for i in body["series"][key]["items"])
        assert got == len(jan), f"{key} total {got} != January {len(jan)}"
    # the monthly chart itself keeps ALL months (self-exclusion) for further picking
    assert [i["key"] for i in body["series"]["by-month"]["items"]] == list(range(1, 13))
    # severity donut = January only
    assert body["severity"]["total"] == len(jan)
    assert sum(i["count"] for i in body["severity"]["items"]) == len(jan)


def test_dashboard_combined_month_and_severity(client):
    df = client.app.state.analytics_service.data
    both = df[(df["month"] == 1) & (df["severity"] == "Fatal")]
    body = client.get("/api/analytics/dashboard?month=1&severity=Fatal").json()
    # KPIs = May AND Fatal equivalent (January AND Fatal here)
    assert body["summary"]["total_collisions"] == len(both)
    counts = {s["severity"]: s["count"] for s in body["summary"]["severity_distribution"]}
    assert counts["Fatal"] == len(both) and counts["Slight"] == 0 and counts["Serious"] == 0
    # severity chart ignores its own filter -> January across all severities
    assert body["severity"]["total"] == len(df[df["month"] == 1])
    # monthly chart ignores its own filter -> Fatal across all months
    assert sum(i["total"] for i in body["series"]["by-month"]["items"]) == len(df[df["severity"] == "Fatal"])
    # remaining charts = January AND Fatal
    assert sum(i["total"] for i in body["series"]["by-weather"]["items"]) == len(both)


def test_dashboard_invalid_filters_rejected(client):
    assert client.get("/api/analytics/dashboard?month=13").status_code == 422
    assert client.get("/api/analytics/dashboard?hour=24").status_code == 422
    assert client.get("/api/analytics/dashboard?weekday=9").status_code == 422
    assert client.get("/api/analytics/dashboard?severity=Bogus").status_code == 422
    # string codes (incl. the dataset's 'unknown' bucket) are accepted
    assert client.get("/api/analytics/dashboard?weather=2").status_code == 200


# ------------------------------------------------------------------ prediction
def test_predict_valid(client):
    r = client.post("/api/predict", json=VALID)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["prediction"] in {"Slight", "Serious", "Fatal"}
    assert list(body["probabilities"]) == ["Slight", "Serious", "Fatal"]
    assert abs(sum(body["probabilities"].values()) - 1) < 1e-6
    assert body["confidence"] == pytest.approx(max(body["probabilities"].values()))
    assert "should not be treated as a definitive" in body["disclaimer"]


def test_predict_probabilities_come_from_the_model(client):
    body = client.post("/api/predict", json=VALID).json()
    svc = client.app.state.prediction_service
    direct = svc.model.predictor.pipeline.predict_proba(
        svc.model.predictor._frame([svc.model.predictor.validate(VALID)]))[0]
    assert [body["probabilities"][c] for c in ("Slight", "Serious", "Fatal")] == pytest.approx(list(direct))


@pytest.mark.parametrize("payload, expected", [
    ({**VALID, "hour": 25}, 422),                    # out of range
    ({**VALID, "day_of_week": 0}, 422),              # out of range
    ({**VALID, "number_of_vehicles": 0}, 422),       # below minimum
    ({**VALID, "road_type": 777}, 422),              # not a known STATS19 code
    ({k: v for k, v in VALID.items() if k != "weather_conditions"}, 422),   # missing required
    ({**VALID, "number_of_casualties": 3}, 422),     # leakage field rejected
    ({**VALID, "bogus_field": 1}, 422),              # unknown field rejected
])
def test_predict_invalid_inputs_rejected(client, payload, expected):
    r = client.post("/api/predict", json=payload)
    assert r.status_code == expected, r.text


def test_predict_batch(client):
    payload = {"records": [VALID, {**VALID, "hour": 8, "speed_limit": 30, "light_conditions": 1}]}
    r = client.post("/api/predict/batch", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["count"] == 2 and len(body["predictions"]) == 2
    assert body["predictions"][0]["probabilities"] != body["predictions"][1]["probabilities"]
    assert client.post("/api/predict/batch", json={"records": []}).status_code == 422
