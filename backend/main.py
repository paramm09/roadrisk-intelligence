"""ROADRISK Intelligence - FastAPI application.

Run from the project root:
    uvicorn backend.main:app --reload

Design rules:
- The trained artifact and the processed dataset are loaded ONCE at startup
  (lifespan); nothing re-trains and no CSV is read per request.
- Every number returned by /api/analytics/* is computed from the real
  processed STATS19 data; every prediction comes from the real model.
- If artifacts are missing, the API still starts: /health reports
  model_loaded=false and prediction endpoints answer 503 with instructions.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .config import BackendConfig, get_config
from .schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionRequest,
    PredictionResponse,
)
from .services.analytics_service import AnalyticsService, Filters
from .services.model_registry import ModelNotAvailableError
from .services.model_service import ModelService
from .services.prediction_service import PredictionService


def create_app(cfg: BackendConfig | None = None) -> FastAPI:
    cfg = cfg or get_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model_service.load()
        app.state.analytics_service.load()
        yield

    app = FastAPI(
        title="ROADRISK Intelligence API",
        description=(
            "Road collision severity prediction and road safety analytics over "
            "UK DfT STATS19 data. Model output is an analytical prediction based "
            "on historical reported collision data and should not be treated as a "
            "definitive determination of collision severity or causation."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.cfg = cfg
    app.state.model_service = ModelService(cfg)
    app.state.analytics_service = AnalyticsService(cfg, app.state.model_service)
    app.state.prediction_service = PredictionService(app.state.model_service)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(cfg.cors_origins),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    _register_routes(app)
    return app


# --------------------------------------------------------------------------------------
def _model(app: FastAPI) -> ModelService:
    return app.state.model_service


def _analytics(app: FastAPI) -> AnalyticsService:
    return app.state.analytics_service


def _require_model(app: FastAPI) -> ModelService:
    ms = _model(app)
    if not ms.loaded:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts are not available. Train first: "
                   "python scripts/inspect_data.py && python scripts/prepare_data.py "
                   "&& python scripts/train_models.py && python scripts/evaluate_models.py")
    return ms


def _require_data(app: FastAPI) -> AnalyticsService:
    an = _analytics(app)
    if not an.loaded:
        raise HTTPException(
            status_code=503,
            detail=f"Processed dataset not available ({an.error}). "
                   "Run: python scripts/prepare_data.py")
    return an


def _filters(year: int | None, severity: str | None, road_type: int | str | None,
             weather: int | str | None, lighting: int | str | None, speed_limit: int | None,
             urban_rural: int | str | None, month: int | None = None, hour: int | None = None,
             weekday: int | None = None) -> Filters:
    if severity is not None and severity not in {"Slight", "Serious", "Fatal"}:
        raise HTTPException(status_code=422, detail="severity must be Slight, Serious or Fatal")
    if month is not None and not 1 <= month <= 12:
        raise HTTPException(status_code=422, detail="month must be 1..12")
    if hour is not None and not 0 <= hour <= 23:
        raise HTTPException(status_code=422, detail="hour must be 0..23")
    if weekday is not None and not 0 <= weekday <= 6:
        raise HTTPException(status_code=422, detail="weekday must be 0..6 (Mon..Sun)")
    return Filters(year=year, month=month, severity=severity, road_type=road_type,
                   weather=weather, lighting=lighting, speed_limit=speed_limit,
                   urban_rural=urban_rural, hour=hour, weekday=weekday)


FILTER_QUERY = dict(
    year=Query(None, description="Collision year, e.g. 2023"),
    month=Query(None, description="Month 1..12 (1=Jan)"),
    severity=Query(None, description="Slight | Serious | Fatal"),
    road_type=Query(None, description="STATS19 road_type code"),
    weather=Query(None, description="STATS19 weather_conditions code"),
    lighting=Query(None, description="STATS19 light_conditions code"),
    speed_limit=Query(None, description="Speed limit mph"),
    urban_rural=Query(None, description="STATS19 urban_or_rural_area code (1/2)"),
    hour=Query(None, description="Hour of day 0..23"),
    weekday=Query(None, description="Weekday 0..6 (Mon..Sun)"),
)


def _register_routes(app: FastAPI) -> None:

    # ------------------------------------------------------------------ health
    @app.get("/health", tags=["system"])
    def health() -> dict:
        ms, an = _model(app), _analytics(app)
        return {
            "status": "ok",
            "model_loaded": ms.loaded,
            "model_error": ms.error,
            "analytics_loaded": an.loaded,
            "analytics_error": an.error,
        }

    # ------------------------------------------------------------------ model
    @app.get("/api/model/info", tags=["model"])
    def model_info() -> dict:
        return _require_model(app).info()

    @app.get("/api/model/metrics", tags=["model"])
    def model_metrics() -> dict:
        return _require_model(app).metrics()

    @app.get("/api/model/feature-importance", tags=["model"])
    def feature_importance() -> dict:
        try:
            return _require_model(app).feature_importance()
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

    @app.get("/api/models", tags=["model"])
    def list_models() -> dict:
        """Genuinely available trained models (artifact files on disk) + real metrics."""
        return _require_model(app).list_models()

    # ------------------------------------------------------------------ analytics
    @app.get("/api/analytics/dashboard", tags=["analytics"])
    def analytics_dashboard(
        year: int | None = FILTER_QUERY["year"], month: int | None = FILTER_QUERY["month"],
        severity: str | None = FILTER_QUERY["severity"], road_type: str | None = FILTER_QUERY["road_type"],
        weather: str | None = FILTER_QUERY["weather"], lighting: str | None = FILTER_QUERY["lighting"],
        speed_limit: int | None = FILTER_QUERY["speed_limit"],
        urban_rural: str | None = FILTER_QUERY["urban_rural"],
        hour: int | None = FILTER_QUERY["hour"], weekday: int | None = FILTER_QUERY["weekday"]) -> dict:
        """Everything the cross-filtered dashboard needs in ONE request.

        KPI summary uses all filters; each series excludes only its own dimension so
        the clicked chart keeps showing every category (selected one emphasised).
        """
        return _require_data(app).dashboard(
            _filters(year, severity, road_type, weather, lighting, speed_limit,
                     urban_rural, month=month, hour=hour, weekday=weekday))

    @app.get("/api/analytics/summary", tags=["analytics"])
    def analytics_summary() -> dict:
        return _require_data(app).summary()

    @app.get("/api/analytics/severity-distribution", tags=["analytics"])
    def severity_distribution(
        year: int | None = FILTER_QUERY["year"], severity: str | None = FILTER_QUERY["severity"],
        road_type: int | None = FILTER_QUERY["road_type"], weather: int | None = FILTER_QUERY["weather"],
        lighting: int | None = FILTER_QUERY["lighting"], speed_limit: int | None = FILTER_QUERY["speed_limit"],
        urban_rural: int | None = FILTER_QUERY["urban_rural"]) -> dict:
        return _require_data(app).severity_distribution(
            _filters(year, severity, road_type, weather, lighting, speed_limit, urban_rural))

    @app.get("/api/analytics/by-year", tags=["analytics"])
    def by_year(
        year: int | None = FILTER_QUERY["year"], severity: str | None = FILTER_QUERY["severity"],
        road_type: int | None = FILTER_QUERY["road_type"], weather: int | None = FILTER_QUERY["weather"],
        lighting: int | None = FILTER_QUERY["lighting"], speed_limit: int | None = FILTER_QUERY["speed_limit"],
        urban_rural: int | None = FILTER_QUERY["urban_rural"]) -> dict:
        return _require_data(app).by_year(
            _filters(year, severity, road_type, weather, lighting, speed_limit, urban_rural))

    @app.get("/api/analytics/by-road-type", tags=["analytics"])
    def by_road_type(
        year: int | None = FILTER_QUERY["year"], severity: str | None = FILTER_QUERY["severity"],
        road_type: int | None = FILTER_QUERY["road_type"], weather: int | None = FILTER_QUERY["weather"],
        lighting: int | None = FILTER_QUERY["lighting"], speed_limit: int | None = FILTER_QUERY["speed_limit"],
        urban_rural: int | None = FILTER_QUERY["urban_rural"]) -> dict:
        return _require_data(app).by_road_type(
            _filters(year, severity, road_type, weather, lighting, speed_limit, urban_rural))

    @app.get("/api/analytics/by-weather", tags=["analytics"])
    def by_weather(
        year: int | None = FILTER_QUERY["year"], severity: str | None = FILTER_QUERY["severity"],
        road_type: int | None = FILTER_QUERY["road_type"], weather: int | None = FILTER_QUERY["weather"],
        lighting: int | None = FILTER_QUERY["lighting"], speed_limit: int | None = FILTER_QUERY["speed_limit"],
        urban_rural: int | None = FILTER_QUERY["urban_rural"]) -> dict:
        return _require_data(app).by_weather(
            _filters(year, severity, road_type, weather, lighting, speed_limit, urban_rural))

    @app.get("/api/analytics/by-lighting", tags=["analytics"])
    def by_lighting(
        year: int | None = FILTER_QUERY["year"], severity: str | None = FILTER_QUERY["severity"],
        road_type: int | None = FILTER_QUERY["road_type"], weather: int | None = FILTER_QUERY["weather"],
        lighting: int | None = FILTER_QUERY["lighting"], speed_limit: int | None = FILTER_QUERY["speed_limit"],
        urban_rural: int | None = FILTER_QUERY["urban_rural"]) -> dict:
        return _require_data(app).by_lighting(
            _filters(year, severity, road_type, weather, lighting, speed_limit, urban_rural))

    @app.get("/api/analytics/by-speed-limit", tags=["analytics"])
    def by_speed_limit(
        year: int | None = FILTER_QUERY["year"], severity: str | None = FILTER_QUERY["severity"],
        road_type: int | None = FILTER_QUERY["road_type"], weather: int | None = FILTER_QUERY["weather"],
        lighting: int | None = FILTER_QUERY["lighting"], speed_limit: int | None = FILTER_QUERY["speed_limit"],
        urban_rural: int | None = FILTER_QUERY["urban_rural"]) -> dict:
        return _require_data(app).by_speed_limit(
            _filters(year, severity, road_type, weather, lighting, speed_limit, urban_rural))

    @app.get("/api/analytics/by-hour", tags=["analytics"])
    def by_hour(
        year: int | None = FILTER_QUERY["year"], severity: str | None = FILTER_QUERY["severity"],
        road_type: int | None = FILTER_QUERY["road_type"], weather: int | None = FILTER_QUERY["weather"],
        lighting: int | None = FILTER_QUERY["lighting"], speed_limit: int | None = FILTER_QUERY["speed_limit"],
        urban_rural: int | None = FILTER_QUERY["urban_rural"]) -> dict:
        return _require_data(app).by_hour(
            _filters(year, severity, road_type, weather, lighting, speed_limit, urban_rural))

    # -- extra series the dashboard/analytics pages need (same real aggregations) --------
    @app.get("/api/analytics/by-weekday", tags=["analytics"])
    def by_weekday(
        year: int | None = FILTER_QUERY["year"], severity: str | None = FILTER_QUERY["severity"],
        road_type: int | None = FILTER_QUERY["road_type"], weather: int | None = FILTER_QUERY["weather"],
        lighting: int | None = FILTER_QUERY["lighting"], speed_limit: int | None = FILTER_QUERY["speed_limit"],
        urban_rural: int | None = FILTER_QUERY["urban_rural"]) -> dict:
        return _require_data(app).by_weekday(
            _filters(year, severity, road_type, weather, lighting, speed_limit, urban_rural))

    @app.get("/api/analytics/by-month", tags=["analytics"])
    def by_month(
        year: int | None = FILTER_QUERY["year"], severity: str | None = FILTER_QUERY["severity"],
        road_type: int | None = FILTER_QUERY["road_type"], weather: int | None = FILTER_QUERY["weather"],
        lighting: int | None = FILTER_QUERY["lighting"], speed_limit: int | None = FILTER_QUERY["speed_limit"],
        urban_rural: int | None = FILTER_QUERY["urban_rural"]) -> dict:
        return _require_data(app).by_month(
            _filters(year, severity, road_type, weather, lighting, speed_limit, urban_rural))

    @app.get("/api/analytics/by-urban-rural", tags=["analytics"])
    def by_urban_rural(
        year: int | None = FILTER_QUERY["year"], severity: str | None = FILTER_QUERY["severity"],
        road_type: int | None = FILTER_QUERY["road_type"], weather: int | None = FILTER_QUERY["weather"],
        lighting: int | None = FILTER_QUERY["lighting"], speed_limit: int | None = FILTER_QUERY["speed_limit"],
        urban_rural: int | None = FILTER_QUERY["urban_rural"]) -> dict:
        return _require_data(app).by_urban_rural(
            _filters(year, severity, road_type, weather, lighting, speed_limit, urban_rural))

    @app.get("/api/analytics/by-junction", tags=["analytics"])
    def by_junction(
        year: int | None = FILTER_QUERY["year"], severity: str | None = FILTER_QUERY["severity"],
        road_type: int | None = FILTER_QUERY["road_type"], weather: int | None = FILTER_QUERY["weather"],
        lighting: int | None = FILTER_QUERY["lighting"], speed_limit: int | None = FILTER_QUERY["speed_limit"],
        urban_rural: int | None = FILTER_QUERY["urban_rural"]) -> dict:
        return _require_data(app).by_junction(
            _filters(year, severity, road_type, weather, lighting, speed_limit, urban_rural))

    @app.get("/api/analytics/map-points", tags=["analytics"])
    def map_points(
        year: int | None = FILTER_QUERY["year"], severity: str | None = FILTER_QUERY["severity"],
        road_type: int | None = FILTER_QUERY["road_type"], weather: int | None = FILTER_QUERY["weather"],
        lighting: int | None = FILTER_QUERY["lighting"], speed_limit: int | None = FILTER_QUERY["speed_limit"],
        urban_rural: int | None = FILTER_QUERY["urban_rural"]) -> dict:
        return _require_data(app).map_points(
            _filters(year, severity, road_type, weather, lighting, speed_limit, urban_rural))

    @app.get("/api/insights", tags=["analytics"])
    def insights() -> dict:
        return _require_data(app).insights()

    # ------------------------------------------------------------------ prediction
    @app.post("/api/predict", response_model=PredictionResponse, tags=["prediction"])
    def predict(request: PredictionRequest) -> dict:
        _require_model(app)
        try:
            return app.state.prediction_service.predict(request)
        except ModelNotAvailableError as e:      # user explicitly chose a model -> 400
            raise HTTPException(status_code=400, detail=str(e)) from e
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

    @app.post("/api/predict/batch", response_model=BatchPredictionResponse, tags=["prediction"])
    def predict_batch(request: BatchPredictionRequest) -> dict:
        _require_model(app)
        try:
            return app.state.prediction_service.predict_batch(request)
        except ModelNotAvailableError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

    return app


app = create_app()
