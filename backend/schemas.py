"""Pydantic schemas.

PredictionRequest mirrors EXACTLY the model's input_fields (see
ml/artifacts/feature_metadata.json -> input_fields, produced by ml/train.py):
only information that would be known at prediction time, no post-outcome
fields (no casualties, no attendance, no identifiers).

Code-value whitelists (e.g. road_type must be a real STATS19 code) are
enforced by ml.predict.Predictor.validate, which raises a ValueError that the
API converts into HTTP 422.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    # strict: unknown/misspelled fields are rejected (leakage guard as well)
    model_config = ConfigDict(extra="forbid")

    # --- MODEL SELECTION ----------------------------------------------------------------------
    model: str | None = Field(
        None,
        description="Model id from GET /api/models (e.g. 'random_forest'). "
                    "null or 'auto' uses the automatically selected best-validation model.",
    )

    # --- ROAD ---------------------------------------------------------------------------------
    speed_limit: int = Field(..., description="mph (STATS19: 20/30/40/50/60/70)")
    road_type: int = Field(..., description="STATS19 road_type code")
    junction_detail: int = Field(..., description="STATS19 junction_detail code")
    junction_control: int | None = Field(None, description="STATS19 junction_control code (optional)")
    first_road_class: int | None = Field(None, description="STATS19 first_road_class code (optional)")
    trunk_road_flag: int | None = Field(None, description="STATS19 trunk_road_flag code (optional)")

    # --- ENVIRONMENT -------------------------------------------------------------------------
    light_conditions: int = Field(..., description="STATS19 light_conditions code")
    weather_conditions: int = Field(..., description="STATS19 weather_conditions code")
    road_surface_conditions: int = Field(..., description="STATS19 road_surface_conditions code")
    special_conditions_at_site: int | None = Field(None, description="STATS19 special_conditions_at_site code")
    carriageway_hazards: int | None = Field(None, description="STATS19 carriageway_hazards code")

    # --- AREA --------------------------------------------------------------------------------
    urban_or_rural_area: int = Field(..., description="STATS19 urban_or_rural_area code (1=Urban, 2=Rural)")

    # --- COLLISION CONTEXT -------------------------------------------------------------------
    number_of_vehicles: int = Field(..., ge=1, description="Vehicles involved (1..cap); known when the system is asked to judge a *new* reported collision")
    hour: int = Field(..., ge=0, le=23, description="Hour of day 0-23")
    day_of_week: int = Field(..., ge=1, le=7, description="STATS19 coding: 1=Sunday ... 7=Saturday")
    month: int | None = Field(None, ge=1, le=12, description="Month 1-12 (optional)")


class PredictionResponse(BaseModel):
    prediction: str
    probabilities: dict[str, float]
    confidence: float
    confidence_note: str | None = None
    top_factors: list[dict] = []
    factors_note: str | None = None
    model: str | None = None
    disclaimer: str | None = None


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    records: list[PredictionRequest] = Field(..., min_length=1, max_length=500)


class BatchPredictionResponse(BaseModel):
    count: int
    predictions: list[PredictionResponse]
