import numpy as np
import pandas as pd

from ml.feature_config import ENGINEERED_FEATURES
from ml.feature_engineering import build_features, engineer_features, resolve_feature_lists


def _row(**kw):
    base = {"hour": 8, "day_of_week": 2, "month": 1, "speed_limit": 30, "number_of_vehicles": 2, "road_type": 6,
            "junction_detail": 3, "light_conditions": 1, "weather_conditions": 2, "road_surface_conditions": 2,
            "urban_or_rural_area": 1}
    base.update(kw)
    return pd.DataFrame([base])


def test_time_features_weekday_rush_hour():
    d, _ = build_features(_row())                       # Monday 08:00, January
    r = d.iloc[0]
    assert (r.weekday, r.is_weekend, r.is_rush_hour, r.is_night, r.season) == (0, 0, 1, 0, "winter")


def test_weekend_night():
    d, _ = build_features(_row(hour=23, day_of_week=7, month=7))   # Saturday 23:00, July
    r = d.iloc[0]
    assert (r.weekday, r.is_weekend, r.is_rush_hour, r.is_night, r.season) == (5, 1, 0, 1, "summer")
    assert abs(r.hour_sin) < 0.3 and r.hour_cos > 0.9


def test_date_column_preferred_over_codes():
    df = _row().drop(columns=["month", "day_of_week"])
    df["date"] = pd.to_datetime(["2099-03-10"])
    d, _ = build_features(df)
    assert d.iloc[0].month == 3 and d.iloc[0].weekday == pd.Timestamp("2099-03-10").dayofweek


def test_groupings_and_unknown_codes():
    d, _ = build_features(_row(weather_conditions=9, light_conditions=6, junction_detail=99, road_type=12,
                               road_surface_conditions=4, speed_limit=99))
    r = d.iloc[0]
    assert r.weather_group == "unknown" and r.lighting_group == "dark_unlit" and r.junction_group == "unknown"
    assert r.road_type_group == "one_way_or_slip" and r.surface_group == "snow_ice"
    assert r.weather_conditions == "unknown" and np.isnan(r.speed_limit) and r.speed_band == "unknown"


def test_speed_bands_and_vehicle_cap():
    d, _ = build_features(pd.DataFrame({"speed_limit": [20, 30, 40, 50, 60, 70], "number_of_vehicles": [1, 2, 3, 15, 0, 5]}), 10)
    assert d.speed_band.tolist() == ["20_or_less", "30", "40_50", "40_50", "60", "70"]
    assert d.number_of_vehicles.iloc[3] == 10 and np.isnan(d.number_of_vehicles.iloc[4])
    assert d.multi_vehicle.iloc[0] == 0 and d.multi_vehicle.iloc[1] == 1


def test_nothing_is_invented_when_sources_missing():
    d, rep = build_features(pd.DataFrame({"speed_limit": [30, 60]}))
    assert "speed_band" in rep["created"]
    for f in ("hour", "is_night", "weather_group", "season"):
        assert f in rep["skipped"] and f not in d.columns
    numeric, categorical, missing = resolve_feature_lists(d)
    assert "hour" not in numeric and "hour" in missing


def test_deterministic_and_documented():
    a, rep = build_features(_row())
    b, _ = build_features(_row())
    pd.testing.assert_frame_equal(a, b)
    assert all(f in ENGINEERED_FEATURES and ENGINEERED_FEATURES[f] for f in rep["created"] if f in ENGINEERED_FEATURES)
    assert set(rep["reasons"]) == {f for f in rep["created"] if f in ENGINEERED_FEATURES}
