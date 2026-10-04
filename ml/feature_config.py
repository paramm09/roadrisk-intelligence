"""Feature whitelist, exclusions (with reasons), STATS19 code handling and groupings.

SOURCE OF TRUTH for leakage control. Every column the model may use is listed in
ALLOWED_FEATURES; everything else is excluded and must have a reason in EXCLUDED_FEATURES.

The code dictionaries below follow the DfT STATS19 data guide as understood when this
was written. They are DATA, not logic: if your year's data guide differs, edit them here.
The data-report step prints the observed code values so you can verify them.
"""
from __future__ import annotations

import re

# ------------------------------------------------------------------ target
TARGET = "accident_severity"        # canonical name; DfT files call it collision_severity
ID_COL = "collision_index"
YEAR_COL = "collision_year"
# Expected raw coding (VERIFIED AT RUNTIME by ml.data_quality.validate_target):
TARGET_CODE_TO_LABEL = {1: "Fatal", 2: "Serious", 3: "Slight"}
CLASS_NAMES = ["Slight", "Serious", "Fatal"]               # model class order (0,1,2)
LABEL_TO_INT = {n: i for i, n in enumerate(CLASS_NAMES)}
CODE_TO_INT = {3: 0, 2: 1, 1: 2}
FATAL_INT = 2

# ------------------------------------------------------------------ column naming
# Column names are normalised (lower-case, non-alphanumerics -> "_") then aliased.
COLUMN_ALIASES = {
    "collision_severity": TARGET,
    "accident_severity": TARGET,
    "accident_index": ID_COL,
    "accident_year": YEAR_COL,
    "accident_reference": "collision_ref_no",
    "1st_road_class": "first_road_class",
    "1st_road_number": "first_road_number",
    "2nd_road_class": "second_road_class",
    "2nd_road_number": "second_road_number",
}


def normalise_column_name(name: str) -> str:
    n = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip().lower()).strip("_")
    return COLUMN_ALIASES.get(n, n)


# ------------------------------------------------------------------ raw feature columns
# column -> codes that mean "unknown / missing / out of range / unallocated" (-> "unknown").
# -1 is the DfT "data missing or out of range" code in every coded field.
CATEGORICAL_RAW_INVALID: dict[str, set] = {
    "road_type": {-1, 9},
    "junction_detail": {-1, 99},
    "junction_control": {-1, 9},
    "light_conditions": {-1},
    "weather_conditions": {-1, 9},
    "road_surface_conditions": {-1, 9},
    "urban_or_rural_area": {-1, 3},          # 3 = unallocated
    "first_road_class": {-1},
    "trunk_road_flag": {-1},
    "special_conditions_at_site": {-1, 9},
    "carriageway_hazards": {-1, 9},
    "pedestrian_crossing_human_control": {-1, 9},
    "pedestrian_crossing_physical_facilities": {-1, 9},
}
SPEED_LIMIT_VALID = {20, 30, 40, 50, 60, 70}   # any other value (e.g. 99, -1) -> missing

# Columns read from the raw CSVs (everything else is never loaded -> saves RAM).
SOURCE_COLUMNS = (
    {TARGET, ID_COL, YEAR_COL, "date", "time", "day_of_week", "latitude", "longitude",
     "speed_limit", "number_of_vehicles"} | set(CATEGORICAL_RAW_INVALID)
)
NUMERIC_SOURCE_COLUMNS = (
    {TARGET, YEAR_COL, "day_of_week", "latitude", "longitude", "speed_limit",
     "number_of_vehicles"} | set(CATEGORICAL_RAW_INVALID)
)

# ------------------------------------------------------------------ model features
MODEL_NUMERIC = [
    "speed_limit", "number_of_vehicles", "hour", "hour_sin", "hour_cos", "month", "weekday",
    "is_weekend", "is_rush_hour", "is_night", "multi_vehicle",
]
MODEL_CATEGORICAL = list(CATEGORICAL_RAW_INVALID) + [
    "season", "speed_band", "weather_group", "surface_group", "lighting_group",
    "junction_group", "road_type_group",
]
ALLOWED_FEATURES = list(MODEL_NUMERIC) + list(MODEL_CATEGORICAL)

# Engineered features -> documented reason (feature-engineering report)
ENGINEERED_FEATURES = {
    "hour": "Hour of day (0-23) from the recorded collision time; risk and traffic mix vary through the day.",
    "hour_sin": "Cyclical encoding of hour so 23:00 and 00:00 are neighbours (helps linear models).",
    "hour_cos": "Cosine partner of hour_sin.",
    "month": "Month of year from collision date; captures seasonal effects.",
    "weekday": "Day of week (0=Mon..6=Sun) from date (falls back to the STATS19 day_of_week code).",
    "is_weekend": "Weekend flag; leisure traffic differs from commuter traffic.",
    "is_rush_hour": "Weekday 07-09 or 16-18; congestion lowers speeds but raises collision counts.",
    "is_night": "Hour >= 21 or < 6; proxy for reduced visibility and fatigue/impairment (approximate, ignores season).",
    "season": "Meteorological season from month (Dec-Feb winter, ...).",
    "speed_band": "Speed limit grouped (<=20, 30, 40-50, 60, 70, unknown); kinetic energy rises non-linearly with speed.",
    "multi_vehicle": "More than one vehicle involved; single-vehicle crashes have a different severity profile.",
    "weather_group": "Weather codes collapsed to fine / rain / snow / high_wind / fog / other / unknown.",
    "surface_group": "Surface codes collapsed to dry / wet / snow_ice / contaminated / unknown.",
    "lighting_group": "Light codes collapsed to daylight / dark_lit / dark_unlit / unknown.",
    "junction_group": "Junction detail collapsed to no_junction / roundabout / t_junction / crossroads_multi / slip_road / private_drive / other / unknown.",
    "road_type_group": "Road type collapsed to roundabout / one_way_or_slip / dual_carriageway / single_carriageway / unknown.",
}

# Code -> group (keys are strings because raw codes are cleaned to strings first)
GROUP_MAPS = {
    "weather_conditions": ("weather_group", {
        "1": "fine", "2": "rain", "5": "rain", "3": "snow", "6": "snow",
        "4": "high_wind", "7": "fog", "8": "other"}),
    "road_surface_conditions": ("surface_group", {
        "1": "dry", "2": "wet", "5": "wet", "3": "snow_ice", "4": "snow_ice",
        "6": "contaminated", "7": "contaminated"}),
    "light_conditions": ("lighting_group", {
        "1": "daylight", "4": "dark_lit", "5": "dark_unlit", "6": "dark_unlit", "7": "dark_unlit"}),
    "junction_detail": ("junction_group", {
        "0": "no_junction", "1": "roundabout", "2": "roundabout", "3": "t_junction",
        "6": "crossroads_multi", "7": "crossroads_multi", "5": "slip_road",
        "8": "private_drive", "9": "other"}),
    "road_type": ("road_type_group", {
        "1": "roundabout", "2": "one_way_or_slip", "7": "one_way_or_slip", "12": "one_way_or_slip",
        "3": "dual_carriageway", "6": "single_carriageway"}),
}

# Feature "families": raw + derived columns that carry the same information. Permutation
# importance and per-prediction explanations are computed per family (permuting only one of
# two redundant columns would understate its importance).
FEATURE_FAMILIES = {
    "Speed limit": ["speed_limit", "speed_band"],
    "Time of day": ["hour", "hour_sin", "hour_cos", "is_rush_hour", "is_night"],
    "Day of week": ["weekday", "is_weekend"],
    "Month / season": ["month", "season"],
    "Road type": ["road_type", "road_type_group"],
    "Junction": ["junction_detail", "junction_control", "junction_group"],
    "Lighting": ["light_conditions", "lighting_group"],
    "Weather": ["weather_conditions", "weather_group"],
    "Road surface": ["road_surface_conditions", "surface_group"],
    "Urban / rural": ["urban_or_rural_area"],
    "Vehicles involved": ["number_of_vehicles", "multi_vehicle"],
    "Road class": ["first_road_class", "trunk_road_flag"],
    "Pedestrian crossings": ["pedestrian_crossing_human_control", "pedestrian_crossing_physical_facilities"],
    "Special conditions / hazards": ["special_conditions_at_site", "carriageway_hazards"],
}

# ------------------------------------------------------------------ exclusions
EXCLUDED_FEATURES = {
    TARGET: "TARGET. Never a feature.",
    "collision_severity": "TARGET (DfT's name for accident_severity).",
    "casualty_severity": "LEAKAGE: severity of individual casualties; the collision severity is defined as the worst casualty severity. Casualty file is never loaded.",
    "casualty_class": "LEAKAGE/POST-EVENT: casualty-file field (describes who was hurt). Casualty file is never loaded.",
    "casualty_type": "LEAKAGE/POST-EVENT: casualty-file field. Casualty file is never loaded.",
    "number_of_casualties": "LEAKAGE: only known after the outcome, and mechanically linked to severity (a fatal collision has at least one casualty; more casualties -> more likely severe).",
    "did_police_officer_attend_scene_of_accident": "POST-EVENT: attendance is a response to the collision and is more likely for severe collisions, so it encodes the outcome.",
    "collision_index": "IDENTIFIER: no predictive meaning; kept only to de-duplicate and verify the split.",
    "collision_ref_no": "IDENTIFIER.",
    "collision_year": "Excluded to avoid memorising year-level drift (reporting changes, COVID-19 2020); kept only for analytics and temporal splitting.",
    "date": "Raw date replaced by engineered month / weekday / season.",
    "time": "Raw time replaced by engineered hour and derived flags.",
    "day_of_week": "Raw STATS19 code (1=Sunday) replaced by engineered weekday / is_weekend.",
    "latitude": "HIGH-CARDINALITY continuous geography. Not used as a model feature (location memorisation, weak generalisation); rounded copy kept for the historical map and grid aggregation only.",
    "longitude": "As latitude.",
    "location_easting_osgr": "As latitude (OSGR coordinates).",
    "location_northing_osgr": "As latitude (OSGR coordinates).",
    "police_force": "Administrative area with force-specific reporting practice; would let the model learn reporting differences rather than collision circumstances. (Move to ALLOWED_FEATURES to test it.)",
    "local_authority_district": "HIGH-CARDINALITY administrative identifier (~380 levels).",
    "local_authority_ons_district": "HIGH-CARDINALITY administrative identifier.",
    "local_authority_highway": "HIGH-CARDINALITY administrative identifier.",
    "lsoa_of_accident_location": "HIGH-CARDINALITY small-area identifier (~35k levels).",
    "first_road_number": "HIGH-CARDINALITY road identifier.",
    "second_road_number": "HIGH-CARDINALITY road identifier.",
    "second_road_class": "Structurally missing away from junctions and largely redundant with junction_detail.",
}

# Any model feature whose name matches one of these fails the leakage audit.
LEAKAGE_NAME_PATTERNS = ["casualt", "severity", "fatal", "killed", "injur", "outcome", "hospital", "died", "death"]

# ------------------------------------------------------------------ labels (for UI / reports)
LABELS = {
    "road_type": {"1": "Roundabout", "2": "One way street", "3": "Dual carriageway",
                  "6": "Single carriageway", "7": "Slip road", "9": "Unknown",
                  "12": "One way street / slip road"},
    "junction_detail": {"0": "Not at junction (or within 20m)", "1": "Roundabout", "2": "Mini-roundabout",
                        "3": "T or staggered junction", "5": "Slip road", "6": "Crossroads",
                        "7": "More than 4 arms", "8": "Private drive or entrance", "9": "Other junction"},
    "junction_control": {"0": "Not at junction", "1": "Authorised person", "2": "Automatic traffic signal",
                         "3": "Stop sign", "4": "Give way or uncontrolled"},
    "light_conditions": {"1": "Daylight", "4": "Darkness - lights lit", "5": "Darkness - lights unlit",
                         "6": "Darkness - no lighting", "7": "Darkness - lighting unknown"},
    "weather_conditions": {"1": "Fine, no high winds", "2": "Raining, no high winds", "3": "Snowing, no high winds",
                           "4": "Fine + high winds", "5": "Raining + high winds", "6": "Snowing + high winds",
                           "7": "Fog or mist", "8": "Other"},
    "road_surface_conditions": {"1": "Dry", "2": "Wet or damp", "3": "Snow", "4": "Frost or ice",
                                "5": "Flood over 3cm deep", "6": "Oil or diesel", "7": "Mud"},
    "urban_or_rural_area": {"1": "Urban", "2": "Rural"},
    "first_road_class": {"1": "Motorway", "2": "A(M)", "3": "A", "4": "B", "5": "C", "6": "Unclassified"},
    "trunk_road_flag": {"1": "Trunk road", "2": "Non-trunk road"},
    "special_conditions_at_site": {"0": "None", "1": "Auto traffic signal - out", "2": "Auto signal part defective",
                                   "3": "Road sign or marking defective or obscured", "4": "Roadworks",
                                   "5": "Road surface defective", "6": "Oil or diesel", "7": "Mud"},
    "carriageway_hazards": {"0": "None", "1": "Vehicle load on road", "2": "Other object on road",
                            "3": "Previous accident", "4": "Dog on road", "5": "Other animal on road",
                            "6": "Pedestrian in carriageway", "7": "Any animal in carriageway"},
    "pedestrian_crossing_human_control": {"0": "None within 50m", "1": "School crossing patrol",
                                          "2": "Other authorised person"},
    "pedestrian_crossing_physical_facilities": {"0": "No facility within 50m", "1": "Zebra", "4": "Pelican/puffin/toucan",
                                                "5": "Pedestrian phase at junction signals", "7": "Footbridge or subway",
                                                "8": "Central refuge"},
}
DAY_OF_WEEK_LABELS = {"1": "Sunday", "2": "Monday", "3": "Tuesday", "4": "Wednesday",
                      "5": "Thursday", "6": "Friday", "7": "Saturday"}
