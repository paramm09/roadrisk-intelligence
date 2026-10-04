"""Generate the SYNTHETIC, TEST-ONLY fixture data/sample/test_fixture.csv.

!! THIS DATA IS FAKE. It exists only so unit/integration tests can run without STATS19.
!! It must never be used for real training, reported metrics, README results or dashboards.
Rows have collision_index starting 'TEST' and collision_year 2099; the pipeline refuses to
train on them unless ROADRISK_ALLOW_TEST_DATA=1 (set only by the tests).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["collision_index", "collision_year", "collision_ref_no", "location_easting_osgr", "location_northing_osgr",
           "longitude", "latitude", "police_force", "collision_severity", "number_of_vehicles", "number_of_casualties",
           "date", "day_of_week", "time", "local_authority_district", "first_road_class", "first_road_number",
           "road_type", "speed_limit", "junction_detail", "junction_control", "second_road_class",
           "pedestrian_crossing_human_control", "pedestrian_crossing_physical_facilities", "light_conditions",
           "weather_conditions", "road_surface_conditions", "special_conditions_at_site", "carriageway_hazards",
           "urban_or_rural_area", "did_police_officer_attend_scene_of_accident", "trunk_road_flag",
           "lsoa_of_accident_location"]


def make_fixture(n: int = 3000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.to_datetime("2099-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D")
    hour = rng.choice(24, n, p=np.r_[np.full(6, .015), np.full(18, .91 / 18)] / (np.r_[np.full(6, .015), np.full(18, .91 / 18)]).sum())
    speed = rng.choice([20, 30, 40, 50, 60, 70], n, p=[.05, .55, .1, .08, .15, .07])
    rural = rng.choice([1, 2], n, p=[.65, .35])
    light = np.where((hour < 6) | (hour > 20), rng.choice([4, 5, 6], n, p=[.6, .1, .3]), 1)
    veh = rng.choice([1, 2, 3, 4], n, p=[.3, .55, .12, .03])
    z = -3.2 + 0.03 * (speed - 30) + 0.6 * (light == 6) + 0.4 * (rural == 2) + 0.15 * (veh == 1)
    p_ser = 1 / (1 + np.exp(-(z + 1.0)))
    u = rng.random(n)
    p_fatal = 1 / (1 + np.exp(-(z - 0.8)))
    sev = np.where(u < p_fatal * 0.3, 1, np.where(u < p_ser * 0.6, 2, 3))
    def codes(vals, p=None):
        return rng.choice(vals, n, p=p)
    df = pd.DataFrame({
        "collision_index": [f"TEST{i:07d}" for i in range(n)], "collision_year": 2099,
        "collision_ref_no": [f"R{i}" for i in range(n)],
        "location_easting_osgr": rng.integers(400000, 500000, n), "location_northing_osgr": rng.integers(100000, 300000, n),
        "longitude": rng.uniform(-3, 0.5, n).round(6), "latitude": rng.uniform(51, 54, n).round(6),
        "police_force": codes([1, 3, 4, 5]), "collision_severity": sev, "number_of_vehicles": veh,
        "number_of_casualties": np.where(sev == 1, rng.integers(1, 5, n), rng.integers(1, 3, n)),
        "date": dates.strftime("%d/%m/%Y"), "day_of_week": ((dates.dayofweek + 1) % 7) + 1,
        "time": [f"{h:02d}:{m:02d}" for h, m in zip(hour, rng.integers(0, 60, n))],
        "local_authority_district": codes([1, 2, 3]), "first_road_class": codes([3, 4, 5, 6], [.3, .2, .2, .3]),
        "first_road_number": rng.integers(0, 999, n), "road_type": codes([1, 2, 3, 6, 7, 9], [.05, .03, .12, .72, .06, .02]),
        "speed_limit": speed, "junction_detail": codes([0, 1, 2, 3, 6, 9, 99], [.45, .05, .02, .3, .1, .06, .02]),
        "junction_control": codes([-1, 0, 2, 4, 9], [.05, .45, .1, .35, .05]), "second_road_class": codes([-1, 3, 4], [.5, .3, .2]),
        "pedestrian_crossing_human_control": codes([0, 9, -1], [.9, .05, .05]),
        "pedestrian_crossing_physical_facilities": codes([0, 1, 5, 9], [.85, .05, .07, .03]),
        "light_conditions": light, "weather_conditions": codes([1, 2, 3, 7, 8, 9, -1], [.75, .15, .02, .02, .03, .02, .01]),
        "road_surface_conditions": codes([1, 2, 4, 9, -1], [.7, .25, .02, .02, .01]),
        "special_conditions_at_site": codes([0, 4, 9], [.95, .03, .02]), "carriageway_hazards": codes([0, 2, 9], [.95, .03, .02]),
        "urban_or_rural_area": rural, "did_police_officer_attend_scene_of_accident": codes([1, 2, 3], [.8, .15, .05]),
        "trunk_road_flag": codes([1, 2, -1], [.1, .88, .02]), "lsoa_of_accident_location": [f"E{i % 500:08d}" for i in range(n)],
    })[COLUMNS]
    df = pd.concat([df, df.iloc[:5]], ignore_index=True)          # 5 exact duplicates to test de-duplication
    return df


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "data" / "sample" / "test_fixture.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    make_fixture().to_csv(out, index=False)
    print(f"wrote SYNTHETIC TEST-ONLY fixture -> {out}")
