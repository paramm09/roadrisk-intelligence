import pytest

from ml.config import Settings, parse_years
from ml.data_loader import SchemaError, discover_raw_files, load_years
from tests.conftest import make_cfg


def test_parse_years():
    assert parse_years("2021-2025") == [2021, 2022, 2023, 2024, 2025]
    assert parse_years("2019,2021,2023") == [2019, 2021, 2023]
    assert parse_years("2022") == [2022]
    with pytest.raises(ValueError):
        parse_years("2025-2021")


def test_default_years_are_2021_2025_and_configurable(monkeypatch):
    monkeypatch.delenv("ROADRISK_YEARS", raising=False)
    assert Settings().years == [2021, 2022, 2023, 2024, 2025]
    monkeypatch.setenv("ROADRISK_YEARS", "2020-2024")
    assert Settings().years == [2020, 2021, 2022, 2023, 2024]


def _write(tmp_path, df, name):
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw / name, index=False)


def test_legacy_column_names_are_normalised(tmp_path, fixture_df):
    legacy = fixture_df.rename(columns={"collision_index": "Accident_Index", "collision_severity": "Accident_Severity",
                                        "first_road_class": "1st_Road_Class", "collision_year": "Accident_Year"})
    legacy.columns = [c if c[0].isupper() or c[0].isdigit() else c.title() for c in legacy.columns]
    _write(tmp_path, legacy, "accidents_2099.csv")
    df, rep = load_years(make_cfg(tmp_path))
    assert {"accident_severity", "collision_index", "first_road_class", "speed_limit"} <= set(df.columns)
    assert rep["years_used"] == [2099]
    assert "number_of_casualties" not in df.columns        # excluded column is never even loaded


def test_missing_target_column_fails_clearly(tmp_path, fixture_df):
    _write(tmp_path, fixture_df.drop(columns=["collision_severity"]), "collision-2099.csv")
    with pytest.raises(SchemaError, match="severity"):
        load_years(make_cfg(tmp_path))


def test_labelled_export_is_rejected(tmp_path, fixture_df):
    d = fixture_df.copy()
    d["road_type"] = "Single carriageway"
    _write(tmp_path, d, "collision-2099.csv")
    with pytest.raises(SchemaError, match="labelled"):
        load_years(make_cfg(tmp_path))


def test_missing_year_reported_and_strict_fails(tmp_path, fixture_df):
    _write(tmp_path, fixture_df, "collision-2099.csv")
    cfg = make_cfg(tmp_path)
    cfg.years = [2098, 2099]
    found, missing = discover_raw_files(cfg)
    assert list(found) == [2099] and missing == [2098]
    _, rep = load_years(cfg)
    assert rep["years_missing"] == [2098] and rep["years_used"] == [2099]
    with pytest.raises(FileNotFoundError):
        load_years(cfg, strict=True)


def test_test_fixture_refused_without_flag(tmp_path, fixture_df):
    _write(tmp_path, fixture_df, "collision-2099.csv")
    cfg = make_cfg(tmp_path)
    cfg.allow_test_data = False
    with pytest.raises(RuntimeError, match="TEST"):
        load_years(cfg)


def test_row_limit_is_recorded(tmp_path, fixture_df):
    _write(tmp_path, fixture_df, "collision-2099.csv")
    cfg = make_cfg(tmp_path)
    cfg.max_rows_per_year = 500
    df, rep = load_years(cfg)
    assert len(df) == 500 and rep["files"][0]["row_limit_applied"] == 500
