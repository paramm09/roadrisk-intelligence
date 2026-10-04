import pandas as pd
import pytest

from ml.data_loader import SchemaError
from ml.data_quality import validate_target


def _df(slight, serious, fatal, extra=()):
    codes = [3] * slight + [2] * serious + [1] * fatal + list(extra)
    return pd.DataFrame({"accident_severity": pd.Series(codes, dtype="float32")})


def test_mapping_and_distribution():
    out, rep = validate_target(_df(800, 150, 50))
    assert rep["class_counts"] == {"Slight": 800, "Serious": 150, "Fatal": 50}
    assert rep["frequency_sanity_check_passed"] is True
    assert set(out["severity"]) == {"Slight", "Serious", "Fatal"}
    assert out.loc[out["severity"] == "Fatal", "target"].eq(2).all()
    assert rep["imbalance_ratio_majority_to_minority"] == 16.0


def test_rare_bad_values_dropped_and_reported():
    out, rep = validate_target(_df(1000, 150, 50, extra=[9]))
    assert rep["n_unexpected_codes"] == 1 and rep["n_rows_after"] == 1200 and len(out) == 1200


def test_many_bad_values_fail():
    with pytest.raises(SchemaError):
        validate_target(_df(500, 100, 50, extra=[9] * 50))


def test_inverted_frequencies_warn():
    _, rep = validate_target(_df(50, 150, 800))
    assert rep["frequency_sanity_check_passed"] is False and "WARNING" in rep["sanity_check_note"]
