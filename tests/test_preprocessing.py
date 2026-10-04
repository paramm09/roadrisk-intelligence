import numpy as np
import pandas as pd

from ml.feature_config import MODEL_CATEGORICAL, MODEL_NUMERIC
from ml.feature_engineering import build_features, resolve_feature_lists
from ml.preprocessing import assign_splits, build_preprocessor, clean_codes
from tests.conftest import make_cfg


def test_special_codes_become_unknown_not_numbers():
    d = clean_codes(pd.DataFrame({"road_type": [6, -1, 9, 3], "urban_or_rural_area": [1, 3, 2, -1],
                                  "junction_detail": [0, 99, -1, 3], "speed_limit": [30, 99, -1, 60]}))
    assert d.road_type.tolist() == ["6", "unknown", "unknown", "3"]
    assert d.urban_or_rural_area.tolist() == ["1", "unknown", "2", "unknown"]
    assert d.junction_detail.tolist() == ["0", "unknown", "unknown", "3"]
    assert d.speed_limit.isna().tolist() == [False, True, True, False]


def _features(fixture_df):
    raw = fixture_df.rename(columns={"collision_severity": "accident_severity"}).copy()
    raw["hour"] = pd.to_numeric(raw["time"].str[:2])
    raw["date"] = pd.to_datetime(raw["date"], format="%d/%m/%Y")
    df, _ = build_features(raw)
    return df, *resolve_feature_lists(df)[:2]


def test_preprocessor_fits_transforms_and_handles_unseen(fixture_df):
    df, num, cat = _features(fixture_df)
    X = df[num + cat]
    prep = build_preprocessor(num, cat, scale_numeric=True).fit(X)
    out = prep.transform(X)
    assert out.shape[0] == len(X) and out.shape[1] > len(num) and not np.isnan(out).any()
    unseen = X.iloc[[0]].copy()
    unseen["road_type"] = "never_seen_code"
    unseen["speed_limit"] = np.nan
    assert prep.transform(unseen).shape[1] == out.shape[1]


def test_scaling_only_when_requested(fixture_df):
    df, num, cat = _features(fixture_df)
    with_scale = build_preprocessor(num, cat, True)
    without = build_preprocessor(num, cat, False)
    assert "scale" in dict(with_scale.transformers[0][1].steps)
    assert "scale" not in dict(without.transformers[0][1].steps)


def test_random_split_proportions_stratified_reproducible_disjoint():
    rng = np.random.default_rng(0)
    y = rng.choice(3, 10000, p=[.85, .13, .02])
    cfg = make_cfg(__import__("pathlib").Path("."))
    a, b = assign_splits(y, None, cfg), assign_splits(y, None, cfg)
    assert (a == b).all()
    shares = {s: (a == s).mean() for s in ("train", "val", "test")}
    assert abs(shares["train"] - .70) < .01 and abs(shares["val"] - .15) < .01 and abs(shares["test"] - .15) < .01
    for s in ("train", "val", "test"):
        assert abs((y[a == s] == 2).mean() - 0.02) < 0.004
    assert set(a) == {"train", "val", "test"}


def test_temporal_split_respects_time():
    dates = pd.Series(pd.date_range("2099-01-01", periods=1000))
    cfg = make_cfg(__import__("pathlib").Path("."))
    cfg.split_strategy = "temporal"
    lab = assign_splits(np.zeros(1000, dtype=int), dates.sample(frac=1, random_state=1).reset_index(drop=True), cfg)
    d = dates.sample(frac=1, random_state=1).reset_index(drop=True)
    assert d[lab == "train"].max() <= d[lab == "val"].min() and d[lab == "val"].max() <= d[lab == "test"].min()
