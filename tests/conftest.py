"""Test setup. All tests use the SYNTHETIC fixture (scripts/make_test_fixture.py) inside temp dirs.
Nothing here is real STATS19 data and nothing produced here is a reportable result."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from make_test_fixture import make_fixture  # noqa: E402
from ml.config import Settings  # noqa: E402


def make_cfg(root: Path, **kw) -> Settings:
    return Settings(root=root, years=[2099], allow_test_data=True, chunk_size=700, tuning_max_rows=1500,
                    search_n_iter=2, cv_folds=2, permutation_sample=500, permutation_repeats=1, **kw)


@pytest.fixture(scope="session")
def fixture_df():
    return make_fixture()


@pytest.fixture(scope="session")
def env(tmp_path_factory, fixture_df):
    """prepare + train on the synthetic fixture in a temp dir (test set untouched)."""
    from ml.prepare import run_prepare
    from ml.train import run_training
    root = tmp_path_factory.mktemp("roadrisk_env")
    (root / "data" / "raw").mkdir(parents=True)
    fixture_df.to_csv(root / "data" / "raw" / "dft-road-casualty-statistics-collision-2099.csv", index=False)
    cfg = make_cfg(root)
    run_prepare(cfg)
    run_training(cfg)
    return cfg
