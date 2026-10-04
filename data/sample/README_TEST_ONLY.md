# TEST ONLY - SYNTHETIC DATA

`test_fixture.csv` is **randomly generated fake data** in the STATS19 collision-file layout
(`python scripts/make_test_fixture.py`, seed 7). It exists solely for unit and integration tests.

It must NEVER be used for real training, reported metrics, README results, dashboards,
screenshots or final model artifacts. Its `collision_index` values start with `TEST` and the
pipeline refuses to load them unless `ROADRISK_ALLOW_TEST_DATA=1` (set only by the tests).
