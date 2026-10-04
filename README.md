# ROADRISK Intelligence

**Machine Learning-Based Road Collision Severity Prediction & Road Safety Analytics**
(UK Department for Transport STATS19 data — multi-class classification: Slight / Serious / Fatal)

> ## Status — pipeline executed end-to-end on the real data
> Every number below was produced by running the scripts in this repository on the five real
> STATS19 collision CSVs in `data/raw/` and is written to `reports/` and `ml/artifacts/`.
> Nothing in this README, the API, or the frontend is hardcoded, fabricated, or taken from the
> synthetic test fixture (which lives only in `data/sample/` + `tests/` and is refused by the
> training loader).
>
> | Stage | Result |
> |---|---|
> | Dataset | 513,801 real collisions, 2021–2025 |
> | Leakage audit | PASSED (29 whitelisted features) |
> | Models trained | Dummy baseline, Logistic Regression, Random Forest, HistGradientBoosting |
> | Selected model | **Logistic Regression** (validation score rule, see §7) |
> | Final test (once, untouched) | macro F1 **0.3455**, balanced accuracy **0.5054**, Fatal recall **0.6531** |
> | Tests | **87 passed** (pytest) |
> | Backend | FastAPI, all endpoints verified live |
> | Frontend | Next.js 16 — `npm run build` ✓, `eslint` ✓, headless-Chrome E2E ✓ |

---

## 1. Problem statement

Road collision severity is highly imbalanced: most collisions are slight, but a small share are
fatal or serious. The application predicts the severity of a *new* reported collision using only
information that would be available at the moment the incident is judged — road, environment,
time, vehicles involved — and provides an analytics layer that describes what five years of real
collision data actually show.

Two products in one repository:

1. **ML pipeline** (Python, scikit-learn) — inspect → clean → EDA → feature engineering →
   leakage audit → split → preprocess → imbalance handling → baselines → models → tuning →
   validation → single final test evaluation → versioned artifact.
2. **Web application** — FastAPI serving the artifact + cached real aggregations, and a
   Next.js dashboard / prediction studio / analytics / insights / map / model-performance UI.

## 2. Dataset

| Item | Value |
|---|---|
| Source | UK DfT **STATS19** road safety open data, *collision* files (data.gov.uk) |
| Licence | Open Government Licence v3.0 |
| Period | **2021, 2022, 2023, 2024, 2025** (configurable, see §11) |
| Files | `data/raw/dft-road-casualty-statistics-collision-{year}.csv` |
| Rows | 101,087 + 106,004 + 104,258 + 100,927 + 101,525 = **513,801** |
| Target | `collision_severity` (the real column name in these files), codes verified at runtime: 1 = Fatal, 2 = Serious, 3 = Slight |
| Target distribution | Slight 389,435 (75.795%) · Serious 116,813 (22.735%) · Fatal 7,553 (1.470%) |
| Duplicates | 0 full-row, 0 duplicate collision_index |
| Schema | Identical across all five years (checked; `column_differences_between_years` in the schema report) |

`python scripts/inspect_data.py` → `reports/schema_report.json` (columns, dtypes, row count per
year, total row count, missing %, unique counts, example values, duplicate counts, target values
and distribution, cross-year differences). The casualty and vehicle files are **never loaded** —
they contain outcomes.

## 3. Data preparation

`python scripts/prepare_data.py` → `data/processed/roadrisk_processed.parquet`
(an identical copy is kept under the loader's historical name `stats19_model_data.parquet`),
plus `reports/data_preparation_report.json`, `reports/target_distribution.json`,
`reports/preprocessing_report.json`, `reports/feature_engineering_report.json`.

Steps: load configured years (chunked, only whitelisted columns) → normalise column aliases →
validate target (drop missing/unexpected) → drop duplicate ids → map DfT special codes
(`-1`, `9`, `99`, unallocated → category `unknown`; speed limit outside {20..70} → missing) →
parse date/time → engineer features → stratified split.

Result: **513,801 rows × 29 model features (11 numeric + 18 categorical → 118 encoded columns)**,
split 359,660 / 77,070 / 77,071.

## 4. Leakage prevention

The question asked of every column: *"Will this value be known at the moment the system is
supposed to make the prediction?"* Source of truth: `ml/feature_config.py` (whitelist +
documented exclusions), audited by `ml/leakage.py` before every training run →
`reports/leakage_report.json`.

| Feature | Status | Reason |
|---|---|---|
| `collision_severity` | EXCLUDED | Target variable |
| `number_of_casualties` | EXCLUDED | Only known after the outcome; mechanically linked to severity |
| `casualty_severity` (casualty file) | EXCLUDED | The collision severity *is* the worst casualty severity |
| `did_police_officer_attend_scene_of_accident` | EXCLUDED | Post-event response that encodes the outcome |
| `enhanced_severity_collision`, `collision_adjusted_severity_*`, `collision_injury_based` | EXCLUDED | Target-derived fields present in the raw files |
| identifiers (`collision_index`, refs, road numbers, LSOA, LA, police force) | EXCLUDED | No predictive meaning / high cardinality / reporting-practice bias |
| `latitude` / `longitude` | EXCLUDED from model | Location memorisation; rounded copies kept only for the historical map |

The audit fails training (non-zero exit) if a target-like name, post-outcome pattern, or the
target itself enters the feature list; it also reports single-feature association vs the target.

## 5. Feature engineering

`reports/feature_engineering_report.json` documents every engineered feature with its original
column, transformation and justification. Highlights: `hour` (+ cyclical sin/cos), `month`,
`weekday`, `is_weekend`, `is_rush_hour`, `is_night`, `season`, `speed_band`, `multi_vehicle`,
and grouped `weather_group`, `surface_group`, `lighting_group`, `junction_group`,
`road_type_group`. Features are skipped (never invented) when a source column is absent — the
two `pedestrian_crossing_*` columns are named `*_historic` in these files and are therefore not
loaded; the schema report records this.

## 6. EDA

`python scripts/run_eda.py` → `reports/eda_summary.json`, notebooks
`notebooks/01_data_understanding.ipynb` and `notebooks/02_eda.ipynb`
(regenerate with `python scripts/build_notebooks.py`). EDA runs on **train+validation only** —
the test split is never touched. Covers target distribution, missing values, duplicates,
numeric/categorical distributions, and severity cross-tabs against road type, speed limit,
lighting, weather, surface, urban/rural, junction, hour, weekday, month, year.

## 7. Training, validation and model selection

`python scripts/train_models.py` → artifacts + `reports/leakage_report.json`,
`balancing_report.json`, `hyperparameter_results.json`, `model_comparison.json`,
`model_selection.json`.

* **Split:** stratified 70/15/15, `random_state = 42`, made *before* preprocessing is fitted.
  Test ids are SHA-256 hashed and re-verified at evaluation time.
* **Preprocessing:** `ColumnTransformer` — numeric: median imputation + missing indicators
  (+ scaling for LR); categorical: `"unknown"` imputation + `OneHotEncoder(handle_unknown="ignore")`
  → `ml/artifacts/preprocessing_pipeline.joblib`.
* **Imbalance (measured first, `reports/balancing_report.json`):** logistic regression on the
  tuning subsample — no balancing: macro F1 0.2955, Fatal recall **0.0000**;
  `class_weight="balanced"`: macro F1 0.3445, Fatal recall **0.6443** → **CLASS_WEIGHT selected**.
  SMOTE not used (justified: class weights already recover minority recall; no resampling of
  validation/test ever happens).
* **Models:** DummyClassifier (baseline), LogisticRegression, RandomForestClassifier,
  HistGradientBoostingClassifier — each non-baseline tuned with `RandomizedSearchCV(cv=3,
  scoring=f1_macro, n_iter=8, n_jobs=1)` on ≤100k stratified training rows.

**Validation results (77,070 rows) — `reports/model_comparison.json`:**

| Model | Accuracy | Macro F1 | Balanced acc. | Fatal recall | Fatal F1 | Selection score |
|---|---:|---:|---:|---:|---:|---:|
| Dummy baseline | 0.7579 | 0.2874 | 0.3333 | 0.0000 | 0.0000 | — (ineligible) |
| **Logistic Regression** | 0.5046 | 0.3456 | 0.5077 | 0.6611 | 0.0748 | 0.4104 |
| Random Forest | 0.6001 | 0.3889 | 0.4452 | 0.2621 | 0.0884 | 0.4114 |
| HistGradientBoosting | 0.5198 | 0.3536 | 0.5019 | 0.6046 | 0.0784 | 0.4129 |

Tuned hyperparameters (`reports/hyperparameter_results.json`): LR `C=0.01`;
RF `n_estimators=100, max_depth=18, min_samples_leaf=10, max_features=0.3`;
HGB `learning_rate=0.1, max_iter=200, max_leaf_nodes=63, l2_regularization=0.1`.

**Selection rule (not “highest accuracy”):** score = 0.6 × macro F1 + 0.4 × balanced accuracy
on validation; candidates within 0.01 are tied → faster inference, then smaller model, then more
interpretable wins. All three real models tied (0.4104–0.4129), so **Logistic Regression**
(4.8 ms/1000 rows, 6 KB, coefficients interpretable) was selected — see
`reports/model_selection.json`. Accuracy is deliberately ignored as the selector: the
majority-class baseline reaches 0.7579 by predicting “Slight” every time.

## 8. Final test evaluation (executed exactly once)

`python scripts/evaluate_models.py` → `reports/final_evaluation.json`,
`reports/failure_analysis.json`, `reports/feature_importance.json`, `reports/model_report.md`.
The script verifies the test-id hash first (“hash verified: test split identical…”).

**Test metrics (77,071 untouched rows), final model = Logistic Regression:**

| Metric | Value |
|---|---:|
| Accuracy | 0.5057 |
| Macro precision | 0.3840 |
| Macro recall (= balanced accuracy) | 0.5054 |
| **Macro F1** | **0.3455** |
| Weighted precision | 0.6938 |
| Weighted recall | 0.5057 |
| Weighted F1 | 0.5767 |
| Balanced accuracy | 0.5054 |

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Slight | 0.8299 | 0.5651 | 0.6724 | 58,416 |
| Serious | 0.2827 | 0.2979 | 0.2901 | 17,522 |
| **Fatal** | **0.0393** | **0.6531** | **0.0741** | 1,133 |

Confusion matrix (rows = true, cols = predicted; order Slight, Serious, Fatal):

```
[[33011, 13012, 12393],
 [ 6607,  5220,  5695],
 [  161,   232,   740]]
```

**Fatal-class performance is flagged POOR (Fatal F1 < 0.30) and reported as measured:**
the class-weighted model predicted “Fatal” 18,828 times for 1,133 true fatal collisions —
high recall, very low precision. Fatal predictions must not be treated as reliable identifiers
of fatal collisions; this is stated on the Model page and in every report
(`poor_fatal_performance_flag: true`).

**Permutation importance (macro-F1 drop, family level, top 5):** Vehicles involved 0.0212 ·
Road class 0.0076 · Special conditions / hazards 0.0040 · Speed limit 0.0034 ·
Road type 0.0016. Caveat recorded in the report: *associations, not causes*.

## 9. Architecture

```
data/raw/*.csv  (real STATS19, 2021-2025)
   │  scripts/inspect_data.py  -> reports/schema_report.json
   │  scripts/prepare_data.py  -> data/processed/roadrisk_processed.parquet (+ reports)
   │  scripts/run_eda.py       -> reports/eda_summary.json
   │  scripts/train_models.py  -> ml/artifacts/*.joblib + tuning/selection reports
   │  scripts/evaluate_models.py -> final_evaluation / failure_analysis / feature_importance
   ▼
backend/  FastAPI  (model + analytics loaded once at startup)
   ├── GET  /health                         service status (model/analytics loaded?)
   ├── GET  /api/model/info                 model identity + accepted input fields
   ├── GET  /api/model/metrics              validation (all models) + final test metrics
   ├── GET  /api/model/feature-importance   reports/feature_importance.json
   ├── GET  /api/models                     available trained models + real validation/test metrics
   ├── GET  /api/analytics/dashboard        Tableau-style cross-filtered dashboard: KPI summary
   │         + every chart in ONE response; accepts all filters incl. ?month=&hour=&weekday=;
   │         summary uses ALL filters, each series excludes only its own dimension
   ├── GET  /api/analytics/summary          totals, severity %, years, model headline
   ├── GET  /api/analytics/severity-distribution
   ├── GET  /api/analytics/by-year | by-road-type | by-weather | by-lighting
   ├── GET  /api/analytics/by-speed-limit   (+ by-hour, by-weekday, by-month,
   │         by-urban-rural, by-junction, map-points, /api/insights)
   │         all accept filters: ?year=&severity=&road_type=&weather=&lighting=
   │                             &speed_limit=&urban_rural=
   ├── POST /api/predict                    Pydantic + model whitelist validation
   │         optional "model": "random_forest" | ... (null/"auto" = selected model;
   │         unknown id -> HTTP 400, no silent fallback, no retraining)
   └── POST /api/predict/batch              (≤500 records)
   ▼
frontend/ Next.js 16 + TypeScript + Tailwind + Framer Motion + Recharts + Lucide + Leaflet
   /  /dashboard  /predict  /analytics  /insights  /map  /model  /methodology  /about
```

## 10. Setup

### 10.1 Python environment (Windows, Python 3.13)

```powershell
cd roadrisk-intelligence
py -3.13 -m venv .venv          # normal (not free-threaded) interpreter
.venv\Scripts\activate
pip install -r requirements.txt # binary wheels only (--only-binary=:all:), no GCC needed
pytest                          # 87 tests, runs on the synthetic test-only fixture
```

Verified versions: Python 3.13.3, numpy 2.4.6, pandas 3.0.3, scipy 1.17.1,
scikit-learn 1.8.0, joblib 1.6.0, pyarrow 25.0.1, fastapi, uvicorn, pydantic, pytest 8.4.2,
httpx2 (TestClient).

### 10.2 Run the ML pipeline

```bash
python scripts/run_pipeline.py        # inspect -> prepare -> eda -> train -> evaluate
# or step by step:
python scripts/inspect_data.py
python scripts/prepare_data.py
python scripts/run_eda.py
python scripts/train_models.py
python scripts/evaluate_models.py
```

Each script prints progress and exits non-zero with a clear `ERROR:` if something is wrong —
a stage can never be silently skipped.

### 10.3 Run the application

```bash
# terminal 1 — API (from the project root)
uvicorn backend.main:app --reload          # http://127.0.0.1:8000  (docs at /docs)

# terminal 2 — frontend
cd frontend
npm install
npm run dev                                # http://localhost:3000
# production:  npm run build && npm start
```

The API loads `final_model.joblib` and the processed dataset **once at startup**. If artifacts
are missing it still starts: `/health` reports `model_loaded: false` and prediction endpoints
answer **503** with instructions — it never invents a model or a number.

## 11. Configuration (no hardcoding)

| Setting | How |
|---|---|
| Years | `python scripts/prepare_data.py --years 2020-2024` or env `ROADRISK_YEARS=2019,2021,2023` |
| Seed / split | `ROADRISK_SEED` (42), `ROADRISK_SPLIT=random|temporal` |
| Imbalance | `ROADRISK_IMBALANCE=NONE|CLASS_WEIGHT|SMOTE` (default CLASS_WEIGHT) |
| Tuning budget | `ROADRISK_SEARCH_ITER` (8), `ROADRISK_CV_FOLDS` (3), `ROADRISK_N_JOBS` (1), `ROADRISK_TUNING_ROWS` (100000) |
| API origin | `ROADRISK_CORS_ORIGINS` (default localhost:3000) |
| Frontend API base | `NEXT_PUBLIC_API_BASE` (default `http://127.0.0.1:8000`) |

Use the **same** years for every step. Missing years are reported loudly
(`years_requested` / `years_used` / `years_missing` in reports and model metadata; `--strict`
makes it fatal).

## 12. Testing

`pytest` — **87 tests passing**, covering: config/schema handling, target validation, feature
engineering, preprocessing & splits, leakage detection, the prepare→train→evaluate flow
(hash-verified test split), model loading, prediction shape/determinism, invalid-input
rejection, and the full API surface via FastAPI's TestClient:

* `GET /health` with and without artifacts (503 honesty check)
* `GET /api/model/info` — asserts **no post-outcome field is an accepted input**
* `GET /api/model/metrics`, `/api/model/feature-importance`
* all `GET /api/analytics/*` — counts asserted equal to the underlying dataframe
* `POST /api/predict` valid (probabilities compared against a direct `predict_proba` call)
  and invalid cases: out-of-range hour/weekday, unknown STATS19 code, missing required field,
  **`number_of_casualties` rejected**, unknown field rejected
* `POST /api/predict/batch` + empty-batch rejection

Tests run on a synthetic fixture (`data/sample/test_fixture.csv`, rows tagged `TEST`) inside
temporary folders — the loader refuses those rows for real training, and the fixture can never
reach production routes.

Frontend verification: `npm run lint` (clean), `npm run build` (clean), and
`node e2e-check.cjs` (headless Chrome: all 9 routes render with zero console/page errors,
dashboard shows the real 513,801 total, a prediction is executed through the form with
probabilities summing to 100%, and invalid input is rejected).

## 13. Memory & performance

Designed for a ~16 GB laptop: chunked CSV reads with column pruning, `float32` dtypes, parquet
storage, test rows dropped from memory during training, models saved/freed one at a time,
`n_jobs=1`, tuning on a stratified subsample, backend loads the parquet once (selected columns)
and memoises aggregations — no CSV is read per request. No deep learning.

## 14. Responsible use

Model output is an analytical prediction based on historical reported collision data and should
not be treated as a definitive determination of collision severity or causation. Not for
insurance, enforcement, or liability decisions about individuals. The map is a **historical
collision record**, not a prediction of future accidents. Probabilities are relative model
scores (class-weighted training inflates minority classes), not calibrated frequencies.

## 15. Limitations

* Fatal-class precision is very low (0.039) — flagged, not hidden; Fatal F1 = 0.074.
* Police-reported collisions only; reporting practice varies by force and year.
* Associations are not causal; no exposure (traffic volume) adjustment.
* Collision-level features only — vehicle/driver and casualty files are excluded by design.
* Descriptive statistics reflect reporting and exposure, not only risk.

## 16. Future improvements

XGBoost/LightGBM comparison, probability calibration (and reporting calibrated probabilities),
vehicle-file features (driver age/type — joined without outcome fields), spatial grid features,
temporal validation across years, threshold tuning for the Fatal class with stakeholder input.

## 17. Data attribution

Contains public sector information licensed under the Open Government Licence v3.0
(DfT STATS19). Confirm terms on the DfT page at your download date.

## 18. Academic AI usage disclosure

_Fill in per your course rules._ This project's code was developed with the assistance of an AI
model. All reported results come from running the pipeline in this repository on the real data.
