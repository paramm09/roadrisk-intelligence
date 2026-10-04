"use client";

/** Methodology: the full pipeline stage by stage, with the real artefacts each stage produces. */
import {
  Activity,
  ArrowDown,
  BrainCircuit,
  Database,
  Droplets,
  FlaskConical,
  Gauge,
  Layers,
  LineChart,
  Server,
  Split,
  TestTubes,
  Timer,
  Wrench,
  Webhook,
} from "lucide-react";
import { SectionHeading, Skeleton } from "@/components/ui";
import { FadeIn, Stagger, StaggerItem } from "@/components/motion";
import { api } from "@/lib/api";
import { useApi } from "@/hooks/useApi";

interface Stage {
  key: string;
  title: string;
  icon: typeof Database;
  what: string;
  why: string;
  output: string;
}

const STAGES: Stage[] = [
  {
    key: "data", title: "Data", icon: Database,
    what: "Five real STATS19 collision CSVs (2021-2025) are read chunk-by-chunk; only whitelisted columns are loaded.",
    why: "The dataset must be real and inspectable. Reading only needed columns keeps the pipeline inside a 16 GB laptop's budget.",
    output: "data/raw/*.csv -> reports/schema_report.json",
  },
  {
    key: "cleaning", title: "Cleaning", icon: Droplets,
    what: "Column aliases normalised, target validated (1/2/3 = Fatal/Serious/Slight), missing targets dropped, duplicate ids removed, DfT special codes (-1, 9, 99, unallocated) mapped to 'unknown', dates and times parsed.",
    why: "Dirty codes silently become fake categories; explicit cleaning keeps every value interpretable and reproducible.",
    output: "data/processed/roadrisk_processed.parquet, reports/data_preparation_report.json",
  },
  {
    key: "eda", title: "EDA", icon: LineChart,
    what: "Target distribution, missing-value analysis, duplicates, numerical and categorical distributions, severity cross-tabs against road/environment/time variables.",
    why: "Understand what the data actually shows before modelling; EDA runs on train+validation only, never the test split.",
    output: "reports/eda_summary.json, notebooks/01 & 02",
  },
  {
    key: "features", title: "Feature engineering", icon: Layers,
    what: "Hour, cyclical hour, month, weekday, weekend, rush hour, night, season, speed band, multi-vehicle, and grouped weather/surface/lighting/junction/road-type categories - each documented with a justification.",
    why: "Models learn better from structured, physically meaningful representations than from raw codes and timestamps.",
    output: "reports/feature_engineering_report.json",
  },
  {
    key: "leakage", title: "Leakage audit", icon: Timer,
    what: "An explicit whitelist plus documented exclusions; automated checks fail training if a target-like column, post-outcome field (casualties, police attendance) or identifier enters the feature list.",
    why: "'Will this value be known at the moment of prediction?' - casualty counts and attendance are consequences of the crash, not inputs to it.",
    output: "reports/leakage_report.json",
  },
  {
    key: "split", title: "Split", icon: Split,
    what: "Stratified 70% train / 15% validation / 15% test, random_state = 42, made BEFORE any preprocessing is fitted. The test ids are hashed and re-verified at evaluation time.",
    why: "Preprocessing fitted on all data would leak statistics; the test set must stay untouched until the very end.",
    output: "split column in the processed parquet (metadata in processed_meta.json)",
  },
  {
    key: "preprocessing", title: "Preprocessing", icon: Wrench,
    what: "ColumnTransformer: numeric -> median imputation + missing indicators (+ scaling where needed); categorical -> constant 'unknown' imputation + OneHotEncoder(handle_unknown='ignore').",
    why: "A single fitted pipeline guarantees train/serve consistency and that validation/test only pass through already-fitted transforms.",
    output: "ml/artifacts/preprocessing_pipeline.joblib",
  },
  {
    key: "balancing", title: "Balancing", icon: Split,
    what: "Class imbalance measured first, then NONE vs class_weight='balanced' compared on validation. SMOTE is not used by default; if enabled it would apply to training rows only, inside the pipeline.",
    why: "Oversampling tricks can distort minority precision; the decision is evidence-based and documented, not automatic.",
    output: "reports/balancing_report.json",
  },
  {
    key: "baseline", title: "Baseline", icon: FlaskConical,
    what: "DummyClassifier (most frequent class) plus an interpretable Logistic Regression baseline.",
    why: "A model is only useful if it beats naive guessing - macro F1 of the dummy is the bar.",
    output: "ml/artifacts/baseline_model.joblib",
  },
  {
    key: "training", title: "Model training", icon: BrainCircuit,
    what: "LogisticRegression, RandomForestClassifier and HistGradientBoostingClassifier all trained on the same split with the same preprocessing.",
    why: "Comparing linear, bagged and boosted families reveals what the data supports rather than what one algorithm prefers.",
    output: "ml/artifacts/{logistic_regression,random_forest,gradient_boosting}.joblib",
  },
  {
    key: "tuning", title: "Hyperparameter tuning", icon: Gauge,
    what: "RandomizedSearchCV, cv=3, scoring = macro F1, limited search spaces, n_jobs=1 on a stratified subsample of training rows.",
    why: "Small, honest searches beat overfitting the CV folds; n_jobs=1 keeps memory predictable on a laptop.",
    output: "reports/hyperparameter_results.json",
  },
  {
    key: "validation", title: "Validation", icon: Activity,
    what: "All candidates scored on the validation split: accuracy, macro/weighted precision-recall-F1, balanced accuracy, Fatal recall. Selection = 0.6 x macro F1 + 0.4 x balanced accuracy with tie-breaks on speed/size/interpretability.",
    why: "Accuracy alone would reward predicting 'Slight' every time; minority-class metrics drive the choice.",
    output: "reports/model_comparison.json, reports/model_selection.json",
  },
  {
    key: "final", title: "Final test", icon: TestTubes,
    what: "The selected model is refit on train+validation and evaluated EXACTLY ONCE on the untouched test split (id hash verified first), including failure analysis and calibration tables.",
    why: "One honest test number beats a dozen tuned-to-the-test numbers.",
    output: "reports/final_evaluation.json, reports/failure_analysis.json",
  },
  {
    key: "artifact", title: "Model artifact", icon: Layers,
    what: "Preprocessing + feature engineering + estimator saved as one reproducible joblib pipeline plus JSON metadata (rows, features, hyperparameters, metrics, seed, timestamp).",
    why: "What shipped is what was scored - no silent retraining, no divergence between report and deployment.",
    output: "ml/artifacts/final_model.joblib, model_metadata.json",
  },
  {
    key: "api", title: "FastAPI", icon: Server,
    what: "The artifact is loaded ONCE at startup; /api/predict validates input with Pydantic + the model's own code whitelist; analytics endpoints serve cached aggregations of the processed dataset.",
    why: "Load-once keeps latency low; strict validation means unsupported or post-outcome fields can never reach the model.",
    output: "backend/main.py + backend/services/",
  },
  {
    key: "web", title: "Web application", icon: Webhook,
    what: "Next.js + TypeScript + Tailwind + Framer Motion + Recharts front end: dashboard, prediction studio, analytics, insights, map, model performance.",
    why: "Make the real results usable for stakeholders while keeping the data - not decoration - as the focus.",
    output: "frontend/ (this application)",
  },
];

export default function MethodologyPage() {
  const summary = useApi(() => api.summary());
  const metrics = useApi(() => api.modelMetrics());

  return (
    <div className="mx-auto max-w-7xl px-4 py-10 sm:px-6">
      <FadeIn>
        <SectionHeading
          kicker="End-to-end ML lifecycle"
          title="Methodology"
          description="Every stage the pipeline actually executed to produce this application, in order, with the artefact it writes. Nothing here is aspirational: the reports listed exist on disk, generated from the real data."
        />
      </FadeIn>

      <div className="mt-6 grid gap-3 sm:grid-cols-3">
        {[
          ["Rows processed", summary.data ? summary.data.total_collisions.toLocaleString() : null],
          ["Dataset period", summary.data?.dataset_period ?? null],
          ["Final model", metrics.data?.model ?? null],
        ].map(([label, value]) => (
          <div key={String(label)} className="card p-4">
            <p className="text-xs uppercase tracking-widest text-[#93a4c0]">{label}</p>
            <p className="mt-1.5 text-lg font-bold text-[#f59e0b]">
              {summary.loading || metrics.loading ? <Skeleton className="h-6 w-32" /> : (value ?? "-")}
            </p>
          </div>
        ))}
      </div>

      {/* ---------------------------------------------------------------- flow */}
      <div className="relative mt-10">
        <div className="absolute left-[19px] top-0 hidden h-full w-px bg-gradient-to-b from-[#f59e0b]/60 via-[#1e2a42] to-transparent sm:block" aria-hidden />
        <Stagger className="space-y-5" gap={0.06}>
          {STAGES.map((s, i) => (
            <StaggerItem key={s.key}>
              <div className="relative sm:pl-16">
                <div className="absolute left-0 top-1 hidden h-10 w-10 items-center justify-center rounded-full border border-[#f59e0b]/50 bg-[#111a2b] sm:flex" aria-hidden>
                  <s.icon className="h-4.5 w-4.5 text-[#f59e0b]" />
                </div>
                <div className="card card-hover p-5">
                  <div className="flex flex-wrap items-center gap-3">
                    <span className="rounded-md bg-[#f59e0b]/12 px-2 py-0.5 text-[11px] font-bold text-[#fcd34d]">
                      {String(i + 1).padStart(2, "0")}
                    </span>
                    <h2 className="text-base font-bold uppercase tracking-wide">{s.title}</h2>
                    <s.icon className="h-4 w-4 text-[#f59e0b] sm:hidden" aria-hidden />
                  </div>
                  <p className="mt-3 text-sm leading-relaxed text-[#cbd7ec]">{s.what}</p>
                  <p className="mt-2 text-xs leading-relaxed text-[#93a4c0]">
                    <span className="font-semibold text-[#f59e0b]">Why: </span>
                    {s.why}
                  </p>
                  <p className="mt-3 overflow-x-auto rounded-md bg-[#0e1626] px-3 py-2 font-mono text-[11px] text-[#7dd3fc]">
                    {s.output}
                  </p>
                  {i < STAGES.length - 1 && (
                    <div className="mt-4 flex justify-center sm:hidden" aria-hidden>
                      <ArrowDown className="h-4 w-4 text-[#334155]" />
                    </div>
                  )}
                </div>
              </div>
            </StaggerItem>
          ))}
        </Stagger>
      </div>

      <FadeIn>
        <div className="mt-10 card p-5 text-xs leading-relaxed text-[#93a4c0]">
          <p className="text-sm font-bold text-[#cbd7ec]">Reproduce it yourself</p>
          <pre className="mt-3 overflow-x-auto rounded-md bg-[#0a1120] p-4 font-mono text-[11.5px] leading-relaxed text-[#7dd3fc]">{`python scripts/inspect_data.py      # schema report
python scripts/prepare_data.py      # clean + engineer + split
python scripts/run_eda.py           # exploratory summary
python scripts/train_models.py      # leakage audit, tuning, selection, artifact
python scripts/evaluate_models.py   # single final test evaluation
uvicorn backend.main:app --reload   # API
cd frontend && npm run dev          # this app`}</pre>
        </div>
      </FadeIn>
    </div>
  );
}
