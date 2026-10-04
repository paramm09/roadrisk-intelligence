// Types mirroring the FastAPI response shapes (backend/schemas.py + services).

export interface Health {
  status: string;
  model_loaded: boolean;
  model_error: string | null;
  analytics_loaded: boolean;
  analytics_error: string | null;
}

export interface SeverityItem {
  severity: string;
  count: number;
  percentage: number;
}

export interface Summary {
  total_collisions: number;
  severity_distribution: SeverityItem[];
  years: number[];
  dataset_period: string | null;
  source: string;
  /** true when the numbers were computed under active dashboard filters. */
  filtered?: boolean;
  model?: {
    name: string | null;
    training_date: string | null;
    macro_f1: number | null;
    balanced_accuracy: number | null;
    accuracy: number | null;
    macro_f1_label: string;
  };
}

export interface SeverityDistribution {
  total: number;
  items: SeverityItem[];
}

export interface BreakdownItem {
  key: number | string | null;
  label: string;
  total: number;
  slight: number;
  serious: number;
  fatal: number;
  slight_pct: number;
  serious_pct: number;
  fatal_pct: number;
}

export interface Breakdown {
  items: BreakdownItem[];
}

export interface MapPoint {
  lat: number;
  lon: number;
  total: number;
  slight: number;
  serious: number;
  fatal: number;
}

export interface MapPoints {
  points: MapPoint[];
  total_cells: number;
  truncated: boolean;
  note?: string;
}

export interface Insight {
  id: string;
  category: string;
  title: string;
  stat: string;
  numbers: Record<string, unknown>;
}

export interface Insights {
  findings: Insight[];
  rows_analysed: number;
  note: string;
}

export interface OptionOpt {
  value: number;
  label: string;
  count?: number;
}

export interface InputField {
  name: string;
  type: "int" | "code";
  required: boolean;
  min?: number;
  max?: number;
  options?: (number | OptionOpt)[];
  labels?: Record<string, string>;
  aliases?: string[];
  note?: string;
  description?: string;
}

export interface ModelInfo {
  model_name: string;
  training_date: string;
  dataset_period: string;
  target: string;
  class_names: string[];
  training_rows: number;
  validation_rows: number;
  test_rows: number;
  random_seed: number;
  best_hyperparameters: Record<string, unknown>;
  preprocessing_version?: string;
  imbalance_strategy: string;
  n_features_raw: number;
  n_features_after_encoding: number;
  features: string[];
  input_fields: InputField[];
  labels: Record<string, Record<string, string>>;
  value_options: Record<string, OptionOpt[]>;
  feature_families: Record<string, string[]>;
  years_used: number[];
  selection_rationale?: SelectionRationale | null;
  probability_note?: string;
}

export interface SelectionRationale {
  rule: string;
  baseline_macro_f1: number;
  eligibility_note?: string | null;
  scores: Record<string, number>;
  "tied_within_0.01": string[];
  highest_score_model: string;
  selected: string;
  fatal_class_warnings: Record<string, string>;
}

export interface ClassMetrics {
  precision: number;
  recall: number;
  f1: number;
  support: number;
}

export interface Metrics {
  n?: number;
  accuracy: number;
  macro_precision?: number;
  macro_recall?: number;
  macro_f1: number;
  weighted_precision?: number;
  weighted_recall?: number;
  weighted_f1: number;
  balanced_accuracy: number;
  fatal_recall?: number;
  fatal_precision?: number;
  fatal_f1?: number;
  confusion_matrix?: number[][];
  confusion_matrix_labels?: string[];
  class_wise?: Record<string, ClassMetrics>;
  display_name?: string;
  training_seconds?: number;
  inference_ms_per_1000_rows?: number;
}

export interface ErrorAnalysis {
  fatal_collisions_in_test: number;
  fatal_collisions_predicted_as: Record<string, number>;
  fatal_recall: number;
  fatal_precision: number;
  poor_fatal_performance_flag: boolean;
  predicted_fatal_total: number;
  probability_calibration?: unknown;
  calibration_note?: string;
  segment_performance?: unknown;
}

export interface ModelMetricsResponse {
  model: string | null;
  validation: Record<string, Metrics>;
  test: Metrics | null;
  selection: SelectionRationale | null;
  error_analysis: ErrorAnalysis | null;
  note: string;
}

export interface PermFamily {
  family: string;
  columns: string[];
  macro_f1_drop_mean: number;
  macro_f1_drop_std: number;
  fatal_recall_drop_mean: number;
}

export interface FeatureImportance {
  model: string;
  computed_on: string;
  scoring: string;
  permutation_importance_by_family: PermFamily[];
  native_importance: Record<string, unknown>;
  caveat: string;
}

/** One genuinely trained model from GET /api/models (artifact exists on disk). */
export interface ModelOption {
  id: string;
  name: string;
  artifact: string;
  available: boolean;
  /** true for the model the training pipeline auto-selected (served from final_model.joblib). */
  is_default: boolean;
  validation_metrics: {
    accuracy?: number;
    macro_f1?: number;
    balanced_accuracy?: number;
    fatal_recall?: number;
  } | null;
  /** Only computed for the auto-selected final model (single evaluation on the test split). */
  test_metrics: {
    accuracy?: number;
    macro_f1?: number;
    balanced_accuracy?: number;
    fatal_recall?: number;
  } | null;
}

export interface ModelsResponse {
  models: ModelOption[];
  default_model: string | null;
  default_model_name?: string | null;
  selection_rule?: string | null;
  note?: string;
}

export interface PredictionPayload {
  /** Model id from GET /api/models; null/"auto" = the pipeline's selected best-validation model. */
  model?: string | null;
  speed_limit: number;
  road_type: number;
  junction_detail: number;
  junction_control?: number | null;
  first_road_class?: number | null;
  trunk_road_flag?: number | null;
  light_conditions: number;
  weather_conditions: number;
  road_surface_conditions: number;
  special_conditions_at_site?: number | null;
  carriageway_hazards?: number | null;
  urban_or_rural_area: number;
  number_of_vehicles: number;
  hour: number;
  day_of_week: number;
  month?: number | null;
}

export interface PredictionResult {
  prediction: string;
  probabilities: Record<string, number>;
  confidence: number;
  confidence_note?: string | null;
  top_factors: { factor: string; effect?: string; delta?: number; [k: string]: unknown }[];
  factors_note?: string | null;
  model?: string;
  disclaimer?: string;
}

export interface Filters {
  year?: number | null;
  month?: number | null;
  severity?: string | null;
  road_type?: number | string | null;
  weather?: number | string | null;
  lighting?: number | string | null;
  speed_limit?: number | null;
  urban_rural?: number | string | null;
  hour?: number | null;
  weekday?: number | null;
}

/**
 * One response from GET /api/analytics/dashboard - everything the cross-filtered
 * dashboard needs. `summary` uses ALL active filters; each series excludes only its
 * own dimension (so the clicked chart keeps showing every category); `labels` maps
 * filter values to display labels taken from the real aggregations.
 */
export interface DashboardPayload {
  summary: Summary;
  severity: SeverityDistribution;
  series: Record<string, Breakdown>;
  labels: Record<string, Record<string, string>>;
}
