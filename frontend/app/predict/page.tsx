"use client";

/** Prediction Studio: form built from the trained model's declared input fields. */
import { useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  AlertTriangle,
  BrainCircuit,
  Clock3,
  CloudSun,
  Flag,
  Loader2,
  Route,
  Send,
  Sparkles,
  Users,
  Zap,
} from "lucide-react";
import { api } from "@/lib/api";
import { useApi } from "@/hooks/useApi";
import type { InputField, ModelOption, PredictionPayload, PredictionResult } from "@/types";
import { ErrorState, Panel, SectionHeading, SeverityBadge, Skeleton } from "@/components/ui";
import { FadeIn } from "@/components/motion";
import { SEVERITY_COLORS } from "@/components/charts";

const DAY_LABELS: Record<string, string> = {
  "1": "Sunday", "2": "Monday", "3": "Tuesday", "4": "Wednesday",
  "5": "Thursday", "6": "Friday", "7": "Saturday",
};

interface GroupDef {
  key: string;
  title: string;
  icon: typeof Clock3;
  desc: string;
  fields: string[];
}

const GROUPS: GroupDef[] = [
  { key: "time", title: "Time", icon: Clock3, desc: "When the collision occurred", fields: ["hour", "day_of_week", "month"] },
  { key: "road", title: "Road", icon: Route, desc: "Road geometry and regulation", fields: ["road_type", "speed_limit", "first_road_class", "junction_detail", "junction_control", "trunk_road_flag", "urban_or_rural_area"] },
  { key: "environment", title: "Environment", icon: CloudSun, desc: "Light, weather and surface", fields: ["light_conditions", "weather_conditions", "road_surface_conditions", "special_conditions_at_site", "carriageway_hazards"] },
  { key: "context", title: "Collision context", icon: Users, desc: "Information available when the incident is judged", fields: ["number_of_vehicles"] },
];

function optionsFor(field: InputField): { value: number; label: string }[] {
  if (field.options && field.options.length) {
    return field.options.map((o) => {
      if (typeof o === "number") return { value: o, label: `${o}` };
      const key = String(o.value);
      const label = field.name === "day_of_week" ? (field.labels?.[key] ?? DAY_LABELS[key] ?? key) : o.label;
      return { value: o.value, label };
    });
  }
  if (field.labels) {
    // e.g. day_of_week arrives as labels {1: "Sunday"...} without an options list
    return Object.entries(field.labels).map(([v, label]) => ({ value: Number(v), label }));
  }
  return [];
}

function FieldControl({
  field,
  value,
  error,
  onChange,
}: {
  field: InputField;
  value: string;
  error?: string;
  onChange: (v: string) => void;
}) {
  const options = useMemo(() => optionsFor(field), [field]);
  const id = `f-${field.name}`;
  return (
    <div>
      <label className="field-label" htmlFor={id}>
        {field.name.replace(/_/g, " ")}
        {!field.required && <span className="ml-1 normal-case tracking-normal text-[#64748b]">(optional)</span>}
      </label>
      {field.type === "code" || field.name === "day_of_week" ? (          <select id={id} className="field-input" value={value} onChange={(e) => onChange(e.target.value)} aria-invalid={!!error}>
          <option value="">Select...</option>
          {options.map((o) => (
            <option key={o.value} value={o.value}>
              {String(o.label) === String(o.value) ? o.value : `${o.value} - ${o.label}`}
            </option>
          ))}
        </select>
      ) : (
        <input
          id={id}
          type="number"
          className="field-input"
          value={value}
          min={field.min}
          max={field.max}
          onChange={(e) => onChange(e.target.value)}
          aria-invalid={!!error}
          required={field.required}
        />
      )}
      {error && <p className="mt-1 text-xs text-[#f87171]">{error}</p>}
      {!error && field.note && <p className="mt-1 text-[11px] text-[#64748b]">{field.note}</p>}
    </div>
  );
}

function MetricCell({ group, label, value }: { group: string; label: string; value?: number }) {
  return (
    <div className="rounded-md border border-[#152036] bg-[#0b1220] p-2">
      <p className="text-[10px] uppercase tracking-wider text-[#64748b]">{group}</p>
      <p className="text-[11px] text-[#93a4c0]">{label}</p>
      <p className="mt-0.5 text-sm font-bold tabular-nums text-[#fcd34d]">
        {typeof value === "number" ? value.toFixed(4) : "-"}
      </p>
    </div>
  );
}

/** Real recorded metrics for the model the user is about to run (validation vs test kept apart). */
function ModelMetrics({ model, auto }: { model: ModelOption; auto: boolean }) {
  return (
    <div className="rounded-lg border border-[#1e2a42] bg-[#0e1626] p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="text-xs font-semibold text-[#cbd7ec]">Model: {model.name}</p>
        <span className="rounded-full border border-[#1e2a42] px-2 py-0.5 text-[10px] uppercase tracking-wider text-[#93a4c0]">
          {auto ? "Auto-selected" : "Manually selected"}
        </span>
      </div>
      <div className="mt-2.5 grid grid-cols-2 gap-2">
        <MetricCell group="Validation" label="Macro F1" value={model.validation_metrics?.macro_f1} />
        <MetricCell group="Validation" label="Balanced accuracy" value={model.validation_metrics?.balanced_accuracy} />
        <MetricCell group="Test" label="Macro F1" value={model.test_metrics?.macro_f1} />
        <MetricCell group="Test" label="Balanced accuracy" value={model.test_metrics?.balanced_accuracy} />
      </div>
      <p className="mt-2 text-[10.5px] leading-relaxed text-[#64748b]">
        {model.test_metrics
          ? "Validation metrics: held-out split during training. Test metrics: the single evaluation on the untouched test split."
          : "Validation metrics come from the held-out validation split. Test metrics exist only for the auto-selected final model (the test split is scored once)."}
      </p>
    </div>
  );
}

export default function PredictPage() {
  const info = useApi(() => api.modelInfo());
  const models = useApi(() => api.models());
  const [modelChoice, setModelChoice] = useState<string>("auto");
  const [values, setValues] = useState<Record<string, string>>({});
  const [clientErrors, setClientErrors] = useState<Record<string, string>>({});
  const [serverError, setServerError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [resultChoice, setResultChoice] = useState<string>("auto");

  const modelList = models.data?.models ?? [];
  const shownModel =
    modelChoice === "auto"
      ? (modelList.find((m) => m.id === models.data?.default_model) ?? null)
      : (modelList.find((m) => m.id === modelChoice) ?? null);

  const fields = info.data?.input_fields ?? [];
  const setField = (name: string, v: string) => {
    setValues((old) => ({ ...old, [name]: v }));
    setClientErrors((old) => ({ ...old, [name]: "" }));
  };

  function validateLocal(): PredictionPayload | null {
    const errs: Record<string, string> = {};
    for (const f of fields) {
      const raw = values[f.name];
      if (raw === undefined || raw === "") {
        if (f.required) errs[f.name] = "Required";
        continue;
      }
      const n = Number(raw);
      if (!Number.isFinite(n) || !Number.isInteger(n)) {
        errs[f.name] = "Must be a whole number";
        continue;
      }
      if (f.min !== undefined && n < f.min) errs[f.name] = `Min ${f.min}`;
      if (f.max !== undefined && n > f.max) errs[f.name] = `Max ${f.max}`;
    }
    setClientErrors(errs);
    if (Object.keys(errs).length) return null;

    const payload: Record<string, unknown> = {};
    for (const f of fields) {
      const raw = values[f.name];
      if (raw !== undefined && raw !== "") payload[f.name] = Number(raw);
    }
    return payload as unknown as PredictionPayload;
  }

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setServerError(null);
    setResult(null);
    const payload = validateLocal();
    if (!payload) return;
    setBusy(true);
    try {
      const res = await api.predict({ ...payload, model: modelChoice === "auto" ? null : modelChoice });
      setResult(res);
      setResultChoice(modelChoice);
    } catch (err) {
      setServerError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  const probRows = result
    ? Object.entries(result.probabilities).sort(
        (a, b) => ["Slight", "Serious", "Fatal"].indexOf(a[0]) - ["Slight", "Serious", "Fatal"].indexOf(b[0]),
      )
    : [];

  return (
    <div className="mx-auto max-w-7xl px-4 py-10 sm:px-6">
      <FadeIn>
        <SectionHeading
          kicker="Real-time model endpoint"
          title="Prediction Studio"
          description="Estimate the likely severity of a collision from information available at prediction time. The form is generated from the trained model's declared input fields - unsupported fields are never shown, and post-outcome information (casualties, injuries, police attendance) is excluded by design."
        />
      </FadeIn>

      {info.error && <div className="mt-6"><ErrorState message={info.error} onRetry={info.refetch} /></div>}

      <div className="mt-8 grid gap-6 lg:grid-cols-5">
        {/* ------------------------------------------------------------ FORM */}
        <form onSubmit={onSubmit} className="space-y-5 lg:col-span-3" noValidate>
          {info.loading && (
            <div className="space-y-4">
              {Array.from({ length: 4 }).map((_, i) => (
                <Skeleton key={i} className="h-40" />
              ))}
            </div>
          )}

          {/* ---------------------------------------------------- MODEL SELECTOR */}
          <Panel title="Model" subtitle="Choose which trained model scores this prediction">
            <div className="space-y-3">
              <div>
                <label className="field-label" htmlFor="model-select">
                  Model
                </label>
                <select
                  id="model-select"
                  className="field-input"
                  value={modelChoice}
                  onChange={(e) => setModelChoice(e.target.value)}
                  disabled={models.loading}
                  aria-busy={models.loading}
                >
                  <option value="auto">
                    {models.loading ? "Loading models..." : "Auto — Best Validation Model"}
                  </option>
                  {modelList.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.name}
                      {m.id === models.data?.default_model ? " (auto-selected)" : ""}
                    </option>
                  ))}
                </select>
              </div>

              {models.error && (
                <ErrorState message="Unable to load available models." onRetry={models.refetch} />
              )}

              {!models.error && shownModel && <ModelMetrics model={shownModel} auto={modelChoice === "auto"} />}

              {modelChoice === "auto" && models.data?.selection_rule && (
                <p className="text-[11px] leading-relaxed text-[#64748b]">
                  Auto selection rule: {models.data.selection_rule}
                </p>
              )}
            </div>
          </Panel>

          {GROUPS.map((g) => {
            const groupFields = fields.filter((f) => g.fields.includes(f.name));
            if (!groupFields.length) return null;
            return (
              <Panel key={g.key} title={g.title} subtitle={g.desc}>
                <div className="grid gap-4 sm:grid-cols-2">
                  {groupFields.map((f) => (
                    <FieldControl
                      key={f.name}
                      field={f}
                      value={values[f.name] ?? ""}
                      error={clientErrors[f.name]}
                      onChange={(v) => setField(f.name, v)}
                    />
                  ))}
                </div>
              </Panel>
            );
          })}

          {serverError && <ErrorState message={serverError} />}

          <button type="submit" className="btn-primary inline-flex w-full items-center justify-center gap-2 text-sm sm:w-auto" disabled={busy || info.loading}>
            {busy ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" aria-hidden /> Running model...
              </>
            ) : (
              <>
                <Send className="h-4 w-4" aria-hidden /> Predict severity
              </>
            )}
          </button>
          <p className="text-[11px] text-[#64748b]">
            Code values follow STATS19; the backend validates every value against the trained
            model&apos;s accepted options and rejects anything else (HTTP 422).
          </p>
        </form>

        {/* ------------------------------------------------------------ RESULT */}
        <div className="space-y-5 lg:col-span-2">
          <AnimatePresence mode="wait">
            {busy && (
              <motion.div key="loading" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
                <Panel title="Predicting">
                  <div className="flex flex-col items-center gap-4 py-8">
                    <motion.div
                      animate={{ rotate: 360 }}
                      transition={{ duration: 1.4, repeat: Infinity, ease: "linear" }}
                      className="rounded-full border-2 border-[#f59e0b]/30 border-t-[#f59e0b] h-10 w-10"
                      aria-hidden
                    />
                    <p className="text-sm text-[#93a4c0]">Scoring input through the trained pipeline...</p>
                  </div>
                </Panel>
              </motion.div>
            )}

            {!busy && result && (
              <motion.div key="result" initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ duration: 0.4 }} className="space-y-5">
                <Panel title="Prediction">
                  <div className="mb-4 flex flex-wrap items-center justify-between gap-2 rounded-lg border border-[#1e2a42] bg-[#0e1626] px-3 py-2">
                    <div>
                      <p className="text-[11px] uppercase tracking-widest text-[#93a4c0]">Selected model</p>
                      <p className="mt-0.5 text-sm font-bold text-[#cbd7ec]">{result.model ?? "-"}</p>
                    </div>
                    <p className="text-right text-[10.5px] text-[#64748b]">
                      {resultChoice === "auto"
                        ? "Auto — best validation model"
                        : "Manually selected for this prediction"}
                    </p>
                  </div>
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-xs uppercase tracking-widest text-[#93a4c0]">Predicted severity</p>
                      <p className="mt-2 text-3xl font-black" style={{ color: SEVERITY_COLORS[result.prediction] ?? "#e6edf7" }}>
                        {result.prediction}
                      </p>
                    </div>
                    <div className="text-right">
                      <p className="text-xs uppercase tracking-widest text-[#93a4c0]">Confidence</p>
                      <p className="mt-2 text-3xl font-black text-[#f59e0b]">
                        {(result.confidence * 100).toFixed(1)}%
                      </p>
                    </div>
                  </div>
                  <p className="mt-2 text-[11px] text-[#64748b]">{result.confidence_note}</p>

                  <div className="mt-5 space-y-3">
                    {probRows.map(([name, p]) => (
                      <div key={name}>
                        <div className="flex items-center justify-between text-xs">
                          <span className="flex items-center gap-2 font-medium">
                            <SeverityBadge severity={name} small /> {name}
                          </span>
                          <span className="tabular-nums text-[#cbd7ec]">{(p * 100).toFixed(2)}%</span>
                        </div>
                        <div className="mt-1.5 h-2.5 overflow-hidden rounded-full bg-[#1a2438]">
                          <motion.div
                            className="h-full rounded-full"
                            style={{ background: SEVERITY_COLORS[name] ?? "#64748b" }}
                            initial={{ width: 0 }}
                            animate={{ width: `${p * 100}%` }}
                            transition={{ duration: 0.7, ease: "easeOut" }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                  <p className="mt-3 text-[11px] text-[#64748b]">
                    Probabilities come directly from the trained model. {result.probabilities && "They are relative model scores, not calibrated real-world frequencies."}
                  </p>
                </Panel>

                {result.top_factors?.length > 0 && (
                  <Panel title="Key factors for this input" subtitle={result.factors_note ?? undefined}>
                    <ul className="space-y-2.5">
                      {result.top_factors.map((f, i) => (
                        <li key={i} className="flex items-start justify-between gap-3 rounded-lg border border-[#1e2a42] bg-[#0e1626] px-3 py-2.5 text-xs">
                          <span className="flex items-center gap-2">
                            <Sparkles className="h-3.5 w-3.5 text-[#f59e0b]" aria-hidden />
                            <span className="font-medium text-[#cbd7ec]">{String(f.factor ?? JSON.stringify(f))}</span>
                          </span>
                          <span className="text-right text-[#93a4c0]">
                            {f.effect ?? ""}
                            {typeof f.delta === "number" ? ` (${f.delta > 0 ? "+" : ""}${f.delta.toFixed(3)})` : ""}
                          </span>
                        </li>
                      ))}
                    </ul>
                  </Panel>
                )}

                <div className="flex items-start gap-2 rounded-xl border border-[#f59e0b]/35 bg-[#f59e0b]/8 px-4 py-3 text-xs text-[#fcd34d]">
                  <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
                  <p>{result.disclaimer}</p>
                </div>
              </motion.div>
            )}

            {!busy && !result && (
              <motion.div key="idle" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                <Panel title="Result">
                  <div className="flex flex-col items-center gap-3 py-10 text-center">
                    <BrainCircuit className="h-8 w-8 text-[#334155]" aria-hidden />
                    <p className="text-sm text-[#93a4c0]">
                      Fill the form and submit to score the trained model.
                    </p>
                    <p className="flex items-center gap-1.5 text-[11px] text-[#64748b]">
                      <Zap className="h-3.5 w-3.5" aria-hidden />
                      {shownModel
                        ? modelChoice === "auto"
                          ? `Auto model ready: ${shownModel.name}`
                          : `Selected model ready: ${shownModel.name}`
                        : info.data?.model_name
                          ? `Loaded model: ${info.data.model_name}`
                          : "Waiting for model metadata..."}
                    </p>
                    <p className="flex items-center gap-1.5 text-[11px] text-[#64748b]">
                      <Flag className="h-3.5 w-3.5" aria-hidden />
                      Predictions are served, never fabricated: the API answers 503 if artifacts
                      are missing.
                    </p>
                  </div>
                </Panel>
              </motion.div>
            )}
          </AnimatePresence>

          {info.data && (
            <Panel title="Accepted inputs" subtitle={`${info.data.n_features_raw} raw features -> ${info.data.n_features_after_encoding} encoded`}>
              <div className="flex flex-wrap gap-1.5">
                {info.data.input_fields.map((f) => (
                  <span
                    key={f.name}
                    className={`rounded-md border px-2 py-1 text-[11px] ${
                      f.required
                        ? "border-[#f59e0b]/40 bg-[#f59e0b]/10 text-[#fcd34d]"
                        : "border-[#1e2a42] bg-[#0e1626] text-[#93a4c0]"
                    }`}
                  >
                    {f.name}
                  </span>
                ))}
              </div>
            </Panel>
          )}
        </div>
      </div>
    </div>
  );
}
