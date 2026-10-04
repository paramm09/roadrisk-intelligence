"use client";

/** Model performance: comparison, test metrics, confusion matrix, importance, training info. */
import { AlertTriangle, CheckCircle2, Cpu, Database, Info, Timer } from "lucide-react";
import { api } from "@/lib/api";
import { useApi } from "@/hooks/useApi";
import type { FeatureImportance, ModelInfo, ModelMetricsResponse, ModelsResponse } from "@/types";
import { ErrorState, Panel, SectionHeading, Skeleton, StatCard } from "@/components/ui";
import { FadeIn, Stagger, StaggerItem } from "@/components/motion";
import { HorizontalBars } from "@/components/charts";

const MODEL_ORDER = ["baseline", "logistic_regression", "random_forest", "gradient_boosting"];

const fmt = (v: number | undefined | null, d = 4) => (typeof v === "number" ? v.toFixed(d) : "-");

function ConfusionMatrix({ matrix, labels }: { matrix: number[][]; labels: string[] }) {
  const max = Math.max(...matrix.flat(), 1);
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[420px] border-collapse text-sm" aria-label="Confusion matrix (rows = true class, columns = predicted)">
        <thead>
          <tr>
            <th className="p-2 text-left text-xs font-semibold uppercase tracking-wider text-[#93a4c0]">true \ predicted</th>
            {labels.map((l) => (
              <th key={l} className="p-2 text-center text-xs font-semibold uppercase tracking-wider text-[#93a4c0]">{l}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={labels[i]}>
              <th className="p-2 text-left text-xs font-semibold uppercase tracking-wider text-[#93a4c0]">{labels[i]}</th>
              {row.map((v, j) => {
                const ratio = v / max;
                const diagonal = i === j;
                return (
                  <td
                    key={j}
                    className="border border-[#1e2a42] p-3 text-center tabular-nums"
                    style={{
                      background: diagonal
                        ? `rgba(245, 158, 11, ${0.08 + 0.5 * ratio})`
                        : `rgba(148, 163, 184, ${0.03 + 0.28 * ratio})`,
                      fontWeight: diagonal ? 700 : 500,
                    }}
                  >
                    {v.toLocaleString()}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function ModelPage() {
  const metrics = useApi<ModelMetricsResponse>(() => api.modelMetrics());
  const info = useApi<ModelInfo>(() => api.modelInfo());
  const importance = useApi<FeatureImportance>(() => api.featureImportance());
  const modelList = useApi<ModelsResponse>(() => api.models());

  const testById = new Map((modelList.data?.models ?? []).map((m) => [m.id, m.test_metrics]));

  const m = metrics.data;
  const test = m?.test ?? null;
  const classes = test?.class_wise ? Object.keys(test.class_wise) : ["Slight", "Serious", "Fatal"];

  const permData = (importance.data?.permutation_importance_by_family ?? [])
    .slice()
    .sort((a, b) => b.macro_f1_drop_mean - a.macro_f1_drop_mean)
    .map((r) => ({ label: r.family, value: r.macro_f1_drop_mean }));

  const coefSource = importance.data?.native_importance as
    | { coefficients_by_class?: Record<string, { feature: string; coef: number }[]> }
    | undefined;
  const fatalCoefs = (coefSource?.coefficients_by_class?.["2"] ?? []).slice(0, 12).map((c) => ({
    label: c.feature.replace(/^cat__|^num__/, ""),
    value: c.coef,
  }));

  return (
    <div className="mx-auto max-w-7xl px-4 py-10 sm:px-6">
      <FadeIn>
        <SectionHeading
          kicker="Presentation-ready"
          title="Model Performance"
          description="Validation comparison of every trained candidate, the single final evaluation on the untouched test split, and real feature-importance computations. Poor minority-class results are shown as measured, never hidden."
        />
      </FadeIn>

      {(metrics.error || info.error) && (
        <div className="mt-6 space-y-3">
          {metrics.error && <ErrorState message={metrics.error} onRetry={metrics.refetch} />}
          {info.error && <ErrorState message={info.error} onRetry={info.refetch} />}
        </div>
      )}

      {/* ---------------------------------------------------------- final model */}
      <div className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard label="Final model" icon={Cpu} value={info.data?.model_name ?? "-"} loading={info.loading} sub={info.data ? `Trained ${info.data.training_date.slice(0, 10)}` : undefined} />
        <StatCard label="Test macro F1" icon={CheckCircle2} accent="#38bdf8" value={test?.macro_f1 ?? null} decimals={4} loading={metrics.loading} sub="Untouched test split" />
        <StatCard label="Balanced accuracy" icon={CheckCircle2} accent="#38bdf8" value={test?.balanced_accuracy ?? null} decimals={4} loading={metrics.loading} sub="Mean per-class recall" />
        <StatCard label="Fatal recall" icon={AlertTriangle} accent="#ef4444" value={test?.fatal_recall ?? null} decimals={4} loading={metrics.loading} sub="Share of true Fatal collisions found" />
      </div>

      {m?.error_analysis?.poor_fatal_performance_flag && (
        <div className="mt-5 flex items-start gap-3 rounded-xl border border-[#ef4444]/40 bg-[#ef4444]/8 px-4 py-3.5 text-sm text-[#fca5a5]">
          <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0" aria-hidden />
          <p>
            <strong>Poor Fatal-class performance flag:</strong> Fatal F1 is below 0.30 on the test
            set (Fatal precision {fmt(m?.error_analysis?.fatal_precision, 3)}, recall{" "}
            {fmt(m?.error_analysis?.fatal_recall, 3)}). The model over-predicts Fatal under
            class-weighted training; Fatal predictions must never be treated as reliable
            identifications of fatal collisions.
          </p>
        </div>
      )}

      {/* ---------------------------------------------------------- comparison */}
      <div className="mt-6 grid gap-5 lg:grid-cols-3">
        <Panel title="Model comparison (validation + test)" subtitle="Selection score = 0.6 x macro F1 + 0.4 x balanced accuracy on VALIDATION metrics" className="lg:col-span-2">
          {metrics.loading ? (
            <Skeleton className="h-56 w-full" />
          ) : m ? (
            <div className="overflow-x-auto">
              <table className="w-full min-w-[720px] text-sm">
                <thead>
                  <tr className="border-b border-[#1e2a42] text-left text-xs uppercase tracking-wider text-[#93a4c0]">
                    <th className="p-2">Model</th>
                    <th className="p-2 text-right">Validation macro F1</th>
                    <th className="p-2 text-right">Validation bal. acc.</th>
                    <th className="p-2 text-right">Validation fatal recall</th>
                    <th className="p-2 text-right">Test macro F1</th>
                    <th className="p-2 text-right">Test bal. acc.</th>
                    <th className="p-2 text-right">Score</th>
                    <th className="p-2 text-center">Auto-selected</th>
                  </tr>
                </thead>
                <tbody>
                  {MODEL_ORDER.filter((k) => m.validation?.[k]).map((k) => {
                    const v = m.validation[k];
                    const selected = m.selection?.selected === k;
                    const testM = testById.get(k);
                    return (
                      <tr key={k} className={`border-b border-[#152036] ${selected ? "bg-[#f59e0b]/8" : ""}`}>
                        <td className="p-2 font-semibold">{v.display_name ?? k}</td>
                        <td className="p-2 text-right tabular-nums">{fmt(v.macro_f1)}</td>
                        <td className="p-2 text-right tabular-nums">{fmt(v.balanced_accuracy)}</td>
                        <td className="p-2 text-right tabular-nums">{fmt(v.fatal_recall, 3)}</td>
                        <td className="p-2 text-right tabular-nums">{fmt(testM?.macro_f1)}</td>
                        <td className="p-2 text-right tabular-nums">{fmt(testM?.balanced_accuracy)}</td>
                        <td className="p-2 text-right tabular-nums">{fmt(m.selection?.scores?.[k], 4)}</td>
                        <td className="p-2 text-center">
                          {selected ? (
                            <span className="inline-flex items-center gap-1 rounded-full bg-[#f59e0b]/20 px-2 py-0.5 text-[11px] font-bold text-[#fcd34d]">YES</span>
                          ) : (
                            <span className="text-[#475569]">-</span>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              <p className="mt-3 text-xs text-[#93a4c0]">
                Validation metrics: every candidate on the held-out validation split. Test metrics are
                computed once, for the auto-selected final model only, so the other candidates show a
                dash rather than an invented number.
              </p>
              {m.selection?.rule && (
                <p className="mt-3 flex items-start gap-2 text-xs text-[#93a4c0]">
                  <Info className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden />
                  {m.selection.rule}
                </p>
              )}
            </div>
          ) : (
            <ErrorState message="No validation metrics available." />
          )}
        </Panel>

        <Panel title="Training information">
          {info.loading ? (
            <Skeleton className="h-56 w-full" />
          ) : info.data ? (
            <dl className="space-y-3 text-sm">
              {[
                ["Model", info.data.model_name],
                ["Training rows", info.data.training_rows.toLocaleString()],
                ["Validation rows", info.data.validation_rows.toLocaleString()],
                ["Test rows", info.data.test_rows.toLocaleString()],
                ["Raw features", `${info.data.n_features_raw}`],
                ["After encoding", `${info.data.n_features_after_encoding}`],
                ["Imbalance strategy", info.data.imbalance_strategy],
                ["Random seed", `${info.data.random_seed}`],
                ["Dataset period", info.data.dataset_period],
              ].map(([k, v]) => (
                <div key={k} className="flex items-center justify-between gap-3 border-b border-[#152036] pb-2">
                  <dt className="text-xs uppercase tracking-wider text-[#93a4c0]">{k}</dt>
                  <dd className="text-right font-semibold tabular-nums">{v}</dd>
                </div>
              ))}
              {info.data.best_hyperparameters && (
                <div>
                  <p className="text-xs uppercase tracking-wider text-[#93a4c0]">Tuned hyperparameters</p>
                  <pre className="mt-1.5 overflow-x-auto rounded-lg bg-[#0e1626] p-3 text-[11px] leading-relaxed text-[#cbd7ec]">
                    {JSON.stringify(info.data.best_hyperparameters, null, 2)}
                  </pre>
                </div>
              )}
            </dl>
          ) : (
            <ErrorState message={info.error ?? "No metadata."} onRetry={info.refetch} />
          )}
        </Panel>
      </div>

      {/* ---------------------------------------------------------- test metrics */}
      <div className="mt-6 grid gap-5 lg:grid-cols-2">
        <Panel title="Final test metrics" subtitle="Computed once on the untouched test split">
          {metrics.loading ? (
            <Skeleton className="h-64 w-full" />
          ) : test ? (
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
              {[
                ["Accuracy", test.accuracy],
                ["Macro precision", test.macro_precision],
                ["Macro recall", test.macro_recall],
                ["Macro F1", test.macro_f1],
                ["Weighted precision", test.weighted_precision],
                ["Weighted recall", test.weighted_recall],
                ["Weighted F1", test.weighted_f1],
                ["Balanced accuracy", test.balanced_accuracy],
                ["Fatal recall", test.fatal_recall],
              ].map(([label, v]) => (
                <div key={String(label)} className="rounded-lg border border-[#1e2a42] bg-[#0e1626] p-3">
                  <p className="text-[11px] uppercase tracking-wider text-[#93a4c0]">{label}</p>
                  <p className="mt-1 text-lg font-bold tabular-nums">{fmt(v as number, 4)}</p>
                </div>
              ))}
            </div>
          ) : (
            <ErrorState message="No test metrics yet - run python scripts/evaluate_models.py." onRetry={metrics.refetch} />
          )}
        </Panel>

        <Panel title="Confusion matrix (test)" subtitle="Rows = true class, columns = predicted class">
          {metrics.loading ? (
            <Skeleton className="h-64 w-full" />
          ) : test?.confusion_matrix && test.confusion_matrix_labels ? (
            <ConfusionMatrix matrix={test.confusion_matrix} labels={test.confusion_matrix_labels} />
          ) : (
            <ErrorState message="No confusion matrix yet - run python scripts/evaluate_models.py." onRetry={metrics.refetch} />
          )}
        </Panel>
      </div>

      {/* ---------------------------------------------------------- per-class */}
      <div className="mt-6 grid gap-5 lg:grid-cols-2">
        <Panel title="Per-class metrics (test)">
          {metrics.loading ? (
            <Skeleton className="h-48 w-full" />
          ) : test?.class_wise ? (
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-[#1e2a42] text-left text-xs uppercase tracking-wider text-[#93a4c0]">
                  <th className="p-2">Class</th>
                  <th className="p-2 text-right">Precision</th>
                  <th className="p-2 text-right">Recall</th>
                  <th className="p-2 text-right">F1</th>
                  <th className="p-2 text-right">Support</th>
                </tr>
              </thead>
              <tbody>
                {classes.map((c) => {
                  const cm = test.class_wise![c];
                  return (
                    <tr key={c} className="border-b border-[#152036]">
                      <td className="p-2 font-semibold" style={{ color: c === "Fatal" ? "#ef4444" : c === "Serious" ? "#f59e0b" : "#38bdf8" }}>{c}</td>
                      <td className="p-2 text-right tabular-nums">{fmt(cm.precision, 3)}</td>
                      <td className="p-2 text-right tabular-nums">{fmt(cm.recall, 3)}</td>
                      <td className="p-2 text-right tabular-nums">{fmt(cm.f1, 3)}</td>
                      <td className="p-2 text-right tabular-nums">{cm.support.toLocaleString()}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          ) : (
            <ErrorState message="No per-class metrics yet." onRetry={metrics.refetch} />
          )}
        </Panel>

        <Panel title="Failure analysis (test)" subtitle="Where the model is wrong - shown as measured">
          {metrics.loading ? (
            <Skeleton className="h-48 w-full" />
          ) : m?.error_analysis ? (
            <div className="space-y-3 text-sm">
              <div className="rounded-lg border border-[#1e2a42] bg-[#0e1626] p-3">
                <p className="text-xs uppercase tracking-wider text-[#93a4c0]">True Fatal collisions in test</p>
                <p className="mt-1 text-xl font-bold tabular-nums">{m.error_analysis.fatal_collisions_in_test.toLocaleString()}</p>
              </div>
              <div>
                <p className="text-xs uppercase tracking-wider text-[#93a4c0]">Fatal collisions predicted as</p>
                <div className="mt-2 grid grid-cols-3 gap-2">
                  {Object.entries(m.error_analysis.fatal_collisions_predicted_as).map(([k, v]) => (
                    <div key={k} className="rounded-lg border border-[#1e2a42] bg-[#0e1626] p-2.5 text-center">
                      <p className="text-[11px] text-[#93a4c0]">{k}</p>
                      <p className="text-lg font-bold tabular-nums" style={{ color: k === "Fatal" ? "#38bdf8" : "#f59e0b" }}>{v.toLocaleString()}</p>
                    </div>
                  ))}
                </div>
              </div>
              <div className="rounded-lg border border-[#1e2a42] bg-[#0e1626] p-3">
                <p className="text-xs uppercase tracking-wider text-[#93a4c0]">Total predicted Fatal</p>
                <p className="mt-1 text-xl font-bold tabular-nums">{m.error_analysis.predicted_fatal_total.toLocaleString()}</p>
                <p className="mt-1 text-[11px] text-[#93a4c0]">vs {m.error_analysis.fatal_collisions_in_test.toLocaleString()} actual - severe over-prediction under class weighting.</p>
              </div>
              {m.error_analysis.calibration_note && (
                <p className="flex items-start gap-2 text-[11px] text-[#93a4c0]">
                  <Info className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden /> {m.error_analysis.calibration_note}
                </p>
              )}
            </div>
          ) : (
            <ErrorState message="No failure analysis yet - run python scripts/evaluate_models.py." onRetry={metrics.refetch} />
          )}
        </Panel>
      </div>

      {/* ---------------------------------------------------------- importance */}
      <div className="mt-6 grid gap-5 lg:grid-cols-2">
        <Panel
          title="Permutation importance by feature family"
          subtitle={importance.data ? `${importance.data.computed_on} - ${importance.data.scoring}` : "Macro-F1 drop when the family is shuffled"}
        >
          {importance.loading ? (
            <Skeleton className="h-80 w-full" />
          ) : importance.error ? (
            <ErrorState message={importance.error} onRetry={importance.refetch} />
          ) : permData.length ? (
            <HorizontalBars data={permData} height={330} formatter={(v) => v.toFixed(4)} />
          ) : (
            <Skeleton className="h-80 w-full" />
          )}
        </Panel>

        <Panel
          title="Model coefficients - Fatal class (top |coef|)"
          subtitle={importance.data?.native_importance ? String((importance.data.native_importance as { model_type?: string }).model_type ?? "") : ""}
        >
          {importance.loading ? (
            <Skeleton className="h-80 w-full" />
          ) : fatalCoefs.length ? (
            <>
              <HorizontalBars data={fatalCoefs} height={330} color="#ef4444" formatter={(v) => v.toFixed(3)} />
              <p className="mt-2 text-[11px] leading-relaxed text-[#93a4c0]">
                {(importance.data?.native_importance as { coefficients_note?: string })?.coefficients_note}
              </p>
            </>
          ) : (
            <p className="text-sm text-[#93a4c0]">No native coefficients for this model type - see permutation importance.</p>
          )}
        </Panel>
      </div>

      {importance.data && (
        <FadeIn>
          <div className="mt-6 flex items-start gap-2 rounded-xl border border-[#f59e0b]/30 bg-[#f59e0b]/8 px-4 py-3 text-xs text-[#fcd34d]">
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
            <p>{importance.data.caveat}</p>
          </div>
        </FadeIn>
      )}

      <FadeIn>
        <div className="mt-6 grid gap-3 text-xs text-[#93a4c0] sm:grid-cols-3">
          <p className="flex items-center gap-2"><Database className="h-4 w-4 text-[#f59e0b]" aria-hidden /> {info.data?.training_rows?.toLocaleString() ?? "-"} rows fitted (train+validation)</p>
          <p className="flex items-center gap-2"><Timer className="h-4 w-4 text-[#f59e0b]" aria-hidden /> {info.data?.test_rows?.toLocaleString() ?? "-"} rows held out for the single test evaluation</p>
          <p className="flex items-center gap-2"><CheckCircle2 className="h-4 w-4 text-[#f59e0b]" aria-hidden /> Test split verified by SHA-256 id hash before scoring</p>
        </div>
      </FadeIn>

      <Stagger className="mt-6 grid gap-3 sm:grid-cols-3">
        {m?.note ? <StaggerItem><p className="text-[11px] text-[#64748b]">{m.note}</p></StaggerItem> : null}
        {info.data?.probability_note ? <StaggerItem className="sm:col-span-2"><p className="text-[11px] text-[#64748b]">{info.data.probability_note}</p></StaggerItem> : null}
      </Stagger>
    </div>
  );
}
