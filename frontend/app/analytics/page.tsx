"use client";

/**
 * Interactive analytics + Tableau-style cross-filtering.
 *
 * ONE request (GET /api/analytics/dashboard) carries the KPI summary and every
 * chart for the CURRENT shared filter state (see components/crossfilter.tsx).
 * Chart clicks and the dropdowns write to that same state: clicking "May" in the
 * monthly chart filters every other chart and the KPI cards to May-only records.
 */
import { useMemo } from "react";
import { Car, Layers, Lightbulb, TrafficCone } from "lucide-react";
import { api } from "@/lib/api";
import { useApi } from "@/hooks/useApi";
import type { DashboardPayload, Filters, ModelInfo } from "@/types";
import { ErrorState, Panel, SectionHeading, Skeleton, StatCard } from "@/components/ui";
import { FadeIn } from "@/components/motion";
import { SeverityDonut } from "@/components/charts";
import { ActiveFilters, CrossFilterPanel, useCrossFilters } from "@/components/crossfilter";

const SEVERITIES = ["Slight", "Serious", "Fatal"];

/** Dropdown options for one STATS19 code field, from the model's declared input fields. */
function codeList(
  info: ModelInfo | null,
  fieldName: string,
  numeric: boolean,
): { value: string; label: string }[] {
  const raw = info?.input_fields.find((f) => f.name === fieldName)?.options ?? [];
  if (numeric) {
    return raw
      .filter((o): o is number => typeof o === "number")
      .map((v) => ({ value: String(v), label: `${v} mph` }));
  }
  return raw
    .filter((o): o is { value: number; label: string } => typeof o !== "number")
    .map((o) => ({ value: String(o.value), label: `${o.value} - ${o.label}` }));
}

function SelectFilter({
  label,
  value,
  onChange,
  options,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  options: { value: string; label: string }[];
}) {
  return (
    <div className="min-w-0 flex-1">
      <label className="field-label" htmlFor={`filter-${label}`}>{label}</label>
      <select id={`filter-${label}`} className="field-input" value={value} onChange={(e) => onChange(e.target.value)}>
        <option value="">All</option>
        {options.map((o) => (
          <option key={o.value} value={o.value}>{o.label}</option>
        ))}
      </select>
    </div>
  );
}

export default function AnalyticsPage() {
  const info = useApi<ModelInfo>(() => api.modelInfo());

  // ---- ONE central filter state shared with every chart, KPI card and dropdown ----
  const { filters, activeCount, setFilter, toggle, clearAll } = useCrossFilters();
  const fetchKey = useMemo(() => JSON.stringify(filters), [filters]);
  const dash = useApi<DashboardPayload>(
    () => api.dashboard(filters as Record<string, unknown>),
    [fetchKey],
  );
  const d = dash.data;
  const s = d?.summary;
  const stale = dash.loading && !!d;          // refetch in progress -> dim, don't unmount
  const counts = Object.fromEntries((s?.severity_distribution ?? []).map((x) => [x.severity, x]));

  // dropdowns: same central state as chart clicks (numeric codes / severity string)
  const setNumber = (k: keyof Filters) => (v: string) => setFilter(k, v === "" ? null : Number(v));
  const setSeverity = (v: string) => setFilter("severity", v === "" ? null : v);
  const reset = () => clearAll();

  const roadOptions = useMemo(() => codeList(info.data, "road_type", false), [info.data]);
  const weatherOptions = useMemo(() => codeList(info.data, "weather_conditions", false), [info.data]);
  const lightOptions = useMemo(() => codeList(info.data, "light_conditions", false), [info.data]);
  const urbanOptions = useMemo(() => codeList(info.data, "urban_or_rural_area", false), [info.data]);
  const speedOptions = useMemo(() => codeList(info.data, "speed_limit", true), [info.data]);
  const yearOptions = useMemo(
    () => Object.keys(d?.labels?.year ?? {}).map((y) => ({ value: y, label: y })),
    [d],
  );

  return (
    <div className="mx-auto max-w-7xl px-4 py-10 sm:px-6">
      <FadeIn>
        <SectionHeading
          kicker="Aggregated backend data"
          title="Interactive Analytics"
          description="Click any chart category (or use the dropdowns) to cross-filter the whole dashboard: every other chart and the KPI cards recompute from that selection, and each chip can be removed individually. Charts update from a single cached backend aggregation - raw CSVs never reach the browser."
        />
      </FadeIn>

      {dash.error && !d && (
        <div className="mt-6">
          <ErrorState message={dash.error} onRetry={dash.refetch} />
        </div>
      )}

      {/* ---------------------------------------------------- active filter chips */}
      <div className="mt-6">
        <ActiveFilters labels={d?.labels} loading={stale} />
      </div>

      {/* ---------------------------------------------------- filters (same state) */}
      <FadeIn>
        <div className="card mt-5 p-4">
          <div className="mb-3 flex items-center justify-between">
            <p className="text-xs font-semibold uppercase tracking-widest text-[#93a4c0]">
              Filters
              {activeCount > 0 && (
                <span className="ml-2 rounded-full bg-[#f59e0b]/15 px-2 py-0.5 text-[#f59e0b]">{activeCount} active</span>
              )}
            </p>
            <button
              type="button"
              onClick={reset}
              className="inline-flex items-center gap-1.5 text-xs text-[#93a4c0] hover:text-[#f59e0b]"
            >
              Clear all filters
            </button>
          </div>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4 xl:grid-cols-7">
            <SelectFilter label="Year" value={String(filters.year ?? "")} onChange={setNumber("year")} options={yearOptions} />
            <div>
              <label className="field-label" htmlFor="filter-severity">Severity</label>
              <select id="filter-severity" className="field-input" value={filters.severity ?? ""} onChange={(e) => setSeverity(e.target.value)}>
                <option value="">All</option>
                {SEVERITIES.map((sv) => (
                  <option key={sv} value={sv}>{sv}</option>
                ))}
              </select>
            </div>
            <SelectFilter label="Road type" value={String(filters.road_type ?? "")} onChange={setNumber("road_type")} options={roadOptions} />
            <SelectFilter label="Weather" value={String(filters.weather ?? "")} onChange={setNumber("weather")} options={weatherOptions} />
            <SelectFilter label="Lighting" value={String(filters.lighting ?? "")} onChange={setNumber("lighting")} options={lightOptions} />
            <SelectFilter label="Speed limit" value={String(filters.speed_limit ?? "")} onChange={setNumber("speed_limit")} options={speedOptions} />
            <SelectFilter label="Urban / rural" value={String(filters.urban_rural ?? "")} onChange={setNumber("urban_rural")} options={urbanOptions} />
          </div>
        </div>
      </FadeIn>

      {/* ---------------------------------------------------- KPI cards (cross-filtered) */}
      <div className={`mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4 ${stale ? "opacity-60 transition-opacity" : "transition-opacity"}`} data-testid="kpi-cards">
        <StatCard
          label="Total collisions"
          icon={Car}
          value={s?.total_collisions ?? null}
          loading={!d}
          sub={activeCount > 0 ? "Matching the active filters" : "Police-reported, all loaded years"}
        />
        <StatCard
          label="Slight"
          icon={Layers}
          accent="#38bdf8"
          value={counts.Slight?.count ?? null}
          loading={!d}
          sub={counts.Slight ? `${counts.Slight.percentage.toFixed(1)}% of the selection` : undefined}
        />
        <StatCard
          label="Serious"
          icon={TrafficCone}
          accent="#f59e0b"
          value={counts.Serious?.count ?? null}
          loading={!d}
          sub={counts.Serious ? `${counts.Serious.percentage.toFixed(1)}% of the selection` : undefined}
        />
        <StatCard
          label="Fatal"
          icon={Lightbulb}
          accent="#ef4444"
          value={counts.Fatal?.count ?? null}
          loading={!d}
          sub={counts.Fatal ? `${counts.Fatal.percentage.toFixed(2)}% of the selection` : undefined}
        />
      </div>

      {/* ------------------------------------------------------------ charts */}
      <div className="mt-6 grid gap-5 lg:grid-cols-3">
        <Panel title="Severity split" subtitle="Click a slice to filter every other chart">
          {d ? (
            <div
              className={stale ? "opacity-60 transition-opacity" : "transition-opacity"}
              data-testid="chart-severity"
              data-selected={filters.severity ?? ""}
            >
              <SeverityDonut
                data={d.severity.items}
                selected={filters.severity ?? null}
                onSelect={(sv) => toggle("severity", sv)}
              />
              <p className="mt-2 text-center text-xs text-[#93a4c0]">
                {filters.severity
                  ? "The severity filter is excluded in this donut so all severities stay visible; KPIs and every other chart apply it."
                  : `${d.severity.total.toLocaleString()} collisions match the other active filters`}
              </p>
            </div>
          ) : (
            <Skeleton className="h-[280px]" />
          )}
        </Panel>
        <CrossFilterPanel
          title="Yearly severity"
          subtitle="Click a bar to filter by year"
          seriesKey="by-year"
          dim="year"
          data={d}
          loading={stale}
          height={280}
          className="lg:col-span-2"
        />
      </div>

      <div className="mt-5 grid gap-5 lg:grid-cols-2">
        <CrossFilterPanel title="By road type" subtitle="Click a bar to filter" seriesKey="by-road-type" dim="road_type" data={d} loading={stale} horizontal height={300} />
        <CrossFilterPanel title="By weather" subtitle="Click a bar to filter" seriesKey="by-weather" dim="weather" data={d} loading={stale} horizontal height={300} />
        <CrossFilterPanel title="By lighting" subtitle="Click a bar to filter" seriesKey="by-lighting" dim="lighting" data={d} loading={stale} horizontal height={260} />
        <CrossFilterPanel title="By speed limit" subtitle="Click a bar to filter" seriesKey="by-speed-limit" dim="speed_limit" data={d} loading={stale} horizontal height={260} />
        <CrossFilterPanel title="By hour of day" subtitle="Click a bar to filter" seriesKey="by-hour" dim="hour" data={d} loading={stale} height={280} />
        <CrossFilterPanel title="By weekday" subtitle="Click a bar to filter" seriesKey="by-weekday" dim="weekday" data={d} loading={stale} height={280} />
        <CrossFilterPanel title="By month" subtitle="Click a month (e.g. May) to filter the whole dashboard" seriesKey="by-month" dim="month" data={d} loading={stale} height={280} />
        <CrossFilterPanel title="By urban / rural" subtitle="Click a bar to filter" seriesKey="by-urban-rural" dim="urban_rural" data={d} loading={stale} horizontal height={220} />
      </div>

      <p className="mt-6 text-xs text-[#64748b]">
        All series are computed by the FastAPI analytics service from the processed STATS19
        dataset; descriptive statistics only, no causal interpretation.
      </p>
    </div>
  );
}
