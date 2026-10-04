"use client";

/**
 * Dashboard: KPIs + charts, every value from ONE real backend aggregation that
 * honours the shared cross-filter state (see components/crossfilter.tsx).
 * Clicking a chart category filters the KPI cards and every other chart.
 */
import { useMemo } from "react";
import {
  BarChart3,
  CalendarRange,
  Car,
  CloudSun,
  Gauge,
  Layers,
  Lightbulb,
  MoonStar,
  Timer,
  TrafficCone,
} from "lucide-react";
import { SeverityDonut, TrendLine } from "@/components/charts";
import { ErrorState, Panel, SectionHeading, Skeleton, StatCard } from "@/components/ui";
import { FadeIn } from "@/components/motion";
import { api } from "@/lib/api";
import { useApi } from "@/hooks/useApi";
import type { DashboardPayload } from "@/types";
import { ActiveFilters, CrossFilterPanel, useCrossFilters } from "@/components/crossfilter";

export default function DashboardPage() {
  // ---- ONE central filter state shared with the Analytics page ----
  const { filters, activeCount, toggle } = useCrossFilters();
  const fetchKey = useMemo(() => JSON.stringify(filters), [filters]);
  const dash = useApi<DashboardPayload>(
    () => api.dashboard(filters as Record<string, unknown>),
    [fetchKey],
  );
  const d = dash.data;
  const s = d?.summary;
  const stale = dash.loading && !!d; // refetch in progress -> dim, keep charts mounted
  const sev = Object.fromEntries((s?.severity_distribution ?? []).map((x) => [x.severity, x]));
  const hourlyItems = (d?.series["by-hour"]?.items ?? []).filter((i) => i.key !== null);

  return (
    <div className="mx-auto max-w-7xl px-4 py-10 sm:px-6">
      <FadeIn>
        <div className="flex flex-wrap items-end justify-between gap-4">
          <SectionHeading
            kicker="Live from FastAPI"
            title="Road Safety Dashboard"
            description={`Real descriptive statistics over the processed STATS19 dataset${s?.dataset_period ? ` (${s.dataset_period})` : ""}. Every number on this page is computed from the real data - no hardcoded chart arrays. Click a chart category to cross-filter the whole dashboard.`}
          />
          <div className="flex flex-wrap gap-2">
            {(s?.years ?? []).map((y) => (
              <span key={y} className="rounded-full border border-[#1e2a42] px-3 py-1 text-xs text-[#93a4c0]">
                {y}
              </span>
            ))}
          </div>
        </div>
      </FadeIn>

      {dash.error && !d && (
        <div className="mt-6">
          <ErrorState message={dash.error} onRetry={dash.refetch} />
        </div>
      )}

      {/* ---------------------------------------------------------- active filters */}
      <div className="mt-6">
        <ActiveFilters labels={d?.labels} loading={stale} />
      </div>

      {/* ---------------------------------------------------------- KPI row (cross-filtered) */}
      <div
        className={`mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4 ${stale ? "opacity-60 transition-opacity" : "transition-opacity"}`}
        data-testid="kpi-cards"
      >
        <StatCard
          label="Total collisions"
          icon={Car}
          value={s?.total_collisions ?? null}
          loading={!d}
          sub={activeCount > 0 ? "Matching the active filters" : "Police-reported, all loaded years"}
        />
        <StatCard label="Slight" icon={Layers} accent="#38bdf8" value={sev.Slight?.percentage ?? null} decimals={1} suffix="%" loading={!d} sub={sev.Slight ? `${sev.Slight.count.toLocaleString()} collisions` : undefined} />
        <StatCard label="Serious" icon={TrafficCone} accent="#f59e0b" value={sev.Serious?.percentage ?? null} decimals={1} suffix="%" loading={!d} sub={sev.Serious ? `${sev.Serious.count.toLocaleString()} collisions` : undefined} />
        <StatCard label="Fatal" icon={Lightbulb} accent="#ef4444" value={sev.Fatal?.percentage ?? null} decimals={2} suffix="%" loading={!d} sub={sev.Fatal ? `${sev.Fatal.count.toLocaleString()} collisions` : undefined} />
      </div>

      {/* ---------------------------------------------------------- KPI row 2 (model + years) */}
      <div className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard label="Dataset years" icon={CalendarRange} value={s?.years.length ?? null} loading={!d} sub={s ? s.years.join(", ") : undefined} />
        <StatCard label="Final model" icon={Gauge} value={s?.model?.name ?? (d ? "-" : undefined)} loading={!d} sub="Selected on validation metrics" />
        <StatCard label="Macro F1 (test)" icon={BarChart3} accent="#38bdf8" value={s?.model?.macro_f1 ?? null} decimals={4} loading={!d} sub="Final evaluation, untouched test split" />
        <StatCard label="Balanced accuracy (test)" icon={Gauge} accent="#38bdf8" value={s?.model?.balanced_accuracy ?? null} decimals={4} loading={!d} sub="Final evaluation, untouched test split" />
      </div>

      {/* ---------------------------------------------------------- charts */}
      <div className="mt-8 grid gap-5 lg:grid-cols-3">
        <Panel title="Severity distribution" subtitle="Click a slice to filter the dashboard">
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
                  ? "Severity filter excluded here so all severities stay visible."
                  : `${d.severity.total.toLocaleString()} collisions match the other active filters`}
              </p>
            </div>
          ) : (
            <Skeleton className="h-[280px]" />
          )}
        </Panel>
        <CrossFilterPanel
          title="Severity by year"
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
        <CrossFilterPanel
          title="Collisions by hour of day"
          subtitle="Stacked severity, 00:00-23:00 - click to filter"
          seriesKey="by-hour"
          dim="hour"
          data={d}
          loading={stale}
          height={280}
        />
        <Panel title="Total collisions by hour" subtitle="Volume profile across the day">
          {d ? (
            <div className={stale ? "opacity-60 transition-opacity" : "transition-opacity"}>
              <TrendLine data={hourlyItems} xKey="label" yKey="total" height={280} name="Collisions" />
            </div>
          ) : (
            <Skeleton className="h-[280px]" />
          )}
        </Panel>
      </div>

      <div className="mt-5 grid gap-5 lg:grid-cols-2">
        <CrossFilterPanel title="Severity by road type" subtitle="Click a bar to filter" seriesKey="by-road-type" dim="road_type" data={d} loading={stale} horizontal height={300} />
        <CrossFilterPanel title="Severity by weather" subtitle="Click a bar to filter" seriesKey="by-weather" dim="weather" data={d} loading={stale} horizontal height={300} />
        <CrossFilterPanel title="Severity by lighting" subtitle="Click a bar to filter" seriesKey="by-lighting" dim="lighting" data={d} loading={stale} horizontal height={260} />
        <CrossFilterPanel title="Severity by speed limit" subtitle="Click a bar to filter" seriesKey="by-speed-limit" dim="speed_limit" data={d} loading={stale} horizontal height={260} />
        <CrossFilterPanel title="Collisions by weekday" subtitle="Click a bar to filter" seriesKey="by-weekday" dim="weekday" data={d} loading={stale} height={280} />
        <CrossFilterPanel title="Collisions by month" subtitle="Click a month (e.g. May) to filter the dashboard" seriesKey="by-month" dim="month" data={d} loading={stale} height={280} />
      </div>

      <FadeIn>
        <div className="mt-8 flex flex-wrap items-center gap-4 text-xs text-[#93a4c0]">
          <span className="flex items-center gap-1.5">
            <MoonStar className="h-3.5 w-3.5" aria-hidden /> Percentages are computed from the
            processed dataset at request time and cached in the API.
          </span>
          <span className="flex items-center gap-1.5">
            <Timer className="h-3.5 w-3.5" aria-hidden /> Model metrics come from the single final
            evaluation on the untouched test split.
          </span>
          <span className="flex items-center gap-1.5">
            <CloudSun className="h-3.5 w-3.5" aria-hidden /> Descriptive statistics only - no
            causal claims.
          </span>
        </div>
      </FadeIn>

      {s && (
        <p className="sr-only" aria-live="polite">
          Dashboard loaded {s.total_collisions.toLocaleString()} collisions
          {activeCount > 0 ? " under the active filters" : ""} across {s.years.join(", ")}.
        </p>
      )}
    </div>
  );
}
