"use client";

/**
 * Tableau-style cross-filtering: ONE shared filter state for every dashboard chart
 * and KPI card, plus the "Active filters" bar and the chart panel wiring.
 *
 * - Chart click  -> toggle(dim, key)   (clicking the selected value again clears it)
 * - Chip "x"     -> toggle(dim, key)   (remove one filter independently)
 * - "Clear all"  -> clearAll()         (no page reload, no navigation)
 *
 * The provider lives in the root layout, so /dashboard and /analytics consume the
 * exact same state (a dropdown selection and a chart click are interchangeable).
 */
import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { Filter, RotateCcw, X } from "lucide-react";
import type { BreakdownItem, DashboardPayload, Filters } from "@/types";
import { Panel, Skeleton } from "@/components/ui";
import { StackedSeverityBar } from "@/components/charts";

export const EMPTY_FILTERS: Required<Filters> = {
  year: null,
  month: null,
  severity: null,
  road_type: null,
  weather: null,
  lighting: null,
  speed_limit: null,
  urban_rural: null,
  hour: null,
  weekday: null,
};

export type FilterKey = keyof Filters;
export type FilterValue = number | string;

interface CrossFilterValue {
  filters: Filters;
  activeKeys: FilterKey[];
  activeCount: number;
  /** Clicking the already-selected value removes that filter (toggle). */
  toggle: (key: FilterKey, value: FilterValue) => void;
  setFilter: (key: FilterKey, value: FilterValue | null) => void;
  clearAll: () => void;
}

const CrossFilterContext = createContext<CrossFilterValue | null>(null);

function isActive(v: FilterValue | null | undefined): v is FilterValue {
  return v !== null && v !== undefined && v !== "";
}

export function CrossFilterProvider({ children }: { children: ReactNode }) {
  const [filters, setFilters] = useState<Filters>({ ...EMPTY_FILTERS });

  const setFilter = useCallback((key: FilterKey, value: FilterValue | null) => {
    setFilters((f) => ({ ...f, [key]: value }) as Filters);
  }, []);

  const toggle = useCallback((key: FilterKey, value: FilterValue) => {
    setFilters((f) => {
      const current = f[key];
      return { ...f, [key]: current === value ? null : value } as Filters;
    });
  }, []);

  const clearAll = useCallback(() => setFilters({ ...EMPTY_FILTERS }), []);

  const activeKeys = useMemo(
    () => (Object.keys(EMPTY_FILTERS) as FilterKey[]).filter((k) => isActive(filters[k])),
    [filters],
  );

  const value = useMemo(
    () => ({ filters, activeKeys, activeCount: activeKeys.length, toggle, setFilter, clearAll }),
    [filters, activeKeys, toggle, setFilter, clearAll],
  );

  return <CrossFilterContext.Provider value={value}>{children}</CrossFilterContext.Provider>;
}

export function useCrossFilters(): CrossFilterValue {
  const ctx = useContext(CrossFilterContext);
  if (!ctx) throw new Error("useCrossFilters must be used inside <CrossFilterProvider>");
  return ctx;
}

/* ------------------------------------------------------------------ active filter bar */
export function ActiveFilters({
  labels,
  loading = false,
}: {
  /** value -> display label per dimension, straight from the backend aggregations. */
  labels?: Record<string, Record<string, string>>;
  loading?: boolean;
}) {
  const { filters, activeKeys, toggle, clearAll } = useCrossFilters();

  const chips = activeKeys.map((key) => {
    const value = filters[key] as FilterValue;
    const label = labels?.[key]?.[String(value)] ?? String(value);
    return { key, value, label };
  });

  return (
    <section
      aria-label="Active filters"
      className="card px-4 py-3"
      data-testid="active-filters"
      data-active={chips.length}
    >
      <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
        <p className="flex items-center gap-2 text-xs font-semibold uppercase tracking-widest text-[#93a4c0]">
          <Filter className="h-4 w-4 text-[#f59e0b]" aria-hidden />
          Active filters
        </p>
        <p className="text-sm text-[#93a4c0]" aria-live="polite">
          {chips.length
            ? "Showing records filtered by:"
            : "Showing all available records"}
        </p>
        {chips.length > 0 && (
          <div className="flex flex-wrap items-center gap-2">
            {chips.map((c) => (
              <button
                key={c.key}
                type="button"
                onClick={() => toggle(c.key, c.value)}
                aria-label={`Remove ${c.label} filter`}
                className="inline-flex items-center gap-1.5 rounded-full border border-[#f59e0b]/40 bg-[#f59e0b]/10 px-2.5 py-1 text-xs font-semibold text-[#fcd34d] transition hover:border-[#f59e0b] hover:bg-[#f59e0b]/20"
              >
                {c.label}
                <X className="h-3 w-3" aria-hidden />
              </button>
            ))}
            <button
              type="button"
              onClick={clearAll}
              className="inline-flex items-center gap-1.5 rounded-full border border-[#1e2a42] bg-[#0e1626] px-2.5 py-1 text-xs text-[#93a4c0] transition hover:border-[#f59e0b]/50 hover:text-[#fcd34d]"
            >
              <RotateCcw className="h-3 w-3" aria-hidden /> Clear all
            </button>
          </div>
        )}
        {loading && <span className="text-xs text-[#64748b]">Updating&hellip;</span>}
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ chart panel wiring */
/**
 * Panel + stacked chart wired to the shared filter state: clicking a category
 * toggles that filter; the selected category stays visible and emphasised while
 * the other categories are muted (never hidden).
 */
export function CrossFilterPanel({
  title,
  subtitle,
  seriesKey,
  dim,
  data,
  loading = false,
  height = 300,
  horizontal = false,
  className,
}: {
  title: string;
  subtitle?: string;
  seriesKey: string;
  dim: FilterKey;
  data: DashboardPayload | null;
  loading?: boolean;
  height?: number;
  horizontal?: boolean;
  className?: string;
}) {
  const { filters, toggle } = useCrossFilters();
  const items: BreakdownItem[] | null = data?.series?.[seriesKey]?.items ?? null;
  const selectedKey = (isActive(filters[dim]) ? filters[dim] : null) as FilterValue | null;

  return (
    <Panel title={title} subtitle={subtitle} className={className}>
      {items ? (
        <div
          className={loading ? "opacity-60 transition-opacity" : "transition-opacity"}
          data-testid={`chart-${seriesKey}`}
          data-selected={selectedKey === null ? "" : String(selectedKey)}
        >
          <StackedSeverityBar
            data={items}
            height={height}
            horizontal={horizontal}
            selectedKey={selectedKey}
            onSelect={(item) => {
              if (item.key === null || item.key === undefined) return;
              toggle(dim, item.key);
            }}
          />
        </div>
      ) : (
        <Skeleton className="h-64 w-full" />
      )}
    </Panel>
  );
}
