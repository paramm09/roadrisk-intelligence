"use client";

/** Historical collision map: aggregated coordinates from the API, Leaflet lazy-loaded. */
import dynamic from "next/dynamic";
import { useState } from "react";
import { AlertTriangle, MapPin, RefreshCw } from "lucide-react";
import { api } from "@/lib/api";
import { useApi } from "@/hooks/useApi";
import type { MapPoints, Summary } from "@/types";
import { ErrorState, EmptyState, SectionHeading, Skeleton } from "@/components/ui";
import { FadeIn } from "@/components/motion";

const MapView = dynamic(() => import("@/components/MapView"), {
  ssr: false,
  loading: () => (
    <div className="flex h-[650px] w-full items-center justify-center rounded-xl border border-[#1e2a42] bg-[#0e1626] lg:h-[700px]">
      <div className="flex flex-col items-center gap-3">
        <RefreshCw className="h-6 w-6 animate-spin text-[#f59e0b]" aria-hidden />
        <p className="text-sm text-[#93a4c0]">Loading map...</p>
      </div>
    </div>
  ),
});

const SEVERITIES = ["Slight", "Serious", "Fatal"];

export default function MapPage() {
  const summary = useApi<Summary>(() => api.summary());
  const [year, setYear] = useState<string>("");
  const [severity, setSeverity] = useState<string>("");

  const filters = {
    year: year === "" ? null : Number(year),
    severity: severity === "" ? null : severity,
  };
  const { data, error, loading, refetch } = useApi<MapPoints>(
    () => api.mapPoints(filters as Record<string, unknown>),
    [year, severity],
  );

  return (
    <div className="mx-auto max-w-7xl px-4 py-10 sm:px-6">
      <FadeIn>
        <SectionHeading
          kicker="Historical record"
          title="Historical Collision Map"
          description="Distribution of past police-reported collisions across Great Britain, aggregated into 0.01-degree grid cells. This is a map of what HAS happened - it is not a prediction of where future accidents will occur, and the model does not predict collision locations."
        />
      </FadeIn>

      <FadeIn>
        <div className="mt-5 flex flex-wrap items-end gap-3">
          <div className="min-w-40">
            <label className="field-label" htmlFor="map-year">Year</label>
            <select id="map-year" className="field-input" value={year} onChange={(e) => setYear(e.target.value)}>
              <option value="">All years</option>
              {(summary.data?.years ?? []).map((y) => (
                <option key={y} value={y}>{y}</option>
              ))}
            </select>
          </div>
          <div className="min-w-40">
            <label className="field-label" htmlFor="map-severity">Severity</label>
            <select id="map-severity" className="field-input" value={severity} onChange={(e) => setSeverity(e.target.value)}>
              <option value="">All severities</option>
              {SEVERITIES.map((s) => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
          </div>
          <div className="flex flex-wrap items-center gap-4 pb-1 text-xs text-[#93a4c0]">
            {SEVERITIES.map((s) => (
              <span key={s} className="flex items-center gap-1.5">
                <span
                  className="h-3 w-3 rounded-full border-2"
                  style={{
                    borderColor: s === "Fatal" ? "#ef4444" : s === "Serious" ? "#f59e0b" : "#38bdf8",
                    background: (s === "Fatal" ? "#ef4444" : s === "Serious" ? "#f59e0b" : "#38bdf8") + "55",
                  }}
                  aria-hidden
                />
                {s}
              </span>
            ))}
            <span className="text-[#64748b]">Circle size = collision count</span>
          </div>
        </div>
      </FadeIn>

      <div className="mt-4">
        {error && <ErrorState message={error} onRetry={refetch} />}
        {loading && <Skeleton className="h-[650px] w-full lg:h-[700px]" />}
        {!loading && !error && data && data.points.length === 0 && (
          <EmptyState message="No collision locations found for the selected filters." />
        )}
        {!loading && !error && data && data.points.length > 0 && <MapView points={data.points} />}
      </div>

      {data && data.points.length > 0 && (
        <div className="mt-4 flex flex-wrap items-center justify-between gap-3 text-xs text-[#93a4c0]">
          <p className="flex items-center gap-1.5">
            <MapPin className="h-3.5 w-3.5 text-[#f59e0b]" aria-hidden />
            Showing {data.total_cells.toLocaleString()} grid cells
            {data.truncated ? " (largest cells capped for performance)" : ""} &middot; click a point for severity counts
          </p>
        </div>
      )}

      <FadeIn>
        <div className="mt-5 flex items-start gap-3 rounded-xl border border-[#f59e0b]/30 bg-[#f59e0b]/8 px-4 py-3 text-xs text-[#fcd34d]">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
          <p>
            {data?.note ??
              "Aggregated 0.01-degree grid cells of HISTORICAL police-reported collisions. This is not a prediction of future collisions."}{" "}
            Coordinates are rounded for privacy and aggregated to grid cells; density reflects
            reporting and exposure, not only risk.
          </p>
        </div>
      </FadeIn>
    </div>
  );
}
