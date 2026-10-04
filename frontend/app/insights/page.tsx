"use client";

/** Road-safety insights: descriptive findings computed by the backend from real data. */
import { AlertTriangle, BarChart3, CalendarDays, CloudSun, MapPinned, Tag } from "lucide-react";
import { api } from "@/lib/api";
import { useApi } from "@/hooks/useApi";
import type { Insights as InsightsData } from "@/types";
import { ErrorState, SectionHeading, Skeleton } from "@/components/ui";
import { FadeIn, Stagger, StaggerItem } from "@/components/motion";

const CATEGORY_ICONS: Record<string, typeof Tag> = {
  "Road characteristics": MapPinned,
  "Environmental conditions": CloudSun,
  "Temporal patterns": CalendarDays,
};

export default function InsightsPage() {
  const { data, error, loading, refetch } = useApi<InsightsData>(() => api.insights());

  const categories = Array.from(new Set((data?.findings ?? []).map((f) => f.category)));

  return (
    <div className="mx-auto max-w-7xl px-4 py-10 sm:px-6">
      <FadeIn>
        <SectionHeading
          kicker="Descriptive statistics"
          title="Road Safety Insights"
          description="Every statistic below is calculated by the API from the actual processed STATS19 data. Each item is labelled DATA FINDING: these are descriptive associations in historical police-reported collisions, not causal effects and not predictions."
        />
      </FadeIn>

      <FadeIn>
        <div className="mt-5 flex flex-wrap items-center gap-3 text-xs">
          <span className="inline-flex items-center gap-1.5 rounded-full border border-[#f59e0b]/40 bg-[#f59e0b]/10 px-3 py-1 font-bold tracking-widest text-[#fcd34d]">
            <Tag className="h-3.5 w-3.5" aria-hidden /> DATA FINDING
          </span>
          {data && (
            <span className="text-[#93a4c0]">{data.rows_analysed.toLocaleString()} rows analysed</span>
          )}
        </div>
      </FadeIn>

      {loading && (
        <div className="mt-8 grid gap-4 md:grid-cols-2">
          {Array.from({ length: 6 }).map((_, i) => (
            <Skeleton key={i} className="h-44" />
          ))}
        </div>
      )}

      {error && <div className="mt-8"><ErrorState message={error} onRetry={refetch} /></div>}

      {!loading &&
        categories.map((cat) => {
          const Icon = CATEGORY_ICONS[cat] ?? BarChart3;
          const findings = (data?.findings ?? []).filter((f) => f.category === cat);
          return (
            <section key={cat} className="mt-10">
              <h2 className="flex items-center gap-2 text-lg font-bold">
                <Icon className="h-5 w-5 text-[#f59e0b]" aria-hidden /> {cat}
              </h2>
              <Stagger className="mt-4 grid gap-4 md:grid-cols-2">
                {findings.map((f) => (
                  <StaggerItem key={f.id}>
                    <div className="card card-hover h-full p-5">
                      <div className="flex items-start justify-between gap-3">
                        <h3 className="text-sm font-bold text-[#cbd7ec]">{f.title}</h3>
                        <span className="rounded border border-[#f59e0b]/40 px-1.5 py-0.5 text-[9px] font-bold tracking-widest text-[#fcd34d]">
                          DATA FINDING
                        </span>
                      </div>
                      <p className="mt-3 text-sm leading-relaxed text-[#e6edf7]">{f.stat}</p>
                    </div>
                  </StaggerItem>
                ))}
              </Stagger>
            </section>
          );
        })}

      {data && (
        <FadeIn>
          <div className="mt-10 flex items-start gap-3 rounded-xl border border-[#1e2a42] bg-[#0e1626] px-5 py-4">
            <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-[#f59e0b]" aria-hidden />
            <div>
              <p className="text-sm font-semibold">How to read these findings</p>
              <p className="mt-1.5 text-xs leading-relaxed text-[#93a4c0]">{data.note}</p>
            </div>
          </div>
        </FadeIn>
      )}
    </div>
  );
}
