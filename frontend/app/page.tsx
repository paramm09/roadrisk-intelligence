"use client";

import Link from "next/link";
import {
  Activity,
  ArrowRight,
  BarChart3,
  BrainCircuit,
  Building2,
  Car,
  Database,
  FileSearch,
  FlaskConical,
  Gauge,
  Layers,
  Library,
  Radar,
  ShieldAlert,
  Siren,
  TestTubes,
  TrafficCone,
  Workflow,
} from "lucide-react";
import { motion, useReducedMotion, useScroll, useTransform } from "framer-motion";
import { useRef } from "react";
import { AnimatedCounter, SeverityBadge, Skeleton } from "@/components/ui";
import { FadeIn, Stagger, StaggerItem } from "@/components/motion";
import { api } from "@/lib/api";
import { useApi } from "@/hooks/useApi";

const WORKFLOW = [
  { icon: Database, label: "Real STATS19 CSV data" },
  { icon: FileSearch, label: "Data inspection" },
  { icon: FlaskConical, label: "Cleaning & validation" },
  { icon: BarChart3, label: "EDA" },
  { icon: Layers, label: "Feature engineering" },
  { icon: ShieldAlert, label: "Leakage audit" },
  { icon: TestTubes, label: "Train / val / test split" },
  { icon: BrainCircuit, label: "Baseline + ML models" },
  { icon: Gauge, label: "Hyperparameter tuning" },
  { icon: Activity, label: "Final test evaluation" },
  { icon: Workflow, label: "FastAPI + web app" },
];

const STAKEHOLDERS = [
  { icon: Siren, label: "Road-safety officers", desc: "Prioritise interventions with measured severity patterns." },
  { icon: Radar, label: "Police analysts", desc: "Explore historical collision records and severity mixes." },
  { icon: Building2, label: "Local councils", desc: "Evidence for road design and speed-limit decisions." },
  { icon: TrafficCone, label: "Transport planners", desc: "Understand temporal and environmental risk profiles." },
  { icon: Library, label: "Researchers", desc: "Reproducible ML pipeline with documented methodology." },
];

/** Animated road/network hero visual (decorative, data-free by design). */
function RoadNetwork() {
  const reduced = useReducedMotion();
  const ref = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({ target: ref, offset: ["start end", "end start"] });
  const y = useTransform(scrollYProgress, [0, 1], [0, reduced ? 0 : -40]);

  return (
    <motion.div ref={ref} style={{ y }} className="relative mx-auto w-full max-w-lg" aria-hidden>
      <div className="map-grid absolute inset-0 rounded-2xl opacity-70" />
      <svg viewBox="0 0 420 340" className="relative w-full">
        {/* road network */}
        <g stroke="#1e2a42" strokeWidth="10" fill="none" strokeLinecap="round">
          <path d="M20 300 L140 180 L260 210 L400 90" />
          <path d="M40 60 L160 120 L240 60 L390 150" />
          <path d="M210 330 L210 210" />
          <path d="M60 340 L120 250 L120 130" />
        </g>
        <g stroke="#f59e0b" strokeWidth="2" strokeDasharray="10 12" fill="none" opacity="0.85">
          <motion.path
            d="M20 300 L140 180 L260 210 L400 90"
            initial={{ pathLength: 0 }}
            animate={{ pathLength: 1 }}
            transition={{ duration: reduced ? 0 : 2.4, ease: "easeInOut", repeat: Infinity, repeatType: "loop" }}
          />
          <motion.path
            d="M40 60 L160 120 L240 60 L390 150"
            initial={{ pathLength: 0 }}
            animate={{ pathLength: 1 }}
            transition={{ duration: reduced ? 0 : 3, delay: 0.4, ease: "easeInOut", repeat: Infinity, repeatType: "loop" }}
          />
        </g>
        {/* collision hotspots */}
        {[
          { cx: 140, cy: 180, r: 13, c: "#f59e0b" },
          { cx: 260, cy: 210, r: 9, c: "#38bdf8" },
          { cx: 160, cy: 120, r: 16, c: "#ef4444" },
          { cx: 390, cy: 150, r: 8, c: "#f59e0b" },
          { cx: 210, cy: 330, r: 7, c: "#38bdf8" },
        ].map((p, i) => (
          <motion.circle
            key={i}
            cx={p.cx}
            cy={p.cy}
            r={p.r}
            fill={p.c}
            opacity={0.25}
            initial={{ scale: 0.7 }}
            animate={reduced ? { scale: 1 } : { scale: [0.7, 1.15, 0.7], opacity: [0.2, 0.45, 0.2] }}
            transition={{ duration: 3, delay: i * 0.35, repeat: Infinity }}
            style={{ transformOrigin: `${p.cx}px ${p.cy}px` }}
          />
        ))}
      </svg>
      <div className="absolute -bottom-3 left-4 flex gap-2">
        <SeverityBadge severity="Slight" small />
        <SeverityBadge severity="Serious" small />
        <SeverityBadge severity="Fatal" small />
      </div>
    </motion.div>
  );
}

export default function LandingPage() {
  const { data: summary, loading, error } = useApi(() => api.summary());

  return (
    <div className="overflow-clip">
      {/* ---------------------------------------------------------------- HERO */}
      <section className="relative border-b border-[#1e2a42] bg-[#0b1220]">
        <div className="map-grid absolute inset-0" aria-hidden />
        <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-[#f59e0b]/50 to-transparent" aria-hidden />
        <div className="relative mx-auto grid max-w-7xl items-center gap-12 px-4 pb-20 pt-16 sm:px-6 lg:grid-cols-2 lg:pt-24">
          <motion.div
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, ease: "easeOut" }}
          >
            <p className="kicker">UK DfT STATS19 &middot; 2021&ndash;2025 &middot; {summary?.total_collisions ? <AnimatedCounter value={summary.total_collisions} /> : loading ? <Skeleton className="inline-block h-4 w-20" /> : "5 collision years"} collisions</p>
            <h1 className="mt-4 text-4xl font-black leading-[1.04] tracking-tight sm:text-6xl">
              ROADRISK
              <br />
              <span className="text-[#f59e0b]">INTELLIGENCE</span>
            </h1>
            <div className="road-line road-line-animated mt-5 w-64" aria-hidden />
            <p className="mt-5 max-w-xl text-lg leading-relaxed text-[#cbd7ec]">
              Machine-learning powered road safety intelligence.
            </p>
            <p className="mt-3 max-w-xl text-sm leading-relaxed text-[#93a4c0]">
              Predict road collision severity and explore five years of real UK Department for
              Transport STATS19 collision data &mdash; a complete ML lifecycle from raw CSV to a
              live FastAPI model endpoint and interactive analytics.
            </p>
            <div className="mt-8 flex flex-wrap gap-3">
              <Link href="/predict" className="btn-primary inline-flex items-center gap-2 text-sm">
                Open Prediction Studio <ArrowRight className="h-4 w-4" aria-hidden />
              </Link>
              <Link href="/dashboard" className="btn-ghost inline-flex items-center gap-2 text-sm">
                <BarChart3 className="h-4 w-4" aria-hidden /> View Dashboard
              </Link>
            </div>
            <p className="mt-6 flex items-center gap-2 text-xs text-[#93a4c0]">
              <ShieldAlert className="h-4 w-4 text-[#f59e0b]" aria-hidden />
              Analytical and educational use only. Model output is not a determination of severity
              or causation.
            </p>
          </motion.div>
          <RoadNetwork />
        </div>
      </section>

      {/* ---------------------------------------------------------------- PURPOSE */}
      <section className="mx-auto max-w-7xl px-4 py-16 sm:px-6">
        <FadeIn>
          <p className="kicker">Project purpose</p>
          <h2 className="mt-2 max-w-3xl text-2xl font-bold tracking-tight sm:text-3xl">
            Turn historical collision records into operational road-safety intelligence
          </h2>
          <p className="mt-4 max-w-3xl text-sm leading-relaxed text-[#93a4c0]">
            Road collision severity is unevenly distributed: most collisions are slight, but a small
            share are fatal or serious. ROADRISK Intelligence trains classical machine-learning
            models on collision-level information that is available when a new incident report is
            judged &mdash; road, environment, time and vehicles involved &mdash; to estimate the
            likely severity class, while a full analytics layer describes what five years of real
            data actually show.
          </p>
        </FadeIn>

        <Stagger className="mt-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {[
            { icon: Database, title: "Real data only", desc: "Five STATS19 collision CSVs (2021-2025). No fabricated rows, metrics or chart values." },
            { icon: ShieldAlert, title: "Leakage-controlled", desc: "Post-outcome fields such as casualty information are excluded by an audited whitelist." },
            { icon: TestTubes, title: "Honest evaluation", desc: "Stratified 70/15/15 split, untouched test set, minority-class metrics reported as measured." },
            { icon: Gauge, title: "Practical inference", desc: "A scikit-learn pipeline served once-loaded by FastAPI - no retraining inside the API." },
          ].map((f) => (
            <StaggerItem key={f.title}>
              <div className="card card-hover h-full p-5">
                <f.icon className="h-5 w-5 text-[#f59e0b]" aria-hidden />
                <h3 className="mt-3 text-sm font-bold">{f.title}</h3>
                <p className="mt-2 text-xs leading-relaxed text-[#93a4c0]">{f.desc}</p>
              </div>
            </StaggerItem>
          ))}
        </Stagger>
      </section>

      {/* ---------------------------------------------------------------- WORKFLOW */}
      <section className="border-y border-[#1e2a42] bg-[#0e1626]">
        <div className="mx-auto max-w-7xl px-4 py-16 sm:px-6">
          <FadeIn>
            <p className="kicker">ML workflow</p>
            <h2 className="mt-2 text-2xl font-bold tracking-tight sm:text-3xl">
              From raw CSV to live prediction
            </h2>
          </FadeIn>
          <Stagger className="mt-8 grid gap-3 sm:grid-cols-2 lg:grid-cols-4" gap={0.05}>
            {WORKFLOW.map((step, i) => (
              <StaggerItem key={step.label}>
                <div className="group flex h-full items-center gap-3 rounded-xl border border-[#1e2a42] bg-[#111a2b] px-4 py-3.5 transition-colors hover:border-[#f59e0b]/50">
                  <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-[#f59e0b]/12 text-xs font-bold text-[#f59e0b]">
                    {String(i + 1).padStart(2, "0")}
                  </span>
                  <step.icon className="h-4 w-4 shrink-0 text-[#93a4c0] transition-colors group-hover:text-[#f59e0b]" aria-hidden />
                  <span className="text-[13px] font-medium text-[#cbd7ec]">{step.label}</span>
                </div>
              </StaggerItem>
            ))}
          </Stagger>
        </div>
      </section>

      {/* ---------------------------------------------------------------- DATASET */}
      <section className="mx-auto max-w-7xl px-4 py-16 sm:px-6">
        <div className="grid items-start gap-8 lg:grid-cols-2">
          <FadeIn>
            <p className="kicker">Dataset</p>
            <h2 className="mt-2 text-2xl font-bold tracking-tight sm:text-3xl">
              UK DfT STATS19 collision data
            </h2>
            <p className="mt-4 text-sm leading-relaxed text-[#93a4c0]">
              STATS19 is the Department for Transport&apos;s open dataset of personal-injury
              road collisions reported by the police. This project uses the collision-level files
              for {summary?.years?.length ? summary.years.join(", ") : "2021-2025"} with the
              <code className="mx-1 rounded bg-[#1a2438] px-1.5 py-0.5 text-xs text-[#f59e0b]">collision_severity</code>
              target (1 = Fatal, 2 = Serious, 3 = Slight, verified at runtime).
            </p>
            <div className="mt-6 grid gap-3 sm:grid-cols-3">
              <div className="card p-4">
                <p className="text-xs uppercase tracking-widest text-[#93a4c0]">Rows</p>
                <p className="mt-1.5 text-xl font-bold text-[#f59e0b]">
                  {loading ? <Skeleton className="h-6 w-24" /> : summary ? <AnimatedCounter value={summary.total_collisions} /> : "-"}
                </p>
              </div>
              <div className="card p-4">
                <p className="text-xs uppercase tracking-widest text-[#93a4c0]">Years</p>
                <p className="mt-1.5 text-xl font-bold text-[#f59e0b]">
                  {loading ? <Skeleton className="h-6 w-16" /> : summary?.dataset_period ?? "-"}
                </p>
              </div>
              <div className="card p-4">
                <p className="text-xs uppercase tracking-widest text-[#93a4c0]">Target</p>
                <p className="mt-1.5 text-xl font-bold text-[#f59e0b]">Severity</p>
              </div>
            </div>
            {error && <p className="mt-3 text-xs text-[#f87171]">{error}</p>}
            <p className="mt-4 text-xs text-[#64748b]">
              Source: data.gov.uk / DfT Road Safety Data, Open Government Licence v3.0. Synthetic
              fixtures exist only inside tests/ and are refused by the training loader.
            </p>
          </FadeIn>

          <FadeIn delay={0.12}>
            <p className="kicker">Stakeholders</p>
            <div className="mt-4 grid gap-3">
              {STAKEHOLDERS.map((s) => (
                <div key={s.label} className="card card-hover flex items-start gap-4 p-4">
                  <span className="rounded-lg bg-[#f59e0b]/12 p-2.5">
                    <s.icon className="h-5 w-5 text-[#f59e0b]" aria-hidden />
                  </span>
                  <div>
                    <h3 className="text-sm font-bold">{s.label}</h3>
                    <p className="mt-1 text-xs leading-relaxed text-[#93a4c0]">{s.desc}</p>
                  </div>
                </div>
              ))}
            </div>
          </FadeIn>
        </div>
      </section>

      {/* ---------------------------------------------------------------- RESPONSIBLE USE */}
      <section className="border-t border-[#1e2a42] bg-[#0e1626]">
        <div className="mx-auto max-w-4xl px-4 py-14 text-center sm:px-6">
          <FadeIn>
            <Car className="mx-auto h-7 w-7 text-[#f59e0b]" aria-hidden />
            <h2 className="mt-4 text-xl font-bold sm:text-2xl">Responsible use</h2>
            <p className="mt-4 text-sm leading-relaxed text-[#93a4c0]">
              This platform supports analytical and educational work. Model output is an analytical
              prediction based on historical reported collision data and should not be treated as a
              definitive determination of collision severity or causation. It must not be used for
              insurance, enforcement or liability decisions about individuals, and descriptive
              associations in the data are never presented as causal effects.
            </p>
            <div className="mt-7 flex justify-center gap-3">
              <Link href="/methodology" className="btn-ghost text-sm">Read the methodology</Link>
              <Link href="/about" className="btn-primary text-sm">About this project</Link>
            </div>
          </FadeIn>
        </div>
      </section>
    </div>
  );
}
