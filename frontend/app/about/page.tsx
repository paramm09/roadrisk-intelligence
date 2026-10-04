"use client";

/** About: objective, data source, stakeholders, limitations and responsible use. */
import {
  AlertTriangle,
  BookOpen,
  Database,
  FileWarning,
  History,
  Scale,
  ShieldAlert,
  Siren,
  Users,
} from "lucide-react";
import { SectionHeading } from "@/components/ui";
import { FadeIn, Stagger, StaggerItem } from "@/components/motion";
import { api } from "@/lib/api";
import { useApi } from "@/hooks/useApi";

const LIMITATIONS = [
  {
    icon: History,
    title: "Historical data only",
    body: "The model learns from what already happened between 2021 and 2025. It cannot anticipate one-off events, long-term trend changes, or the effect of interventions introduced after the training period.",
  },
  {
    icon: FileWarning,
    title: "Police-reported collisions only",
    body: "STATS19 records collisions serious enough to be reported to the police. Minor damage-only or unreported incidents are absent, and reporting practice varies between forces and over time.",
  },
  {
    icon: Scale,
    title: "No causal inference",
    body: "Feature importance and severity cross-tabs are associations in historical records. Nothing here establishes that changing a feature would change the outcome - confounding and exposure (traffic volume) are not adjusted for.",
  },
  {
    icon: AlertTriangle,
    title: "Minority-class performance is weak",
    body: "Fatal collisions are rare (~1.5%). Fatal recall and Fatal F1 are reported exactly as measured on the test split - including a poor-Fatal-F1 flag - and Fatal predictions must never be used to identify individual cases.",
  },
  {
    icon: Users,
    title: "Collision-level information only",
    body: "Vehicle and casualty files are deliberately not loaded (casualty outcomes would leak the target). Driver age, vehicle type and similar factors are future work.",
  },
  {
    icon: ShieldAlert,
    title: "Probabilities are model scores",
    body: "With class-weighted training the outputs are relative model scores, not calibrated real-world probabilities of severity. They must not be quoted as frequencies.",
  },
];

const STAKEHOLDERS = [
  "Road-safety officers",
  "Police analysts",
  "Local councils",
  "Transportation planners",
  "Researchers and students",
];

export default function AboutPage() {
  const summary = useApi(() => api.summary());

  return (
    <div className="mx-auto max-w-7xl px-4 py-10 sm:px-6">
      <FadeIn>
        <SectionHeading
          kicker="About the project"
          title="ROADRISK Intelligence"
          description="An academic end-to-end machine-learning application for road-collision severity prediction and road-safety analytics, built on real UK Department for Transport STATS19 data."
        />
      </FadeIn>

      <div className="mt-8 grid gap-6 lg:grid-cols-2">
        {/* ------------------------------------------------- objective */}
        <FadeIn>
          <div className="card h-full p-6">
            <h2 className="flex items-center gap-2 text-base font-bold">
              <BookOpen className="h-5 w-5 text-[#f59e0b]" aria-hidden /> Project objective
            </h2>
            <p className="mt-3 text-sm leading-relaxed text-[#93a4c0]">
              Predict the severity of a road collision (Slight / Serious / Fatal) using only
              information that would be available when a new incident report is assessed - road
              geometry, environment, time and vehicles involved - and provide a real analytics
              layer that describes five years of collision records for road-safety stakeholders.
            </p>
            <p className="mt-3 text-sm leading-relaxed text-[#93a4c0]">
              The whole ML lifecycle is implemented and reproducible: inspection, cleaning, EDA,
              feature engineering, leakage-controlled splitting, preprocessing, imbalance handling,
              baselines, multiple models, hyperparameter tuning, a single final test evaluation,
              a versioned artifact, a FastAPI service and this Next.js interface.
            </p>
          </div>
        </FadeIn>

        {/* ------------------------------------------------- dataset */}
        <FadeIn delay={0.08}>
          <div className="card h-full p-6">
            <h2 className="flex items-center gap-2 text-base font-bold">
              <Database className="h-5 w-5 text-[#f59e0b]" aria-hidden /> Dataset &amp; source
            </h2>
            <dl className="mt-3 space-y-2.5 text-sm">
              {[
                ["Source", "UK DfT STATS19 road safety open data (collision files)"],
                ["Publisher", "Department for Transport via data.gov.uk"],
                ["Licence", "Open Government Licence v3.0"],
                ["Period", summary.data?.dataset_period ?? (summary.loading ? "..." : "2021-2025")],
                ["Rows", summary.data ? summary.data.total_collisions.toLocaleString() : (summary.loading ? "..." : "-")],
                ["Target", "collision_severity (1 = Fatal, 2 = Serious, 3 = Slight)"],
              ].map(([k, v]) => (
                <div key={k} className="flex items-start justify-between gap-4 border-b border-[#152036] pb-2">
                  <dt className="text-xs uppercase tracking-wider text-[#93a4c0]">{k}</dt>
                  <dd className="text-right text-sm font-medium">{v}</dd>
                </div>
              ))}
            </dl>
            <p className="mt-3 text-xs leading-relaxed text-[#64748b]">
              The casualty and vehicle files are deliberately never loaded: they contain outcome
              information that would leak the target. Synthetic fixtures exist only inside tests/
              and are refused by the training loader.
            </p>
          </div>
        </FadeIn>

        {/* ------------------------------------------------- stakeholders */}
        <FadeIn delay={0.12}>
          <div className="card h-full p-6">
            <h2 className="flex items-center gap-2 text-base font-bold">
              <Siren className="h-5 w-5 text-[#f59e0b]" aria-hidden /> Intended stakeholders
            </h2>
            <ul className="mt-3 grid gap-2">
              {STAKEHOLDERS.map((s) => (
                <li key={s} className="flex items-center gap-2.5 rounded-lg border border-[#1e2a42] bg-[#0e1626] px-3 py-2.5 text-sm">
                  <span className="h-1.5 w-1.5 rounded-full bg-[#f59e0b]" aria-hidden />
                  {s}
                </li>
              ))}
            </ul>
            <p className="mt-3 text-xs leading-relaxed text-[#64748b]">
              The platform supports analysis and teaching. It is not a decision system for
              enforcement, insurance or liability.
            </p>
          </div>
        </FadeIn>

        {/* ------------------------------------------------- responsible use */}
        <FadeIn delay={0.16}>
          <div className="card h-full border-[#f59e0b]/40 p-6">
            <h2 className="flex items-center gap-2 text-base font-bold text-[#fcd34d]">
              <ShieldAlert className="h-5 w-5 text-[#f59e0b]" aria-hidden /> Responsible use
            </h2>
            <p className="mt-3 text-sm leading-relaxed text-[#cbd7ec]">
              Model output is an analytical prediction based on historical reported collision data
              and should not be treated as a definitive determination of collision severity or
              causation.
            </p>
            <ul className="mt-4 space-y-2 text-xs leading-relaxed text-[#93a4c0]">
              <li>- Do not use predictions to judge individual drivers, cases or claims.</li>
              <li>- Do not present descriptive associations as causal effects.</li>
              <li>- Do not describe the map as a future-accident forecast - it is a historical record.</li>
              <li>- Quote metrics only with their source report and test-split context.</li>
              <li>- Remember that police reporting practice shapes every statistic shown here.</li>
            </ul>
          </div>
        </FadeIn>
      </div>

      {/* ------------------------------------------------- limitations */}
      <FadeIn>
        <div className="mt-10">
          <SectionHeading
            kicker="Be honest about the limits"
            title="Limitations"
            description="Known limitations of the data, the model and this application - stated up front rather than buried."
          />
        </div>
      </FadeIn>

      <Stagger className="mt-6 grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {LIMITATIONS.map((l) => (
          <StaggerItem key={l.title}>
            <div className="card card-hover h-full p-5">
              <l.icon className="h-5 w-5 text-[#f59e0b]" aria-hidden />
              <h3 className="mt-3 text-sm font-bold">{l.title}</h3>
              <p className="mt-2 text-xs leading-relaxed text-[#93a4c0]">{l.body}</p>
            </div>
          </StaggerItem>
        ))}
      </Stagger>

      <FadeIn>
        <div className="mt-10 card p-5 text-xs leading-relaxed text-[#93a4c0]">
          <p>
            <strong className="text-[#cbd7ec]">Data attribution:</strong> contains public sector
            information licensed under the Open Government Licence v3.0. Statistics and model
            metrics shown in this application are computed at runtime from the processed dataset
            and the evaluation reports written by the pipeline.
          </p>
        </div>
      </FadeIn>
    </div>
  );
}
