"use client";

/** Shared UI primitives: cards, states, animated counters, section headings. */
import { AlertTriangle, Inbox, type LucideIcon } from "lucide-react";
import { animate, useInView, useReducedMotion } from "framer-motion";
import { useEffect, useRef, useState, type ReactNode } from "react";
import { useHydrated } from "@/hooks/useHydrated";

export function Card({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <div className={`card p-5 ${className}`}>{children}</div>;
}

export function Panel({
  children,
  className = "",
  title,
  subtitle,
  action,
  id,
}: {
  children: ReactNode;
  className?: string;
  title?: string;
  subtitle?: string;
  action?: ReactNode;
  id?: string;
}) {
  return (
    <section className={`card p-5 ${className}`} id={id}>
      {(title || action) && (
        <header className="mb-4 flex items-start justify-between gap-3">
          <div>
            {title && <h3 className="text-sm font-bold uppercase tracking-wider text-[#cbd7ec]">{title}</h3>}
            {subtitle && <p className="mt-1 text-xs text-[#93a4c0]">{subtitle}</p>}
          </div>
          {action}
        </header>
      )}
      {children}
    </section>
  );
}

export function SectionHeading({
  kicker,
  title,
  description,
  center = false,
}: {
  kicker?: string;
  title: string;
  description?: string;
  center?: boolean;
}) {
  return (
    <div className={center ? "text-center" : ""}>
      {kicker && <p className="kicker">{kicker}</p>}
      <h2 className="mt-2 text-2xl font-bold tracking-tight sm:text-3xl">{title}</h2>
      {description && <p className={`mt-3 max-w-3xl text-sm leading-relaxed text-[#93a4c0] ${center ? "mx-auto" : ""}`}>{description}</p>}
    </div>
  );
}

export function Skeleton({ className = "" }: { className?: string }) {
  // span (not div): skeletons are rendered inside <p> elements in several cards,
  // and a <div> inside <p> is reparsed by the HTML parser -> hydration mismatch.
  return <span className={`block animate-pulse rounded-md bg-[#1a2438] ${className}`} aria-hidden />;
}

export function ChartSkeleton({ height = 260 }: { height?: number }) {
  return (
    <div className="space-y-3" style={{ height }} aria-label="Loading chart">
      <Skeleton className="h-4 w-1/3" />
      <div className="flex h-[85%] items-end gap-2">
        {Array.from({ length: 10 }).map((_, i) => (
          <Skeleton key={i} className="w-full" />
        ))}
      </div>
    </div>
  );
}

export function ErrorState({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div role="alert" className="flex flex-col items-center gap-3 rounded-xl border border-[#ef4444]/35 bg-[#ef4444]/8 px-6 py-8 text-center">
      <AlertTriangle className="h-7 w-7 text-[#ef4444]" aria-hidden />
      <p className="max-w-xl text-sm text-[#fca5a5]">{message}</p>
      {onRetry && (
        <button type="button" onClick={onRetry} className="btn-ghost text-sm">
          Retry
        </button>
      )}
    </div>
  );
}

export function EmptyState({ message }: { message: string }) {
  return (
    <div className="flex flex-col items-center gap-3 rounded-xl border border-dashed border-[#1e2a42] px-6 py-10 text-center">
      <Inbox className="h-7 w-7 text-[#93a4c0]" aria-hidden />
      <p className="text-sm text-[#93a4c0]">{message}</p>
    </div>
  );
}

/** Counts up to `value` when scrolled into view; snaps to final value if reduced motion. */
export function AnimatedCounter({
  value,
  decimals = 0,
  prefix = "",
  suffix = "",
  duration = 1.1,
}: {
  value: number;
  decimals?: number;
  prefix?: string;
  suffix?: string;
  duration?: number;
}) {
  const ref = useRef<HTMLSpanElement>(null);
  const inView = useInView(ref, { once: true, margin: "-40px" });
  const hydrated = useHydrated();
  const prefersReducedMotion = useReducedMotion();
  const reduced = hydrated && prefersReducedMotion;
  const [display, setDisplay] = useState(0);

  useEffect(() => {
    if (!inView || reduced) return;
    const controls = animate(0, value, {
      duration,
      ease: "easeOut",
      onUpdate: (v) => setDisplay(v),
    });
    return () => controls.stop();
  }, [inView, value, reduced, duration]);

  const shown = reduced ? value : display;

  return (
    <span ref={ref} className="tabular-nums">
      {prefix}
      {shown.toLocaleString(undefined, {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals,
      })}
      {suffix}
    </span>
  );
}

export function StatCard({
  label,
  value,
  sub,
  icon: Icon,
  accent = "#f59e0b",
  decimals = 0,
  prefix = "",
  suffix = "",
  loading = false,
}: {
  label: string;
  value?: number | string | null;
  sub?: string;
  icon?: LucideIcon;
  accent?: string;
  decimals?: number;
  prefix?: string;
  suffix?: string;
  loading?: boolean;
}) {
  return (
    <div className="card card-hover p-5">
      <div className="flex items-center justify-between">
        <p className="text-xs font-semibold uppercase tracking-widest text-[#93a4c0]">{label}</p>
        {Icon && (
          <span className="rounded-md p-1.5" style={{ background: `${accent}1f`, color: accent }}>
            <Icon className="h-4 w-4" aria-hidden />
          </span>
        )}
      </div>
      <p className="mt-3 text-2xl font-bold tracking-tight sm:text-3xl" style={{ color: accent }}>
        {loading ? (
          <Skeleton className="h-8 w-32" />
        ) : typeof value === "number" ? (
          <AnimatedCounter value={value} decimals={decimals} prefix={prefix} suffix={suffix} />
        ) : (
          (value ?? "-")
        )}
      </p>
      {sub && <p className="mt-1.5 text-xs text-[#93a4c0]">{sub}</p>}
    </div>
  );
}

export function SeverityBadge({ severity, small = false }: { severity: string; small?: boolean }) {
  const color =
    severity === "Fatal" ? "#ef4444" : severity === "Serious" ? "#f59e0b" : "#38bdf8";
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full font-semibold ${small ? "px-2 py-0.5 text-[11px]" : "px-3 py-1 text-xs"}`}
      style={{ background: `${color}1a`, color, border: `1px solid ${color}55` }}
    >
      <span className="h-1.5 w-1.5 rounded-full" style={{ background: color }} aria-hidden />
      {severity}
    </span>
  );
}
