"use client";

/** Recharts wrappers shared by every page. All data is passed in from API results. */
import type { BreakdownItem, SeverityItem } from "@/types";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export const SEVERITY_COLORS: Record<string, string> = {
  Slight: "#38bdf8",
  Serious: "#f59e0b",
  Fatal: "#ef4444",
};
export const SEVERITY_ORDER = ["Slight", "Serious", "Fatal"] as const;

export const GRID = "#1e2a42";
export const AXIS = "#93a4c0";

const tooltipStyle = {
  background: "#111a2b",
  border: "1px solid #1e2a42",
  borderRadius: 10,
  color: "#e6edf7",
  fontSize: 12.5,
} as const;

function pct(v: number) {
  return `${v.toFixed(1)}%`;
}

/* ------------------------------------------------------------------ severity donut */
export function SeverityDonut({
  data,
  height = 280,
  onSelect,
  selected,
}: {
  data: SeverityItem[];
  height?: number;
  /** Click a slice to cross-filter the dashboard by that severity (hover = tooltip only). */
  onSelect?: (severity: string) => void;
  selected?: string | null;
}) {
  const total = data.reduce((s, d) => s + d.count, 0);
  const emphasis = (s: string) => (selected === null || selected === undefined ? 1 : s === selected ? 1 : 0.3);
  const handleSectorClick = (sector: unknown) => {
    if (!onSelect) return;
    const entry = (sector ?? {}) as { payload?: SeverityItem; severity?: string };
    const severity = entry.payload?.severity ?? entry.severity;
    if (severity) onSelect(severity);
  };
  return (
    <div style={{ height }} className={`relative ${onSelect ? "cursor-pointer" : ""}`}>
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie
            data={data}
            dataKey="count"
            nameKey="severity"
            innerRadius="62%"
            outerRadius="88%"
            paddingAngle={2}
            stroke="#0b1220"
            strokeWidth={2}
            isAnimationActive
            onClick={onSelect ? handleSectorClick : undefined}
          >
            {data.map((d) => (
              <Cell
                key={d.severity}
                fill={SEVERITY_COLORS[d.severity] ?? "#64748b"}
                fillOpacity={emphasis(d.severity)}
                tabIndex={onSelect ? 0 : undefined}
                role={onSelect ? "button" : undefined}
                aria-label={
                  onSelect
                    ? `${d.severity}: ${d.count.toLocaleString()} collisions - activate to filter`
                    : undefined
                }
                onKeyDown={
                  onSelect
                    ? (e) => {
                        if (e.key === "Enter" || e.key === " ") {
                          e.preventDefault();
                          onSelect(d.severity);
                        }
                      }
                    : undefined
                }
              />
            ))}
          </Pie>
          <Tooltip
            contentStyle={tooltipStyle}
            formatter={(v, name) => {
              const n = Number(v ?? 0);
              return [`${n.toLocaleString()} (${total ? pct((100 * n) / total) : "0%"})`, String(name)];
            }}
          />
          <Legend
            formatter={(v: string) => <span style={{ color: "#cbd7ec", fontSize: 12.5 }}>{v}</span>}
          />
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
}

/* ---------------------------------------------------- stacked severity breakdown bar */
export function StackedSeverityBar({
  data,
  height = 300,
  xKey = "label",
  stacked = true,
  horizontal = false,
  onSelect,
  selectedKey,
}: {
  data: BreakdownItem[];
  height?: number;
  xKey?: "label";
  stacked?: boolean;
  horizontal?: boolean;
  /** Click a bar to cross-filter the dashboard by that category (hover = tooltip only). */
  onSelect?: (item: BreakdownItem) => void;
  /** Currently selected category key: it stays visible, all others are muted. */
  selectedKey?: number | string | null;
}) {
  const interactive = !!onSelect;
  const emphasis = (i: number) => {
    if (selectedKey === null || selectedKey === undefined) return 1;
    return String(data[i]?.key) === String(selectedKey) ? 1 : 0.3;
  };
  const handleClick = (state: unknown) => {
    if (!onSelect) return;
    const s = state as { activeTooltipIndex?: unknown; activeLabel?: unknown } | null;
    const idx = s?.activeTooltipIndex;
    let item: BreakdownItem | undefined;
    if (typeof idx === "number" && idx >= 0 && idx < data.length) {
      item = data[idx];
    } else if (typeof s?.activeLabel === "string") {
      item = data.find((d) => d.label === s.activeLabel);
    }
    if (item && item.key !== null && item.key !== undefined) onSelect(item);
  };
  const cells = (color: string) =>
    data.map((d, i) => (
      <Cell
        key={`${String(d.key)}-${i}`}
        fill={color}
        fillOpacity={emphasis(i)}
        tabIndex={interactive ? 0 : undefined}
        role={interactive ? "button" : undefined}
        aria-label={
          interactive ? `${d.label}: ${d.total.toLocaleString()} collisions - activate to filter` : undefined
        }
        onKeyDown={
          interactive
            ? (e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  if (d.key !== null && d.key !== undefined) onSelect?.(d);
                }
              }
            : undefined
        }
      />
    ));
  return (
    <div style={{ height }} className={interactive ? "cursor-pointer" : undefined}>
      <ResponsiveContainer width="100%" height="100%">
        {horizontal ? (
          <BarChart data={data} layout="vertical" margin={{ left: 8, right: 12, top: 4, bottom: 4 }} onClick={handleClick}>
            <CartesianGrid strokeDasharray="3 3" stroke={GRID} horizontal={false} />
            <XAxis type="number" stroke={AXIS} fontSize={11.5} tickLine={false} />
            <YAxis type="category" dataKey={xKey} stroke={AXIS} fontSize={11.5} tickLine={false} width={132} />
            <Tooltip contentStyle={tooltipStyle} cursor={{ fill: "#ffffff08" }} />
            <Bar dataKey="slight" stackId="s" name="Slight" fill={SEVERITY_COLORS.Slight} radius={horizontal ? [0, 0, 0, 0] : [4, 4, 0, 0]}>
              {cells(SEVERITY_COLORS.Slight)}
            </Bar>
            <Bar dataKey="serious" stackId="s" name="Serious" fill={SEVERITY_COLORS.Serious}>
              {cells(SEVERITY_COLORS.Serious)}
            </Bar>
            <Bar dataKey="fatal" stackId="s" name="Fatal" fill={SEVERITY_COLORS.Fatal} radius={[0, 4, 4, 0]}>
              {cells(SEVERITY_COLORS.Fatal)}
            </Bar>
          </BarChart>
        ) : (
          <BarChart data={data} margin={{ left: -14, right: 8, top: 8, bottom: 4 }} onClick={handleClick}>
            <CartesianGrid strokeDasharray="3 3" stroke={GRID} vertical={false} />
            <XAxis dataKey={xKey} stroke={AXIS} fontSize={11.5} tickLine={false} interval="preserveStartEnd" />
            <YAxis stroke={AXIS} fontSize={11.5} tickLine={false} />
            <Tooltip contentStyle={tooltipStyle} cursor={{ fill: "#ffffff08" }} />
            <Legend formatter={(v: string) => <span style={{ color: "#cbd7ec", fontSize: 12 }}>{v}</span>} />
            {stacked ? (
              <>
                <Bar dataKey="slight" stackId="s" name="Slight" fill={SEVERITY_COLORS.Slight} radius={[0, 0, 0, 0]}>
                  {cells(SEVERITY_COLORS.Slight)}
                </Bar>
                <Bar dataKey="serious" stackId="s" name="Serious" fill={SEVERITY_COLORS.Serious}>
                  {cells(SEVERITY_COLORS.Serious)}
                </Bar>
                <Bar dataKey="fatal" stackId="s" name="Fatal" fill={SEVERITY_COLORS.Fatal} radius={[4, 4, 0, 0]}>
                  {cells(SEVERITY_COLORS.Fatal)}
                </Bar>
              </>
            ) : (
              <Bar dataKey="total" name="Collisions" fill="#f59e0b" radius={[4, 4, 0, 0]}>
                {cells("#f59e0b")}
              </Bar>
            )}
          </BarChart>
        )}
      </ResponsiveContainer>
    </div>
  );
}

/* ------------------------------------------------------------------ line / area trend */
export function TrendLine({
  data,
  xKey = "label",
  yKey = "total",
  height = 280,
  color = "#f59e0b",
  area = true,
  name = "Collisions",
}: {
  data: object[];
  xKey?: string;
  yKey?: string;
  height?: number;
  color?: string;
  area?: boolean;
  name?: string;
}) {
  return (
    <div style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ left: -14, right: 8, top: 8, bottom: 4 }}>
          <defs>
            <linearGradient id={`grad-${yKey}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={color} stopOpacity={0.35} />
              <stop offset="100%" stopColor={color} stopOpacity={0.02} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke={GRID} vertical={false} />
          <XAxis dataKey={xKey} stroke={AXIS} fontSize={11.5} tickLine={false} />
          <YAxis stroke={AXIS} fontSize={11.5} tickLine={false} />
          <Tooltip contentStyle={tooltipStyle} />
          {area ? (
            <Area type="monotone" dataKey={yKey} name={name} stroke={color} strokeWidth={2} fill={`url(#grad-${yKey})`} />
          ) : (
            <Line type="monotone" dataKey={yKey} name={name} stroke={color} strokeWidth={2} dot={{ r: 2.5 }} />
          )}
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

/* ------------------------------------------------------------------ horizontal importance bars */
export function HorizontalBars({
  data,
  height = 320,
  color = "#f59e0b",
  nameKey = "label",
  formatter,
}: {
  data: { label: string; value: number }[];
  height?: number;
  color?: string;
  nameKey?: string;
  formatter?: (v: number) => string;
}) {
  return (
    <div style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ left: 4, right: 16, top: 4, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GRID} horizontal={false} />
          <XAxis type="number" stroke={AXIS} fontSize={11.5} tickLine={false} />
          <YAxis type="category" dataKey={nameKey} stroke={AXIS} fontSize={11.5} tickLine={false} width={168} />
          <Tooltip
            contentStyle={tooltipStyle}
            formatter={(v) => [formatter ? formatter(Number(v ?? 0)) : Number(v ?? 0).toFixed(4), "drop"]}
          />
          <Bar dataKey="value" fill={color} radius={[0, 4, 4, 0]} background={{ fill: "#101a2e" }} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
