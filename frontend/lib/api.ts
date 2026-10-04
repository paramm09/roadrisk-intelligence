// API client: every value rendered by the UI comes from these calls.
// Nothing is hardcoded here - no chart data, no metrics, no predictions.

const BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";

export const API_BASE = BASE;

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

/**
 * Application-level messages shown in the UI when the server answer is a raw,
 * unhelpful status ("Not Found", "HTTP 500", ...). The real HTTP status + body
 * is always logged to the console first, so debugging is never hidden.
 */
const FRIENDLY_MESSAGES: { prefix: string; message: string }[] = [
  { prefix: "/api/analytics/summary", message: "Unable to load dashboard statistics." },
  { prefix: "/api/analytics/map-points", message: "Unable to load historical collision data." },
  { prefix: "/api/insights", message: "Unable to load insights." },
  { prefix: "/api/models", message: "Unable to load available models." },
  { prefix: "/api/model/", message: "Unable to load prediction model information." },
  { prefix: "/api/predict", message: "Unable to run the prediction." },
  { prefix: "/api/analytics/", message: "Unable to load analytics data." },
];

function friendlyFor(path: string): string | undefined {
  return FRIENDLY_MESSAGES.find((m) => path.startsWith(m.prefix))?.message;
}

/** Raw status bodies carry no useful information for the user - swap those for a friendly message. */
function isGenericDetail(status: number, detail: string): boolean {
  const d = detail.trim().toLowerCase();
  if (!d) return true;
  return (
    d === "not found" ||
    d === "internal server error" ||
    d === "bad gateway" ||
    d === "forbidden" ||
    d === "method not allowed" ||
    /^http \d{3}$/.test(d)
  );
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const url = `${BASE}${path}`;
  const method = init?.method ?? "GET";
  let res: Response;
  try {
    res = await fetch(url, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
      cache: "no-store",
    });
  } catch (e) {
    const message = `Cannot reach the ROADRISK API at ${BASE}. Start it with: uvicorn backend.main:app --reload (${String(e)})`;
    console.error(`[ROADRISK API] ${method} ${url} -> unreachable:`, e);
    throw new ApiError(0, message);
  }
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      if (Array.isArray(body?.detail)) detail = body.detail.map((d: { msg?: string }) => d.msg ?? JSON.stringify(d)).join("; ");
      else if (typeof body?.detail === "string") detail = body.detail;
    } catch {
      /* keep generic detail */
    }
    // Preserve the real HTTP error for development/debugging (console).
    console.error(`[ROADRISK API] ${method} ${url} -> ${res.status} ${detail}`);
    // Show a useful application-level message instead of a raw "Not Found".
    const message = isGenericDetail(res.status, detail) ? (friendlyFor(path) ?? detail) : detail;
    throw new ApiError(res.status, message);
  }
  return (await res.json()) as T;
}

export function qs(filters: Record<string, unknown>): string {
  const parts = Object.entries(filters)
    .filter(([, v]) => v !== undefined && v !== null && v !== "")
    .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(String(v))}`);
  return parts.length ? `?${parts.join("&")}` : "";
}

export const api = {
  health: () => request<import("@/types").Health>("/health"),
  summary: () => request<import("@/types").Summary>("/api/analytics/summary"),
  /** Whole cross-filtered dashboard (KPI summary + every chart) in ONE request. */
  dashboard: (filters: Record<string, unknown> = {}) =>
    request<import("@/types").DashboardPayload>(`/api/analytics/dashboard${qs(filters)}`),
  modelInfo: () => request<import("@/types").ModelInfo>("/api/model/info"),
  models: () => request<import("@/types").ModelsResponse>("/api/models"),
  modelMetrics: () => request<import("@/types").ModelMetricsResponse>("/api/model/metrics"),
  featureImportance: () => request<import("@/types").FeatureImportance>("/api/model/feature-importance"),
  insights: () => request<import("@/types").Insights>("/api/insights"),
  breakdown: (path: string, filters: Record<string, unknown> = {}) =>
    request<import("@/types").Breakdown>(`/api/analytics/${path}${qs(filters)}`),
  severityDistribution: (filters: Record<string, unknown> = {}) =>
    request<import("@/types").SeverityDistribution>(`/api/analytics/severity-distribution${qs(filters)}`),
  mapPoints: (filters: Record<string, unknown> = {}) =>
    request<import("@/types").MapPoints>(`/api/analytics/map-points${qs(filters)}`),
  predict: (payload: import("@/types").PredictionPayload) =>
    request<import("@/types").PredictionResult>("/api/predict", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  predictBatch: (records: import("@/types").PredictionPayload[]) =>
    request<{ count: number; predictions: import("@/types").PredictionResult[] }>("/api/predict/batch", {
      method: "POST",
      body: JSON.stringify({ records }),
    }),
};
