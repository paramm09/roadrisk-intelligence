"use client";

import { useCallback, useEffect, useRef, useState } from "react";

export interface UseApi<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
  refetch: () => void;
}

/** Minimal data-fetching hook: loading / error / data + refetch, abort-safe. */
export function useApi<T>(fetcher: () => Promise<T>, deps: unknown[] = []): UseApi<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [nonce, setNonce] = useState(0);
  const alive = useRef(true);

  useEffect(() => {
    alive.current = true;
    // Resetting loading/error when (re)fetching is the standard fetch-hook pattern:
    // the states are set synchronously here on purpose, before the async request starts.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setLoading(true);
    setError(null);
    fetcher()
      .then((d) => {
        if (alive.current) setData(d);
      })
      .catch((e: Error) => {
        if (alive.current) setError(e.message || String(e));
      })
      .finally(() => {
        if (alive.current) setLoading(false);
      });
    return () => {
      alive.current = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, nonce]);

  const refetch = useCallback(() => setNonce((n) => n + 1), []);
  return { data, error, loading, refetch };
}
