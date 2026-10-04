"use client";

import { useSyncExternalStore } from "react";

const subscribe = () => () => {};

/**
 * `false` on the server and during the first client render, `true` afterwards.
 * Use it to gate any browser-derived value so the initial render is identical
 * on both sides and the real value only lands after hydration.
 */
export function useHydrated(): boolean {
  return useSyncExternalStore(
    subscribe,
    () => true,
    () => false,
  );
}
