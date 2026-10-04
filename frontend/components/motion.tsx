"use client";

/**
 * Framer Motion wrappers: fade / slide / stagger with scroll reveal.
 * All motion respects prefers-reduced-motion (framer-motion's useReducedMotion).
 */
import { motion, useReducedMotion, type Variants } from "framer-motion";
import type { ReactNode } from "react";
import { useHydrated } from "@/hooks/useHydrated";

export function FadeIn({
  children,
  delay = 0,
  className,
  y = 18,
}: {
  children: ReactNode;
  delay?: number;
  className?: string;
  y?: number;
}) {
  const hydrated = useHydrated();
  const prefersReducedMotion = useReducedMotion();
  const reduced = hydrated && prefersReducedMotion;
  return (
    <motion.div
      className={className}
      initial={{ opacity: 0, y: reduced ? 0 : y }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-60px" }}
      transition={{ duration: reduced ? 0 : 0.5, delay: reduced ? 0 : delay, ease: "easeOut" }}
    >
      {children}
    </motion.div>
  );
}

export function Stagger({
  children,
  className,
  gap = 0.08,
}: {
  children: ReactNode;
  className?: string;
  gap?: number;
}) {
  const hydrated = useHydrated();
  const prefersReducedMotion = useReducedMotion();
  const reduced = hydrated && prefersReducedMotion;
  const container: Variants = {
    hidden: {},
    show: { transition: { staggerChildren: reduced ? 0 : gap } },
  };
  return (
    <motion.div
      className={className}
      variants={container}
      initial="hidden"
      whileInView="show"
      viewport={{ once: true, margin: "-40px" }}
    >
      {children}
    </motion.div>
  );
}

export function StaggerItem({ children, className }: { children: ReactNode; className?: string }) {
  const hydrated = useHydrated();
  const prefersReducedMotion = useReducedMotion();
  const reduced = hydrated && prefersReducedMotion;
  const item: Variants = {
    hidden: { opacity: 0, y: reduced ? 0 : 16 },
    show: {
      opacity: 1,
      y: 0,
      transition: { duration: reduced ? 0 : 0.45, ease: "easeOut" },
    },
  };
  return (
    <motion.div className={className} variants={item}>
      {children}
    </motion.div>
  );
}
