"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { Menu, X, ShieldAlert, Activity } from "lucide-react";

const LINKS = [
  { href: "/", label: "Home" },
  { href: "/dashboard", label: "Dashboard" },
  { href: "/predict", label: "Prediction Studio" },
  { href: "/analytics", label: "Analytics" },
  { href: "/insights", label: "Insights" },
  { href: "/map", label: "Historical Map" },
  { href: "/model", label: "Model" },
  { href: "/methodology", label: "Methodology" },
  { href: "/about", label: "About" },
];

export default function Navbar() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 8);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  const closeMenu = () => setOpen(false);

  return (
    <header
      className={`sticky top-0 z-50 border-b transition-colors ${
        scrolled ? "bg-[#0b1220]/95 backdrop-blur border-[#1e2a42]" : "bg-transparent border-transparent"
      }`}
    >
      <nav className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3 sm:px-6" aria-label="Main">
        <Link href="/" className="group flex items-center gap-2.5">
          <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-[#f59e0b]/15 border border-[#f59e0b]/40">
            <Activity className="h-5 w-5 text-[#f59e0b]" aria-hidden />
          </span>
          <span className="leading-tight">
            <span className="block text-[15px] font-bold tracking-wider">ROADRISK</span>
            <span className="block text-[10px] tracking-[0.28em] text-[#93a4c0]">INTELLIGENCE</span>
          </span>
        </Link>

        <div className="hidden items-center gap-1 lg:flex">
          {LINKS.map((l) => {
            const active = pathname === l.href;
            return (
              <Link
                key={l.href}
                href={l.href}
                className={`rounded-md px-3 py-2 text-sm transition-colors ${
                  active
                    ? "bg-[#f59e0b]/12 text-[#f59e0b] font-semibold"
                    : "text-[#cbd7ec] hover:bg-white/5 hover:text-white"
                }`}
                aria-current={active ? "page" : undefined}
              >
                {l.label}
              </Link>
            );
          })}
        </div>

        <div className="hidden items-center gap-3 lg:flex">
          <span className="flex items-center gap-1.5 rounded-full border border-[#1e2a42] px-3 py-1 text-[11px] text-[#93a4c0]">
            <ShieldAlert className="h-3.5 w-3.5 text-[#f59e0b]" aria-hidden />
            Analytical use only
          </span>
        </div>

        <button
          type="button"
          className="rounded-md border border-[#1e2a42] p-2 lg:hidden"
          onClick={() => setOpen((o) => !o)}
          aria-expanded={open}
          aria-label={open ? "Close menu" : "Open menu"}
        >
          {open ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
        </button>
      </nav>

      {open && (
        <div className="border-t border-[#1e2a42] bg-[#0b1220] px-4 pb-4 lg:hidden">
          <div className="grid gap-1 pt-3">
            {LINKS.map((l) => {
              const active = pathname === l.href;
              return (
                <Link
                  key={l.href}
                  href={l.href}
                  onClick={closeMenu}
                  className={`rounded-md px-3 py-2.5 text-sm ${
                    active ? "bg-[#f59e0b]/12 text-[#f59e0b] font-semibold" : "text-[#cbd7ec] hover:bg-white/5"
                  }`}
                >
                  {l.label}
                </Link>
              );
            })}
          </div>
        </div>
      )}
    </header>
  );
}
