import Link from "next/link";
import { ShieldAlert } from "lucide-react";

export default function Footer() {
  return (
    <footer className="mt-16 border-t border-[#1e2a42] bg-[#0e1626]">
      <div className="mx-auto grid max-w-7xl gap-8 px-4 py-10 sm:px-6 md:grid-cols-3">
        <div>
          <p className="text-sm font-bold tracking-wider">ROADRISK INTELLIGENCE</p>
          <p className="mt-2 text-sm leading-relaxed text-[#93a4c0]">
            Machine-learning road collision severity prediction and road safety analytics built on
            UK Department for Transport STATS19 data.
          </p>
        </div>
        <div>
          <p className="text-xs font-semibold uppercase tracking-widest text-[#93a4c0]">Application</p>
          <ul className="mt-3 grid gap-2 text-sm">
            <li><Link className="hover:text-[#f59e0b]" href="/dashboard">Dashboard</Link></li>
            <li><Link className="hover:text-[#f59e0b]" href="/predict">Prediction Studio</Link></li>
            <li><Link className="hover:text-[#f59e0b]" href="/analytics">Analytics</Link></li>
            <li><Link className="hover:text-[#f59e0b]" href="/model">Model Performance</Link></li>
          </ul>
        </div>
        <div>
          <p className="flex items-center gap-2 text-xs font-semibold uppercase tracking-widest text-[#93a4c0]">
            <ShieldAlert className="h-4 w-4 text-[#f59e0b]" aria-hidden /> Responsible use
          </p>
          <p className="mt-3 text-sm leading-relaxed text-[#93a4c0]">
            Model output is an analytical prediction based on historical reported collision data and
            should not be treated as a definitive determination of collision severity or causation.
          </p>
          <p className="mt-3 text-xs text-[#64748b]">
            Contains public sector information licensed under the Open Government Licence v3.0.
          </p>
        </div>
      </div>
    </footer>
  );
}
