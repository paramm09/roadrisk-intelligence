"""Raw STATS19 -> validated, cleaned, feature-engineered, split dataset (+ data/preprocessing reports)."""
from _common import parser, run, settings_from

from ml.prepare import run_prepare


def main():
    p = parser(__doc__)
    p.add_argument("--strict", action="store_true", help="fail if any requested year is missing")
    p.add_argument("--max-rows", type=int, help="DEVELOPMENT ONLY: cap rows per year (recorded in the reports)")
    args = p.parse_args()
    cfg = settings_from(args)
    if args.max_rows:
        cfg.max_rows_per_year = args.max_rows
    run_prepare(cfg, strict=args.strict)
    return 0


if __name__ == "__main__":
    run(main)
