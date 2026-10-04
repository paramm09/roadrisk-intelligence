"""Download the official STATS19 collision CSVs for the configured years into data/raw/.

    python scripts/download_data.py                  # uses ROADRISK_YEARS (default 2021-2025)
    python scripts/download_data.py --years 2020-2024
    python scripts/download_data.py --url 2025=https://.../your-2025-file.csv

Only the collision file is downloaded for each year (the casualty/vehicle files are NOT needed
and the casualty file would be a leakage risk). If the official URL pattern has changed, use
--url or edit DEFAULT_URL_TEMPLATE in ml/config.py (or set ROADRISK_URL_TEMPLATE).
"""
import sys
import urllib.error
import urllib.request

from _common import parser, run, settings_from

import pandas as pd

from ml.data_loader import map_columns, read_header
from ml.feature_config import TARGET


def _size(url):
    try:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0 roadrisk-academic"})
        with urllib.request.urlopen(req, timeout=30) as r:
            n = r.headers.get("Content-Length")
            return int(n) if n else None
    except Exception:
        return None


def download(url, dest):
    tmp = dest.with_suffix(".part")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 roadrisk-academic"})
    done = 0
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        while True:
            block = r.read(1 << 20)
            if not block:
                break
            f.write(block)
            done += len(block)
            if total:
                print(f"\r    {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    tmp.replace(dest)
    return done


def verify(path):
    cols, enc = read_header(path)
    if TARGET not in set(map_columns(cols).values()):
        raise ValueError(f"{path.name} has no severity column; wrong file? columns: {cols[:10]}")
    n = len(pd.read_csv(path, nrows=5, encoding=enc))
    if n == 0:
        raise ValueError(f"{path.name} contains no data rows")
    return len(cols)


def main():
    p = parser(__doc__)
    p.add_argument("--url", action="append", default=[], metavar="YEAR=URL", help="override URL for one year")
    p.add_argument("--force", action="store_true", help="re-download existing files")
    args = p.parse_args()
    cfg = settings_from(args)
    for item in args.url:
        y, u = item.split("=", 1)
        cfg.url_overrides[y.strip()] = u.strip()
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    for year in cfg.years:
        url = cfg.url_for(year)
        name = url.rsplit("/", 1)[-1] or f"collision-{year}.csv"
        dest = cfg.raw_dir / name
        print(f"\nYear        : {year}\nURL         : {url}\nFile name   : {name}\nDestination : {dest}")
        if dest.exists() and not args.force:
            print(f"Already present ({dest.stat().st_size / 1e6:.1f} MB) - skipping (use --force to re-download).")
        else:
            size = _size(url)
            print(f"Download size: {'%.1f MB' % (size / 1e6) if size else 'unknown'}")
            try:
                download(url, dest)
            except (urllib.error.URLError, OSError) as e:
                print(f"FAILED: {e}")
                failures.append((year, url, str(e)))
                continue
        try:
            ncols = verify(dest)
            print(f"Verified: readable CSV with {ncols} columns and a severity column.")
        except Exception as e:
            print(f"FAILED verification: {e}")
            failures.append((year, url, f"verification: {e}"))
    if failures:
        print("\nDOWNLOAD PROBLEMS:")
        for y, u, e in failures:
            print(f"  {y}: {e}\n     {u}")
        print("\nFix: pass --url YEAR=URL, edit ml/config.py DEFAULT_URL_TEMPLATE, or download manually "
              "into data/raw/ (see README 'Manual download fallback').")
        return 1
    print(f"\nAll {len(cfg.years)} year(s) present in {cfg.raw_dir}")
    return 0


if __name__ == "__main__":
    run(main)
