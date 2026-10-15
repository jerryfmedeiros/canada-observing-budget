"""
fetch_pwv_long.py — populate the ERA5 PWV cache for 2001-2019 at all eight cities.

long_record.py --good needs nightly PWV back to 2001 to test whether the good-night ranking,
not only the observable one, depends on the six-year window. This fills the per-site monthly
cache that fetch_era5.build_pwv_index reads, one request at a time (CDS runs one request per
account at a time and accepts only about three queued), and retries refused months.

Usage: python3 fetch_pwv_long.py [--workers 1]
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from fetch_era5 import cache_dir_for, download_month
from sites import SITES

YEARS = range(2001, 2020)


def target(slug, y, m):
    return cache_dir_for(slug) / f"era5_pwv_{y}-{m:02d}.nc"


def fetch_one(slug, y, m, tries=6, wait=180):
    """One month, retried with a pause: CDS refuses new requests while too many are queued
    for the account ("Number queued requests ... temporarily limited"), and that refusal
    comes back at once, so without the pause a pass would run through its list unfetched."""
    for _ in range(tries):
        p = download_month("pwv", y, m, SITES[slug].era5_area(), cache_dir_for(slug))
        if p is not None:
            return p
        time.sleep(wait)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    jobs = [(slug, y, m) for slug in SITES for y in YEARS for m in range(1, 13)]
    for attempt in range(1, 6):
        todo = [j for j in jobs if not (target(*j).exists() and target(*j).stat().st_size > 1024)]
        print(f"pass {attempt}: {len(todo)} of {len(jobs)} months to fetch", flush=True)
        if not todo:
            break
        done = 0
        with ThreadPoolExecutor(args.workers) as ex:
            futs = {ex.submit(fetch_one, s, y, m): (s, y, m)
                    for s, y, m in todo}
            for f in as_completed(futs):
                done += 1
                s, y, m = futs[f]
                ok = f.result() is not None
                if done % 25 == 0 or not ok:
                    print(f"  {done}/{len(todo)}  last {s} {y}-{m:02d} {'ok' if ok else 'FAILED'}", flush=True)
    left = [j for j in jobs if not target(*j).exists()]
    print(f"finished: {len(jobs) - len(left)} of {len(jobs)} months cached", flush=True)
    sys.exit(1 if left else 0)


if __name__ == "__main__":
    main()
