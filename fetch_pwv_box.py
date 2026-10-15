"""
fetch_pwv_box.py — ERA5 PWV for all eight cities, one request per month.

CDS charges and queues a request by its number of fields (hours x variables), not by its
area, and it processes one request per account at a time. fetch_pwv_long.py asks for each
city's small box separately (8 requests a month); this asks once a month for one box that
covers all eight, then cuts each city's own box out of it and writes it where
fetch_era5.build_pwv_index looks, under the same name. The cut keeps the same 0.25-degree
grid points the per-city request returns, so the cached files are interchangeable.

Usage: python3 fetch_pwv_box.py [--years 2001 2019] [--check YYYY-MM]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr

from fetch_era5 import _retrieve_month, cache_dir_for
from sites import SITES

PAD = 0.3   # degrees beyond the union of the city boxes


def union_area():
    a = [s.era5_area() for s in SITES.values()]
    return [round(max(x[0] for x in a) + PAD, 2), round(min(x[1] for x in a) - PAD, 2),
            round(min(x[2] for x in a) - PAD, 2), round(max(x[3] for x in a) + PAD, 2)]


def city_target(slug, y, m):
    return cache_dir_for(slug) / f"era5_pwv_{y}-{m:02d}.nc"


def cut(box: Path, slug: str, out: Path):
    n, w, s, e = SITES[slug].era5_area()
    with xr.open_dataset(box) as ds:
        sub = ds.where((ds.latitude <= n) & (ds.latitude >= s), drop=True)
        sub = sub.where((sub.longitude >= w) & (sub.longitude <= e), drop=True)
        if sub.sizes["latitude"] == 0 or sub.sizes["longitude"] == 0:
            raise RuntimeError(f"{slug}: no grid point inside {SITES[slug].era5_area()}")
        sub = sub.load()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part")
    sub.to_netcdf(tmp)
    tmp.replace(out)


def do_month(y, m, box_dir: Path, tries=6, wait=180):
    todo = [s for s in SITES if not city_target(s, y, m).exists()]
    if not todo:
        return True
    box = box_dir / f"box_pwv_{y}-{m:02d}.nc"
    for _ in range(tries):
        if box.exists() and box.stat().st_size > 1024:
            break
        try:
            _retrieve_month("reanalysis-era5-single-levels", ["total_column_water_vapour"],
                            y, m, box, area=union_area())
        except Exception as ex:
            print(f"  {y}-{m:02d} failed: {ex}", flush=True)
            time.sleep(wait)
    if not (box.exists() and box.stat().st_size > 1024):
        return False
    for s in todo:
        cut(box, s, city_target(s, y, m))
    box.unlink()
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--years", type=int, nargs=2, default=(2001, 2019))
    args = ap.parse_args()
    box_dir = Path(__file__).parent / "data" / "era5_box"
    box_dir.mkdir(parents=True, exist_ok=True)
    months = [(y, m) for y in range(args.years[0], args.years[1] + 1) for m in range(1, 13)]
    print(f"box {union_area()}; {len(months)} months", flush=True)
    for attempt in range(1, 4):
        left = [(y, m) for y, m in months if any(not city_target(s, y, m).exists() for s in SITES)]
        print(f"pass {attempt}: {len(left)} months with a city missing", flush=True)
        if not left:
            break
        for i, (y, m) in enumerate(left, 1):
            t = time.time()
            ok = do_month(y, m, box_dir)
            print(f"  {i}/{len(left)} {y}-{m:02d} {'ok' if ok else 'FAILED'} {time.time() - t:.0f}s", flush=True)
    left = [(y, m) for y, m in months if any(not city_target(s, y, m).exists() for s in SITES)]
    print(f"finished: {len(months) - len(left)} of {len(months)} months complete", flush=True)
    sys.exit(1 if left else 0)


if __name__ == "__main__":
    main()
