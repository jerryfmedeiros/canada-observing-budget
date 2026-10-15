"""
naps_coverage.py — what the NAPS PM2.5 files contain, city by city and year by year.

Replaces a one-off live-server log (results/naps_2024_coverage_check.log) with a check that
runs from the cached yearly files, so the coverage statements in the paper (no Calgary rows
in the 2024 national file; Edmonton's 2024 record on a single station) can be rebuilt
offline. Reports, per city and year, the number of station-days in the file, the number of
distinct stations, and the hours with at least one valid PM2.5 value. Each cached file's
size and SHA-256 are printed too, so a re-download that changed upstream shows up as a diff.

Usage: python3 naps_coverage.py
"""
from __future__ import annotations

import csv
import hashlib
import sys

try:
    from fetch_pm25 import CACHE_DIR, MISSING
    from sites import SITES
except ImportError as e:
    sys.exit(f"Run from the repo root: {e}")

YEARS = range(2020, 2025)


def scan(path):
    """{city_key: (station_days, stations, valid_hours)} for the study's NAPS city keys."""
    keys = {s.naps_city for s in SITES.values() if s.naps_city}
    days = {k: 0 for k in keys}
    stations = {k: set() for k in keys}
    hours = {k: 0 for k in keys}
    with open(path, encoding="latin-1", newline="") as f:
        for row in csv.reader(f):
            if len(row) < 32 or not row[0].startswith("PM2"):
                continue
            city = row[3].strip().lower()
            for k in keys:
                if k in city:
                    days[k] += 1
                    stations[k].add(row[2].strip())   # NAPS station ID (row[1] is the method code)
                    for h in range(24):
                        try:
                            if float(row[8 + h]) > MISSING:
                                hours[k] += 1
                        except (ValueError, IndexError):
                            pass
    return {k: (days[k], len(stations[k]), hours[k]) for k in keys}


def main():
    print("Cached NAPS PM2.5 files:")
    tables = {}
    for y in YEARS:
        p = CACHE_DIR / f"PM25_{y}.csv"
        if not p.exists():
            print(f"  {y}: MISSING ({p.name})")
            continue
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        print(f"  {y}: {p.stat().st_size:>10d} bytes  sha256 {sha}")
        tables[y] = scan(p)

    keys = sorted({s.naps_city for s in SITES.values() if s.naps_city})
    print("\nStation-days / stations / valid hours, by city and year:")
    print(f"{'city':12s} " + " ".join(f"{y:>20d}" for y in tables))
    for k in keys:
        cells = [f"{tables[y][k][0]:6d}/{tables[y][k][1]:2d}/{tables[y][k][2]:6d}"
                 for y in tables]
        print(f"{k:12s} " + " ".join(f"{c:>20s}" for c in cells))

    empty = [(k, y) for y in tables for k in keys if tables[y][k][0] == 0]
    single = [(k, y) for y in tables for k in keys if tables[y][k][1] == 1]
    print("\nCity-years with no rows at all: "
          + (", ".join(f"{k} {y}" for k, y in empty) or "none"))
    print("City-years resting on a single station: "
          + (", ".join(f"{k} {y}" for k, y in single) or "none"))


if __name__ == "__main__":
    main()
