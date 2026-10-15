"""
fetch_pm25.py — hourly surface PM2.5 from ECCC NAPS, the quantitative half of the smoke flag.

METAR present weather (FU) records smoke only when an observer notes it at the surface, and
automated stations do not report it at all. Hourly surface PM2.5 is measured day and night,
so it catches smoky skies that the cloud report alone would call clear.

Source: Environment and Climate Change Canada, National Air Pollution Surveillance (NAPS)
program; public, no key. One CSV per pollutant per year covers every Canadian station, with
hour-ending values in local STANDARD time (no DST); each site's offset is in sites.py.

  https://data-donnees.az.ec.gc.ca/.../ContinuousData-DonneesContinu/HourlyData-DonneesHoraires/PM25_{year}.csv

Each hour's value is the mean over the city's reporting stations (smoke is regional, so a
gap at one station does not open a gap in the record). Run as a script, it prints the share
of astronomical-dark hours above each PM2.5 tier, by year and season.

Usage:
  python3 fetch_pm25.py --site calgary --years 2020 2025
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import subprocess
import time
import sys
from collections import defaultdict
from pathlib import Path

try:
    from climatology import calculate_sun_alt, get_season
except ImportError:
    sys.exit("Error: could not import climatology.py.")

PROJECT_ROOT = Path(__file__).parent.resolve()
CACHE_DIR = PROJECT_ROOT / "data" / "pm25_cache"

DEFAULT_CITY = "calgary"                          # substring of the NAPS city column (default site)
DEFAULT_STD_OFFSET_H = -7                         # Calgary MST (NAPS uses local STANDARD time)
MISSING = -999

NAPS_URL = (
    "https://data-donnees.az.ec.gc.ca/api/file?path="
    "/air/monitor/national-air-pollution-surveillance-naps-program/Data-Donnees/"
    "{year}/ContinuousData-DonneesContinu/HourlyData-DonneesHoraires/PM25_{year}.csv"
)

# Smoke tiers (ug/m3). 35 is the smoke threshold used throughout; 55 and 100 mark
# heavier smoke.
TIERS = (35.0, 55.0, 100.0)
UNAVAILABLE_TTL_S = 24 * 3600


def download_year(year: int) -> Path | None:
    """Download a year's PM2.5 file; return None if NAPS has not published it yet."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    local = CACHE_DIR / f"PM25_{year}.csv"
    if local.exists() and local.stat().st_size > 100_000:
        return local
    # A year NAPS has not published yet is remembered for a day, so repeated runs (every
    # script loads every city) do not re-ask the server dozens of times on a slow network.
    unavailable = CACHE_DIR / f"PM25_{year}.unavailable"
    if unavailable.exists() and time.time() - unavailable.stat().st_mtime < UNAVAILABLE_TTL_S:
        return None
    url = NAPS_URL.format(year=year)
    print(f"  GET {url}", file=sys.stderr)
    tmp = local.with_suffix(".part")
    r = subprocess.run(["curl", "-sS", "--fail", "-L", "--retry", "5", "--retry-delay", "5",
                        "--retry-connrefused", "--max-time", "600", "-o", str(tmp), url])
    tmp.unlink(missing_ok=True) if r.returncode != 0 else None
    if r.returncode == 22:  # curl --fail -> HTTP 4xx (year not posted yet)
        print(f"  (NAPS PM2.5 for {year} not available yet — skipping)", file=sys.stderr)
        unavailable.touch()
        return None
    if r.returncode != 0:
        raise RuntimeError(f"curl failed for {url}")
    tmp.replace(local)
    return local


def parse_city_hours(path: Path, city: str = DEFAULT_CITY,
                     std_offset_h: int = DEFAULT_STD_OFFSET_H) -> dict[dt.datetime, list[float]]:
    """Return {utc_hour: [pm25 from each matching station]} for one yearly NAPS file.

    `city` is a lowercase substring matched against the station-city column;
    `std_offset_h` is the city's LOCAL STANDARD-time UTC offset (NAPS has no DST)."""
    local_std = dt.timezone(dt.timedelta(hours=std_offset_h))
    by_hour: dict[dt.datetime, list[float]] = defaultdict(list)
    with open(path, encoding="latin-1", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            # Data rows start with the pollutant token and have >= 32 fields.
            if len(row) < 32 or not row[0].startswith("PM2"):
                continue
            row_city = row[3].strip().lower()
            if city not in row_city:
                continue
            try:
                day = dt.datetime.strptime(row[7].strip(), "%Y-%m-%d")
            except ValueError:
                continue
            for h in range(24):  # H01..H24 are hour-ending LST
                try:
                    val = float(row[8 + h])
                except (ValueError, IndexError):
                    continue
                if val <= MISSING:
                    continue
                # hour-ending h+1 in LST -> stamp at that wall-clock hour, to UTC
                lst = day.replace(tzinfo=local_std) + dt.timedelta(hours=h + 1)
                by_hour[lst.astimezone(dt.timezone.utc)].append(val)
    return by_hour


def build_pm25_index(years: tuple[int, int], city: str = DEFAULT_CITY,
                     std_offset_h: int = DEFAULT_STD_OFFSET_H) -> dict[dt.datetime, float]:
    """Mean PM2.5 (ug/m3) for `city` per UTC hour across the year range.

    The yearly NAPS CSV is national and shared across all cities (download once,
    filter per city); only `city`/`std_offset_h` differ between sites."""
    y0, y1 = years
    index: dict[dt.datetime, float] = {}
    for year in range(y0, y1 + 1):
        path = download_year(year)
        if path is None:
            continue
        per_hour = parse_city_hours(path, city, std_offset_h)
        for ts, vals in per_hour.items():
            if vals:
                index[ts] = sum(vals) / len(vals)
    return index


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", nargs=2, type=int, default=[2020, 2025])
    ap.add_argument("--site", default="calgary",
                    help="site slug from sites.py (sets NAPS city, std offset, coords)")
    args = ap.parse_args()
    y0, y1 = args.years

    from sites import get_site
    site = get_site(args.site)
    lat, lon, elev = site.lat, site.lon, site.elevation_m

    if site.naps_city is None:
        sys.exit(f"{site.name} has no NAPS city configured — nothing to fetch.")
    print(f"{site.name} nighttime PM2.5 smoke proxy {y0}-{y1} (NAPS, hourly)")
    print(f"Tiers (ug/m3): {', '.join(str(t) for t in TIERS)}")
    print("-" * 70)

    index = build_pm25_index((y0, y1), site.naps_city, site.naps_std_offset_h)
    print(f"Loaded {len(index)} {site.name} station-hours")

    # Classify each astronomical-dark (sun<=-18) hour by smoke tier.
    by_year = defaultdict(lambda: defaultdict(int))
    by_season = defaultdict(lambda: defaultdict(int))
    for ts, pm in index.items():
        if calculate_sun_alt(ts, lat, lon, elev) > -18.0:
            continue
        for bucket in (by_year[ts.year], by_season[get_season(ts.month)]):
            bucket["n"] += 1
            for t in TIERS:
                if pm >= t:
                    bucket[t] += 1

    def emit(title, table, keys):
        print(f"\n[{title}] fraction of astronomical-dark hours over PM2.5 tier")
        hdr = " | ".join(f">={int(t)}".rjust(7) for t in TIERS)
        print(f"{'bucket':<8} | {'dark-hrs':>8} | {hdr}")
        print("-" * (22 + 10 * len(TIERS)))
        for k in keys:
            c = table.get(k)
            if not c or c["n"] == 0:
                continue
            n = c["n"]
            cells = " | ".join(f"{c[t]/n:6.1%}" for t in TIERS)
            print(f"{str(k):<8} | {n:8d} | {cells}")

    emit("PM2.5 BY YEAR", by_year, range(y0, y1 + 1))
    emit("PM2.5 BY SEASON", by_season, ["WINTER", "SPRING", "SUMMER", "AUTUMN"])


if __name__ == "__main__":
    main()
