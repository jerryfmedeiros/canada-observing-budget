"""
smoke_trend.py — is the rise in smoke over 2020-2025 real, or a change in detection?

The surface PM2.5 exceedance rate (>= 35 ug/m3) rises over the study years, but NAPS coverage
thins at the end (no 2024 Calgary rows, a single 2024 Edmonton station, no 2025), and the
smoke flag then leans on the METAR FU code. This compares the PM2.5 rate with two sources
whose sensitivity does not change from year to year:
  FU   the observers' smoke code, logged every year at every station; coarse, but coarse the
       same way each year.
  AOD  MERRA-2 column 550 nm AOD, a reanalysis with the same sensitivity in every year and
       city, available for all of 2020-2025.
Rate = share of clear astronomical-dark reports (okta <= 2, Sun <= -18 deg) flagged.

Writes results/smoke_trend_sources.csv, which figures.py plots.

Usage: python3 smoke_trend.py
"""
from __future__ import annotations

import datetime as dt
from collections import defaultdict

import numpy as np

import observable_nights as on
from fetch_metar import fetch_metar_csv
from fetch_pm25 import build_pm25_index
from smoke_column import load_aod
from climatology import calculate_sun_alt
from sites import SITES

YEARS = range(2020, 2026)
PM25_SMOKE = 35.0
AOD_SMOKE = 0.5   # heavy haze / smoke column
WEST = ["calgary", "edmonton", "vancouver", "winnipeg"]
EAST = ["toronto", "montreal", "ottawa", "halifax"]


def city_trend(slug, site):
    aod = load_aod(slug)
    pm = build_pm25_index((2020, 2025), site.naps_city, site.naps_std_offset_h) if site.naps_city else {}
    n = defaultdict(int)
    fu = defaultdict(int); pmx = defaultdict(int); pm_n = defaultdict(int); aodx = defaultdict(int)
    for y in YEARS:
        for o in fetch_metar_csv(site.human, dt.date(y, 1, 1), dt.date(y, 12, 31)):
            if o.coverage_okta > 2:
                continue
            if calculate_sun_alt(o.timestamp, site.lat, site.lon, site.elevation_m) > -18.0:
                continue
            n[y] += 1
            hr = on.round_hour_utc(o.timestamp)
            if o.obscuration == "smoke":
                fu[y] += 1
            p = pm.get(hr)
            if p is not None:
                pm_n[y] += 1
                if p >= PM25_SMOKE:
                    pmx[y] += 1
            a = aod.get((f"{hr:%Y-%m-%d}", hr.hour))
            if a is not None and a >= AOD_SMOKE:
                aodx[y] += 1
    return n, fu, pmx, pm_n, aodx


def main():
    import csv
    from pathlib import Path
    print(f"Clear-dark-ob smoke rate (%) by year. FU = METAR smoke; PM25 = surface >=35 "
          f"(cov=PM2.5 obs); AOD = MERRA-2 550nm >= {AOD_SMOKE}.\n")
    pools = {"west": {k: defaultdict(int) for k in ("n", "fu", "pmx", "pm_n", "aodx")},
             "east": {k: defaultdict(int) for k in ("n", "fu", "pmx", "pm_n", "aodx")}}
    for region, slugs in (("west", WEST), ("east", EAST)):
        for slug in slugs:
            n, fu, pmx, pm_n, aodx = city_trend(slug, SITES[slug])
            if region == "west":
                rate = lambda num, den: " ".join(
                    f"{100*num[y]/den[y]:6.1f}" if den[y] else f"{'--':>6s}" for y in YEARS)
                print(f"=== {slug.upper()} ===   " + " ".join(f"{y:>6d}" for y in YEARS))
                print(f"  FU  rate %          " + rate(fu, n))
                print(f"  PM25 rate % (cov)   " + rate(pmx, pm_n))
                print(f"  AOD rate %          " + rate(aodx, n) + "\n")
            for y in YEARS:
                for k, d in (("n", n), ("fu", fu), ("pmx", pmx), ("pm_n", pm_n), ("aodx", aodx)):
                    pools[region][k][y] += d[y]

    rows = []
    for region in ("west", "east"):
        p = pools[region]
        for y in YEARS:
            rows.append({"year": y, "region": region, "source": "fu",
                         "rate_pct": round(100 * p["fu"][y] / p["n"][y], 3) if p["n"][y] else ""})
            rows.append({"year": y, "region": region, "source": "pm25",
                         "rate_pct": round(100 * p["pmx"][y] / p["pm_n"][y], 3) if p["pm_n"][y] else ""})
            rows.append({"year": y, "region": region, "source": "aod",
                         "rate_pct": round(100 * p["aodx"][y] / p["n"][y], 3) if p["n"][y] else ""})
    out = Path("results") / "smoke_trend_sources.csv"
    out.parent.mkdir(exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["year", "region", "source", "rate_pct"])
        w.writeheader()
        w.writerows(rows)
    r = lambda reg, s: " ".join(
        f"{[x['rate_pct'] for x in rows if x['year']==y and x['region']==reg and x['source']==s][0]:>6}"
        if [x['rate_pct'] for x in rows if x['year']==y and x['region']==reg and x['source']==s][0] != "" else f"{'--':>6s}"
        for y in YEARS)
    print("=== WESTERN POOLED ===  " + " ".join(f"{y:>6d}" for y in YEARS))
    for s in ("fu", "pm25", "aod"):
        print(f"  {s:5s} rate %        " + r("west", s))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
