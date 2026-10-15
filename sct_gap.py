"""
sct_gap.py — the bias-correction gap when SCT also counts as usable sky.

The gap is the human-minus-GOES night-day difference of differences with "usable" = SKC+FEW
(okta <= 2). Counting SCT as usable (sct_threshold.py) changes the human side of that
difference but not the satellite side, whose clear mask has no okta scale. The gap for the
looser definition is therefore

    gap_4 = gap_2 + (human_delta_4 - human_delta_2)

where human_delta_k is the human station's own night-day usable delta at okta <= k, measured
as the companion paper measures it: snow-free ground (ERA5-Land depth <= 1 cm on the date),
night = Sun <= -12 deg, day = Sun >= +6 deg, the mean per night or day over units with at
least 2 reports, 2020-2025. The okta <= 2 delta is first checked against the companion
paper's value. The shift is measured over all snow-free units rather than the GOES-matched
subset, which assumes the change of definition acts the same way on both.

Usage: python3 sct_gap.py
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import sys
from collections import defaultdict
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

try:
    from climatology import calculate_sun_alt
    from fetch_metar import fetch_metar_csv
    from gaps import gap
    from sites import SITES
except ImportError as e:
    sys.exit(f"Run from the repo root: {e}")

YEARS = (2020, 2025)
SNOW_MAX_M = 0.01
ROOT = Path(__file__).parent
PAPER1 = ROOT / "data" / "paper1" / "dynamic_tables.csv"


def snow_free_dates(slug: str) -> set[str]:
    d = json.loads((ROOT / "data" / "snow_cache" / f"snow_{slug}.json").read_text())["daily"]
    return {t for t, v in zip(d["time"], d["snow_depth_max"]) if v is not None and v <= SNOW_MAX_M}


def human_deltas(site, oktas=(2, 4)):
    """{okta: (delta_pp, n_night_units, n_day_units)} on snow-free units."""
    tz = ZoneInfo(site.iana_tz)
    free = snow_free_dates(site.slug)
    night = defaultdict(list)   # night key -> [okta]
    day = defaultdict(list)     # local date -> [okta]
    for y in range(YEARS[0], YEARS[1] + 1):
        for o in fetch_metar_csv(site.human, dt.date(y, 1, 1), dt.date(y, 12, 31)):
            a = calculate_sun_alt(o.timestamp, site.lat, site.lon, site.elevation_m)
            if a <= -12.0:
                nk = (o.timestamp.astimezone(tz) - dt.timedelta(hours=12)).date()
                night[nk].append(o.coverage_okta)
            elif a >= 6.0:
                day[o.timestamp.astimezone(tz).date()].append(o.coverage_okta)
    out = {}
    for k in oktas:
        nf = [np.mean([x <= k for x in v]) for d, v in night.items()
              if len(v) >= 2 and d.isoformat() in free]
        df = [np.mean([x <= k for x in v]) for d, v in day.items()
              if len(v) >= 2 and d.isoformat() in free]
        out[k] = (100.0 * (np.mean(nf) - np.mean(df)), len(nf), len(df))
    return out


def paper1_delta(slug: str, icao: str):
    with PAPER1.open() as f:
        for r in csv.DictReader(f):
            if r["site"] == slug and r["record"] == f"metar_{icao}" and r["threshold_cm"] == "1.0":
                return float(r["delta_pp"]), int(r["n_night"]), int(r["n_day"])
    return None


def sct_gaps(verbose: bool = False) -> dict[str, float]:
    """{slug: loose-definition satellite gap, pp}."""
    out = {}
    for slug, site in SITES.items():
        d = human_deltas(site)
        g2 = gap("satellite", slug)
        out[slug] = g2 + (d[4][0] - d[2][0])
        if verbose:
            ref = paper1_delta(slug, site.human)
            refs = f"{ref[0]:5.1f} ({ref[1]}/{ref[2]})" if ref else "  n/a"
            print(f"{slug:10s} {d[2][0]:5.1f} ({d[2][1]}/{d[2][2]})  {refs}   "
                  f"{d[4][0]:5.1f}   {g2:5.1f} -> {out[slug]:5.1f}", flush=True)
    return out


def main():
    print("Human night-day usable delta (pp, snow-free units night/day) and the loose gap\n")
    print(f"{'city':10s} {'okta<=2 here':>18s}  {'companion paper':>16s}   {'okta<=4':>7s}   "
          f"{'gap_2 -> gap_4':>14s}")
    sct_gaps(verbose=True)


if __name__ == "__main__":
    main()
