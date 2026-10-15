"""
threshold_sweep.py — the good-night count under stricter and looser cutoffs.

Recomputes the good-night count per city under strict, default and lenient values of the
transparency cutoff (default: median dark-hour PWV <= 15 mm) and, when BUDGET_JET=1, of the
jet-stream cutoffs (200 hPa wind <= 30 m/s, 10 m wind <= 8 m/s), to check that the city
order, not the count itself, is stable to the choice. The smoke flag and the observable bar
are held fixed.

It repeats the per-night classification of observable_nights.py: per night with at least 2
astronomical-dark reports, the usable fraction, the dark hours, the median ERA5 PWV and
winds over the night's dark report hours, and the smoke flag (METAR FU or PM2.5 >= 35 on any
clear dark report). Nights with fewer reports cannot reach the bar and are left out.

--corrected applies the bias correction (nearer-satellite gap, measured shape) to every
night's usable fraction before the observable test, so the check runs on the corrected
counts the paper reports rather than the raw ones.

Usage: python3 threshold_sweep.py [--corrected]
"""
from __future__ import annotations

import datetime as dt
from collections import defaultdict
from zoneinfo import ZoneInfo

import numpy as np

import observable_nights as on
from fetch_era5 import (build_pwv_index, build_wind_index, build_wind10_index, cache_dir_for)
from fetch_metar import fetch_metar_csv
from climatology import calculate_sun_alt
from sites import SITES
from nights import JET_TEST, OBSERVABLE_H, apply_correction
from gaps import gap
import argparse

YEARS = (2020, 2025)
MIN_OBS = 2
PM25_SMOKE = 35.0

# (label, PWV mm, 200 hPa wind m/s, 10 m wind m/s)
SETS = [("strict ", 10.0, 25.0, 6.0),
        ("default", 15.0, 30.0, 8.0),
        ("lenient", 20.0, 35.0, 10.0)]


def daterange(y0, y1):
    d = dt.date(y0, 1, 1)
    while d <= dt.date(y1, 12, 31):
        yield d
        d += dt.timedelta(days=1)


def per_night(site, gap_pp: float = 0.0):
    """Return list of (observable, pwv_med, w200_med, w10_med_or_None, smoky) over covered nights."""
    on.LAT, on.LON, on.ELEVATION_M = site.lat, site.lon, site.elevation_m
    on.LOCAL_TZ = ZoneInfo(site.iana_tz)
    tz = on.LOCAL_TZ
    area, cache = site.era5_area(), cache_dir_for(site.slug)
    pwv = build_pwv_index(YEARS, site.lat, site.lon, area, cache)
    w200 = build_wind_index(YEARS, site.lat, site.lon, area, cache)
    w10 = build_wind10_index(YEARS, site.lat, site.lon, area, cache)
    from fetch_pm25 import build_pm25_index
    pm = build_pm25_index(YEARS, site.naps_city, site.naps_std_offset_h) if site.naps_city else {}

    usable = defaultdict(list)
    npwv, nw200, nw10 = defaultdict(list), defaultdict(list), defaultdict(list)
    smoke = defaultdict(lambda: False)
    for y in range(YEARS[0], YEARS[1] + 1):
        for o in fetch_metar_csv(site.human, dt.date(y, 1, 1), dt.date(y, 12, 31)):
            if calculate_sun_alt(o.timestamp, site.lat, site.lon, site.elevation_m) > -18.0:
                continue
            nk = on.night_key(o.timestamp)
            u = o.coverage_okta <= 2
            usable[nk].append(u)
            hr = on.round_hour_utc(o.timestamp)
            for idx, store in ((pwv, npwv), (w200, nw200), (w10, nw10)):
                v = idx.get(hr)
                if v is not None:
                    store[nk].append(v)
            if u:
                p = pm.get(hr)
                if o.obscuration == "smoke" or (p is not None and p >= PM25_SMOKE):
                    smoke[nk] = True

    out = []
    for nk, us in usable.items():
        if len(us) < MIN_OBS or not (YEARS[0] <= nk.year <= YEARS[1]):
            continue
        win = on.dark_window(nk, -18.0)
        dh = (win[1] - win[0]).total_seconds() / 3600.0 if win else 0.0
        cd = float(apply_correction(np.mean(us), gap_pp)) * dh
        pmed = float(np.median(npwv[nk])) if nk in npwv else None
        wmed = float(np.median(nw200[nk])) if nk in nw200 else None
        w10med = float(np.median(nw10[nk])) if nk in nw10 else None
        out.append((cd >= OBSERVABLE_H, pmed, wmed, w10med, smoke[nk]))
    return out


def good_count(nights, pwv_t, w200_t, w10_t, n_years):
    n = 0
    for o2, pmed, wmed, w10med, smoky in nights:
        if not o2 or smoky:
            continue
        transparent = pmed is not None and pmed <= pwv_t
        calm_aloft = wmed is not None and wmed <= w200_t
        calm_surface = (w10med is None) or (w10med <= w10_t)
        if transparent and ((calm_aloft and calm_surface) or not JET_TEST):
            n += 1
    return n / n_years


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corrected", action="store_true")
    args = ap.parse_args()
    ny = YEARS[1] - YEARS[0] + 1
    print(f"{'city':10s} " + " ".join(f"{lbl:>8s}" for lbl, *_ in SETS))
    results = {}
    for slug, site in SITES.items():
        nights = per_night(site, gap("satellite", slug) if args.corrected else 0.0)
        results[slug] = [good_count(nights, p, w, w10, ny) for _, p, w, w10 in SETS]
        print(f"{slug:10s} " + " ".join(f"{g:8.0f}" for g in results[slug]))
    print("\nRanking (city order) under each set:")
    for i, (lbl, *_ ) in enumerate(SETS):
        order = sorted(results, key=lambda s: -results[s][i])
        print(f"  {lbl}: " + " > ".join(f"{s[:3]}" for s in order))


if __name__ == "__main__":
    main()
