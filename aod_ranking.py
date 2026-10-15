"""
aod_ranking.py — the good-night ranking with a column-AOD limit added to transparency.

The transparency test uses water vapour alone (median dark-hour PWV <= 15 mm), but column
aerosol also dims the sky. This recomputes the good-night count with the added condition
that the night's median dark-hour MERRA-2 column AOD (550 nm) stays below a limit, and
reports how the city ranking moves. transparency_aod.py replaces PWV by AOD instead.

Uses the per-night classification of threshold_sweep.py (default cutoffs) and the MERRA-2
cache (fetch_merra2_aod.py).

--corrected applies the bias correction (nearer-satellite gap, measured shape) to every
night's usable fraction before the observable test, so the check runs on the corrected
counts the paper reports rather than the raw ones.

Usage: python3 aod_ranking.py [--corrected]
"""
from __future__ import annotations

import datetime as dt
from collections import defaultdict
from zoneinfo import ZoneInfo

import numpy as np

import observable_nights as on
from fetch_era5 import build_pwv_index, build_wind_index, build_wind10_index, cache_dir_for
from fetch_metar import fetch_metar_csv
from fetch_pm25 import build_pm25_index
from smoke_column import load_aod
from climatology import calculate_sun_alt
from sites import SITES
from nights import JET_TEST, OBSERVABLE_H, apply_correction
from gaps import gap
import argparse

YEARS = (2020, 2025)
MIN_OBS = 2
PM25_SMOKE = 35.0
PWV_T, W200_T, W10_T = 15.0, 30.0, 8.0       # default good-night cutoffs
AOD_TS = [None, 0.30, 0.20, 0.15]            # None = no AOD term (baseline)


def per_night(site, aod, gap_pp: float = 0.0):
    on.LAT, on.LON, on.ELEVATION_M = site.lat, site.lon, site.elevation_m
    on.LOCAL_TZ = ZoneInfo(site.iana_tz)
    cache = cache_dir_for(site.slug)
    pwv = build_pwv_index(YEARS, site.lat, site.lon, site.era5_area(), cache)
    w200 = build_wind_index(YEARS, site.lat, site.lon, site.era5_area(), cache)
    w10 = build_wind10_index(YEARS, site.lat, site.lon, site.era5_area(), cache)
    pm = build_pm25_index(YEARS, site.naps_city, site.naps_std_offset_h) if site.naps_city else {}

    usable = defaultdict(list)
    npwv, nw200, nw10, naod = (defaultdict(list) for _ in range(4))
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
                a = aod.get((f"{hr:%Y-%m-%d}", hr.hour))
                if a is not None:
                    naod[nk].append(a)

    out = []
    for nk, us in usable.items():
        if len(us) < MIN_OBS or not (YEARS[0] <= nk.year <= YEARS[1]):
            continue
        win = on.dark_window(nk, -18.0)
        dh = (win[1] - win[0]).total_seconds() / 3600.0 if win else 0.0
        o2 = (float(apply_correction(np.mean(us), gap_pp)) * dh) >= OBSERVABLE_H
        good = (o2 and not smoke[nk]
                 and nk in npwv and np.median(npwv[nk]) <= PWV_T
                 and (not JET_TEST or (nk in nw200 and np.median(nw200[nk]) <= W200_T
                                       and (nk not in nw10 or np.median(nw10[nk]) <= W10_T))))
        med_aod = float(np.median(naod[nk])) if naod[nk] else None
        out.append((good, med_aod))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corrected", action="store_true")
    args = ap.parse_args()
    ny = YEARS[1] - YEARS[0] + 1
    res = {}
    for slug, site in SITES.items():
        nights = per_night(site, load_aod(slug),
                           gap("satellite", slug) if args.corrected else 0.0)
        counts = []
        for thr in AOD_TS:
            if thr is None:
                c = sum(g for g, _ in nights)
            else:
                # AOD term only excludes; nights with no AOD sample are kept (rare)
                c = sum(g and (a is None or a <= thr) for g, a in nights)
            counts.append(c / ny)
        res[slug] = counts

    cols = ["baseline"] + [f"AOD<={t}" for t in AOD_TS[1:]]
    print(f"{'city':10s} " + " ".join(f"{c:>9s}" for c in cols))
    for slug in sorted(res, key=lambda s: -res[s][0]):
        print(f"{slug:10s} " + " ".join(f"{v:9.0f}" for v in res[slug]))
    print("\nRanking under each column:")
    for i, c in enumerate(cols):
        order = sorted(res, key=lambda s: -res[s][i])
        print(f"  {c:10s}: " + " > ".join(s[:3] for s in order))


if __name__ == "__main__":
    main()
