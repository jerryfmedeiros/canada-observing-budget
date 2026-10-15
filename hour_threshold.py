"""
hour_threshold.py — the observable and good-night counts at bars of 2 to 6 hours (raw).

Sweeps the observable-night bar H from 2 to 6 h for both counts, per city, and reports the
city order at each H. The corrected counts at 2, 3, 4 and 6 h, which the paper quotes, are
in headline_corrected.py.

Per night, the expected clear-dark hours are cd = (usable fraction of dark reports) x (dark
hours), as in observable_nights.py. observable(H) means cd >= H; good(H) means observable(H),
smoke-clean and transparent (PWV <= 15 mm), plus the jet-stream test when BUDGET_JET=1. cd
is an expected value, not an unbroken run (see contiguous_runs.py).

Usage: python3 hour_threshold.py
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
from climatology import calculate_sun_alt
from sites import SITES
from nights import JET_TEST

YEARS = (2020, 2025)
MIN_OBS = 2
PM25_SMOKE = 35.0
PWV_T, W200_T, W10_T = 15.0, 30.0, 8.0
HOURS = [2, 3, 4, 5, 6]


def per_night(site):
    """Per covered night -> (cd_expected_clear_dark_hours, good_quality_bool)."""
    on.LAT, on.LON, on.ELEVATION_M = site.lat, site.lon, site.elevation_m
    on.LOCAL_TZ = ZoneInfo(site.iana_tz)
    cache = cache_dir_for(site.slug)
    pwv = build_pwv_index(YEARS, site.lat, site.lon, site.era5_area(), cache)
    w200 = build_wind_index(YEARS, site.lat, site.lon, site.era5_area(), cache)
    w10 = build_wind10_index(YEARS, site.lat, site.lon, site.era5_area(), cache)
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
            if u and (o.obscuration == "smoke" or ((pm.get(hr) or 0) >= PM25_SMOKE)):
                smoke[nk] = True

    out = []
    for nk, us in usable.items():
        if len(us) < MIN_OBS or not (YEARS[0] <= nk.year <= YEARS[1]):
            continue
        win = on.dark_window(nk, -18.0)
        dh = (win[1] - win[0]).total_seconds() / 3600.0 if win else 0.0
        cd = float(np.mean(us)) * dh
        quality = (not smoke[nk]
                   and nk in npwv and np.median(npwv[nk]) <= PWV_T
                   and (not JET_TEST or (nk in nw200 and np.median(nw200[nk]) <= W200_T
                                         and (nk not in nw10 or np.median(nw10[nk]) <= W10_T))))
        out.append((cd, quality))
    return out


def main():
    ny = YEARS[1] - YEARS[0] + 1
    obs, good = {}, {}
    for slug, site in SITES.items():
        nights = per_night(site)
        obs[slug] = {H: sum(cd >= H for cd, _ in nights) / ny for H in HOURS}
        good[slug] = {H: sum(cd >= H and q for cd, q in nights) / ny for H in HOURS}

    for label, d in (("OBSERVABLE nights/yr (cd >= H)", obs), ("GOOD nights/yr", good)):
        print(f"\n{label}:")
        print(f"{'city':10s} " + " ".join(f"{'>='+str(H)+'h':>6s}" for H in HOURS) + "   retain 2->4h")
        for slug in sorted(d, key=lambda s: -d[s][2]):
            row = d[slug]
            print(f"{slug:10s} " + " ".join(f"{row[H]:6.0f}" for H in HOURS)
                  + f"   {100*row[4]/row[2]:4.0f}%")
        print("Ranking:")
        for H in HOURS:
            order = sorted(d, key=lambda s: -d[s][H])
            print(f"  >= {H}h: " + " > ".join(s[:3] for s in order))


if __name__ == "__main__":
    main()
