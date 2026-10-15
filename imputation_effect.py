"""
imputation_effect.py — what the monthly-mean imputation changes.

A night with fewer than two astronomical-dark reports (mostly short summer nights) takes its
month's mean usable fraction. Such nights have too little darkness to reach the observable
bar, so the imputation leaves the night counts unchanged. It matters for the annual
clear-dark hours: a sparse night still has a nonzero dark window, and dropping it would bias
the hour totals low. This gives clear-dark and moonless clear-dark hours per year with the
imputation (as reported) and without it (sparse nights excluded), per city.

Usage: python3 imputation_effect.py
"""
from __future__ import annotations

import datetime as dt
from collections import defaultdict
from zoneinfo import ZoneInfo

import numpy as np

import observable_nights as on
from fetch_metar import fetch_metar_csv
from climatology import calculate_sun_alt
from sites import SITES

YEARS = (2020, 2025)
MIN_OBS = 2


def daterange(y0, y1):
    d = dt.date(y0, 1, 1)
    while d <= dt.date(y1, 12, 31):
        yield d
        d += dt.timedelta(days=1)


def main():
    ny = YEARS[1] - YEARS[0] + 1
    print(f"{'city':10s} {'imp.nights/yr':>13s}  {'cd hrs w/o':>10s} {'cd hrs WITH':>11s} "
          f"{'Δ':>5s}  {'moon w/o':>8s} {'moon WITH':>9s} {'Δ':>5s}")
    for slug, site in SITES.items():
        on.LAT, on.LON, on.ELEVATION_M = site.lat, site.lon, site.elevation_m
        tz = ZoneInfo(site.iana_tz)
        dark18, moon18 = {}, {}
        for nd in daterange(*YEARS):
            win = on.dark_window(nd, -18.0)
            dark18[nd] = (win[1] - win[0]).total_seconds() / 3600.0 if win else 0.0
            moon18[nd] = on.moonless_hours(win) if win else 0.0

        night_obs = defaultdict(list)
        for y in range(YEARS[0], YEARS[1] + 1):
            for o in fetch_metar_csv(site.human, dt.date(y, 1, 1), dt.date(y, 12, 31)):
                if calculate_sun_alt(o.timestamp, site.lat, site.lon, site.elevation_m) <= -18.0:
                    nk = (o.timestamp.astimezone(tz) - dt.timedelta(hours=12)).date()
                    night_obs[nk].append(o.coverage_okta <= 2)

        frac = {nk: float(np.mean(v)) for nk, v in night_obs.items() if len(v) >= MIN_OBS}
        mm = defaultdict(list)
        for nk, f in frac.items():
            mm[nk.month].append(f)
        mm = {m: float(np.mean(v)) for m, v in mm.items()}

        cd_with = cd_wo = ml_with = ml_wo = 0.0
        imp = 0
        for nd, h in dark18.items():
            if nd in frac:
                cd_wo += frac[nd] * h
                ml_wo += frac[nd] * moon18[nd]
                cd_with += frac[nd] * h
                ml_with += frac[nd] * moon18[nd]
            elif nd.month in mm:                       # imputed
                imp += 1
                cd_with += mm[nd.month] * h
                ml_with += mm[nd.month] * moon18[nd]
        print(f"{slug:10s} {imp/ny:13.1f}  {cd_wo/ny:10.0f} {cd_with/ny:11.0f} "
              f"{(cd_with-cd_wo)/ny:+5.1f}  {ml_wo/ny:8.0f} {ml_with/ny:9.0f} "
              f"{(ml_with-ml_wo)/ny:+5.1f}")


if __name__ == "__main__":
    main()
