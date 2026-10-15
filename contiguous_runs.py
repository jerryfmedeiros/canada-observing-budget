"""
contiguous_runs.py — the budget on the longest unbroken clear run instead of expected hours.

The budget counts expected clear-dark hours (usable fraction x dark hours), which a night of
scattered cloud can reach without one unbroken stretch. This recomputes the counts on the
longest contiguous clear run of each night and reports whether the city order changes.

Within each night's astronomical-dark window we step 15 min at a time; a step is clear if the
nearest dark METAR report within 30 min is usable (okta <= 2), and a missing or cloudy
report ends the run. The transparency, jet-stream and smoke flags are per night, as in
observable_nights.py.

--auto compares the human station with its automated partner at the four cities that have
one. Contiguity is a property of the report sequence, not of a night's usable fraction, so
the bias correction cannot be applied to it. Instead, this gives the share of nights that
meet the expected-hours bar and also hold an unbroken run of that length, by the human
station and by the ceilometer.

Usage: python3 contiguous_runs.py [--auto]
"""
from __future__ import annotations

import bisect
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
from nights import JET_TEST, OBSERVABLE_H
import argparse

YEARS = (2020, 2025)
MIN_OBS = 2
PM25_SMOKE = 35.0
PWV_T, W200_T, W10_T = 15.0, 30.0, 8.0
HOURS = [2, 3, 4, 5, 6]
STEP = 0.25      # h, discretisation
TOL = 0.5        # h, a step inherits the nearest report only within this gap (hourly METAR -> +-30 min)


def longest_run(obs, ws, we):
    """obs: sorted [(utc_datetime, usable_bool)]; ws,we dark-window bounds. Returns hours."""
    if not obs or we <= ws:
        return 0.0
    times = [o[0] for o in obs]
    best = cur = 0.0
    t = ws
    while t < we:
        i = bisect.bisect_left(times, t)
        cands = [j for j in (i - 1, i) if 0 <= j < len(times)]
        near = min(cands, key=lambda j: abs((times[j] - t).total_seconds()))
        gap_h = abs((times[near] - t).total_seconds()) / 3600.0
        clear = obs[near][1] if gap_h <= TOL else False
        cur = cur + STEP if clear else 0.0
        best = max(best, cur)
        t += dt.timedelta(hours=STEP)
    return best


def per_night(site):
    """Per covered night -> (contiguous_clear_run_hours, good_quality_bool)."""
    on.LAT, on.LON, on.ELEVATION_M = site.lat, site.lon, site.elevation_m
    on.LOCAL_TZ = ZoneInfo(site.iana_tz)
    cache = cache_dir_for(site.slug)
    pwv = build_pwv_index(YEARS, site.lat, site.lon, site.era5_area(), cache)
    w200 = build_wind_index(YEARS, site.lat, site.lon, site.era5_area(), cache)
    w10 = build_wind10_index(YEARS, site.lat, site.lon, site.era5_area(), cache)
    pm = build_pm25_index(YEARS, site.naps_city, site.naps_std_offset_h) if site.naps_city else {}

    dark_obs = defaultdict(list)   # nk -> [(ts, usable)]
    npwv, nw200, nw10 = defaultdict(list), defaultdict(list), defaultdict(list)
    smoke = defaultdict(lambda: False)
    for y in range(YEARS[0], YEARS[1] + 1):
        for o in fetch_metar_csv(site.human, dt.date(y, 1, 1), dt.date(y, 12, 31)):
            if calculate_sun_alt(o.timestamp, site.lat, site.lon, site.elevation_m) > -18.0:
                continue
            nk = on.night_key(o.timestamp)
            u = o.coverage_okta <= 2
            dark_obs[nk].append((o.timestamp, u))
            hr = on.round_hour_utc(o.timestamp)
            for idx, store in ((pwv, npwv), (w200, nw200), (w10, nw10)):
                v = idx.get(hr)
                if v is not None:
                    store[nk].append(v)
            if u and (o.obscuration == "smoke" or ((pm.get(hr) or 0) >= PM25_SMOKE)):
                smoke[nk] = True

    out = []
    for nk, obs in dark_obs.items():
        if len(obs) < MIN_OBS or not (YEARS[0] <= nk.year <= YEARS[1]):
            continue
        win = on.dark_window(nk, -18.0)
        if not win:
            continue
        obs.sort()
        run = longest_run(obs, win[0], win[1])
        quality = (not smoke[nk]
                   and nk in npwv and np.median(npwv[nk]) <= PWV_T
                   and (not JET_TEST or (nk in nw200 and np.median(nw200[nk]) <= W200_T
                                         and (nk not in nw10 or np.median(nw10[nk]) <= W10_T))))
        out.append((run, quality))
    return out


def station_nights(site, icao):
    """Per covered night -> (expected clear-dark hours, longest contiguous clear run)."""
    on.LAT, on.LON, on.ELEVATION_M = site.lat, site.lon, site.elevation_m
    on.LOCAL_TZ = ZoneInfo(site.iana_tz)
    dark_obs = defaultdict(list)
    for y in range(YEARS[0], YEARS[1] + 1):
        for o in fetch_metar_csv(icao, dt.date(y, 1, 1), dt.date(y, 12, 31)):
            if calculate_sun_alt(o.timestamp, site.lat, site.lon, site.elevation_m) > -18.0:
                continue
            dark_obs[on.night_key(o.timestamp)].append((o.timestamp, o.coverage_okta <= 2))
    out = []
    for nk, obs in dark_obs.items():
        if len(obs) < MIN_OBS or not (YEARS[0] <= nk.year <= YEARS[1]):
            continue
        win = on.dark_window(nk, -18.0)
        if not win:
            continue
        obs.sort()
        dh = (win[1] - win[0]).total_seconds() / 3600.0
        out.append((float(np.mean([u for _, u in obs])) * dh, longest_run(obs, win[0], win[1])))
    return out


NOT_CEILOMETER = {"vancouver", "halifax"}   # their 'auto' neighbours are human-augmented


def main_auto():
    bar = OBSERVABLE_H
    print(f"Nights meeting the {bar:g} h expected-hours bar that also hold an unbroken "
          f"{bar:g} h clear run, human station vs fully automated partner\n")
    print(f"{'city':10s} {'station':8s} {'nights/yr >= bar':>16s} {'with run >= bar':>16s} "
          f"{'share':>6s}")
    for slug, site in SITES.items():
        if not site.auto or slug in NOT_CEILOMETER:
            continue
        for icao in (site.human, site.auto):
            n = station_nights(site, icao)
            ny = YEARS[1] - YEARS[0] + 1
            meet = [r for e, r in n if e >= bar]
            both = sum(r >= bar for r in meet)
            print(f"{slug:10s} {icao:8s} {len(meet) / ny:16.1f} {both / ny:16.1f} "
                  f"{100 * both / max(len(meet), 1):5.0f}%", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--auto", action="store_true")
    if ap.parse_args().auto:
        return main_auto()
    ny = YEARS[1] - YEARS[0] + 1
    good = {}
    for slug, site in SITES.items():
        nights = per_night(site)
        good[slug] = {H: sum(r >= H and q for r, q in nights) / ny for H in HOURS}

    print("GOOD nights/yr by LONGEST CONTIGUOUS clear run (>= H h):")
    print(f"{'city':10s} " + " ".join(f"{'>='+str(H)+'h':>6s}" for H in HOURS))
    for slug in sorted(good, key=lambda s: -good[s][2]):
        print(f"{slug:10s} " + " ".join(f"{good[slug][H]:6.0f}" for H in HOURS))
    print("Ranking (contiguous):")
    for H in HOURS:
        order = sorted(good, key=lambda s: -good[s][H])
        print(f"  >= {H}h: " + " > ".join(s[:3] for s in order))


if __name__ == "__main__":
    main()
