"""
correction_shape.py — how the nighttime clearing artifact is distributed over nights.

The bias correction needs two things: its size (each city's gap, from gaps.py) and its
shape, i.e. how much to take from a night given the usable fraction the observer reported.
Subtracting the same amount from every night is one assumption; it also takes from nights
the observer reported fully clear, so that such a night needs 4 / (1 - gap) hours of
darkness to count. This script measures the shape instead.

At the four cities with a screened fully automated partner, nights are grouped by the human
station's usable fraction over astronomical-dark reports. For each group the ceilometer's
mean usable fraction on the same nights is compared with the same quantity over daytime
units (Sun >= +6 deg) grouped the same way. The night-minus-day difference is the artifact
at that reported fraction: how much less clear the sky was, for the same human report, at
night than by day. The ceilometer carries no night-vision bias, and its own blindness to
high cloud is the same at every hour, so it cancels in the difference.

The pooled profile is written to data/derived/correction_shape.csv and is the shape
nights.corrected_frac applies (scaled by each city's gap). It is near zero on nights
reported fully clear and largest on partly cloudy nights.

Usage: python3 correction_shape.py
"""
from __future__ import annotations

import datetime as dt
import sys
from collections import defaultdict
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

try:
    import observable_nights as on
    from climatology import calculate_sun_alt
    from fetch_metar import fetch_metar_csv
    from sites import SITES
except ImportError as e:
    sys.exit(f"Run from the repo root: {e}")

YEARS = (2020, 2025)
MIN_OBS = 2
NOT_CEILOMETER = {"vancouver", "halifax"}   # their 'auto' neighbours are human-augmented
PAIRS = [(slug, s.human, s.auto) for slug, s in SITES.items()
         if s.auto and slug not in NOT_CEILOMETER]
# Human-fraction groups: none, three thirds of the partly cloudy range, and fully clear.
BINS = [(0.0, 1e-9, "0"), (1e-9, 1 / 3, "0-1/3"), (1 / 3, 2 / 3, "1/3-2/3"),
        (2 / 3, 1 - 1e-9, "2/3-1"), (1 - 1e-9, 1.01, "1")]
OUT = Path(__file__).parent / "data" / "derived" / "correction_shape.csv"


def units(icao: str, site):
    """({night: usable fraction over astronomical-dark reports}, {day: fraction over day reports})."""
    tz = ZoneInfo(site.iana_tz)
    on.LAT, on.LON, on.ELEVATION_M = site.lat, site.lon, site.elevation_m
    on.LOCAL_TZ = tz
    night, day = defaultdict(list), defaultdict(list)
    for y in range(YEARS[0], YEARS[1] + 1):
        for o in fetch_metar_csv(icao, dt.date(y, 1, 1), dt.date(y, 12, 31)):
            a = calculate_sun_alt(o.timestamp, site.lat, site.lon, site.elevation_m)
            u = o.coverage_okta <= on.USABLE_OKTA
            if a <= -18.0:
                night[on.night_key(o.timestamp)].append(u)
            elif a >= 6.0:
                day[o.timestamp.astimezone(tz).date()].append(u)
    keep = lambda d: {k: float(np.mean(v)) for k, v in d.items() if len(v) >= MIN_OBS}
    return keep(night), keep(day)


def main():
    pooled = {lab: dict(hn=[], hd=[], an=[], ad=[]) for _, _, lab in BINS}
    print("Mean AUTOMATED usable fraction on units grouped by the HUMAN usable fraction.")
    print("night - day = the artifact at that reported fraction (pp).\n")
    for slug, human, auto in PAIRS:
        site = SITES[slug]
        hn, hd = units(human, site)
        an, ad = units(auto, site)
        print(f"{slug} ({human} vs {auto})")
        print(f"  {'human fraction':15s} {'n night':>8s} {'auto night':>11s} {'n day':>7s} "
              f"{'auto day':>9s} {'night-day':>10s}")
        for lo, hi, lab in BINS:
            nk = [k for k in hn if k in an and lo <= hn[k] < hi]
            dk = [k for k in hd if k in ad and lo <= hd[k] < hi]
            a_n, a_d = [an[k] for k in nk], [ad[k] for k in dk]
            pooled[lab]["hn"] += [hn[k] for k in nk]
            pooled[lab]["hd"] += [hd[k] for k in dk]
            pooled[lab]["an"] += a_n
            pooled[lab]["ad"] += a_d
            print(f"  {lab:15s} {len(a_n):8d} {100 * np.mean(a_n):10.1f}% {len(a_d):7d} "
                  f"{100 * np.mean(a_d):8.1f}% {100 * (np.mean(a_n) - np.mean(a_d)):+9.1f}", flush=True)

    print("\nPOOLED (four cities)")
    print(f"  {'human fraction':15s} {'n night':>8s} {'mean human':>11s} {'auto night':>11s} "
          f"{'auto day':>9s} {'artifact pp':>12s}")
    rows = []
    for _, _, lab in BINS:
        p = pooled[lab]
        art = 100 * (np.mean(p["ad"]) - np.mean(p["an"]))      # positive = sky less clear at night
        rows.append((lab, len(p["an"]), float(np.mean(p["hn"])), art,
                     float(np.mean(p["hd"])), float(np.mean(p["an"])), float(np.mean(p["ad"]))))
        print(f"  {lab:15s} {len(p['an']):8d} {np.mean(p['hn']):11.3f} {100 * np.mean(p['an']):10.1f}% "
              f"{100 * np.mean(p['ad']):8.1f}% {art:+11.1f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        f.write("# Artifact (pp) by human-reported night usable fraction, pooled over the four\n"
                "# human/ceilometer pairs. Written by correction_shape.py; read by nights.py.\n"
                "# The last three columns (day mean human fraction, ceilometer night and day\n"
                "# usable fractions) let correction_scale.py map between the two scales.\n"
                "bin,n_night,mean_human_fraction,artifact_pp,mean_human_day,auto_night,auto_day\n")
        for lab, n, mh, art, mhd, an_, ad_ in rows:
            f.write(f"{lab},{n},{mh:.4f},{art:.2f},{mhd:.4f},{an_:.4f},{ad_:.4f}\n")
    print(f"\nwrote {OUT.relative_to(Path(__file__).parent)}")


if __name__ == "__main__":
    main()
