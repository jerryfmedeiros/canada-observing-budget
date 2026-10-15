"""
surface_wind.py — what the 10 m wind condition adds to the low-jet test.

A low-jet night needs a median 200 hPa wind <= 30 m/s and a median 10 m wind <= 8 m/s. This
script counts low-jet nights a year (corrected, nearer-satellite gap) with and without the 10 m
condition, to show how many nights the surface condition removes on its own.

Usage: python3 surface_wind.py
"""
from __future__ import annotations

import sys

try:
    import observable_nights as on
    from gaps import gap
    from nights import OBSERVABLE_H, YEARS, corrected_frac, per_night
    from sites import SITES
except ImportError as e:
    sys.exit(f"Run from the repo root: {e}")

N_YEARS = YEARS[1] - YEARS[0] + 1


def lowjet_per_year(site, slug):
    rec = per_night(site)
    obs = (corrected_frac(rec, gap("satellite", slug)) * rec.dark_h >= OBSERVABLE_H) & rec.valid
    return (obs & rec.quality & rec.low_jet).sum() / N_YEARS


def main():
    print(f"Low-jet nights a year (corrected), with and without the 10 m wind <= "
          f"{on.WIND10_JET:g} m/s condition\n")
    print(f"{'city':10s} {'with':>6s} {'without':>8s} {'removed':>8s}")
    limit = on.WIND10_JET
    for slug, site in SITES.items():
        on.WIND10_JET = limit
        with_10m = lowjet_per_year(site, slug)
        on.WIND10_JET = float("inf")
        without = lowjet_per_year(site, slug)
        print(f"{slug:10s} {with_10m:6.2f} {without:8.2f} {without - with_10m:8.2f}")
    on.WIND10_JET = limit


if __name__ == "__main__":
    main()
