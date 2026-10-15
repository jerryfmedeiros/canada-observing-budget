"""
auto_flag.py — how many nighttime reports at the eight primary stations carry AUTO.

The paper selects stations that are human-augmented at night. auto_screen.py writes the
screen as rates to data/derived/<slug>/auto_screen.json; this script prints the exact counts
behind them (nighttime = Sun at or below -12 degrees, as in auto_screen.py), 2020-2025.

Usage: python3 auto_flag.py
"""
from __future__ import annotations

import datetime as dt
import sys

try:
    from auto_screen import NIGHT_ALT
    from climatology import calculate_sun_alt
    from fetch_metar import fetch_metar_csv
    from sites import SITES
except ImportError as e:
    sys.exit(f"Run from the repo root: {e}")

YEARS = range(2020, 2026)


def main():
    print(f"Nighttime (Sun <= {NIGHT_ALT:.0f} deg) METAR reports at the primary stations, "
          f"{YEARS[0]}-{YEARS[-1]}\n")
    print(f"{'city':10s} {'station':7s} {'reports':>8s} {'AUTO':>5s} {'share':>7s}")
    tot = tot_auto = 0
    for slug, site in SITES.items():
        n = n_auto = 0
        for y in YEARS:
            for o in fetch_metar_csv(site.human, dt.date(y, 1, 1), dt.date(y, 12, 31)):
                if calculate_sun_alt(o.timestamp, site.lat, site.lon, site.elevation_m) <= NIGHT_ALT:
                    n += 1
                    n_auto += bool(o.is_auto)
        tot, tot_auto = tot + n, tot_auto + n_auto
        print(f"{slug:10s} {site.human:7s} {n:8d} {n_auto:5d} {100 * n_auto / n:6.2f}%")
    print(f"{'all':10s} {'':7s} {tot:8d} {tot_auto:5d} {100 * tot_auto / tot:6.2f}%")


if __name__ == "__main__":
    main()
