"""
smoke_clear_share.py — how much of the surface smoke falls under a clear sky.

The smoke table counts PM2.5 exceedances only on clear astronomical-dark report hours, so a
city whose heaviest smoke came under cloud shows a low rate. This script gives, for each
city over the whole year, how many astronomical-dark report hours had PM2.5 >= 35 ug/m3,
and what share of them were clear reports (the paper quotes Montreal's share). It reuses the
report reader of smoke_summer.py.

Usage: python3 smoke_clear_share.py
"""
from __future__ import annotations

import sys

try:
    import observable_nights as on
    from sites import SITES
    from smoke_summer import reports
except ImportError as e:
    sys.exit(f"Run from the repo root: {e}")

ALL_YEAR = tuple(range(1, 13))


def main():
    print(f"Astronomical-dark report hours with PM2.5 >= {on.PM25_SMOKE:.0f} ug/m3, all months, "
          f"2020-2025 (NAPS where available)\n")
    print(f"{'city':10s} {'exceedances':>11s} {'on clear':>8s} {'share':>6s}")
    for slug in SITES:
        rows = reports(slug, ALL_YEAR)
        ex = [r for r in rows if r[4] is not None and r[4] >= on.PM25_SMOKE]
        clr = sum(1 for r in ex if r[3])
        share = f"{100 * clr / len(ex):5.1f}%" if ex else "    -"
        print(f"{slug:10s} {len(ex):11d} {clr:8d} {share:>6s}")


if __name__ == "__main__":
    main()
