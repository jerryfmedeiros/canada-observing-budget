"""
good_months.py — when in the year the good nights fall.

From the corrected good nights (nearer-satellite gap) of each city: the share falling between
September and May, the share in March and October together, the two best months, and the
share of July nights that pass the transparency (PWV) test, of the July nights with any
astronomical-dark report.

Usage: python3 good_months.py
"""
from __future__ import annotations

import sys

import numpy as np

try:
    from gaps import gap
    from nights import OBSERVABLE_H, corrected_frac, per_night
    from sites import SITES
except ImportError as e:
    sys.exit(f"Run from the repo root: {e}")

NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def main():
    print("Corrected good nights by part of the year, 2020-2025\n")
    print(f"{'city':10s} {'good':>5s} {'Sep-May':>8s} {'Mar+Oct':>8s}  {'best two (ties)':22s} "
          f"{'Jul transp':>9s}")
    for slug, site in SITES.items():
        rec = per_night(site)
        months = np.array([d.month for d in rec.dates])
        obs = (corrected_frac(rec, gap("satellite", slug)) * rec.dark_h >= OBSERVABLE_H) & rec.valid
        good = obs & rec.quality
        by_m = np.array([good[months == m].sum() for m in range(1, 13)])
        sep_may = by_m[[8, 9, 10, 11, 0, 1, 2, 3, 4]].sum() / by_m.sum()
        mar_oct = (by_m[2] + by_m[9]) / by_m.sum()
        order = np.argsort(-by_m, kind="stable")
        best = [m for m in order if by_m[m] >= by_m[order[1]]]   # top two, ties included
        jul = (months == 7) & ~np.isnan(rec.pwv_median)
        jul_pass = rec.transparent[jul].mean()
        print(f"{slug:10s} {by_m.sum():5d} {100 * sep_may:7.1f}% {100 * mar_oct:7.1f}%  "
              f"{', '.join(f'{NAMES[m]} {by_m[m]}' for m in best):22s} {100 * jul_pass:9.1f}%")


if __name__ == "__main__":
    main()
