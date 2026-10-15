"""
seasonal_cycles.py — the seasonal cycles of transparency, the jet stream, and cloud.

For every city and season (DJF, MAM, JJA, SON), over the nights that have astronomical
darkness: the mean of the nightly median dark-hour PWV, the share of nights that pass the
jet-stream test (median 200 hPa wind <= 30 m/s and 10 m wind <= 8 m/s), and the mean usable
fraction of the dark reports, raw and bias-corrected (nights with their own reports only).

Usage: python3 seasonal_cycles.py
"""
from __future__ import annotations

import sys

import numpy as np

try:
    from gaps import gap
    from nights import corrected_frac, per_night
    from sites import SITES
except ImportError as e:
    sys.exit(f"Run from the repo root: {e}")

SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}


def main():
    print("Per season, nights with astronomical darkness: n nights, mean PWV (mm), jet test "
          "passed (%), usable fraction raw/corrected (%)\n")
    print(f"{'city':10s} " + " | ".join(f"{k:^27s}" for k in SEASONS))
    for slug, site in SITES.items():
        r = per_night(site)
        mon = np.array([d.month for d in r.dates])
        cf = corrected_frac(r, gap("satellite", slug))
        cells = []
        for ms in SEASONS.values():
            dk = np.isin(mon, ms) & (r.dark_h > 0)
            own = dk & r.valid & ~r.imputed
            cells.append(f"{dk.sum():4d} {np.nanmean(r.pwv_median[dk]):5.1f} "
                         f"{100 * r.low_jet[dk].mean():4.0f}% "
                         f"{100 * r.frac[own].mean():3.0f}/{100 * cf[own].mean():3.0f}")
        print(f"{slug:10s} " + " | ".join(cells), flush=True)


if __name__ == "__main__":
    main()
