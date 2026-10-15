"""
night_record.py — facts about the per-night record that the Methods quote.

For each city: the nights in the study calendar; the longest astronomical night and the usable
fraction it needs to hold the observable bar (so nights reported below it cannot count); the
longest astronomical night in each month May-August; and the nights whose fraction is imputed
from the monthly mean, by month, with how many of them reach the bar (raw and corrected).
Also prints the reference gap that scales the measured correction profile (nights.py).

Usage: python3 night_record.py
"""
from __future__ import annotations

import sys

import numpy as np

try:
    from gaps import gap
    from nights import OBSERVABLE_H, SHAPE_REF_GAP, corrected_frac, per_night
    from sites import SITES
except ImportError as e:
    sys.exit(f"Run from the repo root: {e}")

SUMMER = (5, 6, 7, 8)
MONTHS = "JFMAMJJASOND"


def main():
    print(f"Observable bar {OBSERVABLE_H:g} h of clear astronomical darkness; "
          f"correction reference gap (mean ceilometer gap, four paired cities) {SHAPE_REF_GAP:.2f} pp\n")
    print(f"{'city':10s} {'nights':>6s} {'longest h':>9s} {'frac needed':>11s}   "
          f"longest dark h:  {'May':>5s} {'Jun':>5s} {'Jul':>5s} {'Aug':>5s}")
    imputed_rows = []
    for slug, site in SITES.items():
        rec = per_night(site)
        months = np.array([d.month for d in rec.dates])
        longest = rec.dark_h.max()
        by_m = [rec.dark_h[months == m].max() for m in SUMMER]
        print(f"{slug:10s} {len(rec.dates):6d} {longest:9.2f} {OBSERVABLE_H / longest:11.3f}   "
              f"{'':16s} " + " ".join(f"{h:5.2f}" for h in by_m))
        obs_raw = (rec.frac * rec.dark_h >= OBSERVABLE_H) & rec.valid
        obs_cor = (corrected_frac(rec, gap("satellite", slug)) * rec.dark_h >= OBSERVABLE_H) & rec.valid
        imp = rec.imputed
        imputed_rows.append((slug, np.bincount(months[imp], minlength=13)[1:],
                             int((imp & obs_raw).sum()), int((imp & obs_cor).sum())))

    print("\nImputed nights (fewer than the minimum dark reports), by month, 2020-2025 total; "
          "how many reach the bar raw / corrected")
    print(f"{'city':10s} " + " ".join(f"{m:>3s}" for m in MONTHS) + f" {'total':>5s} {'obs':>7s}")
    for slug, bym, n_raw, n_cor in imputed_rows:
        print(f"{slug:10s} " + " ".join(f"{int(v):3d}" for v in bym)
              + f" {int(bym.sum()):5d} {n_raw:3d} / {n_cor:d}")


if __name__ == "__main__":
    main()
