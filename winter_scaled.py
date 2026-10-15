"""
winter_scaled.py — what the year-round gap costs in winter.

The gap is measured on snow-free ground and applied all year. winter_dod.py shows that the
ceilometers, which snow does not affect, see the artifact in winter too, but not at the same
size everywhere: 6.2 pp against 15.0 all-season at Toronto, 9.7 against 12.3 at Montreal,
14.4 against 16.5 at Edmonton, 18.3 against 17.4 at Calgary. This scales the December to
February correction at those four cities by the winter / all-season ratio and reports what
changes.

Usage: python3 winter_scaled.py
"""
from __future__ import annotations

import sys

import numpy as np

try:
    from gaps import gap
    from nights import OBSERVABLE_H, apply_correction, per_night
    from sites import SITES
except ImportError as e:
    sys.exit(f"Run from the repo root: {e}")

# winter and all-season human-minus-ceilometer difference-of-differences (pp), winter_dod.py
WINTER_ALL = {"calgary": (18.3, 17.4), "edmonton": (14.4, 16.5),
              "toronto": (6.2, 15.0), "montreal": (9.7, 12.3)}


def main():
    print(f"{'city':10s} {'winter/all':>10s} | {'observable':>21s} | {'good':>21s} | "
          f"{'winter good':>21s}")
    print(f"{'':10s} {'':10s} | {'year-round':>10s} {'scaled':>10s} | {'year-round':>10s} "
          f"{'scaled':>10s} | {'year-round':>10s} {'scaled':>10s}")
    for slug, (w, a) in WINTER_ALL.items():
        rec = per_night(SITES[slug])
        ny = rec.dates[-1].year - rec.dates[0].year + 1
        g = gap("satellite", slug)
        winter = np.isin([d.month for d in rec.dates], [12, 1, 2])
        f_year = apply_correction(rec.frac, g)
        f_scaled = np.where(winter, apply_correction(rec.frac, g * w / a), f_year)
        row = []
        for f in (f_year, f_scaled):
            obs = (f * rec.dark_h >= OBSERVABLE_H) & rec.valid
            row.append((obs.sum() / ny, (obs & rec.quality).sum() / ny,
                        (obs & rec.quality & winter).sum() / ny))
        (o1, g1, w1), (o2, g2, w2) = row
        print(f"{slug:10s} {w / a:10.2f} | {o1:10.1f} {o2:10.1f} | {g1:10.1f} {g2:10.1f} | "
              f"{w1:10.1f} {w2:10.1f}", flush=True)


if __name__ == "__main__":
    main()
