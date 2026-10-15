"""
fu_only.py — does uneven NAPS PM2.5 coverage move the budgets?

Flags smoke from the METAR FU code alone (the one smoke detector logged the same way at every
station in every year) and compares observable-smoke-clean and good-night counts with the
paper's FU-or-PM2.5 flag, raw and bias-corrected.

Usage: python3 fu_only.py
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

HOURS = OBSERVABLE_H   # nights.py; 4 h unless BUDGET_HOURS is set


def main():
    print(f"{'city':10s} {'':9s} {'good (FU+PM)':>13s} {'good (FU)':>10s} {'d':>5s}")
    worst = {"raw": 0.0, "corrected": 0.0}
    ranks = {}
    for label in ("raw", "corrected"):
        ranks[label] = {}
    for slug, site in SITES.items():
        both = per_night(site)
        fu = per_night(site, use_pm25=False)
        ny = both.dates[-1].year - both.dates[0].year + 1
        for label in ("raw", "corrected"):
            g = gap("satellite", slug) if label == "corrected" else 0.0
            obs = ((corrected_frac(both, g) * both.dark_h) >= HOURS) & both.valid
            a = (obs & both.quality).sum() / ny
            b = (obs & fu.quality).sum() / ny
            worst[label] = max(worst[label], abs(b - a))
            ranks[label][slug] = (a, b)
            print(f"{slug:10s} {label:9s} {a:13.1f} {b:10.1f} {b - a:+5.1f}", flush=True)

    for label in ("raw", "corrected"):
        r = ranks[label]
        o_both = sorted(r, key=lambda s: -r[s][0])
        o_fu = sorted(r, key=lambda s: -r[s][1])
        print(f"\n{label}: largest change {worst[label]:.1f} nights/yr; "
              f"ordering unchanged: {o_both == o_fu}")
        print(f"  FU+PM2.5: {' > '.join(s[:3] for s in o_both)}")
        print(f"  FU only : {' > '.join(s[:3] for s in o_fu)}")


if __name__ == "__main__":
    main()
