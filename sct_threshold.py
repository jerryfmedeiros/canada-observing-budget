"""
sct_threshold.py — counting SCT as usable sky.

Recomputes the observable and good-night counts with SCT (okta <= 4) counted as usable as
well as the paper's SKC+FEW (okta <= 2), and reports the city order under each.

--corrected applies the bias correction (nearer-satellite gap, measured shape) to every
night's usable fraction before the observable test, so the check runs on the corrected
counts the paper reports rather than the raw ones. The SCT variant is corrected with its own gap (sct_gap.py), since the artifact is
smaller when SCT also counts as usable.

Usage: python3 sct_threshold.py [--corrected]
"""
from __future__ import annotations

import sys

try:
    import argparse
    from gaps import gap
    from nights import OBSERVABLE_H, corrected_frac, per_night
    from sites import SITES
except ImportError as e:
    sys.exit(f"Run from the repo root: {e}")

HOURS = OBSERVABLE_H   # nights.py; 4 h unless BUDGET_HOURS is set
VARIANTS = [(2, "SKC+FEW (paper)"), (4, "+SCT (loose)")]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corrected", action="store_true")
    args = ap.parse_args()
    gaps = {2: {s: (gap("satellite", s) if args.corrected else 0.0) for s in SITES}}
    if args.corrected:
        from sct_gap import sct_gaps
        gaps[4] = sct_gaps()
        print("Corrected: okta<=2 with the nearer-satellite gap, okta<=4 with its own (sct_gap.py)")
    else:
        gaps[4] = {s: 0.0 for s in SITES}
    obs, good = {}, {}
    for okta, _ in VARIANTS:
        obs[okta], good[okta] = {}, {}
        for slug, site in SITES.items():
            rec = per_night(site, usable_okta=okta)
            n_years = rec.dates[-1].year - rec.dates[0].year + 1
            o = ((corrected_frac(rec, gaps[okta][slug]) * rec.dark_h) >= HOURS) & rec.valid
            obs[okta][slug] = o.sum() / n_years
            good[okta][slug] = (o & rec.quality).sum() / n_years
        print(f"  computed okta<={okta}", file=sys.stderr, flush=True)

    for label, d in (("OBSERVABLE nights/yr", obs), ("GOOD nights/yr", good)):
        print(f"\n{label}:")
        print(f"{'city':10s} " + " ".join(f"{n:>17s}" for _, n in VARIANTS) + "   change")
        for slug in sorted(d[2], key=lambda s: -d[2][s]):
            a, b = d[2][slug], d[4][slug]
            print(f"{slug:10s} " + f"{a:17.1f} {b:17.1f}" + f"   {100 * (b - a) / a:+5.0f}%")
        print("Ranking:")
        for okta, name in VARIANTS:
            order = sorted(d[okta], key=lambda s: -d[okta][s])
            print(f"  {name:17s}: " + " > ".join(s[:3] for s in order))

    same_obs = (sorted(obs[2], key=lambda s: -obs[2][s])
                == sorted(obs[4], key=lambda s: -obs[4][s]))
    top3_abs = set(sorted(good[2], key=lambda s: -good[2][s])[:3])
    top3_sct = set(sorted(good[4], key=lambda s: -good[4][s])[:3])
    mean_lift = sum(obs[4][s] / obs[2][s] for s in obs[2]) / len(obs[2]) - 1
    print(f"\nCounting SCT raises the observable count by {100 * mean_lift:.0f}% on average.")
    print(f"  observable ordering unchanged at all eight positions: {same_obs}")
    print(f"  good-night top three unchanged: {top3_abs == top3_sct} "
          f"({', '.join(sorted(top3_abs))})")


if __name__ == "__main__":
    main()
