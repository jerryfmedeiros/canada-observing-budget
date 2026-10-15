"""
pwv_percentile.py — absolute against city-relative transparency test.

The transparency test uses a fixed 15 mm of precipitable water, which by construction favours
the dry interior. This replaces it with each city's own Pth percentile, which passes the same
share of every city's nights, and reports the good-night counts and city order under both,
with the July pass rate under each.

--corrected applies the bias correction (nearer-satellite gap, measured shape) to every
night's usable fraction before the observable test, so the check runs on the corrected
counts the paper reports rather than the raw ones.

Usage: python3 pwv_percentile.py [--pct 40] [--corrected]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

try:
    from gaps import gap
    from nights import JET_TEST, OBSERVABLE_H, corrected_frac, per_night
    from sites import SITES
    import observable_nights as on
except ImportError as e:
    sys.exit(f"Run from the repo root: {e}")

HOURS = OBSERVABLE_H   # nights.py; 4 h unless BUDGET_HOURS is set


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pct", type=float, default=40.0,
                    help="percentile of each city's own dark-night PWV to use as its bar")
    ap.add_argument("--corrected", action="store_true")
    args = ap.parse_args()

    print(f"Transparency bar: absolute {on.PWV_TRANSPARENT:.0f} mm vs each city's own "
          f"{args.pct:.0f}th-percentile dark-night PWV\n")
    print(f"{'city':10s} {'own p' + str(int(args.pct)):>8s} {'vs 15mm':>8s} | "
          f"{'good(abs)':>10s} {'good(pct)':>10s} | {'July pass abs':>13s} {'pct':>6s}")

    abs_counts, pct_counts, pctile = {}, {}, {}
    for slug, site in SITES.items():
        rec = per_night(site)
        med = rec.pwv_median
        n_years = rec.dates[-1].year - rec.dates[0].year + 1
        g = gap("satellite", slug) if args.corrected else 0.0
        obs = ((corrected_frac(rec, g) * rec.dark_h) >= HOURS) & rec.valid

        have = ~np.isnan(med)
        # The city's own bar, over the nights that actually carry a dark-hour PWV median.
        p = float(np.percentile(med[have], args.pct))
        pctile[slug] = p

        transp_abs = have & (med <= on.PWV_TRANSPARENT)
        transp_pct = have & (med <= p)
        jet_ok = rec.low_jet if JET_TEST else np.ones(len(rec.dates), bool)
        good_abs = obs & rec.smoke_clean & transp_abs & jet_ok
        good_pct = obs & rec.smoke_clean & transp_pct & jet_ok
        abs_counts[slug] = good_abs.sum() / n_years
        pct_counts[slug] = good_pct.sum() / n_years

        july = np.array([d.month == 7 for d in rec.dates]) & have
        ja = 100 * transp_abs[july].sum() / july.sum() if july.sum() else float("nan")
        jp = 100 * transp_pct[july].sum() / july.sum() if july.sum() else float("nan")

        print(f"{slug:10s} {p:7.1f}mm {'below' if p < on.PWV_TRANSPARENT else 'ABOVE':>8s} | "
              f"{abs_counts[slug]:10.1f} {pct_counts[slug]:10.1f} | "
              f"{ja:12.1f}% {jp:5.1f}%", flush=True)

    def rank(d):
        return " > ".join(s[:3] for s in sorted(d, key=lambda s: -d[s]))

    print(f"\nEvery city's own p{args.pct:.0f} lies "
          f"{'BELOW' if all(v < on.PWV_TRANSPARENT for v in pctile.values()) else 'not below'}"
          f" 15 mm ({min(pctile.values()):.1f}-{max(pctile.values()):.1f} mm), so the "
          f"percentile bar is stricter everywhere.")
    print(f"  ranking, absolute bar : {rank(abs_counts)}")
    print(f"  ranking, percentile   : {rank(pct_counts)}")
    top_abs = sorted(abs_counts, key=lambda s: -abs_counts[s])[:2]
    top_pct = sorted(pct_counts, key=lambda s: -pct_counts[s])[:2]
    # The two leaders are not statistically separable, so compare them as a set: the claim
    # under test is which cities lead, not which of the pair is nominally first.
    lowers_all = all(pct_counts[s] <= abs_counts[s] for s in abs_counts)
    print(f"  lowers every count: {lowers_all}")
    print(f"  top two, absolute : {', '.join(top_abs)}")
    print(f"  top two, percentile: {', '.join(top_pct)}")
    print(f"  same two cities on top: {set(top_abs) == set(top_pct)}")


if __name__ == "__main__":
    main()
