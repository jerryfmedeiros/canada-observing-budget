"""
pairwise_all.py — which city-to-city differences in good nights are resolved.

ranking_intervals.py reports a handful of pairs. With eight cities there are 28, and at the
95% level about one in twenty unresolved differences would look resolved by chance. This
computes every pair on the same resampled night blocks (circular block bootstrap, 10-night
blocks, the gap redrawn from its interval in each replicate) and reports each one at the
plain 95% level and with a Bonferroni adjustment for 28 comparisons. The adjusted tail is
0.09% a side, so the bootstrap uses 20000 replicates rather than 2000.

The gaps that set the correction's size are the measurements of Fernandes Medeiros (2026) (gaps.py).
Two options test how much the ranking leans on them. --scale K multiplies every city's gap
(and its interval) by K, so --scale 0 is the uncorrected record. --equal-gap gives every city
the mean of the eight satellite gaps, keeping the interval width of its own gap, so only
the size of the correction comes from outside and the city-to-city differences in it do not.

Usage:
  python3 pairwise_all.py                 # nearer-satellite gaps
  python3 pairwise_all.py --scale 0       # no correction
  python3 pairwise_all.py --equal-gap     # one gap for every city
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import sys

import numpy as np

try:
    from gaps import SATELLITE, gap, gap_ci
    from nights import OBSERVABLE_H, apply_correction, calendar, corrected_frac, per_night
    from sites import SITES
except ImportError as e:
    sys.exit(f"Run from the repo root: {e}")

SPEC = "satellite"
B, BLOCK, SEED, CHUNK = 20000, 10, 42, 2000


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiply every gap by this factor (0 = uncorrected)")
    ap.add_argument("--equal-gap", action="store_true",
                    help="give every city the mean of the eight satellite gaps")
    args = ap.parse_args()
    k = args.scale
    mean_gap = float(np.mean([v[0] for v in SATELLITE.values()]))

    rng = np.random.default_rng(SEED)
    n = len(calendar())
    n_blocks = -(-n // BLOCK)
    starts = rng.integers(0, n, size=(B, n_blocks))
    ix = (starts[:, :, None] + np.arange(BLOCK)[None, None, :]).reshape(B, -1)[:, :n] % n

    reps, point = {}, {}
    for slug, site in SITES.items():
        rec = per_night(site)
        ny = rec.dates[-1].year - rec.dates[0].year + 1
        lo, hi = gap_ci(SPEC, slug)
        grng = np.random.default_rng(
            int(hashlib.sha256(f"{SEED}|{SPEC}|{slug}".encode()).hexdigest()[:8], 16))
        g0 = mean_gap if args.equal_gap else gap(SPEC, slug)
        draws = k * grng.normal(g0, (hi - lo) / (2 * 1.96), B)
        keep = rec.valid & rec.quality
        out = np.zeros(B)
        for i0 in range(0, B, CHUNK):          # chunks keep the (B, nights) grid in memory
            sl = slice(i0, i0 + CHUNK)
            cd = apply_correction(rec.frac[None, :], draws[sl, None]) * rec.dark_h[None, :]
            out[sl] = np.take_along_axis((cd >= OBSERVABLE_H) & keep[None, :], ix[sl], axis=1).sum(1) / ny
        reps[slug] = out
        point[slug] = ((corrected_frac(rec, k * g0) * rec.dark_h >= OBSERVABLE_H) & keep).sum() / ny
        print(f"  loaded {slug}", file=sys.stderr, flush=True)

    order = sorted(point, key=lambda s: -point[s])
    pairs = list(itertools.combinations(order, 2))
    alpha = 0.05 / len(pairs)
    label = f"{SPEC} gap"
    if args.equal_gap:
        label = f"equal gap ({mean_gap:.2f} pp, the satellite mean) at every city"
    if k != 1.0:
        label = "no correction" if k == 0 else f"{label} x {k:g}"
    print(f"Good nights/yr, {label}, {B} replicates; {len(pairs)} pairs, "
          f"Bonferroni tail {100 * alpha / 2:.3f}% a side\n")
    print("  " + "  ".join(f"{s[:3]} {point[s]:.0f}" for s in order) + "\n")
    print(f"{'pair':22s} {'diff':>6s} {'95% interval':>16s} {'adjusted interval':>18s}  resolved")
    r95, radj = {}, {}
    for a, b in pairs:
        d = reps[a] - reps[b]
        l95, h95 = np.percentile(d, [2.5, 97.5])
        la, ha = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        r95[(a, b)], radj[(a, b)] = l95 * h95 > 0, la * ha > 0
        tag = "adjusted" if radj[(a, b)] else ("95% only" if r95[(a, b)] else "no")
        print(f"{a + ' - ' + b:22s} {d.mean():+6.1f} [{l95:+6.1f},{h95:+6.1f}] "
              f"[{la:+6.1f},{ha:+6.1f}]  {tag}")

    print("\n'**' resolved after the adjustment, '*' at 95% only, '.' not resolved")
    print("       " + " ".join(f"{s[:3]:>4s}" for s in order))
    for a in order:
        cells = []
        for b in order:
            if a == b:
                cells.append("   -")
            else:
                k = (a, b) if (a, b) in r95 else (b, a)
                cells.append("  **" if radj[k] else ("   *" if r95[k] else "   ."))
        print(f"  {a[:3]:4s} " + " ".join(cells))
    print(f"\nresolved at 95%: {sum(r95.values())} of {len(pairs)}; "
          f"after the adjustment: {sum(radj.values())} of {len(pairs)}")
    print("adjusted: " + "; ".join(f"{a[:3]} > {b[:3]}" for (a, b), v in radj.items() if v))


if __name__ == "__main__":
    main()
