"""
ranking_intervals.py — block-bootstrap intervals on the good-night ranking.

Good nights come in spells of several days, so consecutive nights are not independent and a
binomial interval is too narrow. Intervals here are circular block bootstraps over the 2192
nights of 2020-2025 in 10-night blocks, with block lengths of 5 and 15 also run to show the
choice does not drive the result.

Reports the raw count, the corrected count under each gap specification (the gap drawn from
its published interval in every replicate, so the correction's own uncertainty carries
through), and city-to-city differences resampled on the same night blocks at both cities.

Usage: python3 ranking_intervals.py
"""
from __future__ import annotations

import hashlib
import sys

import numpy as np

try:
    from gaps import SPEC_MOVES, SPECS, gap, gap_ci
    from nights import OBSERVABLE_H, apply_correction, calendar, corrected_frac, per_night
    from sites import SITES
except ImportError as e:
    sys.exit(f"Run from the repo root: {e}")

B = 2000          # bootstrap replicates
BLOCK = 10        # nights per block (primary); BLOCK_SWEEP checks the choice
BLOCK_SWEEP = (5, 10, 15)
SEED = 42
HOURS = OBSERVABLE_H   # nights.py; 4 h unless BUDGET_HOURS is set

PAIRS = [("calgary", "winnipeg"), ("winnipeg", "edmonton"),
         ("calgary", "edmonton"), ("edmonton", "montreal")]

# The ranking is reported under each of the three gap estimates; the
# specifications and the provenance of each value live in gaps.
PRIMARY = "satellite"


def block_index(rng, n: int, block: int) -> np.ndarray:
    """(B, n) index matrix: B circular block-bootstrap resamples of n nights."""
    n_blocks = -(-n // block)                      # ceiling division
    starts = rng.integers(0, n, size=(B, n_blocks))
    offsets = np.arange(block)
    return (starts[:, :, None] + offsets[None, None, :]).reshape(B, -1)[:, :n] % n


def ci(reps: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))


def main():
    rng = np.random.default_rng(SEED)
    # Draw the night blocks first, so the resampling stream does not depend on how many
    # cities are loaded or in what order their gaps are drawn.
    ix = block_index(rng, len(calendar()), BLOCK)

    good, recs, n_years = {}, {}, None
    for slug, site in SITES.items():
        rec = recs[slug] = per_night(site)
        n_years = rec.dates[-1].year - rec.dates[0].year + 1
        keep = rec.valid & rec.quality
        good[slug] = ((rec.frac * rec.dark_h) >= HOURS) & keep
        print(f"  loaded {slug}", file=sys.stderr, flush=True)

    def corrected(slug: str, spec: str):
        """(point-estimate indicator, (B, n) replicate indicator grid) under one spec.

        The gap draws come from an RNG seeded per (spec, city) rather than from the shared
        stream, so each city's corrected interval is identical no matter how many
        specifications the run loops over or in what order. Adding a specification cannot
        move the numbers already published for the others.
        """
        rec = recs[slug]
        keep = rec.valid & rec.quality
        point = (corrected_frac(rec, gap(spec, slug)) * rec.dark_h >= HOURS) & keep
        lo, hi = gap_ci(spec, slug)
        grng = np.random.default_rng(
            int(hashlib.sha256(f"{SEED}|{spec}|{slug}".encode()).hexdigest()[:8], 16))
        gaps = grng.normal(gap(spec, slug), (hi - lo) / (2 * 1.96), B) / 100.0
        cd = apply_correction(rec.frac[None, :], 100.0 * gaps[:, None]) * rec.dark_h[None, :]
        return point, (cd >= HOURS) & keep[None, :]

    order = sorted(good, key=lambda s: -good[s].sum())
    raw_reps = {s: good[s][ix].sum(1) / n_years for s in order}

    spec_point, spec_reps = {}, {}
    for spec in SPECS:
        spec_point[spec], spec_reps[spec] = {}, {}
        for slug in order:
            pt, grid = corrected(slug, spec)
            spec_point[spec][slug] = pt.sum() / n_years
            spec_reps[spec][slug] = np.take_along_axis(grid, ix, axis=1).sum(1) / n_years

    print(f"\nCircular block bootstrap, {BLOCK}-night blocks, {B} replicates, seed {SEED}\n")
    print(f"{'city':10s} {'raw':>4s} {'raw 95% CI':>13s}   "
          f"{'corr':>4s} {'corrected 95% CI':>16s}    (gap spec: {PRIMARY})")
    for slug in order:
        raw = good[slug].sum() / n_years
        rlo, rhi = ci(raw_reps[slug])
        clo, chi = ci(spec_reps[PRIMARY][slug])
        print(f"{slug:10s} {raw:4.0f} [{rlo:5.0f},{rhi:5.0f}]   "
              f"{spec_point[PRIMARY][slug]:4.0f} [{clo:5.0f},{chi:5.0f}]")

    print("\nCorrected counts and intervals under every gap specification:")
    print(f"{'city':10s} " + " ".join(f"{s:>20s}" for s in SPECS))
    for slug in order:
        cells = []
        for spec in SPECS:
            lo, hi = ci(spec_reps[spec][slug])
            mark = "*" if slug in SPEC_MOVES[spec] else " "
            cells.append(f"{spec_point[spec][slug]:3.0f} [{lo:3.0f},{hi:3.0f}]{mark}")
        print(f"{slug:10s} " + " ".join(f"{c:>20s}" for c in cells))
    print("  (* = this city's gap differs from the satellite specification)")
    for spec in SPECS:
        o = sorted(order, key=lambda s: -spec_point[spec][s])
        # Join equal counts with "=" rather than letting dict order invent a winner.
        parts = [o[0][:3]]
        for prev, cur in zip(o, o[1:]):
            sep = " = " if abs(spec_point[spec][cur] - spec_point[spec][prev]) < 1e-9 else " > "
            parts.append(sep + cur[:3])
        print(f"  ranking, {spec:10s}: " + "".join(parts))

    print("\nPairwise differences, same night blocks resampled at both cities:")
    for a, b in PAIRS:
        dr = raw_reps[a] - raw_reps[b]
        rlo, rhi = ci(dr)
        rv = "resolved" if rlo * rhi > 0 else "unresolved"
        print(f"  {a:9s} - {b:9s}: raw  {dr.mean():+5.1f} [{rlo:+5.1f},{rhi:+5.1f}] {rv}")
        for spec in SPECS:
            dc = spec_reps[spec][a] - spec_reps[spec][b]
            clo, chi = ci(dc)
            cv = "resolved" if clo * chi > 0 else "unresolved"
            print(f"  {'':9s}   {'':9s}  {spec:10s} {dc.mean():+5.1f} "
                  f"[{clo:+5.1f},{chi:+5.1f}] {cv}")

    print("\nBlock-length sensitivity on the raw interval (nights/yr):")
    print(f"{'city':10s} " + " ".join(f"{'blk ' + str(k):>14s}" for k in BLOCK_SWEEP))
    for slug in order:
        cells = []
        for k in BLOCK_SWEEP:
            jx = block_index(np.random.default_rng(SEED), len(good[slug]), k)
            lo, hi = ci(good[slug][jx].sum(1) / n_years)
            cells.append(f"[{lo:5.0f},{hi:5.0f}]")
        print(f"{slug:10s} " + " ".join(f"{c:>14s}" for c in cells))


if __name__ == "__main__":
    main()
