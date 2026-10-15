"""
corrected_budget.py — observable and good-night counts after the bias correction.

Recomputes each city's counts with every night's usable fraction corrected and floored at
zero, under each gap specification in gaps.py, and compares the measured correction shape
with the uniform one. The raw count is the uncorrected upper bound.

Good nights are corrected night by night: a corrected good night must still clear the
observable bar on its reduced fraction. Rescaling the raw good-night count by the
observable-night ratio would assume the correction removes good and other nights in the
same proportion, which it does not.

One gap is applied all year rather than season by season, because on the snow-free ground
the satellite comparison needs, the winter samples are too small at five of the eight
cities. winter_dod.py and winter_scaled.py check that choice against the ceilometers,
which snow does not affect.

Usage: python3 corrected_budget.py
"""
from __future__ import annotations

import sys

try:
    from budgets import read_budget
    from gaps import (MOONLESS_INCREMENT, SPEC_MOVES, SPECS, gap,
                                  moonless_gap)
    from nights import counts, floor_rate, per_night
    from sites import SITES
except ImportError as e:
    sys.exit(f"Run from the repo root: {e}")

# The nearer-satellite gap, which every count in the paper uses; gaps.py carries it and the
# two alternatives, with the source of every value.
GAP_PP = {slug: gap("satellite", slug) for slug in SPECS["satellite"]}
MOONLESS_GAP = {slug: moonless_gap("satellite", slug) for slug in SPECS["satellite"]}

# Vancouver is the one city whose daytime human-GOES offset sits outside the band the
# difference-of-differences cancellation assumes, so its correction is stress-tested by
# doubling it.
DOUBLED = "vancouver"


def main():
    print(f"{'site':10s} {'raw':>5s} {'corrected':>9s}   "
          f"{'good raw':>9s} {'good corr':>10s}")
    rows = []
    recs = {}
    for slug, site in SITES.items():
        rec = recs[slug] = per_night(site)
        raw_obs, raw_good = counts(rec)
        cor_obs, cor_good = counts(rec, GAP_PP[slug])
        raw_ml, _ = counts(rec, moonless=True)
        cor_ml, _ = counts(rec, MOONLESS_GAP[slug], moonless=True)

        # A partly-fetched ERA5 cache deflates good nights without erroring, because a
        # night with no PWV just fails the transparency test. Catch that here.
        published = read_budget(slug)
        if abs(raw_good - published.good) > 1 or abs(raw_obs - published.obs) > 1:
            print(f"  WARNING {slug}: recomputed raw obs/good {raw_obs:.0f}/{raw_good:.0f} "
                  f"but data/derived says {published.obs}/{published.good} -- regenerate "
                  f"data/derived/{slug}/observable.txt", file=sys.stderr)

        rows.append((slug, raw_obs, cor_obs, raw_ml, cor_ml, raw_good, cor_good))
        print(f"{slug:10s} {raw_obs:5.0f} {cor_obs:9.0f}   "
              f"{raw_good:9.0f} {cor_good:10.0f}", flush=True)

    print("\nMoonless-dark obs>=bar (raw -> dark-stratum corrected):")
    for r in rows:
        print(f"  {r[0]:10s} {r[3]:5.0f} -> {r[4]:5.0f}  ({r[4] - r[3]:+.0f})")

    print("\nRanking by corrected good nights:")
    for r in sorted(rows, key=lambda x: -x[6]):
        slug, raw_obs, cor_obs, raw_ml, cor_ml, raw_good, cor_good = r
        print(f"  {slug:10s} good {cor_good:.0f} (raw {raw_good:.0f}); "
              f"obs>=bar {cor_obs:.0f} (raw {raw_obs:.0f}); "
              f"moonless {cor_ml:.0f} (raw {raw_ml:.0f})")

    # ---- corrected good nights under all three gap specifications -------------------
    print("\nCorrected good nights under each gap specification:")
    print(f"{'city':10s} {'raw':>5s} " + " ".join(f"{s:>11s}" for s in SPECS))
    spec_counts = {}
    for slug, site in SITES.items():
        rec = recs[slug]
        _, raw_good = counts(rec)
        line = f"{slug:10s} {raw_good:5.0f} "
        for spec in SPECS:
            _, g = counts(rec, gap(spec, slug))
            spec_counts.setdefault(spec, {})[slug] = g
            moved = "*" if slug in SPEC_MOVES[spec] else " "
            line += f" {g:9.0f}{moved} "
        print(line)
    print("  (* = this city's gap differs from the satellite specification)")
    for spec, d in spec_counts.items():
        order = sorted(d, key=lambda s: -d[s])
        print(f"  ranking, {spec:10s}: " + " > ".join(s[:3] for s in order))

    # ---- how often the zero floor engages --------------------------------------------
    print("\nZero-floor engagement (share of nights with any clear dark sky):")
    fr = {slug: floor_rate(recs[slug], GAP_PP[slug]) for slug in SITES}
    for slug in sorted(fr, key=lambda s: fr[s]):
        print(f"  {slug:10s} {100 * fr[slug]:5.1f}%")
    lo, hi = min(fr, key=lambda s: fr[s]), max(fr, key=lambda s: fr[s])
    print(f"  range: {100 * fr[lo]:.0f}% ({lo}) to {100 * fr[hi]:.0f}% ({hi})")

    # ---- doubling the one correction whose daytime anchor is out of band -------------
    rec = recs[DOUBLED]
    _, g1 = counts(rec, GAP_PP[DOUBLED])
    _, g2 = counts(rec, 2 * GAP_PP[DOUBLED])
    o1, _ = counts(rec, GAP_PP[DOUBLED])
    o2, _ = counts(rec, 2 * GAP_PP[DOUBLED])
    print(f"\nDoubling the {DOUBLED} gap ({GAP_PP[DOUBLED]:.1f} -> "
          f"{2 * GAP_PP[DOUBLED]:.1f} pp):")
    print(f"  good nights      {g1:.0f} -> {g2:.0f}")
    print(f"  obs>=bar nights   {o1:.0f} -> {o2:.0f}")

    # ---- the shape of the correction ---------------------------------------------------
    # The paper applies the measured (ceilometer) shape. The uniform shape, which also takes
    # from nights reported fully clear, is the sensitivity: under it a fully clear night
    # needs bar / (1 - gap) hours of darkness to count.
    from nights import OBSERVABLE_H
    print("\nCorrection SHAPE: good nights/yr under the measured shape and the uniform one")
    print(f"{'city':10s} {'measured':>9s} {'uniform':>8s} {'delta':>6s}   "
          f"{'uniform needs D >=':>19s}")
    shp = {}
    for slug in SITES:
        _, gm = counts(recs[slug], GAP_PP[slug], shape="ceilometer")
        _, gu = counts(recs[slug], GAP_PP[slug], shape="uniform")
        shp[slug] = (gm, gu)
        need = OBSERVABLE_H / (1 - GAP_PP[slug] / 100.0)
        print(f"{slug:10s} {gm:9.0f} {gu:8.0f} {gu - gm:+6.0f}   {need:17.2f} h")
    worst = max(shp, key=lambda s: abs(shp[s][1] - shp[s][0]))
    print(f"  largest move: {worst} {abs(shp[worst][1] - shp[worst][0]):.0f} nights")
    for label, i in (("measured", 0), ("uniform", 1)):
        order = sorted(shp, key=lambda s: -shp[s][i])
        print(f"  ranking, {label:8s}: " + " > ".join(s[:3] for s in order))


if __name__ == "__main__":
    main()
