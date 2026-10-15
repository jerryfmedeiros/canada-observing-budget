"""
correction_scale.py — how much the correction removes, and the scale it is applied on.

1. Mean amount removed per night. The gap (human minus satellite night-day change) sets
   the scale of the correction profile, not the amount removed: nights reported fully
   cloudy have nothing to remove, and the profile puts the artifact on partly cloudy
   nights. This prints, per city, the gap and the mean removed per night (nights with
   astronomical darkness and their own reports) under the measured shape, the uniform
   shape, and the day-equivalent mapping.

2. The scale question. The profile is a difference in the CEILOMETER's usable fraction
   between night and day for the same human report, and it is subtracted from the
   OBSERVER's fraction. Two ways to look at it on the observer's scale:
     a. day-equivalent mapping (nights.day_equivalent_removal): send each night through
        the ceilometer curve to the human daytime fraction with the same mean ceilometer
        value. Inverting E[ceilometer | human] exaggerates differences where the curve is
        flat (regression dilution), so it leans toward too large a correction.
     b. the reverse regression: group units by the CEILOMETER fraction and compare the
        observer's mean fraction at night and by day, i.e. the observer's excess at the
        same ceilometer sky, on the observer's scale. It is large where the ceilometer
        reads clear, because that is where the high cloud the ceilometer cannot see (and
        the night observer misses) sits; the two conditionings differ for that reason, and
        neither is exact. The counts are given under (a) so the effect can be judged.

Usage: python3 correction_scale.py
"""
from __future__ import annotations

import sys

import numpy as np

try:
    from correction_shape import BINS, PAIRS, units
    from gaps import gap
    from nights import OBSERVABLE_H, apply_correction, per_night
    from sites import SITES
except ImportError as e:
    sys.exit(f"Run from the repo root: {e}")

SHAPES = ("ceilometer", "uniform", "day_equivalent")


def main():
    print("1. Mean removed per night (pp) and counts per year under each correction shape\n")
    print(f"{'city':10s} {'gap':>5s} | {'removed: measured':>17s} {'uniform':>8s} {'day-eq':>7s} | "
          f"{'obs: meas':>9s} {'unif':>5s} {'d-eq':>5s} | {'good: meas':>10s} {'unif':>5s} {'d-eq':>5s}")
    good = {sh: {} for sh in SHAPES}
    for slug, site in SITES.items():
        r = per_night(site)
        g = gap("satellite", slug)
        own = r.valid & ~r.imputed & (r.dark_h > 0)
        ny = r.dates[-1].year - r.dates[0].year + 1
        rem, obs = {}, {}
        for sh in SHAPES:
            cf = apply_correction(r.frac, g, sh)
            rem[sh] = 100 * float((r.frac[own] - cf[own]).mean())
            o = (cf * r.dark_h >= OBSERVABLE_H) & r.valid
            obs[sh] = o.sum() / ny
            good[sh][slug] = (o & r.quality).sum() / ny
        print(f"{slug:10s} {g:5.1f} | {rem['ceilometer']:17.1f} {rem['uniform']:8.1f} "
              f"{rem['day_equivalent']:7.1f} | {obs['ceilometer']:9.1f} {obs['uniform']:5.1f} "
              f"{obs['day_equivalent']:5.1f} | {good['ceilometer'][slug]:10.1f} "
              f"{good['uniform'][slug]:5.1f} {good['day_equivalent'][slug]:5.1f}", flush=True)
    print("\nGood-night order:")
    for sh in SHAPES:
        order = sorted(good[sh], key=lambda s: -good[sh][s])
        print(f"  {sh:15s}: " + " > ".join(f"{s[:3]} {good[sh][s]:.0f}" for s in order))

    print("\n2b. Observer's excess at the same ceilometer sky (units grouped by the CEILOMETER "
          "fraction), pooled over the four pairs\n")
    pooled = {lab: dict(hn=[], hd=[]) for _, _, lab in BINS}
    for slug, human, auto in PAIRS:
        site = SITES[slug]
        hn, hd = units(human, site)
        an, ad = units(auto, site)
        for lo, hi, lab in BINS:
            pooled[lab]["hn"] += [hn[k] for k in an if k in hn and lo <= an[k] < hi]
            pooled[lab]["hd"] += [hd[k] for k in ad if k in hd and lo <= ad[k] < hi]
    print(f"  {'ceilometer fraction':20s} {'n night':>8s} {'human night':>12s} {'n day':>6s} "
          f"{'human day':>10s} {'excess pp':>10s}")
    for _, _, lab in BINS:
        p = pooled[lab]
        print(f"  {lab:20s} {len(p['hn']):8d} {100 * np.mean(p['hn']):11.1f}% {len(p['hd']):6d} "
              f"{100 * np.mean(p['hd']):9.1f}% {100 * (np.mean(p['hn']) - np.mean(p['hd'])):+10.1f}")


if __name__ == "__main__":
    main()
