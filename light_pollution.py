"""
light_pollution.py — city-centre VIIRS radiance against the corrected good nights.

Reads the 2022 VIIRS mean radiance per city (results/viirs_dnb.csv, from fetch_viirs_dnb.py)
and prints, for the eight cities, the radiance (nW/cm2/sr), its ratio to a natural dark
site, and the Spearman rank correlation between radiance and corrected good nights. With
eight cities only |rho| >= 0.74 is significant at p < 0.05. The paper's light-pollution
figures come from the calibrated atlas instead (falchi.py); this module also supplies the
rank correlation the other light-pollution scripts use.

Usage: python3 light_pollution.py
"""
from __future__ import annotations

import csv
from pathlib import Path

from budgets import good_counts

NATURAL = 0.25   # approx natural-sky VIIRS DNB radiance, nW/cm2/sr (order-of-magnitude anchor)
CSV = Path(__file__).parent / "results" / "viirs_dnb.csv"


def _midranks(values: dict) -> dict:
    """Rank a dict's values ascending, averaging ranks within ties."""
    order = sorted(values, key=lambda k: values[k])
    ranks, i = {}, 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2.0          # midrank for the tied group order[i..j]
        for k in order[i:j + 1]:
            ranks[k] = shared
        i = j + 1
    return ranks


def spearman(a: dict, b: dict) -> float:
    """Spearman rank correlation over the shared keys, tie-safe (Pearson on midranks)."""
    keys = [k for k in a if k in b]
    ra, rb = _midranks({k: a[k] for k in keys}), _midranks({k: b[k] for k in keys})
    n = len(keys)
    ma, mb = sum(ra.values()) / n, sum(rb.values()) / n
    cov = sum((ra[k] - ma) * (rb[k] - mb) for k in keys)
    va = sum((ra[k] - ma) ** 2 for k in keys) ** 0.5
    vb = sum((rb[k] - mb) ** 2 for k in keys) ** 0.5
    return cov / (va * vb)


def main():
    good = good_counts()

    rad = {}
    with CSV.open() as f:
        for r in csv.DictReader(f):
            if r["slug"] in good:
                rad[r["slug"]] = float(r["radiance"])
    missing = set(good) - set(rad)
    if missing:
        raise SystemExit(f"{CSV} has no radiance row for: {', '.join(sorted(missing))}")

    print(f"{'city':10s} {'good':>5s} {'radiance':>9s} {'x natural':>10s}")
    for c in sorted(good, key=lambda x: -good[x]):
        print(f"{c:10s} {good[c]:5d} {rad[c]:9.1f} {rad[c]/NATURAL:9.0f}x")

    rng = (min(rad.values()), max(rad.values()))
    print(f"\nradiance range: {rng[0]:.0f}-{rng[1]:.0f} nW/cm2/sr "
          f"({rng[0]/NATURAL:.0f}-{rng[1]/NATURAL:.0f}x natural; all bright-urban/suburban)")
    print(f"brightest: {max(rad, key=rad.get)}  darkest: {min(rad, key=rad.get)}")
    print(f"Spearman(corrected good nights, radiance) = {spearman(good, rad):+.2f}  "
          f"(n = 8; |rho| >= 0.74 needed for p < 0.05)")


if __name__ == "__main__":
    main()
