"""
falchi.py — calibrated sky brightness and escape distance from the Falchi et al. (2016) atlas.

The New World Atlas of Artificial Night Sky Brightness (Falchi et al. 2016, GFZ Data Services,
doi:10.5880/GFZ.1.4.2016.001) gives modelled artificial zenith brightness in mcd/m^2 on a
30-arcsec grid, propagated from 2014 VIIRS radiances with a radiative-transfer model. That makes
it a calibrated counterpart to skyglow.py, which uses the 2022 VIIRS composite but only a
relative Walker-law index.

For each study city this reports:
  - the artificial zenith brightness downtown (3x3-pixel mean), its ratio to the natural sky
    (0.171 mcd/m^2, Falchi et al.'s reference, ~22.0 mag/arcsec^2), and the total sky
    brightness in V mag/arcsec^2;
  - escape distances: the shortest straight-line distance from downtown to LAND (Natural Earth
    land minus lakes) where the artificial brightness falls below each of Falchi et al.'s
    class boundaries of 688, 87 and 14 ucd/m^2 (artificial = 4.0, 0.5 and 0.08 x natural).

Put World_Atlas_2015.tif in data/falchi_cache/ (download World_Atlas_2015.zip from the DOI above).

Usage: python3 falchi.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

try:
    import geopandas as gpd
    import rasterio
    from rasterio import features
    from rasterio.windows import Window
except ImportError as e:
    sys.exit(f"needs geopandas and rasterio: {e}")

from fetch_viirs_dnb import CITY_CENTRES
from light_pollution import spearman


def spearman_p(a: dict, b: dict) -> float:
    """Exact two-sided permutation p for Spearman's rho: the share of all n! pairings of b's
    values with a's whose |rho| is at least the observed one (40,320 pairings for n = 8)."""
    from itertools import permutations
    keys = [k for k in a if k in b]
    obs = abs(spearman(a, b)) - 1e-12
    vals = [b[k] for k in keys]
    hits = total = 0
    for perm in permutations(vals):
        total += 1
        hits += abs(spearman(a, dict(zip(keys, perm)))) >= obs
    return hits / total

ROOT = Path(__file__).parent
TIF = ROOT / "data" / "falchi_cache" / "World_Atlas_2015.tif"
NE = ROOT / "data" / "ne_cache"
NATURAL = 0.171168          # mcd/m^2, Falchi et al. (2016) natural zenith reference
THRESH_UCD = (688, 87, 14)  # Falchi et al. class boundaries, micro-cd/m^2
REGION_KM = 450.0
KM_PER_DEG = 111.195
STUDY = ["calgary", "winnipeg", "edmonton", "montreal", "vancouver", "ottawa", "toronto", "halifax"]
PRESERVES = {
    "Beaver Hills (AB)": (53.60, -112.85),
    "Cypress Hills (AB/SK)": (49.65, -110.00),
    "Mont-Megantic (QC)": (45.455, -71.152),
    "Torrance Barrens (ON)": (44.94, -79.52),
    "Kejimkujik (NS)": (44.38, -65.22),
}


def mag(artificial_mcd: float) -> float:
    """Total zenith sky brightness in V mag/arcsec^2 for a given artificial brightness."""
    L = (artificial_mcd + NATURAL) * 1e-3          # cd/m^2
    return -2.5 * np.log10(L / 1.08e5)


def main():
    if not TIF.exists():
        sys.exit(f"missing {TIF}; see the docstring")
    land = gpd.read_file(f"zip://{NE / 'ne_10m_land.zip'}")
    lakes = gpd.read_file(f"zip://{NE / 'ne_10m_lakes.zip'}")

    rows = {}
    with rasterio.open(TIF) as ds:
        def point(lat, lon, half=1):
            r, c = ds.index(lon, lat)
            b = ds.read(1, window=Window(c - half, r - half, 2 * half + 1, 2 * half + 1))
            return float(np.mean(b))

        for slug in STUDY:
            lat, lon = CITY_CENTRES[slug]
            dlat = REGION_KM / KM_PER_DEG
            dlon = dlat / np.cos(np.radians(lat))
            r0, c0 = ds.index(lon - dlon, lat + dlat)
            r1, c1 = ds.index(lon + dlon, lat - dlat)
            win = Window(c0, r0, c1 - c0 + 1, r1 - r0 + 1)
            a = ds.read(1, window=win).astype(np.float64)
            tr = ds.window_transform(win)
            m_land = features.rasterize(((g, 1) for g in land.geometry), out_shape=a.shape,
                                        transform=tr)
            m_lake = features.rasterize(((g, 1) for g in lakes.geometry), out_shape=a.shape,
                                        transform=tr)
            L = (m_land == 1) & (m_lake == 0)
            lons = tr.c + (np.arange(a.shape[1]) + 0.5) * tr.a
            lats = tr.f + (np.arange(a.shape[0]) + 0.5) * tr.e
            dy = (lats[:, None] - lat) * KM_PER_DEG
            dx = (lons[None, :] - lon) * KM_PER_DEG * np.cos(np.radians(lat))
            dist = np.hypot(dy, dx)
            esc = []
            for t in THRESH_UCD:
                ok = L & (a < t / 1000.0) & (dist <= REGION_KM)
                esc.append(float(dist[ok].min()) if ok.any() else float("nan"))
            core = point(lat, lon)
            rows[slug] = (core, esc)

        pres = {n: point(la, lo, 0) for n, (la, lo) in PRESERVES.items()}

    print("Falchi et al. (2016) atlas: artificial zenith brightness (2014 VIIRS basis)")
    print(f"{'city':10s} {'art. mcd/m2':>11s} {'x natural':>9s} {'V mag/as2':>9s}   "
          f"escape km to artificial < " + " / ".join(f"{t}" for t in THRESH_UCD) + " ucd/m2")
    for slug in sorted(STUDY, key=lambda s: -rows[s][0]):
        core, esc = rows[slug]
        print(f"{slug:10s} {core:11.2f} {core / NATURAL:9.1f} {mag(core):9.2f}   "
              + "  ".join(f"{e:5.0f}" for e in esc))

    print(f"\nclass boundaries as total sky brightness: "
          + ", ".join(f"{t} ucd/m2 -> {mag(t / 1000):.2f}" for t in THRESH_UCD)
          + f"; natural {mag(0):.2f} mag/arcsec^2")
    print("\nReference: RASC Dark-Sky Preserves")
    for n, v in pres.items():
        print(f"  {n:24s} {v * 1000:8.1f} ucd/m2   {mag(v):.2f} mag/arcsec^2")

    # Rank correlations against the corrected good and low-jet nights (headline_corrected.py).
    core = {s: rows[s][0] for s in STUDY}
    hc = ROOT / "results" / "headline_corrected.csv"
    with hc.open() as f:
        recs = list(csv.DictReader(f))
    cg = {r["site"]: float(r["good"]) for r in recs}
    cd = {r["site"]: float(r["lowjet"]) for r in recs}
    print("\nSpearman rank correlations with the corrected counts (n = 8; |rho| >= 0.74 needed "
          "for p < 0.05 two-sided)")
    for i, t in enumerate(THRESH_UCD):
        e = {s: rows[s][1][i] for s in STUDY}
        print(f"  escape<{t:>3d}: good {spearman(cg, e):+.2f} (p = {spearman_p(cg, e):.2f})"
              f"  low-jet {spearman(cd, e):+.2f}")
    print(f"  downtown  : good {spearman(cg, core):+.2f}  low-jet {spearman(cd, core):+.2f}")

    # Cross-check against the 2022 Walker-law result (skyglow.py), if present.
    sg = ROOT / "results" / "skyglow.txt"
    if sg.exists():
        walker = {}
        for line in sg.read_text().splitlines():
            p = line.split()
            if p and p[0] in STUDY and len(p) >= 7:
                walker[p[0]] = float(p[5])          # escape at median/300
        if len(walker) == 8:
            e87 = {s: rows[s][1][1] for s in STUDY}
            print(f"\nescape<87 vs Walker 2022 escape (1/300): Spearman {spearman(e87, walker):+.2f}")


if __name__ == "__main__":
    main()
