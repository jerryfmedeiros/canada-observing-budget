"""
skyglow.py — a Walker-law skyglow index and escape distance from the 2022 VIIRS lights.

The VIIRS radiance of fetch_viirs_dnb.py is the mean over an 11x11 pixel box (about 5 by
3 km) on each downtown; it measures the core's own lights, not the sky an observer sees,
which integrates every source in the region weighted by distance. This builds two
quantities from the same 2022 composite. The paper uses them to check the escape distances
of falchi.py, whose atlas is built on 2014 lights.

  1. A Walker-law skyglow index at each downtown: the sum over 1-arcmin cells of
     radiance x cell area x d^-2.5 (d in km, sources within 250 km). Walker's law is the
     empirical fall-off of artificial sky brightness with distance from a city; the index is
     relative and is reported normalised to the eight-city median.

  2. An escape distance: the shortest straight-line distance from downtown to any land cell
     (Natural Earth land minus lakes) whose index falls below a threshold. The thresholds are
     fractions of the eight-city median downtown index (1/30, 1/100, 1/300, 1/1000), so they
     are common to all cities; the index at the RASC Dark-Sky Preserves is printed to show
     what each threshold means on the ground.

The index is relative, not a calibrated sky brightness: it ignores differences in
extinction, terrain, and the insensitivity of VIIRS to the blue light of LEDs.

Usage: python3 skyglow.py
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
    from rasterio.transform import from_origin
    from rasterio.windows import from_bounds
    from scipy.signal import fftconvolve
except ImportError as e:
    sys.exit(f"needs geopandas, rasterio, scipy: {e}")

from fetch_viirs_dnb import CACHE_DIR, CITY_CENTRES, TIF_NAME
from light_pollution import spearman

ROOT = Path(__file__).parent
SUBSET = CACHE_DIR / "canada_south_1arcmin.npz"
NE = ROOT / "data" / "ne_cache"

# Southern-Canada bounding box (plus the adjacent US), degrees.
W, S, E, N = -130.0, 38.0, -56.0, 58.0
AGG = 4                      # 4x4 15-arcsec pixels -> 1 arcmin cells
CELL_DEG = 1.0 / 60.0
KM_PER_DEG = 111.195
EXPONENT = 2.5               # Walker's law
KERNEL_KM = 250.0
REGION_KM = 450.0            # half-width of the grid searched for escape distance
FRACTIONS = (30, 100, 300, 1000)   # thresholds = median downtown index / these
STUDY = ["calgary", "winnipeg", "edmonton", "montreal", "vancouver", "ottawa", "toronto", "halifax"]

# RASC Dark-Sky Preserves used as on-the-ground references for the thresholds.
PRESERVES = {
    "Beaver Hills (AB)": (53.60, -112.85),
    "Cypress Hills (AB/SK)": (49.65, -110.00),
    "Mont-Megantic (QC)": (45.455, -71.152),
    "Torrance Barrens (ON)": (44.94, -79.52),
    "Kejimkujik (NS)": (44.38, -65.22),
}


def load_subset() -> tuple[np.ndarray, float, float]:
    """1-arcmin radiance grid over the bounding box (row 0 = north), cached after first read."""
    if SUBSET.exists():
        z = np.load(SUBSET)
        return z["rad"], float(z["west"]), float(z["north"])
    src = CACHE_DIR / TIF_NAME
    if not src.exists():
        sys.exit(f"missing {src}; run fetch_viirs_dnb.py first")
    print("reading VIIRS subset (one-time, slow: the composite is gzipped)...", file=sys.stderr)
    with rasterio.open(f"/vsigzip/{src}") as ds:
        win = from_bounds(W, S, E, N, ds.transform).round_offsets().round_lengths()
        raw = ds.read(1, window=win).astype(np.float32)
        west, north = ds.window_transform(win) * (0, 0)
    raw = np.nan_to_num(np.maximum(raw, 0.0))
    r, c = (raw.shape[0] // AGG) * AGG, (raw.shape[1] // AGG) * AGG
    rad = raw[:r, :c].reshape(r // AGG, AGG, c // AGG, AGG).mean(axis=(1, 3))
    np.savez_compressed(SUBSET, rad=rad, west=west, north=north)
    return rad, west, north


def land_mask(shape, west, north) -> np.ndarray:
    land = gpd.read_file(f"zip://{NE / 'ne_10m_land.zip'}")
    lakes = gpd.read_file(f"zip://{NE / 'ne_10m_lakes.zip'}")
    tr = from_origin(west, north, CELL_DEG, CELL_DEG)
    m_land = features.rasterize(((g, 1) for g in land.geometry), out_shape=shape, transform=tr)
    m_lake = features.rasterize(((g, 1) for g in lakes.geometry), out_shape=shape, transform=tr)
    return (m_land == 1) & (m_lake == 0)


def rc(lat, lon, west, north):
    return int((north - lat) / CELL_DEG), int((lon - west) / CELL_DEG)


def index_field(rad, west, north, lat0, lon0):
    """Walker index on a (2*REGION_KM)-wide grid around (lat0, lon0), plus that grid's geometry."""
    ky = KM_PER_DEG * CELL_DEG                       # km per cell, north-south
    kx = ky * np.cos(np.radians(lat0))               # km per cell, east-west (local)
    hr_k, hc_k = int(KERNEL_KM / ky), int(KERNEL_KM / kx)
    hr_g, hc_g = int(REGION_KM / ky), int(REGION_KM / kx)
    r0, c0 = rc(lat0, lon0, west, north)
    # Source window = search grid padded by the kernel radius.
    R0, R1 = r0 - hr_g - hr_k, r0 + hr_g + hr_k + 1
    C0, C1 = c0 - hc_g - hc_k, c0 + hc_g + hc_k + 1
    src = np.zeros((R1 - R0, C1 - C0), np.float64)
    rr0, cc0 = max(R0, 0), max(C0, 0)
    rr1, cc1 = min(R1, rad.shape[0]), min(C1, rad.shape[1])
    src[rr0 - R0:rr1 - R0, cc0 - C0:cc1 - C0] = rad[rr0:rr1, cc0:cc1]
    lats = north - (np.arange(R0, R1) + 0.5) * CELL_DEG
    src *= (ky * ky * np.cos(np.radians(lats)) / np.cos(np.radians(lat0)) * (kx / ky))[:, None]
    dy = np.arange(-hr_k, hr_k + 1)[:, None] * ky
    dx = np.arange(-hc_k, hc_k + 1)[None, :] * kx
    d = np.hypot(dy, dx)
    d[hr_k, hc_k] = 0.5 * np.sqrt(kx * ky)          # own cell: mean distance within it
    kern = np.where(d <= KERNEL_KM, d ** -EXPONENT, 0.0)
    field = fftconvolve(src, kern, mode="valid")    # shape = search grid
    return field, (r0 - hr_g, c0 - hc_g), (ky, kx)


def point_index(rad, west, north, lat, lon):
    f, (g_r, g_c), _ = index_field(rad, west, north, lat, lon)
    r, c = rc(lat, lon, west, north)
    return float(f[r - g_r, c - g_c])


def main():
    rad, west, north = load_subset()
    land = land_mask(rad.shape, west, north)

    fields, core = {}, {}
    for slug in STUDY:
        lat, lon = CITY_CENTRES[slug]
        f, origin, km = index_field(rad, west, north, lat, lon)
        r, c = rc(lat, lon, west, north)
        core[slug] = float(f[r - origin[0], c - origin[1]])
        fields[slug] = (f, origin, km, r, c)
    med = float(np.median(list(core.values())))

    print("Walker-law skyglow index (VIIRS VNL 2022, d^-2.5 within 250 km), "
          "normalised to the eight-city median downtown value = 1.00")
    print(f"{'city':10s} {'downtown':>8s} {'box rad':>8s}  escape km to index <= median/"
          + "/".join(str(x) for x in FRACTIONS))
    import csv
    box = {r["slug"]: float(r["radiance"])
           for r in csv.DictReader(open(ROOT / "results" / "viirs_dnb.csv"))}
    esc = {x: {} for x in FRACTIONS}
    for slug in sorted(STUDY, key=lambda s: -core[s]):
        f, (g_r, g_c), (ky, kx), r, c = fields[slug]
        L = np.zeros(f.shape, bool)
        r_a, c_a = max(g_r, 0), max(g_c, 0)
        r_b, c_b = min(g_r + f.shape[0], land.shape[0]), min(g_c + f.shape[1], land.shape[1])
        L[r_a - g_r:r_b - g_r, c_a - g_c:c_b - g_c] = land[r_a:r_b, c_a:c_b]
        yy = (np.arange(f.shape[0]) - (r - g_r))[:, None] * ky
        xx = (np.arange(f.shape[1]) - (c - g_c))[None, :] * kx
        dist = np.hypot(yy, xx)
        row = []
        for x in FRACTIONS:
            ok = L & (f <= med / x)
            e = float(dist[ok].min()) if ok.any() else float("nan")
            esc[x][slug] = e
            row.append(f"{e:6.0f}")
        print(f"{slug:10s} {core[slug] / med:8.2f} {box[slug]:8.1f}  " + "  ".join(row))

    print("\nReference: index at RASC Dark-Sky Preserves (same normalisation)")
    for name, (lat, lon) in PRESERVES.items():
        print(f"  {name:24s} {point_index(rad, west, north, lat, lon) / med:.4f}"
              f"  (= median/{med / point_index(rad, west, north, lat, lon):.0f})")

    # Rank correlations against the corrected good and low-jet nights (headline_corrected.py).
    with (Path(__file__).parent / "results" / "headline_corrected.csv").open() as f:
        recs = list(csv.DictReader(f))
    good = {r["site"]: float(r["good"]) for r in recs}
    lowjet = {r["site"]: float(r["lowjet"]) for r in recs}
    print("\nSpearman rank correlations (n = 8; |rho| >= 0.74 needed for p < 0.05 two-sided)")
    for label, a in (("good", good), ("low-jet", lowjet)):
        print(f"  {label:8s} vs downtown index {spearman(a, core):+.2f}; vs box radiance "
              f"{spearman(a, box):+.2f}; vs escape (1/100) {spearman(a, esc[100]):+.2f}")

if __name__ == "__main__":
    main()
