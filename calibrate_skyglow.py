"""
calibrate_skyglow.py — bring the Falchi et al. (2016) sky-brightness atlas up to 2025 lights.

The atlas (falchi.py) is calibrated but built on 2014 VIIRS radiances. This script:

  1. runs the Walker-law model of skyglow.py on the 2014 VIIRS annual composite (EOG VNL v2.1,
     the year the atlas used) and fits it to the atlas at every land cell within 450 km of the
     eight study cities: a single scale factor per exponent, with the exponent chosen to
     minimise the RMS difference in total zenith brightness (mag/arcsec^2) over cells with
     artificial brightness >= 5 ucd/m^2;
  2. validates the calibrated 2014 model against the atlas downtown and in escape distance;
  3. applies the same calibration to the 2025 composite (EOG VNL v2, NPP), giving downtown
     brightness and escape distances for current lights, and the 2014->2025 change.

Both composites are NPP-only, but they come from different EOG processing versions (v2.1 and
v2), and VIIRS is nearly blind to the blue light of LEDs; the 2025 values are therefore a lower
bound on current brightness, and the change is indicative.

Usage: python3 calibrate_skyglow.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds

import skyglow as sg
from falchi import NATURAL, PRESERVES, mag
from fetch_viirs_dnb import CACHE_DIR, CITY_CENTRES

ROOT = Path(__file__).parent
FALCHI = ROOT / "data" / "falchi_cache" / "World_Atlas_2015.tif"
YEARS = {
    2014: "VNL_v21_npp_2014_global_vcmslcfg_c202205302300.average_masked.dat.tif.gz",
    2025: "VNL_npp_2025_global_vcmslcfg_v2_c202604011200.average_masked.dat.tif.gz",
}
EXPONENTS = (2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5)
FIT_FLOOR = 5e-3             # mcd/m^2: fit only cells with artificial brightness >= 5 ucd/m^2
THRESH = (0.087, 0.014)      # mcd/m^2, escape thresholds (Falchi class boundaries)
STUDY = sg.STUDY


def load_year(year: int):
    """1-arcmin radiance subset for one composite year, cached (same box and grid as skyglow)."""
    cache = CACHE_DIR / f"canada_south_1arcmin_{year}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["rad"], float(z["west"]), float(z["north"])
    src = CACHE_DIR / YEARS[year]
    if not src.exists():
        sys.exit(f"missing {src}")
    print(f"reading {year} composite (one-time)...", file=sys.stderr)
    with rasterio.open(f"/vsigzip/{src}") as ds:
        win = from_bounds(sg.W, sg.S, sg.E, sg.N, ds.transform).round_offsets().round_lengths()
        raw = ds.read(1, window=win).astype(np.float32)
        west, north = ds.window_transform(win) * (0, 0)
    raw = np.nan_to_num(np.maximum(raw, 0.0))
    r, c = (raw.shape[0] // sg.AGG) * sg.AGG, (raw.shape[1] // sg.AGG) * sg.AGG
    rad = raw[:r, :c].reshape(r // sg.AGG, sg.AGG, c // sg.AGG, sg.AGG).mean(axis=(1, 3))
    np.savez_compressed(cache, rad=rad, west=west, north=north)
    return rad, west, north


def region(rad, west, north, land, slug):
    """Model field around one city plus its cell-centre coordinates, distances and land mask."""
    lat, lon = CITY_CENTRES[slug]
    f, (g_r, g_c), (ky, kx) = sg.index_field(rad, west, north, lat, lon)
    r, c = sg.rc(lat, lon, west, north)
    L = np.zeros(f.shape, bool)
    r_a, c_a = max(g_r, 0), max(g_c, 0)
    r_b, c_b = min(g_r + f.shape[0], land.shape[0]), min(g_c + f.shape[1], land.shape[1])
    L[r_a - g_r:r_b - g_r, c_a - g_c:c_b - g_c] = land[r_a:r_b, c_a:c_b]
    lats = north - (g_r + np.arange(f.shape[0]) + 0.5) * sg.CELL_DEG
    lons = west + (g_c + np.arange(f.shape[1]) + 0.5) * sg.CELL_DEG
    dist = np.hypot((np.arange(f.shape[0]) - (r - g_r))[:, None] * ky,
                    (np.arange(f.shape[1]) - (c - g_c))[None, :] * kx)
    return f, lats, lons, dist, L, (r - g_r, c - g_c)


def falchi_at(ds, lats, lons):
    """Atlas value (mcd/m^2) at each cell centre, nearest 30-arcsec pixel; NaN off the map."""
    win = from_bounds(lons.min(), lats.min(), lons.max(), lats.max(), ds.transform)
    win = win.round_offsets().round_lengths()
    a = ds.read(1, window=Window(win.col_off - 1, win.row_off - 1, win.width + 2, win.height + 2))
    tr = ds.window_transform(Window(win.col_off - 1, win.row_off - 1, win.width + 2, win.height + 2))
    rows = np.clip(((lats - tr.f) / tr.e).astype(int), 0, a.shape[0] - 1)
    cols = np.clip(((lons - tr.c) / tr.a).astype(int), 0, a.shape[1] - 1)
    out = a[rows[:, None], cols[None, :]].astype(np.float64)
    out[out < 0] = np.nan
    return out


def escapes(field, dist, L):
    res = []
    for t in THRESH:
        ok = L & (field < t) & (dist <= sg.REGION_KM)
        res.append(float(dist[ok].min()) if ok.any() else float("nan"))
    return res


def main():
    rad14, west, north = load_year(2014)
    rad25, w2, n2 = load_year(2025)
    assert (w2, n2) == (west, north) and rad25.shape == rad14.shape
    land = sg.land_mask(rad14.shape, west, north)

    with rasterio.open(FALCHI) as ds:
        atlas = {}
        for slug in STUDY:
            sg.EXPONENT = 2.5
            f, lats, lons, dist, L, ctr = region(rad14, west, north, land, slug)
            atlas[slug] = (falchi_at(ds, lats, lons), dist, L, ctr)

    # 1. Fit: for each exponent, one scale factor (median log ratio), scored by RMS mag error.
    print("Calibration of the Walker-law model (2014 VIIRS v2.1) to the Falchi et al. atlas")
    print(f"{'exponent':>8s} {'log10 k':>8s} {'RMS mag':>8s} {'n cells':>8s}")
    best = None
    fields14 = {}
    for e in EXPONENTS:
        sg.EXPONENT = e
        xs, ys, per = [], [], {}
        for slug in STUDY:
            f = region(rad14, west, north, land, slug)[0]
            F, dist, L, _ = atlas[slug]
            per[slug] = f
            m = L & (dist <= sg.REGION_KM) & np.isfinite(F) & (F >= FIT_FLOOR) & (f > 0)
            xs.append(f[m]); ys.append(F[m])
        x, y = np.concatenate(xs), np.concatenate(ys)
        logk = float(np.median(np.log10(y) - np.log10(x)))
        rms = float(np.sqrt(np.mean((mag(10 ** logk * x) - mag(y)) ** 2)))
        print(f"{e:8.2f} {logk:8.3f} {rms:8.3f} {x.size:8d}")
        if best is None or rms < best[2]:
            best = (e, logk, rms)
            fields14 = per
    e, logk, rms = best
    k = 10 ** logk
    print(f"best exponent {e}, RMS {rms:.2f} mag/arcsec^2")

    # 2. Validate the calibrated 2014 model against the atlas.
    print("\nValidation, 2014 lights: atlas vs calibrated model")
    print(f"{'city':10s} {'atlas mag':>9s} {'model mag':>9s}   atlas esc 87/14   model esc 87/14")
    for slug in STUDY:
        F, dist, L, (r, c) = atlas[slug]
        mod = k * fields14[slug]
        ea, em = escapes(np.nan_to_num(F, nan=np.inf), dist, L), escapes(mod, dist, L)
        print(f"{slug:10s} {mag(F[r, c]):9.2f} {mag(mod[r, c]):9.2f}   "
              f"{ea[0]:5.0f} {ea[1]:5.0f}       {em[0]:5.0f} {em[1]:5.0f}")

    # 3. Apply to 2025 lights.
    sg.EXPONENT = e
    print(f"\nCalibrated model, 2025 lights (VIIRS v2, NPP), exponent {e}")
    print(f"{'city':10s} {'art. mcd':>8s} {'x nat':>6s} {'mag':>6s} {'esc87':>6s} {'esc14':>6s}"
          f"   downtown change 2014->2025")
    for slug in sorted(STUDY, key=lambda s: -k * fields14[s][atlas[s][3]]):
        f25, _, _, dist, L, (r, c) = region(rad25, west, north, land, slug)
        mod25, mod14 = k * f25, k * fields14[slug]
        e25 = escapes(mod25, dist, L)
        ch = mod25[r, c] / mod14[r, c] - 1
        print(f"{slug:10s} {mod25[r, c]:8.2f} {mod25[r, c] / NATURAL:6.1f} {mag(mod25[r, c]):6.2f} "
              f"{e25[0]:6.0f} {e25[1]:6.0f}   {100 * ch:+5.0f}%")

    print("\nDark-Sky Preserves, calibrated model: 2014 -> 2025 (mag/arcsec^2)")
    for name, (lat, lon) in PRESERVES.items():
        v14 = k * sg.point_index(rad14, west, north, lat, lon)
        v25 = k * sg.point_index(rad25, west, north, lat, lon)
        print(f"  {name:24s} {mag(v14):6.2f} -> {mag(v25):6.2f}")


if __name__ == "__main__":
    main()
