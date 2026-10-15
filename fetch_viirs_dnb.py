"""
fetch_viirs_dnb.py — city-centre VIIRS night-lights radiance.

Writes results/viirs_dnb.csv: the mean upward radiance of an 11x11 pixel box on each city
core, from the VIIRS Nighttime Lights V2.2 annual composite for 2022 (Elvidge et al. 2021,
Earth Observation Group). light_pollution.py reads it, and skyglow.py uses a 1-arcmin
subset of the same composite.

At 15 arc-second resolution an 11x11 box spans about 5.1 km north-south and 3.3 km east-west
at Canadian latitudes. This measures light escaping upward from the core, not sky brightness
at any observing site. Centres are downtown coordinates, not the airports in sites.py.

The composite (about 310 MB) is too large to ship. The repository instead holds
data/viirs_dnb_cache/city_cores_15arcsec_2022.npz, a 41x41 pixel window of the composite
around each centre (EOG's CC BY 4.0 licence), and the script reads that when the composite
is absent. To work from the composite itself, download it with a free EOG account (or set
EOG_TOKEN); --cores rebuilds the windows from it.

The mean is taken in the raster's own float32. Accumulating in float64 changes the eighth
significant figure, which does not matter for the radiance but breaks byte-identity with
the shipped CSV.

Usage:
  python3 fetch_viirs_dnb.py            # write results/viirs_dnb.csv
  python3 fetch_viirs_dnb.py --verify   # recompute and diff against it
  python3 fetch_viirs_dnb.py --cores    # rebuild the city windows from the composite
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

try:
    import numpy as np
    import rasterio
    from rasterio.transform import rowcol
    from rasterio.windows import Window
    from affine import Affine
except ImportError as e:
    sys.exit(f"needs numpy and rasterio (pip install rasterio): {e}")

ROOT = Path(__file__).parent
CACHE_DIR = ROOT / "data" / "viirs_dnb_cache"
TIF_NAME = "VNL_v22_npp-j01_2022_global_vcmslcfg_c202303062300.average_masked.dat.tif.gz"
URL = ("https://eogdata.mines.edu/nighttime_light/annual/v22/2022/" + TIF_NAME)
OUT = ROOT / "results" / "viirs_dnb.csv"
CORES = CACHE_DIR / "city_cores_15arcsec_2022.npz"
CORE_HALF = 20       # windows are 41x41 pixels, wide enough for the box with room to spare

YEAR = 2022
BOX_PX = 11          # 11x11 15-arcsec pixels, about 5.1 km N-S by 3.3 km E-W here

# Canadian metro cores, (lat, lon) in degrees, in the order the published CSV lists them.
# The paper uses only the eight study cities; the other CMAs are kept for context.
CITY_CENTRES = {
    "toronto": (43.6532, -79.3832),
    "montreal": (45.5017, -73.5673),
    "vancouver": (49.2827, -123.1207),
    "ottawa": (45.4215, -75.6972),
    "calgary": (51.0447, -114.0719),
    "edmonton": (53.5461, -113.4938),
    "quebec_city": (46.8139, -71.2080),
    "winnipeg": (49.8951, -97.1384),
    "hamilton": (43.2557, -79.8711),
    "kitchener": (43.4516, -80.4925),
    "london": (42.9849, -81.2453),
    "halifax": (44.6488, -63.5752),
    "stcatharines": (43.1594, -79.2469),
    "windsor": (42.3149, -83.0364),
    "oshawa": (43.8971, -78.8658),
    "victoria": (48.4284, -123.3656),
    "saskatoon": (52.1332, -106.6700),
    "regina": (50.4452, -104.6189),
    "sherbrooke": (45.4042, -71.8929),
    "stjohns": (47.5615, -52.7126),
    "kelowna": (49.8880, -119.4960),
    "barrie": (44.3894, -79.6903),
    "abbotsford": (49.0504, -122.3045),
    "guelph": (43.5448, -80.2482),
    "kingston": (44.2312, -76.4860),
    "sudbury": (46.4917, -80.9930),
    "troisrivieres": (46.3432, -72.5430),
    "saguenay": (48.4167, -71.0667),     # Chicoutimi, the lit core of the amalgamated city
    "moncton": (46.0878, -64.7782),
    "saintjohn": (45.2733, -66.0633),
    "thunderbay": (48.3809, -89.2477),
    "fredericton": (45.9636, -66.6431),
    "charlottetown": (46.2382, -63.1311),
}

DISPLAY = {
    "quebec_city": "Quebec City", "stcatharines": "St. Catharines-Niagara",
    "kitchener": "Kitchener-Cambridge-Waterloo", "stjohns": "St. John's",
    "abbotsford": "Abbotsford-Mission", "sudbury": "Greater Sudbury",
    "troisrivieres": "Trois-Rivieres", "saintjohn": "Saint John",
    "thunderbay": "Thunder Bay",
}


def display_name(slug: str) -> str:
    return DISPLAY.get(slug, slug.replace("_", " ").title())


def ensure_raster() -> Path:
    """Return the cached composite, downloading it only if EOG_TOKEN is set."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    local = CACHE_DIR / TIF_NAME
    if local.exists():
        return local
    token = os.environ.get("EOG_TOKEN")
    if not token:
        sys.exit(
            f"missing {local}\n"
            f"The EOG annual products sit behind a free account, so this cannot be fetched\n"
            f"anonymously. Either download it in a browser from\n  {URL}\n"
            f"and save it at that path, or register at\n"
            f"  https://eogdata.mines.edu/products/register/\n"
            f"and re-run with EOG_TOKEN=<bearer token>.")
    import urllib.request
    print(f"downloading {TIF_NAME} (~295 MB) ...", flush=True)
    req = urllib.request.Request(URL, headers={"Authorization": f"Bearer {token}"})
    tmp = local.with_suffix(local.suffix + ".part")
    with urllib.request.urlopen(req) as r, tmp.open("wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.rename(local)
    return local


def city_radiance(path: Path) -> dict[str, float]:
    """Mean radiance of the BOX_PX x BOX_PX box at each city centre, nW/cm^2/sr."""
    half = BOX_PX // 2
    out = {}
    # /vsigzip/ lets GDAL read the compressed file in place, so no ~12 GB decompression.
    with rasterio.open(f"/vsigzip/{path}") as ds:
        for slug, (lat, lon) in CITY_CENTRES.items():
            row, col = ds.index(lon, lat)
            win = Window(col - half, row - half, BOX_PX, BOX_PX)
            # Mean in the raster's own float32, which is what produced the published CSV.
            # Accumulating in float64 instead shifts the 8th significant figure -- far below
            # any meaning for a ~100 nW/cm^2/sr radiance, but it would break byte-identity.
            block = ds.read(1, window=win)
            if block.shape != (BOX_PX, BOX_PX):
                sys.exit(f"{slug}: window fell outside the raster -- check its coordinates")
            out[slug] = float(np.nanmean(block))
    return out


def write_cores(path: Path) -> None:
    """Save a (2*CORE_HALF+1)-pixel square of the composite around each city centre."""
    arrays = {}
    with rasterio.open(f"/vsigzip/{path}") as ds:
        for slug, (lat, lon) in CITY_CENTRES.items():
            row, col = ds.index(lon, lat)
            side = 2 * CORE_HALF + 1
            arrays[slug] = ds.read(1, window=Window(col - CORE_HALF, row - CORE_HALF, side, side))
            arrays[f"{slug}_origin"] = np.array([row - CORE_HALF, col - CORE_HALF])
        transform = np.array(ds.transform[:6])
    np.savez_compressed(CORES, transform=transform, **arrays)
    print(f"wrote {CORES} ({len(CITY_CENTRES)} cities)")


def core_radiance() -> dict[str, float]:
    """The same box means as city_radiance, read from the saved city windows."""
    half = BOX_PX // 2
    z = np.load(CORES)
    transform = Affine(*z["transform"])
    out = {}
    for slug, (lat, lon) in CITY_CENTRES.items():
        row, col = rowcol(transform, lon, lat)
        r0, c0 = z[f"{slug}_origin"]
        r, c = row - r0, col - c0
        block = z[slug][r - half:r + half + 1, c - half:c + half + 1]
        out[slug] = float(np.nanmean(block))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify", action="store_true",
                    help="recompute and diff against the existing CSV instead of writing it")
    ap.add_argument("--cores", action="store_true",
                    help="rebuild the city windows from the composite")
    args = ap.parse_args()

    if args.cores:
        write_cores(ensure_raster())
        return 0
    if (CACHE_DIR / TIF_NAME).exists():
        rad = city_radiance(CACHE_DIR / TIF_NAME)
    elif CORES.exists():
        rad = core_radiance()
    else:
        rad = city_radiance(ensure_raster())

    if args.verify:
        if not OUT.exists():
            sys.exit(f"{OUT} does not exist yet; run without --verify first")
        old = {r["slug"]: float(r["radiance"]) for r in csv.DictReader(OUT.open())}
        bad = 0
        for slug, val in rad.items():
            ref = old.get(slug)
            if ref is None:
                print(f"  {slug:15s} NEW (not in CSV)")
                bad += 1
            elif abs(ref - val) > 1e-4:
                print(f"  {slug:15s} {val:10.4f} != {ref:10.4f} in CSV")
                bad += 1
        print(f"{len(rad) - bad}/{len(rad)} city values reproduce the published CSV")
        return 0 if bad == 0 else 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["year", "slug", "name", "radiance"])
        for slug, val in rad.items():
            w.writerow([YEAR, slug, display_name(slug), repr(val)])
    print(f"wrote {OUT} ({len(rad)} cities, {BOX_PX}x{BOX_PX} box, VNL v2.2 {YEAR})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
