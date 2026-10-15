"""
fetch_era5.py — ERA5 water vapour (PWV) and 200 hPa and 10 m wind for each site.

PWV (total column water vapour, mm) is the transparency test. The 200 hPa wind is the
jet-stream test: it indicates how fast turbulence moves aloft and is used in site testing
as a seeing proxy, but it is not a seeing measurement and is not validated at these cities,
so the paper reports it only as a subset of the good nights. The 10 m wind is a secondary
condition of the same test.

Source: Copernicus Climate Data Store (CDS); needs a free CDS account.

Setup (once):
  1. Create an account at https://cds.climate.copernicus.eu and log in.
  2. Accept the ERA5 licence on the dataset pages
       reanalysis-era5-single-levels  and  reanalysis-era5-pressure-levels.
  3. pip install "cdsapi>=0.7"
  4. Put your Personal Access Token in ~/.cdsapirc:
       url: https://cds.climate.copernicus.eu/api
       key: <your-personal-access-token>

Requests are made one month at a time to stay under the CDS cost limit (the ERA5 timeseries
datasets do not carry these variables). The CDS queues each request, so a full pull takes
hours; cached months are skipped, so an interrupted run can be restarted.
  python3 fetch_era5.py --site calgary --years 2020 2025

NetCDF is read with netCDF4; xarray is not needed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
import sys
from collections import defaultdict
from pathlib import Path

import netCDF4
import numpy as np

try:
    from climatology import calculate_sun_alt, get_season
except ImportError:
    sys.exit("Error: could not import climatology.py.")

PROJECT_ROOT = Path(__file__).parent.resolve()
CACHE_DIR = PROJECT_ROOT / "data" / "era5_cache"

LAT, LON, ELEVATION_M = 51.05, -114.07, 1043     # Calgary, the default when run alone
# Small box around the site (ERA5 is 0.25 deg); we take the nearest grid cell.
AREA = [51.25, -114.25, 51.0, -114.0]   # N, W, S, E
SEEING_LEVEL = "200"                      # hPa


def cache_dir_for(slug: str | None) -> Path:
    """Per-site ERA5 cache directory, data/era5_cache/<slug>/ (Calgary when slug is None)."""
    return CACHE_DIR / (slug or "calgary")

# Transparency tiers (PWV mm): lower is better. Continental dry nights ~ <10 mm.
PWV_TIERS = (10.0, 15.0, 20.0)
# 200 hPa wind tiers (m/s): above about 30 m/s the jet stream is overhead.
WIND_TIERS = (20.0, 30.0, 40.0)
# 10 m wind tiers (m/s). Strong surface wind stirs turbulence near the ground; ERA5's grid
# is about 31 km, so this is a regional value, not the wind at the airport.
WIND10_TIERS = (5.0, 8.0, 12.0)

SEASONS = ["WINTER", "SPRING", "SUMMER", "AUTUMN"]


# --------------------------------------------------------------------------- #
# CDS download (one NetCDF per year per dataset; cached)
# --------------------------------------------------------------------------- #
# CDS enforces a per-request cost (field-count) limit, so a full year of hourly data
# in one request is rejected ("request too large"). There is no pressure-levels
# timeseries dataset, and the single-levels timeseries subset omits total column
# water vapour -- so we use the standard hourly datasets, chunked by month. Each
# request is ~1 month x 31 days x 24 h x (1-2 vars), comfortably under the limit.
def _retrieve_month(dataset, variables, year, month, target, level=None, area=None):
    import cdsapi  # imported lazily so the module loads without it installed
    target.parent.mkdir(parents=True, exist_ok=True)
    req = {
        "product_type": ["reanalysis"],
        "variable": variables,
        "year": [str(year)],
        "month": [f"{month:02d}"],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": area if area is not None else AREA,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    if level is not None:
        req["pressure_level"] = [level]
    tmp = target.with_suffix(".part")
    cdsapi.Client().retrieve(dataset, req, str(tmp))
    tmp.replace(target)


def download_month(kind: str, year: int, month: int,
                   area: list[float] | None = None,
                   cache_dir: Path = CACHE_DIR / "calgary") -> Path | None:
    """Download one month of PWV, 200 hPa wind, or 10 m wind; cached. None on failure.

    `area` (ERA5 [N,W,S,E] box) and `cache_dir` default to Calgary's."""
    if kind == "pwv":
        target = cache_dir / f"era5_pwv_{year}-{month:02d}.nc"
        dataset, variables, level = ("reanalysis-era5-single-levels",
                                     ["total_column_water_vapour"], None)
    elif kind == "wind10":
        target = cache_dir / f"era5_wind10_{year}-{month:02d}.nc"
        dataset, variables, level = ("reanalysis-era5-single-levels",
                                     ["10m_u_component_of_wind", "10m_v_component_of_wind"], None)
    else:
        target = cache_dir / f"era5_wind{SEEING_LEVEL}_{year}-{month:02d}.nc"
        dataset, variables, level = ("reanalysis-era5-pressure-levels",
                                     ["u_component_of_wind", "v_component_of_wind"], SEEING_LEVEL)
    if target.exists() and target.stat().st_size > 1024:
        return target
    print(f"  CDS {kind} {year}-{month:02d} (queued server-side) ...", file=sys.stderr)
    try:
        _retrieve_month(dataset, variables, year, month, target, level, area=area)
    except Exception as e:
        print(f"  {kind} {year}-{month:02d} failed: {e}", file=sys.stderr)
        return None
    return target


# --------------------------------------------------------------------------- #
# NetCDF parsing (netCDF4; tolerant of new/old CDS coordinate names)
# --------------------------------------------------------------------------- #
def _first_var(nc, candidates):
    for name in candidates:
        if name in nc.variables:
            return name
    return None


def _time_values(nc):
    tname = _first_var(nc, ["valid_time", "time"])
    tvar = nc.variables[tname]
    times = netCDF4.num2date(tvar[:], tvar.units,
                             only_use_cftime_datetimes=False, only_use_python_datetimes=True)
    return [t.replace(tzinfo=dt.timezone.utc) for t in np.atleast_1d(times)]


def _nearest_idx(arr, value):
    return int(np.argmin(np.abs(np.asarray(arr) - value)))


def sample_scalar_series(path: Path, var_candidates, deriver=None,
                         lat_site: float = LAT, lon_site: float = LON) -> dict[dt.datetime, float]:
    """Return {utc_hour: value} at the grid cell nearest (lat_site, lon_site).

    deriver(dict_of_vars_at_cell_over_time) -> np.array lets us combine u,v -> speed.
    """
    out: dict[dt.datetime, float] = {}
    with netCDF4.Dataset(str(path)) as nc:
        times = _time_values(nc)
        lat_name = _first_var(nc, ["latitude", "lat"])
        lon_name = _first_var(nc, ["longitude", "lon"])
        # Nearest grid index per axis, computed independently: the ERA5 area box
        # can come back as a 1x1, 1x2, 2x1, or 2x2 cell block depending on how its
        # edges fall on the 0.25-deg grid, and lat and lon need not both be
        # singletons (Toronto's box straddles a longitude line -> longitude size 2).
        iy = ix = 0
        if lat_name and nc.variables[lat_name].size > 1:
            iy = _nearest_idx(np.asarray(nc.variables[lat_name][:]), lat_site)
        if lon_name and nc.variables[lon_name].size > 1:
            lonarr = np.asarray(nc.variables[lon_name][:])
            lon_target = lon_site if lonarr.min() < 0 else lon_site % 360
            ix = _nearest_idx(lonarr, lon_target)

        def cell_series(varname):
            # Index by dimension NAME rather than by squeezing (which is ambiguous
            # when a spatial axis has size 2): keep the whole time axis, pick the
            # nearest lat/lon cell, and take index 0 of any other axis (e.g. the
            # singleton pressure_level left by the single-level request).
            v = nc.variables[varname]
            data = np.asarray(v[:])
            sel = []
            for d in v.dimensions:
                dl = d.lower()
                if "time" in dl:
                    sel.append(slice(None))
                elif d == lat_name or "lat" in dl:
                    sel.append(iy)
                elif d == lon_name or "lon" in dl:
                    sel.append(ix)
                else:
                    sel.append(0)  # pressure_level or other singleton axis
            return np.atleast_1d(data[tuple(sel)])

        if deriver is None:
            vname = _first_var(nc, var_candidates)
            series = cell_series(vname)
        else:
            series = deriver({c: cell_series(c) for c in var_candidates})

        for t, val in zip(times, np.atleast_1d(series)):
            if np.ma.is_masked(val) or (isinstance(val, float) and math.isnan(val)):
                continue
            out[t.replace(minute=0, second=0, microsecond=0)] = float(val)
    return out


# The three builders default to Calgary; the analysis scripts pass each site's lat/lon,
# area and cache_dir_for(slug).
def build_pwv_index(years: tuple[int, int], lat: float = LAT, lon: float = LON,
                    area: list[float] | None = None,
                    cache_dir: Path = CACHE_DIR / "calgary") -> dict[dt.datetime, float]:
    idx: dict[dt.datetime, float] = {}
    for year in range(years[0], years[1] + 1):
        for month in range(1, 13):
            p = download_month("pwv", year, month, area=area, cache_dir=cache_dir)
            if p:
                idx.update(sample_scalar_series(p, ["tcwv", "p54.162", "total_column_water_vapour"],
                                                lat_site=lat, lon_site=lon))
        print(f"  PWV {year}: {len(idx)} cumulative hours", file=sys.stderr)
    return idx


def build_wind_index(years: tuple[int, int], lat: float = LAT, lon: float = LON,
                     area: list[float] | None = None,
                     cache_dir: Path = CACHE_DIR / "calgary") -> dict[dt.datetime, float]:
    idx: dict[dt.datetime, float] = {}
    spd = lambda d: np.hypot(d["u"], d["v"])
    for year in range(years[0], years[1] + 1):
        for month in range(1, 13):
            p = download_month("wind", year, month, area=area, cache_dir=cache_dir)
            if p:
                idx.update(sample_scalar_series(p, ["u", "v"], deriver=spd,
                                                lat_site=lat, lon_site=lon))
        print(f"  wind {year}: {len(idx)} cumulative hours", file=sys.stderr)
    return idx


def build_wind10_index(years: tuple[int, int], lat: float = LAT, lon: float = LON,
                       area: list[float] | None = None,
                       cache_dir: Path = CACHE_DIR / "calgary") -> dict[dt.datetime, float]:
    """10 m wind speed (m/s)."""
    idx: dict[dt.datetime, float] = {}
    spd = lambda d: np.hypot(d["u10"], d["v10"])
    for year in range(years[0], years[1] + 1):
        for month in range(1, 13):
            p = download_month("wind10", year, month, area=area, cache_dir=cache_dir)
            if p:
                idx.update(sample_scalar_series(p, ["u10", "v10"], deriver=spd,
                                                lat_site=lat, lon_site=lon))
        print(f"  wind10 {year}: {len(idx)} cumulative hours", file=sys.stderr)
    return idx


# --------------------------------------------------------------------------- #
# Climatology report
# --------------------------------------------------------------------------- #
def _tier_table(title, index, tiers, lo_is_good, y0, y1, unit,
                lat=LAT, lon=LON, elevation=ELEVATION_M):
    by_year = defaultdict(lambda: defaultdict(int))
    by_season = defaultdict(lambda: defaultdict(int))
    for ts, val in index.items():
        if calculate_sun_alt(ts, lat, lon, elevation) > -18.0:
            continue
        for bucket in (by_year[ts.year], by_season[get_season(ts.month)]):
            bucket["n"] += 1
            bucket["sum"] += val
            for t in tiers:
                if (val <= t) if lo_is_good else (val >= t):
                    bucket[t] += 1

    rel = "<=" if lo_is_good else ">="
    print(f"\n[{title}] dark-hours by tier ({unit}); fraction {rel} tier")
    hdr = " | ".join(f"{rel}{int(t)}".rjust(7) for t in tiers)
    print(f"{'bucket':<8} | {'dark-hrs':>8} | {'mean':>6} | {hdr}")
    print("-" * (34 + 10 * len(tiers)))
    for k in list(range(y0, y1 + 1)) + SEASONS:
        c = (by_year if isinstance(k, int) else by_season).get(k)
        if not c or c["n"] == 0:
            continue
        n = c["n"]
        cells = " | ".join(f"{c[t]/n:6.1%}" for t in tiers)
        print(f"{str(k):<8} | {n:8d} | {c['sum']/n:6.1f} | {cells}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", nargs=2, type=int, default=[2020, 2025])
    ap.add_argument("--site", default="calgary",
                    help="site slug from sites.py (sets lat/lon/area/cache subdir)")
    args = ap.parse_args()
    y0, y1 = args.years

    from sites import get_site
    site = get_site(args.site)
    lat, lon, elev = site.lat, site.lon, site.elevation_m
    area, cache_dir = site.era5_area(), cache_dir_for(site.slug)

    print(f"ERA5 water vapour (PWV) and 200 hPa and 10 m wind for "
          f"{site.name} {y0}-{y1}")
    print("=" * 78)
    pwv = build_pwv_index((y0, y1), lat, lon, area, cache_dir)
    wind = build_wind_index((y0, y1), lat, lon, area, cache_dir)
    wind10 = build_wind10_index((y0, y1), lat, lon, area, cache_dir)
    print(f"Loaded {len(pwv)} PWV, {len(wind)} 200hPa-wind, {len(wind10)} 10m-wind hours")

    if pwv:
        _tier_table("TRANSPARENCY (PWV)", pwv, PWV_TIERS, lo_is_good=True,
                    y0=y0, y1=y1, unit="mm", lat=lat, lon=lon, elevation=elev)
    if wind:
        _tier_table("JET STREAM (200 hPa wind)", wind, WIND_TIERS, lo_is_good=True,
                    y0=y0, y1=y1, unit="m/s", lat=lat, lon=lon, elevation=elev)
    if wind10:
        _tier_table("SURFACE WIND (10 m)", wind10, WIND10_TIERS, lo_is_good=True,
                    y0=y0, y1=y1, unit="m/s", lat=lat, lon=lon, elevation=elev)


if __name__ == "__main__":
    main()
