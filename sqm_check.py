"""
sqm_check.py — test the Falchi et al. (2016) atlas against citizen SQM readings.

Globe at Night (globeatnight.org/maps-data, CC BY 4.0) publishes every report by year, including
Sky Quality Meter readings (mag/arcsec^2). This takes the SQM readings within 150 km of the
eight study cities, keeps those reported clear, taken with the Sun below -18 deg and the Moon
below the horizon, and compares each with the atlas total zenith brightness at its location.

Reports the residual (SQM minus atlas; negative = sky brighter than the atlas), pooled and by
city, and its trend with year (city fixed effects), as a ground check on the atlas's 2014 basis.
The 2020 file is unavailable from the archive and is skipped. Readings are sparse outside
Toronto, Vancouver, Montreal and Ottawa, and SQMs see a ~20 deg cone rather than the zenith,
so this is a check on the atlas, not a replacement for it.

Usage: python3 sqm_check.py
"""
from __future__ import annotations

import glob
from pathlib import Path

import ephem
import numpy as np
import pandas as pd
import rasterio

from falchi import mag
from fetch_viirs_dnb import CITY_CENTRES

ROOT = Path(__file__).parent
FALCHI = ROOT / "data" / "falchi_cache" / "World_Atlas_2015.tif"
STUDY = ["calgary", "winnipeg", "edmonton", "montreal", "vancouver", "ottawa", "toronto", "halifax"]
RADIUS_KM = 150.0
B, SEED = 2000, 42


def load():
    dfs = []
    for f in sorted(glob.glob(str(ROOT / "data" / "sqm_cache" / "GaN20*.csv"))):
        d = pd.read_csv(f, low_memory=False)
        d["year"] = int(Path(f).stem[-4:])
        dfs.append(d)
    d = pd.concat(dfs, ignore_index=True)
    d["sqm"] = pd.to_numeric(d["SQMReading"], errors="coerce")
    d = d[d.sqm.between(14, 23)].copy()
    d["city"], d["dist"] = None, np.inf
    for s in STUDY:
        la, lo = CITY_CENTRES[s]
        dist = np.hypot((d.Latitude - la) * 111.2, (d.Longitude - lo) * 111.2 * np.cos(np.radians(la)))
        take = (dist < RADIUS_KM) & (dist < d["dist"])
        d.loc[take, "city"], d.loc[take, "dist"] = s, dist[take]
    return d[d.city.notna()].copy()


def dark_and_moonless(row) -> bool:
    try:
        obs = ephem.Observer()
        obs.lat, obs.lon = str(row.Latitude), str(row.Longitude)
        obs.date = f"{row.UTDate} {row.UTTime}"
    except Exception:
        return False
    sun, moon = ephem.Sun(obs), ephem.Moon(obs)
    return np.degrees(sun.alt) < -18 and np.degrees(moon.alt) < 0


def main():
    d = load()
    n_all = len(d)
    d = d[d.CloudCover.astype(str).str.contains("clear", case=False)]
    n_clear = len(d)
    d = d[d.apply(dark_and_moonless, axis=1)].copy()
    print(f"Globe at Night SQM readings within {RADIUS_KM:.0f} km: {n_all}; clear {n_clear}; "
          f"clear, astronomically dark and moonless {len(d)}")

    with rasterio.open(FALCHI) as ds:
        vals = np.array([v[0] for v in ds.sample(zip(d.Longitude, d.Latitude))], float)
    vals[vals < 0] = np.nan
    d["atlas"] = mag(vals)
    d["resid"] = d.sqm - d.atlas
    d = d[np.isfinite(d.resid)]

    rng = np.random.default_rng(SEED)
    def med_ci(x):
        x = np.asarray(x)
        reps = np.median(rng.choice(x, (B, x.size)), axis=1)
        return np.median(x), *np.percentile(reps, [2.5, 97.5])

    m, lo, hi = med_ci(d.resid)
    print(f"\nResidual SQM - atlas (mag/arcsec^2; negative = brighter than atlas), n = {len(d)}")
    print(f"  pooled median {m:+.2f} [{lo:+.2f}, {hi:+.2f}]; "
          f"Spearman(SQM, atlas) = {d[['sqm', 'atlas']].corr(method='spearman').iloc[0, 1]:+.2f}")
    for s in STUDY:
        x = d[d.city == s]
        if len(x) >= 5:
            m, lo, hi = med_ci(x.resid)
            print(f"  {s:10s} n {len(x):3d}  median {m:+.2f} [{lo:+.2f}, {hi:+.2f}]  "
                  f"years {x.year.min()}-{x.year.max()}")
        else:
            print(f"  {s:10s} n {len(x):3d}  (too few)")

    for label, sel in (("2014-2017", d.year <= 2017), ("2021-2025", d.year >= 2021)):
        x = d[sel]
        if len(x) >= 5:
            m, lo, hi = med_ci(x.resid)
            print(f"  {label}  n {len(x):3d}  median {m:+.2f} [{lo:+.2f}, {hi:+.2f}]")

    # Trend: residual on year with city fixed effects (demeaned), bootstrap over readings.
    x = d.copy()
    x["r_dm"] = x.resid - x.groupby("city").resid.transform("mean")
    x["y_dm"] = x.year - x.groupby("city").year.transform("mean")
    x = x[x.groupby("city").year.transform("nunique") > 1]
    def slope(df):
        v = (df.y_dm ** 2).sum()
        return (df.y_dm * df.r_dm).sum() / v if v > 0 else np.nan
    s0 = slope(x)
    reps = [slope(x.sample(len(x), replace=True, random_state=int(rng.integers(1 << 31))))
            for _ in range(B)]
    lo, hi = np.nanpercentile(reps, [2.5, 97.5])
    print(f"\nTrend in residual, city fixed effects (n = {len(x)}): {s0:+.3f} mag/arcsec^2/yr "
          f"[{lo:+.3f}, {hi:+.3f}]  (negative = brightening relative to the 2014-based atlas)")


if __name__ == "__main__":
    main()
