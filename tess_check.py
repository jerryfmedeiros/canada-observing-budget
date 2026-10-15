"""
tess_check.py — ground truth for the sky-brightness atlas from the Quebec TESS photometers.

STARS4ALL TESS-W photometers (tess.stars4all.eu; data ODbL 1.0) log zenith night-sky brightness
(MSAS, mag/arcsec^2, ~17 deg field) about once a minute, together with an infrared sky
temperature. The Quebec network (SEPAQ parks and partners) has 10 sites (14 photometers)
within 200 km of Montreal and Ottawa, from the suburban edge to the Mont-Megantic Dark-Sky
Preserve.

For each site this keeps readings with the Sun below -18 deg and the Moon below -5 deg in
snow-free months (May-October), classes a reading clear when the sky is at least 20 C colder
than the enclosure (the thermopile cloud test), and takes each clear night's median with at
least 60 clear minutes. It then reports, per site: the clear-night median brightness, its
comparison with the Falchi et al. (2016) atlas at the site, and its median by year
(2020-2026), with a pooled trend with site fixed effects.

Usage:
  python3 tess_check.py            # from the shipped data/tess_cache/ (through August 2026)
  python3 tess_check.py --fetch    # first add any newer files from the archive
"""
from __future__ import annotations

import datetime as dt
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import ephem
import numpy as np
import pandas as pd
import rasterio
import requests

from falchi import mag

ROOT = Path(__file__).parent
CACHE = ROOT / "data" / "tess_cache"
FALCHI = ROOT / "data" / "falchi_cache" / "World_Atlas_2015.tif"
DAV = "https://fta-cloud.fis.ucm.es/public.php/webdav/"
AUTH = ("Gr9DAbfiX8Pdm86", "")

# Site -> photometer ids (a replaced unit keeps its site), coordinates, nearest study city (km).
SITES = {
    "Mont-Saint-Bruno":       (["stars645"], 45.546983, -73.345384, "Montreal 18 km"),
    "Plaisance":              (["stars650"], 45.59891, -75.108436, "Ottawa 50 km"),
    "Yamaska":                (["stars643"], 45.428368, -72.61595, "Montreal 75 km"),
    "Mont-Orford":            (["stars655"], 45.36316, -72.23611, "Montreal 105 km"),
    "Mont-Tremblant Pimbina": (["stars473", "stars980"], 46.400249, -74.270052, "Montreal 114 km"),
    "Mont-Tremblant Diable":  (["stars472", "stars979"], 46.316627, -74.511577, "Montreal 117 km"),
    "Sherbrooke Mont-Bellevue": (["stars471", "stars781"], 45.37380198, -71.92304529, "Montreal 129 km"),
    "Saint-Camille":          (["stars60"], 45.734626, -71.676243, "Montreal 150 km"),
    "Mont-Megantic":          (["stars237", "stars657"], 45.42332, -71.125955, "Montreal 190 km"),
    "Frontenac":              (["stars654"], 45.84042, -71.16255, "Montreal 191 km"),
}
YEARS = range(2020, 2027)
MONTHS = range(5, 11)                  # snow-free
SUN_MAX, MOON_MAX = -18.0, -5.0
DT_CLEAR = -20.0                       # sky minus enclosure temperature, C
MIN_CLEAR_MIN = 60
B, SEED = 2000, 42


def dav_list(path):
    r = requests.request("PROPFIND", DAV + path, auth=AUTH, headers={"Depth": "1"}, timeout=60)
    r.raise_for_status()
    return [h.split("/public.php/webdav/")[1] for h in re.findall(r"<d:href>([^<]*)</d:href>", r.text)][1:]


def fetch(pid):
    (CACHE / pid).mkdir(parents=True, exist_ok=True)
    for f in dav_list(pid + "/"):
        m = re.search(r"_(\d{4})-(\d{2})\.dat$", f)
        if not m or int(m.group(1)) not in YEARS or int(m.group(2)) not in MONTHS:
            continue
        out = CACHE / f
        if not out.exists():
            r = requests.get(DAV + f, auth=AUTH, timeout=300)
            r.raise_for_status()
            out.write_bytes(r.content)


def read(pid) -> pd.DataFrame:
    frames = []
    for f in sorted((CACHE / pid).glob("*.dat")):
        d = pd.read_csv(f, sep=";", comment="#", header=None, usecols=[0, 2, 3, 5],
                        names=["utc", "t_box", "t_sky", "msas"], on_bad_lines="skip")
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True)
    d["utc"] = pd.to_datetime(d.utc, errors="coerce", utc=True)
    for c in ("t_box", "t_sky", "msas"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.dropna()


def sun_moon(times: pd.Series, lat, lon):
    """Sun and Moon altitude (deg) at each time, evaluated on a 10-min grid and looked up."""
    grid = times.dt.floor("10min").drop_duplicates()
    obs = ephem.Observer()
    obs.lat, obs.lon = str(lat), str(lon)
    s, m = ephem.Sun(), ephem.Moon()
    alt = {}
    for t in grid:
        obs.date = t.to_pydatetime().replace(tzinfo=None)
        s.compute(obs); m.compute(obs)
        alt[t] = (np.degrees(s.alt), np.degrees(m.alt))
    a = np.array([alt[t] for t in times.dt.floor("10min")])
    return a[:, 0], a[:, 1]


def clear_nights(d, lat, lon) -> pd.DataFrame:
    d = d[(d.msas > 10) & (d.msas < 23)].copy()
    d["sun"], d["moon"] = sun_moon(d.utc, lat, lon)
    d = d[(d.sun < SUN_MAX) & (d.moon < MOON_MAX) & ((d.t_sky - d.t_box) <= DT_CLEAR)]
    d["night"] = (d.utc - pd.Timedelta(hours=12)).dt.date      # UTC-noon night key (UTC-5)
    g = d.groupby("night").msas.agg(["median", "size"])
    g = g[g["size"] >= MIN_CLEAR_MIN].rename(columns={"median": "msas"})
    g["year"] = [n.year for n in g.index]
    return g


def main():
    # The shipped cache (through August 2026) is what the paper used. --fetch adds any
    # newer monthly files from the archive, which changes the results.
    if "--fetch" in sys.argv:
        CACHE.mkdir(exist_ok=True)
        ids = [p for v in SITES.values() for p in v[0]]
        with ThreadPoolExecutor(8) as ex:
            list(ex.map(fetch, ids))

    with rasterio.open(FALCHI) as ds:
        atlas = {k: mag(max(float(next(ds.sample([(v[2], v[1])]))[0]), 0.0)) for k, v in SITES.items()}

    rng = np.random.default_rng(SEED)
    rows, pooled = [], []
    print("Quebec TESS-W photometers: clear, dark, moonless, snow-free nights (May-Oct)")
    print(f"{'site':26s} {'near':16s} {'nights':>6s} {'TESS':>6s} {'atlas':>6s} {'diff':>6s}  "
          "yearly medians")
    for site, (pids, lat, lon, near) in SITES.items():
        d = pd.concat([read(p) for p in pids], ignore_index=True)
        if d.empty:
            print(f"{site:26s} no data"); continue
        g = clear_nights(d, lat, lon)
        if len(g) < 10:
            print(f"{site:26s} {near:16s} {len(g):6d}  (too few clear nights)"); continue
        med = float(g.msas.median())
        yearly = g.groupby("year").msas.agg(["median", "size"])
        ytxt = "  ".join(f"{y}:{r['median']:.2f}({int(r['size'])})" for y, r in yearly.iterrows()
                         if r["size"] >= 5)
        print(f"{site:26s} {near:16s} {len(g):6d} {med:6.2f} {atlas[site]:6.2f} "
              f"{med - atlas[site]:+6.2f}  {ytxt}")
        rows.append((site, med, atlas[site]))
        g = g.assign(site=site)
        pooled.append(g)

    diffs = np.array([m - a for _, m, a in rows])
    print(f"\nTESS minus atlas across {len(rows)} sites: median {np.median(diffs):+.2f}, "
          f"range {diffs.min():+.2f} to {diffs.max():+.2f} mag/arcsec^2 (negative = brighter)")
    r = pd.DataFrame(rows, columns=["site", "tess", "atlas"])
    print(f"Spearman(TESS, atlas) = {r[['tess', 'atlas']].corr(method='spearman').iloc[0, 1]:+.2f}")

    # Pooled trend with site fixed effects, bootstrap over nights.
    p = pd.concat(pooled)
    p = p[p.groupby("site").year.transform("nunique") >= 3].copy()
    p["m_dm"] = p.msas - p.groupby("site").msas.transform("mean")
    p["y_dm"] = p.year - p.groupby("site").year.transform("mean")
    def slope(df):
        return (df.y_dm * df.m_dm).sum() / (df.y_dm ** 2).sum()
    s0 = slope(p)
    idx = np.arange(len(p))
    reps = [slope(p.iloc[rng.choice(idx, idx.size)]) for _ in range(B)]
    lo, hi = np.percentile(reps, [2.5, 97.5])
    print(f"\nPooled trend, site fixed effects ({p.site.nunique()} sites, {len(p)} nights): "
          f"{s0:+.3f} mag/arcsec^2/yr [{lo:+.3f}, {hi:+.3f}]; "
          f"= {100 * (10 ** (-0.4 * s0) - 1):+.1f}%/yr in brightness (negative mag = brightening)")


if __name__ == "__main__":
    main()
