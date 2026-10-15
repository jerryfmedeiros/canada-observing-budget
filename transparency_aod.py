"""
transparency_aod.py — what the water-vapour transparency bar does and does not measure.

The good-night filter calls a night transparent when its median dark-hour PWV is <= 15 mm.
That follows observing practice (the Clear Sky Chart's transparency forecast is computed from
total column water vapour and does not consider smoke), but in the visible band water vapour
is not itself the absorber: haze is. This script asks two things of MERRA-2 column AOD
(550 nm), the direct measure of visible extinction:

  1. Are humid nights hazier? Association of nightly PWV and AOD on smoke-clean nights, raw
     and with each calendar month's median removed, and the AOD distribution either side of
     the 15 mm bar.
  2. Does the ranking depend on the choice? Good nights with the PWV test replaced by an AOD
     ceiling (aod_ranking.py adds AOD to PWV instead), bias-corrected, at the paper's
     observable-night bar.

Nightly AOD is the median over the night's astronomical-dark report hours, the same hours the
PWV median uses.

Usage: python3 transparency_aod.py
"""
from __future__ import annotations

import datetime as dt
import sys
from collections import defaultdict

import numpy as np

try:
    import observable_nights as on
    from climatology import calculate_sun_alt
    from fetch_metar import fetch_metar_csv
    from gaps import gap
    from nights import JET_TEST, OBSERVABLE_H, YEARS, corrected_frac, per_night
    from sites import SITES
    from smoke_column import load_aod
except ImportError as e:
    sys.exit(f"Run from the repo root: {e}")

AOD_CEILINGS = (0.10, 0.15, 0.20)
HAZY = 0.30   # the column depth the smoke checks treat as hazy


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def nightly_aod(site, dates) -> np.ndarray:
    """Median MERRA-2 AOD over each night's astronomical-dark report hours (NaN if none)."""
    aod = load_aod(site.slug)
    on.LAT, on.LON, on.ELEVATION_M = site.lat, site.lon, site.elevation_m
    from zoneinfo import ZoneInfo
    on.LOCAL_TZ = ZoneInfo(site.iana_tz)
    vals = defaultdict(list)
    for year in range(YEARS[0], YEARS[1] + 1):
        for o in fetch_metar_csv(site.human, dt.date(year, 1, 1), dt.date(year, 12, 31)):
            if calculate_sun_alt(o.timestamp, site.lat, site.lon, site.elevation_m) > -18.0:
                continue
            hr = on.round_hour_utc(o.timestamp)
            a = aod.get((f"{hr:%Y-%m-%d}", hr.hour))
            if a is not None:
                vals[on.night_key(o.timestamp)].append(a)
    return np.array([float(np.median(vals[d])) if vals.get(d) else np.nan for d in dates])


def main():
    bar = OBSERVABLE_H
    print("1. Are humid nights hazier? (smoke-clean nights with both PWV and AOD)\n")
    print(f"{'city':10s} {'n':>5s} {'rho':>6s} {'rho, month':>10s}   "
          f"{'median AOD':>21s}   {'share AOD > ' + str(HAZY):>21s}")
    print(f"{'':10s} {'':5s} {'':6s} {'removed':>10s}   "
          f"{'<=15 mm':>10s} {'>15 mm':>10s}   {'<=15 mm':>10s} {'>15 mm':>10s}")
    pooled_p, pooled_a = [], []
    good = {}
    for slug, site in SITES.items():
        rec = per_night(site)
        a = nightly_aod(site, rec.dates)
        p = rec.pwv_median
        mon = np.array([d.month for d in rec.dates])
        ok = ~np.isnan(a) & ~np.isnan(p) & rec.smoke_clean

        pd_, ad_ = p.copy(), a.copy()
        for m in range(1, 13):
            k = ok & (mon == m)
            if k.sum() > 5:
                pd_[k] -= np.median(p[k])
                ad_[k] -= np.median(a[k])
        dry, wet = ok & (p <= on.PWV_TRANSPARENT), ok & (p > on.PWV_TRANSPARENT)
        print(f"{slug:10s} {ok.sum():5d} {spearman(p[ok], a[ok]):+6.2f} "
              f"{spearman(pd_[ok], ad_[ok]):+10.2f}   "
              f"{np.median(a[dry]):10.3f} {np.median(a[wet]):10.3f}   "
              f"{100 * np.mean(a[dry] > HAZY):9.1f}% {100 * np.mean(a[wet] > HAZY):9.1f}%",
              flush=True)
        pooled_p += list(p[ok])
        pooled_a += list(a[ok])

        ny = rec.dates[-1].year - rec.dates[0].year + 1
        obs = (corrected_frac(rec, gap("satellite", slug)) * rec.dark_h >= bar) & rec.valid
        base = obs & rec.smoke_clean & (rec.low_jet if JET_TEST else True)
        row = {"PWV<=15": (base & rec.transparent).sum() / ny}
        for t in AOD_CEILINGS:
            row[f"AOD<={t:.2f}"] = (base & (a <= t)).sum() / ny
        row["none"] = base.sum() / ny
        good[slug] = row

    pp, pa = np.array(pooled_p), np.array(pooled_a)
    dry, wet = pp <= on.PWV_TRANSPARENT, pp > on.PWV_TRANSPARENT
    print(f"{'POOLED':10s} {len(pp):5d} {spearman(pp, pa):+6.2f} {'':10s}   "
          f"{np.median(pa[dry]):10.3f} {np.median(pa[wet]):10.3f}   "
          f"{100 * np.mean(pa[dry] > HAZY):9.1f}% {100 * np.mean(pa[wet] > HAZY):9.1f}%")
    d_ext = 1.086 * (np.median(pa[wet]) - np.median(pa[dry]))
    print(f"\nMedian extra zenith extinction above the bar: {d_ext:.2f} mag "
          f"(1.086 x the AOD difference).")

    cols = list(next(iter(good.values())))
    print(f"\n2. Good nights/yr with the transparency test replaced (corrected, "
          f">= {bar:g} h bar)\n")
    print(f"{'city':10s} " + " ".join(f"{c:>10s}" for c in cols))
    for slug in sorted(good, key=lambda s: -good[s]["PWV<=15"]):
        print(f"{slug:10s} " + " ".join(f"{good[slug][c]:10.1f}" for c in cols))
    print("\nRanking under each test:")
    for c in cols:
        print(f"  {c:10s}: " + " > ".join(s[:3] for s in sorted(good, key=lambda s: -good[s][c])))


if __name__ == "__main__":
    main()
