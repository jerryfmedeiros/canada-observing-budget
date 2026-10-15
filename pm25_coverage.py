"""
pm25_coverage.py — how exposed is the smoke screen where NAPS PM2.5 is missing?

A night with no surface PM2.5 behind its clear dark reports passes the smoke test unless the
observer logged FU. This script reports how many observable nights are in that position, by
city and by year, and asks MERRA-2 column AOD (550 nm) how hazy those nights were. The AOD of
a night is taken over its astronomical-dark report hours, the hours the smoke test uses;
both the median and the peak are checked against the threshold.

Three groups are compared:
  - uncovered observable nights with high AOD (did the other tests remove them?),
  - uncovered good nights with high AOD (smoky nights that got through),
  - covered good nights with high AOD (what the full screen lets through, for scale).

--corrected applies the bias correction (nearer-satellite gap, measured shape) to every
night's usable fraction before the observable test, so the check runs on the corrected
counts the paper reports rather than the raw ones.

Usage: python3 pm25_coverage.py [--corrected]
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from collections import defaultdict
from zoneinfo import ZoneInfo

import numpy as np

try:
    import observable_nights as on
    from climatology import calculate_sun_alt
    from fetch_metar import fetch_metar_csv
    from gaps import gap
    from nights import OBSERVABLE_H, YEARS, corrected_frac, per_night
    from sites import SITES
    from smoke_column import AOD_THR, load_aod
except ImportError as e:
    sys.exit(f"Run from the repo root: {e}")

HOURS = OBSERVABLE_H   # nights.py; 4 h unless BUDGET_HOURS is set


def night_aod(site, slug, dates):
    """(median, peak) MERRA-2 AOD over each night's astronomical-dark report hours."""
    aod = load_aod(slug)
    on.LAT, on.LON, on.ELEVATION_M = site.lat, site.lon, site.elevation_m
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
    med = np.array([float(np.median(vals[d])) if vals.get(d) else np.nan for d in dates])
    peak = np.array([float(np.max(vals[d])) if vals.get(d) else np.nan for d in dates])
    return med, peak


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corrected", action="store_true")
    args = ap.parse_args()
    years = list(range(YEARS[0], YEARS[1] + 1))
    print(f"NAPS PM2.5 coverage behind the smoke test, and MERRA-2 column AOD "
          f"(threshold {AOD_THR:.2f}) on the nights it misses")
    print(f"Observable = >= {HOURS:g} h expected clear dark"
          f"{', bias-corrected' if args.corrected else ', raw'}. "
          f"AOD over each night's astronomical-dark report hours.\n")

    by_year, rows = {}, {}
    tot = defaultdict(int)
    for slug, site in SITES.items():
        rec = per_night(site)
        g = gap("satellite", slug) if args.corrected else 0.0
        obs = ((corrected_frac(rec, g) * rec.dark_h) >= HOURS) & rec.valid
        good = obs & rec.quality
        med, peak = night_aod(site, slug, rec.dates)
        yr = np.array([d.year for d in rec.dates])
        by_year[slug] = [100.0 * (obs & rec.pm25_covered & (yr == y)).sum()
                         / max((obs & (yr == y)).sum(), 1) for y in years]
        u, gu, gc = obs & ~rec.pm25_covered, good & ~rec.pm25_covered, good & rec.pm25_covered
        r = dict(obs=obs.sum(), uncov=u.sum(),
                 u_med=(u & (med > AOD_THR)).sum(), u_peak=(u & (peak > AOD_THR)).sum(),
                 g_uncov=gu.sum(),
                 gu_med=(gu & (med > AOD_THR)).sum(), gu_peak=(gu & (peak > AOD_THR)).sum(),
                 g_cov=gc.sum(),
                 gc_med=(gc & (med > AOD_THR)).sum(), gc_peak=(gc & (peak > AOD_THR)).sum())
        r = {k: int(v) for k, v in r.items()}
        r["top"] = float(np.nanmax(np.where(gu, peak, np.nan))) if gu.any() else float("nan")
        rows[slug] = r
        for k, v in r.items():
            if k != "top":
                tot[k] += v

    print("1. Share of observable nights with PM2.5 behind them, by year (%)")
    print(f"{'city':10s} " + " ".join(f"{y:>5d}" for y in years))
    for slug in SITES:
        print(f"{slug:10s} " + " ".join(f"{v:5.0f}" for v in by_year[slug]))

    print(f"\n2. Nights with no PM2.5 behind them, and their column AOD "
          f"(median / peak above {AOD_THR:.2f})")
    hdr = (f"{'city':10s} {'obs':>5s} {'uncov':>6s} {'uncov%':>7s} | {'median':>6s} {'peak':>5s} | "
           f"{'good unc.':>9s} {'median':>6s} {'peak':>5s} {'top peak':>8s} | "
           f"{'good cov.':>9s} {'median':>6s} {'peak':>5s}")
    print(hdr)

    def line(name, r, top):
        print(f"{name:10s} {r['obs']:5d} {r['uncov']:6d} {100.0 * r['uncov'] / r['obs']:6.1f}% | "
              f"{r['u_med']:6d} {r['u_peak']:5d} | {r['g_uncov']:9d} {r['gu_med']:6d} "
              f"{r['gu_peak']:5d} {top:8.2f} | {r['g_cov']:9d} {r['gc_med']:6d} {r['gc_peak']:5d}")

    for slug in SITES:
        line(slug, rows[slug], rows[slug]["top"])
    print("-" * len(hdr))
    line("POOLED", tot, float(np.nanmax([r["top"] for r in rows.values()])))

    shares = {s: 100.0 * r["uncov"] / r["obs"] for s, r in rows.items()}
    worst = max(shares, key=lambda s: shares[s])
    rest = sorted(v for k, v in shares.items() if k != worst)
    print(f"\nCoverage is thinnest at {worst} ({shares[worst]:.0f}% of its observable nights "
          f"have no surface PM2.5), against {rest[0]:.0f}-{rest[-1]:.0f}% elsewhere.")
    print(f"Uncovered observable nights with median AOD above {AOD_THR:.2f}: {tot['u_med']}; "
          f"of the uncovered good nights, {tot['gu_med']} (peak above: {tot['gu_peak']}).")
    print(f"Good nights with no PM2.5 behind them: {tot['g_uncov']} pooled, "
          f"{tot['g_uncov'] / len(years):.0f}/yr; with PM2.5: {tot['g_cov']}, of which "
          f"{tot['gc_med']} have median AOD above {AOD_THR:.2f} ({tot['gc_peak']} by peak).")


if __name__ == "__main__":
    main()
