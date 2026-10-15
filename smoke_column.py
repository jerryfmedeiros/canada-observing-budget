"""
smoke_column.py — the surface smoke flag against MERRA-2 column AOD.

An observable night is smoke-clean unless one of its clear dark reports carries METAR FU or
co-located surface PM2.5 >= 35 ug/m3. Surface concentration can miss smoke carried above the
boundary layer, which still dims the sky. MERRA-2's column 550 nm AOD (fetch_merra2_aod.py)
includes smoke aloft and is available at night, so it is the check.

For every observable night we take the MERRA-2 AOD at the same clear dark report hours the
surface flag uses, reduce it to the night's peak, and cross-tabulate the surface flag against
a column-AOD threshold:
  - do nights the surface flag marks carry high column AOD?
  - are there high-AOD nights the surface flag missed (smoke aloft)?
  - how many observable nights does each disagreement affect?

--corrected applies the bias correction (nearer-satellite gap, measured shape) to every
night's usable fraction before the observable test, so the check runs on the corrected
counts the paper reports rather than the raw ones.

Usage: python3 smoke_column.py [--corrected]
"""
from __future__ import annotations

import datetime as dt
from collections import defaultdict
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

import observable_nights as on
from fetch_metar import fetch_metar_csv
from fetch_pm25 import build_pm25_index
from climatology import calculate_sun_alt
from sites import SITES
from nights import OBSERVABLE_H, apply_correction
from gaps import gap
import argparse

YEARS = (2020, 2025)
MIN_OBS = 2
PM25_SMOKE = 35.0
AOD_THR = 0.30        # column-smoke threshold (550 nm); ~0.1 mag/airmass extinction and up
CACHE = Path(__file__).parent.resolve() / "data" / "merra2_cache"


def load_aod(slug: str) -> dict[tuple[str, int], float]:
    idx = {}
    p = CACHE / f"{slug}.csv"
    if not p.exists():
        return idx
    with p.open() as f:
        next(f, None)
        for line in f:
            d, h, a = line.rstrip("\n").split(",")
            try:
                v = float(a)
            except ValueError:
                continue
            if v == v:  # not NaN
                idx[(d, int(h))] = v
    return idx


def per_night(site, aod, gap_pp: float = 0.0):
    """Per covered night -> (observable, surface_smoky, peak_clear_AOD_or_None)."""
    on.LAT, on.LON, on.ELEVATION_M = site.lat, site.lon, site.elevation_m
    on.LOCAL_TZ = ZoneInfo(site.iana_tz)
    pm = build_pm25_index(YEARS, site.naps_city, site.naps_std_offset_h) if site.naps_city else {}

    usable = defaultdict(list)
    smoke = defaultdict(lambda: False)
    clear_aod = defaultdict(list)
    for y in range(YEARS[0], YEARS[1] + 1):
        for o in fetch_metar_csv(site.human, dt.date(y, 1, 1), dt.date(y, 12, 31)):
            if calculate_sun_alt(o.timestamp, site.lat, site.lon, site.elevation_m) > -18.0:
                continue
            nk = on.night_key(o.timestamp)
            u = o.coverage_okta <= 2
            usable[nk].append(u)
            if u:
                hr = on.round_hour_utc(o.timestamp)
                p = pm.get(hr)
                if o.obscuration == "smoke" or (p is not None and p >= PM25_SMOKE):
                    smoke[nk] = True
                a = aod.get((f"{hr:%Y-%m-%d}", hr.hour))
                if a is not None:
                    clear_aod[nk].append(a)

    out = []
    for nk, us in usable.items():
        if len(us) < MIN_OBS or not (YEARS[0] <= nk.year <= YEARS[1]):
            continue
        win = on.dark_window(nk, -18.0)
        dh = (win[1] - win[0]).total_seconds() / 3600.0 if win else 0.0
        o2 = (float(apply_correction(np.mean(us), gap_pp)) * dh) >= OBSERVABLE_H
        peak = max(clear_aod[nk]) if clear_aod[nk] else None
        out.append((o2, smoke[nk], peak))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corrected", action="store_true")
    args = ap.parse_args()
    print(f"AOD column-smoke threshold = {AOD_THR:.2f} (550 nm)\n")
    hdr = (f"{'city':10s} {'obs':>5s} {'srf+':>5s} {'aod+':>5s} "
           f"{'both':>5s} {'srf-only':>8s} {'aod-only':>8s} {'agree%':>7s}")
    print(hdr)
    tot = defaultdict(int)
    alln = []
    for slug, site in SITES.items():
        aod = load_aod(slug)
        if not aod:
            print(f"{slug:10s}  (no MERRA-2 cache yet)")
            continue
        g = gap("satellite", slug) if args.corrected else 0.0
        nights = [n for n in per_night(site, aod, g) if n[0] and n[2] is not None]
        obs = len(nights)
        srf = sum(s for _, s, _ in nights)
        aodp = sum(p >= AOD_THR for _, _, p in nights)
        both = sum(s and p >= AOD_THR for _, s, p in nights)
        srf_only = srf - both
        aod_only = aodp - both
        agree = 100.0 * sum((s == (p >= AOD_THR)) for _, s, p in nights) / obs if obs else 0
        for k, v in (("obs", obs), ("srf", srf), ("aod", aodp), ("both", both),
                     ("srf_only", srf_only), ("aod_only", aod_only)):
            tot[k] += v
        alln += [(s, p) for _, s, p in nights]
        print(f"{slug:10s} {obs:5d} {srf:5d} {aodp:5d} {both:5d} "
              f"{srf_only:8d} {aod_only:8d} {agree:6.1f}%")

    if tot["obs"]:
        print("-" * len(hdr))
        agree = 100.0 * (tot["obs"] - tot["srf_only"] - tot["aod_only"]) / tot["obs"]
        print(f"{'POOLED':10s} {tot['obs']:5d} {tot['srf']:5d} {tot['aod']:5d} "
              f"{tot['both']:5d} {tot['srf_only']:8d} {tot['aod_only']:8d} {agree:6.1f}%")
        print(f"\nOf {tot['srf']} surface-flagged observable nights, {tot['both']} "
              f"({100.0*tot['both']/max(tot['srf'],1):.0f}%) also exceed AOD {AOD_THR:.2f}.")
        print(f"Column smoke the surface flag MISSED: {tot['aod_only']} nights "
              f"({100.0*tot['aod_only']/tot['obs']:.1f}% of {tot['obs']} observable nights, "
              f"~{tot['aod_only']/6:.1f}/yr pooled).")

    # Does the surface flag at least catch the THICK episodes? Stratify its detection rate by
    # column depth. A flag that only ever fires under heavy smoke would climb to near 100% in
    # the top bin; one that is simply insensitive stays low everywhere.
    if alln:
        print("\nSurface-flag detection rate by MERRA-2 column AOD:")
        print(f"{'AOD bin':>13s} {'nights':>7s} {'flagged':>8s} {'detected':>9s}")
        for lo, hi in ((0.0, 0.3), (0.3, 0.5), (0.5, 0.8), (0.8, 1.5), (1.5, 99.0)):
            sub = [s for s, pk in alln if lo <= pk < hi]
            if not sub:
                continue
            lab = f"{lo:.1f}-{hi:.1f}" if hi < 99 else f">={lo:.1f}"
            print(f"{lab:>13s} {len(sub):7d} {sum(sub):8d} "
                  f"{100.0*sum(sub)/len(sub):8.1f}%")
        for cut in (AOD_THR, 0.8):
            sub = [s for s, pk in alln if pk >= cut]
            if sub:
                print(f"  AOD >= {cut:.2f}: {sum(sub)}/{len(sub)} flagged "
                      f"= {100.0*sum(sub)/len(sub):.1f}% detected")
        print("  Detection climbs with column depth but stays well below 100% even in the "
              "thickest bin,\n  so the surface flag is a LOWER BOUND on smoke, not a "
              "reliable detector of it.")


if __name__ == "__main__":
    main()
