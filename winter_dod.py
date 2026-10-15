"""
winter_dod.py — the nighttime clearing artifact in winter, measured against the ceilometers.

The bias correction uses a gap measured on snow-free ground, because snow on the ground
confuses the GOES cloud mask, so the satellite comparison cannot measure the artifact in
winter. A ceilometer measures cloud base directly, the same way by day and by night, and
snow does not affect it. At the four cities with an automated partner, the
human-minus-ceilometer difference of differences computed in winter (DJF) is therefore an
estimate of the winter artifact.

The definitions follow the bias measurement, not the budget: night = Sun <= -12 deg, day =
Sun >= +6 deg, usable = okta <= 2, averaged per night or day and then over units, with a
bootstrap over units. Pairs come from sites.py: Calgary CYYC/CYBW, Edmonton CYEG/CZVL,
Toronto CYYZ/CYTZ, Montreal CYUL/CYHU. winter_scaled.py applies the result.

Usage: python3 winter_dod.py
"""
from __future__ import annotations

import datetime as dt
import random
from collections import defaultdict
from zoneinfo import ZoneInfo

import numpy as np

import observable_nights as on
from fetch_metar import fetch_metar_csv
from climatology import calculate_sun_alt, get_season
from sites import SITES

YEARS = (2020, 2025)
SEASONS = ["WINTER", "SPRING", "SUMMER", "AUTUMN"]
# the four automated ceilometer partners, read from the registry rather than repeated
# here. Vancouver's CYXX and Halifax's CYAW are human-augmented (see auto_screen.py) and
# sites.py only as screening candidates, so they are excluded explicitly.
NOT_CEILOMETER = {"vancouver", "halifax"}
PAIRS = [(slug, s.human, s.auto) for slug, s in SITES.items()
         if s.auto and slug not in NOT_CEILOMETER]
random.seed(42)


def unit_fracs(icao, site):
    """season -> (night_unit_fracs, day_unit_fracs): per-unit usable means, >=2 obs/unit."""
    tz = ZoneInfo(site.iana_tz)
    nightu = defaultdict(lambda: defaultdict(list))  # season -> night_key -> [usable]
    dayu = defaultdict(lambda: defaultdict(list))    # season -> local date -> [usable]
    for y in range(YEARS[0], YEARS[1] + 1):
        for o in fetch_metar_csv(icao, dt.date(y, 1, 1), dt.date(y, 12, 31)):
            a = calculate_sun_alt(o.timestamp, site.lat, site.lon, site.elevation_m)
            u = o.coverage_okta <= 2
            if a <= -12.0:
                nk = (o.timestamp.astimezone(tz) - dt.timedelta(hours=12)).date()
                nightu[get_season(nk.month)][nk].append(u)
            elif a >= 6.0:
                d = o.timestamp.astimezone(tz).date()
                dayu[get_season(d.month)][d].append(u)
    nf = {s: [float(np.mean(v)) for v in u.values() if len(v) >= 2] for s, u in nightu.items()}
    df = {s: [float(np.mean(v)) for v in u.values() if len(v) >= 2] for s, u in dayu.items()}
    return nf, df


def delta(nf, df):
    return 100.0 * (np.mean(nf) - np.mean(df)) if nf and df else float("nan")


def main():
    print("Night(sun<=-12)-minus-day usable-fraction delta (pp) by season; DoD = human - auto.\n")
    for slug, human, auto in PAIRS:
        s = SITES[slug]
        hnf, hdf = unit_fracs(human, s)
        anf, adf = unit_fracs(auto, s)
        print(f"=== {slug.upper()}  human {human} / auto {auto} ===")
        print(f"{'season':7s} {'H n%':>6s} {'H d%':>6s} {'H Δ':>6s} | "
              f"{'A n%':>6s} {'A d%':>6s} {'A Δ':>6s} | {'DoD':>6s}")
        dods = {}
        allnf_h = {"WINTER": [], "SPRING": [], "SUMMER": [], "AUTUMN": []}
        for se in SEASONS + ["ALL"]:
            if se == "ALL":
                g = lambda d: [x for s2 in SEASONS for x in d.get(s2, [])]
                hn, hd, an, ad = g(hnf), g(hdf), g(anf), g(adf)
            else:
                hn, hd, an, ad = hnf.get(se, []), hdf.get(se, []), anf.get(se, []), adf.get(se, [])
            hdel, adel = delta(hn, hd), delta(an, ad)
            dods[se] = hdel - adel
            print(f"{se:7s} {100*np.mean(hn):6.1f} {100*np.mean(hd):6.1f} {hdel:+6.1f} | "
                  f"{100*np.mean(an):6.1f} {100*np.mean(ad):6.1f} {adel:+6.1f} | {dods[se]:+6.1f}")

        # night/day-unit bootstrap CI on the WINTER DoD
        def boot(nf, df):
            nb = np.random.choice(nf, len(nf), replace=True)
            db = np.random.choice(df, len(df), replace=True)
            return np.mean(nb) - np.mean(db)
        hw_n, hw_d = hnf["WINTER"], hdf["WINTER"]
        aw_n, aw_d = anf["WINTER"], adf["WINTER"]
        np.random.seed(42)
        reps = [100 * (boot(hw_n, hw_d) - boot(aw_n, aw_d)) for _ in range(2000)]
        lo, hi = np.percentile(reps, [2.5, 97.5])
        print(f"  WINTER DoD = {dods['WINTER']:+.1f} pp  95% CI [{lo:+.1f}, {hi:+.1f}]   "
              f"(ALL-season DoD {dods['ALL']:+.1f}; validate Calgary ALL ~ +17.4)\n")


if __name__ == "__main__":
    main()
