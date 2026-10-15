"""
smoke_summer.py — what the Prairie summer smoke rates rest on.

The seasonal smoke table gives the share of clear astronomical-dark report hours with
surface PM2.5 >= 35 ug/m3. In summer the Prairie cities have little astronomical darkness
(none in June, almost none in July at Edmonton), so those rates come from few hours. This
script lists, for each Prairie city and year, the summer (June-August) dark reports behind
the rate: how many were clear, how many clear hours had PM2.5 and exceeded the threshold,
and on how many distinct nights. For comparison it gives the same rate over all dark hours
regardless of cloud, the METAR FU rate, and the share of hours with MERRA-2 column AOD above
0.3, which is available in every year at every city.

Usage: python3 smoke_summer.py
"""
from __future__ import annotations

import datetime as dt
import sys
from zoneinfo import ZoneInfo

try:
    import observable_nights as on
    from climatology import calculate_sun_alt
    from fetch_metar import fetch_metar_csv
    from fetch_pm25 import build_pm25_index
    from sites import SITES
    from smoke_column import AOD_THR, load_aod
except ImportError as e:
    sys.exit(f"Run from the repo root: {e}")

CITIES = ("calgary", "edmonton", "winnipeg")
YEARS = range(2020, 2026)
PM_YEARS = (2020, 2024)      # verified NAPS files
SUMMER = (6, 7, 8)


def reports(slug, months=SUMMER):
    """Astronomical-dark reports in `months` (summer by default):
    (year, month, night, clear, pm25, fu, aod)."""
    site = SITES[slug]
    on.LAT, on.LON, on.ELEVATION_M = site.lat, site.lon, site.elevation_m
    on.LOCAL_TZ = ZoneInfo(site.iana_tz)
    pm = build_pm25_index(PM_YEARS, site.naps_city, site.naps_std_offset_h)
    aod = load_aod(slug)
    out = []
    for y in YEARS:
        for o in fetch_metar_csv(site.human, dt.date(y, 1, 1), dt.date(y, 12, 31)):
            if calculate_sun_alt(o.timestamp, site.lat, site.lon, site.elevation_m) > -18.0:
                continue
            nk = on.night_key(o.timestamp)
            if nk.month not in months:
                continue
            hr = on.round_hour_utc(o.timestamp)
            out.append((nk.year, nk.month, nk, o.coverage_okta <= on.USABLE_OKTA, pm.get(hr),
                        o.obscuration == "smoke", aod.get((f"{hr:%Y-%m-%d}", hr.hour))))
    return out


def pct(a, b):
    return f"{100 * a / b:5.1f}%" if b else "    -"


def main():
    print(f"Summer (June-August) astronomical-dark reports, PM2.5 >= {on.PM25_SMOKE:.0f} ug/m3, "
          f"AOD > {AOD_THR:.2f}\n")
    for slug in CITIES:
        rows = reports(slug)
        print(f"{slug}")
        print(f"  {'':7s} {'dark':>5s} {'clear':>5s} | {'clear with PM2.5':>16s} {'smoky':>5s} "
              f"{'nights':>6s} {'rate':>6s} | {'all-dark rate':>13s} | {'clear FU':>8s} | "
              f"{'clear AOD>thr':>13s} {'all AOD>thr':>11s}")

        def line(lab, sel):
            r = [x for x in rows if sel(x)]
            cl = [x for x in r if x[3]]
            cp = [x for x in cl if x[4] is not None]
            cs = [x for x in cp if x[4] >= on.PM25_SMOKE]
            ap = [x for x in r if x[4] is not None]
            ca = [x for x in cl if x[6] is not None]
            aa = [x for x in r if x[6] is not None]
            print(f"  {lab:7s} {len(r):5d} {len(cl):5d} | {len(cp):16d} {len(cs):5d} "
                  f"{len({x[2] for x in cs}):6d} {pct(len(cs), len(cp))} | "
                  f"{pct(sum(x[4] >= on.PM25_SMOKE for x in ap), len(ap)):>13s} | "
                  f"{pct(sum(x[5] for x in cl), len(cl)):>8s} | "
                  f"{pct(sum(x[6] > AOD_THR for x in ca), len(ca)):>13s} "
                  f"{pct(sum(x[6] > AOD_THR for x in aa), len(aa)):>11s}")

        for y in YEARS:
            line(str(y), lambda x, y=y: x[0] == y)
        line("2020-23", lambda x: x[0] <= 2023)
        line("2020-24", lambda x: x[0] <= 2024)
        line("2020-25", lambda x: True)
        for m in SUMMER:
            line(dt.date(2000, m, 1).strftime("%B")[:7], lambda x, m=m: x[1] == m)
        print()


if __name__ == "__main__":
    main()
