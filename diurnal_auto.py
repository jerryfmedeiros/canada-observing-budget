"""
diurnal_auto.py — the diurnal cycle of usable sky at the human stations and their ceilometers.

The human reports show usable sky lowest mid-afternoon and highest after midnight. Part of
the overnight rise is the observers' nighttime clearing artifact. This script asks which parts
of the cycle the four screened fully automated partners (CYBW, CYHU, CZVL, CYTZ), which carry
no night-vision bias, also show. For each pair it gives the usable fraction (okta <= 2) by
local clock hour, 2020-2025, for both stations, and the summary contrasts:

  afternoon  = 13:00-16:00 local
  evening    = 21:00-23:00 local
  late night = 01:00-04:00 local
  morning    = 07:00-10:00 local

A rise from afternoon to evening in the ceilometer record is real cloud change; a rise from
evening to late night that the human record shows and the ceilometer does not is the artifact.

Usage: python3 diurnal_auto.py
"""
from __future__ import annotations

import datetime as dt
import sys
from collections import defaultdict
from zoneinfo import ZoneInfo

import numpy as np

try:
    import observable_nights as on
    from fetch_metar import fetch_metar_csv
    from sites import SITES
except ImportError as e:
    sys.exit(f"Run from the repo root: {e}")

YEARS = (2020, 2025)
NOT_CEILOMETER = {"vancouver", "halifax"}   # their 'auto' neighbours are human-augmented
WINDOWS = {"afternoon": range(13, 17), "evening": range(21, 24),
           "late night": range(1, 5), "morning": range(7, 11)}


def by_hour(icao, tz):
    n, u = np.zeros(24), np.zeros(24)
    for y in range(YEARS[0], YEARS[1] + 1):
        for o in fetch_metar_csv(icao, dt.date(y, 1, 1), dt.date(y, 12, 31)):
            h = o.timestamp.astimezone(tz).hour
            n[h] += 1
            u[h] += o.coverage_okta <= on.USABLE_OKTA
    return u, n


def window(u, n, hours):
    hs = list(hours)
    return 100 * u[hs].sum() / n[hs].sum()


def main():
    print("Usable-sky fraction (%) by local hour, human station vs fully automated partner\n")
    summary = []
    for slug, site in SITES.items():
        if not site.auto or slug in NOT_CEILOMETER:
            continue
        tz = ZoneInfo(site.iana_tz)
        hu, hn = by_hour(site.human, tz)
        au, an = by_hour(site.auto, tz)
        print(f"{slug} ({site.human} human, {site.auto} automated)")
        print("  hour   " + " ".join(f"{h:>4d}" for h in range(24)))
        print("  human  " + " ".join(f"{100 * hu[h] / hn[h]:4.0f}" for h in range(24)))
        print("  auto   " + " ".join(f"{100 * au[h] / an[h]:4.0f}" for h in range(24)))
        w = {k: (window(hu, hn, v), window(au, an, v)) for k, v in WINDOWS.items()}
        summary.append((slug, w))
        print()

    print("Window means (%): human / automated")
    print(f"{'city':10s} " + " ".join(f"{k:>15s}" for k in WINDOWS)
          + f" {'eve-aft':>13s} {'late-eve':>13s}")
    for slug, w in summary:
        cells = " ".join(f"{w[k][0]:6.1f} /{w[k][1]:6.1f}" for k in WINDOWS)
        ea = (w["evening"][0] - w["afternoon"][0], w["evening"][1] - w["afternoon"][1])
        le = (w["late night"][0] - w["evening"][0], w["late night"][1] - w["evening"][1])
        print(f"{slug:10s} {cells} {ea[0]:+6.1f}/{ea[1]:+6.1f} {le[0]:+6.1f}/{le[1]:+6.1f}")


if __name__ == "__main__":
    main()
