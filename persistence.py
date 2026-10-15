"""
persistence.py — does a clear evening stay clear?

Takes the report nearest 22:00 local as each night's evening state and the reports nearest
01:00 and 03:00 as the outcomes, each within 45 minutes, and gives three conditional rates
per city with Wilson intervals: a clear evening still clear at 01:00, still clear at 03:00,
and a cloudy evening clear by 03:00.

All three rest on late-night reports and so inherit the nighttime clearing bias. The rate of
clearing after a cloudy evening is therefore an upper bound; the two hold rates, ratios of
similarly biased quantities, are less affected but still optimistic.

--auto computes the same rates from the four fully automated partners (CYBW, CYHU, CZVL,
CYTZ) alongside their human stations. The ceilometers have no night-vision bias (they miss
high thin cloud, but equally at every hour), so the comparison shows how much the clearing
artifact changes the rates.

Usage: python3 persistence.py [--auto]
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from collections import defaultdict
from zoneinfo import ZoneInfo

try:
    import observable_nights as on
    from fetch_metar import fetch_metar_csv
    from sites import SITES
except ImportError as e:
    sys.exit(f"Run from the repo root: {e}")

YEARS = (2020, 2025)
EVENING_H = 22          # local hour taken as the evening state
OUTCOME_H = (1, 3)      # local hours taken as the outcomes
MATCH_MIN = 45          # a report counts for a slot only within this many minutes


def nearest_states(site, icao: str | None = None):
    """Per night: {local_hour: usable_bool} for the evening and outcome slots."""
    tz = ZoneInfo(site.iana_tz)
    on.LAT, on.LON, on.ELEVATION_M = site.lat, site.lon, site.elevation_m
    on.LOCAL_TZ = tz

    # night -> slot -> (minutes_off, usable)
    best: dict = defaultdict(dict)
    for year in range(YEARS[0], YEARS[1] + 1):
        for o in fetch_metar_csv(icao or site.human, dt.date(year, 1, 1), dt.date(year, 12, 31)):
            local = o.timestamp.astimezone(tz)
            for slot in (EVENING_H,) + OUTCOME_H:
                target = local.replace(hour=slot, minute=0, second=0, microsecond=0)
                off = abs((local - target).total_seconds()) / 60.0
                if off > MATCH_MIN:
                    continue
                nk = on.night_key(o.timestamp)
                prev = best[nk].get(slot)
                if prev is None or off < prev[0]:
                    best[nk][slot] = (off, o.coverage_okta <= on.USABLE_OKTA)
    return {nk: {s: v[1] for s, v in slots.items()} for nk, slots in best.items()}


def rates(st):
    """(hold@01, hold@03, recover@03), each as (pct, lo, hi, n)."""
    h1 = [v for v in st.values() if EVENING_H in v and 1 in v and v[EVENING_H]]
    h3 = [v for v in st.values() if EVENING_H in v and 3 in v and v[EVENING_H]]
    rc = [v for v in st.values() if EVENING_H in v and 3 in v and not v[EVENING_H]]

    def rate(sample, slot):
        k = sum(1 for v in sample if v[slot])
        n = len(sample)
        lo, hi = on.wilson_ci(k, n) if n else (0.0, 0.0)
        return 100 * k / n if n else float("nan"), 100 * lo, 100 * hi, n

    return rate(h1, 1), rate(h3, 3), rate(rc, 3)


NOT_CEILOMETER = {"vancouver", "halifax"}   # their 'auto' neighbours are human-augmented


def main_auto():
    print("Hold-or-fold, human station vs its fully automated partner "
          f"({YEARS[0]}-{YEARS[1]})\n")
    print(f"{'city':10s} {'station':8s} {'hold@01':>22s} {'hold@03':>22s} {'recover@03':>22s}")
    for slug, site in SITES.items():
        if not site.auto or slug in NOT_CEILOMETER:
            continue
        for icao in (site.human, site.auto):
            a, b, c = rates(nearest_states(site, icao))
            print(f"{slug:10s} {icao:8s} {a[0]:6.1f}% [{a[1]:5.1f},{a[2]:5.1f}] "
                  f"{b[0]:6.1f}% [{b[1]:5.1f},{b[2]:5.1f}] "
                  f"{c[0]:6.1f}% [{c[1]:5.1f},{c[2]:5.1f}]", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--auto", action="store_true")
    if ap.parse_args().auto:
        return main_auto()
    print(f"Evening hold-or-fold, {YEARS[0]}-{YEARS[1]} (evening = {EVENING_H:02d}:00 local, "
          f"outcomes {OUTCOME_H[0]:02d}:00 / {OUTCOME_H[1]:02d}:00, +/-{MATCH_MIN} min)\n")
    print(f"{'city':10s} {'hold@01':>22s} {'hold@03':>22s} {'recover@03':>22s}   "
          f"{'n(clear eve)':>12s} {'n(cloudy eve)':>13s}")
    rows = {}
    for slug, site in SITES.items():
        st = nearest_states(site)
        h1 = [v for v in st.values() if EVENING_H in v and 1 in v and v[EVENING_H]]
        h3 = [v for v in st.values() if EVENING_H in v and 3 in v and v[EVENING_H]]
        rc = [v for v in st.values() if EVENING_H in v and 3 in v and not v[EVENING_H]]

        def rate(sample, slot):
            k = sum(1 for v in sample if v[slot])
            n = len(sample)
            lo, hi = on.wilson_ci(k, n) if n else (0.0, 0.0)
            return 100 * k / n if n else float("nan"), 100 * lo, 100 * hi, n

        a = rate(h1, 1)
        b = rate(h3, 3)
        c = rate(rc, 3)
        rows[slug] = (a, b, c)
        print(f"{slug:10s} {a[0]:6.1f}% [{a[1]:5.1f},{a[2]:5.1f}] "
              f"{b[0]:6.1f}% [{b[1]:5.1f},{b[2]:5.1f}] "
              f"{c[0]:6.1f}% [{c[1]:5.1f},{c[2]:5.1f}]   "
              f"{b[3]:12d} {c[3]:13d}", flush=True)

    h1 = [rows[s][0][0] for s in rows]
    h3 = [rows[s][1][0] for s in rows]
    rc = [rows[s][2][0] for s in rows]
    print(f"\nAcross the eight cities:")
    print(f"  a usable 22:00 is still usable at 01:00 on {min(h1):.0f}-{max(h1):.0f}% of nights")
    print(f"  ... and at 03:00 on {min(h3):.0f}-{max(h3):.0f}%")
    print(f"  a cloudy 22:00 has recovered by 03:00 on {min(rc):.0f}-{max(rc):.0f}%")
    stickiest = min(rows, key=lambda s: rows[s][2][0])
    print(f"  stickiest (lowest recovery): {stickiest} "
          f"(holds {rows[stickiest][1][0]:.0f}% to 03:00, recovers {rows[stickiest][2][0]:.0f}%)")
    changeable = max(rows, key=lambda s: rows[s][2][0])
    print(f"  most changeable (best recovery): {changeable} "
          f"(holds {rows[changeable][1][0]:.0f}%, recovers {rows[changeable][2][0]:.0f}%)")


if __name__ == "__main__":
    main()
