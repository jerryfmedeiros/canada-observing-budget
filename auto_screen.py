"""
auto_screen.py — regenerate data/derived/<slug>/auto_screen.json from the cache.

The station selection the study rests on: a primary station must be human-augmented at
night (many airports revert to AUTO once the tower closes), and a partner used as the
machine reference must be fully automated. Both are read off the METAR AUTO flag at night
rather than assumed from the station's role.

CYXX (Abbotsford) and CYAW (Shearwater) are carried in sites.py as candidate partners for
Vancouver and Halifax and fail the screen, being human-augmented themselves; that is why
those two cities have no ceilometer comparison.

Usage:
  python3 auto_screen.py           # rewrite the JSON files
  python3 auto_screen.py --check   # recompute and compare, writing nothing (used by verify.py)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import defaultdict
from pathlib import Path

try:
    from fetch_metar import fetch_metar_csv
    from climatology import calculate_sun_alt
    from sites import SITES
except ImportError as e:
    sys.exit(f"Run from the repo root: {e}")

YEARS = (2020, 2025)
NIGHT_ALT = -12.0
HUMAN_MAX_PCT_AUTO = 5.0
AUTO_MIN_PCT = 95.0


def screen(site, station: str, role: str) -> dict:
    n_all = n_auto = n_night = n_night_auto = n_night_usable = 0
    for year in range(YEARS[0], YEARS[1] + 1):
        for o in fetch_metar_csv(station, dt.date(year, 1, 1), dt.date(year, 12, 31)):
            n_all += 1
            n_auto += bool(o.is_auto)
            if calculate_sun_alt(o.timestamp, site.lat, site.lon, site.elevation_m) <= NIGHT_ALT:
                n_night += 1
                n_night_auto += bool(o.is_auto)
                n_night_usable += o.coverage_okta <= 2
    pa = 100.0 * n_night_auto / n_night if n_night else None
    if role == "human":
        ok = pa is not None and pa < HUMAN_MAX_PCT_AUTO
        verdict = ("OK human-augmented" if ok else
                   "AUTO at night -> DROP" if pa is not None and pa >= 50 else
                   "MIXED -- inspect" if pa is not None else "no reports")
    else:
        ok = pa is not None and pa > AUTO_MIN_PCT
        verdict = "OK automated" if ok else "not fully AUTO -- inspect"
    return {
        "station": station, "role": role,
        "n_reports": n_all,
        "pct_auto_all": None if not n_all else round(100.0 * n_auto / n_all, 2),
        "night_n": n_night,
        "night_pct_auto": None if pa is None else round(pa, 1),
        "night_usable_pct": None if not n_night else round(100.0 * n_night_usable / n_night, 1),
        "qualifies": ok, "verdict": verdict,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="recompute and compare against the committed JSON without "
                         "writing; exits non-zero if any city differs")
    args = ap.parse_args()

    print(f"{'site':10s} {'station':7s} {'role':6s} {'night N':>8s} "
          f"{'%AUTO night':>11s} {'%AUTO all':>9s}  verdict")
    stale = []
    for slug, site in SITES.items():
        rows = [screen(site, site.human, "human")]
        if site.auto:
            rows.append(screen(site, site.auto, "auto"))
        out = Path("data") / "derived" / slug
        payload = json.dumps(
            {"site": slug, "human": site.human, "auto": site.auto,
             "years": list(YEARS), "night_solar_alt_max_deg": NIGHT_ALT,
             "stations": rows}, indent=2) + "\n"
        target = out / "auto_screen.json"
        if args.check:
            # Read-only: this is what verify.py runs, so it must not touch the
            # committed evidence it is checking.
            if not target.exists() or target.read_text() != payload:
                stale.append(slug)
        else:
            out.mkdir(parents=True, exist_ok=True)
            target.write_text(payload)
        for r in rows:
            pa = "n/a" if r["night_pct_auto"] is None else f"{r['night_pct_auto']:.1f}%"
            aa = "n/a" if r["pct_auto_all"] is None else f"{r['pct_auto_all']:.1f}%"
            print(f"{slug:10s} {r['station']:7s} {r['role']:6s} {r['night_n']:8d} "
                  f"{pa:>11s} {aa:>9s}  {r['verdict']}")

    if args.check and stale:
        print(f"\nauto_screen.json differs from the code at: {', '.join(stale)}",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
