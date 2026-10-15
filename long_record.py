"""
long_record.py — is 2020-2025 a representative six years?

Recomputes the observable-night count back to 2001 and places the study window among all
rolling six-year windows. --good does the same for good nights, using ERA5 PWV back to 2001
(fetch_pwv_long.py) and smoke from the METAR FU code alone, since PM2.5 was not assembled
before 2020.

Year 2000 is excluded. The FEW cloud group entered the Canadian METAR code with the 2001
changeover, station by station rather than nationally: six of the eight stations report it
in under 0.2% of their 2000 observations against 14-19% afterwards. Because "usable" is
SKC+FEW, those stations could not record a usable sky in 2000, and their counts fall 32-67
nights below their 2001-2005 means. FEW_MIN_PCT detects this from the data and drops the
year at every station, so the cities stay comparable.

Read the result as a range rather than a trend; staffing and reporting practice also
changed after 2001.

--corrected applies each city's 2020-2025 nearer-satellite gap to every year. That assumes
the artifact was the same size back to 2001, which nothing measures.

Usage:
  python3 long_record.py --fetch-only   # populate the METAR cache, retrying when throttled
  python3 long_record.py [--good] [--corrected]
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from collections import defaultdict
from zoneinfo import ZoneInfo

import numpy as np

try:
    import observable_nights as on
    from fetch_metar import fetch_metar_csv
    from climatology import calculate_sun_alt
    from sites import SITES
    from nights import OBSERVABLE_H, apply_correction
    from gaps import gap
except ImportError as e:
    sys.exit(f"Run from the repo root: {e}")

START_YEAR = 2000
END_YEAR = 2025
WINDOW = 6                      # the study window's length, in years
STUDY = (2020, 2025)
HOURS = OBSERVABLE_H   # nights.py; 4 h unless BUDGET_HOURS is set
RETRY_WAIT_S = 120
MAX_TRIES = 20
MIN_DARK_REPORTS = 2000   # a normally-staffed year carries ~10k dark reports
FEW_MIN_PCT = 5.0         # below this share of FEW reports, the station-year predates FEW
PWV_FROM = 2001           # first year of the ERA5 PWV cache (fetch_pwv_long.py)


def fetch_year(station: str, year: int):
    """fetch_metar_csv with retries, because IEM sheds load under bulk requests."""
    for attempt in range(1, MAX_TRIES + 1):
        try:
            rows = list(fetch_metar_csv(station, dt.date(year, 1, 1), dt.date(year, 12, 31)))
            if rows:
                return rows
            print(f"    {station} {year}: empty response (try {attempt}/{MAX_TRIES})",
                  file=sys.stderr, flush=True)
        except Exception as e:                                  # network/throttle
            print(f"    {station} {year}: {type(e).__name__} (try {attempt}/{MAX_TRIES})",
                  file=sys.stderr, flush=True)
        if attempt < MAX_TRIES:
            time.sleep(RETRY_WAIT_S)
    print(f"    {station} {year}: GIVING UP", file=sys.stderr, flush=True)
    return []


def observable_by_year(site, years, gap_pp: float = 0.0, pwv: dict | None = None):
    """{year: observable nights that year}, plus {year: n dark reports} for coverage.

    With `pwv` (an ERA5 {utc_hour: mm} index) it also counts good nights: observable nights
    with no FU on a clear dark report and median dark-hour PWV <= the transparency bar.
    Smoke is taken from FU alone, the one detector logged the same way in every year."""
    tz = ZoneInfo(site.iana_tz)
    on.LAT, on.LON, on.ELEVATION_M = site.lat, site.lon, site.elevation_m
    on.LOCAL_TZ = tz

    usable = defaultdict(list)
    npwv, fu = defaultdict(list), defaultdict(bool)
    n_reports = defaultdict(int)
    few = defaultdict(int)          # reports coded FEW, per calendar year (all hours)
    all_rep = defaultdict(int)
    for year in years:
        for o in fetch_year(site.human, year):
            if o.coverage_okta is not None:
                all_rep[year] += 1
                few[year] += (o.coverage_okta == 2)
            if calculate_sun_alt(o.timestamp, site.lat, site.lon, site.elevation_m) > -18.0:
                continue
            nk = on.night_key(o.timestamp)
            u = o.coverage_okta <= on.USABLE_OKTA
            usable[nk].append(u)
            n_reports[nk.year] += 1
            if pwv is not None:
                v = pwv.get(on.round_hour_utc(o.timestamp))
                if v is not None:
                    npwv[nk].append(v)
                if u and o.obscuration == "smoke":
                    fu[nk] = True

    covered = {nk: float(np.mean(v)) for nk, v in usable.items()
               if len(v) >= on.MIN_OBS_PER_NIGHT}
    # Monthly means are formed per YEAR, so no year is imputed from another year's climate.
    by_ym = defaultdict(list)
    for nk, f in covered.items():
        by_ym[(nk.year, nk.month)].append(f)
    ym_mean = {k: float(np.mean(v)) for k, v in by_ym.items()}

    counts, good = defaultdict(int), defaultdict(int)
    for year in years:
        d, end = dt.date(year, 1, 1), dt.date(year, 12, 31)
        while d <= end:
            win = on.dark_window(d, -18.0)
            dh = (win[1] - win[0]).total_seconds() / 3600.0 if win else 0.0
            f = covered.get(d, ym_mean.get((d.year, d.month)))
            if f is not None and float(apply_correction(f, gap_pp)) * dh >= HOURS:
                counts[year] += 1
                if (pwv is not None and not fu[d] and d in npwv
                        and float(np.median(npwv[d])) <= on.PWV_TRANSPARENT):
                    good[year] += 1
            d += dt.timedelta(days=1)
    few_pct = {y: (100.0 * few[y] / all_rep[y] if all_rep[y] else 0.0) for y in years}
    return counts, n_reports, few_pct, good


def order_report(per_city, kept, dropped, start, label):
    """Rank the cities in every six-year window and count how often each pair keeps the
    order it has in the study window. The study window's offsets can be as large as the
    gaps between neighbouring cities, so the order, not only each count, is checked."""
    starts = [s for s in range(start, END_YEAR - WINDOW + 2)
              if not any(y in dropped for y in range(s, s + WINDOW))]
    win_mean = {slug: {s: np.mean([c[y] for y in range(s, s + WINDOW)]) for s in starts}
                for slug, c in per_city.items()}
    lt_mean = {slug: np.mean([c[y] for y in kept]) for slug, c in per_city.items()}
    st = STUDY[0]
    study_order = sorted(per_city, key=lambda k: -win_mean[k][st])
    lt_order = sorted(per_city, key=lambda k: -lt_mean[k])
    print(f"\n{label}: order of the cities, study window: "
          + " > ".join(f"{k[:3]} {win_mean[k][st]:.0f}" for k in study_order))
    print(f"{label}: order of the cities, {kept[0]}-{kept[-1]} mean: "
          + " > ".join(f"{k[:3]} {lt_mean[k]:.0f}" for k in lt_order))
    print(f"\nRank of each city in each of the {len(starts)} windows (1 = most nights):")
    print(f"{'city':10s} " + " ".join(f"{s % 100:02d}" for s in starts) + "   range  study  long-term")
    rk = {s: {k: i + 1 for i, k in enumerate(sorted(per_city, key=lambda k: -win_mean[k][s]))}
          for s in starts}
    for k in study_order:
        r = [rk[s][k] for s in starts]
        print(f"{k:10s} " + " ".join(f"{v:2d}" for v in r)
              + f"   {min(r)}-{max(r)}    {rk[st][k]:>3d}  {lt_order.index(k) + 1:>9d}")
    print(f"\nPairs (study-window order) and the share of the {len(starts)} windows that keep it:")
    keep_all, flip = [], []
    for i, a in enumerate(study_order):
        for b in study_order[i + 1:]:
            n = sum(win_mean[a][s] > win_mean[b][s] for s in starts)
            (keep_all if n == len(starts) else flip).append((a, b, n))
    print(f"  kept in every window: {len(keep_all)} of {len(keep_all) + len(flip)} pairs")
    for a, b, n in flip:
        print(f"  {a:10s} > {b:10s} in {n:2d} of {len(starts)} windows   "
              f"(study {win_mean[a][st] - win_mean[b][st]:+.0f}, long-term {lt_mean[a] - lt_mean[b]:+.0f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fetch-only", action="store_true",
                    help="populate the METAR cache and stop")
    ap.add_argument("--start", type=int, default=START_YEAR)
    ap.add_argument("--corrected", action="store_true")
    ap.add_argument("--good", action="store_true",
                    help="also rank the cities by good nights (ERA5 PWV + FU), 2001 on")
    args = ap.parse_args()
    years = list(range(args.start, END_YEAR + 1))

    if args.fetch_only:
        for slug, site in SITES.items():
            for year in years:
                fetch_year(site.human, year)
            print(f"  cached {slug} ({site.human}) {years[0]}-{years[-1]}", flush=True)
        return

    per_city, flagged, few_by_city, good_by_city = {}, set(), {}, {}
    for slug, site in SITES.items():
        pwv = None
        if args.good:
            from fetch_era5 import build_pwv_index, cache_dir_for
            pwv = build_pwv_index((max(args.start, PWV_FROM), END_YEAR), site.lat, site.lon,
                                  site.era5_area(), cache_dir_for(slug))
        counts, nrep, few_pct, good = observable_by_year(
            site, years, gap("satellite", slug) if args.corrected else 0.0, pwv)
        good_by_city[slug] = good
        few_by_city[slug] = few_pct
        thin = [y for y in years if nrep[y] < MIN_DARK_REPORTS]
        # A year that predates the FEW group cannot be compared under an SKC+FEW definition.
        nofew = [y for y in years if nrep[y] >= MIN_DARK_REPORTS and few_pct[y] < FEW_MIN_PCT]
        per_city[slug] = counts
        flagged |= set(thin) | set(nofew)
        note = ""
        if thin:
            note += f"  thin: {thin}"
        if nofew:
            note += f"  pre-FEW: {nofew} ({', '.join(f'{few_pct[y]:.1f}%' for y in nofew)})"
        print(f"  {slug} done{note}", file=sys.stderr, flush=True)

    dropped = sorted(flagged)
    kept = [y for y in years if y not in dropped]
    n_win = len(kept) - WINDOW + 1
    if dropped:
        print(f"\nEXCLUDED years (inhomogeneous at one or more stations): {dropped}")
    print(f"\nObservable nights/yr: {kept[0]}-{kept[-1]} ({len(kept)} yr) vs the "
          f"{STUDY[0]}-{STUDY[1]} study window ({n_win} rolling {WINDOW}-yr windows)\n")
    print(f"{'city':10s} {'long-term':>9s} {'study':>7s} {'diff':>6s} "
          f"{'study rank':>11s}  {'best..worst window':>20s}")
    diffs, ranks, nwins = {}, {}, {}
    for slug, counts in per_city.items():
        lt = np.mean([counts[y] for y in kept])
        study = np.mean([counts[y] for y in range(STUDY[0], STUDY[1] + 1)])
        wins = {}
        for s in range(args.start, END_YEAR - WINDOW + 2):
            ys = list(range(s, s + WINDOW))
            if any(y in dropped for y in ys):
                continue
            wins[s] = np.mean([counts[y] for y in ys])
        order = sorted(wins, key=lambda s: -wins[s])
        rank = order.index(STUDY[0]) + 1 if STUDY[0] in wins else None
        diffs[slug], ranks[slug], nwins[slug] = study - lt, rank, len(wins)
        print(f"{slug:10s} {lt:9.0f} {study:7.0f} {study - lt:+6.0f} "
              f"{str(rank) + '/' + str(len(wins)):>11s}  "
              f"{max(wins.values()):.0f}..{min(wins.values()):.0f}")

    mx = max(abs(v) for v in diffs.values())
    within6 = sum(1 for v in diffs.values() if abs(v) <= 6)
    print(f"\nEvery study-window mean sits within {mx:.0f} nights of its long-term mean "
          f"(within 6 at {within6} cities).")
    # A window's rank is against that city's own windows, so 1 is its best six years and
    # n its worst; the ranks are not comparable across cities.
    for slug, d in sorted(diffs.items(), key=lambda kv: -abs(kv[1])):
        r, n = ranks[slug], nwins[slug]
        if r is None:
            tag = "window not evaluable"
        elif r == 1:
            tag = f"BEST of its {n}"
        elif r == n:
            tag = f"WORST of its {n}"
        else:
            tag = f"rank {r} of {n}"
        print(f"  {slug:10s} {d:+5.0f} nights vs long-term   ({tag})")

    order_report(per_city, kept, dropped, args.start, "Observable nights")
    if args.good:
        print(f"\n{'=' * 78}\nGOOD nights/yr (observable, FU-clean, median dark PWV <= "
              f"{on.PWV_TRANSPARENT:g} mm), {kept[0]}-{kept[-1]}")
        for slug, g in good_by_city.items():
            lt = np.mean([g[y] for y in kept])
            st = np.mean([g[y] for y in range(STUDY[0], STUDY[1] + 1)])
            ws = [np.mean([g[y] for y in range(s0, s0 + WINDOW)])
                  for s0 in range(args.start, END_YEAR - WINDOW + 2)
                  if not any(y in dropped for y in range(s0, s0 + WINDOW))]
            rank = sorted(ws, reverse=True).index(st) + 1
            print(f"  {slug:10s} long-term {lt:5.1f}  study {st:5.1f}  diff {st - lt:+5.1f}  "
                  f"study rank {rank}/{len(ws)}  windows {min(ws):.0f}..{max(ws):.0f}")
        order_report(good_by_city, kept, dropped, args.start, "Good nights")

    # ---- why the pre-FEW years are dropped, in numbers -------------------------------
    # The paper quotes a FEW share and an observable-night deficit for the dropped years;
    # both are printed here.
    if dropped:
        base = [y for y in range(max(dropped) + 1, max(dropped) + 6) if y in years]
        print(f"\nFEW cloud-group share by station, dropped year(s) {dropped} vs "
              f"{base[0]}-{base[-1]}:")
        print(f"{'city':10s} {'FEW% dropped':>13s} {'FEW% after':>11s}   "
              f"{'obs dropped':>11s} {'obs ' + str(base[0]) + '-' + str(base[-1]):>13s} "
              f"{'deficit':>8s}")
        pre_lo, pre_hi, post_lo, post_hi, defs, n_pre = 100.0, 0.0, 100.0, 0.0, [], 0
        for slug, counts in per_city.items():
            fp = few_by_city[slug]
            pre = np.mean([fp[y] for y in dropped])
            post = np.mean([fp[y] for y in base])
            obs_pre = np.mean([counts[y] for y in dropped])
            obs_post = np.mean([counts[y] for y in base])
            deficit = obs_pre - obs_post
            print(f"{slug:10s} {pre:12.1f}% {post:10.1f}%   "
                  f"{obs_pre:11.0f} {obs_post:13.0f} {deficit:+8.0f}")
            if pre < FEW_MIN_PCT:          # only the stations that had not adopted FEW yet
                n_pre += 1
                pre_lo, pre_hi = min(pre_lo, pre), max(pre_hi, pre)
                post_lo, post_hi = min(post_lo, post), max(post_hi, post)
                defs.append(deficit)
        print(f"  {n_pre} of {len(per_city)} stations code FEW in under "
              f"{max(pre_hi, 0.0):.1f}% of their {dropped[0]} reports, against "
              f"{post_lo:.0f}-{post_hi:.0f}% over {base[0]}-{base[-1]};")
        print(f"  their observable counts fall {abs(max(defs)):.0f}-{abs(min(defs)):.0f} "
              f"nights below their {base[0]}-{base[-1]} means.")


if __name__ == "__main__":
    main()
