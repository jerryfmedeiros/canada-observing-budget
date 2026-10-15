"""
observable_nights.py — one city's night budget, raw and bias-corrected.

Writes data/derived/<slug>/observable.txt, the per-city budget released with the paper:

  - dark and moonless-dark hours per night, from the Sun and Moon positions (ephem);
  - per year and per month, the clear-dark hours, the observable nights (at least
    OBSERVABLE_H expected hours of clear astronomical darkness), the smoke-clean ones, the
    good nights (also transparent), and the low-jet nights among them, each raw and after
    the bias correction (nearer-satellite gap, measured shape; see nights.py and gaps.py);
  - the diurnal cycle of usable sky, the smoke and obscuration rates on clear dark
    reports, and the night-day usable-sky difference for each station.

The cloud usable fraction is the only empirical input: the share of a night's
astronomical-dark METAR reports at SKC or FEW (okta <= 2). A night with fewer than two
such reports takes its month's mean. Expected clear-dark hours are that fraction times the
night's dark hours. A city with an automated partner gets a second, raw-only block for it.

Usage:
  python3 observable_nights.py --site calgary > data/derived/calgary/observable.txt
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
import os
import sys
from collections import defaultdict
from zoneinfo import ZoneInfo

import ephem
import numpy as np

try:
    from fetch_metar import fetch_metar_csv
    from climatology import calculate_sun_alt, get_season
except ImportError:
    sys.exit("Error: could not import fetch_metar.py / climatology.py.")

# Site geometry (shared by a site's stations; they are close enough to share one sky
# budget). Defaults are Calgary; --site rebinds these globals at the top of main() so
# the per-night ephem helpers (which read them at call time) use the right location.
LAT, LON, ELEVATION_M = 51.05, -114.07, 1043
LOCAL_TZ = ZoneInfo("America/Edmonton")

USABLE_OKTA = 2          # SKC+FEW
MIN_OBS_PER_NIGHT = 2    # need >=2 dark obs to estimate a night's usable fraction
# The observable-night bar: a night counts if it delivers at least this many expected clear
# astronomical-dark hours. Every budget and check reads it from here (nights.OBSERVABLE_H).
# The paper uses 4 h; BUDGET_HOURS=2 reproduces the earlier 2 h results in results/bar2h/.
OBSERVABLE_H = float(os.environ.get("BUDGET_HOURS", "4"))
# A good night is observable, smoke-clean and transparent. The jet-stream test (median
# 200 hPa wind <= 30 m/s, and 10 m wind <= 8 m/s) is reported as a subset, the low-jet
# nights. BUDGET_JET=1 makes it part of the good-night test, which reproduces the earlier
# three-test results in results/jet_primary/.
JET_TEST = os.environ.get("BUDGET_JET", "0") == "1"
GAP_SPEC = "satellite"   # the nearer-satellite gap (gaps.py)
SEASONS = ["WINTER", "SPRING", "SUMMER", "AUTUMN"]
PM25_SMOKE = 35.0        # ug/m3 — a clear night above this is smoke-degraded
PWV_TRANSPARENT = 15.0   # mm — median dark-hour PWV at or below this = transparent
WIND200_JET = 30.0  # m/s — the jet-stream test: median 200 hPa wind at or below this
WIND10_JET = 8.0    # m/s — and median 10 m wind at or below this


def round_hour_utc(ts: dt.datetime) -> dt.datetime:
    """Round a timestamp to the nearest UTC hour (to match NAPS hour-ending keys)."""
    ts = ts.astimezone(dt.timezone.utc)
    floor = ts.replace(minute=0, second=0, microsecond=0)
    return floor + dt.timedelta(hours=1) if ts.minute >= 30 else floor


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion; returns (lo, hi) as fractions."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _boot_dist(arr: np.ndarray, B: int, rng: np.random.Generator) -> np.ndarray:
    """Bootstrap distribution of the mean (resampling the unit = element)."""
    n = len(arr)
    idx = rng.integers(0, n, size=(B, n))
    return arr[idx].mean(axis=1)


def bootstrap_mean_ci(values, B: int = 2000, seed: int = 0):
    """Point mean + percentile 95% CI from a night/day block bootstrap."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    bd = _boot_dist(arr, B, rng)
    return (float(arr.mean()), float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5)))


# --------------------------------------------------------------------------- #
# Deterministic astronomy (sun/moon geometry)
# --------------------------------------------------------------------------- #
def _observer(date_utc: dt.datetime, horizon: str | None = None) -> ephem.Observer:
    o = ephem.Observer()
    o.lat, o.lon = str(LAT), str(LON)
    o.elevation = ELEVATION_M
    if horizon is not None:
        o.horizon = horizon
    o.date = ephem.Date(date_utc)
    return o


def moon_alt_deg(ts_utc: dt.datetime) -> float:
    return math.degrees(ephem.Moon(_observer(ts_utc)).alt)


def dark_window(night_date: dt.date, threshold_deg: float):
    """(set_dt, rise_dt) UTC when the sun is below threshold_deg for this night, or None.

    The night belonging to `night_date` runs from local noon that date to local noon
    the next day. None means the sun never crosses the threshold (e.g. -18 deg in
    high-summer at 51 deg N -> no astronomical darkness)."""
    local_noon = dt.datetime.combine(night_date, dt.time(12, 0)).replace(tzinfo=LOCAL_TZ)
    o = _observer(local_noon.astimezone(dt.timezone.utc), horizon=str(threshold_deg))
    sun = ephem.Sun()
    try:
        set_t = o.next_setting(sun, use_center=True)
        rise_t = o.next_rising(sun, use_center=True)
    except (ephem.AlwaysUpError, ephem.NeverUpError):
        return None
    set_dt = set_t.datetime().replace(tzinfo=dt.timezone.utc)
    rise_dt = rise_t.datetime().replace(tzinfo=dt.timezone.utc)
    if rise_dt <= set_dt:
        return None
    return (set_dt, rise_dt)


def moonless_hours(window, step_min: int = 5) -> float:
    """Hours within `window` (a dark interval) during which the Moon is below the horizon."""
    if window is None:
        return 0.0
    start, end = window
    step = dt.timedelta(minutes=step_min)
    t, total = start, 0.0
    while t < end:
        if moon_alt_deg(t) < 0.0:
            total += step_min / 60.0
        t += step
    return total


def astro_budget(night_date: dt.date) -> dict:
    """Deterministic dark-hour and moonless-dark-hour budgets for both thresholds."""
    out = {}
    for tag, thr in (("18", -18.0), ("12", -12.0)):
        win = dark_window(night_date, thr)
        dark_h = (win[1] - win[0]).total_seconds() / 3600.0 if win else 0.0
        out[f"dark{tag}"] = dark_h
        out[f"moonless{tag}"] = moonless_hours(win) if win else 0.0
    return out


def night_key(ts_utc: dt.datetime) -> dt.date:
    """Map a UTC obs to the night it belongs to (local-noon to local-noon window)."""
    loc = ts_utc.astimezone(LOCAL_TZ)
    return (loc - dt.timedelta(hours=12)).date()


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #
def daterange(y_start: int, y_end: int):
    d = dt.date(y_start, 1, 1)
    end = dt.date(y_end, 12, 31)
    while d <= end:
        yield d
        d += dt.timedelta(days=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", nargs=2, type=int, default=[2020, 2025])
    ap.add_argument("--stations", nargs="+", default=None,
                    help="ICAO IDs (default: the site's human+auto)")
    ap.add_argument("--site", default="calgary", help="site slug from sites.py")
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap resamples")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pm25", action=argparse.BooleanOptionalAction, default=True,
                    help="load NAPS PM2.5 to flag smoke-degraded clear nights")
    ap.add_argument("--era5", action=argparse.BooleanOptionalAction, default=True,
                    help="load ERA5 PWV and winds for the transparency and jet-stream tests")
    ap.add_argument("--okta-thresh", type=int, default=2,
                    help="max okta counted as 'usable/clear' (default 2 = SKC+FEW; use 0 for SKC/CLR only)")
    args = ap.parse_args()

    # Rebind the site-geometry globals from the registry; the ephem helpers and
    # night_key read them at call time, so this must happen before any computation.
    from sites import get_site
    site = get_site(args.site)
    global LAT, LON, ELEVATION_M, LOCAL_TZ, USABLE_OKTA
    LAT, LON, ELEVATION_M = site.lat, site.lon, site.elevation_m
    LOCAL_TZ = ZoneInfo(site.iana_tz)
    USABLE_OKTA = args.okta_thresh
    if not args.stations:
        args.stations = list(site.stations)

    y_start, y_end = args.years
    print(f"Observable-night climatology {y_start}-{y_end}  ({site.name}, local tz {LOCAL_TZ.key})")
    print(f"Usable = okta<={USABLE_OKTA}; darkness -18deg & -12deg; moonless = Moon below horizon")
    print(f"Observable = at least {OBSERVABLE_H:g} h of expected clear astronomical darkness")
    print("=" * 92)

    # Optional PM2.5 smoke index (NAPS). Independent of cloud; flags milky clear skies.
    pm25_index: dict[dt.datetime, float] = {}
    if args.pm25 and site.naps_city:
        try:
            from fetch_pm25 import build_pm25_index
            print("Loading NAPS PM2.5 (smoke proxy) ...")
            pm25_index = build_pm25_index((y_start, y_end), site.naps_city, site.naps_std_offset_h)
            print(f"  {len(pm25_index)} {site.name} PM2.5 station-hours loaded")
        except Exception as e:
            print(f"  PM2.5 unavailable ({e}); continuing without smoke flags")
    elif args.pm25:
        print(f"  No NAPS city configured for {site.name}; continuing without smoke flags")

    # Optional ERA5 transparency (PWV) and jet-stream (200 hPa and 10 m wind) indices.
    pwv_index: dict[dt.datetime, float] = {}
    wind_index: dict[dt.datetime, float] = {}
    wind10_index: dict[dt.datetime, float] = {}
    if args.era5:
        try:
            from fetch_era5 import (build_pwv_index, build_wind_index,
                                    build_wind10_index, cache_dir_for)
            print("Loading ERA5 PWV + 200 hPa wind + 10 m wind ...")
            e_area, e_cache = site.era5_area(), cache_dir_for(site.slug)
            pwv_index = build_pwv_index((y_start, y_end), site.lat, site.lon, e_area, e_cache)
            wind_index = build_wind_index((y_start, y_end), site.lat, site.lon, e_area, e_cache)
            wind10_index = build_wind10_index((y_start, y_end), site.lat, site.lon, e_area, e_cache)
            print(f"  {len(pwv_index)} PWV, {len(wind_index)} 200hPa-wind, "
                  f"{len(wind10_index)} 10m-wind hours loaded")
        except Exception as e:
            print(f"  ERA5 unavailable ({e}); continuing without transparency/jet-stream flags")

    # The correction applies to the human station only; the automated partner is raw.
    from gaps import gap
    from nights import apply_correction
    gap_pp = gap(GAP_SPEC, site.slug)

    # 1) Deterministic astronomy budgets (shared by all stations) ------------- #
    print("Computing deterministic dark/moonless budgets per night ...")
    budgets: dict[dt.date, dict] = {nd: astro_budget(nd) for nd in daterange(y_start, y_end)}

    # Dark-time budget report (astronomy only) ------------------------------- #
    by_month = defaultdict(lambda: defaultdict(list))
    for nd, b in budgets.items():
        for k in ("dark18", "dark12", "moonless18", "moonless12"):
            by_month[nd.month][k].append(b[k])
    print("\n[DARK-TIME BUDGET] mean hours/night by month (deterministic)")
    print(f"{'Mon':<4} | {'dark-18':>8} | {'moonless-18':>11} | {'dark-12':>8} | {'moonless-12':>11}")
    print("-" * 56)
    for m in range(1, 13):
        d = by_month[m]
        print(f"{m:<4} | {np.mean(d['dark18']):8.2f} | {np.mean(d['moonless18']):11.2f} | "
              f"{np.mean(d['dark12']):8.2f} | {np.mean(d['moonless12']):11.2f}")
    tot_dark18 = sum(b["dark18"] for b in budgets.values()) / (y_end - y_start + 1)
    tot_moonless18 = sum(b["moonless18"] for b in budgets.values()) / (y_end - y_start + 1)
    print(f"\nAnnual astronomical-dark hours (sky maximum): {tot_dark18:7.0f} h/yr "
          f"(moonless {tot_moonless18:6.0f} h/yr)")

    # 2) Per-station empirical cloud sampling -------------------------------- #
    # night_usable[station][night_date]['18'|'12'] = list[bool usable]
    # day_usable[station][date] = list[bool usable]   (sun >= 6 deg)
    for station in args.stations:
        print("\n" + "=" * 92)
        print(f"STATION {station}")
        night_usable = defaultdict(lambda: {"18": [], "12": []})
        day_usable = defaultdict(list)
        # Transparency: among clear (usable) astronomical-dark obs, count obscurations.
        # cd_obsc[bucket]['n'|'smoke'|'haze'|'any'] where bucket = year or season.
        cd_obsc = defaultdict(lambda: defaultdict(int))
        cd_pm = defaultdict(lambda: defaultdict(int))   # PM2.5 on clear dark obs
        night_smoke = defaultdict(lambda: {"fu": False, "pm": False})  # per-night smoke flags
        night_pwv = defaultdict(list)    # ERA5 PWV (mm) over each night's dark hours
        night_wind = defaultdict(list)   # ERA5 200 hPa wind (m/s) over each night's dark hours
        night_wind10 = defaultdict(list) # ERA5 10 m wind (m/s) over each night's dark hours
        diurnal = defaultdict(lambda: [0, 0])  # local clock hour -> [total obs, usable obs]

        for year in range(y_start, y_end + 1):
            try:
                obs_list = fetch_metar_csv(station, dt.date(year, 1, 1), dt.date(year, 12, 31))
            except Exception as e:
                print(f"  Error fetching {year}: {e}")
                continue
            for o in obs_list:
                sun_alt = calculate_sun_alt(o.timestamp, LAT, LON, ELEVATION_M)
                usable = o.coverage_okta <= USABLE_OKTA
                lh = o.timestamp.astimezone(LOCAL_TZ).hour  # diurnal cycle (all obs)
                diurnal[lh][0] += 1
                diurnal[lh][1] += int(usable)
                if sun_alt <= -12.0:
                    nk = night_key(o.timestamp)
                    night_usable[nk]["12"].append(usable)
                    if sun_alt <= -18.0:
                        night_usable[nk]["18"].append(usable)
                        if pwv_index or wind_index or wind10_index:  # sky-state-independent
                            hr = round_hour_utc(o.timestamp)
                            pv, wv, w10 = pwv_index.get(hr), wind_index.get(hr), wind10_index.get(hr)
                            if pv is not None:
                                night_pwv[nk].append(pv)
                            if wv is not None:
                                night_wind[nk].append(wv)
                            if w10 is not None:
                                night_wind10[nk].append(w10)
                        if usable:  # clear sky but maybe smoky/hazy
                            pm = pm25_index.get(round_hour_utc(o.timestamp))
                            for bucket in (nk.year, get_season(nk.month)):
                                c = cd_obsc[bucket]
                                c["n"] += 1
                                if o.obscuration == "smoke":
                                    c["smoke"] += 1
                                elif o.obscuration == "haze":
                                    c["haze"] += 1
                                if o.obscuration is not None:
                                    c["any"] += 1
                                if pm is not None:
                                    p = cd_pm[bucket]
                                    p["n"] += 1
                                    if pm >= PM25_SMOKE:
                                        p["smoke"] += 1
                            # per-night smoke flags (FU from human obs, or high PM2.5)
                            if o.obscuration == "smoke":
                                night_smoke[nk]["fu"] = True
                            if pm is not None and pm >= PM25_SMOKE:
                                night_smoke[nk]["pm"] = True
                elif sun_alt >= 6.0:
                    day_usable[o.timestamp.astimezone(LOCAL_TZ).date()].append(usable)

        # Per-night observable estimates. The cloud usable-fraction is empirical;
        # the dark/moonless budget is deterministic. Annual totals SUM the
        # deterministic budget over ALL nights, imputing the cloud fraction by
        # monthly mean where a night lacks >=MIN_OBS dark obs. This correctly gives
        # near-zero-darkness summer nights their tiny contribution instead of
        # over-weighting long, well-sampled winter nights (a mean x 365.25 would).
        night_frac, day_frac = [], []          # bias arrays (night-level)
        covered: dict[dt.date, tuple] = {}     # nk -> (frac18, frac12) for sampled nights
        month_f18, month_f12 = defaultdict(list), defaultdict(list)

        for nk, bythr in night_usable.items():
            if len(bythr["12"]) >= MIN_OBS_PER_NIGHT:
                night_frac.append(float(np.mean(bythr["12"])))  # bias uses sun<=-12 "night"
            if nk in budgets and len(bythr["18"]) >= MIN_OBS_PER_NIGHT:
                f18 = float(np.mean(bythr["18"]))
                f12 = float(np.mean(bythr["12"])) if bythr["12"] else f18
                covered[nk] = (f18, f12)
                month_f18[nk.month].append(f18)
                month_f12[nk.month].append(f12)
        for lst in day_usable.values():
            if len(lst) >= MIN_OBS_PER_NIGHT:
                day_frac.append(float(np.mean(lst)))
        mm18 = {m: float(np.mean(v)) for m, v in month_f18.items()}
        mm12 = {m: float(np.mean(v)) for m, v in month_f12.items()}

        # Full-year accumulation with monthly imputation for un-sampled nights.
        # obs = observable; clean = observable and smoke-clean (no clear dark report with FU
        # or PM2.5 >= threshold); good = clean and transparent; lowjet = good nights that
        # also pass the jet-stream test. The *_c counts use the bias-corrected fraction.
        corrected = station == site.human
        keys = ("obs", "clean", "good", "lowjet", "obs_c", "good_c")
        per_year = defaultdict(lambda: {"cd": 0.0, "cm": 0.0, "cd_c": 0.0, "n": 0, "imp": 0,
                                        **{k: 0 for k in keys}})
        per_month = defaultdict(lambda: {"cd": 0.0, "cm": 0.0, "cd_c": 0.0, "n": 0,
                                         **{k: 0 for k in keys}})
        per_season = defaultdict(list)  # covered-night records for empirical probabilities
        n_years = y_end - y_start + 1
        for nk, b in budgets.items():
            if not (y_start <= nk.year <= y_end):
                continue
            if nk in covered:
                f18, f12, imp = (*covered[nk], 0)
            elif nk.month in mm18:
                f18, f12, imp = mm18[nk.month], mm12[nk.month], 1
            else:
                continue
            cd, cm = f18 * b["dark18"], f18 * b["moonless18"]
            cd_c = float(apply_correction(f18, gap_pp)) * b["dark18"] if corrected else 0.0
            smoky = night_smoke[nk]["fu"] or night_smoke[nk]["pm"] if nk in night_smoke else False
            transparent = nk in night_pwv and float(np.median(night_pwv[nk])) <= PWV_TRANSPARENT
            # The jet-stream test: calm aloft and, where ERA5 has it, at 10 m.
            calm_aloft = nk in night_wind and float(np.median(night_wind[nk])) <= WIND200_JET
            calm_surface = (nk not in night_wind10) or \
                float(np.median(night_wind10[nk])) <= WIND10_JET
            low_jet = calm_aloft and calm_surface
            quality = not smoky and transparent and (low_jet or not JET_TEST)
            obs, obs_c = cd >= OBSERVABLE_H, cd_c >= OBSERVABLE_H
            night = {"obs": obs, "clean": obs and not smoky, "good": obs and quality,
                     "lowjet": obs and quality and low_jet,
                     "obs_c": obs_c, "good_c": obs_c and quality}
            py, pmn = per_year[nk.year], per_month[nk.month]
            for acc in (py, pmn):
                acc["cd"] += cd; acc["cm"] += cm; acc["cd_c"] += cd_c; acc["n"] += 1
                for k in keys:
                    acc[k] += int(night[k])
            py["imp"] += imp
            if not imp:
                per_season[get_season(nk.month)].append((cd, cm, obs))

        # --- Report: observable budget per year (full-year sums) ------------- #
        smoke_note = "smoke-clean = no FU/PM2.5>=35" if pm25_index else "smoke-clean = no METAR FU"
        has_era5 = bool(pwv_index and wind_index)
        cols = ["obs", "clean"] + (["good", "lowjet"] if has_era5 else [])
        if corrected:
            cols += ["obs_c"] + (["good_c"] if has_era5 else [])
        names = {"obs": "observable", "clean": "smoke-clean", "good": "good", "lowjet": "low-jet",
                 "obs_c": "obs corr", "good_c": "good corr"}
        print(f"\n[OBSERVABLE BUDGET] {station} — full-year totals, astronomical dark (-18), "
              f"bar {OBSERVABLE_H:g} h")
        print(f"  ({smoke_note}; corr = bias-corrected, {GAP_SPEC} gap {gap_pp:.1f} pp)"
              if corrected else f"  ({smoke_note}; raw only)")
        hcorr = f" | {'clr-dark h corr':>15}" if corrected else ""
        print(f"{'Year':<5} | {'clear-dark h':>12} | {'clr-moonless h':>14} | "
              + " | ".join(f"{names[c]:>11}" for c in cols) + f"{hcorr} | {'imput':>6}")
        print("-" * (44 + 14 * len(cols) + (18 if corrected else 0)))
        sums = defaultdict(list)
        for year in range(y_start, y_end + 1):
            py = per_year.get(year)
            if not py or py["n"] == 0:
                continue
            for k in ("cd", "cm", "cd_c", *cols):
                sums[k].append(py[k])
            hc = f" | {py['cd_c']:15.0f}" if corrected else ""
            print(f"{year:<5} | {py['cd']:12.0f} | {py['cm']:14.0f} | "
                  + " | ".join(f"{py[c]:11d}" for c in cols) + f"{hc} | {py['imp']/py['n']:5.0%}")
        if sums:
            m = {k: float(np.mean(v)) for k, v in sums.items()}
            print(f"{'mean':<5} | {m['cd']:12.0f} | {m['cm']:14.0f} | "
                  + " | ".join(f"{m[c]:11.0f}" for c in cols)
                  + (f" | {m['cd_c']:15.0f}" if corrected else "")
                  + f" | (smoke -{m['obs'] - m['clean']:.0f}/yr)")
            if has_era5:
                print(f"  Attrition (raw): observable {m['obs']:.0f} -> smoke-clean {m['clean']:.0f} "
                      f"-> transparent (good) {m['good']:.0f} nights/yr, of which low-jet "
                      f"{m['lowjet']:.0f} (PWV<={PWV_TRANSPARENT:.0f}mm; low-jet: 200hPa wind"
                      f"<={WIND200_JET:.0f}m/s, 10m wind<={WIND10_JET:.0f}m/s)")

        # --- Report: seasonal (per covered/sampled night; empirical CIs) ---- #
        print(f"\n[OBSERVABLE BY SEASON] {station} — per sampled night (-18 dark)")
        print(f"{'Season':<8} | {'clear-dark h/nt':>15} | {'moonless h/nt':>13} | "
              f"{'P(observable) 95% CI':>20} | {'N':>5}")
        print("-" * 73)
        for s in SEASONS:
            recs = per_season[s]
            n = len(recs)
            if n == 0:
                continue
            cd = np.mean([r[0] for r in recs])
            cm = np.mean([r[1] for r in recs])
            k = sum(r[2] for r in recs)
            lo, hi = wilson_ci(k, n)
            print(f"{s:<8} | {cd:15.2f} | {cm:13.2f} | "
                  f"{k/n:5.1%} [{lo:.0%},{hi:.0%}] | {n:5d}")

        # --- Report: month-by-month observing calendar ---------------------- #
        mnames = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        mcols = ["obs"] + (["good"] if has_era5 else [])
        if corrected:
            mcols += ["obs_c"] + (["good_c"] if has_era5 else [])
        print(f"\n[OBSERVING CALENDAR] {station} — per month (astronomical dark), nights per year")
        print(f"{'Mon':<4} | {'clr-dark h/nt':>13} | {'moonless h/nt':>13} | "
              + " | ".join(f"{names[c]:>11}" for c in mcols)
              + (f" | {'clr-dark h/nt corr':>18}" if corrected else ""))
        print("-" * (36 + 14 * len(mcols) + (21 if corrected else 0)))
        for m in range(1, 13):
            pmn = per_month.get(m)
            if not pmn or pmn["n"] == 0:
                continue
            print(f"{mnames[m]:<4} | {pmn['cd']/pmn['n']:13.2f} | {pmn['cm']/pmn['n']:13.2f} | "
                  + " | ".join(f"{pmn[c]/n_years:11.1f}" for c in mcols)
                  + (f" | {pmn['cd_c']/pmn['n']:18.2f}" if corrected else ""))

        # --- Report: diurnal cycle (usable fraction by local hour) ---------- #
        print(f"\n[DIURNAL] {station} — usable-sky (okta<=2) fraction by local clock hour")
        print(f"{'Hour':<5} | {'usable':>7} | {'N':>7}    (all obs, {LOCAL_TZ.key})")
        print("-" * 40)
        for h in range(24):
            tot, us = diurnal[h]
            if tot == 0:
                continue
            print(f"{h:02d}:00 | {us/tot:6.1%} | {tot:7d}")

        # --- Report: transparency (smoke/haze on clear dark nights) --------- #
        # Of clear-sky astronomical-dark observations, how many were smoke- or
        # haze-degraded? METAR coverage says "usable" but the sky may be milky.
        print(f"\n[TRANSPARENCY] {station} — obscuration among CLEAR dark obs (okta<=2, sun<=-18)")
        print(f"{'Year':<6} | {'clear-dark obs':>14} | {'smoke':>8} | {'haze':>8} | {'any obsc':>9}")
        print("-" * 56)
        for year in range(y_start, y_end + 1):
            c = cd_obsc.get(year)
            if not c or c["n"] == 0:
                continue
            n = c["n"]
            print(f"{year:<6} | {n:14d} | {c['smoke']/n:7.1%} | {c['haze']/n:7.1%} | {c['any']/n:8.1%}")
        print(f"{'season':<6} |")
        for s in SEASONS:
            c = cd_obsc.get(s)
            if not c or c["n"] == 0:
                continue
            n = c["n"]
            print(f"{s:<6} | {n:14d} | {c['smoke']/n:7.1%} | {c['haze']/n:7.1%} | {c['any']/n:8.1%}")

        # --- Report: PM2.5 smoke on clear dark nights (independent of METAR) -- #
        if cd_pm:
            print(f"\n[PM2.5 SMOKE] {station} — PM2.5>={PM25_SMOKE:.0f} ug/m3 among CLEAR dark obs "
                  f"(matched hours; independent of METAR FU)")
            print(f"{'bucket':<8} | {'matched obs':>11} | {'smoke (PM2.5)':>13}")
            print("-" * 40)
            for k in list(range(y_start, y_end + 1)) + SEASONS:
                p = cd_pm.get(k)
                if not p or p["n"] == 0:
                    continue
                print(f"{str(k):<8} | {p['n']:11d} | {p['smoke']/p['n']:12.1%}")

        # --- Report: night-level bias (usable fraction day vs night) -------- #
        nm, nlo, nhi = bootstrap_mean_ci(night_frac, B=args.boot, seed=args.seed)
        dm, dlo, dhi = bootstrap_mean_ci(day_frac, B=args.boot, seed=args.seed + 1)
        print(f"\n[BIAS night-level] {station} usable fraction (okta<=2), unit = night/day")
        print(f"  Night (sun<=-12): {nm:6.1%}  95% CI [{nlo:.1%}, {nhi:.1%}]  ({len(night_frac)} nights)")
        print(f"  Day   (sun>= 6 ): {dm:6.1%}  95% CI [{dlo:.1%}, {dhi:.1%}]  ({len(day_frac)} days)")
        # bootstrap the night-day delta directly (independent resampling)
        rng = np.random.default_rng(args.seed + 100)
        nd_arr = np.asarray(night_frac, float)
        dd_arr = np.asarray(day_frac, float)
        if nd_arr.size and dd_arr.size:
            delta = _boot_dist(nd_arr, args.boot, rng) - _boot_dist(dd_arr, args.boot, rng)
            print(f"  Night-Day delta: {(nm-dm)*100:+.1f} pp  95% CI "
                  f"[{np.percentile(delta,2.5)*100:+.1f}, {np.percentile(delta,97.5)*100:+.1f}] pp")
            # stash for cross-station delta-of-deltas
            _STASH[station] = (nd_arr, dd_arr)

    # 3) Human minus automated night-day difference of differences ------------ #
    # Cities without an automated partner have no such comparison here; their gap comes
    # from the satellite comparison of Fernandes Medeiros (2026) (gaps.py).
    human, auto = site.human, site.auto
    if auto and human in _STASH and auto in _STASH:
        rng = np.random.default_rng(args.seed + 999)
        B = args.boot
        nh, dh = _STASH[human]
        na, da = _STASH[auto]
        d_human = _boot_dist(nh, B, rng) - _boot_dist(dh, B, rng)
        d_auto = _boot_dist(na, B, rng) - _boot_dist(da, B, rng)
        dod = d_human - d_auto
        point = (nh.mean() - dh.mean()) - (na.mean() - da.mean())
        print("\n" + "=" * 92)
        print(f"[BIAS] human({human}) vs automated({auto}) night-day usable delta-of-deltas")
        print(f"  {human} night-day: {(nh.mean()-dh.mean())*100:+.1f} pp;  "
              f"{auto} night-day: {(na.mean()-da.mean())*100:+.1f} pp")
        print(f"  Delta-of-deltas: {point*100:+.1f} pp  95% CI "
              f"[{np.percentile(dod,2.5)*100:+.1f}, {np.percentile(dod,97.5)*100:+.1f}] pp")
        excl = np.percentile(dod, 2.5) * np.percentile(dod, 97.5) > 0
        print(f"  -> CI excludes zero: {excl} (human nighttime clearing is observer-specific)")
    else:
        why = ("no automated partner configured" if not auto
               else f"{auto} data unavailable this run")
        print("\n" + "=" * 92)
        print(f"[BIAS] {site.name}: {why}; the gap comes from the satellite comparison "
              f"(gaps.py).")


_STASH: dict = {}

if __name__ == "__main__":
    main()
