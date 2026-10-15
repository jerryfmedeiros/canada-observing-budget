"""
nights.py — per-night records shared by the budget scripts.

Rebuilds, for one site, the quantities observable_nights.py works from: the usable fraction
of each night's astronomical-dark reports (monthly-mean imputed where a night has too few),
the dark and moonless-dark hours, and the smoke, transparency and jet-stream flags. Returned
as arrays over the full 2020-2025 calendar so a block bootstrap can slice night blocks
directly. Also holds the bias correction (apply_correction) that every corrected count uses.

The tests treat a missing input differently. A night with no PWV fails the transparency test
and one with no 200 hPa wind fails the jet-stream test; a night with no 10 m wind or no
PM2.5 passes those conditions. No night lacks 10 m wind, but a partial re-fetch of one
variable would bias the counts in one direction or the other.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from collections import defaultdict
from typing import NamedTuple
from zoneinfo import ZoneInfo

import numpy as np

import observable_nights as on
from fetch_era5 import build_pwv_index, build_wind_index, build_wind10_index, cache_dir_for
from fetch_metar import fetch_metar_csv
from fetch_pm25 import build_pm25_index
from gaps import gap as _gap
from climatology import calculate_sun_alt

YEARS = (2020, 2025)

# The observable-night bar and the jet-test switch are set in observable_nights.py
# (BUDGET_HOURS, BUDGET_JET) so that the budget file and every check use the same values.
OBSERVABLE_H = on.OBSERVABLE_H
JET_TEST = on.JET_TEST


class NightRecords(NamedTuple):
    """Per-night arrays over the full study calendar, one element per night."""

    dates: list          # night keys (local-noon-to-local-noon buckets), len N
    frac: np.ndarray     # usable fraction of the night's dark reports (imputed where needed)
    dark_h: np.ndarray   # deterministic astronomical-dark hours
    moonless_h: np.ndarray  # deterministic moonless astronomical-dark hours
    quality: np.ndarray  # bool: smoke-clean AND transparent (AND the jet test if JET_TEST)
    imputed: np.ndarray  # bool: frac came from the monthly mean, not this night's reports
    valid: np.ndarray    # bool: frac is usable at all (False only if the month is empty)
    smoke_clean: np.ndarray   # bool: no FU and no PM2.5 >= threshold on a clear dark ob
    transparent: np.ndarray   # bool: median dark-hour PWV <= PWV_TRANSPARENT
    low_jet: np.ndarray       # bool: the jet-stream test, 200 hPa and (where present) 10 m wind
    pwv_median: np.ndarray    # median dark-hour PWV in mm, NaN where the night has none
    pm25_covered: np.ndarray  # bool: at least one clear dark report had a PM2.5 value behind it


def calendar(years: tuple[int, int] = YEARS) -> list[dt.date]:
    """Every night key in the study window, in order."""
    d, end = dt.date(years[0], 1, 1), dt.date(years[1], 12, 31)
    out = []
    while d <= end:
        out.append(d)
        d += dt.timedelta(days=1)
    return out


def per_night(site, years: tuple[int, int] = YEARS,
              usable_okta: int | None = None, use_pm25: bool = True) -> NightRecords:
    """Build the per-night record arrays for one site from the cached METAR/ERA5/PM2.5 data.

    `usable_okta` overrides the SKC+FEW definition of a usable report (default
    observable_nights.USABLE_OKTA = 2); pass 4 to count SCT as usable as well.
    `use_pm25=False` flags smoke from the METAR FU code alone, the one smoke detector logged
    the same way at every station in every year (observable_nights.py --no-pm25).
    """
    okta_max = on.USABLE_OKTA if usable_okta is None else usable_okta
    on.LAT, on.LON, on.ELEVATION_M = site.lat, site.lon, site.elevation_m
    on.LOCAL_TZ = ZoneInfo(site.iana_tz)

    cache = cache_dir_for(site.slug)
    pwv = build_pwv_index(years, site.lat, site.lon, site.era5_area(), cache)
    w200 = build_wind_index(years, site.lat, site.lon, site.era5_area(), cache)
    w10 = build_wind10_index(years, site.lat, site.lon, site.era5_area(), cache)
    pm = (build_pm25_index(years, site.naps_city, site.naps_std_offset_h)
          if site.naps_city and use_pm25 else {})

    usable: dict = defaultdict(list)
    npwv, nw200, nw10 = defaultdict(list), defaultdict(list), defaultdict(list)
    smoky: dict = defaultdict(bool)
    # Whether the smoke screen actually had a surface measurement behind it on this night.
    # NAPS coverage is uneven across cities and years, so a night can pass "smoke-clean"
    # on the absence of a PM2.5 reading rather than on a low one.
    pm_seen: dict = defaultdict(bool)

    for year in range(years[0], years[1] + 1):
        for o in fetch_metar_csv(site.human, dt.date(year, 1, 1), dt.date(year, 12, 31)):
            if calculate_sun_alt(o.timestamp, site.lat, site.lon, site.elevation_m) > -18.0:
                continue
            nk = on.night_key(o.timestamp)
            u = o.coverage_okta <= okta_max
            usable[nk].append(u)
            # ERA5 is sky-state independent, so it is collected from every dark report,
            # including nights too sparse to give their own usable fraction.
            hr = on.round_hour_utc(o.timestamp)
            for index, store in ((pwv, npwv), (w200, nw200), (w10, nw10)):
                v = index.get(hr)
                if v is not None:
                    store[nk].append(v)
            # Smoke is only meaningful under an otherwise-clear sky, matching observable_nights.
            if u:
                if pm.get(hr) is not None:
                    pm_seen[nk] = True
                if o.obscuration == "smoke" or ((pm.get(hr) or 0) >= on.PM25_SMOKE):
                    smoky[nk] = True

    # Nights with enough dark reports carry their own fraction; the rest take the monthly mean.
    covered = {nk: float(np.mean(v)) for nk, v in usable.items()
               if len(v) >= on.MIN_OBS_PER_NIGHT}
    by_month: dict = defaultdict(list)
    for nk, f in covered.items():
        by_month[nk.month].append(f)
    month_mean = {m: float(np.mean(v)) for m, v in by_month.items()}

    dates = calendar(years)
    n = len(dates)
    frac = np.zeros(n)
    dark_h = np.zeros(n)
    moonless_h = np.zeros(n)
    quality = np.zeros(n, bool)
    imputed = np.zeros(n, bool)
    valid = np.zeros(n, bool)
    smoke_clean = np.zeros(n, bool)
    transparent_a = np.zeros(n, bool)
    low_jet_a = np.zeros(n, bool)
    pwv_median = np.full(n, np.nan)
    pm25_covered = np.zeros(n, bool)

    for i, nk in enumerate(dates):
        win = on.dark_window(nk, -18.0)
        dark_h[i] = (win[1] - win[0]).total_seconds() / 3600.0 if win else 0.0
        moonless_h[i] = on.moonless_hours(win) if win else 0.0
        if nk in covered:
            frac[i], valid[i] = covered[nk], True
        elif nk.month in month_mean:
            frac[i], valid[i], imputed[i] = month_mean[nk.month], True, True
        if nk in npwv:
            pwv_median[i] = float(np.median(npwv[nk]))
        transparent = nk in npwv and pwv_median[i] <= on.PWV_TRANSPARENT
        calm_aloft = nk in nw200 and float(np.median(nw200[nk])) <= on.WIND200_JET
        calm_surface = (nk not in nw10) or \
            float(np.median(nw10[nk])) <= on.WIND10_JET
        smoke_clean[i] = not smoky[nk]
        pm25_covered[i] = pm_seen[nk]
        transparent_a[i] = transparent
        low_jet_a[i] = calm_aloft and calm_surface
        quality[i] = smoke_clean[i] and transparent and (low_jet_a[i] or not JET_TEST)

    return NightRecords(dates, frac, dark_h, moonless_h, quality, imputed, valid,
                        smoke_clean, transparent_a, low_jet_a, pwv_median, pm25_covered)


# How the correction is distributed over nights. "ceilometer" is the measured shape
# (correction_shape.py): the artifact at each human-reported night fraction, from the four
# human/ceilometer pairs. It is near zero on nights reported fully clear and largest on
# partly cloudy nights. A city's gap sets the size: the measured profile corresponds to a
# city whose gap is the paired cities' mean ceilometer gap (SHAPE_REF_GAP), and is scaled
# by gap / SHAPE_REF_GAP elsewhere.
DEFAULT_SHAPE = "ceilometer"
_SHAPE_FILE = Path(__file__).parent / "data" / "derived" / "correction_shape.csv"
_PAIRED = ("calgary", "edmonton", "toronto", "montreal")
SHAPE_REF_GAP = float(np.mean([_gap("ceilometer", s) for s in _PAIRED]))


def _load_shape():
    """(human fraction, artifact pp) knots for fractions above zero, in ascending order."""
    xs, ys = [], []
    for line in _SHAPE_FILE.read_text().splitlines():
        if line.startswith("#") or line.startswith("bin,"):
            continue
        mh, art = line.split(",")[2:4]
        if float(mh) > 0.0:           # a night with no clear report has nothing to take
            xs.append(float(mh))
            ys.append(float(art))
    return np.array(xs), np.array(ys)


_SHAPE_X, _SHAPE_Y = _load_shape()


def _load_curves():
    """Per-bin (human night, ceilometer night) and (human day, ceilometer day) knots, all bins."""
    hn, an, hd, ad = [], [], [], []
    for line in _SHAPE_FILE.read_text().splitlines():
        if line.startswith("#") or line.startswith("bin,"):
            continue
        parts = line.split(",")
        if len(parts) < 7:
            return None
        hn.append(float(parts[2])); hd.append(float(parts[4]))
        an.append(float(parts[5])); ad.append(float(parts[6]))
    return np.array(hn), np.array(an), np.array(hd), np.array(ad)


_CURVES = _load_curves()


def day_equivalent_removal(frac):
    """Removal (fraction) that maps a night's human fraction to the human DAY fraction with
    the same mean ceilometer reading: f -> g_day^-1(g_night(f)), g = E[ceilometer | human].
    A sensitivity test of the scale question: the profile is a difference in the
    ceilometer's fraction, subtracted from the observer's. Inverting a regression curve
    exaggerates differences where it is flat, so this leans toward too large a correction."""
    hn, an, hd, ad = _CURVES
    c = np.interp(frac, hn, an)
    return np.clip(frac - np.interp(c, ad, hd), 0.0, None)


def artifact_pp(frac):
    """The measured artifact (pp) at a human-reported night usable fraction."""
    return np.interp(frac, _SHAPE_X, _SHAPE_Y)


def apply_correction(frac, gap_pp, shape: str = DEFAULT_SHAPE):
    """Usable fraction(s) after the bias correction, floored at zero.

    `frac` and `gap_pp` may be scalars or arrays that broadcast (the bootstraps pass a
    column of gap draws against a row of nights). `shape`:

      "ceilometer"  the measured shape (the paper's correction): gap / SHAPE_REF_GAP times
                    the artifact profile of correction_shape.py.
      "day_equivalent"  maps each night to the observer's daytime scale through the
                    ceilometer (see day_equivalent_removal), scaled the same way; a test.
      "uniform"     the same amount from every night. This also takes from nights reported
                    fully clear, so such a night needs 4 / (1 - gap) hours of darkness to
                    count; kept as a sensitivity.
    """
    frac = np.asarray(frac, dtype=float)
    gap_pp = np.asarray(gap_pp, dtype=float)
    if shape == "ceilometer":
        delta = (gap_pp / SHAPE_REF_GAP) * artifact_pp(frac) / 100.0
    elif shape == "uniform":
        delta = gap_pp / 100.0
    elif shape == "day_equivalent":
        delta = (gap_pp / SHAPE_REF_GAP) * day_equivalent_removal(frac)
    else:
        raise ValueError(f"unknown correction shape: {shape!r}")
    return np.maximum(frac - delta, 0.0)


def corrected_frac(rec: NightRecords, gap_pp: float, shape: str = DEFAULT_SHAPE) -> np.ndarray:
    """Each night's usable fraction after the bias correction, floored at zero (see
    apply_correction for `shape`)."""
    return apply_correction(rec.frac, gap_pp, shape)


def counts(rec: NightRecords, gap_pp: float = 0.0, hours: float = OBSERVABLE_H,
           moonless: bool = False, shape: str = DEFAULT_SHAPE) -> tuple[float, float]:
    """(observable, good) nights per year after the bias correction of size `gap_pp`.

    The correction is applied to each night's usable fraction and floored at zero, so a
    corrected night must still clear the `hours` bar on its reduced fraction -- night by
    night, not as a proportional rescale of the annual total. `shape` is passed through to
    corrected_frac.
    """
    window = rec.moonless_h if moonless else rec.dark_h
    cd = corrected_frac(rec, gap_pp, shape) * window
    obs = (cd >= hours) & rec.valid
    n_years = rec.dates[-1].year - rec.dates[0].year + 1
    return obs.sum() / n_years, (obs & rec.quality).sum() / n_years


def floor_rate(rec: NightRecords, gap_pp: float) -> float:
    """Fraction of nights with any clear dark sky whose corrected fraction hits the zero floor.

    The floor is where the correction stops being a subtraction and becomes a deletion, so
    this is the share of the marginal nights on which the correction is conservative by
    construction.
    """
    has_clear = rec.valid & (rec.frac > 0) & (rec.dark_h > 0)
    if not has_clear.any():
        return float("nan")
    return float(np.mean(apply_correction(rec.frac[has_clear], gap_pp) <= 0.0))
