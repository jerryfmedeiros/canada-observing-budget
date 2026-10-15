"""
faint_moon.py — is "moonless" too strict?

The moonless count takes only hours with the Moon below the horizon. This also credits
hours with the Moon up but at most 25% illuminated, and reports whether any count or city
position moves.

For every credited moon-up minute it evaluates the Krisciunas & Schaefer (1991) model at the
zenith. The result is given as a lunar contribution in nanolamberts and as the brightening
it causes against several baseline skies, because the same contribution is 0.4 mag against
a pristine zenith and 0.03 mag against an urban one.

--corrected applies the bias correction: the nearer-satellite gap to the observable test,
and the dark-sky (moonless) gap to the moonless and faint-Moon tests.

Usage: python3 faint_moon.py [--corrected]
"""
from __future__ import annotations

import datetime as dt
import math
import sys

import numpy as np

try:
    import ephem
    import observable_nights as on
    import argparse
    from gaps import gap, moonless_gap
    from nights import OBSERVABLE_H, corrected_frac, per_night
    from sites import SITES
except ImportError as e:
    sys.exit(f"Run from the repo root: {e}")

HOURS = OBSERVABLE_H   # nights.py; 4 h unless BUDGET_HOURS is set
MAX_ILLUM = 0.25       # credit moon-up hours no more than this fraction illuminated
STEP_MIN = 5           # same sampling step as observable_nights.moonless_hours
K_EXT = 0.20           # V-band extinction coefficient, typical lowland site
V_DARK = 21.6          # pristine dark-sky zenith brightness, mag/arcsec^2
# The added magnitude depends on the sky the Moon is added to: the same lunar contribution
# is large against a natural sky and small against a bright urban one. The eight city cores
# are far brighter than natural, so it is reported against several baselines rather than
# one assumed sky.
V_BASELINES = (21.6, 20.5, 19.5, 18.5)


def _nl(v_mag: float) -> float:
    """Sky brightness in nanoLamberts from mag/arcsec^2 (Krisciunas & Schaefer eq. 27)."""
    return 34.08 * math.exp(20.7233 - 0.92104 * v_mag)


def moon_brightness_nl(obs) -> float:
    """Moon's contribution to zenith sky brightness in nanoLamberts, K&S 1991.

    This is the baseline-independent quantity; how many magnitudes it adds depends on the
    sky it is added to (see V_BASELINES).
    """
    moon = ephem.Moon(obs)
    alt = math.degrees(moon.alt)
    if alt <= 0.0:
        return 0.0
    alpha = 180.0 - abs(math.degrees(float(moon.elong)))   # lunar phase angle
    i_star = 10 ** (-0.4 * (3.84 + 0.026 * abs(alpha) + 4e-9 * alpha ** 4))
    rho = 90.0 - alt                                        # Moon-to-zenith angle
    f_rho = 10 ** 5.36 * (1.06 + math.cos(math.radians(rho)) ** 2) \
        + 10 ** (6.15 - rho / 40.0)
    z_moon = math.radians(90.0 - alt)
    x_moon = (1 - 0.96 * math.sin(z_moon) ** 2) ** -0.5
    return f_rho * i_star * 10 ** (-0.4 * K_EXT * x_moon) * (1 - 10 ** (-0.4 * K_EXT))


def added_mag(b_moon: float, v_base: float) -> float:
    """Magnitudes the Moon adds to a sky of baseline brightness v_base."""
    b = _nl(v_base)
    return 2.5 * math.log10((b_moon + b) / b)


def faint_moon_hours(window, site):
    """(hours creditable, worst moon contribution in nL over the credited moon-up minutes)."""
    if window is None:
        return 0.0, 0.0
    start, end = window
    step = dt.timedelta(minutes=STEP_MIN)
    obs = ephem.Observer()
    obs.lat, obs.lon, obs.elevation = str(site.lat), str(site.lon), site.elevation_m
    t, total, worst = start, 0.0, 0.0
    while t < end:
        obs.date = t
        moon = ephem.Moon(obs)
        if math.degrees(moon.alt) < 0.0:
            total += STEP_MIN / 60.0
        elif moon.moon_phase <= MAX_ILLUM:
            total += STEP_MIN / 60.0
            worst = max(worst, moon_brightness_nl(obs))
        t += step
    return total, worst


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corrected", action="store_true")
    args = ap.parse_args()
    print(f"Crediting moon-up hours up to {MAX_ILLUM:.0%} illuminated "
          f"(Krisciunas & Schaefer 1991 zenith model, k={K_EXT}, V_dark={V_DARK})\n")
    print(f"{'city':10s} {'mnl good':>9s} {'faint good':>11s} {'d':>4s} | "
          f"{'mnl obs':>8s} {'faint obs':>9s} {'d':>4s} | {'worst moon nL':>13s}")

    strict_g, faint_g, worst_all = {}, {}, 0.0
    strict_o, faint_o = {}, {}
    for slug, site in SITES.items():
        rec = per_night(site)
        n_years = rec.dates[-1].year - rec.dates[0].year + 1
        on.LAT, on.LON, on.ELEVATION_M = site.lat, site.lon, site.elevation_m

        faint_h = np.zeros(len(rec.dates))
        worst = 0.0
        for i, nk in enumerate(rec.dates):
            win = on.dark_window(nk, -18.0)
            h, w = faint_moon_hours(win, site)
            faint_h[i] = h
            worst = max(worst, w)
        worst_all = max(worst_all, worst)

        g = gap("satellite", slug) if args.corrected else 0.0
        gm = moonless_gap("satellite", slug) if args.corrected else 0.0
        f_all, f_dark = corrected_frac(rec, g), corrected_frac(rec, gm)
        obs2 = ((f_all * rec.dark_h) >= HOURS) & rec.valid
        s_o = ((f_dark * rec.moonless_h) >= HOURS) & rec.valid
        f_o = ((f_dark * faint_h) >= HOURS) & rec.valid
        strict_o[slug] = s_o.sum() / n_years
        faint_o[slug] = f_o.sum() / n_years
        strict_g[slug] = (obs2 & rec.quality & s_o).sum() / n_years
        faint_g[slug] = (obs2 & rec.quality & f_o).sum() / n_years
        print(f"{slug:10s} {strict_g[slug]:9.1f} {faint_g[slug]:11.1f} "
              f"{faint_g[slug] - strict_g[slug]:+4.1f} | "
              f"{strict_o[slug]:8.1f} {faint_o[slug]:9.1f} "
              f"{faint_o[slug] - strict_o[slug]:+4.1f} | {worst:13.1f}", flush=True)

    def rank(d):
        return " > ".join(s[:3] for s in sorted(d, key=lambda s: -d[s]))

    dg = max(abs(faint_g[s] - strict_g[s]) for s in strict_g)
    do = max(abs(faint_o[s] - strict_o[s]) for s in strict_o)
    print(f"\nlargest change, moonless GOOD nights:     {dg:.2f} nights/yr")
    print(f"largest change, moonless OBSERVABLE nights: {do:.2f} nights/yr")
    print(f"\nWorst Moon contribution over all credited moon-up time: {worst_all:.1f} nL.")
    print("  added zenith sky, by what it is added to:")
    for v in V_BASELINES:
        print(f"    baseline V={v:.1f} mag/arcsec^2  ->  +{added_mag(worst_all, v):.2f} mag")
    print(f"  ranking, Moon strictly below horizon: {rank(strict_g)}")
    print(f"  ranking, faint-Moon credited        : {rank(faint_g)}")
    print(f"  ordering unchanged: {rank(strict_g) == rank(faint_g)}")


if __name__ == "__main__":
    main()
