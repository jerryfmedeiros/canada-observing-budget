"""
headline_corrected.py — the paper's counts, with the bias correction applied.

Every count the paper reports is corrected (nearer-satellite gap, measured shape); the raw
counts are the upper bound shown alongside. This script gives:

  * clear-dark hours a year, observable nights, the attrition to good nights, and the
    conversion rate, with block-bootstrap intervals (10-night blocks, the gap drawn from
    its interval in each replicate, as in ranking_intervals.py)
  * low-jet nights (good nights that also pass the jet-stream test) under each gap
    specification, and the city-to-city differences the text quotes
  * the best months, good nights by year and by month, the counts at bars of 2, 3, 4 and
    6 h, the moonless share, July and August counts, and clear-dark hours per night by
    month and season
  * results/headline_corrected.csv (read by falchi.py and skyglow.py), and the attrition,
    map and calendar figures (figures/*_corrected.png)

Each city's counts are first checked against its budget file, data/derived/<slug>/observable.txt.

Usage: python3 headline_corrected.py
"""
from __future__ import annotations

import hashlib
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from budgets import read_budget
    from gaps import SPECS, gap, gap_ci, moonless_gap
    from nights import OBSERVABLE_H, apply_correction, calendar, corrected_frac, per_night
    from sites import SITES
except ImportError as e:
    sys.exit(f"Run from the repo root: {e}")

SPEC = "satellite"
B, BLOCK, SEED = 2000, 10, 42
HOURS = OBSERVABLE_H   # nights.py; 4 h unless BUDGET_HOURS is set
BARS = (2, 3, 4, 6)     # the sensitivity sweep of the bar itself
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
SEASONS = {"win": (12, 1, 2), "spr": (3, 4, 5), "sum": (6, 7, 8), "aut": (9, 10, 11)}
PAIRS = [("calgary", "winnipeg"), ("winnipeg", "toronto"), ("calgary", "toronto"),
         ("edmonton", "vancouver"), ("halifax", "vancouver"), ("edmonton", "montreal"),
         ("edmonton", "halifax"), ("edmonton", "ottawa"), ("halifax", "toronto"),
         ("winnipeg", "edmonton"), ("calgary", "edmonton"), ("calgary", "vancouver"),
         ("calgary", "ottawa"), ("calgary", "montreal"), ("edmonton", "toronto"),
         ("vancouver", "toronto"), ("edmonton", "halifax"), ("winnipeg", "vancouver"),
         ("winnipeg", "ottawa"), ("winnipeg", "montreal")]
OUT_DIR = "figures"
# (slug, display name, lat, lon) for the map, south to north.
MAP_SITES = [
    ("toronto",   "Toronto",   43.68,  -79.63),
    ("halifax",   "Halifax",   44.88,  -63.51),
    ("ottawa",    "Ottawa",    45.32,  -75.67),
    ("montreal",  "Montréal",  45.47,  -73.74),
    ("vancouver", "Vancouver", 49.19, -123.18),
    ("winnipeg",  "Winnipeg",  49.92,  -97.24),
    ("calgary",   "Calgary",   51.11, -114.02),
    ("edmonton",  "Edmonton",  53.31, -113.58),
]


def block_index(rng, n: int, block: int) -> np.ndarray:
    n_blocks = -(-n // block)
    starts = rng.integers(0, n, size=(B, n_blocks))
    offsets = np.arange(block)
    return (starts[:, :, None] + offsets[None, None, :]).reshape(B, -1)[:, :n] % n


def ci(reps):
    return float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))


def gap_draws(slug: str, spec: str = SPEC) -> np.ndarray:
    """Per-replicate gaps (fraction), seeded per city exactly as ranking_intervals.py does."""
    lo, hi = gap_ci(spec, slug)
    grng = np.random.default_rng(
        int(hashlib.sha256(f"{SEED}|{spec}|{slug}".encode()).hexdigest()[:8], 16))
    return grng.normal(gap(spec, slug), (hi - lo) / (2 * 1.96), B) / 100.0


def main():
    rng = np.random.default_rng(SEED)
    ix = block_index(rng, len(calendar()), BLOCK)

    R = {}
    lowjet_reps, good_reps = {}, {}
    lowjet_spec = {sp: {} for sp in SPECS}   # low-jet point and replicates under every gap
    for slug, site in SITES.items():
        rec = per_night(site)
        ny = rec.dates[-1].year - rec.dates[0].year + 1
        g = gap(SPEC, slug)
        months = np.array([d.month for d in rec.dates])
        years = np.array([d.year for d in rec.dates])

        raw_cd = rec.frac * rec.dark_h
        cf = corrected_frac(rec, g)
        cd = cf * rec.dark_h
        obs_raw = (raw_cd >= HOURS) & rec.valid
        obs = (cd >= HOURS) & rec.valid

        # Check against the budget file, which observable_nights.py writes at the same bar.
        pub = read_budget(slug)
        for label, mine, theirs in (("raw obs", obs_raw.sum() / ny, pub.obs),
                                    ("corrected obs", obs.sum() / ny, pub.obs_corr),
                                    ("corrected good", (obs & rec.quality).sum() / ny,
                                     pub.good_corr)):
            if abs(mine - theirs) > 1:
                print(f"WARNING {slug}: {label} {mine:.0f} but data/derived says {theirs} -- "
                      f"regenerate data/derived/{slug}/observable.txt", file=sys.stderr)

        smoke = obs & rec.smoke_clean
        transp = smoke & rec.transparent
        good = obs & rec.quality
        lowjet = obs & rec.smoke_clean & rec.transparent & rec.low_jet   # low-jet good nights

        # Moonless: the same bar of moonless clear dark on the dark-stratum gap, among good nights.
        ml_cd = corrected_frac(rec, moonless_gap(SPEC, slug)) * rec.moonless_h
        good_ml = good & (ml_cd >= HOURS)

        # Replicates: gap drawn per replicate, nights resampled in shared blocks.
        gd = gap_draws(slug)
        cd_rep = apply_correction(rec.frac[None, :], 100.0 * gd[:, None]) * rec.dark_h[None, :]
        ok_rep = (cd_rep >= HOURS) & rec.valid[None, :]
        lowjet_grid = ok_rep & (rec.smoke_clean & rec.transparent & rec.low_jet)[None, :]
        good_grid = ok_rep & rec.quality[None, :]
        lowjet_reps[slug] = np.take_along_axis(lowjet_grid, ix, axis=1).sum(1) / ny
        dmask = rec.valid & rec.smoke_clean & rec.transparent & rec.low_jet
        for sp in SPECS:
            gs = gap_draws(slug, sp)
            cds = apply_correction(rec.frac[None, :], 100.0 * gs[:, None]) * rec.dark_h[None, :]
            pt = ((corrected_frac(rec, gap(sp, slug)) * rec.dark_h >= HOURS) & dmask).sum() / ny
            reps_sp = np.take_along_axis((cds >= HOURS) & dmask[None, :], ix, axis=1).sum(1) / ny
            lowjet_spec[sp][slug] = (pt, reps_sp)
        good_reps[slug] = np.take_along_axis(good_grid, ix, axis=1).sum(1) / ny

        by_month_good = {m: (good & (months == m)).sum() / ny for m in range(1, 13)}
        best = sorted(by_month_good, key=lambda m: -by_month_good[m])[:2]
        cal_raw = {m: float(np.mean(raw_cd[(months == m) & rec.valid])) for m in range(1, 13)}
        cal_cor = {m: float(np.mean(cd[(months == m) & rec.valid])) for m in range(1, 13)}
        season_cor = {s: float(np.mean(cd[np.isin(months, ms) & rec.valid]))
                      for s, ms in SEASONS.items()}

        R[slug] = dict(
            name=site.name, gap=g,
            clr_h=cd[rec.valid].sum() / ny, clr_h_raw=raw_cd[rec.valid].sum() / ny,
            obs=obs.sum() / ny, obs_raw=obs_raw.sum() / ny,
            smoke=smoke.sum() / ny, transp=transp.sum() / ny,
            good=good.sum() / ny, good_raw=(obs_raw & rec.quality).sum() / ny,
            lowjet=lowjet.sum() / ny, lowjet_raw=(obs_raw & rec.smoke_clean & rec.transparent & rec.low_jet).sum() / ny,
            good_ml=good_ml.sum() / ny,
            bars={h: ((cd >= h) & rec.valid & rec.quality).sum() / ny for h in BARS},
            bars_raw={h: ((raw_cd >= h) & rec.valid & rec.quality).sum() / ny for h in BARS},
            best=best, by_month_good=by_month_good,
            by_year={y: int((good & (years == y)).sum()) for y in range(2020, 2026)},
            jul_obs=(obs & (months == 7)).sum() / ny, aug_obs=(obs & (months == 8)).sum() / ny,
            cal_raw=cal_raw, cal_cor=cal_cor, season=season_cor,
        )
        print(f"  loaded {slug}", file=sys.stderr, flush=True)

    order = sorted(R, key=lambda s: -R[s]["good"])
    print(f"Corrected budget (gap spec: {SPEC}, measured ceilometer shape)\n")
    print(f"{'city':10s} {'gap':>5s} {'clr-h':>6s} {'(raw)':>6s} {'obs':>5s} {'(raw)':>5s} "
          f"{'-smoke':>6s} {'-transp':>7s} {'good':>5s} {'good CI':>10s} {'conv':>5s} "
          f"{'-jet':>5s} {'lowjet':>6s} {'lowjet CI':>10s} {'(raw)':>5s} best (good)")
    for s in order:
        r = R[s]
        glo, ghi = ci(good_reps[s])
        dlo, dhi = ci(lowjet_reps[s])
        print(f"{s:10s} {r['gap']:5.1f} {r['clr_h']:6.0f} {r['clr_h_raw']:6.0f} "
              f"{r['obs']:5.0f} {r['obs_raw']:5.0f} "
              f"{r['obs'] - r['smoke']:6.1f} {r['smoke'] - r['transp']:7.1f} "
              f"{r['good']:5.0f} [{glo:3.0f},{ghi:3.0f}]  {100 * r['good'] / r['obs']:4.0f}% "
              f"{r['good'] - r['lowjet']:5.1f} {r['lowjet']:6.0f} [{dlo:3.0f},{dhi:3.0f}]  "
              f"{r['lowjet_raw']:4.0f} {MONTHS[r['best'][0] - 1]}, {MONTHS[r['best'][1] - 1]}")

    print("\nLow-jet ranking: " + " > ".join(sorted(R, key=lambda s: -R[s]["lowjet"])))
    print("\nPairwise corrected differences (shared night blocks):")
    for a, b in PAIRS:
        for label, reps in (("good", good_reps), ("lowjet", lowjet_reps)):
            d = reps[a] - reps[b]
            lo, hi = ci(d)
            v = "resolved" if lo * hi > 0 else "unresolved"
            print(f"  {a:9s} - {b:9s} {label:6s} {d.mean():+5.1f} [{lo:+5.1f},{hi:+5.1f}] {v}")

    print("\nLow-jet nights under every gap specification, and the top pairs:")
    for sp in SPECS:
        d = lowjet_spec[sp]
        o = sorted(d, key=lambda s: -d[s][0])
        print(f"  {sp:10s} " + "  ".join(f"{s[:3]} {d[s][0]:.0f}" for s in o))
        for a, b in (("winnipeg", "calgary"), ("winnipeg", "toronto"), ("calgary", "toronto")):
            diff = d[a][1] - d[b][1]
            lo, hi = ci(diff)
            print(f"      {a[:3]}-{b[:3]} {diff.mean():+5.1f} [{lo:+5.1f},{hi:+5.1f}] "
                  f"{'resolved' if lo * hi > 0 else 'unresolved'}")

    print(f"\nGood nights at other bars (corrected), moonless share at {HOURS:g} h, "
          "July/August observable:")
    for s in order:
        r = R[s]
        print(f"  {s:10s} " + "  ".join(f"{h}h {r['bars'][h]:5.1f}" for h in BARS) +
              f"  moonless {r['good_ml']:5.1f} ({100 * r['good_ml'] / r['good']:3.0f}%)"
              f"  Jul obs {r['jul_obs']:4.1f}  Aug obs {r['aug_obs']:4.1f}"
              f"  Jul good {r['by_month_good'][7]:3.1f}  Aug good {r['by_month_good'][8]:3.1f}")

    print("\nGood nights by year (corrected):")
    for s in order:
        by = R[s]["by_year"]
        print(f"  {s:10s} " + " ".join(f"{y}:{by[y]:3d}" for y in by)
              + f"   min {min(by.values())} max {max(by.values())}")

    print("\nClear-dark hours per night by season (corrected):")
    for s in order:
        print(f"  {s:10s} " + " ".join(f"{k} {v:4.2f}" for k, v in R[s]["season"].items()))

    print("\nGood nights by month (corrected):")
    for s in order:
        print(f"  {s:10s} " + " ".join(f"{R[s]['by_month_good'][m]:4.1f}" for m in range(1, 13)))

    print("\nClear-dark hours per night by month (corrected), with the peak month(s):")
    for s in order:
        c = R[s]["cal_cor"]
        top = max(c[m] for m in range(1, 13) if not np.isnan(c[m]))
        peaks = [m for m in range(1, 13) if not np.isnan(c[m]) and c[m] >= top - 0.05]
        print(f"  {s:10s} " + " ".join(f"{c[m]:4.2f}" for m in range(1, 13))
              + "   peak " + "/".join(f"{m:d}" for m in peaks))

    with open(os.path.join("results", "headline_corrected.csv"), "w") as f:
        f.write("site,obs,good,lowjet,obs_raw,good_raw,lowjet_raw\n")
        for s in order:
            r = R[s]
            f.write(f"{s},{r['obs']:.2f},{r['good']:.2f},{r['lowjet']:.2f},"
                    f"{r['obs_raw']:.2f},{r['good_raw']:.2f},{r['lowjet_raw']:.2f}\n")

    figures(R)


def figures(R):
    os.makedirs(OUT_DIR, exist_ok=True)
    # Attrition cascade, corrected.
    order = sorted(R, key=lambda s: R[s]["good"])
    y = np.arange(len(order))
    good = np.array([R[s]["good"] for s in order])                   # good nights
    lj = np.array([R[s]["lowjet"] for s in order])                      # of which low-jet
    tr = np.array([R[s]["smoke"] - R[s]["transp"] for s in order])
    sm = np.array([R[s]["obs"] - R[s]["smoke"] for s in order])
    # Good nights are one segment; the low-jet nights, a subset of them, are drawn as a thin
    # bar inside it so that they read as part of the good nights, not a second kind. Print is
    # black and white: lightness steps of one hue, and smoke (the thin outer slice) is hatched.
    from figstyle import RAMP_DARK, RAMP_LIGHT, RAMP_MID, small_multiples
    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    kw = dict(edgecolor="white", linewidth=0.8, zorder=3)
    ax.barh(y, good, color=RAMP_MID, label="good night", **kw)
    ax.barh(y, lj, height=0.32, color=RAMP_DARK, zorder=4, label="of which low-jet")
    ax.barh(y, tr, left=good, color=RAMP_LIGHT, label="lost to transparency", **kw)
    ax.barh(y, sm, left=good + tr, facecolor="white", edgecolor="#333333", linewidth=0.5,
            hatch="//////", label="lost to smoke", zorder=3)
    for i, s in enumerate(order):
        ax.text(R[s]["obs"] + 1.5, i, f"{R[s]['good']:.0f} of {R[s]['obs']:.0f}  "
                f"({100 * R[s]['good'] / R[s]['obs']:.0f}%),  low-jet {R[s]['lowjet']:.0f}",
                va="center", fontsize=7, color="#111111")
    ax.set_yticks(y)
    ax.set_yticklabels([R[s]["name"] for s in order], fontsize=8)
    ax.set_xlabel("Nights per year", fontsize=8)
    ax.set_xlim(0, max(R[s]["obs"] for s in order) * 1.38)
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="x", lw=0.4, color="#e6e6e6", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4,
              frameon=False, handlelength=1.6, columnspacing=1.4)
    fig.tight_layout(pad=0.5)
    fig.savefig(os.path.join(OUT_DIR, "fig_attrition_corrected.png"), dpi=300,
                bbox_inches="tight")
    plt.close(fig)

    # Monthly calendar, corrected: one panel per city (eight lines on one axis cannot be
    # told apart in print). A month with no astronomical darkness at all (June on the
    # Prairies) has no valid nights; it contributes zero clear-dark hours.
    x = np.arange(1, 13)
    cal = {s: np.nan_to_num([R[s]["cal_cor"][m] for m in x]) for s in R}
    # The axis is set from the data (next half hour above the highest month), so no peak
    # is clipped when the counts change.
    cal_top = np.ceil(2 * 1.04 * max(v.max() for v in cal.values())) / 2
    small_multiples(cal, x, [1, 4, 7, 10], ["Jan", "Apr", "Jul", "Oct"], "",
                    "Clear-dark hours per night", "fig_cal_corrected", ymax=cal_top,
                    out_dir=OUT_DIR)

    # Map, corrected.
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError:
        print("    (cartopy not installed — skipping corrected map)")
        return
    from matplotlib.colors import Normalize
    proj = ccrs.LambertConformal(central_longitude=-96, central_latitude=49,
                                 standard_parallels=(49, 77))
    pc = ccrs.PlateCarree()
    fig = plt.figure(figsize=(5.5, 4.2))
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.91, bottom=0.14)
    ax.set_extent([-135, -52, 41, 61], crs=pc)
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f5f5f0", zorder=0)
    ax.add_feature(cfeature.STATES.with_scale("50m"), edgecolor="#cccccc", lw=0.4, zorder=1)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor="#888888", lw=0.7, zorder=2)
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#d0d0d0", zorder=3)
    ax.add_feature(cfeature.LAKES.with_scale("50m"), facecolor="#d0d0d0", zorder=4)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor="#888888", lw=0.5, zorder=5)
    ax.gridlines(lw=0.25, color="#bbbbbb", linestyle="--",
                 xlocs=range(-130, -50, 10), ylocs=range(42, 62, 4))
    # Conversion is a magnitude: one hue, light to dark (the light end is trimmed so the
    # palest marker still separates from the land fill). Marker AREA is proportional to
    # good nights, with no offset, and a size key carries the scale.
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "conv", plt.get_cmap("Blues")(np.linspace(0.3, 1.0, 256)))
    # Colour and size scales are set from the data, so they stay meaningful when the
    # counts change: conversion spans the observed range rounded out to 5%, and the size
    # key shows three round counts across the observed range of good nights.
    convs = [100 * R[s]["good"] / R[s]["obs"] for s in R]
    norm = Normalize(vmin=5 * np.floor(min(convs) / 5), vmax=5 * np.ceil(max(convs) / 5))
    goods = [R[s]["good"] for s in R]
    key_vals = [int(v) for v in np.linspace(10 * np.floor(min(goods) / 10),
                                            10 * np.ceil(max(goods) / 10), 3)]
    AREA_PER_NIGHT = 9.0   # points^2 per good night a year
    label_cfg = {"Vancouver": (-2.5, +1.2, "right", "bottom"),
                 "Calgary": (+1.3, -1.6, "left", "top"),
                 "Edmonton": (+1.4, +0.9, "left", "bottom"),
                 "Winnipeg": (+1.9, +0.4, "left", "center"),
                 "Toronto": (+1.5, -1.4, "left", "top"),
                 "Ottawa": (-1.8, +0.6, "right", "bottom"),
                 "Montréal": (+0.8, +1.8, "left", "bottom"),
                 "Halifax": (+2.3, 0.0, "left", "center")}
    for s, name, lat, lon in MAP_SITES:
        r = R[s]
        conv = 100 * r["good"] / r["obs"]
        ax.scatter(lon, lat, s=AREA_PER_NIGHT * r["good"], c=[cmap(norm(conv))],
                   edgecolor="#333333", linewidth=0.6, transform=pc, zorder=6)
        dlon, dlat, ha, va = label_cfg[name]
        ax.text(lon + dlon, lat + dlat, f"{name}\n{r['good']:.0f} good/yr", transform=pc,
                fontsize=6, ha=ha, va=va, zorder=7, linespacing=1.2)
    smp = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    smp.set_array([])
    cb = fig.colorbar(smp, ax=ax, fraction=0.030, pad=0.02, shrink=0.55)   # match the map's height
    cb.set_label("conversion (good / observable, %)", fontsize=6)
    cb.ax.tick_params(labelsize=6)
    keys = [ax.scatter([], [], s=AREA_PER_NIGHT * v, facecolor="none", edgecolor="#333333",
                       linewidth=0.6) for v in key_vals]
    ax.legend(keys, [str(v) for v in key_vals], title="good nights / yr\n", loc="upper center",
              bbox_to_anchor=(0.5, -0.03), ncol=3, fontsize=6, title_fontsize=6,
              frameon=False, columnspacing=3.5, handletextpad=2.2, borderpad=1.0)
    fig.savefig(os.path.join(OUT_DIR, "fig_map_corrected.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
