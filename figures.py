"""
figures.py — the smoke-trend and diurnal-cycle figures of the paper.

  figures/fig_smoke_trend.pdf/.png        smoke rate by year from three sources
                                          (reads results/smoke_trend_sources.csv, smoke_trend.py)
  figures/fig_diurnal_multisite.pdf/.png  usable-sky fraction by local clock hour, one panel
                                          per city (reads data/derived/<slug>/observable.txt)

The attrition, map and calendar figures use the bias-corrected counts and are drawn by
headline_corrected.py. Run from the repo root:
  python3 figures.py
"""

import csv
import os
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)


def save(fig, stem):
    for ext in ("pdf", "png"):
        path = os.path.join(OUT_DIR, f"{stem}.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  → {OUT_DIR}/{stem}.pdf/.png")


def make_smoke():
    """Smoke rate by year at the western cities from three sources: surface PM2.5, the METAR
    FU code, and MERRA-2 column AOD. Reads results/smoke_trend_sources.csv (smoke_trend.py).
    PM2.5 is drawn only through 2023, after which NAPS coverage thins (smoke_trend.py)."""
    print("Figure: smoke trend (three sources) …")

    path = Path("results") / "smoke_trend_sources.csv"
    if not path.exists():
        print("    (no results/smoke_trend_sources.csv — run smoke_trend.py)")
        return
    data = defaultdict(dict)
    for r in csv.DictReader(path.open()):
        if r["region"] == "west" and r["rate_pct"] not in ("", None):
            data[r["source"]][int(r["year"])] = float(r["rate_pct"])

    # Print is black and white: the three sources differ by line style and marker, and are
    # labelled on the plot rather than through a legend that would cover the 2021 and 2023
    # peaks.
    SERIES = [
        ("pm25", "surface PM2.5", "#111111", "-", "o"),
        ("aod",  "column AOD",    "#111111", "--", "s"),
        ("fu",   "METAR FU code", "#555555", ":", "^"),
    ]
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for yr in (2021, 2023):
        ax.axvspan(yr - 0.3, yr + 0.3, color="#e9e9e9", lw=0, zorder=0)
    ends = {}
    for key, label, color, ls, mk in SERIES:
        d = data.get(key, {})
        if key == "pm25":  # NAPS coverage thins after 2023 (no 2024 Calgary, single 2024 Edm)
            d = {y: v for y, v in d.items() if y <= 2023}
        yrs = sorted(d)
        ax.plot(yrs, [d[y] for y in yrs], color=color, ls=ls, marker=mk, markersize=4.5,
                lw=1.6, zorder=3, markerfacecolor="white" if key == "aod" else color)
        ends[key] = (yrs[-1], d[yrs[-1]], label)
    for key, (xe, ye, label) in ends.items():
        dy = {"pm25": 0.10, "aod": 0.12, "fu": -0.16}[key]
        ax.annotate(label, (xe, ye), xytext=(xe + (0.0 if key == "pm25" else 0.08), ye + dy),
                    fontsize=7, color="#111111", ha="center" if key == "pm25" else "left",
                    va="bottom" if dy > 0 else "top")
    ax.text(2021, 0.04, "fire\nseason", fontsize=6.5, color="#555555", ha="center", va="bottom")
    ax.text(2023, 0.04, "fire\nseason", fontsize=6.5, color="#555555", ha="center", va="bottom")
    ax.set_ylabel("Clear dark hours flagged as smoke (%)", fontsize=8)
    ax.set_ylim(0, 2.9)
    ax.set_xlim(2019.7, 2026.2)
    ax.set_xticks(list(range(2020, 2026)))
    ax.set_xticklabels([str(y) for y in range(2020, 2026)], fontsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", lw=0.4, color="#e6e6e6")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout(pad=0.5)
    save(fig, "fig_smoke_trend")
    plt.close(fig)


def parse_diurnal(slug):
    """Return dict hour(0-23)->usable_pct from the first [DIURNAL] block (human station)
    of data/derived/<slug>/observable.txt."""
    path = Path("data") / "derived" / slug / "observable.txt"
    if not path.exists():
        return {}
    m = re.search(r"\[DIURNAL\].*?\n.*?\n-+\n(.*?)(?:\n\s*\n|\Z)", path.read_text(), re.S)
    if not m:
        return {}
    out = {}
    for line in m.group(1).splitlines():
        parts = [q.strip() for q in line.split("|")]
        if len(parts) >= 2 and re.match(r"\d{2}:00$", parts[0]):
            try:
                out[int(parts[0][:2])] = float(parts[1].replace("%", "").strip())
            except ValueError:
                pass
    return out


def make_diurnal():
    """Usable-sky fraction by local clock hour, one panel per city, noon to noon so that the
    night is one continuous stretch in the middle of each panel."""
    print("Figure: diurnal \u2026")
    from figstyle import CITY_ORDER, small_multiples
    data = {slug: parse_diurnal(slug) for slug in CITY_ORDER}
    missing = [s for s, d in data.items() if len(d) < 24]
    if missing:
        print(f"    (no diurnal data for {', '.join(missing)} \u2014 run observable_nights.py)")
        return
    hours = list(range(12, 24)) + list(range(0, 12))          # noon -> noon
    x = np.arange(24)
    small_multiples({s: [data[s][h] for h in hours] for s in CITY_ORDER},
                    x, [0, 6, 12, 18], ["12", "18", "0", "6"], "Local clock hour",
                    "Usable-sky fraction, human station (%)", "fig_diurnal_multisite",
                    ymax=40, marker=False)


if __name__ == "__main__":
    make_smoke()     # results/smoke_trend_sources.csv
    make_diurnal()   # data/derived/<slug>/observable.txt
    print("Done.")
