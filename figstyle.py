"""
figstyle.py — shared figure conventions for print.

JRASC prints figures in black and white (colour survives only in the electronic copy), so
no figure may rely on hue. The conventions here:

  * eight cities never share one set of axes: small_multiples() gives each city its own
    panel, drawn in black over the other seven in light grey, so every line is identified
    by position and none by colour;
  * magnitudes use one hue running light to dark (RAMP), which reduces to a lightness scale
    in greyscale;
  * a category that must be told apart from a lightness step uses a 45-degree hatch.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

# West to east, the reading order of the panels.
CITY_ORDER = ["vancouver", "calgary", "edmonton", "winnipeg",
              "toronto", "ottawa", "montreal", "halifax"]
CITY_NAME = {"vancouver": "Vancouver", "calgary": "Calgary", "edmonton": "Edmonton",
             "winnipeg": "Winnipeg", "toronto": "Toronto", "ottawa": "Ottawa",
             "montreal": "Montréal", "halifax": "Halifax"}

INK = "#111111"        # the highlighted series and text
CONTEXT = "#c4c4c4"    # the other seven cities, behind
GRID = "#e6e6e6"

# One hue, light to dark; lightness steps stay distinct in greyscale.
RAMP_DARK, RAMP_MID, RAMP_LIGHT = "#08306b", "#4f8fc4", "#c6dbef"


def small_multiples(series: dict, x, xticks, xticklabels, xlabel: str, ylabel: str,
                    path_stem: str, ymax: float | None = None, out_dir: str = "figures",
                    marker: bool = True):
    """One panel per city (2 x 4, west to east), shared axes.

    `series` maps slug -> y values aligned with `x`. Each panel draws all eight in light grey
    and its own city in black on top.
    """
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.0), sharex=True, sharey=True)
    for ax, slug in zip(axes.ravel(), CITY_ORDER):
        for other in CITY_ORDER:
            if other != slug:
                ax.plot(x, series[other], color=CONTEXT, lw=0.8, zorder=1)
        ax.plot(x, series[slug], color=INK, lw=1.6, zorder=3,
                marker="o" if marker else None, markersize=2.6)
        ax.set_title(CITY_NAME[slug], fontsize=8, pad=3)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, fontsize=6.5)
        ax.tick_params(axis="y", labelsize=6.5)
        ax.grid(axis="y", lw=0.4, color=GRID, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0, 0].set_ylim(0, ymax)
    fig.supxlabel(xlabel, fontsize=8, y=0.01)
    fig.supylabel(ylabel, fontsize=8, x=0.01)
    fig.tight_layout(pad=0.6, w_pad=0.8, h_pad=0.9)
    import os
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{path_stem}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
