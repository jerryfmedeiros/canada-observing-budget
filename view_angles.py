"""
view_angles.py — which geostationary satellite views each city more directly.

The nearer-satellite gap uses, at each city, the satellite with the smaller view zenith angle
(gaps.py). This prints the angle from GOES-East (75.2 W) and GOES-West (137.2 W) for every
city, with the gap each specification assigns.

Usage: python3 view_angles.py
"""
from __future__ import annotations

import sys

import numpy as np

try:
    from gaps import SPECS, gap
    from sites import SITES
except ImportError as e:
    sys.exit(f"Run from the repo root: {e}")

SUBLON = {"GOES-East": -75.2, "GOES-West": -137.2}
R_RATIO = 6378.137 / 42164.0   # Earth radius over geostationary orbit radius


def view_zenith(lat: float, lon: float, sublon: float) -> float:
    """View zenith angle (deg) of a geostationary satellite at `sublon`, seen from (lat, lon)."""
    la, dl = np.radians(lat), np.radians(lon - sublon)
    c = np.arccos(np.cos(la) * np.cos(dl))
    return float(np.degrees(np.arctan2(np.sin(c), np.cos(c) - R_RATIO)))


def main():
    print(f"{'city':10s} {'GOES-East':>10s} {'GOES-West':>10s} {'nearer':>7s}   "
          + "  ".join(f"{k:>10s}" for k in SPECS) + "   (gap, pp)")
    for slug, s in SITES.items():
        e = view_zenith(s.lat, s.lon, SUBLON["GOES-East"])
        w = view_zenith(s.lat, s.lon, SUBLON["GOES-West"])
        print(f"{slug:10s} {e:9.1f}° {w:9.1f}° {'West' if w < e else 'East':>7s}   "
              + "  ".join(f"{gap(k, slug):10.1f}" for k in SPECS))


if __name__ == "__main__":
    main()
