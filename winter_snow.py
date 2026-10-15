"""
winter_snow.py — winter snow-cover fraction per city.

Tabulates the fraction of DJF days with ERA5-Land snow depth over 1 cm, which is the filter
threshold used elsewhere, to show that the ceilometer ground-truth cities are heavy-snow.

NOTE: Vancouver's own grid cell is water-masked, CYVR sitting on Sea Island, so fetch_snow.py
widens the request box and takes the nearest valid land cell 5.3 km away. An earlier cache
wrote all-null for Vancouver and this table read those nulls as bare ground and reported
0.0%. A city with no valid days is now called out rather than printed as a number.

Usage: python3 winter_snow.py
"""
import json
from pathlib import Path

CITIES = ["calgary", "edmonton", "winnipeg", "montreal", "ottawa", "toronto", "vancouver", "halifax"]
CACHE = Path(__file__).parent / "data" / "snow_cache"


def winter_snow_frac(slug):
    d = json.loads((CACHE / f"snow_{slug}.json").read_text())["daily"]
    djf = [v for t, v in zip(d["time"], d["snow_depth_max"]) if int(t[5:7]) in (12, 1, 2)]
    valid = [v for v in djf if v is not None]
    frac = 100.0 * sum(1 for v in valid if v > 0.01) / len(valid) if valid else float("nan")
    return frac, len(djf), len(valid)


def main():
    print("Winter (DJF) days with snow on ground (ERA5-Land depth > 1 cm), 2020-2025:")
    for slug in CITIES:
        frac, n, valid = winter_snow_frac(slug)
        if not valid:
            print(f"  {slug:10s}     --    (n={n}, NO VALID SNOW DATA -- refetch with "
                  f"fetch_snow.py; the request box is all water)")
        else:
            note = "" if valid == n else f", {n - valid} days missing"
            print(f"  {slug:10s} {frac:5.1f}%   (n={valid}{note})")


if __name__ == "__main__":
    main()
