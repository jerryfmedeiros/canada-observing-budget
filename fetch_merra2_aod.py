"""
fetch_merra2_aod.py — MERRA-2 hourly column aerosol optical depth for the eight cities.

The smoke flag rests on surface PM2.5 (NAPS) and the METAR FU code. Surface concentration
can miss wildfire smoke carried above the boundary layer, and the satellite aerosol products
(GOES, VIIRS) are daytime retrievals. MERRA-2 is a reanalysis, so it has a column AOD at
every hour, including at night. This pulls the total 550 nm AOD (TOTEXTTAU) from the
M2T1NXAER hourly aerosol diagnostics, per city, to check the surface smoke flag and the
water-vapour transparency test.

Access is GES DISC OPeNDAP with a subset of one grid cell (each request returns that cell's
24 hourly values), authenticated with the Earthdata bearer token in ~/.edl_token (the GES DISC
application must be authorized in the Earthdata profile). The MERRA-2 stream number is 400,
except 401 for the September 2020 to September 2021 reprocessing; it is probed per month.

Cache: data/merra2_cache/<slug>.csv with columns date,hour,aod (24 rows a day). Resumable.

Usage: python3 fetch_merra2_aod.py [--years 2020 2025] [--workers 6] [--sites all|calgary ...]
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from sites import SITES

PROJECT_ROOT = Path(__file__).parent.resolve()
CACHE = PROJECT_ROOT / "data" / "merra2_cache"
BASE = "https://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/MERRA2/M2T1NXAER.5.12.4"
FILL = 1e14  # MERRA-2 undefined is ~1e15


def token() -> str:
    t = os.environ.get("EARTHDATA_TOKEN", "").strip()
    if t:
        return t
    p = Path.home() / ".edl_token"
    if p.exists() and p.read_text().strip():
        return p.read_text().strip()
    sys.exit("No Earthdata token in EARTHDATA_TOKEN or ~/.edl_token.")


TOK = None


def grid_index(lat: float, lon: float) -> tuple[int, int]:
    return round((lat + 90) / 0.5), round((lon + 180) / 0.625)


def url_for(stream: int, d: dt.date, li: int, lo: int) -> str:
    fn = f"MERRA2_{stream}.tavg1_2d_aer_Nx.{d:%Y%m%d}.nc4"
    return (f"{BASE}/{d:%Y}/{d:%m}/{fn}.ascii"
            f"?TOTEXTTAU%5B0:23%5D%5B{li}%5D%5B{lo}%5D")


def fetch_ascii(url: str, timeout: int = 60) -> str | None:
    # --fail turns throttled HTTP errors into a nonzero exit (else curl hands back an
    # error page that silently fails to parse); --retry then rides out transient 5xx/resets.
    r = subprocess.run(["curl", "-sS", "-g", "-L", "--fail", "--retry", "5",
                        "--retry-delay", "1", "--max-time", str(timeout),
                        "-H", f"Authorization: Bearer {TOK}", url],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return None
    return r.stdout


def parse_values(text: str) -> list[float] | None:
    """Pull the 24 hourly TOTEXTTAU values (time order) from the OPeNDAP ascii dump."""
    vals = []
    for line in text.splitlines():
        if line.startswith("TOTEXTTAU.TOTEXTTAU[") and "," in line:
            try:
                v = float(line.rsplit(",", 1)[1].strip())
            except ValueError:
                return None
            vals.append(v if v < FILL else float("nan"))
    return vals if len(vals) == 24 else None


def resolve_streams(years, li, lo):
    """One working stream per (year, month), probed on day 15 (401 window fallback)."""
    streams = {}
    for y in range(years[0], years[1] + 1):
        for m in range(1, 13):
            d = dt.date(y, m, 15)
            if d > dt.date.today():
                continue
            for s in (400, 401):
                txt = fetch_ascii(url_for(s, d, li, lo))
                if txt and parse_values(txt) is not None:
                    streams[(y, m)] = s
                    break
            else:
                print(f"  WARN no stream for {y}-{m:02d}", flush=True)
    return streams


def daterange(y0, y1):
    d, end = dt.date(y0, 1, 1), min(dt.date(y1, 12, 31), dt.date.today())
    while d <= end:
        yield d
        d += dt.timedelta(days=1)


def load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    with path.open() as f:
        next(f, None)
        for line in f:
            done.add(line.split(",", 1)[0])
    return done


def fetch_site(slug, site, years, workers):
    li, lo = grid_index(site.lat, site.lon)
    path = CACHE / f"{slug}.csv"
    path.parent.mkdir(exist_ok=True)
    if not path.exists():
        path.write_text("date,hour,aod\n")
    done = load_done(path)
    print(f"[{slug}] cell [{li}][{lo}]; {len(done)} days cached", flush=True)
    streams = resolve_streams(years, li, lo)
    todo = [d for d in daterange(*years) if f"{d:%Y-%m-%d}" not in done
            and (d.year, d.month) in streams]

    def one(d):
        txt = fetch_ascii(url_for(streams[(d.year, d.month)], d, li, lo))
        vals = parse_values(txt) if txt else None
        return d, vals

    got = 0
    with path.open("a") as out, ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(one, d): d for d in todo}
        for fut in as_completed(futs):
            d, vals = fut.result()
            if vals is None:
                continue
            for h, v in enumerate(vals):
                out.write(f"{d:%Y-%m-%d},{h},{v:.5f}\n")
            got += 1
            if got % 250 == 0:
                out.flush()
                print(f"[{slug}] {got}/{len(todo)}", flush=True)
    print(f"[{slug}] done, +{got} days", flush=True)


def main():
    global TOK
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs=2, type=int, default=[2020, 2025])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--sites", nargs="+", default=["all"])
    a = ap.parse_args()
    TOK = token()
    slugs = list(SITES) if a.sites == ["all"] else a.sites
    for slug in slugs:
        fetch_site(slug, SITES[slug], tuple(a.years), a.workers)


if __name__ == "__main__":
    main()
