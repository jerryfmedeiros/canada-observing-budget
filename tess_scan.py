"""
tess_scan.py — find STARS4ALL TESS photometers near the study cities.

The TESS network (tess.stars4all.eu) archives every photometer's monthly NSBM-standard files
on a public Nextcloud share. This lists the share over WebDAV, reads the header of each
photometer's latest file, and writes results/tess_stations.csv (id, lat, lon, location, first
and last month). Usage: python3 tess_scan.py
"""
from __future__ import annotations

import csv
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

SHARE = "Gr9DAbfiX8Pdm86"
DAV = "https://fta-cloud.fis.ucm.es/public.php/webdav/"
OUT = Path(__file__).parent / "results" / "tess_stations.csv"
AUTH = (SHARE, "")


def listing(path=""):
    r = requests.request("PROPFIND", DAV + path, auth=AUTH, headers={"Depth": "1"}, timeout=60)
    r.raise_for_status()
    return [h.split("/public.php/webdav/")[1] for h in re.findall(r"<d:href>([^<]*)</d:href>", r.text)][1:]


def header(folder):
    try:
        files = sorted(f for f in listing(folder) if f.endswith(".dat"))
        if not files:
            return None
        r = requests.get(DAV + files[-1], auth=AUTH, headers={"Range": "bytes=0-2500"}, timeout=60)
        txt = r.text
        pos = re.search(r"# Position:\s*([-\d.]+),\s*([-\d.]+)", txt)
        loc = re.search(r"# Location name:\s*(.*)", txt)
        m = lambda f: re.search(r"_(\d{4}-\d{2})\.dat$", f).group(1)
        return [folder.strip("/"), pos.group(1) if pos else "", pos.group(2) if pos else "",
                (loc.group(1).strip() if loc else ""), m(files[0]), m(files[-1]), len(files)]
    except Exception as e:
        return [folder.strip("/"), "", "", f"ERROR {e}", "", "", 0]


def main():
    folders = listing()
    with ThreadPoolExecutor(16) as ex:
        rows = [r for r in ex.map(header, folders) if r]
    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "lat", "lon", "location", "first", "last", "n_months"])
        w.writerows(rows)
    print(f"{len(rows)} photometers -> {OUT}")


if __name__ == "__main__":
    main()
