"""
fetch_metar.py — METAR download from the IEM ASOS archive, and the report parser.

fetch_metar_csv(station, start, end) returns one MetarObs per report: the highest cloud
coverage in oktas, the lowest base of a layer of three oktas or more, any CB/TCU genus, the
AUTO flag, and the present-weather obscuration (smoke, dust, haze, fog, mist). The cloud
groups are parsed from the raw METAR text, because IEM's structured sky columns keep only
part of a multi-layer report.

Raw CSV is cached by station and date range under data/metar_cache/, so analysis scripts
download only what is missing. Source: Iowa Environmental Mesonet ASOS archive
(mesonet.agron.iastate.edu), public, no account.
"""
from __future__ import annotations

import csv
import datetime as dt
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


def http_get(url: str, timeout: int = 60) -> str:
    """GET a URL with curl, which uses the system certificate store (the python.org
    macOS installer ships its own CA bundle, which can fail where curl does not)."""
    r = subprocess.run(
        # IEM limits each IP to one request a second (HTTP 429) and returns 503 under
        # load; curl --retry backs off on both, so a throttled request is retried
        # rather than ending the run.
        ["curl", "-sS", "--fail", "--retry", "5", "--retry-delay", "2",
         "--max-time", str(timeout), url],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"curl failed for {url}: {r.stderr.strip()}")
    return r.stdout


PROJECT_ROOT = Path(__file__).parent.resolve()
# Raw IEM CSV is cached by (station, date range), since IEM allows one request a second.
# Delete a file to force a refresh, e.g. for a trailing year that may gain late reports.
METAR_CACHE_DIR = PROJECT_ROOT / "data" / "metar_cache"

ASOS_URL = (
    "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
    "?station={station}&data=metar&year1={y1}&month1={m1}&day1={d1}"
    "&year2={y2}&month2={m2}&day2={d2}&tz=Etc/UTC&format=onlycomma"
    "&latlon=yes&missing=M&trace=T&direct=yes"
)
# Cloud-group regex: SKC/CLR/NCD/NSC/CAVOK, or 3-letter coverage + 3-digit base.
# Optional CB/TCU genus suffix.
CLOUD_GROUP_RE = re.compile(
    r"\b(SKC|CLR|NCD|NSC|CAVOK|NOCLO|(FEW|SCT|BKN|OVC|VV)(\d{3})(CB|TCU)?)\b"
)

COVERAGE_TO_OKTA = {
    "SKC": 0, "CLR": 0, "NCD": 0, "NSC": 0, "CAVOK": 0, "NOCLO": 0,
    "FEW": 2, "SCT": 4, "BKN": 6, "OVC": 8, "VV": 8,
}

# Present-weather obscurations that degrade sky transparency even under a clear
# sky. Ordered by effect on transparency; the first match wins. Smoke (FU) and volcanic
# ash (VA) are both counted as smoke.
_OBSCURATION_PATTERNS = [
    ("smoke", re.compile(r"\b(?:FU|VA)\b")),
    ("dust",  re.compile(r"\b(?:BLDU|BLSA|DRDU|DRSA|DS|SS|PO|DU|SA)\b")),
    ("haze",  re.compile(r"\bHZ\b")),
    ("fog",   re.compile(r"\b(?:MI|BC|PR|FZ)?FG\b")),
    ("mist",  re.compile(r"\bBR\b")),
]


def parse_visibility_sm(text: str) -> float | None:
    """Prevailing visibility in statute miles from a METAR body (Canadian/US form).

    Handles '10SM', '3/4SM', '1 1/2SM', 'P6SM' (>6 -> 6), 'M1/4SM' (<1/4 -> 1/4)."""
    m = re.search(r"\b(\d+)\s+(\d+)/(\d+)SM\b", text)        # whole + fraction
    if m:
        return int(m.group(1)) + int(m.group(2)) / int(m.group(3))
    m = re.search(r"\b[MP]?(\d+)/(\d+)SM\b", text)           # fraction (M/P prefix)
    if m:
        return int(m.group(1)) / int(m.group(2))
    m = re.search(r"\b[MP]?(\d+)SM\b", text)                 # whole (M/P prefix)
    if m:
        return float(m.group(1))
    return None


def parse_obscuration(raw_metar: str) -> tuple[float | None, str | None]:
    """Return (visibility_sm, obscuration_category) from the report body (pre-RMK)."""
    body = raw_metar.split(" RMK", 1)[0]
    for cat, pat in _OBSCURATION_PATTERNS:
        if pat.search(body):
            return parse_visibility_sm(body), cat
    return parse_visibility_sm(body), None


@dataclass
class MetarObs:
    station: str
    timestamp: dt.datetime  # UTC
    raw: str
    coverage_okta: int            # 0..8
    cloud_base_m: float | None    # meters AGL; None if no clouds reported
    genus_hint: str | None        # CB, TCU, or None
    layers: list[tuple[str, int]] # [(coverage_str, base_ft), ...] all layers
    is_auto: bool = False         # report carried the AUTO modifier (no human augmentation)
    visibility_sm: float | None = None  # prevailing visibility, statute miles
    obscuration: str | None = None      # smoke|dust|haze|fog|mist|None (transparency)


def parse_metar_cloud(raw_metar: str) -> tuple[int, float | None, str | None, list[tuple[str, int]]]:
    """Return (max_okta_coverage, lowest_base_m_with_okta>=3, genus_hint, all_layers).

    Highest-coverage layer wins for okta. CB/TCU on any layer flags genus_hint.
    """
    layers: list[tuple[str, int]] = []
    max_okta = 0
    genus_hint: str | None = None
    lowest_base_ft: int | None = None

    for m in CLOUD_GROUP_RE.finditer(raw_metar):
        token = m.group(1)
        if token in COVERAGE_TO_OKTA:
            okta = COVERAGE_TO_OKTA[token]
            layers.append((token, 0))
            max_okta = max(max_okta, okta)
            continue
        cov = m.group(2)
        base_ft = int(m.group(3)) * 100
        suffix = m.group(4)
        if not cov or not COVERAGE_TO_OKTA.get(cov):
            continue
        okta = COVERAGE_TO_OKTA[cov]
        layers.append((cov, base_ft))
        if okta > max_okta:
            max_okta = okta
        if okta >= 3 and (lowest_base_ft is None or base_ft < lowest_base_ft):
            lowest_base_ft = base_ft
        if suffix and not genus_hint:
            genus_hint = suffix

    cloud_base_m = lowest_base_ft * 0.3048 if lowest_base_ft is not None else None
    return max_okta, cloud_base_m, genus_hint, layers


def altitude_bucket_from_base_m(base_m: float | None) -> str | None:
    if base_m is None:
        return None
    if base_m < 2000:
        return "low"
    if base_m < 6000:
        return "mid"
    return "high"


def _asos_csv_text(station: str, start: dt.date, end_inclusive: dt.date) -> str:
    """Raw IEM ASOS CSV for [start, end_inclusive], cached on disk by (station, range).

    A header-only response (no data rows) is returned but NOT cached, so a
    transient empty result retries on the next run instead of sticking."""
    cache = METAR_CACHE_DIR / f"{station}_{start.isoformat()}_{end_inclusive.isoformat()}.csv"
    if cache.exists() and cache.stat().st_size > 0:
        return cache.read_text()

    end = end_inclusive + dt.timedelta(days=1)  # IEM's day2 is exclusive
    url = ASOS_URL.format(
        station=station,
        y1=start.year, m1=start.month, d1=start.day,
        y2=end.year, m2=end.month, d2=end.day,
    )
    print(f"  GET {url}", file=sys.stderr)
    text = http_get(url, timeout=60)
    # Persist only when there is at least one data row (header + >=1 line).
    if sum(1 for ln in text.splitlines() if ln.strip()) > 1:
        METAR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".part")
        tmp.write_text(text)
        tmp.replace(cache)
    return text


def fetch_metar_csv(station: str, start: dt.date, end_inclusive: dt.date) -> list[MetarObs]:
    text = _asos_csv_text(station, start, end_inclusive)

    obs: list[MetarObs] = []
    reader = csv.DictReader(text.splitlines())
    for row in reader:
        raw = (row.get("metar") or "").strip()
        valid_str = (row.get("valid") or "").strip()
        if not raw or not valid_str:
            continue
        try:
            ts = dt.datetime.strptime(valid_str, "%Y-%m-%d %H:%M").replace(tzinfo=dt.timezone.utc)
        except ValueError:
            continue
        okta, base_m, genus, layers = parse_metar_cloud(raw)
        vis_sm, obsc = parse_obscuration(raw)
        obs.append(MetarObs(
            station=station, timestamp=ts, raw=raw,
            coverage_okta=okta, cloud_base_m=base_m,
            genus_hint=genus, layers=layers,
            is_auto=bool(re.search(r"\bAUTO\b", raw)),
            visibility_sm=vis_sm, obscuration=obsc,
        ))
    return obs
