"""sites.py — the eight cities: station, reference point, timezone, and NAPS filter.

Every fetcher and analysis script reads its site constants from here.

`human` is the airport's human-augmented METAR station. `auto` is a nearby fully automated
station used as the ceilometer reference, or None. Both are candidates until
auto_screen.py checks them against the METAR AUTO flag at night: CYXX (Abbotsford) and
CYAW (Shearwater) are carried for Vancouver and Halifax but fail the screen, being
human-augmented themselves, so only four cities have a ceilometer partner.

The reference point (ephemeris, ERA5 grid cell, darkness) is the human station's position
from the IEM station metadata (json/network.py?network=CA_<prov>_ASOS).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Site:
    slug: str
    name: str
    human: str                              # human-augmented station ICAO
    auto: str | None                        # co-located automated ICAO, or None
    lat: float                              # reference point: ephemeris, ERA5 cell, darkness
    lon: float
    elevation_m: int
    iana_tz: str                            # local time for keying nights (handles DST)
    naps_city: str | None                   # lowercase substring matched against the NAPS city column
    naps_std_offset_h: int                  # NAPS local STANDARD-time UTC offset (no DST)

    @property
    def stations(self) -> tuple[str, ...]:
        """Station ICAOs to analyse: (human,) or (human, auto)."""
        return (self.human,) if self.auto is None else (self.human, self.auto)

    def era5_area(self, pad: float = 0.15) -> list[float]:
        """ERA5 request box [N, W, S, E] around the reference point (0.25 deg grid)."""
        return [self.lat + pad, self.lon - pad, self.lat - pad, self.lon + pad]


# Keyed by slug.
SITES: dict[str, Site] = {
    "calgary": Site(
        slug="calgary", name="Calgary", human="CYYC", auto="CYBW",
        lat=51.1139, lon=-114.0203, elevation_m=1084, iana_tz="America/Edmonton",
        naps_city="calgary", naps_std_offset_h=-7,
    ),
    "vancouver": Site(
        slug="vancouver", name="Vancouver", human="CYVR", auto="CYXX",
        lat=49.1830, lon=-123.1682, elevation_m=4, iana_tz="America/Vancouver",
        naps_city="vancouver", naps_std_offset_h=-8,
    ),
    "edmonton": Site(
        # CZVL (Villeneuve), about 44 km NW, is fully automated at night and reports sky
        # condition through 2020-2025.
        slug="edmonton", name="Edmonton", human="CYEG", auto="CZVL",
        lat=53.3097, lon=-113.5797, elevation_m=723, iana_tz="America/Edmonton",
        naps_city="edmonton", naps_std_offset_h=-7,
    ),
    "winnipeg": Site(
        slug="winnipeg", name="Winnipeg", human="CYWG", auto=None,
        lat=49.9167, lon=-97.2333, elevation_m=239, iana_tz="America/Winnipeg",
        naps_city="winnipeg", naps_std_offset_h=-6,
    ),
    "toronto": Site(
        # CYKZ (Buttonville) closed in November 2023 and has no IEM feed; CYTZ (Billy
        # Bishop, about 20 km SE) is fully automated at night.
        slug="toronto", name="Toronto", human="CYYZ", auto="CYTZ",
        lat=43.6772, lon=-79.6306, elevation_m=173, iana_tz="America/Toronto",
        naps_city="toronto", naps_std_offset_h=-5,
    ),
    "ottawa": Site(
        slug="ottawa", name="Ottawa", human="CYOW", auto=None,
        lat=45.3225, lon=-75.6692, elevation_m=114, iana_tz="America/Toronto",
        naps_city="ottawa", naps_std_offset_h=-5,
    ),
    "montreal": Site(
        slug="montreal", name="Montreal", human="CYUL", auto="CYHU",
        lat=45.4683, lon=-73.7414, elevation_m=36, iana_tz="America/Toronto",
        naps_city="montr", naps_std_offset_h=-5,  # matches montreal / montréal
    ),
    "halifax": Site(
        slug="halifax", name="Halifax", human="CYHZ", auto="CYAW",
        lat=44.8808, lon=-63.5086, elevation_m=145, iana_tz="America/Halifax",
        naps_city="halifax", naps_std_offset_h=-4,
    ),
}


def get_site(slug: str) -> Site:
    """Look up a site by slug, with a helpful error listing valid slugs."""
    try:
        return SITES[slug]
    except KeyError:
        raise SystemExit(
            f"Unknown site {slug!r}. Known sites: {', '.join(sorted(SITES))}"
        )


def resolve_sites(slugs: list[str]) -> list[Site]:
    """Resolve a list of slugs (or ['all']) to Site objects, preserving registry order."""
    if slugs == ["all"]:
        return list(SITES.values())
    return [get_site(s) for s in slugs]
