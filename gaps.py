"""
gaps.py — the bias-correction gap specifications the ranking is tested under.

Each city's usable fraction is corrected by one annual night-day gap, the human record's
night-day delta minus a reference's. That gap has more than one defensible estimate, so the
corrected ranking is reported under all three:

  satellite   the nearer-satellite gap: human-minus-GOES, with the geostationary satellite
              that views the city at the smaller zenith angle. Every count the paper
              reports uses it. (The key is part of the bootstrap seeds in
              ranking_intervals.py and headline_corrected.py; renaming it changes the
              published intervals.) GOES-West is
              the nearer satellite for Vancouver, Calgary and Edmonton (view zenith angles
              58, 63 and 65 deg, against 72, 69 and 70 deg from GOES-East); GOES-East for
              the other five.
  ceilometer  human-minus-automated at the four cities with a screened 100%-AUTO partner
              (CYTZ, CYHU, CYBW, CZVL), the nearer-satellite gap elsewhere
  east        human-minus-GOES-East at every city, i.e. the farther satellite at the three
              western cities. At Calgary this is the outlier of its three references
              (23.2 pp against 18.8 and 17.0): GOES-East reads Calgary's snow-free nights
              cloudier than its days, mostly in summer, where GOES-West and the ceilometer
              see almost no difference.

Values are night-unit bootstrap estimates with 95% intervals, read from
data/paper1/dynamic_tables.csv at threshold_cm = 1.0 (records human_minus_goes,
human_minus_goes_west, human_minus_goes_east). The moonless budget adds each city's dark-sky
increment from the VIIRS moonlight split of Fernandes Medeiros (2026) on top of whichever gap applies;
Edmonton takes none (VIIRS null there) and Montreal's is floored at zero.
"""
from __future__ import annotations

# Annual gap point estimate and 95% interval (pp), per specification.
# The nearer-satellite gap at each city.
SATELLITE = {
    "calgary": (18.8, 15.5, 22.2), "toronto": (13.7, 11.0, 16.3),      # West / East
    "edmonton": (19.5, 16.2, 22.8), "montreal": (15.5, 12.5, 18.5),    # West / East
    "winnipeg": (13.6, 10.3, 16.9), "ottawa": (10.4, 7.5, 13.4),
    "vancouver": (9.0, 6.6, 11.3), "halifax": (12.3, 9.3, 15.5),       # West / East
}

# Where a screened automated partner exists, its DoD replaces the satellite gap.
_CEILOMETER = {
    "toronto": (18.0, 14.1, 21.7), "montreal": (13.5, 9.4, 17.6),
    "calgary": (17.0, 12.8, 21.2), "edmonton": (17.7, 13.7, 21.8),
}

# GOES-East at the three western cities, where it is the farther satellite.
_EAST = {"calgary": (23.2, 19.9, 26.4), "edmonton": (16.1, 12.8, 19.4),
         "vancouver": (10.7, 8.3, 13.3)}

CEILOMETER = {**SATELLITE, **_CEILOMETER}
EAST = {**SATELLITE, **_EAST}

SPECS = {"satellite": SATELLITE, "ceilometer": CEILOMETER, "east": EAST}

# The cities whose gap differs from the nearer-satellite one under each specification.
SPEC_MOVES = {"satellite": (), "ceilometer": tuple(_CEILOMETER), "east": tuple(_EAST)}

# Dark-sky increment added to the annual gap for the MOONLESS budget (pp).
MOONLESS_INCREMENT = {
    "calgary": 2.6, "toronto": 0.4, "edmonton": 0.0, "montreal": 0.0,
    "winnipeg": 1.3, "ottawa": 0.0, "vancouver": 0.3, "halifax": 0.1,
}


def gap(spec: str, slug: str) -> float:
    """Annual gap point estimate (pp) for one city under one specification."""
    return SPECS[spec][slug][0]


def gap_ci(spec: str, slug: str) -> tuple[float, float]:
    """95% interval ends (pp) on that gap."""
    return SPECS[spec][slug][1], SPECS[spec][slug][2]


def moonless_gap(spec: str, slug: str) -> float:
    """Dark-stratum gap (pp) for the moonless budget under one specification."""
    return round(gap(spec, slug) + MOONLESS_INCREMENT[slug], 1)
