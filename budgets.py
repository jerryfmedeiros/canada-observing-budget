"""
budgets.py — read the per-city budgets out of data/derived/.

Parses the 6-year mean row of each city's observable.txt (written by observable_nights.py),
so scripts take their counts from the file that holds them instead of a hand-copied constant.

Stdlib only, so it can be used without the METAR/ERA5 stack.

Usage: python3 budgets.py       # print the table
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple

DERIVED = Path(__file__).parent / "data" / "derived"

# The first [OBSERVABLE BUDGET] block belongs to the human-augmented station; a site with an
# automated partner writes a second, raw-only block for it, which is not the paper's budget.
_BLOCK = re.compile(r"\[OBSERVABLE BUDGET\].*?(?=\n\[|\Z)", re.S)
_MEAN = re.compile(r"^mean\s*\|(.*)$", re.M)


class Budget(NamedTuple):
    """The 6-year mean row of a city's budget (human station, astronomical dark), per year."""

    clear_dark_h: int       # clear-dark hours, raw
    moonless_h: int         # clear moonless-dark hours, raw
    obs: int                # observable nights, raw
    clean: int              # of those, smoke-clean
    good: int               # of those, also transparent: good nights, raw
    lowjet: int             # good nights that also pass the jet-stream test, raw
    obs_corr: int           # observable nights, bias-corrected
    good_corr: int          # good nights, bias-corrected
    clear_dark_h_corr: int  # clear-dark hours, bias-corrected

    @property
    def conversion_pct(self) -> int:
        """Corrected good nights as a percentage of corrected observable nights."""
        return round(100 * self.good_corr / self.obs_corr)


def read_budget(slug: str, derived: Path | None = None) -> Budget:
    """Parse data/derived/<slug>/observable.txt and return its mean-row budget.

    Raises FileNotFoundError if the city has not been generated, and ValueError if the file
    lacks the good-night and corrected columns, which means observable_nights.py ran with
    --no-era5.
    """
    path = (derived or DERIVED) / slug / "observable.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing -- run: python3 observable_nights.py --site {slug} "
            f"> data/derived/{slug}/observable.txt")
    block = _BLOCK.search(path.read_text())
    if not block:
        raise ValueError(f"{path}: no [OBSERVABLE BUDGET] block")
    mean = _MEAN.search(block.group(0))
    if not mean:
        raise ValueError(f"{path}: budget block has no 'mean' row")
    cells = [c.strip() for c in mean.group(1).split("|")]
    if len(cells) < 9:
        raise ValueError(
            f"{path}: budget has {len(cells)} columns, expected >= 9 -- generated without "
            f"ERA5, so there is no good-night count to read")
    return Budget(*(int(c) for c in cells[:9]))


def all_budgets(derived: Path | None = None) -> dict[str, Budget]:
    """Every generated city, keyed by slug, ordered by corrected good nights descending."""
    root = derived or DERIVED
    out = {}
    for path in sorted(root.glob("*/observable.txt")):
        out[path.parent.name] = read_budget(path.parent.name, root)
    return dict(sorted(out.items(), key=lambda kv: -kv[1].good_corr))


def good_counts(derived: Path | None = None) -> dict[str, int]:
    """Corrected good nights per year per city, the ranking of Table tab:budget."""
    return {s: b.good_corr for s, b in all_budgets(derived).items()}


if __name__ == "__main__":
    print(f"{'city':10s} {'clr-dark':>8s} {'moonless':>8s} {'obs':>5s} {'clean':>5s} "
          f"{'good':>5s} {'lowjet':>6s} {'obs c':>6s} {'good c':>6s} {'clr-dark c':>10s} "
          f"{'conv':>5s}")
    for slug, b in all_budgets().items():
        print(f"{slug:10s} {b.clear_dark_h:8d} {b.moonless_h:8d} {b.obs:5d} {b.clean:5d} "
              f"{b.good:5d} {b.lowjet:6d} {b.obs_corr:6d} {b.good_corr:6d} "
              f"{b.clear_dark_h_corr:10d} {b.conversion_pct:4d}%")
