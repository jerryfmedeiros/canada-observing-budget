# Usable Observing Nights in Eight Canadian Cities: Cloud, Transparency, and Smoke, 2020–2025

Code and data behind:

**Fernandes Medeiros, J. (2026).** *Usable Observing Nights in Eight Canadian Cities: Cloud,
Transparency, and Smoke, 2020–2025.* Submitted to the Journal of the Royal Astronomical
Society of Canada.

The paper counts, for Toronto, Montréal, Ottawa, Halifax, Winnipeg, Calgary, Edmonton and
Vancouver, how many nights a year have at least 4 h of clear astronomical darkness (hourly
METAR cloud reports), and how many of those are also transparent (ERA5 water vapour) and
free of surface smoke (NAPS PM2.5 and the METAR smoke code). After correcting the human
cloud reports for their nighttime optimism, the cities have 58–98 such nights a year with
clear darkness and 48–70 good nights; Winnipeg and Calgary have the most good nights.

The nighttime-clearing bias the counts are corrected for is measured in the companion paper,
*Human cloud reports overstate nighttime clearing* ([code and data](https://doi.org/10.5281/zenodo.21698181)).
Its per-station tables are shipped here under `data/paper1/`, so the correction can be
reproduced without that archive.

Every number in the paper's tables and every figure can be rebuilt from the data shipped
here, with no downloads. The exception is the part of the light-pollution section that uses
the Falchi et al. (2016) atlas, whose licence does not allow it to be redistributed (see
*Light pollution*). All random steps are seeded (seed 42); bootstraps use 2000 replicates
unless stated.

To confirm that the shipped outputs still match the code:

```bash
pip install -r requirements.txt
python3 verify.py
```

## Definitions

- **Usable sky**: a METAR report at SKC or FEW (okta ≤ 2).
- **Clear-dark hours** of a night: the share of its astronomical-dark reports (Sun ≤ −18°)
  that are usable, times its hours of astronomical darkness. A night with fewer than two
  such reports takes its month's mean share.
- **Observable night**: at least 4 h of clear-dark hours (`OBSERVABLE_H` in
  `observable_nights.py`).
- **Good night**: an observable night that is smoke-clean (no clear dark report with METAR FU
  or PM2.5 ≥ 35 µg m⁻³) and transparent (median dark-hour PWV ≤ 15 mm).
- **Low-jet night**: a good night whose median 200 hPa wind is ≤ 30 m s⁻¹ and median 10 m
  wind ≤ 8 m s⁻¹. Reported as a subset of the good nights, not as a condition of them.
- **Bias correction**: each night's usable share is reduced by the measured artifact
  profile (`correction_shape.py`), scaled by the city's gap. The paper's counts use the
  nearer-satellite gap (`"satellite"` in `gaps.py`); the ceilometer and GOES-East gaps are
  sensitivity tests.

Two environment variables reproduce earlier configurations: `BUDGET_HOURS=2` sets a 2 h bar,
and `BUDGET_JET=1` makes the jet-stream test part of the good-night test. Their outputs are
kept under `results/archive/` (below).

## Layout

The scripts sit in one folder and import each other by name; run them from the repository
root.

```
Shared modules
  sites.py                the eight cities: stations, reference point, timezone, NAPS filter
  observable_nights.py    the per-city budget: writes data/derived/<slug>/observable.txt
  nights.py               per-night records and the bias correction, used by every check
  gaps.py                 the three gap specifications and their values
  budgets.py              reads the per-city budget files
  climatology.py          solar altitude and season helpers; METAR sky-state summary
  figstyle.py             shared print conventions for the figures (greyscale-safe)
  verify.py               regenerates every shipped output and diffs it against the copy here

Data fetchers (only needed to rebuild the inputs)
  fetch_metar.py          IEM ASOS METAR download and report parser
  fetch_era5.py           ERA5 PWV and 200 hPa / 10 m wind per city (Copernicus CDS)
  fetch_pwv_long.py       ERA5 PWV for 2001-2019, one city at a time (for long_record.py --good)
  fetch_pwv_box.py        the same, one request a month covering all eight cities
  fetch_snow.py           ERA5-Land daily snow depth per city (CDS)
  fetch_pm25.py           ECCC NAPS hourly surface PM2.5
  fetch_merra2_aod.py     MERRA-2 hourly column aerosol optical depth (NASA GES DISC)
  fetch_viirs_dnb.py      city-centre VIIRS radiance from the 2022 composite

Station screen and the bias correction
  auto_screen.py          which stations are human-augmented at night and which fully automated
  auto_flag.py            the AUTO-flag counts behind that screen
  correction_shape.py     how the artifact is spread over nights, from the four ceilometer pairs
  correction_scale.py     how much the correction removes per night, and on which scale
  view_angles.py          the GOES-East and GOES-West view angle at each city
  diurnal_auto.py         the diurnal cycle at the human stations and their ceilometers
  winter_dod.py           the artifact in winter, measured against the ceilometers
  winter_scaled.py        the counts with the winter correction scaled to it
  winter_snow.py          winter snow-cover share per city

The counts (Table 3, Figures 2-4)
  headline_corrected.py   every corrected count, interval and difference the paper quotes,
                          and the attrition, map and calendar figures
  corrected_budget.py     observable and good nights under each gap and correction shape
  ranking_intervals.py    block-bootstrap intervals on the good-night ranking
  pairwise_all.py         all 28 city pairs, at 95% and with a Bonferroni adjustment
  good_months.py          when in the year the good nights fall
  seasonal_cycles.py      seasonal water vapour, jet-stream test and cloud
  surface_wind.py         what the 10 m wind condition adds to the low-jet test
  night_record.py         facts about the per-night record quoted in the Methods
  long_record.py          the study window against 2001-2025

Sensitivity tests
  hour_threshold.py       the counts at bars of 2 to 6 h (raw)
  contiguous_runs.py      expected hours against the longest unbroken clear run
  threshold_sweep.py      stricter and looser transparency cutoffs
  pwv_percentile.py       a fixed PWV limit against each city's own percentile
  aod_ranking.py          a column-AOD limit added to the transparency test
  transparency_aod.py     water vapour against column AOD; AOD in place of PWV
  sct_threshold.py        SCT counted as usable sky
  sct_gap.py              the gap for that looser definition
  faint_moon.py           crediting hours with a faint Moon up
  imputation_effect.py    what the monthly-mean imputation changes
  persistence.py          whether a clear evening stays clear

Smoke
  smoke_column.py         the surface smoke flag against MERRA-2 column AOD
  smoke_trend.py          smoke by year from three sources (Figure 1)
  smoke_summer.py         the summer smoke rates at the Prairie cities
  smoke_clear_share.py    how much of the surface smoke falls under a clear sky
  fu_only.py              smoke from the METAR FU code alone
  pm25_coverage.py        where NAPS PM2.5 is missing, and what that costs
  naps_coverage.py        what the cached NAPS files contain, with their SHA-256

Light pollution (all but skyglow.py and light_pollution.py need the Falchi et al. atlas)
  falchi.py               zenith brightness and escape distances from the Falchi et al. atlas
  skyglow.py              a Walker-law index and escape distances from the 2022 VIIRS lights
  calibrate_skyglow.py    the Walker-law model fitted to the atlas, applied to 2025 lights
  light_pollution.py      city-centre VIIRS radiance against the good nights
  sqm_check.py            the atlas against Globe at Night SQM readings
  tess_scan.py            finds the TESS photometers near the cities
  tess_check.py           the atlas against the Quebec TESS photometers, and their trend

Figures
  figures.py              Figure 1 (smoke trend) and Figure 5 (diurnal cycle)
                          (Figures 2-4 are drawn by headline_corrected.py)
```

```
data/metar_cache/        METAR, 14 stations, 2000-2025 (303 MB)
data/era5_cache/<slug>/  ERA5 PWV and winds, 2020-2025, and PWV back to 2001 (285 MB)
data/pm25_cache/         NAPS hourly PM2.5, 2020-2024 (56 MB)
data/merra2_cache/       MERRA-2 column AOD per city (8.7 MB)
data/snow_cache/         ERA5-Land daily snow depth per city
data/paper1/             tables from the companion paper: the gaps used here
data/derived/<slug>/     observable.txt (the city's budget) and auto_screen.json
data/derived/correction_shape.csv   the measured correction profile
data/ne_cache/           Natural Earth land and lakes (escape distances)
data/sqm_cache/          Globe at Night reports, 2014-2023 (sqm_check.py)
data/viirs_dnb_cache/    crops of the VIIRS annual composites for 2014, 2022 and 2025 (9 MB)
data/tess_cache/         Quebec TESS-W photometer files, 2020 to August 2026 (454 MB)
results/                 the output of every script, as listed by verify.py --list
results/archive/         outputs of earlier configurations (see below)
figures/                 the five figures of the paper
```

`data/derived/<slug>/observable.txt` is each city's budget: dark and moonless hours by month;
clear-dark hours, observable, smoke-clean, good and low-jet nights by year and by month, raw
and corrected; the diurnal cycle; and the smoke rates on clear dark reports. A city with an
automated partner has a second, raw-only block for that station.

## Reproducing the paper

```bash
# One city's budget (all eight are shipped in data/derived/).
python3 observable_nights.py --site calgary > data/derived/calgary/observable.txt

# The correction profile, from the four ceilometer pairs; every corrected count reads it.
python3 correction_shape.py

# The paper's counts, intervals and differences, and Figures 2-4.
python3 headline_corrected.py
python3 corrected_budget.py
python3 ranking_intervals.py
python3 pairwise_all.py                 # also --scale 0 (no correction), --equal-gap

# Sensitivity tests; those marked take --corrected to run on the corrected counts.
python3 hour_threshold.py
python3 contiguous_runs.py              # also --auto: against the ceilometers
python3 threshold_sweep.py --corrected
python3 pwv_percentile.py --corrected
python3 aod_ranking.py --corrected
python3 transparency_aod.py
python3 sct_threshold.py --corrected
python3 faint_moon.py --corrected
python3 long_record.py --good --corrected
python3 persistence.py --auto

# Smoke.
python3 smoke_column.py --corrected
python3 smoke_trend.py
python3 pm25_coverage.py --corrected
python3 naps_coverage.py

# Figures 1 and 5.
python3 figures.py
```

`verify.py --list` shows every target with the command that produces it.

Year 2000 is left out of `long_record.py`. The FEW cloud group entered the Canadian METAR code
with the 2001 changeover, station by station: six of the eight stations report it in under
0.2% of their 2000 observations against 14–19% afterwards. Since usable sky is SKC+FEW, those
stations could not record a usable sky in 2000. The script detects this from the data and
drops the year at every station.

## Light pollution

The Falchi et al. (2016) atlas is the one input that is not shipped: its licence states that
"further distribution of these files and commercial use are both generally prohibited."
Download `World_Atlas_2015.zip` from GFZ Data Services (doi:10.5880/GFZ.1.4.2016.001) and
put the unzipped `World_Atlas_2015.tif` (about 3 GB) in `data/falchi_cache/`. `falchi.py`,
`calibrate_skyglow.py`, `sqm_check.py` and `tess_check.py` need it.

The full VIIRS annual composites (280–340 MB each) are not in the repository. The scripts
read crops of them instead: a 1-arcmin grid over southern Canada for each year
(`canada_south_1arcmin*.npz`) and a 41×41 pixel window at full resolution around each city
centre (`city_cores_15arcsec_2022.npz`). These reproduce the shipped results exactly. To
rebuild the crops, download the composites named in `fetch_viirs_dnb.py` and
`calibrate_skyglow.py` from the Earth Observation Group (free account) into
`data/viirs_dnb_cache/`, delete the `.npz` files, and run `fetch_viirs_dnb.py --cores`,
`skyglow.py` and `calibrate_skyglow.py`. `skyglow.py` and `light_pollution.py` run from the
shipped crops. `tess_check.py` reads
the shipped photometer files, through August 2026, which are the ones the paper used;
`tess_check.py --fetch` first adds newer months from the archive, which changes the results.

## Rebuilding the inputs from scratch

```bash
pip install -r requirements.txt

# METAR fills itself: every script downloads and caches the station-years it needs.
# To pull the whole 2001-2025 record up front:
python3 long_record.py --fetch-only

# ERA5 is per city; the CDS queues each month, so this takes hours. Needs ~/.cdsapirc.
for s in calgary vancouver edmonton winnipeg toronto ottawa montreal halifax; do
    python3 fetch_era5.py --site $s --years 2020 2025
done
python3 fetch_pwv_long.py                  # PWV for 2001-2019 (long_record.py --good)

python3 fetch_snow.py --site all           # ERA5-Land, same CDS account
python3 fetch_pm25.py --years 2020 2025    # ECCC NAPS, public; one national file a year
python3 fetch_merra2_aod.py --sites all --years 2020 2025   # needs an Earthdata token
```

## Earlier configurations

`results/archive/` keeps the outputs of four earlier configurations of the analysis, for
comparison. They use the older wording ("great night" for a night that also passed the
jet-stream test) and are not checked by `verify.py`.

- `bar2h/`: a 2 h observable bar (`BUDGET_HOURS=2`)
- `jet_primary/`: the jet-stream test as part of the good-night test (`BUDGET_JET=1`)
- `east_primary/`: the GOES-East gap as the primary correction
- `uniform_shape/`: the uniform correction shape as the primary correction

## Data sources and terms

| Source | Provider | Access |
|---|---|---|
| METAR (14 stations, 2000–2025) | Iowa Environmental Mesonet ASOS archive | public |
| ERA5 PWV, 200 hPa and 10 m wind | Copernicus Climate Data Store | free CDS account |
| ERA5-Land daily snow depth | Copernicus Climate Data Store | free CDS account |
| Surface PM2.5, hourly | ECCC National Air Pollution Surveillance | public |
| MERRA-2 column AOD | NASA GMAO / GES DISC | free Earthdata account |
| VIIRS Nighttime Lights V2 | Earth Observation Group, Colorado School of Mines (CC BY 4.0) | free account |
| World Atlas of Artificial Night Sky Brightness | Falchi et al. (2016), GFZ Data Services | public; not redistributable |
| Sky Quality Meter reports | Globe at Night (CC BY 4.0) | public |
| TESS-W photometers | STARS4ALL (ODbL 1.0) | public |
| Land and lakes | Natural Earth | public domain |

NAPS coverage is uneven and the paper says so: 2025 was not published at the time of
writing, the 2024 national file has no Calgary rows, and smaller gaps affect Halifax,
Winnipeg and Ottawa. `pm25_coverage.py` measures what that costs.
`results/naps_2024_coverage_check.log` records a check of the live 2024 file that a later run
cannot repeat; `naps_coverage.py` rebuilds the same statements from the cached files.

The shipped data files pin the exact inputs of the analysis and remain under the
providers' terms. Apart from the VIIRS crops, they are redistributed unmodified. In
particular:

- The VIIRS files in `data/viirs_dnb_cache/`, and `results/viirs_dnb.csv` derived from them,
  are spatial crops (and, for the 1-arcmin grids, 4×4 pixel averages) of the EOG annual
  composites. They were made utilizing VIIRS Nighttime Lights (VNL V2) data produced by the
  Earth Observation Group, Payne Institute for Public Policy, Colorado School of Mines, and
  are redistributed under CC BY 4.0. Cite Elvidge et al. (2021), Remote Sensing 13, 922,
  doi:10.3390/rs13050922.
- The TESS-W files in `data/tess_cache/` are from the STARS4ALL network and remain under the
  Open Database License (ODbL 1.0).
- The Globe at Night reports in `data/sqm_cache/` remain under CC BY 4.0.

## License and citation

The code is MIT-licensed (see LICENSE). If you use it, please cite the paper; `CITATION.cff`
has the metadata, and the DOI will be added on deposit.
