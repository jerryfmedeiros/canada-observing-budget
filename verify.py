"""
verify.py — check that the shipped outputs still match what the code produces.

Regenerates each file under results/ and data/derived/ into a scratch directory and diffs
it against the shipped copy. Exit status is 0 only if every target matches.

Shipping the outputs lets a reader check the paper without re-fetching the inputs, but it
also lets them drift: a partly failed ERA5 fetch once wrote deflated good-night counts for
Montreal and Winnipeg, and nothing noticed until a fresh run was compared with them.

viirs_dnb.csv is rebuilt from the city windows of the 2022 VIIRS composite in
data/viirs_dnb_cache/, and reported as a skip if they are missing. The NAPS coverage statements are rebuilt offline by
naps_coverage.py, which also prints each cached file's SHA-256. skyglow.py is checked; the
other light-pollution scripts (falchi.py, calibrate_skyglow.py, sqm_check.py, tess_check.py)
need the Falchi et al. atlas, which cannot be shipped, and are not checked here.

data/derived/<slug>/auto_screen.json is covered by auto_screen.py --check, which recomputes
and compares without writing, so a stale screen shows up as an ERROR on that target.

Usage:
  python3 verify.py              # check everything
  python3 verify.py --only moon  # check targets whose name contains a substring
  python3 verify.py --list       # list targets and stop
  python3 verify.py --diff       # show a unified diff on mismatch
  python3 verify.py --update     # rewrite the shipped outputs from the code
"""
from __future__ import annotations

import argparse
import difflib
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
DERIVED = ROOT / "data" / "derived"

CITIES = ["calgary", "vancouver", "edmonton", "winnipeg",
          "toronto", "ottawa", "montreal", "halifax"]

# Scripts whose stdout is the committed results/<stem>.txt. Names line up, so this is just
# the list of stems; anything needing arguments goes in ARGS.
STEMS = [
    "corrected_budget", "ranking_intervals", "hour_threshold", "contiguous_runs",
    "aod_ranking", "threshold_sweep", "imputation_effect", "winter_dod", "winter_snow",
    "smoke_column", "light_pollution", "persistence", "pwv_percentile", "sct_threshold",
    "faint_moon", "long_record", "pm25_coverage", "auto_screen",
]
# --check keeps auto_screen read-only; it also validates data/derived/*/auto_screen.json.
ARGS = {"auto_screen": ["--check"]}

# Checks that also run on the bias-corrected counts the paper reports (--corrected).
CORRECTED_STEMS = [
    "threshold_sweep", "sct_threshold", "pwv_percentile", "aod_ranking", "faint_moon",
    "long_record", "smoke_column", "pm25_coverage",
]
EXTRA: list[tuple[str, Path, list[str]]] = [
    ("headline_corrected", RESULTS / "headline_corrected.txt", ["headline_corrected.py"]),
    ("sct_gap", RESULTS / "sct_gap.txt", ["sct_gap.py"]),
    ("fu_only", RESULTS / "fu_only.txt", ["fu_only.py"]),
    ("persistence_auto", RESULTS / "persistence_auto.txt", ["persistence.py", "--auto"]),
    ("contiguous_runs_auto", RESULTS / "contiguous_runs_auto.txt", ["contiguous_runs.py", "--auto"]),
    ("naps_coverage", RESULTS / "naps_coverage.txt", ["naps_coverage.py"]),
    ("smoke_summer", RESULTS / "smoke_summer.txt", ["smoke_summer.py"]),
    ("smoke_clear_share", RESULTS / "smoke_clear_share.txt", ["smoke_clear_share.py"]),
    ("auto_flag", RESULTS / "auto_flag.txt", ["auto_flag.py"]),
    ("night_record", RESULTS / "night_record.txt", ["night_record.py"]),
    ("surface_wind", RESULTS / "surface_wind.txt", ["surface_wind.py"]),
    ("good_months", RESULTS / "good_months.txt", ["good_months.py"]),
    ("diurnal_auto", RESULTS / "diurnal_auto.txt", ["diurnal_auto.py"]),
    ("seasonal_cycles", RESULTS / "seasonal_cycles.txt", ["seasonal_cycles.py"]),
    ("long_record_good", RESULTS / "long_record_good.txt", ["long_record.py", "--corrected", "--good"]),
    ("long_record_good_raw", RESULTS / "long_record_good_raw.txt", ["long_record.py", "--good"]),
    ("correction_scale", RESULTS / "correction_scale.txt", ["correction_scale.py"]),
    ("transparency_aod", RESULTS / "transparency_aod.txt", ["transparency_aod.py"]),
    ("correction_shape", RESULTS / "correction_shape.txt", ["correction_shape.py"]),
    ("view_angles", RESULTS / "view_angles.txt", ["view_angles.py"]),
    ("pairwise_all", RESULTS / "pairwise_all.txt", ["pairwise_all.py"]),
    # How much the ranking leans on the companion paper's gaps: none, halved, x1.5, one gap.
    ("pairwise_all_raw", RESULTS / "pairwise_all_raw.txt", ["pairwise_all.py", "--scale", "0"]),
    ("pairwise_all_x0.5", RESULTS / "pairwise_all_x0.5.txt", ["pairwise_all.py", "--scale", "0.5"]),
    ("pairwise_all_x1.5", RESULTS / "pairwise_all_x1.5.txt", ["pairwise_all.py", "--scale", "1.5"]),
    ("pairwise_all_equal", RESULTS / "pairwise_all_equal.txt", ["pairwise_all.py", "--equal-gap"]),
    ("winter_scaled", RESULTS / "winter_scaled.txt", ["winter_scaled.py"]),
    ("skyglow", RESULTS / "skyglow.txt", ["skyglow.py"]),
]

TARGETS: list[tuple[str, Path, list[str]]] = [
    (s, RESULTS / f"{s}.txt", [f"{s}.py", *ARGS.get(s, [])]) for s in STEMS
] + [
    (f"{s}_corrected", RESULTS / f"{s}_corrected.txt", [f"{s}.py", "--corrected"])
    for s in CORRECTED_STEMS
] + EXTRA + [
    (f"observable/{c}", DERIVED / c / "observable.txt",
     ["observable_nights.py", "--site", c]) for c in CITIES
]

# Scripts that write their file directly instead of to stdout. Checked by stashing the
# committed copy, re-running, and comparing; the original is restored either way.
IN_PLACE: list[tuple[str, Path, list[str]]] = [
    ("smoke_trend_sources", RESULTS / "smoke_trend_sources.csv",
     ["smoke_trend.py"]),
]

# The VIIRS city radiances are rebuilt from the shipped city windows of the 2022 composite
# (or from the composite itself if it has been downloaded); a skip only if both are missing.
VIIRS_DIR = ROOT / "data" / "viirs_dnb_cache"
VIIRS_SRC = [VIIRS_DIR / "city_cores_15arcsec_2022.npz",
             VIIRS_DIR / "VNL_v22_npp-j01_2022_global_vcmslcfg_c202303062300.average_masked.dat.tif.gz"]
HAVE_VIIRS = any(p.exists() for p in VIIRS_SRC)
SKIPPED = {} if HAVE_VIIRS else {
    "viirs_dnb.csv": "needs data/viirs_dnb_cache/city_cores_15arcsec_2022.npz or the 2022 "
                     "VIIRS composite; then: fetch_viirs_dnb.py --verify",
}


def regenerate(argv: list[str], out: Path) -> tuple[bool, str]:
    """Run argv, capturing stdout to `out`. Progress goes to stderr and is discarded."""
    with out.open("w") as f:
        p = subprocess.run([sys.executable, *argv], stdout=f, stderr=subprocess.PIPE,
                           cwd=ROOT, text=True)
    return p.returncode == 0, p.stderr[-400:] if p.returncode else ""


def verify_in_place(label: str, path: Path, argv: list[str], diff: bool) -> bool:
    """Re-run a script that writes `path` itself, and report whether the content changed."""
    before = path.read_text()
    backup = path.with_suffix(path.suffix + ".verifybak")
    path.replace(backup)
    try:
        p = subprocess.run([sys.executable, *argv], stdout=subprocess.DEVNULL,
                           stderr=subprocess.PIPE, cwd=ROOT, text=True)
        if p.returncode != 0 or not path.exists():
            print(f"  ERROR    {label:26s} {' '.join(argv)} failed\n"
                  f"           {p.stderr[-400:].strip()}")
            return False
        after = path.read_text()
        same = after == before
        if same:
            print(f"  ok       {label:26s} {len(after.splitlines())} lines")
        else:
            print(f"  DIFFERS  {label:26s} content changed on re-run")
            if diff:
                for line in list(difflib.unified_diff(
                        before.splitlines(), after.splitlines(),
                        "committed", "regenerated", lineterm=""))[:40]:
                    print(f"           {line}")
        return same
    finally:
        # Always put the committed copy back; --update is the only path that rewrites it.
        if backup.exists():
            if path.exists():
                path.unlink()
            backup.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="verify only targets whose label contains this substring")
    ap.add_argument("--list", action="store_true", help="list targets and exit")
    ap.add_argument("--diff", action="store_true", help="print a unified diff on mismatch")
    ap.add_argument("--update", action="store_true",
                    help="rewrite the committed outputs from the code instead of checking "
                         "them (review the resulting diff before committing)")
    args = ap.parse_args()

    targets = [t for t in TARGETS if not args.only or args.only in t[0]]
    inplace = [t for t in IN_PLACE if not args.only or args.only in t[0]]
    if args.list:
        for label, path, argv in targets + inplace:
            print(f"  {label:26s} {path.relative_to(ROOT)}  <-  {' '.join(argv)}")
        for name, why in SKIPPED.items():
            print(f"  {'(skipped)':26s} {name}  -- {why}")
        return 0

    if args.update:
        wrote = bad = 0
        for label, _, argv in inplace:          # these write their own file
            p = subprocess.run([sys.executable, *argv], stdout=subprocess.DEVNULL,
                               stderr=subprocess.PIPE, cwd=ROOT, text=True)
            print(f"  {'wrote' if p.returncode == 0 else 'ERROR':8s} {label}", flush=True)
            wrote += p.returncode == 0
            bad += p.returncode != 0
        for label, path, argv in targets:
            path.parent.mkdir(parents=True, exist_ok=True)
            ran, err = regenerate(argv, path)
            if ran:
                print(f"  wrote    {label:26s} {len(path.read_text().splitlines())} lines",
                      flush=True)
                wrote += 1
            else:
                print(f"  ERROR    {label:26s} {err.strip()}", flush=True)
                bad += 1
        print(f"\n{wrote} rewritten, {bad} failed. Review the diff before committing.")
        return 0 if not bad else 1

    ok = failed = missing = 0
    with tempfile.TemporaryDirectory() as tmp:
        for label, path, argv in targets:
            if not path.exists():
                print(f"  MISSING  {label:26s} {path.relative_to(ROOT)} not committed")
                missing += 1
                continue
            out = Path(tmp) / (label.replace("/", "_") + ".txt")
            ran, err = regenerate(argv, out)
            if not ran:
                print(f"  ERROR    {label:26s} {' '.join(argv)} failed\n           {err.strip()}")
                failed += 1
                continue
            want = path.read_text().splitlines()
            got = out.read_text().splitlines()
            if want == got:
                print(f"  ok       {label:26s} {len(got)} lines")
                ok += 1
            else:
                n = sum(1 for a, b in zip(want, got) if a != b) + abs(len(want) - len(got))
                print(f"  DIFFERS  {label:26s} {n} line(s) changed")
                failed += 1
                if args.diff:
                    for line in list(difflib.unified_diff(
                            want, got, "committed", "regenerated", lineterm=""))[:40]:
                        print(f"           {line}")

        for label, path, argv in inplace:
            if path.exists():
                if verify_in_place(label, path, argv, args.diff):
                    ok += 1
                else:
                    failed += 1
            else:
                print(f"  MISSING  {label:26s} not committed")
                missing += 1

    if HAVE_VIIRS and (not args.only or args.only in "viirs_dnb"):
        if args.update:
            p = subprocess.run([sys.executable, "fetch_viirs_dnb.py"], cwd=ROOT,
                               capture_output=True, text=True)
            print(f"  {'wrote' if p.returncode == 0 else 'ERROR'}    {'viirs_dnb':26s}")
            ok, failed = (ok + 1, failed) if p.returncode == 0 else (ok, failed + 1)
        else:
            p = subprocess.run([sys.executable, "fetch_viirs_dnb.py", "--verify"], cwd=ROOT,
                               capture_output=True, text=True)
            last = (p.stdout.strip().splitlines() or ["(no output)"])[-1]
            if p.returncode == 0:
                print(f"  ok       {'viirs_dnb':26s} {last}")
                ok += 1
            else:
                print(f"  DIFFERS  {'viirs_dnb':26s} {last}")
                failed += 1

    for name, why in SKIPPED.items():
        print(f"  skip     {name:26s} {why}")

    print(f"\n{ok} matched, {failed} differed or errored, {missing} missing "
          f"({len(SKIPPED)} not rebuildable here).")
    if failed or missing:
        print("A mismatch means the committed evidence no longer matches the code. "
              "Regenerate the target and check what moved before trusting either.")
    return 0 if not (failed or missing) else 1


if __name__ == "__main__":
    sys.exit(main())
