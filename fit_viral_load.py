"""
Fit (or refit) the viral-load calibration from runs of known concentration.

    python3 fit_viral_load.py run_labels.csv
    python3 fit_viral_load.py run_labels.csv --units ng/mL --out viral_load.json

The labels CSV needs a run column ("run", a run directory name under runs/, or a
path) and a concentration column (any header containing "concentration"); rows
with a blank concentration are skipped. Each labelled run is re-scored offline to
recover its plateau T/C density, and the power law conc = k * density^b is fit to
the pairs. Add more labelled runs and rerun to sharpen the calibration.
"""

import argparse
import csv
import os
import sys

import numpy as np

import analysis
import recorder
import viral_load


def _column(fieldnames, *wanted, contains=None):
    lower = {f.lower(): f for f in fieldnames}
    for w in wanted:
        if w in lower:
            return lower[w]
    if contains:
        for f in fieldnames:
            if contains in f.lower():
                return f
    return None


def read_labels(path, runs_dir):
    """Yield (run_dir, concentration) for each labelled row."""
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)

        run_col = _column(reader.fieldnames, "run", "run_dir", "dir", "name")
        conc_col = _column(reader.fieldnames, "concentration", contains="concentration")

        if run_col is None or conc_col is None:
            sys.exit(f"{path}: need a run column and a concentration column "
                     f"(found {reader.fieldnames})")

        for row in reader:
            raw = (row.get(conc_col) or "").strip()
            if not raw:
                continue
            try:
                conc = float(raw)
            except ValueError:
                print(f"  skip {row[run_col]!r}: concentration {raw!r} is not a number")
                continue

            run = row[run_col].strip()
            run_dir = run if os.path.isdir(run) else os.path.join(runs_dir, run)
            yield run_dir, conc


def collect(labels_path, runs_dir, test_snr=None, control_snr=None):
    """Return ({metric: array}, concentrations) over the labelled runs."""
    samples = {m: [] for m in viral_load.METRICS}
    concs = []

    for run_dir, conc in read_labels(labels_path, runs_dir):
        if not os.path.isfile(os.path.join(run_dir, "profiles.npz")):
            print(f"  skip {run_dir}: no profiles.npz")
            continue

        result = analysis.analyze_run(run_dir, test_snr=test_snr, control_snr=control_snr)

        if not result.get("valid"):
            print(f"  skip {run_dir} (conc {conc:g}): invalid run (control never stable)")
            continue

        density = result["density"]["tc_area_ratio"]
        if density is None or density <= 0:
            print(f"  skip {run_dir} (conc {conc:g}): no usable density")
            continue

        # Rate and time may be absent (a faint or non-latching line); fit() drops
        # the missing ones per metric, so store NaN and keep the density point.
        rate = result["rate"]["test_area_per_s"]
        time = result["time_to_positivity_s"]
        samples["density"].append(density)
        samples["rate"].append(rate if rate is not None else np.nan)
        samples["time"].append(time if time is not None else np.nan)
        concs.append(conc)
        print(f"  {os.path.basename(run_dir):<20} conc {conc:>6g}  "
              f"density {density:.4f}  rate {'--' if rate is None else format(rate, '.4f')}  "
              f"time {'--' if time is None else format(time, '.1f')}")

    return {m: np.array(v) for m, v in samples.items()}, np.array(concs)


def main(argv=None):
    p = argparse.ArgumentParser(description="Fit the viral-load calibration from labelled runs.")
    p.add_argument("labels", help="CSV with run and concentration columns")
    p.add_argument("--runs-dir", default=recorder.RUNS_DIR, help="where run dirs live (default: runs/)")
    p.add_argument("--units", default="", help="concentration units, e.g. ng/mL (for the report)")
    p.add_argument("--out", default=viral_load.CALIBRATION_PATH, help="output calibration path")
    p.add_argument("--test-snr", type=float, help="override test SNR threshold when re-scoring")
    p.add_argument("--control-snr", type=float, help="override control SNR threshold when re-scoring")
    args = p.parse_args(argv)

    print(f"Reading labelled runs from {args.labels} ...")
    samples, concs = collect(args.labels, args.runs_dir, args.test_snr, args.control_snr)

    if concs.size < 3:
        sys.exit(f"\nonly {concs.size} usable calibration points -- need at least 3.")

    calib = viral_load.fit(samples, concs, units=args.units)
    calib.save(args.out)

    print(f"\nFit from {concs.size} runs{'  (' + calib.units + ')' if calib.units else ''}:")
    for name in viral_load.METRICS:
        f = calib.metrics.get(name)
        if f is None:
            print(f"  {name:<8} -- not enough points")
            continue
        star = "  <- most reliable" if name == calib.primary else ""
        # Leave-one-out median fold error for this metric.
        m = calib.metrics[name]
        x = np.array(samples[name], float); c = concs.astype(float)
        ok = np.isfinite(x) & (x > 0)
        lx, ly = np.log(x[ok]), np.log(c[ok]); fold = []
        for i in range(ok.sum()):
            keep = np.ones(ok.sum(), bool); keep[i] = False
            bb, aa = np.polyfit(lx[keep], ly[keep], 1)
            fold.append(max(np.exp(aa + bb * lx[i]) / c[ok][i], c[ok][i] / np.exp(aa + bb * lx[i])))
        print(f"  {name:<8} conc = {np.exp(m.a):8.1f} * x^{m.b:+.3f}   "
              f"68% band x{np.exp(m.resid_sd):.2f}   LOO median x{np.median(fold):.2f}{star}")
    print(f"Saved calibration to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
