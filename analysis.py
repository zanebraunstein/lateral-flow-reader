"""
Offline analysis of a recorded run.

Produces the three figures the reader exists to measure:

  1. time to positivity  -- when the test line first became reliably visible
  2. band density        -- how much dye ended up on the test line
  3. rate of change      -- how fast it got there, onset to plateau

Reads only `profiles.npz`, so detection thresholds can be changed here and
the whole run re-scored without touching a cassette.

    python3 analysis.py                    # newest run under runs/
    python3 analysis.py runs/20260722-1115
    python3 analysis.py --test-snr 8 --json
"""

import argparse
import json
import os
import sys

import numpy as np

import recorder
import strip


# Frames arrive ~10x faster than the chemistry changes. Binning to a fixed
# grid removes frame-to-frame jitter and makes the time axis uniform, which
# differentiation needs.
BIN_SECONDS = 5.0

# The end-point reading: median over the tail of the run
PLATEAU_TAIL_S = 60.0

# Plateau is called at this fraction of the final level
PLATEAU_FRAC = 0.95


def bin_series(t, y, bin_s=BIN_SECONDS):
    """
    Median-reduce (t, y) onto a fixed time grid.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    if t.size == 0:
        return np.array([]), np.array([])

    bins = np.floor((t - t[0]) / bin_s).astype(int)

    out_t = []
    out_y = []

    for b in np.unique(bins):
        m = bins == b
        out_t.append(np.median(t[m]))
        out_y.append(np.median(y[m]))

    return np.asarray(out_t), np.asarray(out_y)


def latch_time(t, present, window=None, votes=None):
    """
    When a boolean detection series becomes reliable.

    Returns (onset, confirmed): `confirmed` is when the vote threshold was
    met, `onset` is the first detection in the window that met it. The latch
    necessarily lags the real appearance of the band, so report onset as the
    time to positivity and keep `confirmed` for auditing.
    """
    window = strip.STABILITY_WINDOW if window is None else window
    votes = strip.STABILITY_VOTES if votes is None else votes

    present = np.asarray(present, dtype=bool)

    for i in range(len(present)):
        lo = max(0, i - window + 1)
        chunk = present[lo:i + 1]

        if chunk.sum() >= votes:
            first = lo + int(np.argmax(chunk))
            return float(t[first]), float(t[i])

    return None, None


def find_plateau(t, y, tail_s=PLATEAU_TAIL_S, frac=PLATEAU_FRAC):
    """
    Return (time plateau was reached, plateau level).

    The level is the median of the run's tail rather than the maximum, so a
    single bright frame cannot define the end point.
    """
    if len(t) < 2:
        return None, None

    tail = y[t >= (t[-1] - tail_s)]

    if tail.size == 0:
        tail = y[-1:]

    level = float(np.median(tail))

    if not np.isfinite(level) or level <= 0:
        return None, level

    reached = np.where(y >= frac * level)[0]

    if reached.size == 0:
        return None, level

    return float(t[reached[0]]), level


def fit_rate(t, y, t0, t1):
    """
    Least-squares slope of y over [t0, t1], with R^2 so a badly non-linear
    rise is visible rather than silently averaged away.
    """
    if t0 is None or t1 is None or t1 <= t0:
        return None, None

    m = (t >= t0) & (t <= t1)

    if m.sum() < 2:
        return None, None

    slope, intercept = np.polyfit(t[m], y[m], 1)

    pred = slope * t[m] + intercept
    ss_res = float(np.sum((y[m] - pred) ** 2))
    ss_tot = float(np.sum((y[m] - np.mean(y[m])) ** 2))

    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return float(slope), float(r2)


def analyze_run(run_dir, test_snr=None, control_snr=None, bin_s=BIN_SECONDS):
    """
    Score a recorded run. Thresholds default to the ones used during capture
    but can be overridden to re-score the same run differently.
    """
    test_snr = strip.TEST_SNR_THRESHOLD if test_snr is None else test_snr
    control_snr = strip.CONTROL_SNR_THRESHOLD if control_snr is None else control_snr

    d = recorder.load_run(run_dir)

    t = np.asarray(d["time_seconds"], dtype=float)

    if t.size == 0:
        return {"run_dir": run_dir, "frames": 0, "error": "run contains no frames"}

    # Re-derive detection from stored SNRs so thresholds are tunable here
    control_present = d["control_snr"] >= control_snr
    test_present = (d["test_snr"] >= test_snr) & control_present

    control_onset, _ = latch_time(t, control_present)
    onset, confirmed = latch_time(t, test_present)

    duration = float(t[-1] - t[0])

    out = {
        "run_dir": run_dir,
        "frames": int(t.size),
        "duration_s": duration,
        "fps": float(t.size / duration) if duration > 0 else float("nan"),
        "thresholds": {"test_snr": float(test_snr), "control_snr": float(control_snr)},
        "valid": control_onset is not None,
        "control_stable_from_s": control_onset,
        "positive": onset is not None,
        "time_to_positivity_s": onset,
        "confirmed_at_s": confirmed,
    }

    # Density and rate both live on the binned series
    tb, area = bin_series(t, d["test_area"], bin_s)
    _, ratio = bin_series(t, d["tc_area_ratio"], bin_s)
    _, peak = bin_series(t, d["test_peak_a"], bin_s)

    t_plateau, area_level = find_plateau(tb, area)
    _, ratio_level = find_plateau(tb, ratio)
    _, peak_level = find_plateau(tb, peak)

    out["density"] = {
        "test_area_a": area_level,
        "test_peak_a": peak_level,
        "tc_area_ratio": ratio_level,
        "plateau_reached_s": t_plateau,
    }

    area_rate, area_r2 = fit_rate(tb, area, onset, t_plateau)
    ratio_rate, ratio_r2 = fit_rate(tb, ratio, onset, t_plateau)

    endpoint = None
    if onset is not None and t_plateau is not None and t_plateau > onset:
        a0 = float(np.interp(onset, tb, area))
        a1 = float(np.interp(t_plateau, tb, area))
        endpoint = (a1 - a0) / (t_plateau - onset)

    out["rate"] = {
        "interval_s": [onset, t_plateau],
        "test_area_per_s": area_rate,
        "test_area_r2": area_r2,
        "endpoint_area_per_s": endpoint,
        "tc_ratio_per_s": ratio_rate,
        "tc_ratio_r2": ratio_r2,
    }

    out["series"] = {"time_s": tb.tolist(), "test_area": area.tolist(),
                     "tc_area_ratio": ratio.tolist()}

    return out


def _fmt(v, spec=".3f", dash="--"):
    return dash if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:{spec}}"


def format_report(r):
    if r.get("error"):
        return f"{r['run_dir']}: {r['error']}"

    lines = [
        f"Run: {r['run_dir']}",
        f"  {r['frames']} frames over {r['duration_s']:.1f} s ({r['fps']:.1f} fps)",
        f"  thresholds: test SNR >= {r['thresholds']['test_snr']:g}, "
        f"control SNR >= {r['thresholds']['control_snr']:g}",
        "",
    ]

    if not r["valid"]:
        lines += [
            "  INVALID: control line never became stable.",
            "  No result should be read from this run.",
            "",
        ]
    else:
        lines.append(f"  Valid: control stable from {r['control_stable_from_s']:.1f} s")
        lines.append("")

    lines.append("Time to positivity")
    if r["positive"]:
        lines.append(f"  onset      {r['time_to_positivity_s']:.1f} s")
        lines.append(f"  confirmed  {r['confirmed_at_s']:.1f} s  "
                     f"({strip.STABILITY_VOTES} of {strip.STABILITY_WINDOW} frames)")
    else:
        lines.append("  never detected -- negative for the run duration")
    lines.append("")

    d = r["density"]
    lines += [
        "Band density (raw a* units, plateau = median of run tail)",
        f"  T area          {_fmt(d['test_area_a'], '.2f')} a*.px",
        f"  T peak          {_fmt(d['test_peak_a'], '.2f')} a*",
        f"  T/C area ratio  {_fmt(d['tc_area_ratio'], '.4f')}   <- compare across runs",
        "",
    ]

    rt = r["rate"]
    lines.append("Rate of change (onset -> plateau)")
    if rt["test_area_per_s"] is None:
        lines.append("  not measurable (no onset, or no plateau reached)")
    else:
        t0, t1 = rt["interval_s"]
        lines += [
            f"  interval        {t0:.1f} -> {t1:.1f} s  ({t1 - t0:.1f} s)",
            f"  T area rate     {_fmt(rt['test_area_per_s'], '.4f')} a*.px/s"
            f"   (linear fit R^2 {_fmt(rt['test_area_r2'], '.3f')})",
            f"  endpoint rate   {_fmt(rt['endpoint_area_per_s'], '.4f')} a*.px/s",
            f"  T/C ratio rate  {_fmt(rt['tc_ratio_per_s'], '.6f')} /s"
            f"   (R^2 {_fmt(rt['tc_ratio_r2'], '.3f')})",
        ]

    return "\n".join(lines)


def latest_run(base=recorder.RUNS_DIR):
    if not os.path.isdir(base):
        return None

    runs = sorted(
        d for d in os.listdir(base)
        if os.path.isfile(os.path.join(base, d, "profiles.npz"))
    )

    return os.path.join(base, runs[-1]) if runs else None


def main(argv=None):
    p = argparse.ArgumentParser(description="Analyse a recorded lateral flow run.")
    p.add_argument("run_dir", nargs="?", help="run directory (default: newest under runs/)")
    p.add_argument("--test-snr", type=float, help="override test detection threshold")
    p.add_argument("--control-snr", type=float, help="override control detection threshold")
    p.add_argument("--bin", type=float, default=BIN_SECONDS, help="binning interval, seconds")
    p.add_argument("--json", action="store_true", help="emit JSON instead of a report")

    args = p.parse_args(argv)

    run_dir = args.run_dir or latest_run()

    if run_dir is None:
        p.error("no run directory given and none found under runs/")

    if not os.path.isfile(os.path.join(run_dir, "profiles.npz")):
        p.error(f"{run_dir} has no profiles.npz")

    result = analyze_run(
        run_dir,
        test_snr=args.test_snr,
        control_snr=args.control_snr,
        bin_s=args.bin
    )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        result.pop("series", None)
        print(format_report(result))

    return 0


if __name__ == "__main__":
    sys.exit(main())
