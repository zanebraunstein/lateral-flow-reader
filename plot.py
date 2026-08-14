"""
Plot a recorded run's kinetics.

Two stacked panels sharing a time axis: band density (a* area) and detection
SNR, with time-to-positivity and plateau marked. Density is the measurement;
SNR shows detection confidence and the thresholds it crosses.

    python3 plot.py                 # newest run -> writes plot.png into it
    python3 plot.py runs/2026...    # a specific run
    python3 plot.py --show          # also open a window
    python3 plot.py --test-snr 8    # re-score at a different threshold first
"""

import argparse
import os
import sys

import analysis
import recorder
import strip


# Okabe-Ito categorical pair (CVD-safe: adjacent dE 21.9). Test = vermilion,
# control = blue; assigned by entity and never swapped.
TEST_COLOR = "#D55E00"
CONTROL_COLOR = "#0072B2"
MARK_COLOR = "#555555"


def build_figure(run_dir, test_snr=None, control_snr=None, bin_s=analysis.BIN_SECONDS):
    """
    Build the figure for a run. Returns (figure, result), or (None, result) if
    the run has no data. Caller sets the matplotlib backend before calling.
    """
    import matplotlib.pyplot as plt

    result = analysis.analyze_run(
        run_dir, test_snr=test_snr, control_snr=control_snr, bin_s=bin_s
    )

    if result.get("error"):
        return None, result

    data = recorder.load_run(run_dir)
    seconds = data["time_seconds"]

    def binned(values):
        t, y = analysis.bin_series(seconds, values, bin_s)
        return t / 60.0, y

    tmin, area_t = binned(data["test_area"])
    _, area_c = binned(data["control_area"])
    _, snr_t = binned(data["test_snr"])
    _, snr_c = binned(data["control_snr"])

    fig, (ax_density, ax_snr) = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 2]}
    )

    # --- Density (the measurement) ---
    ax_density.plot(tmin, area_c, color=CONTROL_COLOR, lw=2, label="Control (C)")
    ax_density.plot(tmin, area_t, color=TEST_COLOR, lw=2, label="Test (T)")
    ax_density.set_ylabel("Band density  (a*·px)")

    verdict = "VALID" if result["valid"] else "INVALID - no stable control"
    outcome = "positive" if result["positive"] else "negative"
    ax_density.set_title(
        f"{os.path.basename(run_dir.rstrip('/'))}   ·   {verdict}   ·   {outcome}"
    )
    ax_density.legend(loc="upper left", frameon=False)
    ax_density.grid(alpha=0.25, lw=0.6)

    # Headroom so the (flat, near-max) control line is off the top frame and the
    # TTP/plateau labels have clear space above it.
    ymax = max(float(area_c.max()), float(area_t.max()), 1.0)
    ax_density.set_ylim(bottom=min(0.0, float(area_t.min())), top=ymax * 1.18)

    top = ax_density.get_ylim()[1]
    onset = result["time_to_positivity_s"]
    plateau = result["density"]["plateau_reached_s"]

    if onset is not None:
        ax_density.axvline(onset / 60, ls="--", color=MARK_COLOR, lw=1.2)
        ax_density.annotate(
            f"TTP {onset / 60:.1f} min", (onset / 60, top),
            xytext=(4, -4), textcoords="offset points",
            va="top", color=MARK_COLOR, fontsize=9
        )

    if plateau is not None:
        ax_density.axvline(plateau / 60, ls=":", color=MARK_COLOR, lw=1.2)
        ax_density.annotate(
            f"plateau {plateau / 60:.1f} min", (plateau / 60, top),
            xytext=(4, -16), textcoords="offset points",
            va="top", color=MARK_COLOR, fontsize=9
        )

    rate = result["rate"]["test_area_per_s"]
    if rate is not None:
        ax_density.text(
            0.985, 0.04,
            f"test rate {rate * 60:.1f} a*·px/min  "
            f"(R²={result['rate']['test_area_r2']:.2f})",
            transform=ax_density.transAxes, ha="right", va="bottom",
            fontsize=9, color=MARK_COLOR,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#dddddd")
        )

    # --- Detection SNR (confidence + thresholds) ---
    ax_snr.plot(tmin, snr_c, color=CONTROL_COLOR, lw=2, label="Control SNR")
    ax_snr.plot(tmin, snr_t, color=TEST_COLOR, lw=2, label="Test SNR")
    ax_snr.axhline(strip.CONTROL_SNR_THRESHOLD, ls="--", color=CONTROL_COLOR, lw=1, alpha=0.5)
    ax_snr.axhline(strip.TEST_SNR_THRESHOLD, ls="--", color=TEST_COLOR, lw=1, alpha=0.5)
    ax_snr.set_ylabel("Detection SNR")
    ax_snr.set_xlabel("Time (minutes)")
    ax_snr.legend(loc="upper left", frameon=False)
    ax_snr.grid(alpha=0.25, lw=0.6)

    fig.tight_layout()

    return fig, result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plot a recorded lateral flow run.")
    parser.add_argument("run_dir", nargs="?", help="run directory (default: newest under runs/)")
    parser.add_argument("--test-snr", type=float, help="override test detection threshold")
    parser.add_argument("--control-snr", type=float, help="override control detection threshold")
    parser.add_argument("--bin", type=float, default=analysis.BIN_SECONDS, help="binning interval, seconds")
    parser.add_argument("--out", help="output PNG path (default: plot.png in the run dir)")
    parser.add_argument("--show", action="store_true", help="open a window as well as saving")

    args = parser.parse_args(argv)

    run_dir = args.run_dir or analysis.latest_run()

    if run_dir is None:
        parser.error("no run directory given and none found under runs/")

    if not os.path.isfile(os.path.join(run_dir, "profiles.npz")):
        parser.error(f"{run_dir} has no profiles.npz")

    try:
        import matplotlib
        if not args.show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required for plotting: pip install matplotlib", file=sys.stderr)
        return 1

    fig, result = build_figure(run_dir, args.test_snr, args.control_snr, args.bin)

    if fig is None:
        print(result.get("error", "nothing to plot"))
        return 1

    out_path = args.out or os.path.join(run_dir, "plot.png")
    fig.savefig(out_path, dpi=120)
    print("Wrote", out_path)

    if args.show:
        plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
