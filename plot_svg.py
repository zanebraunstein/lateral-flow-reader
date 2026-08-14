"""
Zero-dependency SVG plotter for a recorded run.

Same two panels as plot.py -- band density and detection SNR over time -- but
written as an SVG using only the Python standard library. No matplotlib, no
installs (numpy and OpenCV are already present for the reader). Open the .svg
in a browser.

    python3 plot_svg.py                 # newest run -> plot.svg in it
    python3 plot_svg.py runs/2026...
    python3 plot_svg.py --show          # print the path to open
    python3 plot_svg.py --test-snr 8    # re-score threshold first
"""

import argparse
import html
import math
import os
import sys

import analysis
import recorder
import strip


# Okabe-Ito CVD-safe pair, matching plot.py. Colour follows the entity.
TEST_COLOR = "#D55E00"
CONTROL_COLOR = "#0072B2"
AXIS = "#444444"
GRID = "#dddddd"
INK = "#222222"
MUTED = "#666666"

WIDTH = 900
HEIGHT = 720
PANEL_X = 70
PANEL_W = 800


def nice_ticks(lo, hi, target=5):
    """A handful of round tick values spanning [lo, hi]."""
    if hi <= lo:
        return [lo]

    raw = (hi - lo) / target
    mag = 10 ** math.floor(math.log10(raw))

    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag:
            step = m * mag
            break

    start = math.ceil(lo / step) * step
    out = []
    v = start
    while v <= hi + step * 1e-6:
        out.append(round(v, 6))
        v += step

    return out


def _line(x1, y1, x2, y2, color, width=2, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{color}" stroke-width="{width}"{d}/>')


def _text(x, y, s, color=INK, size=13, anchor="start", weight="normal"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" fill="{color}" font-size="{size}" '
            f'text-anchor="{anchor}" font-weight="{weight}">{html.escape(s)}</text>')


def _polyline(points, color, width=2):
    return f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="{width}"/>'


def _panel(px, py, pw, ph, xmax, ylo, yhi, series, ylabel, hlines=(), vlines=()):
    """SVG for one panel. series: list of (xs, ys, color, label)."""
    def sx(v):
        return px + v / (xmax or 1) * pw

    def sy(v):
        return py + ph - (v - ylo) / ((yhi - ylo) or 1) * ph

    els = [f'<rect x="{px}" y="{py}" width="{pw}" height="{ph}" '
           f'fill="none" stroke="{AXIS}" stroke-width="1"/>']

    for yt in nice_ticks(ylo, yhi):
        yy = sy(yt)
        els.append(_line(px, yy, px + pw, yy, GRID, 1))
        els.append(_text(px - 8, yy + 4, f"{yt:g}", MUTED, 11, "end"))

    for xt in nice_ticks(0, xmax):
        xx = sx(xt)
        els.append(_line(xx, py, xx, py + ph, GRID, 1))
        els.append(_text(xx, py + ph + 16, f"{xt:g}", MUTED, 11, "middle"))

    for yv, color in hlines:
        yy = sy(yv)
        els.append(_line(px, yy, px + pw, yy, color, 1, "4 3"))

    for xv, color, label in vlines:
        xx = sx(xv)
        els.append(_line(xx, py, xx, py + ph, color, 1.2, "5 3"))
        els.append(_text(xx + 4, py + 13, label, MUTED, 11, "start"))

    for xs, ys, color, _label in series:
        pts = " ".join(f"{sx(t):.1f},{sy(v):.1f}" for t, v in zip(xs, ys))
        els.append(_polyline(pts, color, 2))

    cx = px - 50
    cy = py + ph / 2
    els.append(f'<text x="{cx:.1f}" y="{cy:.1f}" fill="{INK}" font-size="12" '
               f'text-anchor="middle" transform="rotate(-90 {cx:.1f} {cy:.1f})">'
               f'{html.escape(ylabel)}</text>')

    for i, (_xs, _ys, color, label) in enumerate(series):
        lx = px + 12
        ly = py + 16 + i * 16
        els.append(_line(lx, ly, lx + 18, ly, color, 3))
        els.append(_text(lx + 24, ly + 4, label, INK, 12))

    return "\n".join(els)


def _events(result):
    ev = []
    onset = result["time_to_positivity_s"]
    plateau = result["density"]["plateau_reached_s"]
    if onset is not None:
        ev.append((onset / 60, MUTED, f"TTP {onset / 60:.1f}m"))
    if plateau is not None:
        ev.append((plateau / 60, MUTED, f"plateau {plateau / 60:.1f}m"))
    return ev


def build_svg(run_dir, test_snr=None, control_snr=None, bin_s=analysis.BIN_SECONDS):
    """
    Build the SVG string for a run. Returns (svg, result), or (None, result)
    if the run has no data.
    """
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

    xmax = max(float(tmin.max()), 1e-6)

    dmax = max(float(area_t.max()), float(area_c.max()), 1.0) * 1.18
    dmin = min(0.0, float(area_t.min()))
    density = _panel(
        PANEL_X, 55, PANEL_W, 330, xmax, dmin, dmax,
        [(tmin, area_c, CONTROL_COLOR, "Control (C)"),
         (tmin, area_t, TEST_COLOR, "Test (T)")],
        "Band density (a*.px)",
        vlines=_events(result),
    )

    smax = max(float(snr_t.max()), float(snr_c.max()),
               strip.CONTROL_SNR_THRESHOLD, 1.0) * 1.15
    snr = _panel(
        PANEL_X, 440, PANEL_W, 210, xmax, 0.0, smax,
        [(tmin, snr_c, CONTROL_COLOR, "Control SNR"),
         (tmin, snr_t, TEST_COLOR, "Test SNR")],
        "Detection SNR",
        hlines=[(strip.CONTROL_SNR_THRESHOLD, CONTROL_COLOR),
                (strip.TEST_SNR_THRESHOLD, TEST_COLOR)],
    )

    name = os.path.basename(run_dir.rstrip("/"))
    verdict = "VALID" if result["valid"] else "INVALID"
    outcome = "positive" if result["positive"] else "negative"
    title = f"{name}  -  {verdict}  -  {outcome}"

    rate = result["rate"]["test_area_per_s"]
    rate_text = (
        f"test rate {rate * 60:.1f} a*.px/min (R2={result['rate']['test_area_r2']:.2f})"
        if rate is not None else ""
    )
    # Lower area of the density panel (empty once the test line has risen),
    # left-anchored at an interior x so the whole string fits in the panel.
    rate_el = _text(PANEL_X + PANEL_W - 310, 55 + 330 - 12, rate_text, MUTED, 12)

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" font-family="sans-serif">\n'
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="white"/>\n'
        f'{_text(WIDTH / 2, 30, title, INK, 16, "middle", "bold")}\n'
        f'{density}\n'
        f'{rate_el}\n'
        f'{snr}\n'
        f'{_text(PANEL_X + PANEL_W / 2, HEIGHT - 8, "Time (minutes)", INK, 12, "middle")}\n'
        f'</svg>\n'
    )

    return svg, result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plot a recorded run as SVG (no dependencies).")
    parser.add_argument("run_dir", nargs="?", help="run directory (default: newest under runs/)")
    parser.add_argument("--test-snr", type=float, help="override test detection threshold")
    parser.add_argument("--control-snr", type=float, help="override control detection threshold")
    parser.add_argument("--bin", type=float, default=analysis.BIN_SECONDS, help="binning interval, seconds")
    parser.add_argument("--out", help="output SVG path (default: plot.svg in the run dir)")
    parser.add_argument("--show", action="store_true", help="also print the file path to open")

    args = parser.parse_args(argv)

    run_dir = args.run_dir or analysis.latest_run()

    if run_dir is None:
        parser.error("no run directory given and none found under runs/")

    if not os.path.isfile(os.path.join(run_dir, "profiles.npz")):
        parser.error(f"{run_dir} has no profiles.npz")

    svg, result = build_svg(run_dir, args.test_snr, args.control_snr, args.bin)

    if svg is None:
        print(result.get("error", "nothing to plot"))
        return 1

    out_path = args.out or os.path.join(run_dir, "plot.svg")
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(svg)

    print("Wrote", out_path)

    if args.show:
        print("Open it in a browser:", os.path.abspath(out_path))

    return 0


if __name__ == "__main__":
    sys.exit(main())
