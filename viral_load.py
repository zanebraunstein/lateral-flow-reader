"""
Viral-load prediction: the inverse of measurement.

Measurement turns a run into metrics; this turns those metrics back into an
estimated concentration, using calibrations fit from runs of KNOWN concentration.

Each of the three metrics gets its OWN single-feature power-law calibration and
its own estimate, so a report shows what density, rate, and time each predict on
their own -- three independent readings that should agree. They rarely predict
equally well: on the calibration data density is tightest, then rate, then time
(a line's onset is noisy at low dose). A single combined model is deliberately
NOT used -- with a handful of calibration points, feeding all three into one
regression overfits and predicts worse than density alone. Keeping them separate
makes each metric's estimate and its uncertainty legible, and lets disagreement
between them flag a suspect run.

Each fit lives in log-log space because a dose response is multiplicative: an
error is naturally "within a factor of x", not "within +/- n units". Density and
rate rise with concentration (positive exponent); time-to-positivity falls
(negative exponent). The stored residual spread is a fold (multiplicative) band.

Fit or refit from labelled runs with fit_viral_load.py; this module is the model
and the predictor, and stays free of OpenCV so it can be exercised offline.
"""

import json
import os
from dataclasses import dataclass, asdict

import numpy as np


CALIBRATION_PATH = "viral_load.json"

# The metrics predicted from, in report order, with human labels. Keys match the
# values dict predict() is called with.
METRICS = ("density", "rate", "time")
LABELS = {"density": "density (T/C)", "rate": "rate", "time": "time to pos."}


@dataclass
class MetricFit:
    """
    One metric's power law: ln(conc) = a + b * ln(metric).

    `resid_sd` is the std of the log residuals -- the multiplicative uncertainty,
    so a 68% band is [est / e^sd, est * e^sd]. `lo`/`hi` bound the metric values
    the fit was trained on; outside them is extrapolation.
    """
    a: float
    b: float
    resid_sd: float
    lo: float
    hi: float
    n: int


@dataclass
class LoadCalibration:
    metrics: dict            # metric name -> MetricFit
    units: str = ""
    primary: str = "density"   # the most reliable metric (smallest residual sd)

    def save(self, path=CALIBRATION_PATH):
        with open(path, "w") as handle:
            json.dump(asdict(self), handle, indent=2)


def _fit_one(metric_vals, concentrations):
    """Power-law fit for one metric, or None if fewer than three usable pairs."""
    x = np.asarray(metric_vals, dtype=float)
    c = np.asarray(concentrations, dtype=float)

    ok = np.isfinite(x) & np.isfinite(c) & (x > 0) & (c > 0)
    x, c = x[ok], c[ok]

    if x.size < 3:
        return None

    lx, ly = np.log(x), np.log(c)
    b, a = np.polyfit(lx, ly, 1)
    resid_sd = float((ly - (a + b * lx)).std(ddof=2))

    return MetricFit(a=float(a), b=float(b), resid_sd=resid_sd,
                     lo=float(x.min()), hi=float(x.max()), n=int(x.size))


def fit(samples, concentrations, units=""):
    """
    Fit a power law per metric from paired samples.

    `samples` maps metric name -> array of that metric across the calibration
    runs (aligned with `concentrations`); missing/non-positive entries are
    dropped per metric. A metric with fewer than three usable points is skipped.
    Needs at least `density`.
    """
    fits = {}
    for name in METRICS:
        if name not in samples:
            continue
        fit_one = _fit_one(samples[name], concentrations)
        if fit_one is not None:
            fits[name] = fit_one

    if "density" not in fits:
        raise ValueError("need >= 3 usable density points to calibrate")

    # Most reliable = tightest residual spread; used as the headline estimate.
    primary = min(fits, key=lambda k: fits[k].resid_sd)

    return LoadCalibration(metrics=fits, units=units, primary=primary)


def load(path=CALIBRATION_PATH):
    """Load a saved calibration, or None if there is none."""
    if not os.path.exists(path):
        return None

    with open(path) as handle:
        data = json.load(handle)

    metrics = {k: MetricFit(**v) for k, v in data["metrics"].items()}
    return LoadCalibration(metrics=metrics, units=data.get("units", ""),
                           primary=data.get("primary", "density"))


def _predict_one(value, fit):
    if fit is None or value is None or not np.isfinite(value) or value <= 0:
        return None

    ln_est = fit.a + fit.b * np.log(value)
    sd = fit.resid_sd

    return {
        "estimate": float(np.exp(ln_est)),
        "lo68": float(np.exp(ln_est - sd)),
        "hi68": float(np.exp(ln_est + sd)),
        "lo95": float(np.exp(ln_est - 2 * sd)),
        "hi95": float(np.exp(ln_est + 2 * sd)),
        "value": float(value),
        "extrapolated": value < fit.lo or value > fit.hi,
    }


def predict(values, calib):
    """
    Estimate concentration from each metric independently.

    `values` maps metric name -> measured value (any may be None/missing).
    Returns a dict metric -> estimate dict (or None if that metric was not
    measured or not calibrated), or None if there is no calibration at all.
    """
    if calib is None:
        return None

    out = {}
    for name in METRICS:
        pred = _predict_one(values.get(name), calib.metrics.get(name))
        if pred is not None:
            pred["primary"] = (name == calib.primary)
        out[name] = pred
    return out


def format_prediction(preds):
    """Report lines for a per-metric prediction dict (or a 'no estimate' line)."""
    if not preds or all(p is None for p in preds.values()):
        return ["  not available (no calibration or no test line)"]

    def q(v):
        return f"{v:,.0f}" if v >= 10 else f"{v:.2g}"

    lines, estimates = [], []
    for name in METRICS:
        p = preds.get(name)
        label = LABELS[name]
        if p is None:
            lines.append(f"  {label:<14} --   (not measured)")
            continue

        tag = "   <- most reliable" if p["primary"] else ""
        if p["extrapolated"]:
            tag += "   (metric outside calibrated range)"

        lines.append(f"  {label:<14} {q(p['estimate']):>7}   "
                     f"(68% {q(p['lo68'])}-{q(p['hi68'])}){tag}")
        estimates.append(p["estimate"])

    # If the three readings disagree wildly, the run is suspect -- say so rather
    # than let the reader trust a single number.
    if len(estimates) >= 2 and min(estimates) > 0:
        spread = max(estimates) / min(estimates)
        if spread > 2.0:
            lines.append(f"  note: estimates span x{spread:.1f}; the metrics "
                         f"disagree, so treat this run with caution.")

    return lines
