"""
Viral-load prediction: the inverse of measurement.

Measurement turns a run into metrics; this turns a metric back into an estimated
concentration, using a calibration fit from runs of KNOWN concentration.

Which metric? Empirically (leave-one-out CV on the calibration set), the plateau
T/C density predicts concentration best on its own -- a power law,
conc = k * density^b. Adding rate or time-to-positivity as extra regressors makes
out-of-sample prediction WORSE, not better: with a handful of calibration points
the extra freedom overfits, and time-to-positivity is too noisy at low dose to
carry load information. So the model is deliberately single-feature; rate and
onset stay in the report as independent cross-checks, not as inputs.

The fit lives in log-log space because a dose response is multiplicative: an
error is naturally "within a factor of x", not "within +/- n units". The stored
residual spread is therefore a fold (multiplicative) band.

Fit or refit from labelled runs with fit_viral_load.py; this module is the model
and the predictor, and stays free of OpenCV so it can be exercised offline.
"""

import json
import os
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np


CALIBRATION_PATH = "viral_load.json"


@dataclass
class LoadCalibration:
    """
    Power-law calibration: ln(conc) = a + b * ln(density).

    `resid_sd` is the standard deviation of the log residuals -- the
    multiplicative uncertainty, so a prediction's 68% band is [est / e^sd,
    est * e^sd]. `dens_min`/`dens_max` bound the densities the fit was trained
    on; outside them a prediction is extrapolation. `faint_below` marks the
    low-density region where the fit is least trustworthy (a faint line carries
    little load information).
    """
    a: float
    b: float
    resid_sd: float
    dens_min: float
    dens_max: float
    faint_below: float
    n: int
    feature: str = "density_tc_ratio"
    units: str = ""

    def save(self, path=CALIBRATION_PATH):
        with open(path, "w") as handle:
            json.dump(asdict(self), handle, indent=2)


def fit(densities, concentrations, units=""):
    """
    Fit the power law from paired (density, concentration) samples.

    Densities and concentrations must be positive; non-finite or non-positive
    pairs are dropped. Needs at least three usable points.
    """
    d = np.asarray(densities, dtype=float)
    c = np.asarray(concentrations, dtype=float)

    ok = np.isfinite(d) & np.isfinite(c) & (d > 0) & (c > 0)
    d, c = d[ok], c[ok]

    if d.size < 3:
        raise ValueError(f"need >= 3 usable calibration points, got {d.size}")

    x, y = np.log(d), np.log(c)
    b, a = np.polyfit(x, y, 1)

    resid = y - (a + b * x)
    # ddof=2: two parameters were fitted. Guard the tiny-sample case.
    resid_sd = float(resid.std(ddof=2)) if d.size > 2 else 0.0

    return LoadCalibration(
        a=float(a),
        b=float(b),
        resid_sd=resid_sd,
        dens_min=float(d.min()),
        dens_max=float(d.max()),
        # Least-trustworthy region: the bottom fifth of the trained density
        # range, where the line is faint and load information is thin.
        faint_below=float(d.min() + 0.2 * (d.max() - d.min())),
        n=int(d.size),
        units=units,
    )


def load(path=CALIBRATION_PATH):
    """Load a saved calibration, or None if there is none."""
    if not os.path.exists(path):
        return None

    with open(path) as handle:
        data = json.load(handle)

    return LoadCalibration(**data)


def predict(density, calib):
    """
    Estimate concentration from a density, with a multiplicative uncertainty
    band and confidence flags.

    Returns None if there is nothing to predict (no calibration, or no test
    line). Otherwise a dict with the point estimate, 68%/95% fold bands, and
    `extrapolated` / `low_confidence` flags so a caller never reports a number
    more precisely than it is earned.
    """
    if calib is None or density is None or not np.isfinite(density) or density <= 0:
        return None

    ln_est = calib.a + calib.b * np.log(density)
    est = float(np.exp(ln_est))

    sd = calib.resid_sd

    return {
        "estimate": est,
        "lo68": float(np.exp(ln_est - sd)),
        "hi68": float(np.exp(ln_est + sd)),
        "lo95": float(np.exp(ln_est - 2 * sd)),
        "hi95": float(np.exp(ln_est + 2 * sd)),
        "density": float(density),
        "units": calib.units,
        # Outside the trained range the power law is unverified; below it the
        # estimate is a lower bound, above it an upper bound.
        "extrapolated": density < calib.dens_min or density > calib.dens_max,
        "extrapolation_side": (
            "below" if density < calib.dens_min
            else "above" if density > calib.dens_max
            else None
        ),
        "low_confidence": density < calib.faint_below,
    }


def format_prediction(pred):
    """One or more report lines for a prediction dict (or a 'no estimate' line)."""
    if pred is None:
        return ["  not available (no calibration or no test line)"]

    u = f" {pred['units']}" if pred["units"] else ""

    def q(v):
        return f"{v:,.0f}{u}" if v >= 10 else f"{v:.2g}{u}"

    lines = [
        f"  estimate     {q(pred['estimate'])}",
        f"  68% band     {q(pred['lo68'])} - {q(pred['hi68'])}",
        f"  95% band     {q(pred['lo95'])} - {q(pred['hi95'])}",
    ]

    if pred["extrapolated"]:
        side = pred["extrapolation_side"]
        bound = "lower bound" if side == "below" else "upper bound"
        lines.append(f"  note: density is {side} the calibration range; "
                     f"read the estimate as the {bound}.")
    elif pred["low_confidence"]:
        lines.append("  note: faint line (low density); estimate is the least "
                     "certain part of the range.")

    return lines
