"""
Synthetic test data.

Real cassettes are one-shot and cannot be checked into a repo, so the suite
works on generated strips, frames and runs whose ground truth is known by
construction.
"""

import os

import cv2 as cv
import numpy as np


# Cassette occupies x 400..1500, y 400..800 of a 1920x1080 frame, which maps
# onto the canonical 900x320 warp:  fx = 400 + cx / 900 * 1100
CASSETTE_X0, CASSETTE_X1 = 400, 1500
CASSETTE_Y0, CASSETTE_Y1 = 400, 800

# Canonical x of the bands, chosen to land on EXPECTED_T_FRAC / EXPECTED_C_FRAC
CANON_T_X = 677
CANON_C_X = 739

MEMBRANE = 235


def strip_image(bands, h=40, w=300, noise=0.0, seed=0):
    """
    A strip ROI with reddish bands at the given (fraction, amplitude) pairs.
    """
    rng = np.random.default_rng(seed)

    img = np.full((h, w, 3), MEMBRANE, np.uint8)

    for frac, amp in bands:
        c = int(frac * w)
        img[:, max(0, c - 5):c + 6] = (MEMBRANE - amp, MEMBRANE - amp, MEMBRANE)

    if noise:
        img = np.clip(
            img.astype(np.int16) + rng.normal(0, noise, img.shape), 0, 255
        ).astype(np.uint8)

    return img


def cassette_frame(t_amp=0, c_amp=70):
    """
    A full camera frame (BGR) containing a detectable cassette, with test and
    control bands of the given dye amplitudes. `t_amp=0` means no test line.
    """
    img = np.full((1080, 1920, 3), 30, np.uint8)

    img[CASSETTE_Y0:CASSETTE_Y1, CASSETTE_X0:CASSETTE_X1] = MEMBRANE

    for canon_x, amp in ((CANON_T_X, t_amp), (CANON_C_X, c_amp)):
        if amp <= 0:
            continue

        fx = int(CASSETTE_X0 + canon_x / 900 * (CASSETTE_X1 - CASSETTE_X0))

        img[570:650, fx - 7:fx + 8] = (MEMBRANE - amp, MEMBRANE - amp, MEMBRANE)

    return img


def blank_frame():
    """A frame with no cassette in it."""
    return np.full((1080, 1920, 3), 30, np.uint8)


def true_band_excess(result):
    """
    The a* excess the test band actually puts on the membrane, measured
    straight off the strip ROI without any profile machinery.

    This is the ground truth a density measure is supposed to track.
    """
    a = cv.cvtColor(result.strip_roi, cv.COLOR_BGR2LAB)[:, :, 1].astype(np.float64)

    col = np.median(a, axis=0)

    band = col[result.test.idx - 5:result.test.idx + 6].mean()
    membrane = np.median(np.concatenate([col[:30], col[-30:]]))

    return band - membrane


def make_run_npz(
    path,
    duration=900.0,
    fps=10.0,
    control_from=20.0,
    test_from=None,
    rise_to=600.0,
    plateau_area=200.0,
    control_area=200.0,
    kind="linear",
    noise=0.0,
    seed=0
):
    """
    Write a profiles.npz with kinetics known by construction.

    `test_from=None` produces a negative run; `control_from` far in the future
    produces an invalid one. Returns (times, test_area).
    """
    rng = np.random.default_rng(seed)

    t = np.arange(0, duration, 1.0 / fps)
    n = t.size

    c_snr = np.where(t >= control_from, 20.0, 1.0)

    if test_from is None:
        t_snr = np.full(n, 0.5)
        area = np.zeros(n)
    else:
        t_snr = np.where(t >= test_from, 12.0, 0.5)

        if kind == "linear":
            frac = np.clip((t - test_from) / (rise_to - test_from), 0, 1)
        else:
            tau = (rise_to - test_from) / 3.0
            frac = np.where(t >= test_from, 1 - np.exp(-(t - test_from) / tau), 0.0)

        area = plateau_area * frac

    if noise:
        area = area + rng.normal(0, noise, n)

    c_area = np.full(n, control_area)

    os.makedirs(path, exist_ok=True)

    np.savez(
        os.path.join(path, "profiles.npz"),
        time_seconds=t,
        profiles=np.zeros((n, 178), np.float32),
        test_idx=np.full(n, 80, np.int32),
        control_idx=np.full(n, 142, np.int32),
        test_strength=t_snr,
        control_strength=c_snr,
        test_snr=t_snr,
        control_snr=c_snr,
        tc_ratio=t_snr / np.maximum(c_snr, 1e-6),
        test_present=(t_snr >= 5).astype(np.int8),
        control_present=(c_snr >= 6).astype(np.int8),
        test_area=area,
        control_area=c_area,
        tc_area_ratio=area / c_area,
        test_peak_a=area / 11.5,
        control_peak_a=c_area / 11.5,
        profile_scale=np.full(n, 2.5)
    )

    return t, area
