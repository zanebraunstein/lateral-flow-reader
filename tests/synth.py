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


def window_scene(t_amp=50, c_amp=70, quad=None):
    """
    A frame containing ONLY the results window (bright membrane rectangle with
    C and T bands), warped into place at an arbitrary tilted quad -- the rest
    of the cassette is absent, as in a close-up fixed rig.

    Returns (frame, window_quad). Feeding window_quad back into
    strip.analyze(frame, window_quad) must recover the bands, which is the
    round trip the calibration path relies on.

    Bands are painted at the strip fractions inside the canonical window, so a
    correct warp lands them at EXPECTED_T_FRAC / EXPECTED_C_FRAC of the strip.
    """
    import strip

    w, h = strip.WINDOW_CANON_W, strip.WINDOW_CANON_H

    canvas = np.full((h, w, 3), MEMBRANE, np.uint8)

    sx0, sy0, sx1, sy1 = strip.calibrated_strip_bounds(canvas)
    strip_w = sx1 - sx0

    for frac, amp in ((strip.EXPECTED_T_FRAC, t_amp), (strip.EXPECTED_C_FRAC, c_amp)):
        if amp <= 0:
            continue

        x = int(sx0 + frac * strip_w)
        canvas[sy0:sy1, x - 5:x + 6] = (MEMBRANE - amp, MEMBRANE - amp, MEMBRANE)

    return _paint_window(canvas, quad)


def window_scene_at(control_frac, test_frac, control_amp=70, test_amp=50,
                    quad=None, band_half=5):
    """
    Like window_scene but with the control and test bands placed at explicit
    strip fractions -- so a cassette of either orientation can be built
    (e.g. control on the left with control_frac < test_frac). `band_half` sets
    each band's half-width in pixels (large values make broad bands).
    """
    import strip

    w, h = strip.WINDOW_CANON_W, strip.WINDOW_CANON_H
    canvas = np.full((h, w, 3), MEMBRANE, np.uint8)

    sx0, sy0, sx1, sy1 = strip.calibrated_strip_bounds(canvas)
    strip_w = sx1 - sx0

    xs = np.arange(w, dtype=np.float32)

    for frac, amp in ((control_frac, control_amp), (test_frac, test_amp)):
        if amp <= 0:
            continue

        cx = sx0 + frac * strip_w
        # Gaussian band: reduce blue/green (keep red) so it reads reddish, with
        # band_half as the width. Broad bands (large band_half) exceed the
        # candidate width filter, exercising the snap-to-peak path.
        dip = amp * np.exp(-0.5 * ((xs - cx) / band_half) ** 2)

        region = canvas[sy0:sy1].astype(np.float32)
        region[:, :, 0] -= dip
        region[:, :, 1] -= dip
        canvas[sy0:sy1] = np.clip(region, 0, 255).astype(np.uint8)

    return _paint_window(canvas, quad)


def _paint_window(canvas, quad):
    h, w = canvas.shape[:2]

    if quad is None:
        # An off-centre, slightly tilted placement in a 1080p frame
        quad = np.array(
            [[500, 300], [1400, 330], [1380, 700], [520, 670]],
            dtype=np.float32
        )

    src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    matrix = cv.getPerspectiveTransform(src, quad)

    frame = np.full((1080, 1920, 3), 30, np.uint8)
    cv.warpPerspective(
        canvas, matrix, (1920, 1080),
        dst=frame, borderMode=cv.BORDER_TRANSPARENT
    )

    return frame, quad


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
