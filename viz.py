"""
Display overlays for the live reader.

Every function here draws onto an array it is given; none of them open a
window, so this module stays usable in a headless run.
"""

import cv2 as cv
import numpy as np

from strip import CANON_W, CANON_H


FONT = cv.FONT_HERSHEY_SIMPLEX

COLOR_TEST = (255, 255, 0)
COLOR_CONTROL = (0, 255, 0)
COLOR_CANDIDATE = (0, 255, 255)
COLOR_TEXT = (255, 255, 255)
COLOR_SEARCHING = (0, 0, 255)
COLOR_STRIP_ROI = (255, 0, 255)


def draw_band_line_on_main(
    frame,
    quad,
    canon_x,
    y0,
    y1,
    label,
    color,
    canon_w=CANON_W,
    canon_h=CANON_H
):
    """
    Project a band position from canonical warped space back onto the live
    camera view. `canon_w`/`canon_h` are the dimensions of the warp `quad`
    maps to -- the full cassette when detecting, the results window when
    calibrated.
    """
    dst = np.array([
        [0, 0],
        [canon_w - 1, 0],
        [canon_w - 1, canon_h - 1],
        [0, canon_h - 1]
    ], dtype=np.float32)

    Minv = cv.getPerspectiveTransform(
        dst,
        quad.astype(np.float32)
    )

    pts = np.array([
        [[canon_x, y0]],
        [[canon_x, y1]]
    ], dtype=np.float32)

    pts = cv.perspectiveTransform(
        pts,
        Minv
    ).reshape(-1, 2)

    p0 = tuple(np.round(pts[0]).astype(int))
    p1 = tuple(np.round(pts[1]).astype(int))

    cv.line(frame, p0, p1, color, 2)

    cv.putText(
        frame,
        label,
        (p0[0] + 5, p0[1] - 5),
        FONT,
        0.7,
        color,
        2
    )


def draw_profile(profile, width=600, height=200):
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    p = profile.astype(np.float32)

    p -= np.min(p)

    if np.max(p) > 1e-6:
        p /= np.max(p)

    xs = np.linspace(
        0,
        width - 1,
        len(p)
    ).astype(np.int32)

    ys = (
        height - 1
        - p * (height - 1)
    ).astype(np.int32)

    pts = np.stack([xs, ys], axis=1)

    cv.polylines(
        canvas,
        [pts],
        False,
        COLOR_TEXT,
        2
    )

    return canvas


def mark_bands_on_profile(profile_vis, profile, t_idx, c_idx, candidates):
    """
    Annotate the profile plot with the chosen T/C bands and the runner-up
    candidates. Candidates are drawn last so they sit on top.
    """
    def x_for(idx):
        return int(idx / len(profile) * profile_vis.shape[1])

    if t_idx is not None:
        x = x_for(t_idx)

        cv.line(
            profile_vis,
            (x, 0),
            (x, profile_vis.shape[0]),
            COLOR_TEST,
            2
        )

        cv.putText(profile_vis, "T", (x + 4, 20), FONT, 0.6, COLOR_TEST, 2)

    if c_idx is not None:
        x = x_for(c_idx)

        cv.line(
            profile_vis,
            (x, 0),
            (x, profile_vis.shape[0]),
            COLOR_CONTROL,
            2
        )

        cv.putText(profile_vis, "C", (x + 4, 45), FONT, 0.6, COLOR_CONTROL, 2)

    for idx in candidates[:5]:
        x = x_for(idx)

        cv.line(
            profile_vis,
            (x, 0),
            (x, profile_vis.shape[0]),
            COLOR_CANDIDATE,
            1
        )


def draw_cassette_found(disp, quad):
    cv.polylines(
        disp,
        [quad.astype(np.int32)],
        True,
        COLOR_CONTROL,
        3
    )

    cv.putText(disp, "Cassette detected", (20, 40), FONT, 1.0, COLOR_CONTROL, 2)


def draw_searching(disp):
    cv.putText(disp, "Searching...", (20, 40), FONT, 1.0, COLOR_SEARCHING, 2)


def draw_readout(disp, tc_ratio, t_snr, c_snr, stable_test, stable_control):
    """
    Two-line verdict, using the debounced (stable) detections. The control line
    reports validity; the test line reports the result. SNRs are shown as a
    confidence hint; the full per-frame numbers live in the CSV.
    """
    # Control = validity
    if stable_control:
        control_text, control_color = f"Control: VALID   (SNR {c_snr:.1f})", COLOR_CONTROL
    else:
        control_text, control_color = f"Control: waiting   (SNR {c_snr:.1f})", COLOR_SEARCHING

    cv.putText(disp, control_text, (20, 85), FONT, 0.8, control_color, 2)

    # Test = result, only meaningful once the control is valid
    if not stable_control:
        test_text, test_color = "Test: --", COLOR_TEXT
    elif stable_test:
        test_text = f"Test: POSITIVE   T/C={tc_ratio:.2f}   (SNR {t_snr:.1f})"
        test_color = COLOR_TEST
    else:
        test_text, test_color = "Test: negative", COLOR_TEXT

    cv.putText(disp, test_text, (20, 120), FONT, 0.8, test_color, 2)


def render(frame, result, stable_test=False, stable_control=False):
    """
    Build every display image for one frame.

    Returns an ordered {window name: image} mapping; the caller decides
    whether to show them. Nothing here mutates `frame` or `result`, so a
    headless run can simply skip the call.
    """
    disp = frame.copy()

    if result is None:
        draw_searching(disp)
        return {"Lateral Flow Reader": disp}

    draw_cassette_found(disp, result.quad)

    x0, y0, x1, y1 = result.window_bounds
    sx0, sy0, sx1, sy1 = result.strip_rect

    # The warp `result.quad` maps to may be the full cassette or just the
    # results window; project band lines back using its actual dimensions.
    canon_h, canon_w = result.warped.shape[:2]

    for band, label, color in (
        (result.control, "C", COLOR_CONTROL),
        (result.test, "T", COLOR_TEST)
    ):
        # Only mark a band the reader is actually confident in, so a peak
        # snapped onto noise does not draw a misleading line.
        if band.idx is not None and band.present:
            draw_band_line_on_main(
                disp,
                result.quad,
                x0 + sx0 + band.idx,
                y0,
                y1,
                label,
                color,
                canon_w,
                canon_h
            )

    draw_readout(
        disp,
        result.tc_ratio,
        result.test.snr,
        result.control.snr,
        stable_test,
        stable_control
    )

    # Copy before drawing: `result.warped` is the measured image
    warped = result.warped.copy()

    cv.rectangle(warped, (x0, y0), (x1, y1), COLOR_CONTROL, 2)

    results_window = warped[y0:y1, x0:x1]

    cv.rectangle(results_window, (sx0, sy0), (sx1, sy1), COLOR_STRIP_ROI, 2)

    profile_vis = draw_profile(result.profile)

    mark_bands_on_profile(
        profile_vis,
        result.profile,
        result.test.idx if result.test.present else None,
        result.control.idx if result.control.present else None,
        result.candidates
    )

    return {
        "Results Window": results_window,
        "Strip ROI": result.strip_roi,
        "Signal Profile": profile_vis,
        "Warped Cassette": warped,
        "Lateral Flow Reader": disp
    }
