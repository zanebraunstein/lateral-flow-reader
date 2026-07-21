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


def draw_band_line_on_main(
    frame,
    quad,
    canon_x,
    y0,
    y1,
    label,
    color
):
    """
    Project a band position from canonical cassette space back onto the
    live camera view.
    """
    dst = np.array([
        [0, 0],
        [CANON_W - 1, 0],
        [CANON_W - 1, CANON_H - 1],
        [0, CANON_H - 1]
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


def draw_readout(
    disp,
    t_strength,
    c_strength,
    tc_ratio,
    t_snr,
    c_snr,
    test_present,
    control_present,
    stable_test,
    stable_control
):
    """
    Numeric readout and detection state, drawn every frame a cassette is seen.
    """
    def yes_no(flag):
        return "YES" if flag else "NO"

    cv.putText(
        disp,
        f"T={t_strength:.2f}  C={c_strength:.2f}  T/C={tc_ratio:.2f}",
        (20, 80),
        FONT,
        0.7,
        COLOR_TEXT,
        2
    )

    cv.putText(
        disp,
        (
            f"T SNR={t_snr:.1f}  "
            f"C SNR={c_snr:.1f}  "
            f"C={yes_no(control_present)}  "
            f"T={yes_no(test_present)}"
        ),
        (20, 110),
        FONT,
        0.65,
        COLOR_TEXT,
        2
    )

    cv.putText(
        disp,
        f"Stable Control: {yes_no(stable_control)}",
        (20, 140),
        FONT,
        0.6,
        COLOR_CONTROL,
        2
    )

    cv.putText(
        disp,
        f"Stable Test: {yes_no(stable_test)}",
        (20, 170),
        FONT,
        0.6,
        COLOR_TEST,
        2
    )
