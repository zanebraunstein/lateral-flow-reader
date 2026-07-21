"""
Cassette geometry and strip signal analysis.

Pure OpenCV/NumPy: no camera and no display, so this module can be imported
and exercised on a workstation against recorded frames.
"""

import cv2 as cv
import numpy as np


# Canonical cassette geometry
CANON_W = 900
CANON_H = 320

# Results window ROI in canonical warped cassette view
WINDOW_X0 = 0.55
WINDOW_Y0 = 0.22
WINDOW_X1 = 0.93
WINDOW_Y1 = 0.78

STRIP_Y0_FRAC = 0.44
STRIP_Y1_FRAC = 0.66

STRIP_X0_FRAC = 0.30
STRIP_X1_FRAC = 0.82

EDGE_EXCLUDE_FRAC = 0.12
MIN_BAND_WIDTH = 6
MAX_BAND_WIDTH = 45

EXPECTED_C_FRAC = 0.80
EXPECTED_T_FRAC = 0.45
SEARCH_RADIUS_FRAC = 0.18
MIN_TC_SEPARATION_FRAC = 0.18

# Band is called present above this SNR; test also requires a valid control
CONTROL_SNR_THRESHOLD = 6.0
TEST_SNR_THRESHOLD = 5.0

# A detection must hold for STABILITY_VOTES of the last STABILITY_WINDOW frames
STABILITY_WINDOW = 10
STABILITY_VOTES = 7


def order_points(pts):
    pts = pts.astype(np.float32)

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def find_cassette_quad(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (7, 7), 0)

    edges = cv.Canny(gray, 50, 150)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))
    edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv.findContours(
        edges,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )

    contours = sorted(contours, key=cv.contourArea, reverse=True)

    h, w = frame.shape[:2]
    min_area = 0.03 * (h * w)

    for cnt in contours:
        area = cv.contourArea(cnt)

        if area < min_area:
            continue

        peri = cv.arcLength(cnt, True)

        approx = cv.approxPolyDP(
            cnt,
            0.02 * peri,
            True
        )

        if len(approx) == 4 and cv.isContourConvex(approx):
            return order_points(
                approx.reshape(4, 2)
            )

    return None


def warp_cassette(frame, quad):
    dst = np.array([
        [0, 0],
        [CANON_W - 1, 0],
        [CANON_W - 1, CANON_H - 1],
        [0, CANON_H - 1]
    ], dtype=np.float32)

    M = cv.getPerspectiveTransform(
        quad.astype(np.float32),
        dst
    )

    return cv.warpPerspective(
        frame,
        M,
        (CANON_W, CANON_H)
    )


def results_window_bounds():
    """
    Results window rectangle in canonical warped coordinates.
    """
    return (
        int(WINDOW_X0 * CANON_W),
        int(WINDOW_Y0 * CANON_H),
        int(WINDOW_X1 * CANON_W),
        int(WINDOW_Y1 * CANON_H)
    )


def strip_bounds(results_window):
    """
    Strip rectangle in results-window coordinates.
    """
    h, w = results_window.shape[:2]

    return (
        int(STRIP_X0_FRAC * w),
        int(STRIP_Y0_FRAC * h),
        int(STRIP_X1_FRAC * w),
        int(STRIP_Y1_FRAC * h)
    )


def extract_strip_roi(results_window):
    x0, y0, x1, y1 = strip_bounds(results_window)

    return results_window[y0:y1, x0:x1]


def redness_profile(strip_bgr):
    lab = cv.cvtColor(strip_bgr, cv.COLOR_BGR2LAB)

    a = lab[:, :, 1].astype(np.float32)

    # Collapse vertically into 1D profile
    background = cv.GaussianBlur(
        a,
        (0, 0),
        sigmaX=25,
        sigmaY=25
    )

    normalized = a - background

    profile = np.median(
        normalized,
        axis=0
    )

    # Smooth profile
    kernel = np.ones(15, dtype=np.float32)
    kernel /= kernel.sum()
    profile = np.convolve(profile, kernel, mode="same")

    profile -= np.median(profile)

    mad = np.median(np.abs(profile)) + 1e-6
    profile /= mad

    return profile


def find_peak_candidates(profile):
    n = len(profile)

    if n < 50:
        return []

    pad = int(EDGE_EXCLUDE_FRAC * n)

    candidates = []

    for i in range(2, n - 2):
        if i < pad or i > (n - 1 - pad):
            continue

        if profile[i] >= profile[i - 1] and profile[i] >= profile[i + 1]:
            candidates.append(i)

    candidates.sort(key=lambda i: profile[i], reverse=True)

    return candidates


def band_width_peak(profile, idx, frac=0.5):
    peak = float(profile[idx])
    base = float(np.median(profile))
    level = base + frac * (peak - base)

    left = idx
    while left > 0 and profile[left] > level:
        left -= 1

    right = idx
    while right < len(profile) - 1 and profile[right] > level:
        right += 1

    return right - left


def filter_band_candidates(profile):
    peaks = find_peak_candidates(profile)

    valid = []

    for idx in peaks:
        width = band_width_peak(profile, idx)

        if MIN_BAND_WIDTH <= width <= MAX_BAND_WIDTH:
            valid.append(idx)

    return valid


def pick_peak_near(profile, candidates, expected_frac, radius_frac):
    n = len(profile)

    expected = int(expected_frac * n)
    radius = int(radius_frac * n)

    lo = max(0, expected - radius)
    hi = min(n - 1, expected + radius)

    best = None
    best_score = -1e9

    for idx in candidates:
        if idx < lo or idx > hi:
            continue

        distance_penalty = abs(idx - expected) / (radius + 1e-6)
        score = float(profile[idx]) - 0.35 * distance_penalty

        if score > best_score:
            best_score = score
            best = idx

    return best


def pick_t_c_from_peaks(profile, candidates):
    t_idx = pick_peak_near(profile, candidates, EXPECTED_T_FRAC, SEARCH_RADIUS_FRAC)
    c_idx = pick_peak_near(profile, candidates, EXPECTED_C_FRAC, SEARCH_RADIUS_FRAC)

    if t_idx is not None and c_idx is not None:
        min_sep = int(MIN_TC_SEPARATION_FRAC * len(profile))

        if abs(t_idx - c_idx) < min_sep:
            if profile[c_idx] >= profile[t_idx]:
                t_idx = None
            else:
                c_idx = t_idx
                t_idx = None

    if c_idx is None and t_idx is not None:
        c_idx, t_idx = t_idx, None

    return t_idx, c_idx


def band_signal_snr(profile, idx, half_width=6):
    """
    Return band strength and signal-to-noise ratio.
    """
    if idx is None:
        return 0.0, 0.0

    n = len(profile)

    lo = max(0, idx - half_width)
    hi = min(n, idx + half_width + 1)

    peak = float(np.max(profile[lo:hi]))

    # Exclude the candidate band from background estimation
    mask = np.ones(n, dtype=bool)
    mask[
        max(0, idx - 3 * half_width):
        min(n, idx + 3 * half_width + 1)
    ] = False

    background = profile[mask]

    if background.size < 20:
        background = profile

    baseline = float(np.median(background))
    strength = peak - baseline

    # Robust noise estimate
    noise = float(
        np.median(np.abs(background - baseline)) + 1e-6
    )

    snr = strength / noise

    return strength, snr
