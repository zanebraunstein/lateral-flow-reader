"""
Cassette geometry and strip signal analysis.

Pure OpenCV/NumPy: no camera and no display, so this module can be imported
and exercised on a workstation against recorded frames.
"""

from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

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

# Canonical size of the results window on its own. Chosen to equal the pixel
# size of the window crop in the full-cassette path, so the strip sample count
# is identical either way and every tuning constant below stays valid whether
# the window is found by detection or by calibration.
WINDOW_CANON_W = int((WINDOW_X1 - WINDOW_X0) * CANON_W)
WINDOW_CANON_H = int((WINDOW_Y1 - WINDOW_Y0) * CANON_H)

STRIP_Y0_FRAC = 0.44
STRIP_Y1_FRAC = 0.66

STRIP_X0_FRAC = 0.30
STRIP_X1_FRAC = 0.82

# Calibrated path: symmetric margin trimmed off each side of the marked box, to
# keep its border edges out of the analysed strip while staying centred.
# Larger = cleaner detection (further from the box edges); smaller = the
# analysed strip fills more of the box. Tune if detection is edgy or the box
# feels too cropped.
CAL_STRIP_MARGIN_FRAC = 0.15

# Small: a band can sit near the edge of the strip (e.g. a control line close
# to the window edge), so only the very rim is excluded as warp artifact.
EDGE_EXCLUDE_FRAC = 0.05
MIN_BAND_WIDTH = 6
MAX_BAND_WIDTH = 45

EXPECTED_C_FRAC = 0.80
EXPECTED_T_FRAC = 0.45
SEARCH_RADIUS_FRAC = 0.18
MIN_TC_SEPARATION_FRAC = 0.18

# Detection sensitivity: a band counts as present above this SNR (test also
# requires a valid control). LOWER = more sensitive (catches fainter lines but
# risks noise); raise if it starts calling blanks positive. Check a run's real
# SNR with diagnose.py and set these just below what your true lines produce.
CONTROL_SNR_THRESHOLD = 4.5
TEST_SNR_THRESHOLD = 4.0

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


def warp_quad(frame, quad, out_w, out_h):
    """
    Perspective-warp the region bounded by `quad` to an out_w x out_h image.
    `quad` is four points ordered tl, tr, br, bl in frame coordinates.
    """
    dst = np.array([
        [0, 0],
        [out_w - 1, 0],
        [out_w - 1, out_h - 1],
        [0, out_h - 1]
    ], dtype=np.float32)

    M = cv.getPerspectiveTransform(
        quad.astype(np.float32),
        dst
    )

    return cv.warpPerspective(
        frame,
        M,
        (out_w, out_h)
    )


def warp_cassette(frame, quad):
    return warp_quad(frame, quad, CANON_W, CANON_H)


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


def calibrated_strip_bounds(results_window):
    """
    Strip rectangle for the calibrated path: most of the marked box, trimmed by
    a small SYMMETRIC margin on each side and to the band row vertically.

    The margin keeps the box's edges -- the membrane/plastic border, which
    creates sharp transitions that inflate the noise floor and spawn spurious
    peaks -- out of the analysed region. Being symmetric, the analysed strip
    stays centred under the calibration box.
    """
    h, w = results_window.shape[:2]
    margin = int(CAL_STRIP_MARGIN_FRAC * w)

    return (margin, int(STRIP_Y0_FRAC * h), w - margin, int(STRIP_Y1_FRAC * h))


def extract_strip_roi(results_window):
    x0, y0, x1, y1 = strip_bounds(results_window)

    return results_window[y0:y1, x0:x1]


def raw_redness_profile(strip_bgr):
    """
    Baseline-centred redness profile in LAB a* units.

    These units are physically meaningful: they scale with how much dye is
    on the membrane, which is what a density / viral-load estimate needs.
    """
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

    return profile


def profile_scale(raw_profile):
    """
    Robust spread of a raw profile, used to normalise it for detection.

    NOTE: this grows with the bands themselves, so dividing by it compresses
    the dose response. Good for thresholding, wrong for density -- measure
    density on the raw profile instead.
    """
    return float(np.median(np.abs(raw_profile)) + 1e-6)


def redness_profile(strip_bgr):
    """
    Noise-normalised profile, in units of the strip's own robust spread.
    Use for peak detection and SNR gating, not for density.
    """
    raw = raw_redness_profile(strip_bgr)

    return raw / profile_scale(raw)


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


def band_extent(profile, idx, frac=0.5):
    """
    Edges of the band around `idx`, walked out to `frac` of its peak height.
    """
    peak = float(profile[idx])
    base = float(np.median(profile))
    level = base + frac * (peak - base)

    left = idx
    while left > 0 and profile[left] > level:
        left -= 1

    right = idx
    while right < len(profile) - 1 and profile[right] > level:
        right += 1

    return left, right


def band_width_peak(profile, idx, frac=0.5):
    left, right = band_extent(profile, idx, frac)

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


def snap_to_peak(profile, expected_frac, radius_frac):
    """
    Position of the tallest local maximum within `radius_frac` of the expected
    fractional position -- the visible peak nearest where the band should be.

    Unlike the candidate picker this ignores band width, so a broad band that
    width-filtering would drop is still located; SNR decides presence. Returns
    None only if the window contains no local maximum (a flat region).
    """
    n = len(profile)

    expected = int(expected_frac * n)
    radius = int(radius_frac * n)

    # i-1 and i+1 are read below, so keep i within [1, n-2]
    lo = max(1, expected - radius)
    hi = min(n - 2, expected + radius)

    best = None
    best_height = -1e18

    for i in range(lo, hi + 1):
        # A local maximum with real relief on at least one side, so a flat
        # (saturated) region is not mistaken for a peak.
        rises = profile[i] >= profile[i - 1] and profile[i] >= profile[i + 1]
        strict = profile[i] > profile[i - 1] or profile[i] > profile[i + 1]

        if rises and strict and profile[i] > best_height:
            best_height = profile[i]
            best = i

    return best


def snap_t_c(profile, test_frac, control_frac, radius_frac=SEARCH_RADIUS_FRAC):
    """
    Locate the test and control bands by snapping to the tallest peak near each
    calibrated position. For the fixed-rig path, where the positions are known
    and the width filter only gets in the way.
    """
    t_idx = None if test_frac is None else snap_to_peak(profile, test_frac, radius_frac)
    c_idx = None if control_frac is None else snap_to_peak(profile, control_frac, radius_frac)

    # If both snapped to the same peak, keep it as the control (the anchor).
    if t_idx is not None and t_idx == c_idx:
        t_idx = None

    return t_idx, c_idx


def dominant_two_bands(profile, candidates, min_sep_frac=MIN_TC_SEPARATION_FRAC):
    """
    The strongest candidate peaks that are far enough apart to be a real band
    pair, returned ordered left-to-right by position (0, 1, or 2 of them).

    For a calibrated fixed rig this locates the control and test bands with no
    assumed positions; the caller labels them from the known control side.
    """
    if not candidates:
        return []

    min_sep = int(min_sep_frac * len(profile))

    chosen = [candidates[0]]

    for idx in candidates[1:]:
        if all(abs(idx - c) >= min_sep for c in chosen):
            chosen.append(idx)

            if len(chosen) == 2:
                break

    return sorted(chosen)


def pick_t_c_from_peaks(
    profile,
    candidates,
    test_frac=EXPECTED_T_FRAC,
    control_frac=EXPECTED_C_FRAC,
    radius_frac=SEARCH_RADIUS_FRAC
):
    """
    Assign test and control bands by searching near their expected fractional
    positions. For a fixed rig those come from calibration; otherwise they are
    the module defaults. A None fraction means that band is not searched for.
    """
    t_idx = (
        None if test_frac is None
        else pick_peak_near(profile, candidates, test_frac, radius_frac)
    )
    c_idx = (
        None if control_frac is None
        else pick_peak_near(profile, candidates, control_frac, radius_frac)
    )

    if t_idx is not None and c_idx is not None:
        min_sep = int(MIN_TC_SEPARATION_FRAC * len(profile))

        # Too close to be a real T/C pair, so one of them is spurious. Keep
        # the control-window candidate: the control sits at a fixed, known
        # position and is the validity anchor, whereas picking by peak height
        # can label a band at the test position as the control.
        if abs(t_idx - c_idx) < min_sep:
            t_idx = None

    # A band at the test position is NOT evidence of a control line. If the
    # control never developed the run is invalid, and saying so is safer than
    # promoting the test band and reporting a valid negative.

    return t_idx, c_idx


def band_background(profile, idx, half_width=6, exclude=()):
    """
    Profile samples far enough from the band at `idx` to estimate the local
    baseline and noise without the band contaminating them.

    `exclude` lists other band positions to also mask out -- otherwise a second
    band counts as "noise" and inflates the estimate, deflating this band's SNR.
    """
    n = len(profile)

    mask = np.ones(n, dtype=bool)

    for centre in (idx,) + tuple(exclude):
        if centre is None:
            continue
        mask[
            max(0, centre - 3 * half_width):
            min(n, centre + 3 * half_width + 1)
        ] = False

    background = profile[mask]

    if background.size < 20:
        background = profile

    return background


def band_peak_height(profile, idx, half_width=6, exclude=()):
    """
    Peak height above the local baseline, in whatever units `profile` carries.
    """
    if idx is None:
        return 0.0

    n = len(profile)

    lo = max(0, idx - half_width)
    hi = min(n, idx + half_width + 1)

    peak = float(np.max(profile[lo:hi]))
    baseline = float(np.median(band_background(profile, idx, half_width, exclude)))

    return peak - baseline


def band_area(profile, idx, half_width=6, frac=0.5, exclude=()):
    """
    Integrated signal above the local baseline across the band.

    Pass the RAW profile to get a density in a* units. Area is a better
    density proxy than peak height: it tracks total dye rather than the
    single darkest sample, so it holds up when focus drift spreads a band
    out without changing how much dye is there.
    """
    if idx is None:
        return 0.0

    baseline = float(np.median(band_background(profile, idx, half_width, exclude)))

    left, right = band_extent(profile, idx, frac)

    segment = profile[left:right + 1] - baseline

    # Clip so a dipping baseline cannot subtract from the integral
    return float(np.sum(np.clip(segment, 0.0, None)))


def band_signal_snr(profile, idx, half_width=6, exclude=()):
    """
    Return band strength and signal-to-noise ratio. `exclude` masks other bands
    out of the noise estimate so they do not deflate this band's SNR.
    """
    if idx is None:
        return 0.0, 0.0

    background = band_background(profile, idx, half_width, exclude)

    baseline = float(np.median(background))

    strength = band_peak_height(profile, idx, half_width, exclude)

    # Robust noise estimate
    noise = float(
        np.median(np.abs(background - baseline)) + 1e-6
    )

    snr = strength / noise

    return strength, snr


@dataclass
class BandReading:
    """
    One band (test or control) as measured in a single frame.

    `strength` and `snr` are in normalised units and drive detection.
    `area` and `peak_a` are in raw a* units and carry the density.
    """
    idx: Optional[int]
    strength: float
    snr: float
    present: bool
    area: float = 0.0
    peak_a: float = 0.0


@dataclass
class FrameResult:
    """
    Everything measured from one frame. Holds no image the renderer has
    touched, so drawing can never feed back into the next measurement.
    """
    quad: np.ndarray
    warped: np.ndarray
    window_bounds: Tuple[int, int, int, int]
    strip_rect: Tuple[int, int, int, int]
    strip_roi: np.ndarray
    profile: np.ndarray
    candidates: List[int]
    test: BandReading
    control: BandReading
    tc_ratio: float
    raw_profile: Optional[np.ndarray] = None
    scale: float = 1.0
    tc_area_ratio: float = 0.0


def analyze(frame, window_quad=None, bands=None):
    """
    Measure the test and control bands in one frame.

    With `window_quad` (four points bounding the results window in frame
    coordinates, e.g. from a fixed-rig calibration) the window is warped
    directly and no cassette detection happens -- so it works even when most
    of the cassette is out of frame. Without it, the full cassette is detected
    and its results window taken as a fixed fraction of the warp.

    `bands` is an optional (test_frac, control_frac) pair giving the expected
    band positions along the strip, as learned during calibration. Without it
    the module defaults are used.

    Returns a FrameResult, or None if no cassette was found (detection path).
    """
    if window_quad is not None:
        quad = np.asarray(window_quad, dtype=np.float32)
        warped = warp_quad(frame, quad, WINDOW_CANON_W, WINDOW_CANON_H)
        window_bounds = (0, 0, WINDOW_CANON_W, WINDOW_CANON_H)
        results_window = warped
        # Analyse the full marked box, so it matches the calibration.
        strip_rect = calibrated_strip_bounds(results_window)
    else:
        quad = find_cassette_quad(frame)

        if quad is None:
            return None

        warped = warp_cassette(frame, quad)

        x0, y0, x1, y1 = results_window_bounds()
        window_bounds = (x0, y0, x1, y1)
        results_window = warped[y0:y1, x0:x1]
        strip_rect = strip_bounds(results_window)

    return _measure_strip(results_window, warped, quad, window_bounds, strip_rect, bands)


def _measure_strip(results_window, warped, quad, window_bounds, strip_rect, bands=None):
    """
    Shared measurement core for both the detection and calibration paths.
    """
    if bands is not None:
        test_frac, control_frac = bands
    else:
        test_frac, control_frac = EXPECTED_T_FRAC, EXPECTED_C_FRAC

    sx0, sy0, sx1, sy1 = strip_rect
    # Copy: the result must stay valid even if the caller later draws on `warped`
    strip_roi = results_window[sy0:sy1, sx0:sx1].copy()

    # Detection runs on the normalised profile; density on the raw one
    raw_profile = raw_redness_profile(strip_roi)
    scale = profile_scale(raw_profile)
    profile = raw_profile / scale

    candidates = filter_band_candidates(profile)

    if bands is not None:
        # Calibrated positions known: snap to the tallest peak near each, so a
        # broad band is not lost to the width filter.
        t_idx, c_idx = snap_t_c(profile, test_frac, control_frac)
    else:
        t_idx, c_idx = pick_t_c_from_peaks(profile, candidates, test_frac, control_frac)

    # Each band excludes the other from its noise/baseline, so two strong bands
    # do not deflate each other's SNR.
    t_strength, t_snr = band_signal_snr(profile, t_idx, exclude=(c_idx,))
    c_strength, c_snr = band_signal_snr(profile, c_idx, exclude=(t_idx,))

    t_area = band_area(raw_profile, t_idx, exclude=(c_idx,))
    c_area = band_area(raw_profile, c_idx, exclude=(t_idx,))

    t_peak_a = band_peak_height(raw_profile, t_idx, exclude=(c_idx,))
    c_peak_a = band_peak_height(raw_profile, c_idx, exclude=(t_idx,))

    control_present = c_snr >= CONTROL_SNR_THRESHOLD
    test_present = t_snr >= TEST_SNR_THRESHOLD and control_present

    if c_strength > 1e-6:
        tc_ratio = t_strength / c_strength
    else:
        tc_ratio = 0.0

    # Density ratio: normalising T by C absorbs run-to-run variation in
    # sample volume and flow rate, so this is the load-comparable figure.
    if c_area > 1e-6:
        tc_area_ratio = t_area / c_area
    else:
        tc_area_ratio = 0.0

    return FrameResult(
        quad=quad,
        warped=warped,
        window_bounds=window_bounds,
        strip_rect=strip_rect,
        strip_roi=strip_roi,
        profile=profile,
        candidates=candidates,
        test=BandReading(t_idx, t_strength, t_snr, test_present, t_area, t_peak_a),
        control=BandReading(c_idx, c_strength, c_snr, control_present, c_area, c_peak_a),
        tc_ratio=tc_ratio,
        raw_profile=raw_profile,
        scale=scale,
        tc_area_ratio=tc_area_ratio
    )


class StabilityTracker:
    """
    A band counts as stable once it has been detected in STABILITY_VOTES of
    the last STABILITY_WINDOW frames in which a cassette was visible.
    """

    def __init__(self, window=STABILITY_WINDOW, votes=STABILITY_VOTES):
        self.votes = votes
        self.recent_test = deque(maxlen=window)
        self.recent_control = deque(maxlen=window)

    def update(self, result):
        self.recent_test.append(result.test.present)
        self.recent_control.append(result.control.present)

    @property
    def stable_test(self):
        return sum(self.recent_test) >= self.votes

    @property
    def stable_control(self):
        return sum(self.recent_control) >= self.votes
