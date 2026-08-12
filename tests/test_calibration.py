import cv2 as cv
import numpy as np
import pytest

import calibration as calib
import strip
import synth
import viz


def test_warp_quad_returns_requested_size():
    frame, quad = synth.window_scene()

    warped = strip.warp_quad(frame, quad, strip.WINDOW_CANON_W, strip.WINDOW_CANON_H)

    assert warped.shape[:2] == (strip.WINDOW_CANON_H, strip.WINDOW_CANON_W)


def test_warp_cassette_still_uses_full_canonical_size():
    frame = synth.cassette_frame(t_amp=40)
    quad = strip.find_cassette_quad(frame)

    warped = strip.warp_cassette(frame, quad)

    assert warped.shape[:2] == (strip.CANON_H, strip.CANON_W)


def test_calibration_path_finds_bands():
    """
    Analyse from a window quad alone -- no cassette detection. The profile
    spans the strip, so the bands land at their expected strip fractions.
    """
    frame, quad = synth.window_scene(t_amp=50, c_amp=70)

    result = strip.analyze(frame, window_quad=quad)

    assert result is not None
    assert result.control.present
    assert result.test.present

    n = len(result.profile)
    assert result.test.idx / n == pytest.approx(strip.EXPECTED_T_FRAC, abs=0.05)
    assert result.control.idx / n == pytest.approx(strip.EXPECTED_C_FRAC, abs=0.05)


def test_calibration_measures_with_no_cassette_present():
    """
    The reason this path exists: window_scene draws only the results window,
    no surrounding cassette, yet the bands are still measured.
    """
    frame, quad = synth.window_scene(t_amp=50, c_amp=70)

    result = strip.analyze(frame, window_quad=quad)

    assert result is not None
    assert result.control.present
    assert result.test.present


def test_calibration_control_only_is_negative_not_invalid():
    frame, quad = synth.window_scene(t_amp=0, c_amp=70)

    result = strip.analyze(frame, window_quad=quad)

    assert result.control.present
    assert not result.test.present


def test_calibration_density_tracks_dye_load():
    """
    Density must behave the same on the calibration path as on detection.
    """
    quad = np.array([[500, 300], [1400, 330], [1380, 700], [520, 670]], dtype=np.float32)

    areas = []
    for amp in (20, 40, 60):
        frame, _ = synth.window_scene(t_amp=amp, c_amp=70, quad=quad)
        areas.append(strip.analyze(frame, window_quad=quad).test.area)

    assert areas[0] < areas[1] < areas[2]


def test_band_line_projects_to_the_real_band_location():
    """
    The projected T/C overlay must land on the actual band in the live frame.
    This only holds when the projection uses the window canvas size, not the
    full-cassette size -- so it guards viz passing the right dimensions.
    """
    frame, quad = synth.window_scene(t_amp=50, c_amp=70)
    result = strip.analyze(frame, window_quad=quad)

    x0, y0, x1, y1 = result.window_bounds
    sx0, sy0, sx1, sy1 = result.strip_rect
    canon_h, canon_w = result.warped.shape[:2]

    # Use the T line: it is cyan, which the green window outline does not
    # collide with, so it isolates cleanly by colour.
    canon_x = x0 + sx0 + result.test.idx
    mid_y = (y0 + y1) / 2

    # Where that window-canvas point truly sits in the frame: forward-warp
    # canvas -> quad.
    src = np.array(
        [[0, 0], [canon_w - 1, 0], [canon_w - 1, canon_h - 1], [0, canon_h - 1]],
        dtype=np.float32,
    )
    forward = cv.getPerspectiveTransform(src, quad.astype(np.float32))
    true_x = cv.perspectiveTransform(
        np.array([[[canon_x, mid_y]]], dtype=np.float32), forward
    ).reshape(2)[0]

    # Where viz actually draws it, read back off the rendered frame.
    disp = viz.render(frame, result)["Lateral Flow Reader"]
    ys, xs = np.where(np.all(disp == viz.COLOR_TEST, axis=2))
    band_xs = xs[ys > 250]          # exclude the HUD text near the top

    assert band_xs.size > 0
    assert abs(band_xs.mean() - true_x) < 15, (
        f"T line drawn at x~{band_xs.mean():.0f}, band really at x~{true_x:.0f}"
    )


def test_render_works_on_calibration_result():
    """
    viz must project band lines using the window canvas size, not the full
    cassette size, or the overlay lands in the wrong place / errors.
    """
    frame, quad = synth.window_scene(t_amp=50)
    result = strip.analyze(frame, window_quad=quad)

    windows = viz.render(frame, result)

    assert list(windows) == [
        "Results Window", "Strip ROI", "Signal Profile",
        "Warped Cassette", "Lateral Flow Reader",
    ]
    # warped is the window canvas, not the full cassette
    assert result.warped.shape[:2] == (strip.WINDOW_CANON_H, strip.WINDOW_CANON_W)


def test_calibration_save_load_round_trip(tmp_path):
    path = str(tmp_path / "calibration.json")

    points = [(1400, 330), (520, 670), (500, 300), (1380, 700)]  # unordered
    cal = calib.from_points(points, (1920, 1080))
    cal.save(path)

    loaded = calib.load(path)

    assert loaded is not None
    assert loaded.frame_size == (1920, 1080)
    assert np.allclose(loaded.window_quad, cal.window_quad)


def test_from_points_orders_corners():
    """Corners may be clicked in any order; storage is always tl, tr, br, bl."""
    points = [(1400, 330), (520, 670), (500, 300), (1380, 700)]
    cal = calib.from_points(points, (1920, 1080))

    tl, tr, br, bl = cal.window_quad
    assert tl[0] < tr[0] and bl[0] < br[0]     # left corners left of right
    assert tl[1] < bl[1] and tr[1] < br[1]     # top corners above bottom


def test_load_missing_returns_none(tmp_path):
    assert calib.load(str(tmp_path / "nope.json")) is None


def test_calibration_and_detection_agree_on_the_same_window(tmp_path):
    """
    Given a window quad equal to the results window the detector would use,
    both paths must measure the same bands -- the calibration path is not a
    different measurement, just a different way of locating the same strip.
    """
    frame = synth.cassette_frame(t_amp=50, c_amp=70)

    detected = strip.analyze(frame)
    assert detected is not None

    # Reconstruct, in frame coordinates, the window the detector used: the
    # results-window rectangle in canonical space, mapped back through the quad.
    x0, y0, x1, y1 = detected.window_bounds
    corners_canon = np.array(
        [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32
    ).reshape(-1, 1, 2)

    dst = np.array(
        [[0, 0], [strip.CANON_W - 1, 0],
         [strip.CANON_W - 1, strip.CANON_H - 1], [0, strip.CANON_H - 1]],
        dtype=np.float32,
    )
    inv = cv.getPerspectiveTransform(dst, detected.quad.astype(np.float32))
    window_quad = cv.perspectiveTransform(corners_canon, inv).reshape(-1, 2)

    calibrated = strip.analyze(frame, window_quad=window_quad)

    assert calibrated is not None
    assert calibrated.control.present == detected.control.present
    assert calibrated.test.present == detected.test.present
    # band positions agree to within a couple of samples
    assert abs(calibrated.control.idx - detected.control.idx) <= 3
