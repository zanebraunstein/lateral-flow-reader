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


def test_snap_to_peak_finds_the_tallest_local_max():
    n = 200
    xs = np.arange(n)
    prof = 6.0 * np.exp(-((xs - 60) / 18.0) ** 2)      # tall peak at 60
    prof += 2.0 * np.exp(-((xs - 95) / 3.0) ** 2)      # smaller bump at 95

    assert abs(strip.snap_to_peak(prof, 0.30, 0.30) - 60) <= 2


def test_snap_to_peak_locates_a_broad_band_the_width_filter_drops():
    """
    The real bug: a broad band exceeds MAX_BAND_WIDTH so it never becomes a
    candidate, and the candidate picker lands on a narrow shoulder bump.
    Snapping ignores width and finds the broad peak.
    """
    n = 200
    xs = np.arange(n)
    prof = 5.0 * np.exp(-((xs - 60) / 28.0) ** 2)      # broad -> width > MAX
    prof += 3.0 * np.exp(-((xs - 88) / 2.5) ** 2)      # narrow bump nearby

    candidates = strip.filter_band_candidates(prof)
    picked = strip.pick_peak_near(prof, candidates, 0.30, 0.30)
    snapped = strip.snap_to_peak(prof, 0.30, 0.30)

    # the broad peak is not among the width-filtered candidates...
    assert all(abs(c - 60) > 8 for c in candidates)
    # ...so the candidate picker misses it, but snapping lands on it
    assert picked is None or abs(picked - 60) > 8
    assert abs(snapped - 60) <= 3


def test_snap_to_peak_returns_none_on_a_flat_region():
    assert strip.snap_to_peak(np.zeros(200), 0.5, 0.1) is None


@pytest.mark.parametrize("expected_frac", [0.0, 0.02, 0.5, 0.98, 1.0])
def test_snap_to_peak_handles_positions_at_the_edges(expected_frac):
    """A window running off either end must not index out of bounds."""
    prof = np.random.default_rng(0).normal(0, 1, 178)
    strip.snap_to_peak(prof, expected_frac, 0.18)     # must not raise


def test_snap_t_c_skips_a_missing_test_position():
    xs = np.arange(200)
    prof = 5.0 * np.exp(-((xs - 150) / 5.0) ** 2)

    t_idx, c_idx = strip.snap_t_c(prof, None, 0.75)

    assert t_idx is None
    assert abs(c_idx - 150) <= 3


def test_calibrated_path_uses_snapping_not_the_candidate_picker(monkeypatch):
    """
    The fix: with calibrated positions, analyze() must locate bands by snapping
    to the peak, not via the width-filtered candidate picker.
    """
    called = []
    real_snap = strip.snap_t_c
    monkeypatch.setattr(
        strip, "snap_t_c",
        lambda *a, **k: (called.append("snap"), real_snap(*a, **k))[1]
    )
    monkeypatch.setattr(
        strip, "pick_t_c_from_peaks",
        lambda *a, **k: called.append("pick") or (None, None)
    )

    frame, quad = synth.window_scene_at(control_frac=0.20, test_frac=0.80)
    strip.analyze(frame, window_quad=quad, bands=(0.80, 0.20))

    assert "snap" in called
    assert "pick" not in called


def test_detection_path_still_uses_the_candidate_picker(monkeypatch):
    """The full-cassette path (no calibration) is unchanged."""
    called = []
    monkeypatch.setattr(
        strip, "pick_t_c_from_peaks",
        lambda *a, **k: called.append("pick") or (None, None)
    )
    monkeypatch.setattr(strip, "snap_t_c", lambda *a, **k: called.append("snap") or (None, None))

    strip.analyze(synth.cassette_frame(t_amp=50))

    assert "pick" in called
    assert "snap" not in called


def test_calibrated_path_finds_broad_bands():
    """Calibrated snapping locates both bands at their positions."""
    frame, quad = synth.window_scene_at(
        control_frac=0.20, test_frac=0.80, control_amp=90, test_amp=90, band_half=22
    )
    result = strip.analyze(frame, window_quad=quad, bands=(0.80, 0.20))

    n = len(result.profile)
    assert result.test.idx / n == pytest.approx(0.80, abs=0.05)
    assert result.control.idx / n == pytest.approx(0.20, abs=0.05)


def test_dominant_two_bands_returns_peaks_left_to_right():
    # Make the RIGHT band the stronger one, so strength order (right, left)
    # differs from position order -- only a position sort gives left-to-right.
    frame, quad = synth.window_scene_at(
        control_frac=0.20, test_frac=0.80, control_amp=45, test_amp=75
    )
    result = strip.analyze(frame, window_quad=quad, bands=(0.80, 0.20))

    found = strip.dominant_two_bands(result.profile, result.candidates)

    assert len(found) == 2
    assert found[0] < found[1]     # ordered by position, not strength
    n = len(result.profile)
    assert found[0] / n == pytest.approx(0.20, abs=0.05)
    assert found[1] / n == pytest.approx(0.80, abs=0.05)


def test_learn_bands_assigns_by_control_side():
    """
    The same two bands must map to opposite roles depending on which side the
    control is on -- this is what makes either cassette orientation work.
    """
    # Strong band left (0.20), weaker band right (0.80)
    frame, quad = synth.window_scene_at(
        control_frac=0.20, test_frac=0.80, control_amp=75, test_amp=45
    )
    probe = strip.analyze(frame, window_quad=quad)

    control_frac, test_frac = calib.learn_bands(probe.profile, "left")
    assert control_frac == pytest.approx(0.20, abs=0.05)
    assert test_frac == pytest.approx(0.80, abs=0.05)

    control_frac, test_frac = calib.learn_bands(probe.profile, "right")
    assert control_frac == pytest.approx(0.80, abs=0.05)
    assert test_frac == pytest.approx(0.20, abs=0.05)


def test_learn_bands_finds_a_band_near_the_strip_edge():
    """
    A control line close to the window edge (as on a pregnancy cassette) must
    still be found -- it was being dropped by an over-aggressive edge exclusion.
    """
    n = 200
    xs = np.arange(n)
    prof = 5.0 * np.exp(-((xs - 16) / 4.0) ** 2)       # control near left edge (~0.08)
    prof += 3.0 * np.exp(-((xs - 150) / 4.0) ** 2)     # test on the right

    control_frac, test_frac = calib.learn_bands(prof, "left")

    assert control_frac == pytest.approx(16 / n, abs=0.03)
    assert test_frac == pytest.approx(150 / n, abs=0.03)


def test_learn_bands_finds_a_broad_band_the_width_filter_drops():
    """
    Calibration must learn a broad band's position, even though it would be
    dropped from the width-filtered candidate list -- the bug behind the left
    line landing between the peaks.
    """
    n = 200
    xs = np.arange(n)
    prof = 4.0 * np.exp(-((xs - 150) / 4.0) ** 2)      # narrow control, right
    prof += 3.0 * np.exp(-((xs - 50) / 30.0) ** 2)     # broad test, left (width > MAX)

    # the broad left band is absent from the width-filtered candidates
    assert all(abs(c - 50) > 8 for c in strip.filter_band_candidates(prof))

    control_frac, test_frac = calib.learn_bands(prof, "right")

    assert control_frac == pytest.approx(150 / n, abs=0.03)
    assert test_frac == pytest.approx(50 / n, abs=0.03)


def test_dominant_two_bands_respects_separation():
    """Two peaks too close collapse to one -- they cannot be a real T/C pair."""
    frame, quad = synth.window_scene_at(control_frac=0.48, test_frac=0.52)
    result = strip.analyze(frame, window_quad=quad)

    assert len(strip.dominant_two_bands(result.profile, result.candidates)) == 1


def test_reversed_orientation_control_on_left():
    """
    A cassette with control on the LEFT and test on the RIGHT -- the opposite
    of the module defaults. Learning positions from the cassette and searching
    near them must put control and test on the correct bands.

    This is the real-cassette case that motivated learned band positions.
    """
    # Control (strong) left at 0.20, test (weaker) right at 0.80
    frame, quad = synth.window_scene_at(
        control_frac=0.20, test_frac=0.80, control_amp=70, test_amp=45
    )

    # control on the left -> control_frac < test_frac
    result = strip.analyze(frame, window_quad=quad, bands=(0.80, 0.20))

    assert result.control.present
    assert result.test.present

    n = len(result.profile)
    assert result.control.idx / n == pytest.approx(0.20, abs=0.05)
    assert result.test.idx / n == pytest.approx(0.80, abs=0.05)
    # the control band is the stronger one here; density is measured on test
    assert result.control.peak_a > result.test.peak_a


def test_learned_bands_survive_calibration_round_trip(tmp_path):
    """
    calibrate.py stores learned positions; main.py loads them via .bands.
    """
    cal = calib.from_points(
        [(500, 300), (1400, 330), (1380, 700), (520, 670)],
        (1920, 1080),
        control_frac=0.20, test_frac=0.80,
    )
    cal.save(str(tmp_path / "c.json"))
    loaded = calib.load(str(tmp_path / "c.json"))

    assert loaded.bands == (0.80, 0.20)     # (test_frac, control_frac)


def test_bands_is_none_when_positions_absent():
    cal = calib.from_points([(0, 0), (1, 0), (1, 1), (0, 1)], (10, 10))
    assert cal.bands is None


def test_none_test_frac_means_no_test_band():
    """A calibration that only learned the control searches for no test line."""
    frame, quad = synth.window_scene(t_amp=50, c_amp=70)

    result = strip.analyze(frame, window_quad=quad, bands=(None, strip.EXPECTED_C_FRAC))

    assert result.control.present
    assert result.test.idx is None


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


def test_calibration_and_detection_both_detect_the_same_cassette(tmp_path):
    """
    Both paths detect the same bands on a normal cassette. They analyse
    different-width strips by design (detection sub-crops the window, the
    calibrated path uses the full marked box), so band indices are not directly
    comparable -- the shared invariant is that both find control and test.
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
    assert detected.control.present and detected.test.present
    assert calibrated.control.present and calibrated.test.present
