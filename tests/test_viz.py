import numpy as np

import strip
import synth
import viz


EXPECTED_WINDOWS = [
    "Results Window",
    "Strip ROI",
    "Signal Profile",
    "Warped Cassette",
    "Lateral Flow Reader",
]


def _readout_colors(disp):
    """Colours present in the two-line readout region (top-left)."""
    region = disp[65:135, :700]

    def has(color):
        return bool(np.any(np.all(region == color, axis=2)))

    return {
        "valid": has(viz.COLOR_CONTROL),
        "waiting": has(viz.COLOR_SEARCHING),
        "positive": has(viz.COLOR_TEST),
    }


def _render(stable_test, stable_control):
    frame = synth.cassette_frame(t_amp=50, c_amp=70)
    result = strip.analyze(frame)
    return viz.render(frame, result, stable_test, stable_control)["Lateral Flow Reader"]


def test_readout_shows_waiting_until_control_is_stable():
    colors = _readout_colors(_render(stable_test=False, stable_control=False))

    assert colors["waiting"]
    assert not colors["valid"]
    assert not colors["positive"]


def test_readout_shows_valid_but_not_positive_for_a_negative():
    colors = _readout_colors(_render(stable_test=False, stable_control=True))

    assert colors["valid"]
    assert not colors["waiting"]
    assert not colors["positive"]


def test_readout_shows_positive_only_when_test_is_stable():
    colors = _readout_colors(_render(stable_test=True, stable_control=True))

    assert colors["valid"]
    assert colors["positive"]


def test_render_produces_every_window():
    frame = synth.cassette_frame(t_amp=50)

    assert list(viz.render(frame, strip.analyze(frame))) == EXPECTED_WINDOWS


def test_render_without_cassette_shows_only_main_view():
    frame = synth.blank_frame()

    assert list(viz.render(frame, strip.analyze(frame))) == ["Lateral Flow Reader"]


def test_absent_band_draws_no_marker():
    """
    Snapping to a calibrated position may land on noise when a band is absent;
    the marker must not draw unless the band is actually present.
    """
    # Control present (left), no test line (right)
    frame, quad = synth.window_scene_at(
        control_frac=0.20, test_frac=0.80, control_amp=70, test_amp=0
    )
    result = strip.analyze(frame, window_quad=quad, bands=(0.80, 0.20))

    assert not result.test.present

    disp = viz.render(frame, result)["Lateral Flow Reader"]
    ys, xs = np.where(np.all(disp == viz.COLOR_TEST, axis=2))

    assert xs[ys > 250].size == 0     # no test marker in the window region


def test_render_does_not_touch_the_measurement():
    """
    Overlays were once drawn into the same buffer the strip was sliced from,
    so the reader partly measured its own annotations. Rendering must leave
    every measured array untouched.
    """
    frame = synth.cassette_frame(t_amp=50)
    result = strip.analyze(frame)

    before = {
        "frame": frame.copy(),
        "warped": result.warped.copy(),
        "strip_roi": result.strip_roi.copy(),
        "profile": result.profile.copy(),
        "raw_profile": result.raw_profile.copy(),
    }

    # Twice, so accumulating draws are caught as well as single ones
    viz.render(frame, result, True, False)
    viz.render(frame, result, False, True)

    after = {
        "frame": frame,
        "warped": result.warped,
        "strip_roi": result.strip_roi,
        "profile": result.profile,
        "raw_profile": result.raw_profile,
    }

    for name, original in before.items():
        assert np.array_equal(original, after[name]), f"render() mutated {name}"


def test_reanalysis_after_render_is_identical():
    """
    The end-to-end version of the same property: drawing on a frame must not
    change what the next measurement of it produces.
    """
    frame = synth.cassette_frame(t_amp=50)

    first = strip.analyze(frame)
    viz.render(frame, first)
    second = strip.analyze(frame)

    assert np.allclose(first.profile, second.profile)
    assert first.control.snr == second.control.snr
    assert first.test.area == second.test.area
