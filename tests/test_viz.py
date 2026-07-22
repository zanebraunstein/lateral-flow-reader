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


def test_render_produces_every_window():
    frame = synth.cassette_frame(t_amp=50)

    assert list(viz.render(frame, strip.analyze(frame))) == EXPECTED_WINDOWS


def test_render_without_cassette_shows_only_main_view():
    frame = synth.blank_frame()

    assert list(viz.render(frame, strip.analyze(frame))) == ["Lateral Flow Reader"]


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
