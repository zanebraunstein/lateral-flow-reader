import numpy as np
import pytest

import strip
import synth


def picks(image):
    profile = strip.redness_profile(image)
    candidates = strip.filter_band_candidates(profile)

    return profile, strip.pick_t_c_from_peaks(profile, candidates)


def test_finds_both_bands():
    profile, (t_idx, c_idx) = picks(
        synth.strip_image([(0.45, 45), (0.80, 70)])
    )
    n = len(profile)

    assert abs(t_idx - 0.45 * n) < 5
    assert abs(c_idx - 0.80 * n) < 5


def test_control_only_gives_no_test_band():
    _, (t_idx, c_idx) = picks(synth.strip_image([(0.80, 70)]))

    assert t_idx is None
    assert c_idx is not None


def test_blank_strip_finds_nothing():
    _, (t_idx, c_idx) = picks(synth.strip_image([]))

    assert t_idx is None
    assert c_idx is None


def test_snr_is_zero_for_missing_band():
    profile = strip.redness_profile(synth.strip_image([]))

    assert strip.band_signal_snr(profile, None) == (0.0, 0.0)
    assert strip.band_area(profile, None) == 0.0
    assert strip.band_peak_height(profile, None) == 0.0


def test_raw_profile_is_recoverable_from_normalised():
    """
    Density lives in raw a* units, but only the normalised profile is stored.
    This invariant is what makes `recorder.raw_profiles` correct.
    """
    image = synth.strip_image([(0.45, 45), (0.80, 70)])

    raw = strip.raw_redness_profile(image)
    scale = strip.profile_scale(raw)

    assert np.allclose(raw, strip.redness_profile(image) * scale, atol=1e-10)


def test_band_width_agrees_with_extent():
    profile = strip.redness_profile(synth.strip_image([(0.45, 45), (0.80, 70)]))

    for idx in strip.filter_band_candidates(profile):
        left, right = strip.band_extent(profile, idx)

        assert strip.band_width_peak(profile, idx) == right - left


@pytest.mark.parametrize("amp", [10, 30, 50, 70])
def test_analyze_matches_hand_composed_primitives(amp):
    """
    `analyze` is a convenience wrapper; it must not drift from the functions
    it composes.
    """
    frame = synth.cassette_frame(t_amp=amp)
    result = strip.analyze(frame)

    raw = strip.raw_redness_profile(result.strip_roi)
    profile = raw / strip.profile_scale(raw)

    candidates = strip.filter_band_candidates(profile)
    t_idx, c_idx = strip.pick_t_c_from_peaks(profile, candidates)

    assert result.candidates == candidates
    assert result.test.idx == t_idx
    assert result.control.idx == c_idx
    assert result.test.snr == strip.band_signal_snr(profile, t_idx)[1]
    assert result.test.area == strip.band_area(raw, t_idx)
    assert result.test.peak_a == strip.band_peak_height(raw, t_idx)


def test_analyze_returns_none_without_cassette():
    assert strip.analyze(synth.blank_frame()) is None


def test_lone_test_band_is_not_a_valid_control():
    """
    A cassette with a test line but no control line is INVALID. Reporting it
    as a valid negative is the dangerous direction to fail in.

    Asserted on the outcome rather than on which index gets assigned, so this
    keeps holding whichever way the promotion rule is resolved.
    """
    result = strip.analyze(synth.cassette_frame(t_amp=60, c_amp=0))

    assert not result.control.present, (
        f"band at index {result.control.idx} (the test position) was accepted "
        "as the control line"
    )
    assert not result.test.present


def _result(test_present, control_present=True):
    """Minimal FrameResult carrying just the detection flags."""
    return strip.FrameResult(
        quad=None,
        warped=None,
        window_bounds=None,
        strip_rect=None,
        strip_roi=None,
        profile=None,
        candidates=[],
        test=strip.BandReading(0, 0.0, 0.0, test_present),
        control=strip.BandReading(0, 0.0, 0.0, control_present),
        tc_ratio=0.0
    )


def test_stability_needs_repeated_detections():
    """
    A single noisy frame must not flip the reader to positive; that is the
    whole point of the vote window.
    """
    tracker = strip.StabilityTracker(window=10, votes=7)

    for _ in range(6):
        tracker.update(_result(True))
        assert not tracker.stable_test

    tracker.update(_result(True))
    assert tracker.stable_test


def test_stability_lapses_when_detections_stop():
    tracker = strip.StabilityTracker(window=10, votes=7)

    for _ in range(10):
        tracker.update(_result(True))
    assert tracker.stable_test

    # push the detections out of the window
    for _ in range(4):
        tracker.update(_result(False))

    assert not tracker.stable_test


def test_stability_tracks_test_and_control_separately():
    tracker = strip.StabilityTracker(window=10, votes=7)

    for _ in range(8):
        tracker.update(_result(test_present=False, control_present=True))

    assert tracker.stable_control
    assert not tracker.stable_test


def _density_table(amps):
    truth, area, peak, normalised = [], [], [], []

    for amp in amps:
        result = strip.analyze(synth.cassette_frame(t_amp=amp))

        truth.append(synth.true_band_excess(result))
        area.append(result.test.area)
        peak.append(result.test.peak_a)
        normalised.append(result.test.strength)

    return map(np.asarray, (truth, area, peak, normalised))


def test_density_tracks_dye_load():
    """
    Area and raw peak must be proportional to how much dye is on the band --
    this is what makes them usable as a viral load proxy.
    """
    truth, area, peak, _ = _density_table([10, 20, 30, 40, 50, 60, 70])

    for name, measured in (("area", area), ("peak", peak)):
        r2 = np.corrcoef(truth, measured)[0, 1] ** 2
        ratio = measured / truth
        spread = ratio.max() / ratio.min()

        assert r2 > 0.99, f"{name} R^2 only {r2:.4f}"
        assert spread < 1.3, f"{name} is {spread:.2f}x from proportional"


def test_normalised_strength_is_not_a_density_measure():
    """
    Guards the reason density is measured on the raw profile: the MAD divisor
    grows with the band itself, so normalised strength compresses the dose
    response. If this ever starts passing as proportional, the normalisation
    changed and the density path should be revisited.
    """
    truth, _, _, normalised = _density_table([10, 20, 30, 40, 50, 60, 70])

    ratio = normalised / truth
    spread = ratio.max() / ratio.min()

    assert spread > 1.5, (
        f"normalised strength is unexpectedly proportional ({spread:.2f}x); "
        "re-check whether density still needs the raw profile"
    )
