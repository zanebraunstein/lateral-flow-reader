import numpy as np
import pytest

import analysis
import synth


ONSET = 180.0
RISE_TO = 600.0
LEVEL = 200.0
TRUE_RATE = LEVEL / (RISE_TO - ONSET)


@pytest.fixture
def linear_run(tmp_path):
    """A run whose positivity time, density and rate are known exactly."""
    path = str(tmp_path / "linear")
    synth.make_run_npz(path, test_from=ONSET, rise_to=RISE_TO, plateau_area=LEVEL)

    return path


def test_recovers_time_to_positivity(linear_run):
    result = analysis.analyze_run(linear_run)

    assert result["positive"]
    assert result["time_to_positivity_s"] == pytest.approx(ONSET, abs=1.0)

    # the latch necessarily lags the onset it is derived from
    assert result["confirmed_at_s"] >= result["time_to_positivity_s"]


def test_recovers_density(linear_run):
    density = analysis.analyze_run(linear_run)["density"]

    assert density["test_area_a"] == pytest.approx(LEVEL, rel=0.02)
    assert density["tc_area_ratio"] == pytest.approx(1.0, rel=0.02)


def test_recovers_rate_of_change(linear_run):
    rate = analysis.analyze_run(linear_run)["rate"]

    assert rate["test_area_per_s"] == pytest.approx(TRUE_RATE, rel=0.02)
    assert rate["test_area_r2"] == pytest.approx(1.0, abs=1e-3)

    # the two independent estimates should agree on a linear rise
    assert rate["endpoint_area_per_s"] == pytest.approx(TRUE_RATE, rel=0.02)


def test_plateau_is_found_near_the_true_knee(linear_run):
    result = analysis.analyze_run(linear_run)
    reached = result["density"]["plateau_reached_s"]

    expected = ONSET + analysis.PLATEAU_FRAC * (RISE_TO - ONSET)

    assert result["rate"]["plateaued"]
    assert reached == pytest.approx(expected, abs=analysis.BIN_SECONDS + 1)


@pytest.mark.parametrize("duration,test_from", [
    (900, 60),     # normal, plateaus
    (200, 60),     # stopped while still rising
    (75, 60),      # stopped soon after onset
    (200, 185),    # onset late in a short run
])
def test_detected_onset_always_yields_a_rate(tmp_path, duration, test_from):
    """
    Whenever the positivity onset is detected, a rate must be reported -- the
    onset is the start of the rate interval, extended to the run's end if the
    line has not plateaued.
    """
    path = str(tmp_path / "r")
    synth.make_run_npz(path, duration=duration, test_from=test_from,
                       rise_to=600, plateau_area=200)

    result = analysis.analyze_run(path)

    assert result["positive"]                              # onset detected
    assert result["time_to_positivity_s"] is not None
    assert result["rate"]["test_area_per_s"] is not None   # ...so a rate exists
    assert result["rate"]["interval_s"][0] == pytest.approx(test_from, abs=2)


def test_run_stopped_before_plateau_still_gives_a_rate(tmp_path):
    """
    A run stopped while the line is still rising must still yield a rate,
    measured over onset -> end of run, flagged as not plateaued.
    """
    path = str(tmp_path / "early")
    # would plateau at 600 s, but the run is cut off at 200 s
    synth.make_run_npz(path, duration=200, test_from=60, rise_to=600, plateau_area=200)

    result = analysis.analyze_run(path)
    rate = result["rate"]

    assert not rate["plateaued"]
    assert result["density"]["plateau_reached_s"] is None
    assert rate["test_area_per_s"] == pytest.approx(200 / (600 - 60), rel=0.05)
    # the interval runs to the end of the (short) run
    assert rate["interval_s"][1] == pytest.approx(200, abs=analysis.BIN_SECONDS + 1)

    assert "still rising" in analysis.format_report(result)


def test_test_line_rgb_picks_the_darkest_frame():
    data = {
        "test_present": np.array([1, 1, 1]),
        "test_r": np.array([200, 210, 190]),
        "test_g": np.array([50, 40, 60]),
        "test_b": np.array([50, 40, 60]),
        "test_area": np.array([10.0, 30.0, 20.0]),   # darkest = frame 1 (max area)
    }

    assert analysis.test_line_rgb(data) == (210, 40, 40)


def test_test_line_rgb_none_when_colour_absent():
    # older run without the RGB columns
    data = {"test_present": np.array([1]), "test_area": np.array([10.0])}

    assert analysis.test_line_rgb(data) is None


def test_report_still_works_on_runs_without_rgb(tmp_path):
    path = str(tmp_path / "r")
    synth.make_run_npz(path, test_from=180)          # make_run_npz stores no RGB

    result = analysis.analyze_run(path)

    assert result["density"]["test_rgb"] is None
    assert "T line RGB      --" in analysis.format_report(result)


def test_negative_run_reports_nothing_invented(tmp_path):
    path = str(tmp_path / "negative")
    synth.make_run_npz(path, test_from=None)

    result = analysis.analyze_run(path)

    assert result["valid"]
    assert not result["positive"]
    assert result["time_to_positivity_s"] is None
    assert result["rate"]["test_area_per_s"] is None


def test_run_without_control_is_invalid_and_not_positive(tmp_path):
    """
    A cassette whose control never develops is invalid; a test line on it must
    not be reported as a positive result.
    """
    path = str(tmp_path / "invalid")
    synth.make_run_npz(path, control_from=1e9, test_from=200.0)

    result = analysis.analyze_run(path)

    assert not result["valid"]
    assert not result["positive"]


def test_nonlinear_rise_is_flagged_by_r2(tmp_path):
    path = str(tmp_path / "expo")
    synth.make_run_npz(path, test_from=ONSET, rise_to=RISE_TO, kind="expo", noise=3.0, seed=5)

    result = analysis.analyze_run(path)

    assert result["rate"]["test_area_per_s"] is not None
    assert result["rate"]["test_area_r2"] < 0.95


def test_thresholds_can_be_changed_offline(linear_run):
    """
    The point of recording profiles: re-score a spent run at a new threshold.
    """
    assert analysis.analyze_run(linear_run, test_snr=5.0)["positive"]
    assert not analysis.analyze_run(linear_run, test_snr=13.0)["positive"]


def test_empty_run_is_reported_not_crashed(tmp_path):
    path = str(tmp_path / "empty")
    synth.make_run_npz(path, duration=0.0)

    result = analysis.analyze_run(path)

    assert result["frames"] == 0
    assert "error" in result


def test_report_renders_for_every_outcome(tmp_path):
    for name, kwargs in (
        ("pos", {"test_from": ONSET}),
        ("neg", {"test_from": None}),
        ("bad", {"control_from": 1e9, "test_from": 200.0}),
    ):
        path = str(tmp_path / name)
        synth.make_run_npz(path, **kwargs)

        text = analysis.format_report(analysis.analyze_run(path))

        assert "Time to positivity" in text
        assert "nan" not in text.lower()


def test_binning_reduces_to_a_uniform_grid():
    t = np.array([0.0, 0.1, 0.2, 5.1, 5.2, 10.4])
    y = np.array([1.0, 3.0, 2.0, 10.0, 12.0, 20.0])

    tb, yb = analysis.bin_series(t, y, bin_s=5.0)

    assert len(tb) == 3
    assert yb[0] == pytest.approx(2.0)     # median of 1, 3, 2
    assert yb[1] == pytest.approx(11.0)    # median of 10, 12


def test_latch_backdates_to_first_detection():
    t = np.arange(20, dtype=float)
    present = np.zeros(20, dtype=bool)
    present[5:] = True

    onset, confirmed = analysis.latch_time(t, present, window=10, votes=7)

    assert onset == 5.0
    assert confirmed == 11.0               # 7th consecutive detection
