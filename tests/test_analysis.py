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
    reached = analysis.analyze_run(linear_run)["density"]["plateau_reached_s"]

    expected = ONSET + analysis.PLATEAU_FRAC * (RISE_TO - ONSET)

    assert reached == pytest.approx(expected, abs=analysis.BIN_SECONDS + 1)


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
