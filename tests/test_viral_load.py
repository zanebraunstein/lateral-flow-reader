import numpy as np
import pytest

import viral_load


def test_fit_recovers_a_known_power_law():
    # conc = 500 * density^1.5, exactly
    dens = np.array([0.1, 0.2, 0.3, 0.5, 0.8, 1.0])
    conc = 500 * dens ** 1.5

    calib = viral_load.fit(dens, conc)

    assert calib.b == pytest.approx(1.5, abs=1e-6)
    assert np.exp(calib.a) == pytest.approx(500, rel=1e-6)
    assert calib.resid_sd == pytest.approx(0.0, abs=1e-9)

    pred = viral_load.predict(0.4, calib)
    assert pred["estimate"] == pytest.approx(500 * 0.4 ** 1.5, rel=1e-6)


def test_predict_is_none_without_a_prediction():
    calib = viral_load.fit([0.1, 0.2, 0.5], [10, 40, 200])

    assert viral_load.predict(0.3, None) is None      # no calibration
    assert viral_load.predict(None, calib) is None     # no density (no band)
    assert viral_load.predict(0.0, calib) is None      # zero density
    assert viral_load.predict(-1.0, calib) is None     # nonsense density


def test_extrapolation_is_flagged_on_both_sides():
    calib = viral_load.fit([0.2, 0.4, 0.8], [40, 160, 640])

    below = viral_load.predict(0.1, calib)
    above = viral_load.predict(1.2, calib)
    inside = viral_load.predict(0.4, calib)

    assert below["extrapolated"] and below["extrapolation_side"] == "below"
    assert above["extrapolated"] and above["extrapolation_side"] == "above"
    assert not inside["extrapolated"]


def test_faint_density_is_low_confidence():
    # faint_below = 0.2 + 0.2*(1.0-0.2) = 0.36
    calib = viral_load.fit([0.2, 0.4, 0.6, 1.0], [40, 160, 360, 1000])

    assert viral_load.predict(0.25, calib)["low_confidence"]
    assert not viral_load.predict(0.8, calib)["low_confidence"]


def test_uncertainty_band_is_multiplicative_and_ordered():
    calib = viral_load.fit([0.1, 0.2, 0.4, 0.8], [20, 55, 140, 600.0])

    p = viral_load.predict(0.3, calib)

    # symmetric in log space: est^2 == lo * hi
    assert p["lo68"] * p["hi68"] == pytest.approx(p["estimate"] ** 2, rel=1e-6)
    assert p["lo95"] < p["lo68"] < p["estimate"] < p["hi68"] < p["hi95"]


def test_save_and_load_roundtrip(tmp_path):
    calib = viral_load.fit([0.1, 0.3, 0.9], [15, 120, 700], units="ng/mL")

    path = str(tmp_path / "vl.json")
    calib.save(path)
    back = viral_load.load(path)

    assert back.a == pytest.approx(calib.a)
    assert back.b == pytest.approx(calib.b)
    assert back.units == "ng/mL"
    assert viral_load.load(str(tmp_path / "missing.json")) is None


def test_fit_needs_at_least_three_points():
    with pytest.raises(ValueError):
        viral_load.fit([0.1, 0.2], [10, 20])


def test_format_prediction_reports_absence():
    assert "not available" in viral_load.format_prediction(None)[0]


def test_format_prediction_notes_extrapolation_as_a_bound():
    calib = viral_load.fit([0.2, 0.4, 0.8], [40, 160, 640])

    text = "\n".join(viral_load.format_prediction(viral_load.predict(2.0, calib)))

    assert "upper bound" in text
