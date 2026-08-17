import numpy as np
import pytest

import viral_load


def _calib(units=""):
    """A calibration whose density is an exact power law, rate/time noisier."""
    dens = np.array([0.1, 0.2, 0.3, 0.5, 0.8, 1.0])
    conc = 500 * dens ** 1.5                       # exact: density is tightest
    rng = np.random.default_rng(0)
    rate = (conc / 1000) * np.exp(rng.normal(0, 0.25, dens.size))   # rises, noisy
    time = (1e6 / conc) * np.exp(rng.normal(0, 0.5, dens.size))     # falls, noisier
    return viral_load.fit({"density": dens, "rate": rate, "time": time}, conc, units=units)


def test_fit_recovers_a_known_density_power_law():
    calib = _calib()
    d = calib.metrics["density"]

    assert d.b == pytest.approx(1.5, abs=1e-6)
    assert np.exp(d.a) == pytest.approx(500, rel=1e-6)
    assert d.resid_sd == pytest.approx(0.0, abs=1e-9)


def test_primary_is_the_tightest_metric():
    calib = _calib()
    assert calib.primary == "density"


def test_time_metric_has_a_negative_exponent():
    # time-to-positivity falls as concentration rises
    assert _calib().metrics["time"].b < 0


def test_predict_gives_one_estimate_per_metric():
    calib = _calib()

    preds = viral_load.predict({"density": 0.4, "rate": 0.15, "time": 40.0}, calib)

    assert set(preds) == {"density", "rate", "time"}
    assert all(preds[m] is not None for m in preds)
    assert preds["density"]["primary"] is True
    assert preds["rate"]["primary"] is False
    # density estimate recovers the exact law
    assert preds["density"]["estimate"] == pytest.approx(500 * 0.4 ** 1.5, rel=1e-6)


def test_predict_skips_a_missing_metric():
    calib = _calib()

    preds = viral_load.predict({"density": 0.4, "rate": None, "time": 40.0}, calib)

    assert preds["rate"] is None            # not measured this run
    assert preds["density"] is not None
    assert preds["time"] is not None


def test_predict_is_none_without_a_calibration():
    assert viral_load.predict({"density": 0.4}, None) is None


def test_a_metric_with_too_few_points_is_skipped():
    dens = np.array([0.1, 0.3, 0.6, 0.9])
    conc = 500 * dens ** 1.5
    rate = np.array([np.nan, np.nan, 0.2, np.nan])   # only one usable

    calib = viral_load.fit({"density": dens, "rate": rate}, conc)

    assert "density" in calib.metrics
    assert "rate" not in calib.metrics               # dropped
    assert viral_load.predict({"density": 0.4, "rate": 0.2}, calib)["rate"] is None


def test_fit_requires_density():
    with pytest.raises(ValueError):
        viral_load.fit({"rate": [0.1, 0.2, 0.3]}, [10, 40, 90])


def test_extrapolation_is_flagged():
    calib = _calib()   # density trained on 0.1 .. 1.0

    assert viral_load.predict({"density": 2.0}, calib)["density"]["extrapolated"]
    assert not viral_load.predict({"density": 0.5}, calib)["density"]["extrapolated"]


def test_uncertainty_band_is_multiplicative_and_ordered():
    calib = _calib()
    p = viral_load.predict({"rate": 0.15}, calib)["rate"]

    assert p["lo68"] * p["hi68"] == pytest.approx(p["estimate"] ** 2, rel=1e-6)
    assert p["lo95"] < p["lo68"] < p["estimate"] < p["hi68"] < p["hi95"]


def test_save_and_load_roundtrip(tmp_path):
    calib = _calib(units="ng/mL")
    path = str(tmp_path / "vl.json")
    calib.save(path)

    back = viral_load.load(path)
    assert back.units == "ng/mL"
    assert back.primary == "density"
    assert set(back.metrics) == set(calib.metrics)
    assert back.metrics["density"].b == pytest.approx(calib.metrics["density"].b)
    assert viral_load.load(str(tmp_path / "missing.json")) is None


def test_format_reports_absence():
    assert "not available" in viral_load.format_prediction(None)[0]
    assert "not available" in viral_load.format_prediction(
        {"density": None, "rate": None, "time": None})[0]


def test_format_flags_disagreement_between_metrics():
    calib = _calib()
    # density says ~high, rate/time forced low -> wide spread
    preds = viral_load.predict({"density": 0.9, "rate": 0.02, "time": 200.0}, calib)
    text = "\n".join(viral_load.format_prediction(preds))

    assert "disagree" in text
