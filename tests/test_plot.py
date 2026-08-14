import os

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import plot          # noqa: E402
import synth         # noqa: E402


def make_run(tmp_path, **kwargs):
    run_dir = str(tmp_path / "runs" / "r")
    synth.make_run_npz(run_dir, **kwargs)
    return run_dir


def test_build_figure_for_positive_run(tmp_path):
    run_dir = make_run(tmp_path, test_from=180, rise_to=600)

    fig, result = plot.build_figure(run_dir)

    assert fig is not None
    assert result["positive"]
    assert len(fig.axes) == 2          # density + SNR panels


def test_build_figure_handles_negative_run(tmp_path):
    run_dir = make_run(tmp_path, test_from=None)

    fig, result = plot.build_figure(run_dir)

    assert fig is not None              # a negative run still plots
    assert not result["positive"]


def test_build_figure_empty_run_returns_none(tmp_path):
    run_dir = make_run(tmp_path, duration=0.0)

    fig, result = plot.build_figure(run_dir)

    assert fig is None
    assert "error" in result


def test_main_writes_a_png(tmp_path):
    run_dir = make_run(tmp_path, test_from=180)
    out = str(tmp_path / "out.png")

    assert plot.main([run_dir, "--out", out]) == 0
    assert os.path.getsize(out) > 0


def test_main_defaults_output_into_the_run_dir(tmp_path):
    run_dir = make_run(tmp_path, test_from=180)

    assert plot.main([run_dir]) == 0
    assert os.path.isfile(os.path.join(run_dir, "plot.png"))
