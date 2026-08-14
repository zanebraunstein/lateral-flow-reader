import os
import xml.dom.minidom as minidom

import plot_svg
import synth


def make_run(tmp_path, **kwargs):
    run_dir = str(tmp_path / "runs" / "r")
    synth.make_run_npz(run_dir, **kwargs)
    return run_dir


def test_build_svg_is_well_formed_xml(tmp_path):
    run_dir = make_run(tmp_path, test_from=180, rise_to=600)

    svg, result = plot_svg.build_svg(run_dir)

    assert svg is not None
    minidom.parseString(svg)          # raises if malformed
    assert svg.lstrip().startswith("<svg")
    assert result["positive"]


def test_build_svg_draws_both_series_per_panel(tmp_path):
    run_dir = make_run(tmp_path, test_from=180)

    svg, _ = plot_svg.build_svg(run_dir)

    # two panels x two series = four polylines
    assert svg.count("<polyline") == 4


def test_build_svg_marks_positivity_and_rate(tmp_path):
    run_dir = make_run(tmp_path, test_from=180, rise_to=600)

    svg, _ = plot_svg.build_svg(run_dir)

    assert "TTP" in svg
    assert "plateau" in svg
    assert "test rate" in svg


def test_build_svg_negative_run_has_no_ttp(tmp_path):
    run_dir = make_run(tmp_path, test_from=None)

    svg, result = plot_svg.build_svg(run_dir)

    assert svg is not None
    assert not result["positive"]
    assert "TTP" not in svg


def test_build_svg_empty_run_returns_none(tmp_path):
    run_dir = make_run(tmp_path, duration=0.0)

    svg, result = plot_svg.build_svg(run_dir)

    assert svg is None
    assert "error" in result


def test_main_writes_svg(tmp_path):
    run_dir = make_run(tmp_path, test_from=180)

    assert plot_svg.main([run_dir]) == 0

    out = os.path.join(run_dir, "plot.svg")
    assert os.path.getsize(out) > 0
    minidom.parse(out)


def test_svg_needs_no_matplotlib(monkeypatch):
    """
    The whole point: plotting works without matplotlib installed. Hide it and
    confirm build_svg still imports and runs.
    """
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ImportError("matplotlib blocked for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)

    import importlib
    importlib.reload(plot_svg)          # re-import under the block

    assert plot_svg is not None
