"""
End-to-end: the capture loop against a stubbed camera, then analysis of what
it recorded. Covers the wiring between modules that unit tests miss.
"""

import csv
import os

import analysis
import recorder
import synth


def run_loop(monkeypatch, tmp_path, fake_camera, frames, duration=1.5, **recorder_kwargs):
    import main

    fake_camera.frames = frames

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main, "RUN_DURATION_S", duration)

    if recorder_kwargs:
        monkeypatch.setattr(
            main.recorder.RunRecorder.__init__,
            "__defaults__",
            (recorder_kwargs.get("image_interval_s", 0.5),
             recorder_kwargs.get("autosave_every", 50))
        )

    main.main()

    runs = sorted(os.listdir("runs"))

    return os.path.join(str(tmp_path), "runs", runs[-1])


def developing_run(n=60, appears_at=20):
    """Control present throughout; test line appears partway and darkens."""
    return [
        synth.cassette_frame(
            t_amp=0 if i < appears_at else min(60, (i - appears_at) * 4),
            c_amp=70
        )
        for i in range(n)
    ]


def test_run_records_csv_and_profiles(monkeypatch, tmp_path, fake_camera, headless):
    run_dir = run_loop(monkeypatch, tmp_path, fake_camera, developing_run())

    assert sorted(os.listdir(run_dir)) == ["frames", "profiles.npz", "signal_log.csv"]

    with open(os.path.join(run_dir, "signal_log.csv")) as handle:
        rows = list(csv.DictReader(handle))

    data = recorder.load_run(run_dir)

    assert len(rows) == len(data["time_seconds"])
    assert len(rows) > 0

    # the two records are written by different code paths; they must agree
    for i, row in enumerate(rows):
        for field in ("test_area", "control_area", "tc_area_ratio", "profile_scale"):
            assert float(row[field]) == float(f"{data[field][i]:.4f}")


def test_loop_stops_itself_without_a_keypress(monkeypatch, tmp_path, fake_camera, headless):
    """
    waitKey never returns 'q' in these tests, so only the duration can end it.
    """
    run_dir = run_loop(monkeypatch, tmp_path, fake_camera, developing_run(), duration=1.0)

    times = recorder.load_run(run_dir)["time_seconds"]

    assert times[-1] <= 1.0 + 0.5


def test_display_windows_are_drawn(monkeypatch, tmp_path, fake_camera, headless):
    run_loop(monkeypatch, tmp_path, fake_camera, developing_run())

    assert "Lateral Flow Reader" in headless
    assert "Signal Profile" in headless


def test_recorded_run_analyses_end_to_end(monkeypatch, tmp_path, fake_camera, headless):
    run_dir = run_loop(monkeypatch, tmp_path, fake_camera, developing_run())

    result = analysis.analyze_run(run_dir, bin_s=0.05)

    assert result["valid"], "control line should be detected throughout"
    assert result["positive"], "test line should be detected as it develops"
    assert result["density"]["test_area_a"] > 0
    assert result["time_to_positivity_s"] is not None

    assert "Time to positivity" in analysis.format_report(result)


def test_run_without_a_cassette_records_nothing(monkeypatch, tmp_path, fake_camera, headless):
    run_dir = run_loop(
        monkeypatch, tmp_path, fake_camera, [synth.blank_frame()] * 10, duration=0.5
    )

    with open(os.path.join(run_dir, "signal_log.csv")) as handle:
        rows = list(csv.DictReader(handle))

    assert rows == []
    assert not os.path.exists(os.path.join(run_dir, "profiles.npz"))


def test_camera_is_released_when_the_loop_raises(monkeypatch, tmp_path, fake_camera, headless):
    """
    The cleanup path exists so a crash cannot leave the camera held.
    """
    import main

    fake_camera.frames = developing_run(n=5)
    monkeypatch.chdir(tmp_path)

    cameras = []
    original = main.start_camera

    def tracking_start():
        camera = original()
        cameras.append(camera)
        return camera

    monkeypatch.setattr(main, "start_camera", tracking_start)
    monkeypatch.setattr(main.strip, "analyze", lambda frame: 1 / 0)

    try:
        main.main()
    except ZeroDivisionError:
        pass

    assert cameras and cameras[0].stopped, "camera was not stopped on failure"
