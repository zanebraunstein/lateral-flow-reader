import os

import numpy as np

import recorder
import strip
import synth


def record_frames(run_dir, amps, **kwargs):
    run = recorder.RunRecorder(run_dir, **kwargs)

    for i, amp in enumerate(amps):
        result = strip.analyze(synth.cassette_frame(t_amp=amp))
        run.add(i * 0.1, result)

    return run


def test_round_trip_preserves_readings(tmp_path):
    run_dir = str(tmp_path / "run")
    os.makedirs(os.path.join(run_dir, "frames"))

    amps = [0, 20, 40, 60]
    expected = [strip.analyze(synth.cassette_frame(t_amp=a)) for a in amps]

    run = recorder.RunRecorder(run_dir, image_interval_s=None)
    for i, result in enumerate(expected):
        run.add(i * 0.1, result)
    run.save()

    data = recorder.load_run(run_dir)

    assert data["profiles"].shape == (len(amps), len(expected[0].profile))
    assert not np.isnan(data["profiles"]).any()

    for i, result in enumerate(expected):
        assert data["test_area"][i] == result.test.area
        assert data["control_area"][i] == result.control.area
        assert data["tc_area_ratio"][i] == result.tc_area_ratio
        assert data["profile_scale"][i] == result.scale
        assert data["test_idx"][i] == (-1 if result.test.idx is None else result.test.idx)


def test_raw_profiles_recovers_physical_units(tmp_path):
    run_dir = str(tmp_path / "run")
    os.makedirs(os.path.join(run_dir, "frames"))

    result = strip.analyze(synth.cassette_frame(t_amp=50))

    run = recorder.RunRecorder(run_dir, image_interval_s=None)
    run.add(0.0, result)
    run.save()

    data = recorder.load_run(run_dir)
    raw = recorder.raw_profiles(data)

    assert np.allclose(raw[0], result.raw_profile, rtol=1e-5, atol=1e-4)

    # and the recovered profile must reproduce the recorded density
    area = strip.band_area(raw[0].astype(np.float64), int(data["test_idx"][0]))

    assert abs(area - data["test_area"][0]) < 1e-3


def test_missing_band_stored_as_negative_one(tmp_path):
    run_dir = str(tmp_path / "run")
    os.makedirs(os.path.join(run_dir, "frames"))

    result = strip.analyze(synth.cassette_frame(t_amp=0))
    assert result.test.idx is None

    run = recorder.RunRecorder(run_dir, image_interval_s=None)
    run.add(0.0, result)
    run.save()

    assert recorder.load_run(run_dir)["test_idx"][0] == -1


def test_each_run_gets_its_own_directory(tmp_path, monkeypatch):
    """
    Runs used to write to a fixed path, so restarting the reader destroyed the
    previous run -- and each run costs a cassette.
    """
    monkeypatch.chdir(tmp_path)

    first = recorder.create_run_dir()
    open(os.path.join(first, "signal_log.csv"), "w").write("first")

    monkeypatch.setattr(recorder.time, "strftime", lambda fmt: "20260101-000001")
    second = recorder.create_run_dir()

    assert first != second
    assert open(os.path.join(first, "signal_log.csv")).read() == "first"


def test_autosave_keeps_data_recoverable_without_a_final_save(tmp_path):
    """
    A hard kill skips save(); periodic autosaves bound how much is lost.
    """
    run_dir = str(tmp_path / "run")
    os.makedirs(os.path.join(run_dir, "frames"))

    run = record_frames(run_dir, [40] * 20, image_interval_s=None, autosave_every=7)

    recovered = recorder.load_run(run_dir)["time_seconds"]

    assert len(recovered) == 14           # last autosave at frame 14
    assert run.frame_count == 20


def test_images_are_decimated(tmp_path):
    run_dir = str(tmp_path / "run")
    os.makedirs(os.path.join(run_dir, "frames"))

    run = recorder.RunRecorder(run_dir, image_interval_s=0.5, autosave_every=None)

    result = strip.analyze(synth.cassette_frame(t_amp=40))
    for i in range(20):
        run.add(i * 0.1, result)          # 2.0 s of frames

    saved = os.listdir(os.path.join(run_dir, "frames"))

    assert run.frame_count == 20
    assert 3 <= len(saved) <= 5, f"expected ~4 images over 2 s, got {len(saved)}"


def test_recording_failure_does_not_abort_the_run(tmp_path):
    """
    Recording is best-effort: a write error must not kill a run that is still
    producing good CSV rows.
    """
    run_dir = str(tmp_path / "run")
    os.makedirs(os.path.join(run_dir, "frames"))

    run = recorder.RunRecorder(run_dir, image_interval_s=None, autosave_every=2)
    run.save = lambda: (_ for _ in ()).throw(OSError("disk full"))

    result = strip.analyze(synth.cassette_frame(t_amp=40))

    for i in range(6):
        run.add(i * 0.1, result)          # must not raise

    assert run.frame_count == 6
