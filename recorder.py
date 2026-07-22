"""
Per-run data capture.

An assay run is one-shot: the cassette is spent afterwards and the frames
cannot be recreated. This module writes everything needed to re-analyse a
run offline, so thresholds and ROI geometry can be re-tuned without
consuming another test.

Each run gets its own directory:

    runs/20260722-101530/
        signal_log.csv   per-frame measurements (written by main)
        profiles.npz     per-frame 1-D profiles + band readings
        frames/          decimated warped cassette images
"""

import os
import time

import cv2 as cv
import numpy as np


RUNS_DIR = "runs"

# Warped cassettes are ~860 kB each, so keep only a periodic sample
IMAGE_INTERVAL_S = 2.0

# Profiles are held in memory and rewritten periodically; the CSV is the
# per-frame durable record, this bounds how much profile data a hard kill loses
AUTOSAVE_EVERY = 200


def create_run_dir(base=RUNS_DIR):
    """
    Make a fresh timestamped directory for this run and return its path.
    """
    path = os.path.join(base, time.strftime("%Y%m%d-%H%M%S"))

    os.makedirs(os.path.join(path, "frames"), exist_ok=True)

    return path


class RunRecorder:
    """
    Accumulates per-frame profiles and band readings, and saves a decimated
    set of warped cassette images.
    """

    def __init__(
        self,
        run_dir,
        image_interval_s=IMAGE_INTERVAL_S,
        autosave_every=AUTOSAVE_EVERY
    ):
        self.run_dir = run_dir
        self.image_interval_s = image_interval_s
        self.autosave_every = autosave_every

        self.times = []
        self.profiles = []
        self.readings = []

        self._last_image_t = None
        self._images_saved = 0
        self._warned = False

    @property
    def frame_count(self):
        return len(self.times)

    def add(self, elapsed, result):
        """
        Record one analysed frame.
        """
        self.times.append(elapsed)
        self.profiles.append(np.asarray(result.profile, dtype=np.float32))

        self.readings.append((
            -1 if result.test.idx is None else result.test.idx,
            -1 if result.control.idx is None else result.control.idx,
            result.test.strength,
            result.control.strength,
            result.test.snr,
            result.control.snr,
            result.tc_ratio,
            float(result.test.present),
            float(result.control.present),
            result.test.area,
            result.control.area,
            result.tc_area_ratio,
            result.test.peak_a,
            result.control.peak_a,
            result.scale
        ))

        # Recording is best-effort: a write failure must never abort a run
        # that is still producing good CSV rows.
        try:
            self._maybe_save_image(elapsed, result)

            if self.autosave_every and self.frame_count % self.autosave_every == 0:
                self.save()
        except Exception as e:
            if not self._warned:
                print("Recording degraded:", e)
                self._warned = True

    def _maybe_save_image(self, elapsed, result):
        if self.image_interval_s is None:
            return

        due = (
            self._last_image_t is None
            or (elapsed - self._last_image_t) >= self.image_interval_s
        )

        if not due:
            return

        path = os.path.join(
            self.run_dir,
            "frames",
            f"{elapsed:09.2f}.jpg"
        )

        if cv.imwrite(path, result.warped):
            self._last_image_t = elapsed
            self._images_saved += 1

    def save(self):
        """
        Write profiles and readings to profiles.npz. Safe to call repeatedly.
        """
        if not self.times:
            return

        # Profile length is fixed by the canonical geometry, but guard anyway
        # so one odd frame cannot make the whole array ragged.
        width = max(len(p) for p in self.profiles)

        profiles = np.full(
            (len(self.profiles), width),
            np.nan,
            dtype=np.float32
        )

        for i, p in enumerate(self.profiles):
            profiles[i, :len(p)] = p

        readings = np.asarray(self.readings, dtype=np.float64)

        np.savez(
            os.path.join(self.run_dir, "profiles.npz"),
            time_seconds=np.asarray(self.times, dtype=np.float64),
            profiles=profiles,
            test_idx=readings[:, 0].astype(np.int32),
            control_idx=readings[:, 1].astype(np.int32),
            test_strength=readings[:, 2],
            control_strength=readings[:, 3],
            test_snr=readings[:, 4],
            control_snr=readings[:, 5],
            tc_ratio=readings[:, 6],
            test_present=readings[:, 7].astype(np.int8),
            control_present=readings[:, 8].astype(np.int8),
            # Density, in raw a* units
            test_area=readings[:, 9],
            control_area=readings[:, 10],
            tc_area_ratio=readings[:, 11],
            test_peak_a=readings[:, 12],
            control_peak_a=readings[:, 13],
            # `profiles` are normalised; multiply by this to get raw a* units
            profile_scale=readings[:, 14]
        )

    def summary(self):
        return (
            f"{self.frame_count} frames, "
            f"{self._images_saved} images -> {self.run_dir}"
        )


def load_run(run_dir):
    """
    Load a recorded run back for offline analysis.

    Missing band indices come back as -1; use `test_idx >= 0` to mask.
    """
    with np.load(os.path.join(run_dir, "profiles.npz")) as data:
        return {k: data[k] for k in data.files}


def raw_profiles(data):
    """
    Recover the profiles in raw a* units from a loaded run.

    `profiles` is stored normalised (that is what detection uses); the
    per-frame scale factor puts it back on a physical footing.
    """
    return data["profiles"] * data["profile_scale"][:, None]
