"""
Fixed-rig calibration.

When the camera is mounted at a fixed position over the cassette, the results
window never moves, so it is marked once and reused for every frame. This
removes per-frame cassette detection entirely -- it works even when most of
the cassette is out of frame, and there is no detection jitter to pollute the
measurement.

The calibration is four corner points of the results window in frame
coordinates, stored in calibration.json.
"""

import json
import os
from dataclasses import dataclass

import numpy as np

import strip


CALIBRATION_PATH = "calibration.json"


@dataclass
class Calibration:
    # Four points (tl, tr, br, bl), in the pixel coordinates of the frame the
    # calibration was captured at.
    window_quad: np.ndarray
    frame_size: tuple

    # Band positions along the strip (fraction 0..1), learned from the cassette
    # during calibration. This is what lets the reader handle either cassette
    # orientation without hand-tuned constants. None if not captured.
    control_frac: float = None
    test_frac: float = None

    @property
    def bands(self):
        """(test_frac, control_frac) for strip.analyze, or None if uncaptured."""
        if self.control_frac is None and self.test_frac is None:
            return None

        return (self.test_frac, self.control_frac)

    def save(self, path=CALIBRATION_PATH):
        with open(path, "w") as handle:
            json.dump(
                {
                    "window_quad": self.window_quad.tolist(),
                    "frame_size": list(self.frame_size),
                    "control_frac": self.control_frac,
                    "test_frac": self.test_frac,
                },
                handle,
                indent=2,
            )


def learn_bands(profile, control_side):
    """
    Locate the two bands in a profile and assign control/test by which side
    the control is on. Returns (control_frac, test_frac) as strip fractions;
    either may be None if fewer than two bands are present.

    Uses raw peaks (not the width-filtered candidates), so a broad band is
    still found -- the same reason detection snaps to peaks at runtime. This
    is what lets calibration handle either cassette orientation without
    hand-tuned positions.
    """
    n = len(profile)
    peaks = strip.find_peak_candidates(profile)
    found = strip.dominant_two_bands(profile, peaks)

    if len(found) == 2:
        left, right = found
        control_idx, test_idx = (left, right) if control_side == "left" else (right, left)
    elif len(found) == 1:
        control_idx, test_idx = found[0], None
    else:
        control_idx, test_idx = None, None

    control_frac = None if control_idx is None else control_idx / n
    test_frac = None if test_idx is None else test_idx / n

    return control_frac, test_frac


def from_points(points, frame_size, control_frac=None, test_frac=None):
    """
    Build a Calibration from four clicked corners in any order.
    """
    quad = strip.order_points(np.asarray(points, dtype=np.float32))

    return Calibration(
        window_quad=quad,
        frame_size=tuple(frame_size),
        control_frac=control_frac,
        test_frac=test_frac,
    )


def load(path=CALIBRATION_PATH):
    """
    Load a saved calibration, or None if there is none.
    """
    if not os.path.exists(path):
        return None

    with open(path) as handle:
        data = json.load(handle)

    return Calibration(
        window_quad=np.asarray(data["window_quad"], dtype=np.float32),
        frame_size=tuple(data["frame_size"]),
        control_frac=data.get("control_frac"),
        test_frac=data.get("test_frac"),
    )
