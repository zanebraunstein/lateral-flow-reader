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

    def save(self, path=CALIBRATION_PATH):
        with open(path, "w") as handle:
            json.dump(
                {
                    "window_quad": self.window_quad.tolist(),
                    "frame_size": list(self.frame_size),
                },
                handle,
                indent=2,
            )


def from_points(points, frame_size):
    """
    Build a Calibration from four clicked corners in any order.
    """
    quad = strip.order_points(np.asarray(points, dtype=np.float32))

    return Calibration(window_quad=quad, frame_size=tuple(frame_size))


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
    )
