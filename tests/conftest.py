"""
Test setup.

`main` imports picamera2 and libcamera, which only exist on the Pi. Stub
modules are installed into sys.modules so the capture loop can be exercised
on a workstation; everything else in the project is already Pi-free.
"""

import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class FakeCamera:
    """
    Stands in for Picamera2. Replays a fixed list of RGB frames, repeating the
    last one once exhausted so a timed loop always has something to capture.
    """

    frames = []

    def __init__(self):
        self.index = 0
        self.stopped = False

    def create_video_configuration(self, **kwargs):
        return {}

    def configure(self, config):
        pass

    def start(self):
        pass

    def set_controls(self, controls):
        pass

    def stop(self):
        self.stopped = True

    def capture_array(self):
        if not FakeCamera.frames:
            return np.zeros((1080, 1920, 3), np.uint8)

        frame = FakeCamera.frames[min(self.index, len(FakeCamera.frames) - 1)]
        self.index += 1

        return frame


def _install_stubs():
    if "picamera2" not in sys.modules:
        module = types.ModuleType("picamera2")
        module.Picamera2 = FakeCamera
        sys.modules["picamera2"] = module

    if "libcamera" not in sys.modules:
        module = types.ModuleType("libcamera")

        af = type("AfModeEnum", (), {"Continuous": "continuous"})
        module.controls = type("controls", (), {"AfModeEnum": af})

        sys.modules["libcamera"] = module


_install_stubs()


@pytest.fixture
def fake_camera():
    """Reset the stub camera and hand back the class for frame loading."""
    FakeCamera.frames = []

    return FakeCamera


@pytest.fixture
def headless(monkeypatch):
    """
    Silence the display so the capture loop runs without a window server.
    Returns a dict recording which windows were drawn.
    """
    import cv2 as cv

    shown = {}

    monkeypatch.setattr(cv, "imshow", lambda name, image: shown.setdefault(name, image.shape))
    monkeypatch.setattr(cv, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(cv, "waitKey", lambda delay: 0xFF)

    return shown
