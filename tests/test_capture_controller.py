"""Synthetic integration tests for camera lifecycle and CSV capture wiring."""

from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from camera_registry import CameraBusyError, CameraRegistry
from capture_controller import CaptureController
from pose_detector import PoseObservation


class _FakeCamera:
    """Provide test-only frames through the same OpenCV capture interface."""

    def __init__(self, *_args, **_kwargs):
        self.released = False

    def isOpened(self):
        return True

    def read(self):
        return True, np.zeros((100, 100, 3), dtype=np.uint8)

    def release(self):
        self.released = True


class _FakeDetector:
    """Produce synthetic angle observations in controller unit tests only."""

    def __init__(self, *_args, **_kwargs):
        self.angles = iter([160, 160, 160, 160, 160, 100, 50, 50, 80, 120, 160, 160, 160, 160, 160])
        self.closed = False

    def process(self, frame, arm_side=None):
        angle = next(self.angles, 160)
        return frame, PoseObservation(angle, arm_side, 0.95, "Tracking stable")

    def close(self):
        self.closed = True


class CaptureControllerTests(unittest.TestCase):
    """Ensure real controller integration appends a completed test session."""

    def test_camera_lease_and_completed_session_are_persisted(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = CameraRegistry()
            camera = _FakeCamera()
            with patch("capture_controller.cv2.VideoCapture", return_value=camera), \
                 patch("capture_controller.PoseDetector", _FakeDetector), \
                 patch("capture_controller.GET_READY_SECONDS", 0), \
                 patch("capture_controller.ANGLE_SMOOTHING_WINDOW", 1):
                controller = CaptureController(registry, data_dir=Path(temp_dir))
                session = controller.start("TEST01", "LEFT", 1)
                with self.assertRaises(CameraBusyError):
                    registry.acquire("other-browser", 0, __import__("cv2"))
                for _ in range(20):
                    controller.tick()
                    if controller.state == "completed":
                        break
                self.assertEqual(controller.state, "completed")
                self.assertEqual(controller.completed_repetitions, 1)
                self.assertTrue(camera.released)

            with (Path(temp_dir) / "session_summaries.csv").open(newline="", encoding="utf-8") as file:
                summaries = list(csv.DictReader(file))
            with (Path(temp_dir) / "session_observations.csv").open(newline="", encoding="utf-8") as file:
                observations = list(csv.DictReader(file))
            self.assertEqual(len(summaries), 1)
            self.assertEqual(summaries[0]["session_id"], session.session_id)
            self.assertEqual(summaries[0]["session_status"], "completed")
            self.assertTrue(observations)
            self.assertTrue(all(row["session_id"] == session.session_id for row in observations))


if __name__ == "__main__":
    unittest.main()
