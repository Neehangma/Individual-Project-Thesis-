"""Unit-level wiring test for session setup, webcam observations, and CSV output."""

from __future__ import annotations

import csv
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import app
from data_logger import DataLogger
from pose_detector import PoseObservation


class _FakeCamera:
    """Supply blank frames to the app loop in this isolated integration test."""

    def __init__(self, *_args, **_kwargs):
        self.frames_remaining = 14

    def isOpened(self):
        return True

    def read(self):
        if self.frames_remaining <= 0:
            return False, None
        self.frames_remaining -= 1
        return True, np.zeros((480, 640, 3), dtype=np.uint8)

    def release(self):
        return None


class _FakeDetector:
    """Return test-only pose angles while preserving the detector interface."""

    def __init__(self, *_args, **_kwargs):
        self.index = 0
        self.angles = [160, 160, 160, 160, 160, 160, 160, 160,
                       160, 100, 50, 90, 160, 160]

    def process(self, frame, arm_side=None):
        angle = self.angles[self.index]
        self.index += 1
        return frame, PoseObservation(angle, arm_side or "LEFT", 0.95, "Tracking stable")

    def close(self):
        return None


class LiveAppIntegrationTests(unittest.TestCase):
    """Exercise live setup and completed-session persistence through app.main."""

    def test_selected_arm_target_and_rep_session_are_saved(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            key_values = [ord("i"), ord("p"), ord("0"), ord("0"), ord("1"),
                          13, ord("l"), ord("s"), -1, -1, -1, -1, -1, ord("q")]
            keys = iter(key_values)
            real_logger = DataLogger

            def make_logger(*, min_confidence):
                return real_logger(Path(temp_dir), min_confidence=min_confidence)

            with patch.object(sys, "argv", ["app.py", "--target", "1", "--sets", "1"]), \
                 patch.object(app, "DataLogger", side_effect=make_logger), \
                 patch.object(app.cv2, "VideoCapture", _FakeCamera), \
                 patch.object(app, "PoseDetector", _FakeDetector), \
                 patch.object(app, "GET_READY_SECONDS", 0), \
                 patch.object(app, "ANGLE_SMOOTHING_WINDOW", 1), \
                 patch.object(app.cv2, "waitKey", side_effect=lambda _delay: next(keys)), \
                 patch.object(app.cv2, "imshow"), \
                 patch.object(app.cv2, "getWindowProperty", return_value=1), \
                 patch.object(app.cv2, "destroyAllWindows"):
                self.assertEqual(app.main(), 0)

            with (Path(temp_dir) / "session_summaries.csv").open(newline="", encoding="utf-8") as file:
                summaries = list(csv.DictReader(file))
            with (Path(temp_dir) / "session_observations.csv").open(newline="", encoding="utf-8") as file:
                observations = list(csv.DictReader(file))

            self.assertEqual(len(summaries), 1)
            self.assertEqual(summaries[0]["participant_id"], "P001")
            self.assertEqual(summaries[0]["arm_side"], "LEFT")
            self.assertEqual(summaries[0]["session_status"], "completed")
            self.assertEqual(summaries[0]["completed_repetitions"], "1")
            self.assertEqual(len(observations), 5)
            self.assertTrue(all(row["arm_side"] == "LEFT" for row in observations))


if __name__ == "__main__":
    unittest.main()
