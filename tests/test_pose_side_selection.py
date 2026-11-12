"""Tests that requested anatomical arm labels remain paired with their points."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from pose_detector import PoseDetector


class PoseSideSelectionTests(unittest.TestCase):
    """Test side-specific landmark selection independently of live camera input."""

    def setUp(self):
        self.detector = PoseDetector(min_visibility=0.55)
        self.detector._drawing.draw_landmarks = lambda *args, **kwargs: None
        self.points = [SimpleNamespace(x=0.5, y=0.5, visibility=0.95) for _ in range(33)]
        self._set_arm("LEFT", ((0.2, 0.2), (0.2, 0.4), (0.4, 0.4)))
        self._set_arm("RIGHT", ((0.6, 0.2), (0.6, 0.4), (0.6, 0.6)))
        self.detector._pose.process = lambda _: SimpleNamespace(
            pose_landmarks=SimpleNamespace(landmark=self.points)
        )
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def tearDown(self):
        self.detector.close()

    def _set_arm(self, side, coordinates, visibility=0.95):
        """Set test-only coordinates for the shoulder, elbow, and wrist of a side."""
        landmark = self.detector._landmark
        indices = {
            "LEFT": (landmark.LEFT_SHOULDER, landmark.LEFT_ELBOW, landmark.LEFT_WRIST),
            "RIGHT": (landmark.RIGHT_SHOULDER, landmark.RIGHT_ELBOW, landmark.RIGHT_WRIST),
        }[side]
        for index, (x, y) in zip(indices, coordinates):
            self.points[index.value] = SimpleNamespace(x=x, y=y, visibility=visibility)

    def test_left_and_right_measurements_use_matching_landmarks(self):
        _, left = self.detector.process(self.frame, arm_side="LEFT")
        _, right = self.detector.process(self.frame, arm_side="RIGHT")
        self.assertEqual(left.arm_side, "LEFT")
        self.assertAlmostEqual(left.angle_degrees, 90.0, places=4)
        self.assertEqual(right.arm_side, "RIGHT")
        self.assertAlmostEqual(right.angle_degrees, 180.0, places=4)

    def test_unreliable_requested_arm_is_not_replaced_by_other_side(self):
        self._set_arm("LEFT", ((0.2, 0.2), (0.2, 0.4), (0.4, 0.4)), visibility=0.9)
        self._set_arm("RIGHT", ((0.6, 0.2), (0.6, 0.4), (0.6, 0.6)), visibility=0.2)
        _, observation = self.detector.process(self.frame, arm_side="RIGHT")
        self.assertIsNone(observation.angle_degrees)
        self.assertEqual(observation.arm_side, "RIGHT")
        self.assertEqual(observation.status, "Arm landmarks unclear")


if __name__ == "__main__":
    unittest.main()
