"""MediaPipe Pose wrapper that selects and draws the more visible arm."""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import mediapipe as mp
import numpy as np

from angle_calculator import calculate_elbow_angle


@dataclass
class PoseObservation:
    angle_degrees: float | None
    arm_side: str | None
    visibility: float
    status: str


class PoseDetector:
    """Run pretrained MediaPipe Pose on each RGB frame and annotate it."""

    def __init__(self, min_visibility: float = 0.55) -> None:
        self.min_visibility = min_visibility
        self._selected_side: str | None = None
        self._challenger_side: str | None = None
        self._challenger_frames = 0
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._drawing = mp.solutions.drawing_utils
        self._connections = mp.solutions.pose.POSE_CONNECTIONS
        self._landmark = mp.solutions.pose.PoseLandmark

    def process(self, frame_bgr, arm_side: str | None = None) -> tuple[object, PoseObservation]:
        """Return an annotated frame, optionally measuring a requested anatomical arm."""
        if (not isinstance(frame_bgr, np.ndarray) or frame_bgr.size == 0
                or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3):
            raise ValueError("Expected a non-empty three-channel BGR camera frame")
        if frame_bgr.dtype.name != "uint8":
            raise ValueError("Expected an 8-bit BGR camera frame")
        frame_bgr = frame_bgr.copy()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        result = self._pose.process(frame_rgb)
        frame_rgb.flags.writeable = True
        annotated = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        if not result.pose_landmarks:
            return annotated, PoseObservation(None, None, 0.0, "Person not detected")

        self._drawing.draw_landmarks(
            annotated,
            result.pose_landmarks,
            self._connections,
            landmark_drawing_spec=self._drawing.DrawingSpec(
                color=(205, 205, 205), thickness=2, circle_radius=2
            ),
            connection_drawing_spec=self._drawing.DrawingSpec(
                color=(110, 130, 140), thickness=2
            ),
        )
        landmarks = result.pose_landmarks.landmark
        arms = {
            "LEFT": (self._landmark.LEFT_SHOULDER, self._landmark.LEFT_ELBOW, self._landmark.LEFT_WRIST),
            "RIGHT": (self._landmark.RIGHT_SHOULDER, self._landmark.RIGHT_ELBOW, self._landmark.RIGHT_WRIST),
        }
        candidates = []
        for side, indices in arms.items():
            points = [landmarks[index.value] for index in indices]
            visibility = min(
                max(0.0, min(1.0, float(point.visibility)))
                if math.isfinite(float(point.visibility)) else 0.0
                for point in points
            )
            candidates.append((visibility, side, indices, points))
        by_side = {candidate[1]: candidate for candidate in candidates}
        if arm_side is not None:
            requested_side = arm_side.strip().upper()
            if requested_side not in by_side:
                raise ValueError("arm_side must be LEFT, RIGHT, or None")
            selected = by_side[requested_side]
            self._selected_side = requested_side
            self._challenger_side = None
            self._challenger_frames = 0
        else:
            best = max(candidates, key=lambda item: item[0])
            current = by_side.get(self._selected_side)

            # Keep the current anatomical side through small frame-to-frame
            # confidence changes. Switch only after a clear, sustained improvement,
            # or immediately if the current arm is no longer reliably visible.
            if current is None or current[0] < self.min_visibility:
                selected = best if best[0] >= self.min_visibility else None
                self._challenger_side = None
                self._challenger_frames = 0
            elif best[1] == self._selected_side or best[0] <= current[0] + 0.15:
                selected = current
                self._challenger_side = None
                self._challenger_frames = 0
            else:
                if self._challenger_side == best[1]:
                    self._challenger_frames += 1
                else:
                    self._challenger_side = best[1]
                    self._challenger_frames = 1
                if self._challenger_frames >= 5:
                    selected = best
                    self._challenger_side = None
                    self._challenger_frames = 0
                else:
                    selected = current

        if selected is None:
            self._selected_side = None
            visibility = max(candidate[0] for candidate in candidates)
            for _, side, indices, points in candidates:
                self._draw_arm_landmarks(annotated, side, indices, points)
            return annotated, PoseObservation(None, None, visibility, "Arm landmarks unclear")

        visibility, side, indices, points = selected
        self._draw_arm_landmarks(annotated, side, indices, points)
        if visibility < self.min_visibility:
            return annotated, PoseObservation(None, side, visibility, "Arm landmarks unclear")
        self._selected_side = side

        height, width = annotated.shape[:2]
        elbow_x = int(points[1].x * width)
        elbow_y = int(points[1].y * height)
        try:
            angle = calculate_elbow_angle(
                (points[0].x, points[0].y),
                (points[1].x, points[1].y),
                (points[2].x, points[2].y),
            )
        except ValueError:
            return annotated, PoseObservation(None, side, visibility, "Arm landmarks unclear")

        cv2.putText(annotated, f"{angle:.1f} deg", (elbow_x + 12, elbow_y + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)
        return annotated, PoseObservation(angle, side, visibility, "Tracking stable")

    @staticmethod
    def _draw_arm_landmarks(frame, side, indices, points) -> None:
        """Highlight visible shoulder, elbow, wrist, and their connecting arm segments."""
        height, width = frame.shape[:2]
        color = (0, 145, 255) if side == "LEFT" else (255, 145, 0)
        visible_points: list[tuple[int, int] | None] = []
        for point in points:
            confidence = float(point.visibility)
            coordinates_finite = math.isfinite(float(point.x)) and math.isfinite(float(point.y))
            visible = math.isfinite(confidence) and confidence >= 0.10 and coordinates_finite
            if not visible:
                visible_points.append(None)
                continue
            x = max(0, min(width - 1, int(float(point.x) * width)))
            y = max(0, min(height - 1, int(float(point.y) * height)))
            visible_points.append((x, y))

        for start, end in ((0, 1), (1, 2)):
            if visible_points[start] is not None and visible_points[end] is not None:
                cv2.line(frame, visible_points[start], visible_points[end], color, 4, cv2.LINE_AA)
        for index, point, pixel in zip(indices, points, visible_points):
            if pixel is None:
                continue
            cv2.circle(frame, pixel, 8, color, -1, cv2.LINE_AA)
            label = f"{side} {index.name.split('_')[1]}"
            cv2.putText(frame, label, (pixel[0] + 8, pixel[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    def close(self) -> None:
        self._pose.close()
