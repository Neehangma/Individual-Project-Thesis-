"""MediaPipe Pose wrapper that selects and draws the more visible arm."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import mediapipe as mp

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

    def process(self, frame_bgr) -> tuple[object, PoseObservation]:
        """Return an annotated BGR frame and its measured elbow observation."""
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
            "left": (self._landmark.LEFT_SHOULDER, self._landmark.LEFT_ELBOW, self._landmark.LEFT_WRIST),
            "right": (self._landmark.RIGHT_SHOULDER, self._landmark.RIGHT_ELBOW, self._landmark.RIGHT_WRIST),
        }
        candidates = []
        for side, indices in arms.items():
            points = [landmarks[index.value] for index in indices]
            visibility = min(float(point.visibility) for point in points)
            candidates.append((visibility, side, indices, points))
        visibility, side, indices, points = max(candidates, key=lambda item: item[0])
        if visibility < self.min_visibility:
            return annotated, PoseObservation(None, side, visibility, "Arm landmarks unclear")

        height, width = annotated.shape[:2]
        pixels = []
        for index, point in zip(indices, points):
            x, y = int(point.x * width), int(point.y * height)
            pixels.append((x, y))
            cv2.circle(annotated, (x, y), 8, (0, 145, 255), -1)
            cv2.putText(annotated, index.name.replace("_", " ").title(), (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        try:
            angle = calculate_elbow_angle(
                (points[0].x, points[0].y),
                (points[1].x, points[1].y),
                (points[2].x, points[2].y),
            )
        except ValueError:
            return annotated, PoseObservation(None, side, visibility, "Arm landmarks unclear")

        cv2.putText(annotated, f"{angle:.1f} deg", (pixels[1][0] + 12, pixels[1][1] + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)
        return annotated, PoseObservation(angle, side, visibility, "Tracking stable")

    def close(self) -> None:
        self._pose.close()
