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
            "LEFT": (self._landmark.LEFT_SHOULDER, self._landmark.LEFT_ELBOW, self._landmark.LEFT_WRIST),
            "RIGHT": (self._landmark.RIGHT_SHOULDER, self._landmark.RIGHT_ELBOW, self._landmark.RIGHT_WRIST),
        }
        candidates = []
        for side, indices in arms.items():
            points = [landmarks[index.value] for index in indices]
            visibility = min(float(point.visibility) for point in points)
            candidates.append((visibility, side, indices, points))
        best = max(candidates, key=lambda item: item[0])
        by_side = {candidate[1]: candidate for candidate in candidates}
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
            return annotated, PoseObservation(None, None, best[0], "Arm landmarks unclear")

        visibility, side, indices, points = selected
        self._selected_side = side

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
