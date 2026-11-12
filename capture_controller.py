"""Session-scoped bridge between live webcam frames and the Streamlit UI."""

from __future__ import annotations

from collections import deque
from datetime import datetime
import time
from typing import Any
from uuid import uuid4

import cv2
import numpy as np

from camera_registry import CameraRegistry
from data_logger import DataLogger
from feedback_engine import FeedbackEngine
from pose_detector import PoseDetector, PoseObservation
from rep_counter import RepCounter
from session_manager import ExerciseSession, SessionManager


# Session and angle configuration, shared with the existing repetition logic.
MIN_VISIBILITY = 0.60
POSE_MIN_VISIBILITY = 0.55
ANGLE_SMOOTHING_WINDOW = 5
FLEXED_THRESHOLD_DEGREES = 60.0
EXTENDED_THRESHOLD_DEGREES = 150.0
TARGET_MIN_ANGLE_DEGREES = 40.0
TARGET_MAX_ANGLE_DEGREES = 150.0
TOTAL_SETS = 3
REPS_PER_SET = 10
REST_SECONDS = 10
GET_READY_SECONDS = 3.0
CAMERA_INDEX = 0


def create_counter(target_repetitions: int) -> RepCounter:
    """Create the established hysteresis counter for one configured session."""
    return RepCounter(
        flexed_threshold=FLEXED_THRESHOLD_DEGREES,
        extended_threshold=EXTENDED_THRESHOLD_DEGREES,
        reps_per_set=REPS_PER_SET,
        total_sets=TOTAL_SETS,
        rest_seconds=REST_SECONDS,
        target_min_angle=TARGET_MIN_ANGLE_DEGREES,
        target_max_angle=TARGET_MAX_ANGLE_DEGREES,
        total_target_repetitions=target_repetitions,
    )


class CaptureController:
    """Own one participant's camera, detector, counter, and data lifecycle."""

    def __init__(self, registry: CameraRegistry, *, data_dir=None) -> None:
        """Initialize idle session resources without opening the camera."""
        self.registry = registry
        self.data_dir = data_dir
        self.owner = uuid4().hex
        self.manager = SessionManager()
        self.logger = DataLogger(data_dir, min_confidence=MIN_VISIBILITY)
        self.session: ExerciseSession | None = None
        self.camera: Any | None = None
        self.detector: PoseDetector | None = None
        self.counter: RepCounter | None = None
        self.feedback_engine = FeedbackEngine()
        self.angle_history: deque[float] = deque(maxlen=ANGLE_SMOOTHING_WINDOW)
        self.state = "idle"
        self.last_frame: np.ndarray | None = None
        self.raw_angle: float | None = None
        self.smoothed_angle: float | None = None
        self.arm_side: str | None = None
        self.visibility = 0.0
        self.tracking_status = "Camera is not running."
        self.feedback_message = "Enter session details and start when ready."
        self.movement_state = "READY"
        self.phase = "Extended"
        self.elapsed_seconds = 0.0
        self.ready_until = 0.0
        self.last_record_at = 0.0
        self.last_error = ""
        self._pause_started_at: float | None = None
        self._last_observation = PoseObservation(None, None, 0.0, "Not tracking")

    @property
    def is_running(self) -> bool:
        """Whether a session is collecting observations or in its countdown."""
        return self.state in {"ready", "active", "paused"} and self.session is not None

    @property
    def completed_repetitions(self) -> int:
        """Return current session repetitions, or zero before setup."""
        return self.counter.total_reps if self.counter is not None else 0

    @property
    def target_repetitions(self) -> int:
        """Return the active target or zero when no session has started."""
        return self.session.target_repetitions if self.session else 0

    def start(self, participant_id: str, arm_side: str, target_repetitions: int) -> ExerciseSession:
        """Acquire the live camera and begin a new validated research session."""
        if self.is_running:
            raise RuntimeError("A session is already active in this browser.")
        if not 1 <= int(target_repetitions) <= TOTAL_SETS * REPS_PER_SET:
            raise ValueError(f"Target must be between 1 and {TOTAL_SETS * REPS_PER_SET} repetitions.")
        capture = self.registry.acquire(self.owner, CAMERA_INDEX, cv2)
        detector = None
        try:
            detector = PoseDetector(min_visibility=POSE_MIN_VISIBILITY)
            session = self.manager.start_session(
                participant_id=participant_id,
                arm_side=arm_side,
                target_repetitions=int(target_repetitions),
            )
            self.logger.start_session(session)
        except Exception:
            try:
                if detector is not None:
                    detector.close()
            finally:
                capture.release()
                self.registry.release(self.owner)
            raise
        self.camera, self.detector, self.session = capture, detector, session
        self.counter = create_counter(int(target_repetitions))
        self.angle_history.clear()
        self.state = "ready"
        self.ready_until = time.monotonic() + GET_READY_SECONDS
        self.elapsed_seconds = 0.0
        self.last_error = ""
        self.feedback_message = "Get ready. The session will begin in 3 seconds."
        self.tracking_status = "Position yourself so your selected arm is visible."
        return session

    def tick(self) -> bool:
        """Read and process one real frame, logging at most one sampled observation."""
        if not self.is_running or self.camera is None or self.detector is None:
            return False
        ok, frame = self.camera.read()
        if not ok:
            self.last_error = "The webcam stopped returning frames. The session was saved and the camera released."
            self.end()
            self.state = "error"
            return False

        now = time.monotonic()
        try:
            annotated, observation = self.detector.process(frame, arm_side=self.session.arm_side)
        except Exception as exc:
            self.last_error = f"Pose tracking failed: {exc}"
            self.end()
            self.state = "error"
            return False
        self._last_observation = observation
        valid = (
            observation.angle_degrees is not None
            and observation.visibility >= MIN_VISIBILITY
            and observation.arm_side == self.session.arm_side
        )
        self.raw_angle = observation.angle_degrees if valid else None
        if valid:
            self.angle_history.append(float(observation.angle_degrees))
            self.smoothed_angle = float(np.mean(self.angle_history))
            self.arm_side = observation.arm_side
            self.tracking_status = "Tracking stable"
        else:
            self.angle_history.clear()
            self.smoothed_angle = None
            self.arm_side = observation.arm_side
            self.tracking_status = (
                "Person not detected. Move into the camera view."
                if observation.status == "Person not detected"
                else "Arm not fully visible. Adjust your position."
            )

        self.visibility = observation.visibility
        self.last_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        if self.state == "ready" and now >= self.ready_until:
            self.state = "active"
            self.feedback_message = "Tracking started. Extend your elbow to begin."

        if self.state == "active":
            if valid and self.smoothed_angle is not None:
                event = self.counter.update(self.smoothed_angle, now)
                if self.counter.partial_movement:
                    self.logger.record_incomplete_movement(self.session)
                self.feedback_message = self.feedback_engine.message(
                    tracking_valid=True,
                    movement_state=self.counter.state,
                    phase=self.counter.phase,
                    arm_ready=self.counter.is_armed,
                    now=now,
                    rep_number=event.rep_number if event else None,
                    partial_movement=self.counter.partial_movement,
                )
            else:
                if self.counter.invalidate_tracking(now):
                    self.logger.record_incomplete_movement(self.session)
                self.feedback_message = self.feedback_engine.message(
                    tracking_valid=False,
                    movement_state=self.counter.state,
                    phase=self.counter.phase,
                    arm_ready=self.counter.is_armed,
                    now=now,
                )

            self.elapsed_seconds = self.manager.elapsed_seconds(now)
            # Fragment reruns are throttled; this additional guard prevents duplicate records.
            if now - self.last_record_at >= 0.10:
                self.logger.record_observation(
                    self.session,
                    elapsed_seconds=self.elapsed_seconds,
                    elbow_angle=self.smoothed_angle if valid else None,
                    pose_confidence=self.visibility,
                    tracking_status=self.tracking_status,
                    movement_state=self.counter.state,
                    completed_repetitions=self.counter.total_reps,
                    feedback_message=self.feedback_message,
                )
                self.last_record_at = now
            if self.counter.is_complete:
                self.feedback_message = "Session target reached. Session completed."
                self.end(completed=True)
        elif self.state == "paused":
            self.elapsed_seconds = self.manager.elapsed_seconds(now)
            self.feedback_message = "Session paused. Resume to continue recording."
        else:
            self.elapsed_seconds = self.manager.elapsed_seconds(now)
        self.movement_state = self.counter.state if self.counter else "READY"
        self.phase = self.counter.phase if self.counter else "Extended"
        return True

    def pause(self) -> None:
        """Pause time, repetition counting, and measurement collection."""
        if self.state not in {"ready", "active"} or self.session is None:
            raise RuntimeError("There is no active session to pause.")
        now = time.monotonic()
        if self.state == "active" and self.counter and self.counter.invalidate_tracking(now):
            self.logger.record_incomplete_movement(self.session)
        self.manager.pause(now)
        self._pause_started_at = now
        self.state = "paused"
        self.angle_history.clear()
        self.feedback_message = "Session paused. Resume to continue recording."

    def resume(self) -> None:
        """Resume the session and exclude paused time from timer and rep timing."""
        if self.state != "paused" or self.session is None:
            raise RuntimeError("The session is not paused.")
        now = time.monotonic()
        pause_duration = now - (self._pause_started_at or now)
        self.manager.resume(now)
        if self.counter:
            self.counter.shift_time(pause_duration)
        self._pause_started_at = None
        self.state = "active"
        self.angle_history.clear()

    def end(self, *, completed: bool = False) -> dict[str, Any] | None:
        """Save the summary once and release the detector and webcam resources."""
        summary = None
        try:
            if self.session is not None and self.session.status == "in_progress":
                if self.counter and self.counter.reset_movement():
                    self.logger.record_incomplete_movement(self.session)
                ended = self.manager.end_session(
                    completed_repetitions=self.counter.total_reps if self.counter else 0,
                    valid_observations=self.logger.valid_observation_count(self.session),
                )
                summary = self.logger.finish_session(
                    ended, completed_repetitions=self.counter.total_reps if self.counter else 0
                )
                self.state = "completed" if completed else "ended"
        finally:
            self._release_resources()
        return summary

    def close(self) -> None:
        """Finalize an unfinished session as interrupted and free native resources."""
        if self.session is not None and self.session.status == "in_progress":
            self.end()
        else:
            self._release_resources()

    def __del__(self) -> None:
        """Best-effort cleanup if Streamlit discards a disconnected browser session."""
        try:
            self.close()
        except Exception:
            # Python may be shutting down; the process-wide registry also releases
            # the camera from its atexit handler.
            pass

    def _release_resources(self) -> None:
        """Close MediaPipe and release the exclusive camera lease idempotently."""
        detector, self.detector = self.detector, None
        self.camera = None
        try:
            if detector is not None:
                detector.close()
        finally:
            self.registry.release(self.owner)
