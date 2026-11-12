"""Lifecycle and active-time tracking for a single exercise session."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import re
import time
from uuid import uuid4


class SessionStateError(RuntimeError):
    """Raised when an operation is invalid for the current session state."""


@dataclass
class ExerciseSession:
    """An in-memory session record used to associate live observations safely."""

    session_id: str
    participant_id: str
    arm_side: str
    exercise_type: str
    target_repetitions: int
    start_time: str
    start_monotonic: float = field(repr=False)
    status: str = "in_progress"
    end_time: str = ""
    duration_seconds: float = 0.0
    paused: bool = False
    _active_seconds: float = field(default=0.0, repr=False)
    _running_since: float | None = field(default=None, repr=False)


class SessionManager:
    """Start, pause, resume, and end one session without allowing overlap."""

    PARTICIPANT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$")

    def __init__(self) -> None:
        """Initialize a manager with no active session."""
        self.current: ExerciseSession | None = None

    def start_session(
        self,
        *,
        participant_id: str,
        arm_side: str,
        target_repetitions: int,
        exercise_type: str = "elbow_flexion_extension",
        started_at: datetime | None = None,
        monotonic_now: float | None = None,
    ) -> ExerciseSession:
        """Create a unique session after validating anonymous study settings."""
        if self.current is not None and self.current.status == "in_progress":
            raise SessionStateError("A session is already in progress")
        participant_id = participant_id.strip()
        if not self.PARTICIPANT_ID_PATTERN.fullmatch(participant_id):
            raise ValueError("Participant ID must be 1-32 letters, numbers, '_' or '-' (anonymous code only)")
        normalized_arm = arm_side.strip().upper()
        if normalized_arm not in {"LEFT", "RIGHT"}:
            raise ValueError("Exercise arm must be LEFT or RIGHT")
        if not isinstance(target_repetitions, int) or target_repetitions < 1:
            raise ValueError("Target repetitions must be a positive integer")
        started_at = started_at or datetime.now().astimezone()
        monotonic_now = time.monotonic() if monotonic_now is None else monotonic_now
        stamp = started_at.strftime("%Y%m%d_%H%M%S")
        session = ExerciseSession(
            session_id=f"session_{stamp}_{uuid4().hex[:8]}",
            participant_id=participant_id,
            arm_side=normalized_arm,
            exercise_type=exercise_type,
            target_repetitions=target_repetitions,
            start_time=started_at.isoformat(timespec="seconds"),
            start_monotonic=monotonic_now,
            _running_since=monotonic_now,
        )
        self.current = session
        return session

    def elapsed_seconds(self, now: float | None = None) -> float:
        """Return active elapsed time, excluding time spent paused."""
        if self.current is None:
            raise SessionStateError("No session has been created")
        session = self.current
        if session.status != "in_progress":
            return session.duration_seconds
        if session.paused or session._running_since is None:
            return max(0.0, session._active_seconds)
        now = time.monotonic() if now is None else now
        return max(0.0, session._active_seconds + now - session._running_since)

    def pause(self, now: float | None = None) -> None:
        """Pause the active session timer and reject duplicate pause actions."""
        session = self._require_active()
        if session.paused:
            raise SessionStateError("Session is already paused")
        now = time.monotonic() if now is None else now
        if session._running_since is not None:
            session._active_seconds += max(0.0, now - session._running_since)
        session._running_since = None
        session.paused = True

    def resume(self, now: float | None = None) -> None:
        """Resume an active paused session without adding pause time to duration."""
        session = self._require_active()
        if not session.paused:
            raise SessionStateError("Session is not paused")
        session._running_since = time.monotonic() if now is None else now
        session.paused = False

    def end_session(
        self,
        *,
        completed_repetitions: int,
        valid_observations: int,
        ended_at: datetime | None = None,
        monotonic_now: float | None = None,
    ) -> ExerciseSession:
        """Set a final status and duration based on target and collected data."""
        session = self._require_active()
        ended_at = ended_at or datetime.now().astimezone()
        monotonic_now = time.monotonic() if monotonic_now is None else monotonic_now
        if not session.paused and session._running_since is not None:
            session._active_seconds += max(0.0, monotonic_now - session._running_since)
        session.duration_seconds = round(max(0.0, session._active_seconds), 3)
        session.end_time = ended_at.isoformat(timespec="seconds")
        session.paused = False
        session._running_since = None
        if completed_repetitions >= session.target_repetitions:
            session.status = "completed"
        elif valid_observations > 0 or completed_repetitions > 0:
            session.status = "interrupted"
        else:
            session.status = "cancelled"
        return session

    def _require_active(self) -> ExerciseSession:
        """Return the current session or raise when no session has started."""
        if self.current is None or self.current.status != "in_progress":
            raise SessionStateError("No session is currently in progress")
        return self.current
