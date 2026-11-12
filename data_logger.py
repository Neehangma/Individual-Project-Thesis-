"""Append-only CSV persistence for live observations and session summaries."""

from __future__ import annotations

import csv
from datetime import datetime
import math
from pathlib import Path
from statistics import mean
from typing import Any

from session_manager import ExerciseSession


OBSERVATION_COLUMNS = (
    "session_id", "participant_id", "timestamp", "elapsed_seconds", "arm_side",
    "elbow_angle", "pose_confidence", "tracking_status", "movement_state",
    "completed_repetitions", "target_repetitions", "feedback_message",
)
SUMMARY_COLUMNS = (
    "session_id", "participant_id", "arm_side", "exercise_type", "target_repetitions",
    "completed_repetitions", "incomplete_movements", "start_time", "end_time",
    "total_duration_seconds", "session_status", "average_elbow_angle",
    "minimum_elbow_angle", "maximum_elbow_angle", "tracking_success_rate",
)


class DataIntegrityError(RuntimeError):
    """Raised when stored research data conflicts with the expected CSV schema."""


class DataLogger:
    """Manage append-only research CSVs and valid-angle summary statistics."""

    def __init__(self, data_dir: Path | str | None = None, *, min_confidence: float = 0.60) -> None:
        """Create the data directory and initialize compatible CSV headers."""
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("Minimum pose confidence must be between 0 and 1")
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).resolve().parent / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.observations_path = self.data_dir / "session_observations.csv"
        self.summaries_path = self.data_dir / "session_summaries.csv"
        self.min_confidence = min_confidence
        self._ensure_header(self.observations_path, OBSERVATION_COLUMNS)
        self._ensure_header(self.summaries_path, SUMMARY_COLUMNS)
        self._active: dict[str, dict[str, Any]] = {}
        self._ended_sessions: set[str] = set()

    def start_session(self, session: ExerciseSession) -> None:
        """Register a manager-created session before accepting measurements."""
        if session.status != "in_progress":
            raise DataIntegrityError("Only an in-progress session can collect observations")
        if session.session_id in self._active or session.session_id in self._ended_sessions:
            raise DataIntegrityError("This session has already been registered")
        self._active[session.session_id] = {
            "participant_id": session.participant_id,
            "valid_angles": [],
            "observation_count": 0,
            "valid_count": 0,
            "incomplete_movements": 0,
        }

    def record_observation(
        self,
        session: ExerciseSession,
        *,
        elapsed_seconds: float,
        elbow_angle: float | None,
        pose_confidence: float,
        tracking_status: str,
        movement_state: str,
        completed_repetitions: int,
        feedback_message: str,
        timestamp: datetime | None = None,
    ) -> bool:
        """Append one actual observation, blanking angles from invalid tracking."""
        stats = self._require_collecting(session)
        timestamp = timestamp or datetime.now().astimezone()
        confidence = min(1.0, max(0.0, float(pose_confidence)))
        valid_angle = (
            elbow_angle is not None
            and math.isfinite(float(elbow_angle))
            and confidence >= self.min_confidence
            and session.status == "in_progress"
            and not session.paused
        )
        row = {
            "session_id": session.session_id,
            "participant_id": session.participant_id,
            "timestamp": timestamp.astimezone().isoformat(timespec="milliseconds"),
            "elapsed_seconds": f"{max(0.0, elapsed_seconds):.3f}",
            "arm_side": session.arm_side,
            "elbow_angle": f"{float(elbow_angle):.3f}" if valid_angle else "",
            "pose_confidence": f"{confidence:.4f}",
            "tracking_status": tracking_status,
            "movement_state": movement_state,
            "completed_repetitions": int(completed_repetitions),
            "target_repetitions": session.target_repetitions,
            "feedback_message": feedback_message,
        }
        self._append(self.observations_path, OBSERVATION_COLUMNS, row)
        stats["observation_count"] += 1
        if valid_angle:
            stats["valid_count"] += 1
            stats["valid_angles"].append(float(elbow_angle))
        return bool(valid_angle)

    def record_incomplete_movement(self, session: ExerciseSession) -> None:
        """Increment the summary count for an abandoned partial movement."""
        stats = self._require_collecting(session)
        stats["incomplete_movements"] += 1

    def valid_observation_count(self, session: ExerciseSession) -> int:
        """Return the number of accepted angle observations in this process."""
        return int(self._require_registered(session)["valid_count"])

    def incomplete_movement_count(self, session: ExerciseSession) -> int:
        """Return the number of partial movements tracked for this session."""
        return int(self._require_registered(session)["incomplete_movements"])

    def finish_session(self, session: ExerciseSession, *, completed_repetitions: int) -> dict[str, Any]:
        """Append one summary row and permanently reject more observations."""
        if session.status == "in_progress":
            raise DataIntegrityError("Session manager must end the session before writing its summary")
        stats = self._require_registered(session)
        angles: list[float] = stats["valid_angles"]
        summary = {
            "session_id": session.session_id,
            "participant_id": session.participant_id,
            "arm_side": session.arm_side,
            "exercise_type": session.exercise_type,
            "target_repetitions": session.target_repetitions,
            "completed_repetitions": int(completed_repetitions),
            "incomplete_movements": stats["incomplete_movements"],
            "start_time": session.start_time,
            "end_time": session.end_time,
            "total_duration_seconds": f"{session.duration_seconds:.3f}",
            "session_status": session.status,
            "average_elbow_angle": f"{mean(angles):.3f}" if angles else "",
            "minimum_elbow_angle": f"{min(angles):.3f}" if angles else "",
            "maximum_elbow_angle": f"{max(angles):.3f}" if angles else "",
            "tracking_success_rate": (f"{stats['valid_count'] / stats['observation_count']:.5f}"
                                      if stats["observation_count"] else ""),
        }
        self._append(self.summaries_path, SUMMARY_COLUMNS, summary)
        del self._active[session.session_id]
        self._ended_sessions.add(session.session_id)
        return summary

    def _require_collecting(self, session: ExerciseSession) -> dict[str, Any]:
        """Ensure the session is registered and has not been ended or paused."""
        if session.status != "in_progress" or session.session_id in self._ended_sessions:
            raise DataIntegrityError("Cannot record observations after a session has ended")
        if session.paused:
            raise DataIntegrityError("Cannot record measurements while the session is paused")
        return self._require_registered(session)

    def _require_registered(self, session: ExerciseSession) -> dict[str, Any]:
        """Return registered per-session statistics even during finalization."""
        stats = self._active.get(session.session_id)
        if stats is None or stats["participant_id"] != session.participant_id:
            raise DataIntegrityError("Session is not registered with this data logger")
        return stats

    @staticmethod
    def _ensure_header(path: Path, columns: tuple[str, ...]) -> None:
        """Create a CSV header once or reject an existing incompatible schema."""
        if not path.exists() or path.stat().st_size == 0:
            with path.open("a", newline="", encoding="utf-8") as csv_file:
                csv.writer(csv_file).writerow(columns)
            return
        with path.open("r", newline="", encoding="utf-8-sig") as csv_file:
            existing = next(csv.reader(csv_file), [])
        if tuple(existing) != columns:
            raise DataIntegrityError(f"Existing research CSV has an unexpected header: {path}")

    @staticmethod
    def _append(path: Path, columns: tuple[str, ...], row: dict[str, Any]) -> None:
        """Append and flush one record, raising a readable storage error on failure."""
        try:
            with path.open("a", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=columns, extrasaction="raise")
                writer.writerow(row)
                csv_file.flush()
        except (OSError, csv.Error) as exc:
            raise DataIntegrityError(f"Could not append research data to {path}: {exc}") from exc
