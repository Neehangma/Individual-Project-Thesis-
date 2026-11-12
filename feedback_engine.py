"""State-aware short feedback messages for the live capture screen."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FeedbackEngine:
    """Select concise guidance from tracking, movement, and session state."""

    event_message_seconds: float = 2.0

    def __post_init__(self) -> None:
        self._event_message = ""
        self._event_until = 0.0

    def message(
        self,
        *,
        tracking_valid: bool,
        movement_state: str,
        phase: str,
        arm_ready: bool,
        now: float,
        rep_number: int | None = None,
        partial_movement: bool = False,
        session_paused: bool = False,
    ) -> str:
        """Return a live message that corresponds to the current measured state."""
        if session_paused:
            return "Session paused. Press P to resume."
        if movement_state == "COMPLETED":
            return "Target reached. Session complete."
        if not tracking_valid or movement_state == "PAUSED":
            return "Tracking lost. Reposition yourself."

        if rep_number is not None:
            self._event_message = f"Repetition completed: {rep_number}"
            self._event_until = now + self.event_message_seconds
        elif partial_movement:
            self._event_message = "Partial movement detected."
            self._event_until = now + self.event_message_seconds
        if self._event_message and now < self._event_until:
            return self._event_message

        if movement_state == "FLEXION":
            return "Bend your elbow."
        if movement_state == "EXTENSION":
            return "Slowly extend your elbow."
        if movement_state == "READY":
            return "Arm detected. Get ready." if arm_ready else "Extend your elbow to begin."
        return "Position your arm so the shoulder, elbow and wrist are visible."
