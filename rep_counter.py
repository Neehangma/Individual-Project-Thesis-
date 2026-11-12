"""State-based elbow repetition counting with interruption recovery."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class RepRecord:
    """Measurements recorded when a complete flexion-extension rep is detected."""

    rep_number: int
    set_number: int
    min_angle: float
    max_angle: float
    duration_seconds: float
    target_range_reached: bool


class RepCounter:
    """Count complete extended-flexed-extended cycles using valid angle samples.

    Movement starts only after the elbow is observed at the extended threshold.
    A candidate is discarded on tracking loss, pause, or reversal before flexion.
    The cycle's phase state supplies hysteresis: an angle cannot complete a rep
    until it has passed through both the flexed and extending states.
    """

    def __init__(
        self,
        *,
        flexed_threshold: float,
        extended_threshold: float,
        reps_per_set: int,
        total_sets: int,
        rest_seconds: int,
        target_min_angle: float,
        target_max_angle: float,
        pause_timeout_seconds: float = 5.0,
    ) -> None:
        if not 0 < flexed_threshold < extended_threshold < 180:
            raise ValueError("Movement thresholds must satisfy 0 < flexed < extended < 180")
        if reps_per_set < 1 or total_sets < 1 or rest_seconds < 0:
            raise ValueError("Set, repetition, and rest settings must be non-negative")
        if not 0 <= target_min_angle < target_max_angle <= 180:
            raise ValueError("Target angle range must be within 0..180 degrees")
        if pause_timeout_seconds <= 0:
            raise ValueError("Pause timeout must be greater than zero")
        self.flexed_threshold = flexed_threshold
        self.extended_threshold = extended_threshold
        self.reps_per_set = reps_per_set
        self.total_sets = total_sets
        self.rest_seconds = rest_seconds
        self.target_min_angle = target_min_angle
        self.target_max_angle = target_max_angle
        self.pause_timeout_seconds = pause_timeout_seconds
        self.reset_progress()

    def reset_progress(self) -> None:
        """Reset visible set and repetition progress without deleting session data."""
        self.phase = "Extended"
        self.state = "READY"
        self.set_number = 1
        self.reps_in_set = 0
        self.total_reps = 0
        self.is_complete = False
        self._armed = False
        self._rep_started_at: float | None = None
        self._rep_min: float | None = None
        self._rep_max: float | None = None
        self._reached_flexion = False
        self._rest_until = 0.0
        self._tracking_paused = False
        self._pause_started_at: float | None = None
        self.partial_movement = False

    @property
    def is_armed(self) -> bool:
        """Whether an extended starting position has been observed."""
        return self._armed

    def rest_remaining(self, now: float) -> int:
        """Return whole seconds remaining in the inter-set rest, rounded up."""
        return max(0, math.ceil(self._rest_until - now))

    def shift_time(self, paused_seconds: float) -> None:
        """Exclude a user-paused interval from rep timing, rest, and pause timeout."""
        if paused_seconds <= 0:
            return
        if self._rep_started_at is not None:
            self._rep_started_at += paused_seconds
        if self._rest_until:
            self._rest_until += paused_seconds
        if self._pause_started_at is not None:
            self._pause_started_at += paused_seconds

    def invalidate_tracking(self, now: float) -> None:
        """Pause counting and discard any incomplete movement when tracking is bad."""
        if self.is_complete:
            self.state = "COMPLETED"
            return
        if not self._tracking_paused:
            self._tracking_paused = True
            self._pause_started_at = now
            self._armed = False
            self._cancel_current_rep()
        if self._pause_started_at is not None and now - self._pause_started_at >= self.pause_timeout_seconds:
            self.state = "READY"
        else:
            self.state = "PAUSED"

    def update(self, angle: float, now: float) -> RepRecord | None:
        """Consume one valid smoothed angle, changing state or returning one rep."""
        self.partial_movement = False
        if self.is_complete:
            self.state = "COMPLETED"
            return None

        if self._tracking_paused:
            self._tracking_paused = False
            self._pause_started_at = None
            self._armed = False
            self._cancel_current_rep()
            self.state = "READY"

        if now < self._rest_until:
            self.state = "READY"
            return None
        if self._rest_until:
            self._rest_until = 0.0

        if not self._armed:
            self.state = "READY"
            if angle >= self.extended_threshold:
                self._armed = True
                self.phase = "Extended"
            elif angle <= self.flexed_threshold:
                self.phase = "Flexed"
            else:
                self.phase = "Flexing"
            return None

        if self.phase == "Extended":
            if angle < self.extended_threshold:
                self.phase = "Flexing"
                self.state = "FLEXION"
                self._rep_started_at = now
                self._rep_min = self._rep_max = angle
                self._reached_flexion = angle <= self.flexed_threshold
                if self._reached_flexion:
                    self.phase = "Flexed"
        elif self.phase == "Flexing":
            self.state = "FLEXION"
            self._update_extrema(angle)
            if angle <= self.flexed_threshold:
                self.phase = "Flexed"
                self._reached_flexion = True
            elif angle >= self.extended_threshold:
                self._cancel_current_rep()
                self.phase = "Extended"
                self.state = "READY"
                self.partial_movement = True
        elif self.phase == "Flexed":
            self.state = "FLEXION"
            self._update_extrema(angle)
            if angle > self.flexed_threshold:
                self.phase = "Extending"
                self.state = "EXTENSION"
        elif self.phase == "Extending":
            self.state = "EXTENSION"
            self._update_extrema(angle)
            if angle <= self.flexed_threshold:
                self.phase = "Flexed"
                self.state = "FLEXION"
            elif angle >= self.extended_threshold and self._reached_flexion:
                return self._complete_rep(now)
        return None

    def _update_extrema(self, angle: float) -> None:
        """Include one valid angle in the active repetition's observed range."""
        if self._rep_min is not None:
            self._rep_min = min(self._rep_min, angle)
        if self._rep_max is not None:
            self._rep_max = max(self._rep_max, angle)

    def _cancel_current_rep(self) -> None:
        """Drop any incomplete candidate without changing completed repetitions."""
        self._rep_started_at = None
        self._rep_min = None
        self._rep_max = None
        self._reached_flexion = False

    def _complete_rep(self, now: float) -> RepRecord:
        """Finalize a complete rep, advance set progress, and start rest if due."""
        started_at = self._rep_started_at if self._rep_started_at is not None else now
        minimum = self._rep_min if self._rep_min is not None else 0.0
        maximum = self._rep_max if self._rep_max is not None else 0.0
        self.total_reps += 1
        self.reps_in_set += 1
        record = RepRecord(
            rep_number=self.total_reps,
            set_number=self.set_number,
            min_angle=minimum,
            max_angle=maximum,
            duration_seconds=max(0.0, now - started_at),
            target_range_reached=(minimum <= self.target_min_angle
                                  and maximum >= self.target_max_angle),
        )
        self._cancel_current_rep()
        self.phase = "Extended"
        self.state = "READY"
        if self.reps_in_set >= self.reps_per_set:
            if self.set_number >= self.total_sets:
                self.is_complete = True
                self.state = "COMPLETED"
            else:
                self.set_number += 1
                self.reps_in_set = 0
                self._rest_until = now + self.rest_seconds
        return record
