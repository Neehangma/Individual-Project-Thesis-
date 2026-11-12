"""Unit tests for the rep counter using synthetic angles only."""

from __future__ import annotations

import unittest

from feedback_engine import FeedbackEngine
from rep_counter import RepCounter


def make_counter(*, target: int = 10, sets: int = 3, timeout: float = 5.0) -> RepCounter:
    """Build a counter with the live prototype's configured movement bounds."""
    return RepCounter(
        flexed_threshold=60,
        extended_threshold=150,
        reps_per_set=target,
        total_sets=sets,
        rest_seconds=0,
        target_min_angle=40,
        target_max_angle=150,
        pause_timeout_seconds=timeout,
    )


def complete_one_rep(counter: RepCounter, start: float = 0.0):
    """Feed one test-only extended-flexed-extended angle cycle."""
    counter.update(160, start)
    counter.update(105, start + 1)
    counter.update(50, start + 2)
    counter.update(90, start + 3)
    return counter.update(160, start + 4)


class RepetitionCounterTests(unittest.TestCase):
    """Exercise valid cycles, interruptions, targets, and state recovery."""

    def test_complete_flexion_extension_counts_once(self):
        counter = make_counter()
        rep = complete_one_rep(counter)
        self.assertIsNotNone(rep)
        self.assertEqual(counter.total_reps, 1)
        self.assertEqual(rep.set_number, 1)
        self.assertEqual(counter.state, "READY")

    def test_partial_movement_does_not_count(self):
        counter = make_counter()
        counter.update(160, 0)
        counter.update(110, 1)
        counter.update(130, 2)
        self.assertIsNone(counter.update(160, 3))
        self.assertTrue(counter.partial_movement)
        self.assertEqual(counter.total_reps, 0)

    def test_holding_fixed_angle_does_not_repeat(self):
        counter = make_counter()
        for now, angle in enumerate([160] + [100] * 40):
            self.assertIsNone(counter.update(angle, float(now)))
        self.assertEqual(counter.total_reps, 0)

    def test_noise_around_extension_threshold_does_not_count(self):
        counter = make_counter()
        for now, angle in enumerate([160, 149, 151, 149, 151, 160]):
            self.assertIsNone(counter.update(angle, float(now)))
        self.assertEqual(counter.total_reps, 0)

    def test_multiple_consecutive_repetitions(self):
        counter = make_counter(target=5)
        self.assertIsNotNone(complete_one_rep(counter, 0))
        self.assertIsNotNone(complete_one_rep(counter, 5))
        self.assertEqual(counter.total_reps, 2)

    def test_tracking_loss_discards_in_progress_repetition(self):
        counter = make_counter()
        counter.update(160, 0)
        counter.update(100, 1)
        counter.update(50, 2)
        counter.invalidate_tracking(2.5)
        self.assertEqual(counter.state, "PAUSED")
        self.assertIsNone(counter.update(160, 3))
        self.assertEqual(counter.total_reps, 0)
        self.assertIsNotNone(complete_one_rep(counter, 4))

    def test_pause_resume_requires_a_fresh_extended_start(self):
        counter = make_counter()
        counter.update(160, 0)
        counter.update(100, 1)
        counter.invalidate_tracking(1.5)
        self.assertEqual(counter.state, "PAUSED")
        self.assertIsNone(counter.update(100, 2))
        self.assertEqual(counter.state, "READY")
        self.assertIsNone(counter.update(160, 3))
        self.assertIsNotNone(complete_one_rep(counter, 4))

    def test_tracking_pause_timeout_recovers_to_ready(self):
        counter = make_counter(timeout=2.0)
        counter.invalidate_tracking(1.0)
        self.assertEqual(counter.state, "PAUSED")
        counter.invalidate_tracking(3.1)
        self.assertEqual(counter.state, "READY")
        self.assertEqual(counter.total_reps, 0)

    def test_target_stops_additional_repetitions(self):
        counter = make_counter(target=1, sets=1)
        self.assertIsNotNone(complete_one_rep(counter))
        self.assertTrue(counter.is_complete)
        self.assertEqual(counter.state, "COMPLETED")
        self.assertIsNone(complete_one_rep(counter, 5))
        self.assertEqual(counter.total_reps, 1)

    def test_early_session_can_end_below_target(self):
        counter = make_counter(target=5)
        self.assertIsNotNone(complete_one_rep(counter))
        self.assertFalse(counter.is_complete)
        self.assertEqual(counter.total_reps, 1)

    def test_feedback_matches_tracking_and_movement_state(self):
        feedback = FeedbackEngine()
        self.assertEqual(
            feedback.message(tracking_valid=True, movement_state="READY", phase="Flexed", arm_ready=False, now=0),
            "Extend your elbow to begin.",
        )
        self.assertEqual(
            feedback.message(tracking_valid=False, movement_state="PAUSED", phase="Flexing", arm_ready=False, now=0),
            "Tracking lost. Reposition yourself.",
        )
        self.assertEqual(
            feedback.message(tracking_valid=True, movement_state="FLEXION", phase="Flexing", arm_ready=True, now=1),
            "Bend your elbow.",
        )
        self.assertEqual(
            feedback.message(tracking_valid=True, movement_state="EXTENSION", phase="Extending", arm_ready=True, now=2),
            "Slowly extend your elbow.",
        )
        self.assertEqual(
            feedback.message(tracking_valid=True, movement_state="READY", phase="Extended", arm_ready=True, now=3, rep_number=3),
            "Repetition completed: 3",
        )
        self.assertEqual(
            feedback.message(tracking_valid=True, movement_state="READY", phase="Extended", arm_ready=True, now=4, partial_movement=True),
            "Partial movement detected.",
        )
        self.assertEqual(
            feedback.message(tracking_valid=True, movement_state="READY", phase="Extended", arm_ready=True, now=5, session_paused=True),
            "Session paused. Press P to resume.",
        )
        self.assertEqual(
            feedback.message(tracking_valid=True, movement_state="COMPLETED", phase="Extended", arm_ready=True, now=6),
            "Target reached. Session complete.",
        )


if __name__ == "__main__":
    unittest.main()
