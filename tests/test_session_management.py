"""Unit tests for session lifecycle and persisted research observations."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest

from data_logger import DataIntegrityError, DataLogger
from session_manager import SessionManager, SessionStateError


class SessionManagerTests(unittest.TestCase):
    """Check unique IDs, state changes, active duration, and final statuses."""

    def test_unique_sessions_and_duplicate_start_prevention(self):
        manager = SessionManager()
        started = datetime(2026, 11, 12, 12, 0, tzinfo=timezone.utc)
        first = manager.start_session(
            participant_id="P001", arm_side="LEFT", target_repetitions=2,
            started_at=started, monotonic_now=10,
        )
        with self.assertRaises(SessionStateError):
            manager.start_session(participant_id="P002", arm_side="RIGHT", target_repetitions=1,
                                  started_at=started, monotonic_now=11)
        manager.end_session(completed_repetitions=0, valid_observations=0,
                            ended_at=started, monotonic_now=12)
        second = manager.start_session(
            participant_id="P002", arm_side="RIGHT", target_repetitions=1,
            started_at=started, monotonic_now=13,
        )
        self.assertNotEqual(first.session_id, second.session_id)

    def test_pause_resume_excludes_paused_time(self):
        manager = SessionManager()
        session = manager.start_session(participant_id="P007", arm_side="RIGHT",
                                        target_repetitions=3, monotonic_now=10)
        self.assertEqual(manager.elapsed_seconds(15), 5)
        manager.pause(15)
        self.assertTrue(session.paused)
        self.assertEqual(manager.elapsed_seconds(25), 5)
        manager.resume(25)
        self.assertFalse(session.paused)
        self.assertEqual(manager.elapsed_seconds(28), 8)
        ended = manager.end_session(completed_repetitions=1, valid_observations=4,
                                    monotonic_now=30)
        self.assertEqual(ended.duration_seconds, 10)
        self.assertEqual(ended.status, "interrupted")
        with self.assertRaises(SessionStateError):
            manager.pause(31)

    def test_completed_and_cancelled_statuses(self):
        manager = SessionManager()
        completed = manager.start_session(participant_id="P101", arm_side="LEFT",
                                          target_repetitions=1, monotonic_now=0)
        self.assertEqual(manager.end_session(completed_repetitions=1, valid_observations=1,
                                             monotonic_now=2).status, "completed")
        cancelled = manager.start_session(participant_id="P102", arm_side="RIGHT",
                                          target_repetitions=10, monotonic_now=3)
        self.assertEqual(manager.end_session(completed_repetitions=0, valid_observations=0,
                                             monotonic_now=4).status, "cancelled")

    def test_anonymous_id_arm_and_target_validation(self):
        manager = SessionManager()
        for participant_id, arm, target in [("name with spaces", "LEFT", 1),
                                            ("P1", "BOTH", 1), ("P1", "RIGHT", 0)]:
            with self.assertRaises(ValueError):
                manager.start_session(participant_id=participant_id, arm_side=arm,
                                      target_repetitions=target, monotonic_now=0)


class DataLoggerTests(unittest.TestCase):
    """Verify valid-only statistics, CSV append, restart persistence, and locking."""

    def test_two_sessions_append_and_invalid_angle_is_blank_after_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = SessionManager()
            logger = DataLogger(Path(directory), min_confidence=0.60)
            first = manager.start_session(participant_id="R001", arm_side="LEFT",
                                          target_repetitions=1, monotonic_now=0)
            logger.start_session(first)
            self.assertTrue(logger.record_observation(
                first, elapsed_seconds=0.1, elbow_angle=92.5, pose_confidence=0.9,
                tracking_status="Tracking stable", movement_state="FLEXION",
                completed_repetitions=0, feedback_message="Bend your elbow.",
            ))
            self.assertFalse(logger.record_observation(
                first, elapsed_seconds=0.2, elbow_angle=77.0, pose_confidence=0.3,
                tracking_status="Arm not fully visible", movement_state="PAUSED",
                completed_repetitions=0, feedback_message="Tracking lost.",
            ))
            logger.record_incomplete_movement(first)
            ended_first = manager.end_session(completed_repetitions=0, valid_observations=1,
                                              monotonic_now=1)
            summary_one = logger.finish_session(ended_first, completed_repetitions=0)
            self.assertEqual(summary_one["session_status"], "interrupted")
            self.assertEqual(summary_one["average_elbow_angle"], "92.500")
            self.assertEqual(summary_one["incomplete_movements"], 1)
            with self.assertRaises(DataIntegrityError):
                logger.record_observation(
                    ended_first, elapsed_seconds=1.1, elbow_angle=90, pose_confidence=0.9,
                    tracking_status="Tracking stable", movement_state="READY",
                    completed_repetitions=0, feedback_message="",
                )

            # A new logger instance represents an app restart; old rows stay intact.
            restarted_logger = DataLogger(Path(directory), min_confidence=0.60)
            second = manager.start_session(participant_id="R002", arm_side="RIGHT",
                                           target_repetitions=1, monotonic_now=2)
            restarted_logger.start_session(second)
            self.assertTrue(restarted_logger.record_observation(
                second, elapsed_seconds=0.1, elbow_angle=150.0, pose_confidence=0.95,
                tracking_status="Tracking stable", movement_state="READY",
                completed_repetitions=1, feedback_message="Target reached.",
            ))
            ended_second = manager.end_session(completed_repetitions=1, valid_observations=1,
                                               monotonic_now=3)
            summary_two = restarted_logger.finish_session(ended_second, completed_repetitions=1)
            self.assertEqual(summary_two["session_status"], "completed")

            with (Path(directory) / "session_observations.csv").open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            with (Path(directory) / "session_summaries.csv").open(newline="", encoding="utf-8") as file:
                summaries = list(csv.DictReader(file))
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[1]["elbow_angle"], "")
            self.assertEqual({row["session_id"] for row in rows}, {first.session_id, second.session_id})
            self.assertEqual(len(summaries), 2)
            self.assertEqual([row["session_status"] for row in summaries], ["interrupted", "completed"])

    def test_empty_session_summary_has_blank_angle_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = SessionManager()
            logger = DataLogger(Path(directory))
            session = manager.start_session(participant_id="E001", arm_side="LEFT",
                                            target_repetitions=2, monotonic_now=0)
            logger.start_session(session)
            ended = manager.end_session(completed_repetitions=0, valid_observations=0,
                                        monotonic_now=1)
            summary = logger.finish_session(ended, completed_repetitions=0)
            self.assertEqual(summary["session_status"], "cancelled")
            self.assertEqual(summary["average_elbow_angle"], "")
            self.assertEqual(summary["tracking_success_rate"], "")


if __name__ == "__main__":
    unittest.main()
