"""Tests for the frame and repetition measurements consumed by later reports."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from rep_counter import RepRecord
from session_logger import SessionLogger


class SessionLoggerTests(unittest.TestCase):
    """Verify active observations and settings persist in session files."""

    def test_writes_frame_state_and_json_rep_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = SessionLogger(
                Path(temp_dir),
                target_range=(40, 150),
                settings={"total_sets": 2, "reps_per_set_target": 5},
            )
            logger.log_frame(
                elapsed_time=0.25,
                arm_side="LEFT",
                raw_angle=149.3,
                smoothed_angle=148.9,
                visibility=0.91,
                phase="Flexing",
                movement_state="FLEXION",
                rep_count=0,
                set_number=1,
                valid_frame=True,
            )
            logger.log_frame(
                elapsed_time=0.50,
                arm_side=None,
                raw_angle=None,
                smoothed_angle=None,
                visibility=0.20,
                phase="Flexing",
                movement_state="PAUSED",
                rep_count=0,
                set_number=1,
                valid_frame=False,
            )
            logger.log_rep(RepRecord(1, 1, 38.5, 155.2, 2.4, True))
            csv_path, json_path = logger.finish(duration_seconds=0.5)

            with csv_path.open(newline="", encoding="utf-8") as frame_file:
                frames = list(csv.DictReader(frame_file))
            summary = json.loads(json_path.read_text(encoding="utf-8"))

            self.assertEqual(len(frames), 2)
            self.assertEqual(frames[0]["movement_state"], "FLEXION")
            self.assertEqual(frames[1]["valid_frame"], "False")
            self.assertEqual(summary["total_reps"], 1)
            self.assertEqual(summary["reps_per_set"], {"1": 1, "2": 0})
            self.assertEqual(summary["target_range_degrees"], {"minimum": 40, "maximum": 150})
            self.assertEqual(summary["repetition_data"][0]["min_angle"], 38.5)
            self.assertEqual(summary["configured_settings"]["reps_per_set_target"], 5)


if __name__ == "__main__":
    unittest.main()
