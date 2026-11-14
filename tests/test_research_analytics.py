"""Tests for analytics calculations and non-destructive filtering/export."""

from __future__ import annotations

from datetime import date
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from research_analytics import (
    calculate_analytics,
    duration_by_session,
    export_csv,
    filter_research_data,
    prepare_research_data,
    read_research_csv,
    repetition_progress_by_session,
    session_table,
    tracking_by_session,
)


def sample_frames():
    """Return synthetic research rows isolated to automated tests."""
    summaries = pd.DataFrame([
        {"session_id": "S1", "participant_id": "P01", "arm_side": "LEFT", "target_repetitions": 10,
         "completed_repetitions": 8, "total_duration_seconds": 60, "session_status": "completed",
         "incomplete_movements": 1, "start_time": "2026-10-01T10:00:00+05:45"},
        {"session_id": "S2", "participant_id": "P01", "arm_side": "RIGHT", "target_repetitions": 10,
         "completed_repetitions": 5, "total_duration_seconds": 90, "session_status": "interrupted",
         "incomplete_movements": 2, "start_time": "2026-10-02T10:00:00+05:45"},
        {"session_id": "S3", "participant_id": "P02", "arm_side": "LEFT", "target_repetitions": 10,
         "completed_repetitions": 0, "total_duration_seconds": 10, "session_status": "cancelled",
         "incomplete_movements": 0, "start_time": "2026-10-03T10:00:00+05:45"},
    ])
    observations = pd.DataFrame([
        {"session_id": "S1", "participant_id": "P01", "arm_side": "LEFT", "timestamp": "2026-10-01T10:00:01+05:45",
         "elapsed_seconds": "1", "elbow_angle": "150", "tracking_status": "Tracking stable"},
        {"session_id": "S1", "participant_id": "P01", "arm_side": "LEFT", "timestamp": "2026-10-01T10:00:02+05:45",
         "elapsed_seconds": "2", "elbow_angle": "90", "tracking_status": "Tracking stable"},
        {"session_id": "S2", "participant_id": "P01", "arm_side": "RIGHT", "timestamp": "2026-10-02T10:00:01+05:45",
         "elapsed_seconds": "1", "elbow_angle": "inf", "tracking_status": "Arm not visible"},
        {"session_id": "S2", "participant_id": "P01", "arm_side": "RIGHT", "timestamp": "2026-10-02T10:00:02+05:45",
         "elapsed_seconds": "2", "elbow_angle": "181", "tracking_status": "Tracking stable"},
        {"session_id": "S3", "participant_id": "P02", "arm_side": "LEFT", "timestamp": "2026-10-03T10:00:01+05:45",
         "elapsed_seconds": "1", "elbow_angle": "", "tracking_status": "Person not detected"},
    ])
    return summaries, observations


class ResearchAnalyticsTests(unittest.TestCase):
    """Cover empty/missing data, angle validity, rates, filters, and exports."""

    def test_metrics_exclude_blank_infinite_and_out_of_range_angles(self):
        summaries, observations = sample_frames()
        metrics = calculate_analytics(summaries, observations)

        self.assertEqual(metrics["total_participants"], 2)
        self.assertEqual(metrics["total_sessions"], 3)
        self.assertEqual(metrics["completed_sessions"], 1)
        self.assertEqual(metrics["interrupted_sessions"], 1)
        self.assertEqual(metrics["cancelled_sessions"], 1)
        self.assertEqual(metrics["valid_angle_observations"], 2)
        self.assertEqual(metrics["invalid_angle_observations"], 3)
        self.assertEqual(metrics["average_elbow_angle"], 120.0)
        self.assertEqual(metrics["minimum_elbow_angle"], 90.0)
        self.assertEqual(metrics["maximum_elbow_angle"], 150.0)
        self.assertEqual(metrics["pose_tracking_success_rate"], 3 / 5)
        self.assertEqual(metrics["average_completed_repetitions"], 13 / 3)
        self.assertEqual(metrics["average_session_duration_seconds"], 160 / 3)
        self.assertEqual(metrics["completed_repetitions"], 13)
        self.assertEqual(metrics["incomplete_movements"], 3)
        self.assertAlmostEqual(metrics["session_completion_rate"], 1 / 3)

    def test_empty_and_missing_columns_are_safe(self):
        empty = pd.DataFrame()
        metrics = calculate_analytics(empty, empty)
        self.assertEqual(metrics["total_sessions"], 0)
        self.assertEqual(metrics["valid_angle_observations"], 0)
        self.assertIsNone(metrics["average_elbow_angle"])
        self.assertIsNone(metrics["pose_tracking_success_rate"])
        self.assertIsNone(metrics["session_completion_rate"])
        table = session_table(empty)
        self.assertTrue(table.empty)
        self.assertEqual(len(table.columns), 11)

        incomplete = pd.DataFrame([{"session_id": "S-only"}])
        sparse_observations = pd.DataFrame([
            {"session_id": "S-only", "elbow_angle": "45", "completed_repetitions": 4}
        ])
        normalized_summaries, normalized_observations = prepare_research_data(incomplete, sparse_observations)
        sparse_metrics = calculate_analytics(normalized_summaries, normalized_observations)
        self.assertEqual(sparse_metrics["valid_angle_observations"], 1)
        self.assertIsNone(sparse_metrics["pose_tracking_success_rate"])
        self.assertIsNone(sparse_metrics["average_session_duration_seconds"])
        self.assertIsNone(sparse_metrics["completed_sessions"])
        self.assertIsNone(sparse_metrics["interrupted_sessions"])
        self.assertEqual(sparse_metrics["completed_repetitions"], 4)
        self.assertEqual(sparse_metrics["average_completed_repetitions"], 4.0)

    def test_filters_apply_participant_session_arm_status_and_date(self):
        summaries, observations = sample_frames()
        filtered_summaries, filtered_observations = filter_research_data(
            summaries,
            observations,
            participant_ids=["P01"],
            arm_sides=["right"],
            statuses=["interrupted"],
            date_range=(date(2026, 10, 2), date(2026, 10, 2)),
        )
        self.assertEqual(filtered_summaries["session_id"].tolist(), ["S2"])
        self.assertEqual(set(filtered_observations["session_id"]), {"S2"})

        selected, selected_observations = filter_research_data(summaries, observations, session_ids=["S3"])
        self.assertEqual(selected["participant_id"].tolist(), ["P02"])
        self.assertEqual(set(selected_observations["session_id"]), {"S3"})

    def test_tracking_aggregation_and_export_do_not_mutate_source(self):
        summaries, observations = sample_frames()
        before = observations.copy(deep=True)
        duplicated = pd.concat([observations, observations.iloc[[0]]], ignore_index=True)
        normalized_observations = prepare_research_data(summaries, duplicated)[1]
        self.assertEqual(len(normalized_observations), len(observations))
        tracking = tracking_by_session(normalized_observations).set_index("session_id")
        self.assertEqual(tracking.loc["S1", "successful_observations"], 2)
        self.assertEqual(tracking.loc["S1", "total_observations"], 2)
        self.assertEqual(tracking.loc["S1", "success_rate"], 1.0)

        filtered_summaries, filtered_observations = filter_research_data(
            summaries, observations, participant_ids=["P01"]
        )
        csv_output = export_csv(session_table(filtered_summaries)).decode("utf-8")
        observation_output = export_csv(filtered_observations).decode("utf-8")
        self.assertTrue(csv_output.startswith("session_id,participant_id,arm_side"))
        self.assertIn("S1", csv_output)
        self.assertNotIn("S3", csv_output)
        self.assertIn("S1", observation_output)
        self.assertNotIn("S3", observation_output)
        pd.testing.assert_frame_equal(observations, before)

    def test_observation_only_session_uses_latest_cumulative_reps_and_elapsed_time(self):
        observations = pd.DataFrame([
            {"session_id": "LIVE1", "completed_repetitions": 0, "target_repetitions": 5, "elapsed_seconds": 1},
            {"session_id": "LIVE1", "completed_repetitions": 1, "target_repetitions": 5, "elapsed_seconds": 4},
            {"session_id": "LIVE1", "completed_repetitions": 2, "target_repetitions": 5, "elapsed_seconds": 8},
        ])
        progress = repetition_progress_by_session(pd.DataFrame(), observations)
        durations = duration_by_session(pd.DataFrame(), observations)
        self.assertEqual(len(progress), 1)
        self.assertEqual(progress.iloc[0]["completed_repetitions"], 2)
        self.assertEqual(progress.iloc[0]["target_repetitions"], 5)
        self.assertEqual(progress.iloc[0]["source"], "Latest observation")
        self.assertEqual(durations.iloc[0]["duration_seconds"], 8)
        self.assertEqual(durations.iloc[0]["source"], "Observed elapsed")

    def test_missing_and_empty_csv_files_load_as_empty(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertTrue(read_research_csv(root / "not-created.csv").empty)
            empty_csv = root / "empty.csv"
            empty_csv.write_text("", encoding="utf-8")
            self.assertTrue(read_research_csv(empty_csv).empty)


if __name__ == "__main__":
    unittest.main()
