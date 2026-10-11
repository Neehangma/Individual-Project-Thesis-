"""CSV frame capture and JSON session summary persistence."""

from __future__ import annotations

import csv
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from rep_counter import RepRecord


FRAME_COLUMNS = (
    "timestamp", "elapsed_time", "arm_side", "raw_angle", "smoothed_angle",
    "visibility", "phase", "rep_count", "set_number", "valid_frame",
)


class SessionLogger:
    """Write one row per active-session frame and summarize measurements on exit."""

    def __init__(
        self,
        output_dir: Path | str | None = None,
        *,
        target_range: tuple[float, float] = (40.0, 150.0),
        settings: dict[str, int | float] | None = None,
    ) -> None:
        """Create a uniquely named session CSV under the local sessions directory."""
        self.output_dir = Path(output_dir) if output_dir else Path(__file__).resolve().parent / "sessions"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        started = datetime.now().astimezone()
        base = started.strftime("session_%Y%m%d_%H%M%S")
        suffix = 0
        while True:
            self.session_id = base if suffix == 0 else f"{base}_{suffix:02d}"
            self.csv_path = self.output_dir / f"{self.session_id}.csv"
            self.json_path = self.output_dir / f"{self.session_id}.json"
            if not self.csv_path.exists() and not self.json_path.exists():
                break
            suffix += 1
        self.start_time = started.isoformat(timespec="seconds")
        self.target_range = target_range
        self.settings = dict(settings or {})
        self._csv_file = self.csv_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._csv_file, fieldnames=FRAME_COLUMNS)
        self._writer.writeheader()
        self._angles: list[float] = []
        self._visibilities: list[float] = []
        self._valid_frame_count = 0
        self._invalid_frame_count = 0
        self.rep_records: list[dict[str, Any]] = []
        self.completed_rep_count = 0
        self._finished = False

    def log_frame(
        self,
        *,
        elapsed_time: float,
        arm_side: str | None,
        raw_angle: float | None,
        smoothed_angle: float | None,
        visibility: float,
        phase: str,
        rep_count: int,
        set_number: int,
        valid_frame: bool,
    ) -> None:
        """Append a timestamped observation and update summary aggregates."""
        if self._finished:
            raise RuntimeError("Cannot log to a finished session")
        timestamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
        self._writer.writerow({
            "timestamp": timestamp,
            "elapsed_time": f"{max(0.0, elapsed_time):.3f}",
            "arm_side": arm_side or "",
            "raw_angle": "" if raw_angle is None else f"{raw_angle:.3f}",
            "smoothed_angle": "" if smoothed_angle is None else f"{smoothed_angle:.3f}",
            "visibility": f"{min(1.0, max(0.0, visibility)):.4f}",
            "phase": phase,
            "rep_count": int(rep_count),
            "set_number": int(set_number),
            "valid_frame": bool(valid_frame),
        })
        self._csv_file.flush()
        self._visibilities.append(min(1.0, max(0.0, visibility)))
        if valid_frame and smoothed_angle is not None:
            self._valid_frame_count += 1
            self._angles.append(float(smoothed_angle))
        else:
            self._invalid_frame_count += 1

    def log_rep(self, record: RepRecord) -> None:
        """Store complete per-repetition metrics for the final JSON summary."""
        self.completed_rep_count += 1
        rep = asdict(record)
        rep["rep_number"] = self.completed_rep_count
        self.rep_records.append(rep)

    def finish(self, *, duration_seconds: float) -> tuple[Path, Path]:
        """Close the measurements CSV and write the computed JSON session summary."""
        if self._finished:
            return self.csv_path, self.json_path
        self._csv_file.close()
        minimum = min(self._angles) if self._angles else None
        maximum = max(self._angles) if self._angles else None
        set_count = int(self.settings.get("total_sets", 0))
        reps_per_set: dict[str, int] = {str(index): 0 for index in range(1, set_count + 1)}
        for rep in self.rep_records:
            key = str(rep["set_number"])
            reps_per_set[key] = reps_per_set.get(key, 0) + 1
        durations = [float(rep["duration_seconds"]) for rep in self.rep_records]
        summary = {
            "session_id": self.session_id,
            "start_time": self.start_time,
            "end_time": datetime.now().astimezone().isoformat(timespec="seconds"),
            "session_duration_seconds": round(max(0.0, duration_seconds), 3),
            "total_reps": self.completed_rep_count,
            "reps_per_set": reps_per_set,
            "minimum_angle": minimum,
            "maximum_angle": maximum,
            "average_angle": (sum(self._angles) / len(self._angles)) if self._angles else None,
            "range_of_motion": None if minimum is None else round(maximum - minimum, 3),
            "average_rep_duration_seconds": (sum(durations) / len(durations)) if durations else None,
            "average_visibility": (sum(self._visibilities) / len(self._visibilities)) if self._visibilities else None,
            "valid_frame_count": self._valid_frame_count,
            "low_visibility_frame_count": self._invalid_frame_count,
            "target_range_degrees": {
                "minimum": self.target_range[0],
                "maximum": self.target_range[1],
            },
            "configured_settings": self.settings,
            "repetition_data": self.rep_records,
        }
        with self.json_path.open("w", encoding="utf-8") as summary_file:
            json.dump(summary, summary_file, indent=2)
            summary_file.write("\n")
        self._finished = True
        return self.csv_path, self.json_path
