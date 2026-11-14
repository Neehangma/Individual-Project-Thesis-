"""Pure data loading, cleaning, filtering, and metrics for research analytics."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


SESSION_TABLE_COLUMNS = [
    "session_id",
    "participant_id",
    "arm_side",
    "target_repetitions",
    "completed_repetitions",
    "total_duration_seconds",
    "average_elbow_angle",
    "minimum_elbow_angle",
    "maximum_elbow_angle",
    "tracking_success_rate",
    "session_status",
]

SUMMARY_FILE = "session_summaries.csv"
OBSERVATION_FILE = "session_observations.csv"


def read_research_csv(path: Path | str) -> pd.DataFrame:
    """Read an existing CSV without changing it; missing or malformed files are empty."""
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except (OSError, UnicodeError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()


def load_research_data(data_dir: Path | str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load session summaries and observations from the project's CSV directory."""
    root = Path(data_dir)
    return (
        _prepare_summaries(read_research_csv(root / SUMMARY_FILE)),
        _prepare_observations(read_research_csv(root / OBSERVATION_FILE)),
    )


def prepare_research_data(
    summaries: pd.DataFrame, observations: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return cleaned copies of both tables, preserving the caller's source frames."""
    return _prepare_summaries(summaries), _prepare_observations(observations)


def _prepare_summaries(source: pd.DataFrame) -> pd.DataFrame:
    """Normalize summary columns and remove duplicate session summaries by ID."""
    frame = source.copy(deep=True)
    for column in SESSION_TABLE_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    frame["session_id"] = _clean_text(frame["session_id"])
    frame["participant_id"] = _clean_text(frame["participant_id"])
    frame["arm_side"] = _clean_text(frame["arm_side"]).str.upper()
    frame["session_status"] = _clean_text(frame["session_status"]).str.lower()
    for column in (
        "target_repetitions", "completed_repetitions", "total_duration_seconds",
        "average_elbow_angle", "minimum_elbow_angle", "maximum_elbow_angle",
        "tracking_success_rate", "incomplete_movements",
    ):
        if column not in frame.columns:
            frame[column] = pd.NA
        frame[column] = _numeric(frame[column])
    for column in ("target_repetitions",):
        frame.loc[frame[column] <= 0, column] = np.nan
    for column in ("completed_repetitions", "total_duration_seconds", "incomplete_movements"):
        frame.loc[frame[column] < 0, column] = np.nan
    for column in ("average_elbow_angle", "minimum_elbow_angle", "maximum_elbow_angle"):
        frame.loc[~frame[column].between(0.0, 180.0), column] = np.nan
    frame.loc[~frame["tracking_success_rate"].between(0.0, 1.0), "tracking_success_rate"] = np.nan
    if "start_time" not in frame.columns:
        frame["start_time"] = pd.NA
    # Keep the calendar date in the timestamp's recorded local offset for date filters.
    frame["_session_date"] = pd.to_datetime(
        frame["start_time"].astype(str).str.slice(0, 10), format="%Y-%m-%d", errors="coerce"
    )
    frame = frame.loc[frame["session_id"].ne("")].copy()
    return frame.drop_duplicates(subset=["session_id"], keep="last").reset_index(drop=True)


def _prepare_observations(source: pd.DataFrame) -> pd.DataFrame:
    """Normalize observation IDs, timestamps, confidence, and valid angle values."""
    frame = source.copy(deep=True)
    for column in ("session_id", "participant_id", "arm_side", "timestamp", "elbow_angle", "pose_confidence", "tracking_status"):
        if column not in frame.columns:
            frame[column] = pd.NA
    frame["session_id"] = _clean_text(frame["session_id"])
    frame["participant_id"] = _clean_text(frame["participant_id"])
    frame["arm_side"] = _clean_text(frame["arm_side"]).str.upper()
    frame["tracking_status"] = _clean_text(frame["tracking_status"])
    frame["elbow_angle"] = _numeric(frame["elbow_angle"])
    # A geometric elbow angle is expected in [0, 180]. Outliers are invalid, never zero-filled.
    frame.loc[~frame["elbow_angle"].between(0.0, 180.0), "elbow_angle"] = np.nan
    frame["pose_confidence"] = _numeric(frame["pose_confidence"])
    frame.loc[~frame["pose_confidence"].between(0.0, 1.0), "pose_confidence"] = np.nan
    frame["elapsed_seconds"] = _numeric(frame["elapsed_seconds"]) if "elapsed_seconds" in frame else np.nan
    frame["completed_repetitions"] = _numeric(frame["completed_repetitions"]) if "completed_repetitions" in frame else np.nan
    frame["target_repetitions"] = _numeric(frame["target_repetitions"]) if "target_repetitions" in frame else np.nan
    frame.loc[frame["elapsed_seconds"] < 0, "elapsed_seconds"] = np.nan
    frame.loc[frame["completed_repetitions"] < 0, "completed_repetitions"] = np.nan
    frame.loc[frame["target_repetitions"] <= 0, "target_repetitions"] = np.nan
    frame["_timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    frame["_timestamp_date"] = pd.to_datetime(
        frame["timestamp"].astype(str).str.slice(0, 10), format="%Y-%m-%d", errors="coerce"
    )
    frame = frame.loc[frame["session_id"].ne("")].copy()
    # Drop exact duplicated CSV rows in the analytics view; distinct timestamps remain distinct samples.
    return frame.drop_duplicates().reset_index(drop=True)


def _clean_text(series: pd.Series) -> pd.Series:
    """Convert a column to trimmed strings while retaining missing values as empty text."""
    return series.fillna("").astype(str).str.strip()


def _numeric(series: pd.Series) -> pd.Series:
    """Convert numeric values safely and treat infinities as missing measurements."""
    values = pd.to_numeric(series, errors="coerce")
    return values.replace([np.inf, -np.inf], np.nan)


def session_metadata(summaries: pd.DataFrame, observations: pd.DataFrame) -> pd.DataFrame:
    """Combine summary and observation identifiers for filters, including active sessions."""
    summaries, observations = prepare_research_data(summaries, observations)
    summary_by_id = summaries.set_index("session_id", drop=False) if not summaries.empty else pd.DataFrame()
    rows: list[dict[str, Any]] = []
    all_ids = set(summaries.get("session_id", pd.Series(dtype=str)).dropna().astype(str))
    all_ids.update(observations.get("session_id", pd.Series(dtype=str)).dropna().astype(str))
    for session_id in sorted(item for item in all_ids if item):
        summary = summary_by_id.loc[session_id] if not summary_by_id.empty and session_id in summary_by_id.index else None
        group = observations.loc[observations["session_id"] == session_id]

        def value(column: str, fallback: Any = "") -> Any:
            if summary is not None:
                candidate = summary.get(column, fallback)
                if pd.notna(candidate) and str(candidate).strip():
                    return candidate
            if column in group:
                candidates = group[column].dropna()
                candidates = candidates[candidates.astype(str).str.strip().ne("")]
                if not candidates.empty:
                    return candidates.iloc[0]
            return fallback

        session_date = pd.NaT
        if summary is not None:
            session_date = summary.get("_session_date", pd.NaT)
        if pd.isna(session_date) and not group.empty:
            session_date = group["_timestamp_date"].min()
        rows.append({
            "session_id": session_id,
            "participant_id": value("participant_id"),
            "arm_side": str(value("arm_side")).upper(),
            "session_status": str(value("session_status")).lower(),
            "_session_date": session_date,
        })
    return pd.DataFrame(rows, columns=["session_id", "participant_id", "arm_side", "session_status", "_session_date"])


def filter_research_data(
    summaries: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    participant_ids: Iterable[str] | None = None,
    session_ids: Iterable[str] | None = None,
    arm_sides: Iterable[str] | None = None,
    statuses: Iterable[str] | None = None,
    date_range: tuple[date | None, date | None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter copies by session metadata and keep linked observations in sync."""
    summaries, observations = prepare_research_data(summaries, observations)
    participant_ids = _selected(participant_ids)
    session_ids = _selected(session_ids)
    arm_sides = {item.upper() for item in _selected(arm_sides)}
    statuses = {item.lower() for item in _selected(statuses)}
    metadata = session_metadata(summaries, observations)
    keep = pd.Series(True, index=metadata.index)
    if participant_ids:
        keep &= metadata["participant_id"].isin(participant_ids)
    if session_ids:
        keep &= metadata["session_id"].isin(session_ids)
    if arm_sides:
        keep &= metadata["arm_side"].isin(arm_sides)
    if statuses:
        keep &= metadata["session_status"].isin(statuses)
    if date_range:
        start_date, end_date = date_range
        session_dates = pd.to_datetime(metadata["_session_date"], errors="coerce", utc=True)
        if start_date is not None:
            keep &= session_dates.notna() & (session_dates.dt.date >= start_date)
        if end_date is not None:
            keep &= session_dates.notna() & (session_dates.dt.date <= end_date)
    selected_ids = set(metadata.loc[keep, "session_id"])
    filtered_summaries = summaries.loc[summaries["session_id"].isin(selected_ids)].copy(deep=True)
    filtered_observations = observations.loc[observations["session_id"].isin(selected_ids)].copy(deep=True)
    return filtered_summaries.reset_index(drop=True), filtered_observations.reset_index(drop=True)


def _selected(values: Iterable[str] | None) -> set[str]:
    """Normalize optional multi-select values; empty selection means all values."""
    return {str(item).strip() for item in (values or []) if str(item).strip()}


def calculate_analytics(summaries: pd.DataFrame, observations: pd.DataFrame) -> dict[str, Any]:
    """Calculate descriptive research metrics, excluding invalid angle observations."""
    summaries, observations = prepare_research_data(summaries, observations)
    valid_angle = observations["elbow_angle"].notna()
    valid_angles = observations.loc[valid_angle, "elbow_angle"]
    participants = set(summaries["participant_id"].loc[summaries["participant_id"].ne("")])
    participants.update(observations["participant_id"].loc[observations["participant_id"].ne("")])
    all_session_ids = set(summaries["session_id"]) | set(observations["session_id"])
    status_known = summaries["session_status"].isin({"completed", "interrupted", "cancelled"})
    status_sessions = summaries.loc[status_known]
    completed_count = int(summaries["session_status"].eq("completed").sum())
    interrupted_count = int(summaries["session_status"].eq("interrupted").sum())
    cancelled_count = int(summaries["session_status"].eq("cancelled").sum())
    has_known_status = not status_sessions.empty
    known_tracking = observations["tracking_status"].str.strip().ne("")
    if known_tracking.any():
        tracked = observations["tracking_status"].str.strip().str.casefold().eq("tracking stable")
        tracking_rate = float(tracked.loc[known_tracking].sum() / known_tracking.sum())
    else:
        tracking_rate = None
    # Prefer each saved session summary; for an in-progress observation-only
    # session use its latest cumulative count once, rather than summing frame rows.
    repetitions_by_session = summaries.set_index("session_id")["completed_repetitions"].to_dict()
    if not observations.empty:
        observed_counts = observations.groupby("session_id")["completed_repetitions"].max()
        for session_id, count in observed_counts.items():
            if pd.isna(repetitions_by_session.get(session_id)) and pd.notna(count):
                repetitions_by_session[session_id] = count
    completed_values = pd.to_numeric(pd.Series(repetitions_by_session), errors="coerce").dropna()
    duration_values = summaries["total_duration_seconds"].dropna()
    incomplete_values = summaries["incomplete_movements"].dropna()
    return {
        "total_participants": len(participants),
        "total_sessions": len(all_session_ids),
        "completed_sessions": completed_count if has_known_status else None,
        "interrupted_sessions": interrupted_count if has_known_status else None,
        "cancelled_sessions": cancelled_count if has_known_status else None,
        "average_completed_repetitions": float(completed_values.mean()) if not completed_values.empty else None,
        "average_session_duration_seconds": float(duration_values.mean()) if not duration_values.empty else None,
        "pose_tracking_success_rate": tracking_rate,
        "valid_angle_observations": int(valid_angle.sum()),
        "invalid_angle_observations": int(len(observations) - valid_angle.sum()),
        "average_elbow_angle": float(valid_angles.mean()) if not valid_angles.empty else None,
        "minimum_elbow_angle": float(valid_angles.min()) if not valid_angles.empty else None,
        "maximum_elbow_angle": float(valid_angles.max()) if not valid_angles.empty else None,
        "completed_repetitions": int(completed_values.sum()) if not completed_values.empty else 0,
        "incomplete_movements": int(incomplete_values.sum()) if not incomplete_values.empty else None,
        "session_completion_rate": (
            float(completed_count / len(all_session_ids))
            if all_session_ids and len(status_sessions) == len(all_session_ids) else None
        ),
        "sessions_with_known_status": len(status_sessions),
    }


def session_table(summaries: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with the requested summary columns in a stable order."""
    table = summaries.copy(deep=True)
    for column in SESSION_TABLE_COLUMNS:
        if column not in table:
            table[column] = pd.NA
    table = table[SESSION_TABLE_COLUMNS]
    if not table.empty:
        table = table.sort_values("session_id", ascending=False, kind="stable")
    return table.reset_index(drop=True)


def tracking_by_session(observations: pd.DataFrame) -> pd.DataFrame:
    """Aggregate successful stable-tracking observations for each selected session."""
    columns = ["session_id", "successful_observations", "total_observations", "success_rate"]
    if observations.empty or "tracking_status" not in observations:
        return pd.DataFrame(columns=columns)
    frame = observations.copy(deep=True)
    known = frame["tracking_status"].str.strip().ne("")
    frame = frame.loc[known].copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame["_successful"] = frame["tracking_status"].str.strip().str.casefold().eq("tracking stable")
    output = frame.groupby("session_id", dropna=False)["_successful"].agg(["sum", "count"]).reset_index()
    output.columns = ["session_id", "successful_observations", "total_observations"]
    output["success_rate"] = output["successful_observations"] / output["total_observations"]
    return output[columns]


def repetition_progress_by_session(
    summaries: pd.DataFrame, observations: pd.DataFrame
) -> pd.DataFrame:
    """Build one completed/target pair per session, including active observation-only sessions."""
    summaries, observations = prepare_research_data(summaries, observations)
    rows: dict[str, dict[str, Any]] = {}
    for _, row in summaries.iterrows():
        rows[row["session_id"]] = {
            "session_id": row["session_id"],
            "completed_repetitions": row["completed_repetitions"],
            "target_repetitions": row["target_repetitions"],
            "source": "Saved summary",
        }
    if not observations.empty:
        observed = observations.groupby("session_id").agg(
            completed_repetitions=("completed_repetitions", "max"),
            target_repetitions=("target_repetitions", "max"),
        )
        for session_id, values in observed.iterrows():
            current = rows.get(session_id)
            if current is None:
                rows[session_id] = {
                    "session_id": session_id,
                    "completed_repetitions": values["completed_repetitions"],
                    "target_repetitions": values["target_repetitions"],
                    "source": "Latest observation",
                }
            else:
                for column in ("completed_repetitions", "target_repetitions"):
                    if pd.isna(current[column]) and pd.notna(values[column]):
                        current[column] = values[column]
    result = pd.DataFrame(rows.values(), columns=[
        "session_id", "completed_repetitions", "target_repetitions", "source"
    ])
    return result.dropna(subset=["completed_repetitions", "target_repetitions"]).reset_index(drop=True)


def duration_by_session(summaries: pd.DataFrame, observations: pd.DataFrame) -> pd.DataFrame:
    """Return saved durations, falling back to latest observed elapsed time when unfinished."""
    summaries, observations = prepare_research_data(summaries, observations)
    rows: dict[str, dict[str, Any]] = {}
    for _, row in summaries.iterrows():
        duration = row["total_duration_seconds"]
        if pd.notna(duration):
            rows[row["session_id"]] = {
                "session_id": row["session_id"], "duration_seconds": duration, "source": "Saved summary"
            }
    if not observations.empty:
        observed = observations.groupby("session_id")["elapsed_seconds"].max()
        for session_id, elapsed in observed.items():
            if session_id not in rows and pd.notna(elapsed):
                rows[session_id] = {
                    "session_id": session_id, "duration_seconds": elapsed, "source": "Observed elapsed"
                }
    return pd.DataFrame(rows.values(), columns=["session_id", "duration_seconds", "source"])


def export_csv(frame: pd.DataFrame) -> bytes:
    """Serialize a filtered table to UTF-8 CSV bytes without modifying its source."""
    return frame.to_csv(index=False).encode("utf-8")
