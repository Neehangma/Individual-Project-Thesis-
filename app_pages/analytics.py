"""Research dashboard for filtered descriptive analysis of saved CSV data."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from research_analytics import (
    calculate_analytics,
    export_csv,
    filter_research_data,
    load_research_data,
    duration_by_session,
    repetition_progress_by_session,
    session_metadata,
    session_table,
    tracking_by_session,
)


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CHART_COLORS = ["#168C91", "#5AAE9F", "#8CC7B7", "#F0A45D", "#637C89", "#B6C5CA"]


def _metric_text(value, *, suffix: str = "", decimals: int = 1) -> str:
    """Format a calculated metric for a KPI card, marking unavailable as N/A."""
    if value is None or pd.isna(value):
        return "N/A"
    if isinstance(value, int):
        return f"{value:,}{suffix}"
    return f"{float(value):,.{decimals}f}{suffix}"


def _style_chart(figure):
    """Apply consistent thesis-friendly chart colors and spacing."""
    figure.update_layout(
        template="plotly_white",
        colorway=CHART_COLORS,
        margin=dict(l=18, r=16, t=42, b=20),
        legend_title_text="",
        font=dict(color="#263940"),
    )
    figure.update_xaxes(showgrid=False)
    figure.update_yaxes(gridcolor="#E5ECEE", zerolinecolor="#D4E0E2")
    return figure


st.title("Research Analytics")
st.caption("Descriptive analysis of saved, anonymised exercise sessions and pose observations.")

# Reload on each app run so new CSV rows become visible without cache invalidation.
summaries, observations = load_research_data(DATA_DIR)
metadata = session_metadata(summaries, observations)

if metadata.empty:
    st.info("No research records are available yet. Complete or start a monitored session to collect CSV observations.")
    st.caption("Expected files: data/session_summaries.csv and data/session_observations.csv.")

participant_options = sorted(item for item in metadata["participant_id"].unique() if item)
session_options = sorted(item for item in metadata["session_id"].unique() if item)
arm_options = sorted(item for item in metadata["arm_side"].unique() if item)
status_options = sorted(item for item in metadata["session_status"].unique() if item)

with st.container(border=True):
    st.subheader("Filters")
    filter_row_one = st.columns([1.2, 1.4, 0.8, 1.0])
    selected_participants = filter_row_one[0].multiselect(
        "Participant ID", participant_options, placeholder="All participants", key="analytics_participants"
    )
    selected_sessions = filter_row_one[1].multiselect(
        "Session ID", session_options, placeholder="All sessions", key="analytics_sessions"
    )
    selected_arms = filter_row_one[2].multiselect(
        "Arm", arm_options, placeholder="Both / all", key="analytics_arms"
    )
    selected_statuses = filter_row_one[3].multiselect(
        "Session status", status_options, placeholder="All statuses", key="analytics_statuses"
    )

    available_dates = metadata["_session_date"].dropna()
    if not available_dates.empty:
        min_date: date = available_dates.min().date()
        max_date: date = available_dates.max().date()
        chosen_dates = st.date_input(
            "Session date range", value=(min_date, max_date), min_value=min_date, max_value=max_date,
            key="analytics_date_range",
        )
        if isinstance(chosen_dates, (tuple, list)):
            date_range = (chosen_dates[0] if chosen_dates else None,
                          chosen_dates[1] if len(chosen_dates) > 1 else (chosen_dates[0] if chosen_dates else None))
        else:
            date_range = (chosen_dates, chosen_dates) if chosen_dates else (None, None)
    else:
        st.date_input("Session date range", value=None, disabled=True, key="analytics_date_range_empty")
        date_range = None

filtered_summaries, filtered_observations = filter_research_data(
    summaries,
    observations,
    participant_ids=selected_participants,
    session_ids=selected_sessions,
    arm_sides=selected_arms,
    statuses=selected_statuses,
    date_range=date_range,
)
metrics = calculate_analytics(filtered_summaries, filtered_observations)

st.subheader("Research overview")
overview = st.columns(6)
overview[0].metric("Participants", _metric_text(metrics["total_participants"], decimals=0))
overview[1].metric("Exercise sessions", _metric_text(metrics["total_sessions"], decimals=0))
overview[2].metric("Completed sessions", _metric_text(metrics["completed_sessions"], decimals=0))
overview[3].metric("Interrupted sessions", _metric_text(metrics["interrupted_sessions"], decimals=0))
overview[4].metric("Avg. completed reps", _metric_text(metrics["average_completed_repetitions"]))
overview[5].metric(
    "Avg. duration", _metric_text(metrics["average_session_duration_seconds"], suffix=" s")
)

st.subheader("System performance evaluation")
performance = st.columns(4)
rate = metrics["pose_tracking_success_rate"]
performance[0].metric("Pose tracking success", _metric_text(rate * 100 if rate is not None else None, suffix="%"))
performance[1].metric("Valid angle observations", _metric_text(metrics["valid_angle_observations"], decimals=0))
performance[2].metric("Invalid angle observations", _metric_text(metrics["invalid_angle_observations"], decimals=0))
completion_rate = metrics["session_completion_rate"]
performance[3].metric(
    "Session completion rate", _metric_text(completion_rate * 100 if completion_rate is not None else None, suffix="%")
)
detail_metrics = st.columns(4)
detail_metrics[0].metric("Average elbow angle", _metric_text(metrics["average_elbow_angle"], suffix="°"))
detail_metrics[1].metric("Minimum elbow angle", _metric_text(metrics["minimum_elbow_angle"], suffix="°"))
detail_metrics[2].metric("Maximum elbow angle", _metric_text(metrics["maximum_elbow_angle"], suffix="°"))
detail_metrics[3].metric("Completed repetitions", _metric_text(metrics["completed_repetitions"], decimals=0))
st.caption(
    "Incomplete movements: " + _metric_text(metrics["incomplete_movements"], decimals=0)
    + ". Angle statistics exclude blank, non-numeric, non-finite, and out-of-range observations."
)
st.info(
    "These are descriptive system measurements. Angle accuracy requires independent ground-truth angle references; "
    "repetition-count accuracy requires manually verified counts. Neither accuracy metric is calculated without those references."
)

st.subheader("Exercise performance analysis")
chart_left, chart_right = st.columns(2, gap="large")
with chart_left:
    st.markdown("#### Elbow angle over time")
    angle_sessions = sorted(
        filtered_observations.loc[filtered_observations["elbow_angle"].notna(), "session_id"].unique().tolist()
    )
    if angle_sessions:
        angle_session_id = st.selectbox("Session for angle chart", angle_sessions, key="analytics_angle_session")
        angle_frame = filtered_observations.loc[
            (filtered_observations["session_id"] == angle_session_id)
            & filtered_observations["elbow_angle"].notna()
        ].sort_values("_timestamp")
        if "elapsed_seconds" in angle_frame and angle_frame["elapsed_seconds"].notna().any():
            fig = px.line(
                angle_frame, x="elapsed_seconds", y="elbow_angle", markers=False,
                labels={"elapsed_seconds": "Elapsed time (seconds)", "elbow_angle": "Angle (degrees)"},
            )
            st.plotly_chart(_style_chart(fig), width="stretch")
        else:
            st.info("Valid angles exist, but elapsed-time values are unavailable for this session.")
    else:
        st.info("No valid angle measurements are available for the current filters.")

with chart_right:
    st.markdown("#### Repetition progress")
    repetition_data = repetition_progress_by_session(filtered_summaries, filtered_observations)
    if not repetition_data.empty:
        fig = px.bar(
            repetition_data, x="session_id", y=["completed_repetitions", "target_repetitions"],
            barmode="group", hover_data=["source"],
            labels={"session_id": "Session", "value": "Repetitions", "variable": "Measure", "source": "Record source"},
        )
        st.plotly_chart(_style_chart(fig), width="stretch")
    else:
        st.info("No saved session summaries with repetition targets are available for these filters.")

chart_left, chart_right = st.columns(2, gap="large")
with chart_left:
    st.markdown("#### Session duration analysis")
    duration_data = duration_by_session(filtered_summaries, filtered_observations)
    if not duration_data.empty:
        fig = px.bar(
            duration_data, x="session_id", y="duration_seconds", color="source",
            labels={"session_id": "Session", "duration_seconds": "Duration (seconds)", "source": "Duration source"},
        )
        st.plotly_chart(_style_chart(fig), width="stretch")
        st.caption("Ended sessions use saved summary duration; observation-only sessions use latest elapsed time.")
    else:
        st.info("No saved session durations are available for these filters.")

with chart_right:
    st.markdown("#### Angle distribution")
    angle_data = filtered_observations.dropna(subset=["elbow_angle"])
    if not angle_data.empty:
        fig = px.histogram(
            angle_data, x="elbow_angle", nbins=24,
            labels={"elbow_angle": "Elbow angle (degrees)", "count": "Observations"},
        )
        st.plotly_chart(_style_chart(fig), width="stretch")
    else:
        st.info("No valid angle observations are available for this distribution.")

chart_left, chart_right = st.columns(2, gap="large")
with chart_left:
    st.markdown("#### Tracking performance")
    tracking_data = tracking_by_session(filtered_observations)
    if not tracking_data.empty:
        tracking_data["success_percent"] = tracking_data["success_rate"] * 100
        fig = px.bar(
            tracking_data, x="session_id", y="success_percent",
            labels={"session_id": "Session", "success_percent": "Stable tracking (%)"},
        )
        fig.update_yaxes(range=[0, 100])
        st.plotly_chart(_style_chart(fig), width="stretch")
    else:
        st.info("Tracking status was not recorded in the selected observation data.")

with chart_right:
    st.markdown("#### Session status distribution")
    status_data = filtered_summaries.loc[
        filtered_summaries["session_status"].isin(["completed", "interrupted", "cancelled"]), "session_status"
    ].value_counts().rename_axis("session_status").reset_index(name="sessions")
    if not status_data.empty:
        fig = px.pie(status_data, names="session_status", values="sessions", hole=0.55)
        st.plotly_chart(_style_chart(fig), width="stretch")
    else:
        st.info("No saved session status summaries are available for these filters.")

st.subheader("Filtered session records")
table = session_table(filtered_summaries)
if table.empty:
    st.info("No saved session summaries match the selected filters. Observation-only sessions remain available in the charts and observation export.")
else:
    display_table = table.copy(deep=True)
    display_table["tracking_success_rate"] = display_table["tracking_success_rate"].map(
        lambda value: f"{float(value) * 100:.1f}%" if pd.notna(value) else "N/A"
    )
    st.dataframe(display_table, width="stretch", hide_index=True)

export_col_one, export_col_two = st.columns(2)
export_col_one.download_button(
    "Download filtered session summaries",
    data=export_csv(table),
    file_name="filtered_session_summaries.csv",
    mime="text/csv",
    width="stretch",
)
export_col_two.download_button(
    "Download filtered observations",
    data=export_csv(filtered_observations.drop(columns=["_timestamp", "_timestamp_date"], errors="ignore")),
    file_name="filtered_session_observations.csv",
    mime="text/csv",
    width="stretch",
)
