"""Session-level results from the locally persisted research CSV files."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OBSERVATIONS_PATH = DATA_DIR / "session_observations.csv"
SUMMARIES_PATH = DATA_DIR / "session_summaries.csv"


def _read_csv(path: Path) -> pd.DataFrame:
    """Read a research CSV safely, returning an empty table when unavailable."""
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()


st.title("Session Results")
summaries = _read_csv(SUMMARIES_PATH)
observations = _read_csv(OBSERVATIONS_PATH)
if summaries.empty or "session_id" not in summaries.columns:
    st.info("No saved sessions yet. Finish a live exercise session to see results here.")
    st.stop()

st.subheader("Session history")
st.dataframe(summaries.sort_values("start_time", ascending=False), width="stretch", hide_index=True)
session_ids = summaries["session_id"].astype(str).tolist()
preferred = st.session_state.get("last_session_id")
default_index = session_ids.index(preferred) if preferred in session_ids else 0
selected_id = st.selectbox("Inspect session", session_ids, index=default_index)
summary = summaries[summaries["session_id"].astype(str) == selected_id].iloc[-1]
st.caption(
    f"Session `{summary.get('session_id', selected_id)}` · Started {summary.get('start_time', '—')} · "
    f"Ended {summary.get('end_time', '—')}"
)
session_col = observations.get("session_id", pd.Series(dtype=str)).astype(str)
session_observations = observations[session_col == selected_id].copy()

metrics = [
    ("Participant", summary.get("participant_id", "—")),
    ("Arm", summary.get("arm_side", "—")),
    ("Exercise", summary.get("exercise_type", "—")),
    ("Status", summary.get("session_status", "—")),
    ("Repetitions", f"{summary.get('completed_repetitions', 0)} / {summary.get('target_repetitions', 0)}"),
    ("Duration", f"{float(summary.get('total_duration_seconds', 0) or 0):.1f} s"),
    ("Average angle", f"{float(summary.get('average_elbow_angle')):.1f}°" if pd.notna(summary.get("average_elbow_angle")) else "—"),
    ("Angle range", f"{float(summary.get('minimum_elbow_angle')):.1f}° – {float(summary.get('maximum_elbow_angle')):.1f}°" if pd.notna(summary.get("minimum_elbow_angle")) and pd.notna(summary.get("maximum_elbow_angle")) else "—"),
    ("Tracking success", f"{float(summary.get('tracking_success_rate')) * 100:.1f}%" if pd.notna(summary.get("tracking_success_rate")) else "—"),
    ("Incomplete movements", summary.get("incomplete_movements", 0)),
]
for start in range(0, len(metrics), 5):
    columns = st.columns(5)
    for column, (label, value) in zip(columns, metrics[start:start + 5]):
        column.metric(label, value)

st.subheader("Elbow angle over time")
if (not session_observations.empty and "elbow_angle" in session_observations
        and session_observations["elbow_angle"].notna().any()):
    session_observations["elbow_angle"] = pd.to_numeric(session_observations["elbow_angle"], errors="coerce")
    session_observations["elapsed_seconds"] = pd.to_numeric(session_observations["elapsed_seconds"], errors="coerce")
    valid = session_observations.dropna(subset=["elbow_angle", "elapsed_seconds"])
    chart = px.line(
        valid, x="elapsed_seconds", y="elbow_angle", markers=False,
        labels={"elapsed_seconds": "Elapsed time (seconds)", "elbow_angle": "Elbow angle (degrees)"},
    )
    chart.update_layout(margin=dict(l=10, r=10, t=20, b=10), template="plotly_white")
    st.plotly_chart(chart, width="stretch")
else:
    st.info("This session has no valid angle observations to chart.")

download_col, summary_col = st.columns(2)
if not session_observations.empty:
    download_col.download_button(
        "Download session observations CSV",
        session_observations.to_csv(index=False).encode("utf-8"),
        file_name=f"{selected_id}_observations.csv", mime="text/csv", width="stretch",
    )
summary_col.download_button(
    "Download session summary CSV",
    summary.to_frame().T.to_csv(index=False).encode("utf-8"),
    file_name=f"{selected_id}_summary.csv", mime="text/csv", width="stretch",
)
