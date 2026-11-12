"""Small shared Streamlit fragment that advances live webcam processing."""

from __future__ import annotations

import time

import streamlit as st

from capture_controller import CaptureController


@st.fragment(run_every=0.15)
def _live_capture_fragment(controller: CaptureController) -> None:
    """Process one real frame and display the active session's camera/status."""
    if not controller.is_running:
        return
    if not controller.tick():
        if controller.last_error:
            st.error(controller.last_error)
        return
    session = controller.session
    st.caption(f"Participant **{session.participant_id}** · Session `{session.session_id}` · Arm **{session.arm_side}**")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Repetitions", f"{controller.completed_repetitions} / {session.target_repetitions}")
    metric_cols[1].metric("Movement state", controller.movement_state.title())
    metric_cols[2].metric("Phase", controller.phase)
    metric_cols[3].metric(
        "Elapsed", f"{int(controller.elapsed_seconds // 60):02d}:{int(controller.elapsed_seconds % 60):02d}"
    )
    st.progress(
        min(1.0, controller.completed_repetitions / session.target_repetitions),
        text=f"Session progress · {controller.completed_repetitions} of {session.target_repetitions}",
    )
    if controller.feedback_message:
        st.info(controller.feedback_message)
    st.markdown("#### Live camera")
    image_col, state_col = st.columns([1.65, 1], vertical_alignment="center")
    with image_col:
        if controller.last_frame is not None:
            st.image(controller.last_frame, caption="Live pose tracking", width="stretch")
        else:
            st.info("Waiting for the first camera frame…")
    with state_col:
        angle_text = f"{controller.smoothed_angle:.1f}°" if controller.smoothed_angle is not None else "—"
        st.metric("Elbow angle", angle_text)
        st.metric("Arm", controller.session.arm_side.title())
        st.metric("Pose confidence", f"{controller.visibility:.2f}")
        if controller.tracking_status == "Tracking stable":
            st.success(controller.tracking_status)
        else:
            st.warning(controller.tracking_status)
        if controller.state == "ready":
            seconds = max(0, int(controller.ready_until - time.monotonic() + 0.999))
            st.info(f"Get ready · session starts in {seconds} s")
        if controller.counter and controller.counter.rest_remaining(time.monotonic()):
            st.info(f"Rest between sets · {controller.counter.rest_remaining(time.monotonic())} s")
    if controller.state == "completed":
        st.success("Session target reached and saved. The camera has been released.")
        st.rerun()


def render_live_capture(controller: CaptureController) -> None:
    """Mount the live fragment only while a session owns the webcam."""
    if controller.is_running:
        _live_capture_fragment(controller)
