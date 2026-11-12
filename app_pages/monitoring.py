"""Setup and controls for live webcam exercise monitoring."""

import streamlit as st

from capture_controller import TOTAL_SETS, REPS_PER_SET


def _controller():
    """Get the entrypoint-owned controller from the current Streamlit session."""
    return st.session_state.capture_controller


st.title("Exercise Monitoring")
controller = _controller()

if not controller.is_running:
    st.write("Set up a session. Data collection begins after the camera opens successfully.")
    with st.form("session_setup", clear_on_submit=False, border=True):
        participant_id = st.text_input(
            "Anonymous participant ID",
            max_chars=32,
            help="Use a study code only (letters, numbers, underscore, or hyphen); do not enter a name.",
        )
        st.selectbox("Exercise", ["Elbow Flexion and Extension"])
        arm = st.radio("Exercise arm", ["LEFT", "RIGHT"], horizontal=True)
        target = st.number_input(
            "Target repetitions", min_value=1, max_value=TOTAL_SETS * REPS_PER_SET,
            value=10, step=1,
        )
        started = st.form_submit_button("Start Session", type="primary", width="stretch")
    if started:
        try:
            session = controller.start(participant_id, arm, int(target))
            st.session_state.last_session_id = session.session_id
            st.success(f"Session {session.session_id} started. Camera is initializing.")
            st.rerun()
        except (ValueError, RuntimeError, OSError) as exc:
            st.error(str(exc))
    if controller.last_error:
        st.error(controller.last_error)
    st.info("Live processing uses the selected arm's anatomical MediaPipe landmarks.")
else:
    session = controller.session
    st.caption(f"Controls for participant **{session.participant_id}** · Session `{session.session_id}`")
    control_cols = st.columns(3)
    if controller.state == "paused":
        if control_cols[0].button("Resume", type="primary", width="stretch"):
            controller.resume()
            st.rerun()
    else:
        if control_cols[0].button("Pause", width="stretch", disabled=controller.state == "ready"):
            controller.pause()
            st.rerun()
    if control_cols[1].button("End Session", width="stretch"):
        st.session_state.confirm_end_session = True
    if st.session_state.get("confirm_end_session", False):
        st.warning("End this session and save the collected measurements?")
        confirm_col, keep_col = st.columns(2)
        if confirm_col.button("End and Save", type="primary", key="confirm_end"):
            summary = controller.end()
            st.session_state.confirm_end_session = False
            st.session_state.last_session_id = summary["session_id"] if summary else session.session_id
            st.success("Session saved. View its measurements on Session Results.")
            st.rerun()
        if keep_col.button("Continue Session", key="cancel_end"):
            st.session_state.confirm_end_session = False
            st.rerun()
    st.caption("Camera processing continues while navigating between pages. Ending saves the summary and releases the camera.")
