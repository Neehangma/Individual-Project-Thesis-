"""Home page with project purpose and a path into live exercise monitoring."""

import streamlit as st

st.title("Elbow Rehabilitation Exercise Monitoring System")
st.write(
    "A local academic prototype that uses a webcam and computer vision to estimate "
    "elbow angles and monitor elbow flexion and extension movements."
)

left, right = st.columns([1.5, 1], gap="large")
with left:
    with st.container(border=True):
        st.subheader("Start a monitored session")
        st.write("Select an anonymous participant code, exercise arm, and repetition target.")
        st.page_link("app_pages/monitoring.py", label="Start Exercise Session", icon="🎥")
    with st.container(border=True):
        st.subheader("How it works")
        st.write(
            "MediaPipe estimates shoulder, elbow, and wrist landmarks from live camera frames. "
            "The system calculates an image-plane elbow angle, applies movement thresholds, "
            "and saves timestamped measurements locally as CSV."
        )
with right:
    with st.container(border=True):
        st.subheader("Camera setup")
        st.markdown(
            "- Place the camera steadily at about arm height.\n"
            "- Keep your upper body and the full selected arm in view.\n"
            "- Face the camera side-on enough for the elbow bend to be visible.\n"
            "- Close other apps that may be using the webcam."
        )
    with st.container(border=True):
        st.subheader("Research use")
        st.caption(
            "Use anonymous participant codes and collect data only under the approved consent procedure. "
            "This prototype stores measurements, not camera images or video. It is not a diagnostic system."
        )

st.divider()
st.page_link("app_pages/results.py", label="Review saved session results", icon="📈")
