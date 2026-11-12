"""Project overview, technical method, and limitations."""

import streamlit as st

st.title("About the System")
st.write(
    "This project explores a camera-based approach for monitoring elbow flexion and extension "
    "during a supervised exercise session. It records actual pose-derived angle estimates, "
    "tracking status, movement states, and repetition counts for academic analysis."
)

with st.container(border=True):
    st.subheader("Aim and objectives")
    st.markdown(
        "- Estimate an elbow angle from shoulder, elbow, and wrist landmarks.\n"
        "- Detect completed flexion-extension cycles using configurable thresholds and states.\n"
        "- Provide concise live tracking and movement feedback.\n"
        "- Store anonymised session measurements locally for research review."
    )

with st.container(border=True):
    st.subheader("How the computer vision works")
    st.write(
        "OpenCV supplies live frames to MediaPipe Pose, which estimates body landmarks. "
        "The application selects the configured anatomical arm, checks the visibility of its "
        "shoulder, elbow, and wrist, then calculates the angle between the two elbow-centred vectors. "
        "A short moving average reduces frame-to-frame noise before the existing repetition state machine processes measurements."
    )

left, right = st.columns(2)
with left:
    with st.container(border=True):
        st.subheader("Technologies")
        st.write("Python · Streamlit · OpenCV · MediaPipe Pose · NumPy · Pandas · Plotly · CSV")
with right:
    with st.container(border=True):
        st.subheader("Limitations")
        st.write(
            "A single webcam provides an estimated 2D image-plane angle, not a clinically validated "
            "3D measurement. Occlusion, camera position, clothing, lighting, and body orientation "
            "can affect landmark confidence and angle estimates."
        )

st.warning(
    "This is an academic prototype for exercise monitoring and technical evaluation. "
    "It is not a medical diagnostic system and does not replace professional physiotherapy advice."
)
st.caption("Research CSV files are stored locally in the project's data/ directory. Webcam images and video are not saved.")
