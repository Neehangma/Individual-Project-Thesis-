"""Streamlit entry point for the elbow exercise monitoring prototype."""

from __future__ import annotations

import atexit

import streamlit as st

from camera_registry import CameraRegistry
from capture_controller import CaptureController
from live_capture_view import render_live_capture


st.set_page_config(page_title="Elbow Exercise Monitor", page_icon="🦾", layout="wide", initial_sidebar_state="expanded")


@st.cache_resource
def get_camera_registry() -> CameraRegistry:
    """Share a camera lock across browser sessions and release it on shutdown."""
    registry = CameraRegistry()
    atexit.register(registry.release_all)
    return registry


def get_controller() -> CaptureController:
    """Return this browser session's controller without reopening the camera."""
    if "capture_controller" not in st.session_state:
        st.session_state.capture_controller = CaptureController(get_camera_registry())
    return st.session_state.capture_controller


with st.sidebar:
    st.markdown("### Elbow Exercise Monitor")
    st.caption("Local computer-vision research prototype")
    st.divider()

controller = get_controller()
render_live_capture(controller)

pages = [
    st.Page("app_pages/home.py", title="Home", icon="🏠", default=True),
    st.Page("app_pages/monitoring.py", title="Exercise Monitoring", icon="🎥"),
    st.Page("app_pages/results.py", title="Session Results", icon="📈"),
    st.Page("app_pages/about.py", title="About the System", icon="ℹ️"),
]
navigation = st.navigation(pages, position="sidebar", expanded=True)
navigation.run()
