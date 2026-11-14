"""Streamlit rendering regression test for RGB camera-frame arrays."""

from __future__ import annotations

import unittest

import numpy as np
from streamlit.testing.v1 import AppTest


class LiveFrameRenderingTests(unittest.TestCase):
    """Ensure the persistent image placeholder accepts an encoded live frame."""

    def test_rgb_array_renders_in_streamlit_image_element(self):
        synthetic_frame = np.full((48, 64, 3), (30, 120, 180), dtype=np.uint8)
        payload_hex = synthetic_frame.tobytes().hex()
        source = (
            "import numpy as np\n"
            "import streamlit as st\n"
            "from live_capture_view import render_camera_frame\n"
            "frame = np.frombuffer(bytes.fromhex('" + payload_hex + "'), dtype=np.uint8).reshape(48, 64, 3)\n"
            "render_camera_frame(st.empty(), frame)\n"
        )
        app = AppTest.from_string(source).run()
        self.assertFalse(app.exception)
        self.assertTrue(app.get("imgs"))


if __name__ == "__main__":
    unittest.main()
