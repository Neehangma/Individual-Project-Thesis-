"""Streamlit rendering regression test for encoded camera-frame bytes."""

from __future__ import annotations

import unittest

import cv2
import numpy as np
from streamlit.testing.v1 import AppTest


class LiveFrameRenderingTests(unittest.TestCase):
    """Ensure the persistent image placeholder accepts an encoded live frame."""

    def test_jpeg_bytes_render_in_streamlit_image_element(self):
        synthetic_frame = np.full((48, 64, 3), (30, 120, 180), dtype=np.uint8)
        encoded_ok, encoded_frame = cv2.imencode(".jpg", synthetic_frame)
        self.assertTrue(encoded_ok)
        payload_hex = encoded_frame.tobytes().hex()
        source = (
            "import streamlit as st\n"
            "from live_capture_view import render_camera_frame\n"
            "render_camera_frame(st.empty(), bytes.fromhex('" + payload_hex + "'))\n"
        )
        app = AppTest.from_string(source).run()
        self.assertFalse(app.exception)
        self.assertTrue(app.get("imgs"))


if __name__ == "__main__":
    unittest.main()
