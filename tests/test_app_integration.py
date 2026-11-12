"""Streamlit entrypoint and page navigation smoke tests."""

from __future__ import annotations

from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class StreamlitAppTests(unittest.TestCase):
    """Verify all visible pages render without accessing the camera."""

    def test_home_and_all_navigation_pages_render(self):
        app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=15).run()
        self.assertFalse(app.exception)
        self.assertIn("Elbow Rehabilitation Exercise Monitoring System", [item.value for item in app.title])

        for page_path, expected_title in (
            ("app_pages/monitoring.py", "Exercise Monitoring"),
            ("app_pages/results.py", "Session Results"),
            ("app_pages/about.py", "About the System"),
            ("app_pages/home.py", "Elbow Rehabilitation Exercise Monitoring System"),
        ):
            with self.subTest(page=page_path):
                app.switch_page(page_path).run()
                self.assertFalse(app.exception)
                self.assertIn(expected_title, [item.value for item in app.title])

    def test_empty_results_state_is_safe(self):
        app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=15).run()
        app.switch_page("app_pages/results.py").run()
        self.assertFalse(app.exception)
        self.assertTrue(any("No saved sessions yet" in item.value for item in app.info))


if __name__ == "__main__":
    unittest.main()
