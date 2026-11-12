"""Process-wide webcam lease to prevent concurrent Streamlit camera sessions."""

from __future__ import annotations

import threading
from typing import Any


class CameraBusyError(RuntimeError):
    """Raised when another browser session currently owns the webcam."""


class CameraRegistry:
    """Own at most one OpenCV capture and release it safely at shutdown."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._owner: str | None = None
        self._capture: Any | None = None

    def acquire(self, owner: str, camera_index: int, cv2_module: Any) -> Any:
        """Open the requested camera unless it is already leased."""
        with self._lock:
            if self._capture is not None:
                if self._owner == owner:
                    return self._capture
                raise CameraBusyError("The webcam is already in use by another session. End that session first.")
            capture = (cv2_module.VideoCapture(camera_index, cv2_module.CAP_DSHOW)
                       if hasattr(cv2_module, "CAP_DSHOW") else cv2_module.VideoCapture(camera_index))
            if not capture.isOpened():
                capture.release()
                raise RuntimeError(
                    f"Could not open camera {camera_index}. Check camera permissions, close other camera apps, "
                    "or try another camera index."
                )
            self._owner, self._capture = owner, capture
            return capture

    def release(self, owner: str) -> None:
        """Release the webcam only when called by its current owner."""
        with self._lock:
            if self._owner != owner:
                return
            capture, self._capture = self._capture, None
            self._owner = None
            if capture is not None:
                capture.release()

    def release_all(self) -> None:
        """Release any remaining camera during application shutdown."""
        with self._lock:
            capture, self._capture = self._capture, None
            self._owner = None
            if capture is not None:
                capture.release()
