"""Live webcam capture for elbow rehabilitation exercise monitoring."""

from __future__ import annotations

import argparse
from collections import deque
from datetime import timedelta
import time

import cv2
import numpy as np

from pose_detector import PoseDetector
from rep_counter import RepCounter
from session_logger import SessionLogger


# ============================== Configuration ==============================
CAMERA_INDEX = 0
MIN_VISIBILITY = 0.60
POSE_MIN_VISIBILITY = 0.55
ANGLE_SMOOTHING_WINDOW = 5
FLEXED_THRESHOLD_DEGREES = 60.0
EXTENDED_THRESHOLD_DEGREES = 150.0
TARGET_MIN_ANGLE_DEGREES = 40.0
TARGET_MAX_ANGLE_DEGREES = 150.0
TOTAL_SETS = 3
REPS_PER_SET = 10
REST_SECONDS = 10
GET_READY_SECONDS = 3
ANGLE_BOUNDARY_TOLERANCE_DEGREES = 5.0
WINDOW_NAME = "Elbow Rehabilitation Monitor - Live Capture"


def smooth_angle(history: deque[float], raw_angle: float | None, valid: bool) -> float | None:
    """Return the valid-angle moving average, clearing history on bad tracking."""
    if not valid or raw_angle is None:
        history.clear()
        return None
    history.append(float(raw_angle))
    return float(np.mean(history))


def format_duration(seconds: float) -> str:
    """Format elapsed seconds as minutes and seconds for the live display."""
    duration = timedelta(seconds=max(0, int(seconds)))
    minutes, remainder = divmod(int(duration.total_seconds()), 60)
    return f"{minutes:02d}:{remainder:02d}"


def angle_color(angle: float | None, valid: bool) -> tuple[int, int, int]:
    """Select a BGR color for an angle relative to the configured target range."""
    if angle is None or not valid:
        return (0, 0, 230)
    if TARGET_MIN_ANGLE_DEGREES <= angle <= TARGET_MAX_ANGLE_DEGREES:
        near_edge = min(angle - TARGET_MIN_ANGLE_DEGREES,
                        TARGET_MAX_ANGLE_DEGREES - angle) <= ANGLE_BOUNDARY_TOLERANCE_DEGREES
        return (0, 165, 255) if near_edge else (70, 200, 90)
    distance = min(abs(angle - TARGET_MIN_ANGLE_DEGREES),
                   abs(angle - TARGET_MAX_ANGLE_DEGREES))
    return (0, 165, 255) if distance <= ANGLE_BOUNDARY_TOLERANCE_DEGREES else (0, 0, 230)


def draw_target_bar(frame: np.ndarray, angle: float | None) -> None:
    """Draw the configured target range and current angle as a horizontal bar."""
    height, width = frame.shape[:2]
    left, right = 24, width - 24
    y = min(148, height - 28)
    cv2.putText(frame, f"Target range: {TARGET_MIN_ANGLE_DEGREES:.0f}-{TARGET_MAX_ANGLE_DEGREES:.0f} deg",
                (left, y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.line(frame, (left, y), (right, y), (100, 100, 100), 10, cv2.LINE_AA)
    cv2.line(frame, (left, y), (right, y), (70, 200, 90), 5, cv2.LINE_AA)
    if angle is not None:
        clipped = min(TARGET_MAX_ANGLE_DEGREES, max(TARGET_MIN_ANGLE_DEGREES, angle))
        fraction = (clipped - TARGET_MIN_ANGLE_DEGREES) / max(
            1.0, TARGET_MAX_ANGLE_DEGREES - TARGET_MIN_ANGLE_DEGREES
        )
        marker_x = int(left + fraction * (right - left))
        cv2.circle(frame, (marker_x, y), 10, angle_color(angle, True), -1, cv2.LINE_AA)


def draw_live_ui(
    frame: np.ndarray,
    *,
    angle: float | None,
    arm_side: str | None,
    visibility: float,
    valid: bool,
    tracking_status: str,
    phase: str,
    set_number: int,
    rep_in_set: int,
    elapsed_seconds: float,
    session_state: str,
    rest_remaining: int,
    ready_remaining: int,
) -> np.ndarray:
    """Draw compact live capture guidance, status, progress, and target bar."""
    output = cv2.flip(frame, 1)
    height, width = output.shape[:2]
    panel_height = min(172, height)
    overlay = output.copy()
    cv2.rectangle(overlay, (0, 0), (width, panel_height), (28, 30, 33), -1)
    cv2.addWeighted(overlay, 0.88, output, 0.12, 0, output)

    shown_angle = f"{angle:.1f} deg" if angle is not None else "--"
    side = arm_side if arm_side else "--"
    cv2.putText(output, f"Angle: {shown_angle}    Arm: {side}    Visibility: {visibility:.2f}",
                (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, angle_color(angle, valid), 2, cv2.LINE_AA)
    pause_label = " (paused)" if session_state == "paused" else ""
    cv2.putText(output,
                f"Set {set_number}/{TOTAL_SETS} | Rep {rep_in_set}/{REPS_PER_SET}    Phase: {phase}    Time: {format_duration(elapsed_seconds)}{pause_label}",
                (20, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (245, 245, 245), 2, cv2.LINE_AA)

    if ready_remaining > 0:
        status = f"Get ready: {ready_remaining}"
        status_color = (0, 190, 255)
    elif session_state == "paused":
        status = "Session paused - press P to resume"
        status_color = (0, 190, 255)
    elif not valid:
        status = "Arm not fully visible - adjust position"
        status_color = (0, 0, 255)
    elif session_state == "idle":
        status = "S start | P pause/resume | Q save & end | R reset progress"
        status_color = (220, 220, 220)
    elif rest_remaining > 0:
        status = f"Rest between sets: {rest_remaining}s"
        status_color = (0, 190, 255)
    elif tracking_status != "Tracking stable":
        status = tracking_status
        status_color = (0, 0, 255)
    elif session_state == "complete":
        status = "Target complete - press Q to save and end"
        status_color = (70, 200, 90)
    else:
        status = session_state.title()
        status_color = (70, 200, 90)
    cv2.putText(output, status, (20, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.57,
                status_color, 2, cv2.LINE_AA)
    if height > 155:
        draw_target_bar(output, angle if valid else None)
    return output


def main() -> int:
    """Run the webcam capture loop and save a CSV plus summary when it ends."""
    parser = argparse.ArgumentParser(description="Live elbow exercise capture")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX,
                        help="Webcam device index (default: 0)")
    args = parser.parse_args()

    camera = None
    detector = None
    logger: SessionLogger | None = None
    logger_finished = False
    state = "idle"
    ready_until = 0.0
    session_start = 0.0
    paused_at = 0.0
    paused_total = 0.0
    counter = RepCounter(
        flexed_threshold=FLEXED_THRESHOLD_DEGREES,
        extended_threshold=EXTENDED_THRESHOLD_DEGREES,
        reps_per_set=REPS_PER_SET,
        total_sets=TOTAL_SETS,
        rest_seconds=REST_SECONDS,
        target_min_angle=TARGET_MIN_ANGLE_DEGREES,
        target_max_angle=TARGET_MAX_ANGLE_DEGREES,
    )
    angle_history: deque[float] = deque(maxlen=ANGLE_SMOOTHING_WINDOW)
    try:
        camera = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
        if not camera.isOpened():
            print(f"Could not open camera {args.camera}. Check permissions, close other camera apps, or try --camera 1.")
            return 1
        detector = PoseDetector(min_visibility=POSE_MIN_VISIBILITY)
        print("Camera ready. Use S to start, P to pause/resume, Q to save and end, R to reset set/rep progress.")

        while True:
            ok, frame = camera.read()
            if not ok:
                print("Camera stopped returning frames. Check the webcam connection and try again.")
                break

            now = time.monotonic()
            ready_remaining = max(0, int(np.ceil(ready_until - now))) if state == "ready" else 0
            if state == "ready" and now >= ready_until:
                logger = SessionLogger(
                    target_range=(TARGET_MIN_ANGLE_DEGREES, TARGET_MAX_ANGLE_DEGREES),
                    settings={
                        "minimum_visibility": MIN_VISIBILITY,
                        "flexed_threshold_degrees": FLEXED_THRESHOLD_DEGREES,
                        "extended_threshold_degrees": EXTENDED_THRESHOLD_DEGREES,
                        "angle_smoothing_window": ANGLE_SMOOTHING_WINDOW,
                        "total_sets": TOTAL_SETS,
                        "reps_per_set_target": REPS_PER_SET,
                        "rest_seconds": REST_SECONDS,
                    },
                )
                session_start = now
                paused_total = 0.0
                state = "active"
                ready_remaining = 0
                angle_history.clear()

            annotated, observation = detector.process(frame)
            valid = (observation.angle_degrees is not None
                     and observation.visibility >= MIN_VISIBILITY
                     and observation.arm_side is not None)
            smoothed = smooth_angle(angle_history, observation.angle_degrees, valid)

            if state == "active" and logger is not None:
                if valid and smoothed is not None:
                    event = counter.update(smoothed, now)
                    if event is not None:
                        logger.log_rep(event)
                else:
                    counter.invalidate_tracking()

                elapsed = max(0.0, now - session_start - paused_total)
                logger.log_frame(
                    elapsed_time=elapsed,
                    arm_side=observation.arm_side,
                    raw_angle=observation.angle_degrees,
                    smoothed_angle=smoothed,
                    visibility=observation.visibility,
                    phase=counter.phase,
                    rep_count=logger.completed_rep_count,
                    set_number=counter.set_number,
                    valid_frame=valid,
                )

            if state == "active":
                state_for_display = "complete" if counter.is_complete else "active"
                elapsed = max(0.0, now - session_start - paused_total)
            elif state == "paused":
                state_for_display = "paused"
                elapsed = max(0.0, paused_at - session_start - paused_total)
            else:
                state_for_display = "idle" if state in ("idle", "ready") else state
                elapsed = 0.0

            rest_remaining = counter.rest_remaining(now)
            ui = draw_live_ui(
                annotated,
                angle=smoothed if valid else observation.angle_degrees,
                arm_side=observation.arm_side,
                visibility=observation.visibility,
                valid=valid,
                tracking_status=observation.status,
                phase=counter.phase,
                set_number=counter.set_number,
                rep_in_set=counter.reps_in_set,
                elapsed_seconds=elapsed,
                session_state=state_for_display,
                rest_remaining=rest_remaining,
                ready_remaining=ready_remaining,
            )
            cv2.imshow(WINDOW_NAME, ui)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("s"), ord("S")) and state == "idle":
                state = "ready"
                ready_until = time.monotonic() + GET_READY_SECONDS
            elif key in (ord("p"), ord("P")):
                if state == "active":
                    state = "paused"
                    paused_at = time.monotonic()
                    counter.invalidate_tracking()
                    angle_history.clear()
                elif state == "paused":
                    resumed_at = time.monotonic()
                    pause_duration = resumed_at - paused_at
                    paused_total += pause_duration
                    counter.shift_time(pause_duration)
                    angle_history.clear()
                    state = "active"
            elif key in (ord("r"), ord("R")):
                if state == "ready":
                    state = "idle"
                    ready_until = 0.0
                elif state in ("active", "paused"):
                    counter.reset_progress()
                    angle_history.clear()

            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

        return 0
    except Exception as exc:
        print(f"Live capture stopped because of an error: {exc}")
        return 1
    finally:
        if logger is not None and not logger_finished:
            try:
                end_monotonic = paused_at if state == "paused" else time.monotonic()
                duration = max(0.0, end_monotonic - session_start - paused_total)
                paths = logger.finish(duration_seconds=duration)
                logger_finished = True
                print(f"Session data saved: {paths[0]} and {paths[1]}")
            except Exception as exc:
                print(f"Could not finalize session data: {exc}")
        try:
            if detector is not None:
                detector.close()
        except Exception as exc:
            print(f"Could not close the pose detector cleanly: {exc}")
        finally:
            if camera is not None:
                camera.release()
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
