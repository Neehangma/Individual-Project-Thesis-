"""Live webcam capture for elbow rehabilitation exercise monitoring."""

from __future__ import annotations

import argparse
from collections import deque
from datetime import timedelta
import time

import cv2
import numpy as np

from pose_detector import PoseDetector
from feedback_engine import FeedbackEngine
from rep_counter import RepCounter
from session_manager import ExerciseSession, SessionManager, SessionStateError
from data_logger import DataLogger


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
PAUSE_TIMEOUT_SECONDS = 5.0
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


def draw_target_bar(frame: np.ndarray, angle: float | None, *, y: int = 190) -> None:
    """Draw the configured target range and current angle as a horizontal bar."""
    height, width = frame.shape[:2]
    left, right = 24, width - 24
    y = min(y, height - 28)
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
    participant_id: str,
    selected_arm: str | None,
    session_id: str | None,
    target_repetitions: int,
    setup_message: str,
    angle: float | None,
    arm_side: str | None,
    visibility: float,
    valid: bool,
    feedback_message: str,
    movement_state: str,
    phase: str,
    set_number: int,
    rep_in_set: int,
    total_sets: int,
    completed_repetitions: int,
    elapsed_seconds: float,
    session_state: str,
    rest_remaining: int,
    ready_remaining: int,
) -> np.ndarray:
    """Draw session setup or active research-session details over the webcam."""
    output = cv2.flip(frame, 1)
    height, width = output.shape[:2]
    panel_height = min(224, height)
    overlay = output.copy()
    cv2.rectangle(overlay, (0, 0), (width, panel_height), (28, 30, 33), -1)
    cv2.addWeighted(overlay, 0.88, output, 0.12, 0, output)

    if session_id:
        header = f"Participant: {participant_id}   Session: {session_id}"
    else:
        arm_label = selected_arm or "not selected"
        input_label = " (editing)" if session_state == "editing" else ""
        header = f"Participant code: {participant_id or '<press I to enter>'}{input_label}   Arm: {arm_label}   Target: {target_repetitions}"
    cv2.putText(output, header[:110], (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                (245, 245, 245), 1, cv2.LINE_AA)

    shown_angle = f"{angle:.1f} deg" if angle is not None else "--"
    side = arm_side or selected_arm or "--"
    cv2.putText(output, f"Arm: {side}    Elbow angle: {shown_angle}    Pose confidence: {visibility:.2f}",
                (18, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.62, angle_color(angle, valid), 2, cv2.LINE_AA)
    pause_label = " (paused)" if session_state == "paused" else ""
    cv2.putText(output,
                f"Set {set_number}/{total_sets} (rep {rep_in_set}/{REPS_PER_SET}) | Repetitions: {completed_repetitions}/{target_repetitions} | {movement_state} / {phase} | {format_duration(elapsed_seconds)}{pause_label}",
                (18, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (245, 245, 245), 1, cv2.LINE_AA)

    if session_state == "editing":
        status = setup_message
        color = (0, 190, 255)
    elif ready_remaining > 0:
        status = f"Get ready: {ready_remaining}"
        color = (0, 190, 255)
    elif session_state == "paused":
        status = feedback_message
        color = (0, 190, 255)
    elif session_state == "complete":
        status = feedback_message
        color = (70, 200, 90)
    elif session_state == "idle":
        status = setup_message
        color = (220, 220, 220)
    elif rest_remaining > 0:
        status = f"Rest between sets: {rest_remaining}s"
        color = (0, 190, 255)
    elif not valid:
        status = feedback_message
        color = (0, 0, 255)
    else:
        status = feedback_message
        color = (70, 200, 90)
    cv2.putText(output, status[:110], (18, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                color, 1, cv2.LINE_AA)
    if height > 200:
        draw_target_bar(output, angle if valid else None)
    return output


def create_counter(target_repetitions: int, total_sets: int) -> RepCounter:
    """Create a counter whose per-set maximum can contain the session target."""
    return RepCounter(
        flexed_threshold=FLEXED_THRESHOLD_DEGREES,
        extended_threshold=EXTENDED_THRESHOLD_DEGREES,
        reps_per_set=REPS_PER_SET,
        total_sets=total_sets,
        rest_seconds=REST_SECONDS,
        target_min_angle=TARGET_MIN_ANGLE_DEGREES,
        target_max_angle=TARGET_MAX_ANGLE_DEGREES,
        pause_timeout_seconds=PAUSE_TIMEOUT_SECONDS,
        total_target_repetitions=target_repetitions,
    )


def finalize_session(
    manager: SessionManager,
    logger: DataLogger,
    session: ExerciseSession,
    counter: RepCounter,
) -> dict:
    """End and summarize the session, retaining interruptions and valid metrics."""
    if session.status == "in_progress":
        if not counter.is_complete and counter.reset_movement():
            logger.record_incomplete_movement(session)
        ended = manager.end_session(
            completed_repetitions=counter.total_reps,
            valid_observations=logger.valid_observation_count(session),
        )
    else:
        ended = session
    return logger.finish_session(ended, completed_repetitions=counter.total_reps)


def main() -> int:
    """Run webcam monitoring with setup, session lifecycle, and CSV collection."""
    parser = argparse.ArgumentParser(description="Live elbow exercise session capture")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX,
                        help="Webcam device index (default: 0)")
    parser.add_argument("--target", type=int, default=10,
                        help="Total session repetition target (default: 10)")
    parser.add_argument("--sets", type=int, default=TOTAL_SETS,
                        help=f"Maximum sets available for the target (default: {TOTAL_SETS})")
    args = parser.parse_args()
    max_target = args.sets * REPS_PER_SET
    if args.target < 1 or args.target > max_target:
        parser.error(f"--target must be between 1 and {max_target} for the configured set count")
    if args.sets < 1:
        parser.error("--sets must be at least 1")

    camera = None
    detector = None
    manager = SessionManager()
    logger: DataLogger | None = None
    session: ExerciseSession | None = None
    counter = create_counter(args.target, args.sets)
    feedback_engine = FeedbackEngine()
    angle_history: deque[float] = deque(maxlen=ANGLE_SMOOTHING_WINDOW)
    participant_id = ""
    editing_id = False
    selected_arm: str | None = None
    target_repetitions = args.target
    state = "idle"
    ready_until = 0.0
    paused_at: float | None = None
    setup_message = "I enter anonymous participant code | L/R select arm | +/- target | S start"

    try:
        logger = DataLogger(min_confidence=MIN_VISIBILITY)
        camera = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
        if not camera.isOpened():
            print(f"Could not open camera {args.camera}. Check permissions, close other camera apps, or try --camera 1.")
            return 1
        detector = PoseDetector(min_visibility=POSE_MIN_VISIBILITY)
        print("Enter an anonymous participant code in the camera window; do not enter personal information.")

        while True:
            ok, frame = camera.read()
            if not ok:
                print("Camera stopped returning frames. Check the webcam connection and try again.")
                break

            now = time.monotonic()
            ready_remaining = max(0, int(np.ceil(ready_until - now))) if state == "ready" else 0
            if state == "ready" and now >= ready_until:
                state = "active"
                angle_history.clear()

            annotated, observation = detector.process(frame, arm_side=selected_arm)
            valid = (
                observation.angle_degrees is not None
                and observation.visibility >= MIN_VISIBILITY
                and observation.arm_side is not None
                and (session is None or observation.arm_side == session.arm_side)
            )
            smoothed = smooth_angle(angle_history, observation.angle_degrees, valid)
            event = None
            if state == "active" and session is not None and logger is not None:
                if valid and smoothed is not None:
                    event = counter.update(smoothed, now)
                    if counter.partial_movement:
                        logger.record_incomplete_movement(session)
                elif counter.invalidate_tracking(now):
                    logger.record_incomplete_movement(session)

                feedback = feedback_engine.message(
                    tracking_valid=valid,
                    movement_state=counter.state,
                    phase=counter.phase,
                    arm_ready=counter.is_armed,
                    now=now,
                    rep_number=counter.total_reps if event is not None else None,
                    partial_movement=counter.partial_movement,
                )
                tracking_status = (
                    "Tracking stable" if valid else
                    "Person not detected" if observation.status == "Person not detected" else
                    "Arm not fully visible - adjust position"
                )
                logger.record_observation(
                    session,
                    elapsed_seconds=manager.elapsed_seconds(now),
                    elbow_angle=smoothed if valid else None,
                    pose_confidence=observation.visibility,
                    tracking_status=tracking_status,
                    movement_state=counter.state,
                    completed_repetitions=counter.total_reps,
                    feedback_message=feedback,
                )

                if counter.is_complete:
                    summary = finalize_session(manager, logger, session, counter)
                    state = "complete"
                    print(f"Completed session saved: {summary['session_id']} ({summary['session_status']})")

            if state == "active" and session is not None:
                elapsed = manager.elapsed_seconds(now)
                display_state = "active"
            elif state == "paused" and session is not None:
                elapsed = manager.elapsed_seconds(now)
                display_state = "paused"
            elif state == "complete" and session is not None:
                elapsed = session.duration_seconds
                display_state = "complete"
            else:
                elapsed = 0.0
                display_state = "editing" if editing_id else "idle"

            ready_remaining = max(0, int(np.ceil(ready_until - now))) if state == "ready" else 0
            if state == "active" and session is not None:
                feedback = feedback_engine.message(
                    tracking_valid=valid,
                    movement_state=counter.state,
                    phase=counter.phase,
                    arm_ready=counter.is_armed,
                    now=now,
                )
            elif state == "paused":
                feedback = "Session paused. Press P to resume."
            elif state == "complete":
                feedback = "Target reached. Session complete."
            elif not valid:
                feedback = "Position your arm so the shoulder, elbow and wrist are visible."
            else:
                feedback = "Arm detected. Choose setup options before starting."

            ui = draw_live_ui(
                annotated,
                participant_id=participant_id,
                selected_arm=selected_arm,
                session_id=session.session_id if session is not None else None,
                target_repetitions=target_repetitions,
                setup_message=setup_message,
                angle=smoothed if valid else observation.angle_degrees,
                arm_side=observation.arm_side,
                visibility=observation.visibility,
                valid=valid,
                feedback_message=feedback,
                movement_state=("PAUSED" if state == "paused" else counter.state),
                phase=counter.phase,
                set_number=counter.set_number,
                rep_in_set=counter.reps_in_set,
                total_sets=args.sets,
                completed_repetitions=counter.total_reps,
                elapsed_seconds=elapsed,
                session_state=display_state,
                rest_remaining=counter.rest_remaining(now),
                ready_remaining=ready_remaining,
            )
            cv2.imshow(WINDOW_NAME, ui)
            key = cv2.waitKey(1) & 0xFF

            if editing_id:
                if key in (10, 13):
                    editing_id = False
                    setup_message = "Participant code saved. Select arm with L/R and target with +/-; press S."
                elif key in (8, 127):
                    participant_id = participant_id[:-1]
                elif key == 27:
                    editing_id = False
                elif 0 <= key < 128:
                    character = chr(key)
                    if (character.isalnum() or character in "_-") and len(participant_id) < 32:
                        participant_id += character.upper()
                continue_key = True
            else:
                continue_key = False

            if not continue_key and key in (ord("q"), ord("Q"), 27):
                break
            if not continue_key and state == "idle" and key in (ord("i"), ord("I")):
                editing_id = True
                setup_message = "Type an anonymous code (letters, numbers, _ or -); Enter saves."
            elif not continue_key and state == "idle" and key in (ord("l"), ord("L")):
                selected_arm = "LEFT"
                setup_message = "Left arm selected. Enter participant code, then press S to start."
            elif not continue_key and state == "idle" and key in (ord("r"), ord("R")):
                selected_arm = "RIGHT"
                setup_message = "Right arm selected. Enter participant code, then press S to start."
            elif not continue_key and state == "idle" and key in (ord("+"), ord("=")):
                target_repetitions = min(max_target, target_repetitions + 1)
            elif not continue_key and state == "idle" and key in (ord("-"), ord("_")):
                target_repetitions = max(1, target_repetitions - 1)
            elif not continue_key and state == "idle" and key in (ord("s"), ord("S")):
                try:
                    if not participant_id:
                        raise ValueError("Enter an anonymous participant code with I first")
                    if selected_arm is None:
                        raise ValueError("Select the exercise arm with L or R first")
                    session = manager.start_session(
                        participant_id=participant_id,
                        arm_side=selected_arm,
                        target_repetitions=target_repetitions,
                        monotonic_now=now,
                    )
                    logger.start_session(session)
                    counter = create_counter(target_repetitions, args.sets)
                    angle_history.clear()
                    ready_until = time.monotonic() + GET_READY_SECONDS
                    state = "ready"
                    setup_message = "Session started. Get ready."
                except (ValueError, SessionStateError) as exc:
                    setup_message = str(exc)
            elif not continue_key and key in (ord("p"), ord("P")):
                if state == "active" and session is not None:
                    incomplete = counter.invalidate_tracking(now)
                    if incomplete and logger is not None:
                        logger.record_incomplete_movement(session)
                    manager.pause(now)
                    state = "paused"
                    paused_at = now
                    angle_history.clear()
                elif state == "paused" and session is not None and paused_at is not None:
                    resumed_at = time.monotonic()
                    pause_duration = resumed_at - paused_at
                    manager.resume(resumed_at)
                    counter.shift_time(pause_duration)
                    angle_history.clear()
                    state = "active"
                    paused_at = None
            elif not continue_key and key in (ord("r"), ord("R")):
                if state == "active" and session is not None and logger is not None:
                    if counter.reset_movement():
                        logger.record_incomplete_movement(session)
                    angle_history.clear()

            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

        return 0
    except Exception as exc:
        print(f"Session capture stopped because of an error: {exc}")
        return 1
    finally:
        if session is not None and session.status == "in_progress" and logger is not None:
            try:
                summary = finalize_session(manager, logger, session, counter)
                print(f"Session saved: {summary['session_id']} ({summary['session_status']})")
            except Exception as exc:
                print(f"Could not safely finalize the active session: {exc}")
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
