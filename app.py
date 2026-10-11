"""Milestone 1: live webcam elbow angle prototype (press Q or Esc to exit)."""

from __future__ import annotations

import argparse

import cv2

from pose_detector import PoseDetector


def main() -> int:
    parser = argparse.ArgumentParser(description="Live 2D elbow angle estimator")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--visibility", type=float, default=0.55,
                        help="Minimum visibility for shoulder, elbow and wrist (0..1)")
    args = parser.parse_args()
    if not 0 <= args.visibility <= 1:
        parser.error("--visibility must be between 0 and 1")

    camera = None
    detector = None
    try:
        camera = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
        if not camera.isOpened():
            print(f"Could not open camera {args.camera}. Check permissions, close other camera apps, or try --camera 1.")
            return 1
        detector = PoseDetector(args.visibility)
        print("Camera ready. Press Q or Esc in the video window to quit.")
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Camera stopped returning frames. Check the webcam connection and try again.")
                break
            annotated, observation = detector.process(frame)
            # Run inference on the camera's original orientation so MediaPipe's
            # anatomical LEFT/RIGHT labels stay correct. Mirror only the view.
            annotated = cv2.flip(annotated, 1)
            cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 88), (35, 35, 35), -1)
            angle_text = f"{observation.angle_degrees:.1f} deg" if observation.angle_degrees is not None else "--"
            side_text = observation.arm_side if observation.arm_side else "--"
            cv2.putText(annotated, f"Elbow angle: {angle_text}     Arm: {side_text}", (18, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated, f"{observation.status}  |  visibility {observation.visibility:.2f}",
                        (18, 69), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (70, 220, 120) if observation.angle_degrees is not None else (0, 170, 255),
                        2, cv2.LINE_AA)
            cv2.imshow("Elbow Rehabilitation Monitor - Milestone 1", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    except Exception as exc:
        print(f"Could not initialize or process the webcam pose detector: {exc}")
        return 1
    finally:
        if detector is not None:
            detector.close()
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
