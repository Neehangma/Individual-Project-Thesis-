# Elbow Rehabilitation Exercise Monitor

Milestone 1 prototype: live webcam pose landmarks and an estimated elbow angle.
The camera image is processed locally and is not saved.

## Setup (Windows, Python 3.11)

In PowerShell, from this project directory:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

The prototype opens the default webcam (device 0). Press **Q** or **Esc** while
the camera window is focused to quit. If another device is required, use
`python app.py --camera 1`. If tracking is too permissive or too strict, adjust
the visibility cutoff, for example `python app.py --visibility 0.65`.

## Camera positioning and angle check

Place the camera at roughly elbow height, stand far enough away to keep the
shoulder, elbow, and wrist visible, and turn so the moving arm is seen from the
side. Move slowly through a bend and straighten cycle. The highlighted shoulder,
elbow, and wrist should follow the arm, and the displayed angle should change
with the movement. A straighter arm generally produces an angle nearer 180°;
bending reduces the measured angle. Either arm can be selected, based on the
visibility of its three landmarks. Weak visibility displays a warning instead
of an angle.

This is an estimated 2D image-based angle from one webcam. It is not a
clinically validated three-dimensional measurement and is not medical advice.

## Files

- `app.py`: camera loop and live OpenCV display.
- `pose_detector.py`: pretrained MediaPipe Pose inference and landmark drawing.
- `angle_calculator.py`: geometric elbow angle calculation.
