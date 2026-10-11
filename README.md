# Elbow Rehabilitation Exercise Monitor

Live webcam capture prototype for elbow flexion and extension. It estimates the
2D elbow angle with MediaPipe Pose and saves measurements locally for later
analysis. Camera images and video are not saved.

## Setup (Windows, Python 3.11)

In PowerShell, from this project directory:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

The default camera index is 0. If needed, select another webcam with
`python app.py --camera 1`.

## Live camera controls

- **S**: start a session after a three-second get-ready countdown.
- **P**: pause or resume capture. The session timer excludes paused time.
- **Q** or **Esc**: end capture and save the session files.
- **R**: reset set and rep progress while preserving the observations already
  captured in the current session.

The live screen shows the raw/smoothed elbow angle, detected anatomical arm
side, landmark visibility, movement phase, set and rep progress, session timer,
and a target-range bar. Low-visibility frames are marked invalid, excluded from
smoothing and repetition counting, and retained in the CSV with their validity
flag. Live guidance is limited to tracking and capture status; detailed feedback
is intended for a later post-session view.

## Capture settings

The constants at the top of `app.py` control visibility threshold, five-frame
moving average, flexed and extended thresholds, target range, set/rep target,
inter-set rest, and get-ready countdown. Initial defaults are 3 sets of 10,
flexed below 60°, extended at or above 150°, and target range 40°–150°. These
are configurable software defaults, not universal clinical recommendations.

## Session files

Starting a session creates a `sessions/` directory if needed. On Q/Esc (or
normal application exit), the session is saved as:

- `session_YYYYMMDD_HHMMSS.csv`: one row per active capture frame with local
  timestamp, elapsed session time, arm side, raw and smoothed angle, visibility,
  phase, completed-rep count, set number, and `valid_frame`.
- `session_YYYYMMDD_HHMMSS.json`: session duration, total reps, reps per set,
  min/max angle, range of motion, average rep duration, average visibility,
  low-visibility frame count, target range, and per-rep min/max/duration/target
  range attainment. It also records the thresholds, visibility cutoff,
  smoothing window, and set/rest configuration used for that capture.

A numeric suffix is added if two sessions start within the same second. Files
are local research measurements; use anonymised participant codes and the
approved participant-consent procedure when collecting research data.

## Camera positioning and limitations

Place the camera around elbow height and keep the shoulder, elbow, and wrist in
view. A side-on view usually makes the image-plane angle easier to interpret.
Slowly bend and straighten to see the angle, phase, and repetition state respond.
The angle is an estimate from a single webcam's 2D image, not a clinically
validated three-dimensional measurement or medical advice.
