# Elbow Rehabilitation Exercise Monitor

Local webcam prototype for monitoring elbow flexion and extension. MediaPipe
Pose landmarks feed a 2D elbow-angle estimate, repetition counter, and concise
live feedback. Webcam frames are not saved.

## Setup and run (Windows, Python 3.11)

From PowerShell in this project directory:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

Choose another camera with `python app.py --camera 1`. Configure the total
session repetition target before opening the camera with `--target`; for
example, `python app.py --target 5 --sets 1`. The default target is 10. `--sets`
sets the maximum available sets; each set holds up to 10 repetitions, with the
configured rest between sets.

## Setup and live controls

Before starting, press **I** and enter an anonymous participant code using
letters, numbers, underscores, or hyphens, then press **Enter**. Do not enter a
name or other identifying information. Press **L** or **R** to select the
exercise arm and **+**/**-** to change the target. Press **S** to start; the
session gets a three-second countdown.

- **P**: pause or resume. Paused time is excluded from the session duration.
- **Q** or **Esc**: end the session and save it. Ending below target records an
  interrupted session; ending before valid exercise observations records a
  cancelled session.
- **R** during capture: discard the current incomplete movement and return to
  a fresh extended starting position without erasing completed reps or data.

The live screen shows participant and session IDs, selected arm, elbow angle,
pose confidence, completed repetitions/target, set progress, movement phase and
state, elapsed time, tracking status, and state-aware feedback. Only reliable
angles are counted or used as research measurements. Tracking loss discards an
incomplete rep and requires the selected arm to return to the extended starting
position. The counter stops when the configured total target is reached and
saves the completed session automatically.

## Movement states and settings

`READY` waits for a valid angle at or above the extended threshold. Falling below
that threshold starts `FLEXION`; reaching the flexed threshold confirms the
bend. Moving back above the flexed threshold enters `EXTENSION`; reaching the
extended threshold completes one repetition. Tracking loss enters `PAUSED` and
discards an in-progress cycle. Once tracking returns, a fresh extended starting
position is required. The state becomes `COMPLETED` at the session target.

The constants at the top of `app.py` configure visibility, smoothing, movement
thresholds, target range, default sets/target, rest duration, get-ready time,
and tracking pause timeout. Defaults include a 0.60 minimum pose confidence,
five-frame angle smoothing, flexion below 60°, and extension at or above 150°.
These are software defaults for supervised testing, not universal clinical
recommendations. A single webcam provides an estimated 2D angle, not a
clinically validated 3D measurement.

## Research data

The app creates `data/` and initializes CSV headers automatically. It appends
records to:

- `data/session_observations.csv`: one active-session observation per processed
  frame, with session/participant IDs, timestamp, elapsed seconds, selected arm,
  valid elbow angle (blank when invalid), pose confidence, tracking status,
  movement state, rep count, target, and feedback message.
- `data/session_summaries.csv`: one row per ended session, including status,
  participant and exercise information, target and completed reps, incomplete
  movements, start/end times, active duration, valid-angle min/max/average, and
  tracking success rate.

Valid-angle summary statistics exclude invalid or low-confidence frames.
Tracking loss frames remain in the observations CSV with a blank angle. The CSVs
are opened in append mode and existing headers are checked; previous sessions
remain available after restarting the app. No video or images are written.

Inspect the summary and recent observations from PowerShell:

```powershell
Import-Csv .\data\session_summaries.csv | Format-Table
Import-Csv .\data\session_observations.csv | Select-Object -Last 20 | Format-Table
```

## Tests

Run the unit tests with:

```powershell
python -m unittest discover -s tests -v
```

Synthetic poses and angles are used only in unit tests. The running app uses
measurements from the live webcam.
