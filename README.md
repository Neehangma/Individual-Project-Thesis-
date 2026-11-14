# Elbow Rehabilitation Exercise Monitoring System

A local Streamlit prototype for monitoring elbow flexion and extension with a live webcam. It reuses the project's MediaPipe pose detector, geometrically correct elbow-angle calculation, state-based repetition counter, feedback engine, session manager, and append-only CSV logger. The application does not save camera images or video.

## Setup (Windows, Python 3.11)

Open PowerShell in this project directory:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Alternatively, after dependencies are installed, run `run_project.bat`. Streamlit opens the application in your browser. Use `Ctrl+C` in the terminal to stop the server; the application releases the camera on session end or server shutdown.

To run with the verification environment created in this workspace, use `.\.venv-ui\Scripts\python.exe -m streamlit run app.py`. For a fresh setup, use the `.venv` commands above.

## Pages

- **Home** introduces the system and camera setup.
- **Exercise Monitoring** starts and controls a real webcam session.
- **Session Results** reads persisted summaries and observations, charts a selected session, and downloads its CSV records.
- **Research Analytics** filters the stored research records, shows descriptive system metrics, compares sessions, and exports the filtered tables.
- **About the System** describes the method and its limitations.

## Research Analytics measures

The dashboard reads `data/session_summaries.csv` and `data/session_observations.csv` on each page run. Participant and session totals count unique anonymised IDs across both files, so an observation-only session is still represented. Completed/interrupted/cancelled counts and session duration statistics come from saved summaries. Completed repetition totals prefer one value per summary; for a session with observations but no summary, the latest cumulative observation count is used once. Average repetitions is the mean of those per-session values. Session completion rate is shown only when every filtered session has a known saved status.

An angle observation is valid only when it parses to a finite number from 0 through 180 degrees. Blanks, invalid strings, infinities, and out-of-range values count as invalid and are excluded from angle statistics. Tracking success is the share of filtered observations with a known tracking status marked `Tracking stable`. Date filtering uses a saved session start time, or the first available observation timestamp when there is no summary. Filtering and downloads operate on copies and never rewrite the research CSVs; exact duplicate observation rows are suppressed in the analytics view while timestamp-distinct samples are retained.

The charts compare actual saved angle observations, per-session repetition targets and counts, summary durations and statuses, and tracking status records. No clinical angle accuracy or repetition-count accuracy is calculated: those require independent ground-truth angle measurements and manually verified repetition counts, respectively.

## Live session steps

1. Allow the browser/operating system to grant webcam access. Close other apps using the camera.
2. On Exercise Monitoring, enter an anonymous participant code (1–32 letters, digits, underscores, or hyphens), select the anatomical arm, and set a repetition target from 1 to 30.
3. Select **Start Session**. A three-second get-ready period begins after the camera opens. The selected arm's shoulder, elbow, and wrist must remain in view.
4. Follow the on-screen angle, tracking status, movement phase, repetition progress, and brief live feedback. The counter uses the configured extension/flexion thresholds in `capture_controller.py` and requires a complete extension–flexion–extension cycle.
5. Select **Pause** / **Resume** as needed. **End Session** asks for confirmation before saving. A session reaching its repetition target is marked completed; an early stop with valid measurements is interrupted; a session with no valid exercise observations is cancelled.
6. Use **Session Results** to review the actual session record and download the observation or summary CSV.

The session target defaults to 10 repetitions. Set capacity is three sets of ten, with a ten-second rest between full sets. Change the constants near the top of `capture_controller.py` for supervised research configuration. Thresholds are software settings, not universal clinical recommendations; tune them using supervised pilot testing with the intended camera setup and population.

## Webcam processing and limitations

Streamlit's timed fragment reads and processes one live frame at a time. The per-browser capture controller persists in Streamlit session state, while a process-wide camera lease prevents another browser session from opening the same webcam. The active fragment continues sampling while navigating across pages. At most one observation is logged every 100 ms; invalid or low-confidence angles are stored as blank values. MediaPipe landmark names represent the participant's anatomical left and right sides, independently of the mirrored display.

The elbow angle is a single-camera 2D image-plane estimate, not a clinically validated three-dimensional measurement. Pose quality can be affected by camera placement, occlusion, lighting, clothing, and body orientation. This is an academic prototype for exercise monitoring and technical evaluation; it is not a medical diagnostic system and does not replace professional physiotherapy advice.

## Research data

Data is kept locally in `data/` and appended without overwriting previous records:

- `data/session_observations.csv`: timestamped session and participant IDs, elapsed seconds, selected arm, elbow angle (blank for invalid tracking), pose confidence, tracking status, movement state, completed/target repetitions, and feedback message.
- `data/session_summaries.csv`: one row per session with exercise and arm, target/completed repetitions, incomplete movements, start/end times, duration, session status, valid-angle statistics, and tracking success rate.

Participant codes must be anonymous. Collect research data only under the approved participant-consent procedure. CSV rows are not committed to version control. The application creates headers automatically and checks existing column schemas.

Inspect saved records with PowerShell:

```powershell
Import-Csv .\data\session_summaries.csv | Format-Table
Import-Csv .\data\session_observations.csv | Select-Object -Last 20 | Format-Table
```

## Tests and manual camera check

Run the automated test suite:

```powershell
python -m unittest discover -s tests -v
```

Manual end-to-end checklist:

1. Start `python -m streamlit run app.py`; verify Home, Monitoring, Results, Research Analytics, and About navigation.
2. On Monitoring, use a study code and select LEFT. Start with the selected arm visible. Check that landmarks, an updating elbow angle, confidence, and stable/lost tracking status appear.
3. Slowly bend and extend through the configured thresholds; confirm the count increments once only after a complete cycle. Repeat with RIGHT selected and compare the anatomical label.
4. Pause and resume; confirm the timer excludes the pause and no measurements are collected while paused. End and save.
5. Open Session Results; verify the saved session summary, angle chart (when valid samples exist), and both CSV downloads.
6. Restart the application and confirm the previous session remains listed. Try starting two browser sessions and confirm the second receives the camera-in-use message.
7. Test with the webcam unavailable (close the camera device or choose no accessible camera) and verify the app shows a readable error rather than crashing.
8. On Research Analytics, exercise the participant, session, arm, status, and date filters; download both filtered CSVs and verify the originals remain unchanged.
