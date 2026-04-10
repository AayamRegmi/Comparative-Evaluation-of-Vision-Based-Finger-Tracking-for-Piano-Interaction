# Script Reference

This document explains what each script in the `scripts/` package does, what it depends on, and how it fits into the overall pipeline.

---

## Pipeline Overview

```
key_calibration.py          ← run once per tripod position
        ↓
record.py                   ← capture video + MIDI per participant
        ↓
recalibrate.py              ← (optional) fix calibration on saved video
        ↓
analyse.py                  ← compute MJMPE results per session
        ↓
plot_results.py             ← generate charts and participant reports
data_summary.py             ← count rows across all sessions
```

`live_preview.py` is a standalone live preview/test tool that does not write any data.

---

## config.py

**Purpose:** Central store for every tunable constant in the project. All other scripts import this instead of hard-coding values.

**Depends on:** nothing (no internal imports)

**Key values:**
| Constant | Description |
|---|---|
| `CAP_WIDTH/HEIGHT/FPS` | Camera capture resolution and frame rate |
| `RESIZE_WIDTH/HEIGHT` | Resolution MediaPipe runs inference at |
| `MODEL_COMPLEXITY` | MediaPipe model complexity (0 = lite, 1 = full) |
| `MIN_DETECTION/TRACKING_CONFIDENCE` | MediaPipe thresholds |
| `LUX_DIM_THRESHOLD` | Below this (lux) → Dim label |
| `LUX_BRIGHT_THRESHOLD` | At or above this → Bright label |
| `HAND_SIZE_MIN/MAX_CM` | Validation range for manual hand size input |
| `HAND_SIZE_SMALL_MAX_CM` | Hand length below this → Small category |
| `HAND_SIZE_LARGE_MIN_CM` | Hand length at or above this → Large category |
| `ACCURACY_THRESHOLD_RATIO` | Fraction of white key width used as accuracy threshold |
| `MASK_*` | Colours for the key overlay drawn on-screen |

---

## camera_setup.py

**Purpose:** Opens the camera with the correct backend for the current OS and applies the resolution and FPS from config. Returns a ready `cv2.VideoCapture` object.

**Depends on:** `config`

**Key function:**
- `init_camera(config)` — opens the camera using `CAP_DSHOW` on Windows (required to avoid DirectShow double-open conflicts), sets width/height/FPS, returns `cap`

**Used by:** `record.py`, `key_calibration.py`, `live_preview.py`

---

## lux_calculator.py

**Purpose:** Converts a numeric lux value into a human-readable lighting category label (Dim / Indoor / Bright) using the thresholds in config.

**Depends on:** `config`

**Key function:**
- `lux_to_label(lux)` → `"Dim"` / `"Indoor"` / `"Bright"`

**Used by:** `record.py`, `live_preview.py`

---

## fitzpatrick_detector.py

**Purpose:** Estimates the participant's Fitzpatrick skin type (I–VI) from a live video frame using the ITA (Individual Typology Angle) colorimetric method. Samples CIELab pixel values from the palm convex hull defined by MediaPipe landmarks 0, 1, 5, 9, 13, 17. Runs once per session after the first hand is detected.

**Depends on:** `config`, `cv2`, `numpy`

**Key functions:**
- `detect_skin_type(frame, hand_landmarks, fw, fh)` → `(fitzpatrick_type, label, ita_angle)` or `(None, None, None)` if palm not visible
- `ita_to_fitzpatrick(ita)` → `(type_int, label_str)` — maps ITA angle to the six Fitzpatrick categories using Chardon (1991) thresholds

**Used by:** `record.py`, `live_preview.py`

---

## stats_collector.py

**Purpose:** Rolling statistics tracker for the recording/preview loop. Computes FPS, MediaPipe inference time, and total frame latency over a sliding window. Has a warmup phase (first N frames discarded) before stats stabilise.

**Depends on:** `numpy`, `collections.deque`

**Key class:** `StatsCollector(warmup_frames, stat_frames)`
- `update(loop_start, inference_ms, latency_ms)` — call once per frame
- `fps`, `inference_ms`, `latency_ms` — current rolling averages
- `warmup_done` — True once warmup frames have elapsed
- `print_final_stats(w, h, complexity)` — prints summary on exit

**Used by:** `record.py`, `live_preview.py`

---

## midi_recorder.py

**Purpose:** Threaded MIDI input capture with `time.perf_counter()` timestamps. Uses `mido` and `python-rtmidi`. All timestamps are relative to a shared `t0` origin that is also used by the video frame logger, enabling MIDI-to-frame synchronisation in analysis.

**Depends on:** `mido`, `python-rtmidi`, `json`, `threading`, `pathlib`

**Key classes:**

`MidiRecorder`
- `start(t0)` — begins background daemon thread polling the MIDI port every 1 ms. Skips realtime messages (clock, active sensing).
- `stop()` — joins thread
- `save(out_dir, pid_str)` → `(midi_jsonl_path, midi_mid_path)` — writes `p###_midi.jsonl` (one JSON line per event) and `p###_midi.mid` (standard MIDI file, type 0, 120 BPM)

`FrameLogger`
- `log()` — call once per `writer.write(frame)`. Records `(frame_index, time_s)` relative to `t0`.
- `save(out_dir, pid_str)` → `frames_csv_path` — writes `p###_frames.csv`

`list_midi_input_ports()` — returns list of detected port name strings

**Used by:** `record.py`, `live_preview.py`

---

## key_calibration.py

**Purpose:** Interactive GUI for aligning a virtual piano keyboard overlay over the camera view. The overlay parameters (position, key width/height, starting MIDI note, perspective warp corners) are saved to `data/calibration/key_centers.json`. This file is used by `analyse.py` to know where each key's polygon is in the video frame.

**Depends on:** `config`, `camera_setup`, `cv2`, `numpy`, `json`, `math`

**Key class:** `KeyMask`
- Stores the overlay geometry and computes per-key polygons with optional perspective warp
- `draw(frame)` — blends the key overlay onto a frame in-place
- `on_mouse(event, x, y, flags)` — handles drag-to-move, edge-drag-to-resize, corner-drag-to-warp, scroll-to-add/remove-keys
- `save(cal_dir, fw, fh, filename="key_centers.json")` — writes JSON with all parameters plus pre-computed key centre coordinates
- `KeyMask.load(path)` — reconstructs a `KeyMask` from a saved JSON file

**Key function:**
- `build_key_layout(start_midi, num_white, ox, oy, wkw, wkh)` — exported utility that re-derives centre coordinates from saved parameters; used by `analyse.py`

**Controls (live):**
| Key / Action | Effect |
|---|---|
| Drag interior | Move keyboard |
| Drag left/right edge | Resize key width |
| Drag bottom edge | Resize key height |
| Drag corner | Warp perspective |
| Scroll | Add/remove key on right end |
| `[` / `]` | Shift starting MIDI note |
| `-` / `=` | Fine key width ±1 px |
| `F` | Toggle 180° flip to match recording orientation |
| `V` | Reset warp to rectangle |
| `ENTER` or `S` | Save calibration |
| `ESC` | Quit without saving |

**Used by:** `record.py`, `live_preview.py`, `analyse.py`, `recalibrate.py`

---

## model_manager.py

**Purpose:** Abstract interface and concrete wrappers for switching between hand-tracking models at runtime. Defines a common API (`load`, `infer`, `draw`, `close`) so `live_preview.py` can swap models without changing its main loop.

**Depends on:** `config`, `cv2`, `numpy`, `mediapipe`, `onnxruntime`

**Key classes:**
- `MediaPipeHandModel` — wraps `mp.solutions.hands`; `infer()` returns MediaPipe result object
- `OpenPoseHandModel` — wraps an ONNX hand model; `infer()` uses `cv2.minMaxLoc` to extract the 21 keypoints from heatmaps. Note: operates on a cropped region of the frame, so when both hands are visible the keypoints from different hands can be mixed.

**Used by:** `live_preview.py`, `analyse.py`

---

## record.py

**Purpose:** Main data collection script. Runs a live preview with MediaPipe hand tracking, allows the operator to enter session metadata (lux, hand size, Fitzpatrick type, MIDI port), then records raw video and MIDI to `data/raw/p###/`. Multiple sessions can be recorded back-to-back without restarting — pressing `N` returns to the setup screen with the next participant ID while keeping the camera and MediaPipe model running.

**Depends on:**
| Dependency | Used for |
|---|---|
| `config` | All tunables |
| `camera_setup.init_camera` | Opening the camera |
| `fitzpatrick_detector` | Auto-detecting skin type from palm pixels |
| `lux_calculator.lux_to_label` | Converting entered lux to Dim/Indoor/Bright |
| `stats_collector.StatsCollector` | Rolling FPS / inference / latency stats |
| `midi_recorder.MidiRecorder` | Background MIDI capture thread |
| `midi_recorder.FrameLogger` | Per-frame timestamp logging |
| `key_calibration.KeyMask` | Drawing and interacting with the key overlay |
| `mediapipe` | Live hand landmark detection |
| `cv2`, `numpy`, `mido` | Video capture, display, MIDI port listing |

**Output files per session** (written to `data/raw/p###/`):
| File | Contents |
|---|---|
| `p###.mp4` | Raw video, no overlays, flip baked in if `flip_y=True` |
| `p###_frames.csv` | `frame_index, time_s` — one row per recorded frame |
| `p###_midi.jsonl` | One JSON line per MIDI event with `time_s` offset from recording start |
| `p###_midi.mid` | Standard MIDI file (type 0, 120 BPM) |
| `p###_session.json` | Session metadata (lux, hand size, Fitzpatrick, flip state, file names, MIDI port) |
| `p###_key_centers.json` | Snapshot of the active calibration at the moment recording started |

**Key controls:**
| Key | Action |
|---|---|
| `SPACE` | Start / stop recording |
| `N` | Finish current session and go to setup screen for next participant |
| `S` | Toggle stats overlay |
| `M` | Toggle key overlay visibility |
| `P` | Toggle mask control panel |
| `R` | Re-run Fitzpatrick auto-detection |
| `F` | Toggle 180° frame flip |
| `V` | Reset mask warp |
| `ESC` | Quit |

---

## live_preview.py

**Purpose:** Live hand-tracking preview and informal accuracy test. Mirrors much of `record.py`'s display but **writes nothing to disk**. Shows MJMPE, detection rate, and per-finger stats in a live overlay. At the end (ESC) displays a results summary. Used to verify model performance and system setup before a real recording session.

**Depends on:**
| Dependency | Used for |
|---|---|
| `config` | All tunables |
| `camera_setup.init_camera` | Camera initialisation |
| `key_calibration.KeyMask` | Key overlay for live MJMPE computation |
| `midi_recorder.list_midi_input_ports` | MIDI port listing in setup |
| `stats_collector.StatsCollector` | Rolling performance stats |
| `model_manager` | Swappable MediaPipe / OpenPose model interface |
| `mediapipe` | Hand landmark detection |
| `cv2`, `numpy`, `mido` | Video, display, MIDI |

**Note:** `live_preview.py` also contains `_resolve_hand_label()` which classifies detected hands as L or R using thumb-wrist geometry combined with MediaPipe's `multi_handedness`, accounting for mirrored camera setups. The same logic is duplicated in `analyse.py` as `_hand_label_from_mp()`.

---

## key_calibration.py → recalibrate.py

**Purpose:** GUI tool for aligning the key overlay against a **pre-recorded session video**. Useful when the session calibration file was lost, or the tripod moved between calibrating and recording. Saves `p###_key_centers.json` directly into the session folder so `analyse.py` picks it up automatically.

**Depends on:** `key_calibration.KeyMask`, `cv2`, `numpy`, `json`, `csv`

**How it works:**
1. Opens a **participant picker** — scrollable list of all `data/raw/p###/` folders. Green dot = already calibrated, blue dot = needs calibration.
2. Click a participant → opens the **calibration editor** which plays back that session's video.
3. Align the overlay to match the piano in the video. The `NOW: C5` indicator in the top bar shows which note was being played at the current frame (loaded from `p###_midi.jsonl` and `p###_frames.csv`), and highlights the corresponding key polygon in cyan.
4. Press `ENTER` or `S` to save `p###_key_centers.json` and return to the picker.

**Usage:** `python -m scripts.recalibrate`

---

## analyse.py

**Purpose:** Offline per-session MJMPE computation. For each `note_on` event in the MIDI log, it seeks to the nearest video frame, runs the selected model on that frame, and checks whether any fingertip landmark falls inside the corresponding key polygon. Produces a JSON results file in `data/processed/`.

**Depends on:**
| Dependency | Used for |
|---|---|
| `config` | Tunables (confidence thresholds, accuracy ratio) |
| `key_calibration.KeyMask` | Loading key polygons from calibration file |
| `model_manager.OpenPoseHandModel` | OpenPose inference |
| `mediapipe` | MediaPipe inference |
| `cv2`, `numpy` | Video reading, frame processing |

**Calibration priority:** looks for `data/raw/p###/p###_key_centers.json` first (per-session snapshot); falls back to `data/calibration/key_centers.json` with a warning.

**Three outcome categories per note event:**
| Outcome | Meaning |
|---|---|
| `matched` | A fingertip was inside the key polygon → contributes to MJMPE |
| `detection_fail` | Hands visible but no tip landed inside the polygon |
| `missed` | No hand landmarks detected at all |

**Hand classification:**
- MediaPipe: uses `multi_handedness` label combined with thumb-MCP-to-wrist geometry to determine physical L/R, accounting for mirrored cameras.
- OpenPose: returns `None` (geometry unreliable due to scene-level crop); falls back to keyboard-position split (left of midpoint = L).

**Output:** `data/processed/p###_mediapipe_results.json` and `p###_openpose_results.json`

**Usage:**
```bash
python -m scripts.analyse data/raw/p001          # CLI, both models
python -m scripts.analyse data/raw/p001 --model mediapipe
python -m scripts.analyse --ui                   # graphical session picker
```

---

## plot_results.py

**Purpose:** Loads all processed result JSONs and generates publication-ready charts plus individual participant reports.

**Depends on:** `matplotlib`, `numpy`, `json`, `pathlib`

**Charts generated** (saved to `data/plots/`):
| File | Description |
|---|---|
| `01_model_comparison_mjmpe.png` | Bar chart: MediaPipe vs OpenPose MJMPE per participant |
| `02_per_finger_mjmpe.png` | Mean per-finger MJMPE — MediaPipe L, MediaPipe R, OpenPose combined |
| `03_detection_breakdown.png` | Stacked bar: matched / detection-fail / missed proportions per model |
| `04_mjmpe_by_lux.png` | MJMPE grouped by lighting condition with ±SD error bars |
| `05_mjmpe_by_fitzpatrick.png` | MJMPE grouped by Fitzpatrick type with ±SD error bars |
| `06_mjmpe_vs_handsize.png` | Scatter plot of MJMPE vs hand size with regression lines |
| `07_finger_distribution_mediapipe.png` | Box plots of per-finger MJMPE distribution across sessions |
| `07_finger_distribution_openpose.png` | Same for OpenPose (combined L+R) |
| `08_heatmap_mediapipe.png` | Heatmap: participants × fingers (10 columns, L/R split) |
| `08_heatmap_openpose.png` | Heatmap: participants × fingers (5 columns, combined) |
| `reports/p###_report.png` | Individual participant report (see below) |

**Participant report layout:**
- Top left: session info (PID, skin type, lighting, hand size, notes played)
- Top centre: detection outcome bar chart (Matched / Det. Fail / Missed)
- Top right: overall results (MJMPE, accuracy %, detection rate for both models)
- Bottom left/centre: per-hand bar charts coloured by accuracy tier (≤4 px green / 4–7 px orange / >7 px red)
- Bottom right: highlights (most/least accurate finger, detection rate rating, overall accuracy rating)

**Usage:**
```bash
python -m scripts.plot_results                    # all sessions
python -m scripts.plot_results --pid p001 p003    # specific sessions
python -m scripts.plot_results --out figures/     # custom output folder
```

---

## data_summary.py

**Purpose:** Quick audit of how much data has been collected across all sessions. Prints a table with per-session frame counts, MIDI event counts, note-on counts, and matched notes for each model.

**Depends on:** `json`, `csv`, `pathlib`

**Usage:** `python -m scripts.data_summary`

**Output columns:**
| Column | Description |
|---|---|
| Frames | Video frames recorded |
| MIDI evts | Total MIDI events (all types) |
| note_on | Key-press events (velocity > 0) |
| MP matched | Notes where MediaPipe tip landed on key |
| OP matched | Notes where OpenPose tip landed on key |

---

## Dependency Graph

```
config.py
├── camera_setup.py
├── lux_calculator.py
├── fitzpatrick_detector.py
├── stats_collector.py
├── midi_recorder.py
├── key_calibration.py
│   └── recalibrate.py
├── model_manager.py
│   ├── live_preview.py ──────────────────────── (no data written)
│   └── analyse.py
│       └── plot_results.py
└── record.py  (uses all of the above)

data_summary.py  (reads data/raw/ directly, no script imports)
```
