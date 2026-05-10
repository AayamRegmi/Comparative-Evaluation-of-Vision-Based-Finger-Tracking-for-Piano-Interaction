# Piano Finger Tracking — System & Code Explanation

## Pipeline Overview

```
key_calibration.py
        ↓
   record.py          ←→   test.py  (live MPJPE, no files saved)
        ↓
   analyse.py
        ↓
 plot_results.py
```

---

## 1. `key_calibration.py` — Piano Key Overlay

### What it does
Maps every visible piano key to a pixel region and centre point in the camera frame. All other scripts depend on this file.

### Output
`data/calibration/key_centers.json` — list of dicts, one per key:
```json
{
  "midi_note": 60,
  "note_name": "C4",
  "key_type": "white",
  "center": [412, 874],
  "polygon": [[...]]
}
```

### Core function — `build_key_layout()`
```python
def build_key_layout(start_midi, num_white, ox, oy, wkw, wkh):
    bkw = max(4, int(wkw * 0.55))   # black key width = 55% of white
    bkh = int(wkh * 0.62)           # black key height = 62% of white
    keys = []
    white = 0
    for midi in range(start_midi, 128):
        semi = midi % 12
        if semi in _WHITE_SEMIS:       # {0,2,4,5,7,9,11}
            x = ox + white * wkw
            keys.append({
                'center': (x + wkw // 2,  oy + int(wkh * 0.82))  # 82% down = strike zone
            })
            white += 1
        else:
            cx = ox + white * wkw      # X between adjacent white keys
            keys.append({
                'center': (cx, oy + bkh // 2)                     # 50% down black key
            })
```

**Why 82%?** The finger strikes near the bottom of the key, not the geometric centre. Using the geometric centre would add systematic Y-bias.

### UI Controls
| Key / Action | Effect |
|---|---|
| Drag interior | Move overlay |
| Drag edge / corner handles | Resize width / height |
| Scroll wheel | Add / remove white keys on the right |
| `[` / `]` | Shift starting MIDI note |
| `-` / `=` | Fine-adjust key width (±1 px) |
| `C` | Toggle centre visualisation (line vs dot) |
| ENTER / S | Save calibration |

---

## 2. `record.py` — Data Collection

### What it does
Captures synchronised video + MIDI + session metadata for one participant.

### Setup Screen
Overlaid on the live camera feed. Collects:
- **Lux level** — typed manually; instantly shows Dim / Indoor / Bright label
- **Hand size** — cm, validated against config min/max
- **Fitzpatrick type** — optional (1–6); if blank, auto-detected from palm colour
- **MIDI port** — selected with `[` / `]`; last note received shown live for verification

```python
# Validation on ENTER press
lux_val  = float(fields[0]["value"])
hand_val = float(fields[1]["value"])
if not (config.HAND_SIZE_MIN_CM <= hand_val <= config.HAND_SIZE_MAX_CM):
    error_msg = f"Hand size must be {config.HAND_SIZE_MIN_CM}-{config.HAND_SIZE_MAX_CM} cm"
```

### Fitzpatrick Auto-Detection
Called every frame until 20 samples are collected, then averaged for stability:
```python
if not fitz_detected and results.multi_hand_landmarks:
    _, _, ita_sample = detect_skin_type(frame, results.multi_hand_landmarks[0], fw, fh)
    if ita_sample is not None:
        fitz_ita_buf.append(ita_sample)
        if len(fitz_ita_buf) >= 20:          # 20-frame average
            fitz_avg_ita = float(np.mean(fitz_ita_buf))
            fitz_type, fitz_label = ita_to_fitzpatrick(fitz_avg_ita)
            fitz_detected = True
```
> See §5 for how `detect_skin_type` works.

### Shared Clock — Synchronisation
The single most important line for data integrity:
```python
rec_t0 = time.perf_counter()   # set at the instant SPACE is pressed
```
Both `FrameLogger` and `MidiRecorder` timestamp their events relative to this same origin, making video and MIDI directly comparable in offline analysis.

### What Gets Saved
```
data/raw/p001/
├── p001.mp4                  ← raw video (no overlays)
├── p001_frames.csv           ← frame_index, time_s
├── p001_midi.jsonl           ← one JSON line per MIDI event
├── p001_midi.mid             ← standard MIDI file
├── p001_key_centers.json     ← calibration snapshot at recording time
└── p001_session.json         ← all metadata (lux, hand size, Fitzpatrick, fps…)
```

**Raw frames only (no overlays)** — so the offline analyser sees exactly what the camera saw.

### Key Bindings During Recording
| Key | Action |
|---|---|
| SPACE | Start / Stop recording |
| S | Toggle FPS / inference / latency stats |
| R | Re-run Fitzpatrick detection |
| M | Toggle key mask overlay |
| P | Toggle mask control panel |
| N | Next session (new participant ID) |
| ESC | Quit |

---

## 3. `test.py` — Live MPJPE Test (ephemeral)

### What it does
Identical pipeline to `record.py` but **writes nothing to disk**. Play notes live; MPJPE updates in the top-right corner. On ESC, shows a full results panel.

### Per-Note MPJPE Computation
```python
for msg in midi_port.iter_pending():
    if msg.type == 'note_on' and msg.velocity > 0:
        center = _key_center(mask, msg.note)       # (cx, cy) from calibration
        tips   = _get_fingertips(latest_lms, fw, fh, multi_handedness=...)

        if not tips:
            missed += 1
            continue

        # Polygon containment: is any fingertip INSIDE the key's pixel polygon?
        inside_tips = [t for t in tips
                       if cv2.pointPolygonTest(key_poly, (float(t[1]), float(t[2])), False) >= 0]

        if inside_tips:
            best  = min(inside_tips, key=lambda t: abs(t[1] - cx))   # closest X to centre
            h_err = abs(best_tx - cx)     # HORIZONTAL error only (removes depth bias)
            errors.append(h_err)
        else:
            detection_fail[kb_hand] += 1  # hands visible, but no tip on key
```

### Three Outcome Categories
| Category | Condition | Counted in MPJPE? |
|---|---|---|
| `matched` | Fingertip inside key polygon | Yes |
| `detection_fail` | Hands visible, no tip on polygon | No (separate counter) |
| `missed` | No hand landmarks at all | No |

**Why exclude detection_fail from MPJPE?** Including ghost/misidentified landmarks would inflate the error unfairly — it is a separate research metric from accuracy.

### Why Horizontal Error Only?
```python
h_err = abs(tip_x - key_cx)    # X only
# Y is discarded — finger length, keystroke depth, and hand angle all affect Y
# but say nothing about whether the model placed the landmark on the correct key
```

### Hand Labelling (L/R)
MediaPipe's handedness label is designed for selfie (mirrored) cameras. The system cross-validates it with anatomical geometry:
```python
wrist_x     = hand_lms.landmark[0].x * fw
thumb_mcp_x = hand_lms.landmark[2].x * fw
# For piano (palm-down): left thumb MCP is to the RIGHT of the wrist
geo_label   = 'L' if thumb_mcp_x > wrist_x else 'R'
# If MP label disagrees with geometry -> geometry wins (handles camera mirror)
return mp_label if mp_label == geo_label else geo_label
```

### Visual Flash System
| Colour | Meaning |
|---|---|
| Green line + circle | Matched, error < 20 px |
| Yellow | Matched, error 20–40 px |
| Red | Matched, error ≥ 40 px |
| Magenta crosshair | Detection fail |

A vertical line spans the full key height showing the target X; a horizontal line from tip to centre shows the actual error.

### Results Overlay (on ESC)
Rendered with PIL for clean text, displayed on the last captured frame:
- Overall MPJPE (mean + median), accuracy %, detection rate %
- Per-finger breakdown: Left and Right hands, all 5 fingers

---

## 4. `analyse.py` — Offline Analysis

### What it does
Re-runs hand-landmark detection over the recorded video — frame by frame at each MIDI note event — and writes a JSON results file. Supports **MediaPipe** and **OpenPose**.

### Frame Alignment
```python
def _nearest_frame_idx(event_time, frame_times):
    ts_list = [t for t, _ in frame_times]
    i = bisect.bisect_left(ts_list, event_time)   # binary search
    # Pick whichever neighbour is temporally closer
    if i > 0 and abs(ts_list[i-1] - event_time) < abs(ts_list[i] - event_time):
        i -= 1
    return frame_times[i][1]
```
For each MIDI `note_on`, seek the video to the nearest frame by timestamp, then run the model.

### MediaPipe (offline)
```python
_mp = mp.solutions.hands.Hands(
    static_image_mode=True,   # important: no temporal tracking when seeking random frames
    ...
)
```
`static_image_mode=True` is critical — in live mode MediaPipe uses temporal context from the previous frame, which is meaningless when jumping to arbitrary frames.

### OpenPose — Crop Strategy
```python
# Crop to keyboard area + 500px above (where hands actually are)
x0 = max(0, int(min(key_xs)) - 150)
x1 = min(w, int(max(key_xs)) + 150)
y0 = max(0, int(min(key_ys)) - 500)   # hand height above keys
y1 = min(h, int(max(key_ys)) + 80)
crop = frame[y0:y1, x0:x1]
# Offset keypoint coordinates back to full-frame space after inference
kps_offset = [(int(kp[0]+x0), int(kp[1]+y0), kp[2]) for kp in kps]
```
**Why crop?** Full 1920×1080 squished to OpenPose's 368×368 input makes hands ~20 px — too small to detect reliably.

### Output JSON Structure
```json
{
  "pid": "p001",
  "model": "mediapipe",
  "lux": 320,
  "fitzpatrick_type": 3,
  "hand_size_cm": 18.5,
  "detection_rate_pct": 91.3,
  "mjmpe_px": 14.7,
  "per_hand": {
    "L": {
      "matched": 48, "detection_fail": 4, "mjmpe_px": 13.2,
      "accuracy_pct": 87.5,
      "fingers": { "0": 15.1, "1": 12.4, "2": 11.8, "3": 14.0, "4": 16.3 }
    },
    "R": { "..." }
  }
}
```
> Note: JSON keys use `mjmpe_px` (historical); display labels say MPJPE.

### CLI Usage
```bash
python -m scripts.analyse data/raw/p001/ --model mediapipe
python -m scripts.analyse data/raw/p001/ --model openpose
python -m scripts.analyse --ui    # tkinter GUI listing all sessions
```

---

## 5. `fitzpatrick_detector.py` — Skin Type Detection

### Method — ITA (Individual Typology Angle)
Based on Chardon et al. (1991). No external ML model — pure colorimetry.

```python
# ITA formula (CIELab colour space)
# ITA = arctan((L* - 50) / b*) × (180/π)
```

**Steps:**
1. Extract palm pixel region using MediaPipe landmarks 0,1,5,9,13,17 (wrist + MCP joints) as a convex polygon mask.
2. Convert those pixels from BGR → CIELab.
3. Compute mean L* and b* across all palm pixels.
4. Calculate ITA angle.
5. Map angle to Fitzpatrick type:

```python
_ITA_THRESHOLDS = [
    (55,  1, "I - Very Light"),
    (41,  2, "II - Light"),
    (28,  3, "III - Med Light"),
    (10,  4, "IV - Med Dark"),
    (-30, 5, "V - Dark"),
    # anything below → VI - Very Dark
]
```

Averaged over **20 frames** in `record.py` for stability before locking in the result.

---

## 6. `plot_results.py` — Result Visualisation

Loads all `*_results.json` from `data/processed/` and saves 14 PNG charts to `data/plots/`.

| File | Chart |
|---|---|
| `00_participant_composition.png` | Fitzpatrick, lighting, hand size distributions |
| `01_model_comparison_mpjpe.png` | MediaPipe vs OpenPose MPJPE per participant |
| `02_per_finger_mpjpe.png` | Per-finger MPJPE, L/R subplots, both models |
| `03_detection_breakdown.png` | Matched / detection-fail / missed proportions |
| `04_mpjpe_by_lux.png` | MPJPE by lighting (Dim / Indoor / Bright) |
| `05_mpjpe_by_fitzpatrick.png` | MPJPE by Fitzpatrick skin type |
| `06_mpjpe_vs_handsize.png` | Scatter: MPJPE vs hand size with regression |
| `07_finger_distribution_<model>.png` | Box plots of per-finger MPJPE across sessions |
| `08_heatmap_<model>.png` | Per-finger MPJPE heatmap across participants |
| `09_detection_fail_by_fitzpatrick.png` | Detection-fail rate by skin type |
| `10_detection_fail_by_lux.png` | Detection-fail rate by lighting |
| `11_mpjpe_by_hand.png` | Left vs Right hand MPJPE |
| `12_accuracy_by_finger.png` | Per-finger accuracy — MediaPipe |
| `13_accuracy_by_finger_openpose.png` | Per-finger accuracy — OpenPose |

### Filtering
```bash
python -m scripts.plot_results --pid p001 p003   # specific participants
python -m scripts.plot_results --out figures/    # custom output folder
```

---

## 7. Key Terms Glossary

| Term | Definition |
|---|---|
| **MPJPE** | Mean Per Joint Position Error — average horizontal pixel error across all matched note events |
| **Detection rate** | `matched / (matched + detection_fail + missed)` — how reliably the model places a landmark on the pressed key |
| **Accuracy** | % of matched notes where `h_err < wkw / 2` (within half a white key width) |
| **Detection fail** | Hands visible in frame but no fingertip inside the key polygon |
| **Missed** | No hand landmarks detected at all |
| **ITA angle** | Individual Typology Angle — colorimetric skin tone measure from CIELab: `arctan((L*−50)/b*)×(180/π)` |
| **Horizontal-only error** | Y is discarded; only X deviation from key centre is measured |
| **Shared clock (`rec_t0`)** | `time.perf_counter()` set at SPACE-press; both video frames and MIDI events timestamp from this origin |
| **Polygon containment** | `cv2.pointPolygonTest` — checks if a fingertip pixel is inside the calibrated key polygon |
| **`static_image_mode=True`** | MediaPipe flag used in offline analysis; disables temporal tracking which is invalid for random-frame seeking |
