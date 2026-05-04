# Installation Guide

## Requirements

- **Python 3.10** (required — MediaPipe 0.10.x does not support Python 3.11+)
- A webcam or USB camera
- A MIDI keyboard connected via USB

---

## 1. Clone the repository

```
git clone https://github.com/AayamRegmi/Comparative-Evaluation-of-Vision-Based-Finger-Tracking-for-Piano-Interaction.git
cd Comparative-Evaluation-of-Vision-Based-Finger-Tracking-for-Piano-Interaction
```

---

## 2. Create a virtual environment

```
python -m venv piano_env
```

**Activate it:**

- Windows: `piano_env\Scripts\activate`
- macOS / Linux: `source piano_env/bin/activate`

---

## 3. Install Python dependencies

```
pip install -r requirements.txt
```

This installs: NumPy, Pandas, OpenCV, MediaPipe, Mido, python-rtmidi, Matplotlib, Seaborn, SciPy, and tqdm.

> **Windows note:** `python-rtmidi` requires the Microsoft Visual C++ Build Tools to compile. If the install fails, download and install the [Build Tools for Visual Studio](https://visualstudio.microsoft.com/visual-cpp-build-tools/) (select "Desktop development with C++"), then re-run `pip install python-rtmidi`.

> **tensorflow** is listed in `requirements.txt` for potential future model extensions but is not used by any current script. It can be skipped if disk space is a concern (`pip install -r requirements.txt --ignore-requires-python` and exclude the tensorflow lines manually).

---

## 4. Install additional packages

These packages are used by the analysis and results scripts but are not in `requirements.txt`:

```
pip install Pillow python-pptx python-docx
```

| Package | Used by |
|---|---|
| Pillow | `live_preview.py` — results overlay rendering |
| python-pptx | `redesign_pptx.py` — presentation generation |
| python-docx | Documentation scripts |

---

## 5. OpenPose model files

OpenPose inference uses pre-trained Caffe model weights loaded via OpenCV DNN. The model files are included in the repository under `scripts/`:

- `scripts/openpose_hand.prototxt`
- `scripts/openpose_hand.caffemodel` (~55 MB)

No separate OpenPose installation is required — OpenCV handles inference directly.

---

## 6. Verify the installation

Run the following to confirm everything is working:

```
python -c "import cv2, mediapipe, mido, numpy, matplotlib, PIL; print('All core imports OK')"
```

Then launch the live test:

```
python -m scripts.live_preview
```

---

## 7. MIDI setup

- Connect your MIDI keyboard via USB before launching any script.
- The setup screen will list detected MIDI ports. Use `[` / `]` to cycle between them.
- If no port is detected, check that your keyboard is powered on and that a MIDI driver is installed (Windows may require a driver from the keyboard manufacturer).

---

## Running the scripts

| Script | Command | Purpose |
|---|---|---|
| Key calibration | `python -m scripts.key_calibration` | Align the on-screen keyboard overlay to the camera view |
| Live MPJPE test | `python -m scripts.live_preview` | Real-time accuracy measurement (no data saved) |
| Recording session | `python -m scripts.record` | Record video + MIDI for offline analysis |
| Offline analysis | `python -m scripts.analyse` | Compute per-session MPJPE from recordings |
| Plot results | `python -m scripts.plot_results` | Generate charts from all session results |
