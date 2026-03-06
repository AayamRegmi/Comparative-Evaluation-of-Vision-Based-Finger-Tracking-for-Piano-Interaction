# record.py
# Recording preview with:
#   - Setup screen: manual lux level + hand size input
#   - Auto Fitzpatrick skin-type detection (one-shot on first hand detected)
#   - REC / STOP toggle via SPACE bar
# Later saves raw data to data/raw/<participant_id>/ (video + session metadata)
#
# Key bindings during preview:
#   SPACE  – start / stop recording
#   S      – toggle live stats overlay (FPS / inference / latency)
#   R      – retest Fitzpatrick skin type
#   ESC    – quit

import ctypes
import ctypes.wintypes
import cv2
import json
import mediapipe as mp
import numpy as np
import os
import mido
import time
from datetime import datetime
from pathlib import Path

try:
    from . import config
    from .camera_setup import init_camera
    from .fitzpatrick_detector import detect_skin_type
    from .lux_calculator import lux_to_label
    from .stats_collector import StatsCollector
    from .midi_recorder import MidiRecorder, FrameLogger, list_midi_input_ports
    from .key_calibration import KeyMask, draw_mask_handles, MaskControlPanel
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import config
    from camera_setup import init_camera
    from fitzpatrick_detector import detect_skin_type
    from lux_calculator import lux_to_label
    from stats_collector import StatsCollector
    from midi_recorder import MidiRecorder, FrameLogger, list_midi_input_ports
    from key_calibration import KeyMask, draw_mask_handles, MaskControlPanel

# Root folder for all raw participant data
_RAW_DIR = Path(__file__).parent.parent / "data" / "raw"


# ---------------------------------------------------------------------------
# Recording helpers
# ---------------------------------------------------------------------------

def _hand_size_label(cm: float) -> str:
    if cm < config.HAND_SIZE_SMALL_MAX_CM:
        return "Small"
    if cm >= config.HAND_SIZE_LARGE_MIN_CM:
        return "Large"
    return "Medium"


_NOTE_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

def _midi_note_name(n):
    return _NOTE_NAMES[n % 12] + str(n // 12 - 1)


def _hand_size_label(cm):
    if cm < config.HAND_SIZE_SMALL_MAX_CM: return "Small"
    if cm >= config.HAND_SIZE_LARGE_MIN_CM: return "Large"
    return "Medium"


def _next_participant_id() -> int:
    """Return the next available participant number (scans data/raw/p### folders)."""
    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    nums = []
    for d in _RAW_DIR.iterdir():
        if d.is_dir() and d.name.startswith("p"):
            try:
                nums.append(int(d.name[1:]))
            except ValueError:
                pass
    return max(nums, default=0) + 1


def _start_recording(frame, pid: int, fps: float) -> tuple:
    """
    Create the participant folder, open a VideoWriter, and return
    (writer, out_dir, video_path).
    fps should be the actual measured frame rate, not config.CAP_FPS.
    """
    pid_str = f"p{pid:03d}"
    out_dir = _RAW_DIR / pid_str
    out_dir.mkdir(parents=True, exist_ok=True)

    video_path = out_dir / f"{pid_str}.mp4"
    h, w = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
    return writer, out_dir, video_path


def _save_session_metadata(out_dir: Path, pid: int, lux_value: float,
                            lux_label: str, hand_size_cm: float,
                            hand_size_label: str, fitz_type: int, fitz_label: str,
                            video_path: Path, rec_start: str, rec_stop: str,
                            midi_port: str = None, frames_csv: Path = None,
                            midi_jsonl: Path = None, midi_mid: Path = None,
                            video_fps: float = None):
    """Write session metadata to <pid>_session.json."""
    meta = {
        "participant_id": f"p{pid:03d}",
        "recording_start": rec_start,
        "recording_stop":  rec_stop,
        "lux_value":       lux_value,
        "lux_label":       lux_label,
        "hand_size_cm":    hand_size_cm,
        "hand_size_label": hand_size_label,
        "flip_y":          flip_y,
        "hand_size_label": hand_size_label,
        "fitzpatrick_type":  fitz_type,
        "fitzpatrick_label": fitz_label,
        "video_file":      video_path.name,
        "video_fps":       round(video_fps, 2) if video_fps else None,
        "frames_csv":      frames_csv.name  if frames_csv  else None,
        "midi_port":       midi_port,
        "midi_jsonl":      midi_jsonl.name  if midi_jsonl  else None,
        "midi_mid":        midi_mid.name    if midi_mid    else None,
    }
    out_path = out_dir / f"p{pid:03d}_session.json"
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Session metadata saved → {out_path}")


# ---------------------------------------------------------------------------
# Window helpers
# ---------------------------------------------------------------------------

_WIN_NAME = "Piano Finger Tracking \u2013 Record"


def _get_work_area():
    """Return (width, height) of the usable screen area (taskbar excluded)."""
    try:
        rect = ctypes.wintypes.RECT()
        ctypes.windll.user32.SystemParametersInfoW(48, 0, ctypes.byref(rect), 0)
        return rect.right - rect.left, rect.bottom - rect.top
    except Exception:
        return 1280, 720


def _create_window():
    """Create a resizable window sized to fit inside the taskbar-aware work area."""
    work_w, work_h = _get_work_area()
    # Maintain 16:9 aspect ratio within available space
    if work_w / work_h > 16 / 9:
        win_h = work_h
        win_w = int(work_h * 16 / 9)
    else:
        win_w = work_w
        win_h = int(work_w * 9 / 16)
    cv2.namedWindow(_WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(_WIN_NAME, win_w, win_h)
    cv2.moveWindow(_WIN_NAME, (work_w - win_w) // 2, 0)


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------

def _probe_cameras(max_index: int = 4) -> list:
    """Return list of camera indices that can be opened (probes 0..max_index)."""
    found = []
    for idx in range(max_index + 1):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            found.append(idx)
        cap.release()
    return found if found else [0]


def _open_cap(cam_idx: int) -> cv2.VideoCapture:
    """Open camera at cam_idx with standard config properties."""
    cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.CAP_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAP_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          config.CAP_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   config.CAP_BUFFERSIZE)
    return cap


# ---------------------------------------------------------------------------
# Setup screen
# ---------------------------------------------------------------------------

def _draw_input_box(frame, label, value, unit, y, active):
    """Draw a labelled text-input box onto frame."""
    x, w = 80, 420
    color = (0, 200, 255) if active else (130, 130, 130)

    cv2.putText(frame, label, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)

    box_y = y + 8
    cv2.rectangle(frame, (x, box_y), (x + w, box_y + 40), (40, 40, 40), -1)
    cv2.rectangle(frame, (x, box_y), (x + w, box_y + 40), color, 2)

    cursor = "|" if active else ""
    cv2.putText(frame, f"{value}{cursor}  {unit}", (x + 8, box_y + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def _run_setup_screen(cap, midi_ports: list, midi_port_idx: int,
                      avail_cams: list = None, cam_idx: int = 0):
    """
    Display setup overlay on the live camera feed.
    Collects lux level and hand size (cm) via keyboard.
    Shows detected MIDI port; [ and ] cycle through available ports.
    Shows camera selector; , and . cycle through available cameras.

    TAB: switch field   ENTER: confirm   BACKSPACE: delete
    [/]: cycle MIDI port   ,/.: cycle camera   ESC: quit

    Returns (lux, hand_size_cm, midi_port_idx, cam_idx, cap)
    or (None, None, 0, cam_idx, cap) on ESC.
    """
    if avail_cams is None:
        avail_cams = [cam_idx]
    fields = [
        {"label": "Lux Level", "unit": "lux", "value": ""},
        {"label": "Hand Size", "unit": "cm",  "value": ""},
    ]
    active    = 0
    error_msg = ""

    lux_colors = {
        "Dim":    (100, 100, 255),
        "Indoor": (100, 220, 100),
        "Bright": (0,   255, 255),
    }

    # MIDI port display colours
    MIDI_COLOR_OK   = (80, 220, 80)
    MIDI_COLOR_NONE = (60, 60, 200)
    flip_y = False
    _midi_mon  = None
    _last_note = "---"

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if flip_y:
            frame = cv2.flip(frame, -1)

        if midi_ports:
            if _midi_mon is None:
                try:
                    _midi_mon = mido.open_input(midi_ports[midi_port_idx])
                except Exception:
                    pass
            if _midi_mon is not None:
                for _msg in _midi_mon.iter_pending():
                    if _msg.type == "note_on" and _msg.velocity > 0:
                        _last_note = _midi_note_name(_msg.note) + "  (MIDI " + str(_msg.note) + ")"

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        cv2.putText(frame, "SESSION SETUP", (80, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, config.TEXT_COLOR_SETUP, 3)
        cv2.putText(frame, "Fill in both fields, then press ENTER", (80, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1)

        _draw_input_box(frame, fields[0]["label"], fields[0]["value"],
                        fields[0]["unit"], 160, active == 0)

        if fields[0]["value"]:
            try:
                label_str = lux_to_label(float(fields[0]["value"]))
                col = lux_colors.get(label_str, (200, 200, 200))
                cv2.putText(frame, f"  -> {label_str}", (90, 232),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, 2)
            except ValueError:
                pass

        _draw_input_box(frame, fields[1]["label"], fields[1]["value"],
                        fields[1]["unit"], 280, active == 1)

        # MIDI port display (below hand-size box, not a text-entry field)
        midi_y = 410
        cv2.putText(frame, "MIDI Input Port", (80, midi_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)
        midi_box_y = midi_y + 8
        if midi_ports:
            port_label = midi_ports[midi_port_idx]
            midi_col   = MIDI_COLOR_OK
            cycle_hint = f"  [ ] to cycle  ({midi_port_idx + 1}/{len(midi_ports)})"
        else:
            port_label = "No MIDI device found"
            midi_col   = MIDI_COLOR_NONE
            cycle_hint = "  (connect piano via USB and restart)"
        cv2.rectangle(frame, (80, midi_box_y), (500, midi_box_y + 40), (40, 40, 40), -1)
        cv2.rectangle(frame, (80, midi_box_y), (500, midi_box_y + 40), midi_col, 2)
        cv2.putText(frame, port_label, (88, midi_box_y + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, midi_col, 2)
        cv2.putText(frame, cycle_hint, (80, midi_box_y + 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (130, 130, 130), 1)
        cv2.putText(frame, "Last note: " + _last_note, (80, midi_box_y + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 220), 2)

        # Camera selector (below MIDI section)
        cam_y = 510
        cv2.putText(frame, "Camera", (80, cam_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)
        cam_box_y = cam_y + 8
        cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cam_label = f"Camera {cam_idx}  ({cam_w} x {cam_h})"
        cam_hint2 = (f"  c/C to cycle  ({avail_cams.index(cam_idx) + 1}/{len(avail_cams)})"
                     if len(avail_cams) > 1 else "  (only camera detected)")
        cv2.rectangle(frame, (80, cam_box_y), (500, cam_box_y + 40), (40, 40, 40), -1)
        cv2.rectangle(frame, (80, cam_box_y), (500, cam_box_y + 40), (80, 220, 80), 2)
        cv2.putText(frame, cam_label, (88, cam_box_y + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 220, 80), 2)
        cv2.putText(frame, cam_hint2, (80, cam_box_y + 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (130, 130, 130), 1)
        flip_txt = "  f: flip  [" + ("ON" if flip_y else "off") + "]"
        cv2.putText(frame, flip_txt, (80, cam_box_y + 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 220, 80) if flip_y else (130, 130, 130), 1)

        cv2.putText(frame,
                    "TAB: switch field    ENTER: confirm    [/]: MIDI port    c/C: camera    f: flip    ESC: quit",
                    (80, frame.shape[0] - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140, 140, 140), 1)
        if error_msg:
            cv2.putText(frame, error_msg, (80, frame.shape[0] - 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 80, 255), 2)

        cv2.imshow(_WIN_NAME, frame)

        key = cv2.waitKey(30) & 0xFF

        if key == config.EXIT_KEY:
            if _midi_mon is not None:
                try:
                    _midi_mon.close()
                except Exception:
                    pass
                _midi_mon = None
            return None, None, 0, cam_idx, cap, False

        elif key == ord("[") and midi_ports:
            midi_port_idx = (midi_port_idx - 1) % len(midi_ports)
            _midi_mon = None

        elif key == ord("]") and midi_ports:
            midi_port_idx = (midi_port_idx + 1) % len(midi_ports)
            _midi_mon = None

        elif key in (ord("c"), ord("C")) and len(avail_cams) > 1:
            delta   = -1 if key == ord("C") else 1
            new_idx = avail_cams[(avail_cams.index(cam_idx) + delta) % len(avail_cams)]
            new_cap = _open_cap(new_idx)
            if new_cap.isOpened():
                cap.release()
                cap     = new_cap
                cam_idx = new_idx
            else:
                new_cap.release()

        elif key == 9:                          # Tab
            active = (active + 1) % len(fields)
            error_msg = ""

        elif key in (13, 10):                   # Enter
            try:
                lux_val  = float(fields[0]["value"])
                hand_val = float(fields[1]["value"])
            except ValueError:
                error_msg = "Please fill both fields with valid numbers"
                continue

            if lux_val < 0:
                error_msg = "Lux must be 0 or greater"
                continue
            if not (config.HAND_SIZE_MIN_CM <= hand_val <= config.HAND_SIZE_MAX_CM):
                error_msg = (f"Hand size must be "
                             f"{config.HAND_SIZE_MIN_CM}–{config.HAND_SIZE_MAX_CM} cm")
                continue

        if _midi_mon is not None:
            try:
                _midi_mon.close()
            except Exception:
                pass
            _midi_mon = None
            return lux_val, hand_val, midi_port_idx, cam_idx, cap, flip_y

        elif key in (ord("f"), ord("F")):
            flip_y = not flip_y

        elif key in (8, 127):                   # Backspace
            fields[active]["value"] = fields[active]["value"][:-1]
            error_msg = ""

        elif 0 <= key <= 127 and chr(key) in "0123456789.":
            if key == ord(".") and "." in fields[active]["value"]:
                continue
            fields[active]["value"] += chr(key)
            error_msg = ""

    if _midi_mon is not None:
        try:
            _midi_mon.close()
        except Exception:
            pass
        _midi_mon = None
    return None, None, 0, cam_idx, cap, False


# ---------------------------------------------------------------------------
# Recording preview
# ---------------------------------------------------------------------------

def run_record():
    mp_hands = mp.solutions.hands

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=config.MAX_NUM_HANDS,
        model_complexity=config.MODEL_COMPLEXITY,
        min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
    )

    # --- Probe available cameras BEFORE opening the main cap ---
    # (probing after init_camera would try to open camera 0 twice on DirectShow,
    #  corrupting the stream and causing cap.read() to fail immediately)
    avail_cams = _probe_cameras()
    cam_idx    = config.CAMERA_INDEX
    if cam_idx not in avail_cams:
        avail_cams.insert(0, cam_idx)

    cap = init_camera(config)
    _create_window()

    # --- Detect MIDI ports before setup ---
    midi_ports    = list_midi_input_ports()
    midi_port_idx = 0
    if midi_ports:
        print(f"MIDI ports found: {midi_ports}")
    else:
        print("WARNING: No MIDI input ports detected. Recording will proceed without MIDI.")

    # --- Setup screen ---
    print("Waiting for session setup...")
    lux_value, hand_size_cm, midi_port_idx, cam_idx, cap, flip_y = _run_setup_screen(
        cap, midi_ports, midi_port_idx, avail_cams, cam_idx
    )
    hand_size_label = _hand_size_label(hand_size_cm) if hand_size_cm else "Unknown"

    if lux_value is None:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        return

    # Frame dimensions (may differ if user switched camera during setup)
    _fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    _fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    lux_label_str    = lux_to_label(lux_value)
    hand_size_label  = _hand_size_label(hand_size_cm)
    pid              = _next_participant_id()
    midi_port_name = midi_ports[midi_port_idx] if midi_ports else None
    print(f"Lux: {lux_value:.0f} lux  ->  {lux_label_str}  |  Hand size: {hand_size_cm} cm")
    print(f"Participant ID: p{pid:03d}")
    print(f"MIDI port: {midi_port_name or 'none'}")
    print("SPACE: rec/stop   S: stats   R: retest fitz   ESC: quit\n")

    # --- Session state ---
    fitz_detected   = False
    fitz_type       = 0
    fitz_label      = ""
    fitz_retesting  = False

    recording   = False
    show_stats  = False
    writer      = None
    out_dir     = None
    video_path  = None
    rec_start   = None
    rec_fps     = float(config.CAP_FPS)   # updated to actual FPS when recording starts

    # MIDI / frame-sync state
    midi_rec     = None   # MidiRecorder instance (active during recording)
    frame_logger = None   # FrameLogger instance  (active during recording)
    rec_t0       = 0.0    # perf_counter() at recording start (shared clock origin)

    stats = StatsCollector(config.WARMUP_FRAMES, config.STAT_FRAMES)

    # Load key calibration mask (or create default)
    _cal_dir  = Path(__file__).parent.parent / "data" / "calibration"
    _cal_file = _cal_dir / "key_centers.json"
    if _cal_file.exists():
        try:
            mask = KeyMask.load(_cal_file)
            print(f"Key calibration loaded: {mask.num_white} white keys  (M to toggle, drag to adjust)")
        except Exception as _e:
            print(f"Note: key calibration not loaded ({_e}), using default")
            mask = KeyMask.default(_fw, _fh)
    else:
        mask = KeyMask.default(_fw, _fh)
        print("No calibration file found, using default layout  (M to toggle, drag to adjust)")
    show_mask = True

    # Mouse callback: drag to resize/move, auto-save when drag ends
    def _on_mouse(event, x, y, flags, _):
        if mask.on_mouse(event, x, y, flags):
            mask.save(_cal_dir, _fw, _fh)

    cv2.setMouseCallback(_WIN_NAME, _on_mouse)

    panel = MaskControlPanel(mask, save_fn=lambda: mask.save(_cal_dir, _fw, _fh))

    lux_colors = {
        "Dim":    (100, 100, 255),
        "Indoor": (100, 220, 100),
        "Bright": (0,   255, 255),
    }

    while cap.isOpened():
        loop_start = time.perf_counter()

        success, frame = cap.read()
        if not success:
            print("Frame capture failed")
            break
        if flip_y:
            frame = cv2.flip(frame, -1)

        # Run MediaPipe at full 1080p — no resize for recording quality
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False

        t0      = time.perf_counter()
        results = hands.process(rgb)
        t1      = time.perf_counter()
        inference_ms = (t1 - t0) * 1000

        # --- One-shot Fitzpatrick detection (also re-runs after R pressed) ---
        if not fitz_detected and results.multi_hand_landmarks:
            fitz_type, fitz_label = detect_skin_type(
                frame,
                results.multi_hand_landmarks[0],
                frame.shape[1],
                frame.shape[0],
            )
            if fitz_type > 0:
                fitz_detected  = True
                fitz_retesting = False
                print(f"Fitzpatrick: Type {fitz_type} ({fitz_label})")

        total_latency_ms = (time.perf_counter() - loop_start) * 1000
        stats.update(loop_start, inference_ms, total_latency_ms)

        # --- Save raw frame (no overlays) to video file ---
        if recording and writer is not None:
            writer.write(frame)
            if frame_logger is not None:
                frame_logger.log()

        # --- All overlays go on a display copy, keeping the saved frame clean ---
        display = frame.copy()

        # Key calibration mask (drawn first so all other overlays sit on top)
        if show_mask:
            mask.draw(display)
            if not recording:
                draw_mask_handles(display, mask)

        # Stats overlay (top-left)
        if show_stats:
            ys = 35
            if not stats.warmup_done:
                cv2.putText(display, f"Stats: warming up ({stats.frame_count}/{config.WARMUP_FRAMES})",
                            (10, ys), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (160, 160, 160), 1)
            else:
                avg_fps = np.mean(list(stats.fps_history)[-30:])
                cv2.putText(display, f"FPS: {avg_fps:4.1f}", (10, ys),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, config.TEXT_COLOR_FPS, 2)
                ys += 32
                cv2.putText(display, f"Inf: {inference_ms:4.1f} ms", (10, ys),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.TEXT_COLOR_INF, 2)
                ys += 28
                cv2.putText(display, f"Lat: {total_latency_ms:4.1f} ms", (10, ys),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.TEXT_COLOR_LAT, 2)

        # Session info (top-right)
        fw = display.shape[1]
        yr = 35
        lux_col = lux_colors.get(lux_label_str, config.TEXT_COLOR_SESSION)
        cv2.putText(display, f"Lux: {lux_value:.0f}  ({lux_label_str})",
                    (fw - 340, yr), cv2.FONT_HERSHEY_SIMPLEX, 0.65, lux_col, 2)
        yr += 30
        cv2.putText(display, f"Hand: {hand_size_cm:.1f} cm  ({hand_size_label})",
                    (fw - 340, yr), cv2.FONT_HERSHEY_SIMPLEX, 0.65, config.TEXT_COLOR_SESSION, 2)
        yr += 30
        fitz_text = (f"Fitz: Type {fitz_type}  {fitz_label}"
                     if fitz_detected else "Fitz: show hand to detect...")
        cv2.putText(display, fitz_text,
                    (fw - 340, yr), cv2.FONT_HERSHEY_SIMPLEX, 0.65, config.TEXT_COLOR_FITZPATRICK, 2)

        # REC indicator (top-left, below stats if visible)
        rec_y = 100 if show_stats else 35
        if recording:
            cv2.circle(display, (28, rec_y - 7), 12, (0, 0, 220), -1)
            cv2.putText(display, f"REC  p{pid:03d}", (46, rec_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 220), 2)

        # Fitzpatrick retest overlay
        if fitz_retesting:
            fh, fw2 = display.shape[:2]
            bx, bw, bh = fw2 // 2 - 280, 560, 90
            by = fh // 2 - 45
            cv2.rectangle(display, (bx, by), (bx + bw, by + bh), (30, 30, 30), -1)
            cv2.rectangle(display, (bx, by), (bx + bw, by + bh), config.TEXT_COLOR_FITZPATRICK, 2)
            cv2.putText(display, "FITZPATRICK RETEST",
                        (bx + 20, by + 32), cv2.FONT_HERSHEY_SIMPLEX,
                        0.85, config.TEXT_COLOR_FITZPATRICK, 2)
            cv2.putText(display, "Place your hand in view",
                        (bx + 20, by + 68), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (200, 200, 200), 1)

        # Key hint bar (bottom)
        hint_y    = display.shape[0] - 18
        st_state   = "ON" if show_stats else "OFF"
        rec_state  = "STOP" if recording else "REC"
        mask_part  = f"  M:mask({'ON' if show_mask else 'OFF'})  N:ctrl({'ON' if panel.visible else 'OFF'})  V:reset-warp"
        cv2.putText(display,
                    f"SPACE:{rec_state}  S:stats({st_state})  R:fitz{mask_part}  ESC:quit",
                    (10, hint_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 120), 1)

        cv2.imshow(_WIN_NAME, display)
        panel.render()

        key = cv2.waitKey(1) & 0xFF

        if key == config.EXIT_KEY:
            break
        elif key == ord(" "):
            if not recording:
                # Measure actual FPS from recent loop timing before opening writer
                if stats.warmup_done and stats.fps_history:
                    rec_fps = float(np.mean(list(stats.fps_history)[-30:]))
                    rec_fps = max(1.0, min(rec_fps, float(config.CAP_FPS)))
                else:
                    rec_fps = float(config.CAP_FPS)
                # Start recording — establish shared clock origin first
                rec_t0    = time.perf_counter()
                rec_start = datetime.now().isoformat(timespec="seconds")
                writer, out_dir, video_path = _start_recording(frame, pid, rec_fps)
                frame_logger = FrameLogger(rec_t0)
                if midi_port_name:
                    midi_rec = MidiRecorder(midi_port_name)
                    midi_rec.start(rec_t0)
                recording = True
                print(f"Recording STARTED  ({rec_start})  FPS={rec_fps:.1f}  ->  {video_path}")
                if midi_rec:
                    print(f"MIDI recording on: {midi_port_name}")
            else:
                # Stop recording
                recording = False
                rec_stop  = datetime.now().isoformat(timespec="seconds")

                # Stop MIDI listener before flushing files
                midi_jsonl = midi_mid = frames_csv = None
                if midi_rec is not None:
                    midi_rec.stop()
                    pid_str = f"p{pid:03d}"
                    midi_jsonl, midi_mid = midi_rec.save(out_dir, pid_str)
                    midi_rec = None
                if frame_logger is not None:
                    pid_str    = f"p{pid:03d}"
                    frames_csv = frame_logger.save(out_dir, pid_str)
                    frame_logger = None

                if writer is not None:
                    writer.release()
                    writer = None
                _save_session_metadata(
                    out_dir, pid, lux_value, lux_label_str,
                    hand_size_cm, hand_size_label, fitz_type, fitz_label,
                    video_path, rec_start, rec_stop,
                    midi_port=midi_port_name,
                    frames_csv=frames_csv,
                    midi_jsonl=midi_jsonl,
                    midi_mid=midi_mid,
                    video_fps=rec_fps,
                )
                print(f"Recording STOPPED  ({rec_stop})")
        elif key == ord("s") or key == ord("S"):
            show_stats = not show_stats
        elif key == ord("m") or key == ord("M"):
            show_mask = not show_mask
        elif key == ord("n") or key == ord("N"):
            panel.toggle()
        elif key == ord("r") or key == ord("R"):
            fitz_detected  = False
            fitz_type      = 0
            fitz_label     = ""
            fitz_retesting = True
            print("Fitzpatrick retest triggered — place hand in view")
        elif key == ord("v") or key == ord("V"):
            mask.reset_warp()
            print("Mask warp reset to rectangle.")

    # Ensure writer is closed if user exits mid-recording
    if writer is not None:
        rec_stop = datetime.now().isoformat(timespec="seconds")
        writer.release()

        midi_jsonl = midi_mid = frames_csv = None
        if midi_rec is not None:
            midi_rec.stop()
            if out_dir:
                pid_str = f"p{pid:03d}"
                midi_jsonl, midi_mid = midi_rec.save(out_dir, pid_str)
        if frame_logger is not None and out_dir:
            pid_str    = f"p{pid:03d}"
            frames_csv = frame_logger.save(out_dir, pid_str)

        if out_dir and video_path:
            _save_session_metadata(
                out_dir, pid, lux_value, lux_label_str,
                hand_size_cm, hand_size_label, fitz_type, fitz_label,
                video_path, rec_start, rec_stop,
                midi_port=midi_port_name,
                frames_csv=frames_csv,
                midi_jsonl=midi_jsonl,
                midi_mid=midi_mid,
                video_fps=rec_fps,
            )

    stats.print_final_stats(frame.shape[1], frame.shape[0], config.MODEL_COMPLEXITY)

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    run_record()
