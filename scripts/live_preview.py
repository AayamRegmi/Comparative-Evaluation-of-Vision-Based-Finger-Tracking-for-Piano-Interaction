"""test.py
Live MJMPE (Mean Joint-to-MIDI-Pitch Error) accuracy test.

Mirrors record.py but writes nothing to disk.  At the end, shows a results
overlay with the final MJMPE and note counts, then exits.

MJMPE definition
----------------
For every MIDI note_on (velocity > 0) while at least one hand is visible:
    error_i = min distance (pixels) from any fingertip landmark (4,8,12,16,20)
              to the calibrated centre of the pressed key.
MJMPE = mean(error_i)

Notes with no hand visible are counted as "missed" and excluded from the mean.

Key bindings
------------
  M   – toggle key mask overlay
  N   – toggle mask control panel
  S   – toggle stats overlay
  ESC – stop test and show results
"""

import os
from pathlib import Path

import cv2
import mediapipe as mp
import mido
import numpy as np
import time

try:
    from . import config
    from .camera_setup import init_camera
    from .key_calibration import KeyMask, draw_mask_handles, MaskControlPanel
    from .midi_recorder import list_midi_input_ports
    from .stats_collector import StatsCollector
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import config
    from camera_setup import init_camera
    from key_calibration import KeyMask, draw_mask_handles, MaskControlPanel
    from midi_recorder import list_midi_input_ports
    from stats_collector import StatsCollector

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_WIN_NAME   = "Piano Finger Tracking \u2013 MJMPE Test"
_FINGERTIPS = (4, 8, 12, 16, 20)   # MediaPipe fingertip landmark indices
_FLASH_FRAMES = 4                   # frames to keep the per-note visual feedback


# ---------------------------------------------------------------------------
# Camera helpers  (duplicated from record.py — standalone, no shared state)
# ---------------------------------------------------------------------------

_NOTE_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

def _midi_note_name(n):
    return _NOTE_NAMES[n % 12] + str(n // 12 - 1)


def _probe_cameras(max_index: int = 4) -> list:
    found = []
    for idx in range(max_index + 1):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            found.append(idx)
        cap.release()
    return found if found else [0]


def _open_cap(cam_idx: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.CAP_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAP_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          config.CAP_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   config.CAP_BUFFERSIZE)
    return cap


def _create_window():
    try:
        import ctypes, ctypes.wintypes
        rect = ctypes.wintypes.RECT()
        ctypes.windll.user32.SystemParametersInfoW(48, 0, ctypes.byref(rect), 0)
        work_w = rect.right  - rect.left
        work_h = rect.bottom - rect.top
    except Exception:
        work_w, work_h = 1280, 720

    if work_w / work_h > 16 / 9:
        win_h = work_h;  win_w = int(work_h * 16 / 9)
    else:
        win_w = work_w;  win_h = int(work_w * 9 / 16)

    cv2.namedWindow(_WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(_WIN_NAME, win_w, win_h)
    cv2.moveWindow(_WIN_NAME, (work_w - win_w) // 2, 0)


# ---------------------------------------------------------------------------
# Setup screen
# ---------------------------------------------------------------------------

def _run_test_setup(cap, midi_ports: list, midi_port_idx: int,
                    avail_cams: list, cam_idx: int):
    """
    Simplified setup: MIDI port + camera selection only.
    Returns (midi_port_idx, cam_idx, cap) or (None, cam_idx, cap) on ESC.
    """
    MIDI_OK   = (80, 220, 80)
    MIDI_NONE = (60, 60, 200)
    CAM_COL   = (80, 220, 80)
    flip_y    = False
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

        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)

        cv2.putText(frame, "MJMPE TEST  \u2013  Session Setup", (80, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, config.TEXT_COLOR_SETUP, 3)
        cv2.putText(frame, "Select MIDI port and camera, then press ENTER", (80, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1)

        # MIDI port box
        midi_y     = 180
        midi_box_y = midi_y + 8
        cv2.putText(frame, "MIDI Input Port", (80, midi_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)
        if midi_ports:
            port_label = midi_ports[midi_port_idx]
            midi_col   = MIDI_OK
            hint       = f"  [ / ] to cycle  ({midi_port_idx + 1}/{len(midi_ports)})"
        else:
            port_label = "No MIDI device found"
            midi_col   = MIDI_NONE
            hint       = "  (connect piano via USB and restart)"
        cv2.rectangle(frame, (80, midi_box_y), (500, midi_box_y + 40), (40, 40, 40), -1)
        cv2.rectangle(frame, (80, midi_box_y), (500, midi_box_y + 40), midi_col, 2)
        cv2.putText(frame, port_label, (88, midi_box_y + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, midi_col, 2)
        cv2.putText(frame, hint, (80, midi_box_y + 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (130, 130, 130), 1)
        cv2.putText(frame, "Last note: " + _last_note, (80, midi_box_y + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 220), 2)

        # Camera box
        cam_y     = 310
        cam_box_y = cam_y + 8
        cv2.putText(frame, "Camera", (80, cam_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)
        cam_w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cam_h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cam_label = f"Camera {cam_idx}  ({cam_w} x {cam_h})"
        cam_hint  = (f"  c/C to cycle  ({avail_cams.index(cam_idx) + 1}/{len(avail_cams)})"
                     if len(avail_cams) > 1 else "  (only camera detected)")
        cv2.rectangle(frame, (80, cam_box_y), (500, cam_box_y + 40), (40, 40, 40), -1)
        cv2.rectangle(frame, (80, cam_box_y), (500, cam_box_y + 40), CAM_COL, 2)
        cv2.putText(frame, cam_label, (88, cam_box_y + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, CAM_COL, 2)
        cv2.putText(frame, cam_hint, (80, cam_box_y + 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (130, 130, 130), 1)
        flip_txt = "  f: flip  [" + ("ON" if flip_y else "off") + "]"
        cv2.putText(frame, flip_txt, (80, cam_box_y + 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 220, 80) if flip_y else (130, 130, 130), 1)

        cv2.putText(frame,
                    "ENTER: start test    [/]: MIDI port    c/C: camera    f: flip    ESC: quit",
                    (80, frame.shape[0] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140, 140, 140), 1)

        cv2.imshow(_WIN_NAME, frame)
        key = cv2.waitKey(30) & 0xFF

        if key == config.EXIT_KEY:
            if _midi_mon is not None:
                try:
                    _midi_mon.close()
                except Exception:
                    pass
                _midi_mon = None
            return None, cam_idx, cap, False

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

        elif key in (ord("f"), ord("F")):
            flip_y = not flip_y

        elif key in (13, 10):       # Enter
            if _midi_mon is not None:
                try:
                    _midi_mon.close()
                except Exception:
                    pass
                _midi_mon = None
            return midi_port_idx, cam_idx, cap, flip_y

    if _midi_mon is not None:
        try:
            _midi_mon.close()
        except Exception:
            pass
        _midi_mon = None
    return None, cam_idx, cap, False


# ---------------------------------------------------------------------------
# MJMPE helpers
# ---------------------------------------------------------------------------

def _key_center(mask: KeyMask, midi_note: int):
    """Return (cx, cy) for midi_note, or None if not in calibration."""
    for k in mask.keys:
        if k['midi_note'] == midi_note:
            return k['center']
    return None


def _get_fingertips(multi_hand_lms, fw: int, fh: int) -> list:
    """
    Return (slot, x, y) for every fingertip across all visible hands.
    slot: 0=thumb  1=index  2=middle  3=ring  4=pinky
    """
    tips = []
    if not multi_hand_lms:
        return tips
    for hand_lms in multi_hand_lms:
        for slot, lm_idx in enumerate(_FINGERTIPS):
            lm = hand_lms.landmark[lm_idx]
            tips.append((slot, int(lm.x * fw), int(lm.y * fh)))
    return tips


def _draw_hand_overlay(display: np.ndarray, multi_hand_lms, fw: int, fh: int):
    """Draw skeleton + cyan fingertip markers for all detected hands."""
    if not multi_hand_lms:
        return
    mp_drawing = mp.solutions.drawing_utils
    mp_hands   = mp.solutions.hands
    draw_spec_lm   = mp_drawing.DrawingSpec(color=(200, 200, 200), thickness=1, circle_radius=2)
    draw_spec_conn = mp_drawing.DrawingSpec(color=(100, 100, 100), thickness=1)
    for hand_lms in multi_hand_lms:
        mp_drawing.draw_landmarks(display, hand_lms,
                                  mp_hands.HAND_CONNECTIONS,
                                  draw_spec_lm, draw_spec_conn)
        for lm_idx in _FINGERTIPS:
            lm = hand_lms.landmark[lm_idx]
            px, py = int(lm.x * fw), int(lm.y * fh)
            cv2.circle(display, (px, py), 7, (255, 255, 0), -1)   # cyan fill
            cv2.circle(display, (px, py), 7, (0,   0,   0), 1)    # black outline


# ---------------------------------------------------------------------------
# Results overlay
# ---------------------------------------------------------------------------

_FINGER_NAMES = ("Thumb", "Index", "Middle", "Ring", "Pinky")


def _show_results(last_frame: np.ndarray, errors: list, missed: int,
                  per_finger: dict, accurate: dict, detection_fail: dict, wkw: int):
    """Render results panel on last_frame and wait for any key."""
    display = last_frame.copy()
    fh, fw  = display.shape[:2]

    # Dark centred panel — wider for two-column per-finger layout
    pw, ph   = 700, 560
    px       = max(0, (fw - pw) // 2)
    py       = max(0, (fh - ph) // 2)
    panel    = display[py:py+ph, px:px+pw].copy()
    cv2.rectangle(panel, (0, 0), (pw, ph), (15, 15, 15), -1)
    cv2.rectangle(panel, (0, 0), (pw, ph), (80, 80, 80), 2)
    cv2.addWeighted(panel, 0.92, display[py:py+ph, px:px+pw], 0.08, 0,
                    display[py:py+ph, px:px+pw])
    display[py:py+ph, px:px+pw] = panel

    # Title
    cv2.putText(display, "MJMPE TEST RESULTS",
                (px + 170, py + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (220, 220, 220), 2)
    cv2.line(display, (px + 20, py + 60), (px + pw - 20, py + 60), (60, 60, 60), 1)

    def row(label, value, y, val_color=(200, 200, 200)):
        cv2.putText(display, label, (px + 30, py + y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (150, 150, 150), 1)
        cv2.putText(display, value, (px + 350, py + y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, val_color, 2)

    if errors:
        mjmpe    = np.mean(errors)
        min_e    = np.min(errors)
        max_e    = np.max(errors)
        matched  = len(errors)
        total_df = sum(detection_fail.values())
        all_ev   = matched + total_df + missed
        pct      = 100.0 * sum(accurate.values()) / matched
        det_rate = 100.0 * matched / all_ev if all_ev else 0.0
        half_key = wkw * config.ACCURACY_THRESHOLD_RATIO

        mjmpe_col = ((0, 200, 80) if mjmpe < 20 else
                     (0, 220, 220) if mjmpe < 40 else (60, 80, 255))
        acc_col   = ((0, 200, 80) if pct >= 80 else
                     (0, 220, 220) if pct >= 60 else (60, 80, 255))
        dr_col    = ((0, 200, 80) if det_rate >= 90 else
                     (0, 220, 220) if det_rate >= 70 else (60, 80, 255))

        row("MJMPE  (horizontal)",               f"{mjmpe:.1f} px",        80, mjmpe_col)
        row(f"Accuracy  (<{half_key:.0f} px)",   f"{pct:.1f} %",          112, acc_col)
        row("Detection rate",                    f"{det_rate:.1f} %",      144, dr_col)
        row("Notes matched",                     str(matched),             176)
        row("Detection fails  (no tip on key)",  str(total_df),            208, (200, 0, 200))
        row("Notes missed  (no hands)",          str(missed),              240)
        row("Min / Max error",  f"{min_e:.1f} px  /  {max_e:.1f} px",     272)

        # Divider + two-column per-finger section
        cv2.line(display, (px + 20, py + 292), (px + pw - 20, py + 292), (55, 55, 55), 1)
        cv2.putText(display, "Per-finger breakdown (L = left half of frame, R = right):",
                    (px + 30, py + 312), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 160, 160), 1)

        # Column divider
        cv2.line(display, (px + 348, py + 298), (px + 348, py + ph - 30),
                 (55, 55, 55), 1)

        # Column headers
        lx = px + 20
        rx = px + 368
        cv2.putText(display, "Left Hand",
                    (lx + 90, py + 332), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (140, 170, 220), 1)
        cv2.putText(display, "Right Hand",
                    (rx + 78, py + 332), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (220, 170, 140), 1)

        for i, name in enumerate(_FINGER_NAMES):
            y_off = 356 + i * 34
            for side, col_x in [('L', lx), ('R', rx)]:
                errs = per_finger[side].get(i, [])
                cv2.putText(display, f"{name:<7}", (col_x, py + y_off),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.49, (150, 150, 150), 1)
                if errs:
                    f_mjmpe = np.mean(errs)
                    f_acc   = 100.0 * sum(1 for e in errs if e < half_key) / len(errs)
                    f_col   = ((0, 200, 80) if f_mjmpe < 20 else
                               (0, 220, 220) if f_mjmpe < 40 else (60, 80, 255))
                    cv2.putText(display,
                                f"{f_mjmpe:5.1f}px {f_acc:4.1f}% ({len(errs)})",
                                (col_x + 72, py + y_off),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.46, f_col, 1)
                else:
                    cv2.putText(display, "--",
                                (col_x + 72, py + y_off),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (80, 80, 80), 1)
    else:
        cv2.putText(display, "No notes detected",
                    (px + 170, py + 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (60, 80, 255), 2)
        cv2.putText(display, "Check MIDI connection and retry",
                    (px + 150, py + 260),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (130, 130, 130), 1)

    cv2.putText(display, "Press any key to close",
                (px + 220, py + ph - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 100, 100), 1)

    cv2.imshow(_WIN_NAME, display)
    cv2.waitKey(0)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_test():
    mp_hands = mp.solutions.hands
    hands    = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=config.MAX_NUM_HANDS,
        model_complexity=config.MODEL_COMPLEXITY,
        min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
    )

    # Probe cameras BEFORE opening main cap (DirectShow can't hold two handles)
    avail_cams = _probe_cameras()
    cam_idx    = config.CAMERA_INDEX
    if cam_idx not in avail_cams:
        avail_cams.insert(0, cam_idx)

    cap = init_camera(config)
    _create_window()

    midi_ports    = list_midi_input_ports()
    midi_port_idx = 0
    if midi_ports:
        print(f"MIDI ports: {midi_ports}")
    else:
        print("WARNING: No MIDI ports detected — MJMPE will not be computed.")

    # Setup screen
    midi_port_idx, cam_idx, cap, flip_y = _run_test_setup(cap, midi_ports, midi_port_idx, avail_cams, cam_idx)
    if midi_port_idx is None:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        return

    _fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    _fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    midi_port_name = midi_ports[midi_port_idx] if midi_ports else None
    print(f"MIDI port: {midi_port_name or 'none'}")
    print("Running MJMPE test — press piano keys.  ESC: show results and quit.\n")

    # Load key calibration mask
    _cal_dir  = Path(__file__).parent.parent / "data" / "calibration"
    _cal_file = _cal_dir / "key_centers.json"
    if _cal_file.exists():
        try:
            mask = KeyMask.load(_cal_file)
            print(f"Calibration loaded: {mask.num_white} white keys")
        except Exception as _e:
            print(f"Calibration load failed ({_e}), using default layout")
            mask = KeyMask.default(_fw, _fh)
    else:
        mask = KeyMask.default(_fw, _fh)
        print("No calibration file found — using default layout (accuracy will be lower)")
    show_mask = True

    # Mouse callback: allow live mask adjustment (auto-saves like record.py)
    def _on_mouse(event, x, y, flags, _):
        if mask.on_mouse(event, x, y, flags):
            mask.save(_cal_dir, _fw, _fh)
    cv2.setMouseCallback(_WIN_NAME, _on_mouse)

    panel = MaskControlPanel(mask, save_fn=lambda: mask.save(_cal_dir, _fw, _fh))

    # Open MIDI port directly (no background thread — we poll iter_pending each frame)
    midi_port = None
    if midi_port_name:
        try:
            midi_port = mido.open_input(midi_port_name)
        except Exception as e:
            print(f"Could not open MIDI port: {e}")

    stats = StatsCollector(config.WARMUP_FRAMES, config.STAT_FRAMES)

    # MJMPE accumulators
    errors         : list = []           # per-note horizontal errors (px) — x-axis only
    per_finger     : dict = {'L': {i: [] for i in range(5)},
                             'R': {i: [] for i in range(5)}}
    accurate       : dict = {'L': 0, 'R': 0}  # notes where h-error < wkw/2
    detection_fail : dict = {'L': 0, 'R': 0}  # hands visible but no tip on key polygon
    missed         : int  = 0            # note_on events with no hand visible
    show_stats = False

    # Per-frame flash state: list of (frames_remaining, key_cx, key_cy, tip_x, tip_y, dist)
    flashes: list = []

    latest_lms = None   # most recent multi_hand_landmarks
    last_frame = None   # for results screen

    while cap.isOpened():
        loop_start = time.perf_counter()

        success, frame = cap.read()
        if not success:
            print("Frame capture failed")
            break
        if flip_y:
            frame = cv2.flip(frame, -1)
        last_frame = frame

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        t0      = time.perf_counter()
        results = hands.process(rgb)
        t1      = time.perf_counter()
        inference_ms = (t1 - t0) * 1000

        latest_lms = results.multi_hand_landmarks
        fw, fh     = frame.shape[1], frame.shape[0]

        # --- Poll MIDI events and compute per-note MJMPE contribution ---
        if midi_port is not None:
            for msg in midi_port.iter_pending():
                if msg.is_realtime:
                    continue
                if msg.type == 'note_on' and msg.velocity > 0:
                    center = _key_center(mask, msg.note)
                    if center is None:
                        continue    # note not in calibration range

                    tips = _get_fingertips(latest_lms, fw, fh)
                    if not tips:
                        missed += 1
                        continue

                    cx, cy = center
                    hand   = 'L' if cx < fw / 2 else 'R'
                    # Identify playing finger: prefer fingertip inside key polygon.
                    # If no fingertip is over the key body the model has failed to
                    # place a landmark on the pressed key — count as detection_fail,
                    # do NOT add to MJMPE (would contaminate measurement with ghost
                    # landmarks or misidentified fingers).
                    key_poly = next(
                        (k['polygon'] for k in mask.keys if k['midi_note'] == msg.note),
                        None
                    )
                    if key_poly is not None:
                        inside_tips = [t for t in tips
                                       if cv2.pointPolygonTest(
                                           key_poly, (float(t[1]), float(t[2])), False
                                       ) >= 0]
                    else:
                        inside_tips = []

                    if inside_tips:
                        best_slot, best_tx, best_ty = min(
                            inside_tips, key=lambda t: abs(t[1] - cx)
                        )
                        # Horizontal-only error (removes depth/anatomy bias)
                        h_err = abs(best_tx - cx)
                        errors.append(h_err)
                        per_finger[hand][best_slot].append(h_err)
                        if h_err < mask.wkw * config.ACCURACY_THRESHOLD_RATIO:
                            accurate[hand] += 1
                        flashes.append([_FLASH_FRAMES, cx, cy,
                                        best_tx, best_ty, h_err, 'match'])
                    else:
                        # Detection fail: model missed the key polygon entirely
                        detection_fail[hand] += 1
                        flashes.append([_FLASH_FRAMES, cx, cy,
                                        cx, cy, 0, 'fail'])

        total_latency_ms = (time.perf_counter() - loop_start) * 1000
        stats.update(loop_start, inference_ms, total_latency_ms)

        # --- Build display frame ---
        display = frame.copy()

        if show_mask:
            mask.draw(display)
            draw_mask_handles(display, mask)

        _draw_hand_overlay(display, latest_lms, fw, fh)

        # Flash: key centre circle + error line
        next_flashes = []
        for flash in flashes:
            remaining, cx, cy, tx, ty, dist, kind = flash
            if kind == 'match':
                # Colour by distance: green < 20px, yellow < 40px, red ≥ 40px
                col = ((0, 200, 80) if dist < 20 else
                       (0, 220, 220) if dist < 40 else (60, 80, 255))
                cv2.circle(display, (cx, cy), 14, col, 3)
                cv2.line(display,   (tx, ty), (cx, cy), col, 2)
                cv2.circle(display, (tx, ty), 9, col, -1)
            else:
                # Detection fail: magenta outline — no landmark landed on key
                cv2.circle(display, (cx, cy), 16, (200, 0, 200), 3)
                cv2.line(display, (cx - 10, cy), (cx + 10, cy), (200, 0, 200), 2)
                cv2.line(display, (cx, cy - 10), (cx, cy + 10), (200, 0, 200), 2)
            remaining -= 1
            if remaining > 0:
                next_flashes.append([remaining, cx, cy, tx, ty, dist, kind])
        flashes = next_flashes

        # Stats overlay (top-left)
        if show_stats:
            ys = 35
            if not stats.warmup_done:
                cv2.putText(display,
                            f"Stats: warming up ({stats.frame_count}/{config.WARMUP_FRAMES})",
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

        # MJMPE HUD (top-right)
        yr         = 35
        xr         = fw - 400
        matched    = len(errors)
        total_df   = sum(detection_fail.values())
        all_events = matched + total_df + missed
        if errors:
            m         = np.mean(errors)
            pct_live  = 100.0 * sum(accurate.values()) / matched
            det_rate  = 100.0 * matched / all_events if all_events else 0.0
            mjmpe_col = ((0, 200, 80) if m < 20 else
                         (0, 220, 220) if m < 40 else (60, 80, 255))
            acc_col   = ((0, 200, 80) if pct_live >= 80 else
                         (0, 220, 220) if pct_live >= 60 else (60, 80, 255))
            dr_col    = ((0, 200, 80) if det_rate >= 90 else
                         (0, 220, 220) if det_rate >= 70 else (60, 80, 255))
            cv2.putText(display, f"MJMPE (x): {m:.1f} px",
                        (xr, yr), cv2.FONT_HERSHEY_SIMPLEX, 0.70, mjmpe_col, 2)
            yr += 28
            cv2.putText(display, f"Accuracy:  {pct_live:.1f} %",
                        (xr, yr), cv2.FONT_HERSHEY_SIMPLEX, 0.70, acc_col, 2)
            yr += 28
            cv2.putText(display, f"Det.Rate:  {det_rate:.1f} %",
                        (xr, yr), cv2.FONT_HERSHEY_SIMPLEX, 0.70, dr_col, 2)
        else:
            cv2.putText(display, "MJMPE (x): --",
                        (xr, yr), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (200, 200, 200), 2)
            yr += 28
            cv2.putText(display, "Accuracy:  --",
                        (xr, yr), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (200, 200, 200), 2)
            yr += 28
            cv2.putText(display, "Det.Rate:  --",
                        (xr, yr), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (200, 200, 200), 2)
        yr += 26
        cv2.putText(display, f"Notes: {matched} ok / {total_df} fail / {missed} missed",
                    (xr, yr), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1)

        # Hint bar
        hint_y   = display.shape[0] - 18
        mask_lbl = f"M:mask({'ON' if show_mask else 'OFF'})"
        ctrl_lbl = f"N:ctrl({'ON' if panel.visible else 'OFF'})"
        st_lbl   = f"S:stats({'ON' if show_stats else 'OFF'})"
        cv2.putText(display,
                    f"{mask_lbl}  {ctrl_lbl}  {st_lbl}  V:reset-warp  ESC:results+quit",
                    (10, hint_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 120), 1)

        cv2.imshow(_WIN_NAME, display)
        panel.render()

        key = cv2.waitKey(1) & 0xFF

        if key == config.EXIT_KEY:
            break
        elif key == ord("m") or key == ord("M"):
            show_mask = not show_mask
        elif key == ord("n") or key == ord("N"):
            panel.toggle()
        elif key == ord("s") or key == ord("S"):
            show_stats = not show_stats
        elif key == ord("v") or key == ord("V"):
            mask.reset_warp()

    # --- Results ---
    if midi_port is not None:
        try:
            midi_port.close()
        except Exception:
            pass

    print(f"\n--- MJMPE Results (horizontal error) ---")
    if errors:
        half_key = mask.wkw / 2
        matched  = len(errors)
        total_df = sum(detection_fail.values())
        all_ev   = matched + total_df + missed
        pct      = 100.0 * sum(accurate.values()) / matched
        det_rate = 100.0 * matched / all_ev if all_ev else 0.0
        print(f"  MJMPE (x):        {np.mean(errors):.2f} px")
        print(f"  Accuracy:         {pct:.1f} %  (error < {half_key:.0f} px = ½ key width)")
        print(f"  Detection rate:   {det_rate:.1f} %  (landmark on key / all events)")
        print(f"  Notes matched:    {matched}")
        print(f"  Detection fails:  {total_df}  (hands visible, no tip on key)")
        print(f"  Notes missed:     {missed}  (no hands detected)")
        print(f"  Min error:        {np.min(errors):.2f} px")
        print(f"  Max error:        {np.max(errors):.2f} px")
        print(f"  Per-finger (L = left-frame half, R = right):")
        for side, side_name in [('L', 'Left'), ('R', 'Right')]:
            print(f"    {side_name} hand:")
            for i, name in enumerate(_FINGER_NAMES):
                errs = per_finger[side].get(i, [])
                if errs:
                    f_acc = 100.0 * sum(1 for e in errs if e < half_key) / len(errs)
                    print(f"      {name:<7}  {np.mean(errs):5.2f} px  {f_acc:5.1f}%"
                          f"  ({len(errs)} notes)")
                else:
                    print(f"      {name:<7}  -- (0 notes)")
    else:
        print("  No notes matched (check MIDI connection)")
    print("----------------------------------------\n")

    if last_frame is not None:
        _show_results(last_frame, errors, missed, per_finger, accurate,
                      detection_fail, mask.wkw)

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    run_test()
