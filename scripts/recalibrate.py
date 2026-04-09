"""
recalibrate.py  –  Per-session key-mask calibration against recorded video.

Opens a participant picker showing all sessions in data/raw/.
Green dot = already has a session calibration.
Blue dot  = no calibration yet (needs one before analyse.py can run).

Click any participant to open the calibration editor for that session.
The editor plays back the recorded video so you can find a clear frame,
align the overlay to match the piano exactly, then save.  The calibration
is written as p###_key_centers.json inside the session folder and is
picked up automatically by analyse.py.

Usage:
    python -m scripts.recalibrate

Picker controls:
    Click row             open calibration editor for that session
    Scroll                scroll the participant list
    ESC                   quit

Calibration editor controls:
  Playback
    SPACE                 play / pause video
    LEFT / RIGHT  or A/D  step one frame back / forward
    ,  /  .               jump 10 frames back / forward

  Overlay alignment
    Drag interior         move keyboard overlay
    Drag left/right edge  resize key width
    Drag bottom edge      resize key height
    Drag corner           warp perspective (fine-tune perspective skew)
    Scroll                add / remove a key on the right end
    [ / ]                 shift starting MIDI note one white key down / up
    -  /  =               shrink / grow white key width by 1 px
    V                     reset warp back to a plain rectangle

  General
    H                     toggle controls bar (hide if overlay is in the way)
    ENTER  or  S          save calibration and return to picker
    ESC                   return to picker without saving

Workflow for existing sessions with missing calibrations:
    1. Run:  python -m scripts.recalibrate
    2. Click a blue-dot session in the picker
    3. Press SPACE to play — find a frame where the piano is clearly visible
       and hands are not blocking the keys (first few seconds usually best)
    4. Drag the overlay to align with the piano keys
    5. Press ENTER to save, then pick the next session
    6. Once all sessions are calibrated, re-run:
           python -m scripts.analyse --pid p013 p014 ...
"""

import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    from .key_calibration import (KeyMask, draw_mask_handles, _draw_hud,
                                   _CAL_DIR, _WHITE_SEMIS)
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from key_calibration import (KeyMask, draw_mask_handles, _draw_hud,
                                  _CAL_DIR, _WHITE_SEMIS)

_RAW_DIR   = Path(__file__).parent.parent / "data" / "raw"
_WIN_PICK  = "Recalibrate — Select Participant"
_WIN_CAL   = "Recalibrate — Align Overlay  (ENTER=save  ESC=back)"

# Picker layout constants
_ROW_H     = 38
_PAD       = 12
_PICK_W    = 540
_PICK_ROWS = 16   # visible rows before scrolling


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _session_dirs() -> list[Path]:
    if not _RAW_DIR.exists():
        return []
    return sorted(
        d for d in _RAW_DIR.iterdir()
        if d.is_dir() and d.name.startswith("p") and d.name[1:].isdigit()
    )


def _first_frame(video_path: Path) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def _has_session_cal(session_dir: Path, pid: str) -> bool:
    return (session_dir / f"{pid}_key_centers.json").exists()


_NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

def _midi_note_name(note: int) -> str:
    return _NOTE_NAMES[note % 12] + str(note // 12 - 1)


def _load_frame_times(frames_csv: Path, video_fps: float, total_frames: int) -> list[float]:
    """Return list of time_s indexed by frame_index."""
    if frames_csv.exists():
        times = {}
        with open(frames_csv, newline="") as f:
            for row in csv.DictReader(f):
                times[int(row["frame_index"])] = float(row["time_s"])
        return [times.get(i, i / video_fps) for i in range(total_frames)]
    return [i / video_fps for i in range(total_frames)]


def _load_note_events(midi_jsonl: Path) -> list[tuple[float, int, bool]]:
    """Return list of (time_s, midi_note, is_on) sorted by time."""
    events = []
    if not midi_jsonl.exists():
        return events
    with open(midi_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
                if ev.get("type") == "note_on":
                    is_on = ev.get("velocity", 0) > 0
                    events.append((ev["time_s"], ev["note"], is_on))
                elif ev.get("type") == "note_off":
                    events.append((ev["time_s"], ev["note"], False))
            except Exception:
                pass
    return sorted(events, key=lambda e: e[0])


def _active_notes_at(frame_time: float, note_events: list) -> list[int]:
    """Return list of MIDI notes currently held at frame_time."""
    held: dict[int, bool] = {}
    for t, note, is_on in note_events:
        if t > frame_time:
            break
        held[note] = is_on
    return [n for n, on in held.items() if on]


# ---------------------------------------------------------------------------
# Participant picker
# ---------------------------------------------------------------------------

def _run_picker() -> str | None:
    """
    Show a scrollable list of all sessions.
    Returns the chosen PID string (e.g. 'p013'), or None if the user quits.
    """
    sessions = _session_dirs()
    if not sessions:
        print("No sessions found in data/raw/")
        return None

    scroll   = 0        # index of top visible row
    hovered  = -1
    selected = None

    pick_h = _PAD * 2 + _ROW_H * min(len(sessions), _PICK_ROWS) + 60

    def _render():
        canvas = np.full((_pick_h, _PICK_W, 3), 30, dtype=np.uint8)

        # Title
        cv2.putText(canvas, "Select participant to calibrate",
                    (_PAD, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1)
        cv2.line(canvas, (_PAD, 36), (_PICK_W - _PAD, 36), (80, 80, 80), 1)

        visible = sessions[scroll: scroll + _PICK_ROWS]
        for i, sd in enumerate(visible):
            pid      = sd.name
            cal_done = _has_session_cal(sd, pid)
            gy       = 44 + i * _ROW_H

            row_idx  = scroll + i
            bg_col   = (55, 55, 55) if row_idx == hovered else (38, 38, 38)
            cv2.rectangle(canvas, (_PAD, gy), (_PICK_W - _PAD, gy + _ROW_H - 2),
                          bg_col, -1)

            # Status dot
            dot_col = (50, 200, 80) if cal_done else (60, 130, 220)
            cv2.circle(canvas, (_PAD + 14, gy + _ROW_H // 2), 6, dot_col, -1)

            # PID label
            cv2.putText(canvas, pid.upper(), (_PAD + 28, gy + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (230, 230, 230), 1)

            # Status text
            status = "calibrated" if cal_done else "needs calibration"
            s_col  = (80, 180, 80) if cal_done else (100, 160, 230)
            cv2.putText(canvas, status, (_PAD + 110, gy + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, s_col, 1)

        # Scroll hint
        if len(sessions) > _PICK_ROWS:
            shown_end = min(scroll + _PICK_ROWS, len(sessions))
            hint = f"Showing {scroll + 1}–{shown_end} of {len(sessions)}  (scroll to see more)"
            cv2.putText(canvas, hint,
                        (_PAD, pick_h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, (120, 120, 120), 1)

        cv2.putText(canvas, "ESC: quit",
                    (_PICK_W - 90, pick_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 120), 1)
        return canvas

    # Use a mutable container so the closure can update it
    _pick_h = pick_h

    def on_mouse(event, x, y, flags, _):
        nonlocal hovered, selected, scroll
        # Which row is under cursor?
        row_top = 44
        rel     = y - row_top
        if 0 <= rel < _ROW_H * _PICK_ROWS and _PAD <= x <= _PICK_W - _PAD:
            idx = scroll + rel // _ROW_H
            if idx < len(sessions):
                hovered = idx
                if event == cv2.EVENT_LBUTTONUP:
                    selected = sessions[idx].name
            else:
                hovered = -1
        else:
            hovered = -1

        if event == cv2.EVENT_MOUSEWHEEL:
            delta = -1 if flags > 0 else 1
            scroll = max(0, min(len(sessions) - _PICK_ROWS, scroll + delta))

    cv2.namedWindow(_WIN_PICK, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(_WIN_PICK, _PICK_W, pick_h)
    cv2.setMouseCallback(_WIN_PICK, on_mouse)

    while True:
        cv2.imshow(_WIN_PICK, _render())
        key = cv2.waitKey(30) & 0xFF

        if key == 27:
            cv2.destroyWindow(_WIN_PICK)
            return None

        if selected:
            cv2.destroyWindow(_WIN_PICK)
            return selected


# ---------------------------------------------------------------------------
# HUD drawing with dark background bars (always readable over any video frame)
# ---------------------------------------------------------------------------

_BAR_H = 32   # height of top/bottom dark bars

def _draw_text(img, text, x, y, scale=0.50, color=(220, 220, 220), thickness=1):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def _draw_editor_hud(canvas, mask, total_frames, cur_frame, playing, fw, fh,
                     show_help, active_notes=None):
    # Dark top bar — status line
    cv2.rectangle(canvas, (0, 0), (fw, _BAR_H), (18, 18, 18), -1)
    status = (f"Keys: {mask.num_white} white  |  "
              f"Key: {mask.wkw}x{mask.wkh}px  |  "
              f"Pos: ({mask.ox},{mask.oy})  |  "
              f"Frame {cur_frame}/{total_frames - 1}  "
              f"({'PLAY' if playing else 'PAUSE'})")
    _draw_text(canvas, status, 8, 21, scale=0.48, color=(200, 200, 200))

    # Active note indicator — shown on right side of top bar
    if active_notes:
        note_str = "  ".join(_midi_note_name(n) for n in sorted(active_notes))
        label    = f"NOW: {note_str}"
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        _draw_text(canvas, label, fw - tw - 12, 21, scale=0.55, color=(80, 220, 255))
    else:
        _draw_text(canvas, "—", fw - 30, 21, scale=0.50, color=(100, 100, 100))

    # Dark bottom bar — controls
    bar_y = fh - _BAR_H
    cv2.rectangle(canvas, (0, bar_y), (fw, fh), (18, 18, 18), -1)

    if show_help:
        controls = ("SPACE:play/pause   LEFT/RIGHT:step   ,:‑10   .:+10   "
                    "[ / ]:note   -/=:width   V:reset warp   "
                    "ENTER/S:save   ESC:back   H:hide help")
    else:
        controls = "H: show controls"
    _draw_text(canvas, controls, 8, bar_y + 21, scale=0.44, color=(170, 170, 170))

    # Progress bar just above the bottom bar
    if total_frames > 1:
        filled = int(fw * cur_frame / (total_frames - 1))
        cv2.rectangle(canvas, (0, bar_y - 5), (fw, bar_y - 1), (50, 50, 50), -1)
        cv2.rectangle(canvas, (0, bar_y - 5), (filled, bar_y - 1), (0, 200, 80), -1)


# ---------------------------------------------------------------------------
# Calibration editor — video playback + overlay alignment
# ---------------------------------------------------------------------------

def _run_editor(pid: str) -> bool:
    """
    Open the session video for pid, allow playback and overlay alignment,
    save p###_key_centers.json to the session folder on ENTER/S.
    Returns True if calibration was saved.
    """
    session_dir = _RAW_DIR / pid
    video_path  = session_dir / f"{pid}.mp4"
    out_cal     = session_dir / f"{pid}_key_centers.json"

    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}")
        return False

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: Cannot open {video_path}")
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fw           = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh           = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Canvas height = video + top bar + bottom bar + progress strip
    canvas_h = fh + _BAR_H * 2 + 5

    # Load MIDI + frame timing for note indicator
    frame_times  = _load_frame_times(session_dir / f"{pid}_frames.csv", fps, total_frames)
    note_events  = _load_note_events(session_dir / f"{pid}_midi.jsonl")
    if note_events:
        print(f"  Loaded {len(note_events)} MIDI events for note indicator.")
    else:
        print("  No MIDI data found — note indicator unavailable.")

    # Load calibration: session-local → global → default
    if out_cal.exists():
        try:
            mask = KeyMask.load(out_cal)
            print(f"Loaded existing session calibration for {pid}.")
        except Exception:
            mask = KeyMask.default(fw, fh)
    elif (_CAL_DIR / "key_centers.json").exists():
        try:
            mask = KeyMask.load(_CAL_DIR / "key_centers.json")
            print("Using global calibration as starting point.")
        except Exception:
            mask = KeyMask.default(fw, fh)
    else:
        mask = KeyMask.default(fw, fh)

    cur_frame  = 0
    playing    = False
    show_help  = True
    saved      = False
    frame_cache: dict[int, np.ndarray] = {}

    def _read(idx: int) -> np.ndarray | None:
        if idx in frame_cache:
            return frame_cache[idx].copy()
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, f = cap.read()
        if not ok:
            return None
        if len(frame_cache) > 60:
            del frame_cache[min(frame_cache)]
        frame_cache[idx] = f.copy()
        return f

    # Mouse coords arrive in canvas space; video sits at y-offset _BAR_H
    def on_mouse(event, x, y, flags, _):
        mask.on_mouse(event, x, y - _BAR_H, flags)

    cv2.namedWindow(_WIN_CAL, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(_WIN_CAL, min(fw, 1400), min(canvas_h, 900))
    cv2.setMouseCallback(_WIN_CAL, on_mouse)

    while True:
        raw = _read(cur_frame)
        if raw is None:
            break

        # Draw overlay on video frame
        mask.draw(raw)
        draw_mask_handles(raw, mask)

        # Active notes at this frame
        frame_t      = frame_times[cur_frame] if cur_frame < len(frame_times) else cur_frame / fps
        active_notes = _active_notes_at(frame_t, note_events)

        # Highlight active key centres on the video frame
        for note in active_notes:
            for k in mask.keys:
                if k["midi_note"] == note:
                    cx, cy = int(k["center"][0]), int(k["center"][1])
                    cv2.circle(raw, (cx, cy), 10, (0, 255, 255), 3)
                    cv2.circle(raw, (cx, cy),  4, (0, 255, 255), -1)

        # Compose canvas: top bar | video | progress + bottom bar
        canvas = np.zeros((canvas_h, fw, 3), dtype=np.uint8)
        canvas[_BAR_H: _BAR_H + fh] = raw

        _draw_editor_hud(canvas, mask, total_frames, cur_frame, playing, fw, canvas_h,
                         show_help, active_notes)

        cv2.imshow(_WIN_CAL, canvas)
        wait_ms = max(1, int(1000 / fps)) if playing else 20
        key = cv2.waitKey(wait_ms) & 0xFF

        if playing:
            nxt = cur_frame + 1
            cur_frame = nxt if nxt < total_frames else (playing := False) or cur_frame

        # Key handling
        if key == 27:       # ESC — back to picker
            break

        elif key in (13, 10, ord('s'), ord('S')):
            path = mask.save(session_dir, fw, fh,
                             filename=f"{pid}_key_centers.json")
            print(f"Saved → {path}")
            saved = True
            break

        elif key == ord(' '):
            playing = not playing

        elif key in (81, 2, ord('a')):      # LEFT / A
            cur_frame = max(0, cur_frame - 1)
            playing = False

        elif key in (83, 3, ord('d')):      # RIGHT / D
            cur_frame = min(total_frames - 1, cur_frame + 1)
            playing = False

        elif key == ord(','):
            cur_frame = max(0, cur_frame - 10)
            playing = False

        elif key == ord('.'):
            cur_frame = min(total_frames - 1, cur_frame + 10)
            playing = False

        elif key in (ord('h'), ord('H')):
            show_help = not show_help

        elif key == ord('['):
            m = mask.start_midi - 1
            while m >= 0 and m % 12 not in _WHITE_SEMIS:
                m -= 1
            if m >= 0:
                mask.start_midi = m
                mask.mark_dirty()

        elif key == ord(']'):
            m = mask.start_midi + 1
            while m <= 127 and m % 12 not in _WHITE_SEMIS:
                m += 1
            if m <= 127:
                mask.start_midi = m
                mask.mark_dirty()

        elif key == ord('-'):
            mask.wkw = max(6, mask.wkw - 1)
            mask.reset_warp()

        elif key == ord('='):
            mask.wkw += 1
            mask.reset_warp()

        elif key in (ord('v'), ord('V')):
            mask.reset_warp()

    cap.release()
    cv2.destroyWindow(_WIN_CAL)
    return saved


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    while True:
        pid = _run_picker()
        if pid is None:
            break
        _run_editor(pid)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
