"""analyse.py
Offline per-session MJMPE analysis.

Loads a recorded participant session, runs a hand-landmark model over each
video frame that corresponds to a MIDI note_on event, and writes a JSON
results file.  Supports MediaPipe Hands and OpenPose Hand.

CLI usage
---------
    python scripts/analyse.py data/raw/p001/ --model mediapipe
    python scripts/analyse.py data/raw/p001/ --model openpose
    python scripts/analyse.py data/raw/p001/          # both models

GUI usage
---------
    python scripts/analyse.py --ui

Required session files (inside <session_dir>):
    p###.mp4              – recorded video (no overlays)
    p###_midi.jsonl       – MIDI note events with time_s timestamps
    p###_frames.csv       – frame_index,time_s for every written video frame
    p###_session.json     – metadata (fitzpatrick_type, lux, hand_size_cm, …)

Calibration:
    data/calibration/key_centers.json   (must exist; run key_calibration.py first)

Output (alongside session files, or to --out):
    p###_mediapipe_results.json
    p###_openpose_results.json
"""

import argparse
import bisect
import csv
import json
import os
import queue as _queue
import sys
import threading
from pathlib import Path

import cv2
import numpy as np

try:
    from . import config
    from .key_calibration import KeyMask
    from .model_manager import OpenPoseHandModel
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import config
    from key_calibration import KeyMask
    from model_manager import OpenPoseHandModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROJECT_ROOT = str(Path(__file__).parent.parent)
_CAL_FILE     = Path(_PROJECT_ROOT) / "data" / "calibration" / "key_centers.json"
_RAW_DIR       = Path(_PROJECT_ROOT) / "data" / "raw"
_PROCESSED_DIR = Path(_PROJECT_ROOT) / "data" / "processed"

# Fingertip landmark indices — identical for MediaPipe and OpenPose Hand
_TIPS         = [4, 8, 12, 16, 20]   # Thumb, Index, Middle, Ring, Pinky
_FINGER_NAMES = ("Thumb", "Index", "Middle", "Ring", "Pinky")


# ---------------------------------------------------------------------------
# Session file discovery
# ---------------------------------------------------------------------------

def _find_pid(session_dir: Path) -> str:
    """Infer participant ID ('p001') from the MIDI JSONL filename."""
    for f in sorted(Path(session_dir).glob("p*_midi.jsonl")):
        return f.name.replace("_midi.jsonl", "")
    raise FileNotFoundError(f"No p###_midi.jsonl found in {session_dir}")


def _load_frames(frames_csv: Path) -> list:
    """Return list of (time_s: float, frame_index: int), sorted by time_s."""
    rows = []
    with open(frames_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append((float(row["time_s"]), int(row["frame_index"])))
    rows.sort(key=lambda r: r[0])
    return rows


def _load_midi_notes(midi_jsonl: Path) -> list:
    """Return list of note_on event dicts (velocity > 0)."""
    events = []
    with open(midi_jsonl) as fh:
        for line in fh:
            evt = json.loads(line)
            if evt.get("type") == "note_on" and evt.get("velocity", 0) > 0:
                events.append(evt)
    return events


def _load_session_meta(session_json: Path) -> dict:
    if Path(session_json).exists():
        with open(session_json) as fh:
            return json.load(fh)
    return {}


def _scan_sessions(raw_dir: Path) -> list:
    """
    Return list of session info dicts for every valid session in raw_dir.
    Each dict: {pid, dir, mediapipe: bool, openpose: bool}
    """
    sessions = []
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        return sessions
    for d in sorted(raw_dir.iterdir()):
        if not d.is_dir():
            continue
        midi_files = sorted(d.glob("*_midi.jsonl"))
        if not midi_files:
            continue
        pid = midi_files[0].stem.replace("_midi", "")
        sessions.append({
            "pid":       pid,
            "dir":       d,
            "mediapipe": (_PROCESSED_DIR / f"{pid}_mediapipe_results.json").exists(),
            "openpose":  (_PROCESSED_DIR / f"{pid}_openpose_results.json").exists(),
        })
    return sessions


# ---------------------------------------------------------------------------
# Frame alignment
# ---------------------------------------------------------------------------

def _nearest_frame_idx(event_time: float, frame_times: list) -> int:
    """Return the frame_index whose timestamp is closest to event_time."""
    ts_list = [t for t, _ in frame_times]
    i = bisect.bisect_left(ts_list, event_time)
    i = min(max(i, 0), len(ts_list) - 1)
    if i > 0 and abs(ts_list[i - 1] - event_time) < abs(ts_list[i] - event_time):
        i -= 1
    return frame_times[i][1]


# ---------------------------------------------------------------------------
# Fingertip extractors — common output: [(slot, x, y), ...]
#   slot: 0=Thumb  1=Index  2=Middle  3=Ring  4=Pinky
# ---------------------------------------------------------------------------

def _tips_mediapipe(result, fw: int, fh: int) -> list:
    if not result.multi_hand_landmarks:
        return []
    tips = []
    for hand_lms in result.multi_hand_landmarks:
        for slot, lm_idx in enumerate(_TIPS):
            lm = hand_lms.landmark[lm_idx]
            tips.append((slot, int(lm.x * fw), int(lm.y * fh)))
    return tips


def _tips_openpose(kps: list) -> list:
    tips = []
    for slot, idx in enumerate(_TIPS):
        if kps[idx] is not None:
            x, y, _conf = kps[idx]
            tips.append((slot, int(x), int(y)))
    return tips


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def _analyse_session(session_dir: Path, model_name: str, out_dir: Path,
                     progress_cb=None, cancel_ev: threading.Event = None) -> Path:
    """
    Run one model over a single recorded session and write a results JSON.

    Args:
        progress_cb(done: int, total: int)  — called every ~5 notes (optional)
        cancel_ev                           — stop gracefully if set (optional)

    Returns:
        Path of the output JSON, or None if cancelled.
    """
    session_dir = Path(session_dir)
    pid = _find_pid(session_dir)

    frames_csv   = session_dir / f"{pid}_frames.csv"
    midi_jsonl   = session_dir / f"{pid}_midi.jsonl"
    video_path   = session_dir / f"{pid}.mp4"
    session_json = session_dir / f"{pid}_session.json"

    for p in (frames_csv, midi_jsonl, video_path):
        if not p.exists():
            raise FileNotFoundError(f"Required session file not found: {p}")
    if not _CAL_FILE.exists():
        raise FileNotFoundError(
            f"Key calibration not found: {_CAL_FILE}\n"
            f"Run key_calibration.py first."
        )

    frame_times = _load_frames(frames_csv)
    note_events = _load_midi_notes(midi_jsonl)
    metadata    = _load_session_meta(session_json)
    mask        = KeyMask.load(_CAL_FILE)

    print(f"\n[{pid} | {model_name}]  {len(note_events)} notes  "
          f"|  {len(frame_times)} frames  |  {video_path.name}")

    # --- Load model ---
    if model_name == "mediapipe":
        import mediapipe as mp  # type: ignore
        _mp = mp.solutions.hands.Hands(
            static_image_mode=True,   # offline: seeking random frames, no temporal state
            max_num_hands=config.MAX_NUM_HANDS,
            model_complexity=config.MODEL_COMPLEXITY,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        )
        def _get_tips(frame):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            return _tips_mediapipe(_mp.process(rgb), fw, fh)   # fw/fh from outer scope
        def _close(): _mp.close()
    else:
        _op = OpenPoseHandModel(config, _PROJECT_ROOT)
        _op.load()
        def _get_tips(frame):
            return _tips_openpose(_op.infer(frame))
        def _close(): _op.close()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        _close()
        raise RuntimeError(f"Could not open video: {video_path}")

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Keyboard midpoint for L/R hand split (keyboard may not be centred in frame)
    _key_xs = [k['center'][0] for k in mask.keys] if mask.keys else [fw / 2]
    kb_mid  = (min(_key_xs) + max(_key_xs)) / 2

    errors         : list = []
    per_finger     : dict = {'L': {i: [] for i in range(5)},
                             'R': {i: [] for i in range(5)}}
    accurate       : dict = {'L': 0, 'R': 0}
    detection_fail : dict = {'L': 0, 'R': 0}
    missed         : int  = 0
    skipped        : int  = 0
    total      : int  = len(note_events)
    cancelled  : bool = False

    for i, evt in enumerate(note_events):
        if cancel_ev is not None and cancel_ev.is_set():
            cancelled = True
            break

        if progress_cb is not None and (i % 5 == 0 or i == total - 1):
            progress_cb(i + 1, total)

        if (i + 1) % 25 == 0 or i == total - 1:
            print(f"  {i + 1}/{total} …", end="\r")

        midi_note = evt["note"]
        key_info  = next((k for k in mask.keys if k["midi_note"] == midi_note), None)
        if key_info is None:
            skipped += 1
            continue

        cx, cy   = key_info["center"]
        hand     = 'L' if cx < kb_mid else 'R'
        key_poly = key_info["polygon"]

        frame_idx = _nearest_frame_idx(evt["time_s"], frame_times)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            missed += 1
            continue

        tips = _get_tips(frame)
        if not tips:
            missed += 1
            continue

        # Polygon-containment identification (mirrors test.py exactly).
        # If no tip is inside the polygon the model failed to place a landmark
        # on the pressed key — count as detection_fail, exclude from MJMPE.
        inside_tips = [t for t in tips
                       if cv2.pointPolygonTest(
                           key_poly, (float(t[1]), float(t[2])), False
                       ) >= 0]

        if inside_tips:
            best_slot, best_tx, _ty = min(inside_tips, key=lambda t: abs(t[1] - cx))
            h_err = abs(best_tx - cx)
            errors.append(h_err)
            per_finger[hand][best_slot].append(h_err)
            if h_err < mask.wkw * config.ACCURACY_THRESHOLD_RATIO:
                accurate[hand] += 1
        else:
            detection_fail[hand] += 1

    print()
    cap.release()
    _close()

    if cancelled:
        return None     # don't write partial results

    half_key   = mask.wkw * config.ACCURACY_THRESHOLD_RATIO
    matched    = len(errors)
    total_df   = sum(detection_fail.values())
    all_events = matched + total_df + missed

    def _finger_dict(side: str) -> dict:
        return {
            str(i): {
                "name":         _FINGER_NAMES[i],
                "count":        len(per_finger[side][i]),
                "mjmpe":        (round(float(np.mean(per_finger[side][i])), 3)
                                 if per_finger[side][i] else None),
                "median_px":    (round(float(np.median(per_finger[side][i])), 3)
                                 if per_finger[side][i] else None),
                "accuracy_pct": (round(
                    100.0 * sum(1 for e in per_finger[side][i] if e < half_key)
                    / len(per_finger[side][i]), 2)
                                 if per_finger[side][i] else None),
                "low_sample":   len(per_finger[side][i]) < 5,
            }
            for i in range(5)
        }

    result = {
        "pid":                  pid,
        "model":                model_name,
        "fitzpatrick":          metadata.get("fitzpatrick_type"),
        "lux":                  metadata.get("lux_value"),
        "hand_size_cm":         metadata.get("hand_size_cm"),
        "notes_total":          total,
        "notes_skipped":        skipped,
        "notes_matched":        matched,
        "notes_detection_fail": total_df,
        "notes_missed":         missed,
        "accuracy_threshold_px": half_key,
        "mjmpe_px":             round(float(np.mean(errors)), 3) if errors else None,
        "accuracy_pct":         (round(100.0 * sum(accurate.values()) / matched, 2)
                                 if matched else None),
        "detection_rate_pct":   (round(100.0 * matched / all_events, 2)
                                 if all_events else None),
        "per_hand": {
            "L": {
                "matched":        sum(len(per_finger['L'][i]) for i in range(5)),
                "detection_fail": detection_fail['L'],
                "mjmpe_px":       (round(float(np.mean(
                                       [e for i in range(5) for e in per_finger['L'][i]])), 3)
                                   if any(per_finger['L'][i] for i in range(5)) else None),
                "accuracy_pct":   (round(100.0 * accurate['L']
                                         / sum(len(per_finger['L'][i]) for i in range(5)), 2)
                                   if any(per_finger['L'][i] for i in range(5)) else None),
                "fingers":        _finger_dict('L'),
            },
            "R": {
                "matched":        sum(len(per_finger['R'][i]) for i in range(5)),
                "detection_fail": detection_fail['R'],
                "mjmpe_px":       (round(float(np.mean(
                                       [e for i in range(5) for e in per_finger['R'][i]])), 3)
                                   if any(per_finger['R'][i] for i in range(5)) else None),
                "accuracy_pct":   (round(100.0 * accurate['R']
                                         / sum(len(per_finger['R'][i]) for i in range(5)), 2)
                                   if any(per_finger['R'][i] for i in range(5)) else None),
                "fingers":        _finger_dict('R'),
            },
        },
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{pid}_{model_name}_results.json"
    with open(out_path, "w") as fh_out:
        json.dump(result, fh_out, indent=2)

    print(f"  MJMPE: {result['mjmpe_px']} px  |  Acc: {result['accuracy_pct']}%  "
          f"|  DetRate: {result['detection_rate_pct']}%  "
          f"|  Matched: {matched}  Fail: {total_df}  Missed: {missed}")
    for side, side_name in [('L', 'Left'), ('R', 'Right')]:
        ph = result["per_hand"][side]
        print(f"    {side_name} hand  (matched={ph['matched']}, fail={ph['detection_fail']}):")
        for i, name in enumerate(_FINGER_NAMES):
            pf = ph["fingers"][str(i)]
            if pf["count"]:
                print(f"      {name:<7}  {pf['mjmpe']:6.2f} px  "
                      f"{pf['accuracy_pct']:5.1f}%  ({pf['count']} notes)")
    print(f"  → {out_path}\n")

    return out_path


# ---------------------------------------------------------------------------
# Graphical UI
# ---------------------------------------------------------------------------

def run_ui() -> None:
    """Launch the tkinter analysis management UI."""
    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError:
        print("ERROR: tkinter not available.  Use the CLI interface instead.",
              file=sys.stderr)
        return

    # -----------------------------------------------------------------------
    # State shared between main thread and worker
    # -----------------------------------------------------------------------
    _cancel_ev = threading.Event()
    _q         = _queue.Queue()

    # -----------------------------------------------------------------------
    # Root window
    # -----------------------------------------------------------------------
    root = tk.Tk()
    root.title("MJMPE Offline Analysis")
    root.minsize(580, 560)
    root.resizable(True, False)
    PAD = 8

    style = ttk.Style()
    try:
        style.theme_use("clam")
    except Exception:
        pass

    # -----------------------------------------------------------------------
    # Session list (Treeview)
    # -----------------------------------------------------------------------
    sf = ttk.LabelFrame(root, text=" Sessions ", padding=PAD)
    sf.pack(fill=tk.BOTH, expand=True, padx=PAD, pady=(PAD, 0))

    COLS = ("pid", "mediapipe", "openpose")
    tree = ttk.Treeview(sf, columns=COLS, show="headings", height=9,
                        selectmode="none")
    tree.heading("pid",       text="Participant")
    tree.heading("mediapipe", text="MediaPipe")
    tree.heading("openpose",  text="OpenPose")
    tree.column("pid",       width=180, anchor="w")
    tree.column("mediapipe", width=110, anchor="center")
    tree.column("openpose",  width=110, anchor="center")

    vsb = ttk.Scrollbar(sf, orient=tk.VERTICAL, command=tree.yview)
    tree.configure(yscrollcommand=vsb.set)
    tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    vsb.pack(side=tk.RIGHT, fill=tk.Y)

    # Style processed / unprocessed rows differently
    tree.tag_configure("done",    foreground="#3a3")
    tree.tag_configure("partial", foreground="#a80")
    tree.tag_configure("none",    foreground="#888")

    summary_var = tk.StringVar(value="Scanning…")
    ttk.Label(root, textvariable=summary_var, anchor="w").pack(
        fill=tk.X, padx=PAD + 2, pady=(2, 0))

    # -----------------------------------------------------------------------
    # Options
    # -----------------------------------------------------------------------
    of = ttk.LabelFrame(root, text=" Options ", padding=PAD)
    of.pack(fill=tk.X, padx=PAD, pady=(PAD, 0))

    model_var = tk.StringVar(value="both")
    for txt, val in [("Both models", "both"),
                     ("MediaPipe only", "mediapipe"),
                     ("OpenPose only",  "openpose")]:
        ttk.Radiobutton(of, text=txt, variable=model_var, value=val).pack(
            side=tk.LEFT, padx=(0, PAD))

    skip_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(of, text="Skip already-processed",
                    variable=skip_var).pack(side=tk.RIGHT)

    # -----------------------------------------------------------------------
    # Progress
    # -----------------------------------------------------------------------
    pf = ttk.LabelFrame(root, text=" Progress ", padding=PAD)
    pf.pack(fill=tk.X, padx=PAD, pady=(PAD, 0))

    ttk.Label(pf, text="Current session:").grid(row=0, column=0, sticky="w")
    cur_bar = ttk.Progressbar(pf, mode="determinate", maximum=100)
    cur_bar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(2, 0))
    cur_lbl_var = tk.StringVar(value="—")
    ttk.Label(pf, textvariable=cur_lbl_var).grid(
        row=2, column=0, columnspan=2, sticky="w")

    ttk.Label(pf, text="Overall:").grid(row=3, column=0, sticky="w",
                                        pady=(PAD // 2, 0))
    ovr_bar = ttk.Progressbar(pf, mode="determinate", maximum=100)
    ovr_bar.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(2, 0))
    ovr_lbl_var = tk.StringVar(value="—")
    ttk.Label(pf, textvariable=ovr_lbl_var).grid(
        row=5, column=0, columnspan=2, sticky="w")

    pf.columnconfigure(0, weight=1)

    # -----------------------------------------------------------------------
    # Buttons
    # -----------------------------------------------------------------------
    bf = ttk.Frame(root, padding=(PAD, PAD))
    bf.pack(fill=tk.X)

    refresh_btn = ttk.Button(bf, text="⟳  Refresh")
    start_btn   = ttk.Button(bf, text="▶  Start Analysis")
    cancel_btn  = ttk.Button(bf, text="✕  Cancel", state=tk.DISABLED)

    refresh_btn.pack(side=tk.LEFT)
    cancel_btn.pack(side=tk.RIGHT)
    start_btn.pack(side=tk.RIGHT, padx=(0, 4))

    # -----------------------------------------------------------------------
    # Session data & refresh
    # -----------------------------------------------------------------------
    _sessions: list = []

    def _row_tag(s):
        if s["mediapipe"] and s["openpose"]:
            return "done"
        if s["mediapipe"] or s["openpose"]:
            return "partial"
        return "none"

    def refresh():
        nonlocal _sessions
        _sessions = _scan_sessions(_RAW_DIR)
        for item in tree.get_children():
            tree.delete(item)
        for s in _sessions:
            mp_s = "✓" if s["mediapipe"] else "—"
            op_s = "✓" if s["openpose"]  else "—"
            tree.insert("", tk.END, iid=s["pid"],
                        values=(s["pid"], mp_s, op_s),
                        tags=(_row_tag(s),))
        done    = sum(1 for s in _sessions if s["mediapipe"] and s["openpose"])
        partial = sum(1 for s in _sessions
                      if (s["mediapipe"] or s["openpose"])
                      and not (s["mediapipe"] and s["openpose"]))
        n = len(_sessions)
        summary_var.set(
            f"{n} participant session{'s' if n != 1 else ''} found  "
            f"·  {done} fully processed  ·  {partial} partial"
        )

    refresh()

    # -----------------------------------------------------------------------
    # Worker thread
    # -----------------------------------------------------------------------
    def _worker():
        model_choice = model_var.get()
        skip         = skip_var.get()
        models = (["mediapipe", "openpose"] if model_choice == "both"
                  else [model_choice])

        work = [(s["dir"], s["pid"], m)
                for s in _sessions
                for m in models
                if not (skip and s[m])]

        total_work = len(work)
        if total_work == 0:
            _q.put({"t": "finished",
                    "msg": "Nothing to process — all sessions already done."})
            return

        for step, (sess_dir, pid, model_name) in enumerate(work):
            if _cancel_ev.is_set():
                break

            _q.put({"t": "start", "pid": pid, "model": model_name,
                    "step": step, "total": total_work})
            try:
                # Use default-argument capture to avoid late-binding closure bug
                def _prog(done, total,
                          _s=step, _tw=total_work, _p=pid, _m=model_name):
                    _q.put({"t": "progress", "pid": _p, "model": _m,
                            "done": done, "total": total,
                            "step": _s, "total_work": _tw})

                out = _analyse_session(
                    sess_dir, model_name, _PROCESSED_DIR,
                    progress_cb=_prog, cancel_ev=_cancel_ev,
                )
                if out is not None:
                    _q.put({"t": "session_done", "pid": pid, "model": model_name,
                            "step": step + 1, "total": total_work})
            except Exception as exc:
                _q.put({"t": "error", "pid": pid, "model": model_name,
                        "msg": str(exc)})

        if _cancel_ev.is_set():
            _q.put({"t": "cancelled"})
        else:
            _q.put({"t": "finished", "msg": "All done."})

    # -----------------------------------------------------------------------
    # Queue polling (runs on main thread via after())
    # -----------------------------------------------------------------------
    def poll():
        try:
            while True:
                msg = _q.get_nowait()
                t   = msg["t"]

                if t == "start":
                    cur_bar["value"] = 0
                    cur_lbl_var.set(
                        f"{msg['pid']}  ·  {msg['model']}  ·  starting…")
                    ovr_bar["value"] = 100 * msg["step"] / msg["total"]
                    ovr_lbl_var.set(f"{msg['step']} / {msg['total']} sessions")

                elif t == "progress":
                    pct = 100 * msg["done"] / max(msg["total"], 1)
                    cur_bar["value"] = pct
                    cur_lbl_var.set(
                        f"{msg['pid']}  ·  {msg['model']}  ·  "
                        f"note {msg['done']} / {msg['total']}")
                    step_frac = (msg["step"] + pct / 100) / msg["total_work"]
                    ovr_bar["value"] = 100 * step_frac
                    ovr_lbl_var.set(
                        f"{msg['step']} / {msg['total_work']} sessions  "
                        f"({100 * step_frac:.0f} %)")

                elif t == "session_done":
                    cur_bar["value"] = 100
                    ovr_bar["value"] = 100 * msg["step"] / msg["total"]
                    ovr_lbl_var.set(f"{msg['step']} / {msg['total']} sessions")
                    # Update treeview tick
                    col_idx = list(COLS).index(msg["model"])
                    try:
                        vals = list(tree.item(msg["pid"], "values"))
                        vals[col_idx] = "✓"
                        # Recompute tag
                        mp_done = (vals[1] == "✓")
                        op_done = (vals[2] == "✓")
                        tag = ("done"    if mp_done and op_done else
                               "partial" if mp_done or  op_done else "none")
                        tree.item(msg["pid"], values=vals, tags=(tag,))
                    except Exception:
                        pass

                elif t == "error":
                    cur_lbl_var.set(
                        f"ERROR  {msg['pid']} | {msg['model']}: {msg['msg']}")

                elif t in ("finished", "cancelled"):
                    start_btn.configure(state=tk.NORMAL)
                    cancel_btn.configure(state=tk.DISABLED)
                    _cancel_ev.clear()
                    refresh()
                    if t == "finished":
                        ovr_bar["value"] = 100
                        cur_bar["value"] = 100
                        ovr_lbl_var.set("Complete")
                        cur_lbl_var.set(msg.get("msg", "All done."))
                    else:
                        cur_lbl_var.set("Cancelled.")

        except _queue.Empty:
            pass

        root.after(100, poll)

    # -----------------------------------------------------------------------
    # Button handlers
    # -----------------------------------------------------------------------
    def start():
        _cancel_ev.clear()
        start_btn.configure(state=tk.DISABLED)
        cancel_btn.configure(state=tk.NORMAL)
        cur_bar["value"] = 0
        ovr_bar["value"] = 0
        cur_lbl_var.set("Starting…")
        ovr_lbl_var.set("")
        threading.Thread(target=_worker, daemon=True).start()

    def cancel():
        _cancel_ev.set()
        cancel_btn.configure(state=tk.DISABLED)
        cur_lbl_var.set("Cancelling…")

    start_btn.configure(command=start)
    cancel_btn.configure(command=cancel)
    refresh_btn.configure(command=refresh)

    root.after(100, poll)
    root.mainloop()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Offline per-session MJMPE analysis (MediaPipe or OpenPose)."
    )
    parser.add_argument("session_dir", nargs="?",
                        help="Session directory, e.g. data/raw/p001/ "
                             "(omit when using --ui)")
    parser.add_argument("--model",
                        choices=["mediapipe", "openpose", "both"],
                        default="both",
                        help="Model to run (default: both sequentially)")
    parser.add_argument("--out", default=None,
                        help="Output directory (default: same as session_dir)")
    parser.add_argument("--ui", action="store_true",
                        help="Launch the graphical analysis UI")
    args = parser.parse_args()

    if args.ui:
        run_ui()
        return

    if not args.session_dir:
        parser.error("session_dir is required when not using --ui")

    session_dir = Path(args.session_dir)
    out_dir     = Path(args.out) if args.out else _PROCESSED_DIR
    models      = (["mediapipe", "openpose"] if args.model == "both"
                   else [args.model])

    for model_name in models:
        try:
            _analyse_session(session_dir, model_name, out_dir)
        except FileNotFoundError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
