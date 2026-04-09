"""
data_summary.py  –  Count collected data rows across all sessions.

Usage:
    python -m scripts.data_summary
"""

import json
import csv
from pathlib import Path

RAW_DIR  = Path(__file__).parent.parent / "data" / "raw"
PROC_DIR = Path(__file__).parent.parent / "data" / "processed"

def _count_lines(path: Path) -> int:
    """Count non-empty lines in a text file (works for CSV and JSONL)."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        return sum(1 for l in lines if l.strip())
    except Exception:
        return 0

def main():
    session_dirs = sorted(p for p in RAW_DIR.iterdir() if p.is_dir())

    total_frames      = 0
    total_midi_events = 0
    total_note_on     = 0
    total_matched_mp  = 0
    total_matched_op  = 0
    sessions          = []

    print(f"\n{'PID':<8} {'Frames':>8} {'MIDI evts':>10} {'note_on':>8} "
          f"{'MP matched':>11} {'OP matched':>11}  Session date")
    print("-" * 80)

    for sd in session_dirs:
        pid = sd.name

        # --- frames CSV (header row excluded) ---
        frames_file = sd / f"{pid}_frames.csv"
        frame_rows  = max(0, _count_lines(frames_file) - 1)  # minus header

        # --- MIDI JSONL ---
        midi_file   = sd / f"{pid}_midi.jsonl"
        midi_lines  = _count_lines(midi_file)
        note_on_cnt = 0
        if midi_file.exists():
            for line in midi_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                    if ev.get("type") == "note_on" and ev.get("velocity", 0) > 0:
                        note_on_cnt += 1
                except json.JSONDecodeError:
                    pass

        # --- session date ---
        sess_file = sd / f"{pid}_session.json"
        date_str  = ""
        if sess_file.exists():
            try:
                meta     = json.loads(sess_file.read_text(encoding="utf-8"))
                date_str = meta.get("recording_start", "")[:10]
            except Exception:
                pass

        # --- processed results ---
        mp_matched = op_matched = 0
        for model, var in [("mediapipe", "mp_matched"), ("openpose", "op_matched")]:
            res_file = PROC_DIR / f"{pid}_{model}_results.json"
            if res_file.exists():
                try:
                    res = json.loads(res_file.read_text(encoding="utf-8"))
                    if model == "mediapipe":
                        mp_matched = res.get("notes_matched", 0)
                    else:
                        op_matched = res.get("notes_matched", 0)
                except Exception:
                    pass

        print(f"{pid:<8} {frame_rows:>8,} {midi_lines:>10,} {note_on_cnt:>8,} "
              f"{mp_matched:>11,} {op_matched:>11,}  {date_str}")

        total_frames      += frame_rows
        total_midi_events += midi_lines
        total_note_on     += note_on_cnt
        total_matched_mp  += mp_matched
        total_matched_op  += op_matched

    n = len(session_dirs)
    print("-" * 80)
    print(f"{'TOTAL':<8} {total_frames:>8,} {total_midi_events:>10,} {total_note_on:>8,} "
          f"{total_matched_mp:>11,} {total_matched_op:>11,}  ({n} sessions)")

    print(f"""
Summary
-------
  Sessions recorded          : {n}
  Video frames               : {total_frames:,}
  MIDI events (all types)    : {total_midi_events:,}
  Note-on events (key press) : {total_note_on:,}
  MediaPipe matched notes    : {total_matched_mp:,}
  OpenPose  matched notes    : {total_matched_op:,}
  Avg frames / session       : {total_frames // n if n else 0:,}
  Avg note-ons / session     : {total_note_on // n if n else 0:,}
""")

if __name__ == "__main__":
    main()
