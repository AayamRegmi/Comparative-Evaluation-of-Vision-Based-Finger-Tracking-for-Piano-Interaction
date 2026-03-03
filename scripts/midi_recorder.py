"""midi_recorder.py
Threaded MIDI input recorder with perf_counter timestamps.

All events are timestamped relative to a shared t0 (time.perf_counter())
that must match the t0 recorded when the first video frame is written.
This lets you precisely map any MIDI event to its corresponding video frame.

Output files per session:
  <pid>_midi.jsonl  – one JSON object per line, fields: time_s, type, channel,
                       note, velocity, control, value, program (as applicable)
  <pid>_midi.mid    – standard MIDI file (type 0, 120 BPM)
  <pid>_frames.csv  – frame_index,time_s  for every recorded video frame
"""

import csv
import json
import threading
import time
from pathlib import Path

import mido


def list_midi_input_ports() -> list:
    """Return names of all available MIDI input ports."""
    return mido.get_input_names()


class MidiRecorder:
    """
    Thread-safe MIDI recorder.

    Typical use:
        ports = list_midi_input_ports()
        rec = MidiRecorder(ports[0])
        t0 = time.perf_counter()   # shared with video frame logging
        rec.start(t0)
        ...
        rec.stop()
        jsonl_path, mid_path = rec.save(out_dir, "p001")
    """

    def __init__(self, port_name: str):
        self.port_name = port_name
        self._events: list = []          # list of (time_s: float, msg: mido.Message)
        self._lock = threading.Lock()
        self._port = None
        self._thread = None
        self._stop_evt = threading.Event()
        self._t0 = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, t0: float) -> None:
        """Open the MIDI port and start the listener thread.

        Args:
            t0: time.perf_counter() value at the moment recording begins.
                Must be the same t0 used when logging video frame timestamps.
        """
        self._t0 = t0
        with self._lock:
            self._events.clear()
        self._stop_evt.clear()
        self._port = mido.open_input(self.port_name)
        self._thread = threading.Thread(
            target=self._listen, daemon=True, name="MidiListener"
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the listener to stop, close the port, and join the thread."""
        self._stop_evt.set()
        if self._port is not None:
            try:
                self._port.close()
            except Exception:
                pass
            self._port = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def get_events(self) -> list:
        """Thread-safe snapshot of all recorded (time_s, message) pairs."""
        with self._lock:
            return list(self._events)

    def save(self, out_dir: Path, pid_str: str) -> tuple:
        """Write JSONL and MIDI files.

        Args:
            out_dir:  Participant output directory (e.g. data/raw/p001/).
            pid_str:  Participant string prefix, e.g. "p001".

        Returns:
            (jsonl_path, mid_path)
        """
        events = self.get_events()

        jsonl_path = self._save_jsonl(events, out_dir, pid_str)
        mid_path   = self._save_midi(events, out_dir, pid_str)

        return jsonl_path, mid_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _listen(self) -> None:
        while not self._stop_evt.is_set():
            if self._port is None:
                break
            try:
                for msg in self._port.iter_pending():
                    # Skip timing/clock messages — they're not musically meaningful
                    if msg.is_realtime:
                        continue
                    t = time.perf_counter() - self._t0
                    with self._lock:
                        self._events.append((t, msg))
            except Exception:
                break
            time.sleep(0.001)   # 1 ms poll keeps CPU load negligible

    def _save_jsonl(self, events: list, out_dir: Path, pid_str: str) -> Path:
        """Save events as a JSONL file (one JSON object per line)."""
        path = out_dir / f"{pid_str}_midi.jsonl"
        with open(path, "w", newline="") as fh:
            for t, msg in events:
                row = {"time_s": round(t, 6), "type": msg.type}
                for attr in ("channel", "note", "velocity", "control", "value", "program"):
                    if hasattr(msg, attr):
                        row[attr] = getattr(msg, attr)
                fh.write(json.dumps(row) + "\n")
        print(f"MIDI JSONL  → {path}  ({len(events)} events)")
        return path

    def _save_midi(self, events: list, out_dir: Path, pid_str: str) -> Path:
        """Save events as a standard MIDI file (type 0, 120 BPM)."""
        path = out_dir / f"{pid_str}_midi.mid"

        # 120 BPM → 500 000 µs per beat; 480 ticks per beat → 1 tick ≈ 1.042 ms
        tempo         = 500_000
        ticks_per_beat = 480
        us_per_tick   = tempo / ticks_per_beat

        mid   = mido.MidiFile(type=0, ticks_per_beat=ticks_per_beat)
        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))

        prev_tick = 0
        for t, msg in events:
            abs_tick = int(t * 1_000_000 / us_per_tick)
            delta    = max(0, abs_tick - prev_tick)
            prev_tick = abs_tick
            try:
                track.append(msg.copy(time=delta))
            except Exception:
                # Some sysex/meta messages can't be copied — skip gracefully
                pass

        mid.save(str(path))
        print(f"MIDI file   → {path}")
        return path


# ------------------------------------------------------------------
# Frame timestamp logger (companion to MidiRecorder)
# ------------------------------------------------------------------

class FrameLogger:
    """
    Logs (frame_index, time_s) pairs for every written video frame,
    using the same t0 origin as MidiRecorder.

    Usage:
        fl = FrameLogger(t0)
        fl.log()          # call once per frame.write()
        ...
        fl.save(out_dir, "p001")  # writes p001_frames.csv
    """

    def __init__(self, t0: float):
        self._t0    = t0
        self._rows: list = []   # list of (frame_index, time_s)
        self._idx   = 0

    def log(self) -> None:
        """Record the current frame's timestamp. Call once per writer.write()."""
        self._rows.append((self._idx, round(time.perf_counter() - self._t0, 6)))
        self._idx += 1

    def save(self, out_dir: Path, pid_str: str) -> Path:
        path = out_dir / f"{pid_str}_frames.csv"
        with open(path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["frame_index", "time_s"])
            writer.writerows(self._rows)
        print(f"Frame log   → {path}  ({self._idx} frames)")
        return path
