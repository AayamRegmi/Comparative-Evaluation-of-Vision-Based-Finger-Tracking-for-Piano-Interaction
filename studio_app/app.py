"""
studio_app/app.py
Photoshop-style studio launcher for Piano Finger Tracking.

Layout:
  TITLE BAR  (dark navy + gold accent)
  TOOLBAR | VIEWPORT (live camera, letterboxed) | PROPERTIES PANEL
  CONSOLE (collapsible)
  STATUS BAR

Run from project root:
    python studio_app\app.py
"""

from __future__ import annotations

import abc
import json
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, scrolledtext
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parents[1].resolve()
sys.path.insert(0, str(ROOT))

import scripts.config as cfg
from scripts.camera_setup import init_camera
from scripts.lux_calculator import lux_to_label

PYTHON = ROOT / "piano_env" / "Scripts" / "python.exe"
if not PYTHON.exists():
    PYTHON = Path(sys.executable)

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
BG          = "#0D1B2A"
TOOLBAR_BG  = "#0A1520"
HEADER_BG   = "#0F2240"
PANEL_BG    = "#111E2E"
CONSOLE_BG  = "#080F18"
STATUS_BG   = "#060E18"
ACCENT      = "#E8C86C"
ACCENT2     = "#C4A84E"
FG          = "#D0DDE8"
FG_DIM      = "#6A7E92"
FG_BRIGHT   = "#FFFFFF"
BORDER      = "#1F3550"
REC_RED     = "#E84040"
REC_DARK    = "#8B0000"
DOT_OK      = "#4CAF50"
DOT_RUN     = "#FFC107"
DOT_ERR     = "#F44336"
DOT_IDLE    = "#3A3A3A"
ENTRY_BG    = "#0D1D30"
BTN_BG      = "#1A3A6B"
BTN_HOVER   = "#224A88"
BTN_FG      = "#D0DDE8"

TOOLBAR_W   = 66
PROPS_W     = 236
STATUS_H    = 26
TITLE_H     = 44
CONSOLE_H   = 140

FONT_TITLE  = ("Segoe UI", 11, "bold")
FONT_LABEL  = ("Segoe UI", 9)
FONT_SMALL  = ("Segoe UI", 8)
FONT_MONO   = ("Consolas", 9)
FONT_ICON   = ("Segoe UI Emoji", 18)
FONT_TOOL   = ("Segoe UI Emoji", 14)


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _psec(panel: tk.Frame, title: str) -> tk.Frame:
    """Properties section header + returns inner frame."""
    hdr = tk.Frame(panel, bg=HEADER_BG)
    hdr.pack(fill="x", padx=4, pady=(8, 0))
    tk.Label(hdr, text=title.upper(), bg=HEADER_BG, fg=ACCENT,
             font=FONT_SMALL).pack(side="left", padx=6, pady=3)
    inner = tk.Frame(panel, bg=PANEL_BG)
    inner.pack(fill="x", padx=4, pady=(0, 2))
    return inner


def _prow(parent: tk.Frame, label: str, value: str = "", var: tk.StringVar | None = None) -> tk.Label:
    """Label + value row in properties panel. Returns value Label."""
    row = tk.Frame(parent, bg=PANEL_BG)
    row.pack(fill="x", padx=6, pady=1)
    tk.Label(row, text=label, bg=PANEL_BG, fg=FG_DIM,
             font=FONT_SMALL, width=14, anchor="w").pack(side="left")
    lbl = tk.Label(row, textvariable=var if var else None,
                   text=value if not var else "",
                   bg=PANEL_BG, fg=FG, font=FONT_SMALL, anchor="w")
    lbl.pack(side="left", fill="x", expand=True)
    return lbl


def _pentry(parent: tk.Frame, label: str, var: tk.StringVar, width: int = 10) -> tk.Entry:
    """Label + Entry row in properties panel. Returns Entry."""
    row = tk.Frame(parent, bg=PANEL_BG)
    row.pack(fill="x", padx=6, pady=2)
    tk.Label(row, text=label, bg=PANEL_BG, fg=FG_DIM,
             font=FONT_SMALL, width=14, anchor="w").pack(side="left")
    ent = tk.Entry(row, textvariable=var, width=width,
                   bg=ENTRY_BG, fg=FG, insertbackground=FG,
                   relief="flat", font=FONT_SMALL)
    ent.pack(side="left")
    return ent


def _pbtn(parent: tk.Frame, text: str, cmd, bg: str = BTN_BG,
          fg: str = BTN_FG, font=FONT_LABEL) -> tk.Button:
    """Styled button helper."""
    b = tk.Button(parent, text=text, command=cmd, bg=bg, fg=fg,
                  activebackground=BTN_HOVER, activeforeground=FG_BRIGHT,
                  relief="flat", font=font, cursor="hand2", padx=8, pady=4)
    b.pack(fill="x", padx=6, pady=3)
    return b


# ---------------------------------------------------------------------------
# Base tool
# ---------------------------------------------------------------------------

class BaseTool(abc.ABC):
    name:        str  = ""
    icon:        str  = "?"
    uses_camera: bool = True

    def activate(self, app: "StudioApp") -> None:
        """Called when tool is selected. Open models / resources."""

    def deactivate(self, app: "StudioApp") -> None:
        """Called when tool is deselected. Release resources."""

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Return (possibly annotated) copy or the same frame."""
        return frame

    @abc.abstractmethod
    def build_properties(self, panel: tk.Frame, app: "StudioApp") -> None:
        """Populate the properties panel (panel is already empty)."""


# ---------------------------------------------------------------------------
# Home tool — raw camera feed
# ---------------------------------------------------------------------------

class HomeTool(BaseTool):
    name = "Home"
    icon = "🏠"
    uses_camera = True

    def build_properties(self, panel: tk.Frame, app: "StudioApp") -> None:
        sec = _psec(panel, "Camera")
        _prow(sec, "Device",     str(cfg.CAMERA_INDEX))
        _prow(sec, "Resolution", f"{cfg.CAP_WIDTH}×{cfg.CAP_HEIGHT}")
        _prow(sec, "FPS",        var=app.fps_var)


# ---------------------------------------------------------------------------
# Preview tool — MediaPipe landmarks + key mask overlay
# ---------------------------------------------------------------------------

class PreviewTool(BaseTool):
    name = "Preview"
    icon = "👁"
    uses_camera = True

    def __init__(self):
        self._mm          = None
        self._fitz_frames = []
        self._fitz_label  = None
        self._key_mask    = None
        self._mask_loaded = False
        self._mask_panel  = None
        # Properties vars (set in build_properties)
        self._fitz_var    = None
        self._model_var   = None
        self._hands_var   = None
        self._mask_btn    = None

    def activate(self, app: "StudioApp") -> None:
        from scripts.model_manager import ModelManager
        self._mm = ModelManager(cfg, str(ROOT))
        self._mm.ensure_loaded()
        # Try to load key mask (same file used by Recalibrate and run_calibration)
        mask_path = ROOT / "data" / "calibration" / "key_centers.json"
        if mask_path.exists() and not self._mask_loaded:
            try:
                from scripts.key_calibration import KeyMask, MaskControlPanel
                self._key_mask = KeyMask.load(str(mask_path))
                self._mask_panel = MaskControlPanel(
                    self._key_mask,
                    save_fn=lambda: self._save_mask(mask_path),
                )
                self._mask_loaded = True
            except Exception:
                pass

    def _save_mask(self, path) -> None:
        if self._key_mask is None:
            return
        self._key_mask.save(Path(path).parent, cfg.CAP_WIDTH, cfg.CAP_HEIGHT)

    def toggle_mask_panel(self) -> None:
        if self._mask_panel:
            self._mask_panel.toggle()

    def on_canvas_mouse(self, cv2_event: int, fx: int, fy: int, flags: int = 0) -> None:
        if self._key_mask:
            self._key_mask.on_mouse(cv2_event, fx, fy, flags)

    def on_canvas_scroll(self, delta: int, fx: int, fy: int) -> None:
        if self._key_mask:
            self._key_mask.on_mouse(
                cv2.EVENT_MOUSEWHEEL, fx, fy,
                cv2.EVENT_FLAG_CTRLKEY if delta > 0 else 0,
                param=delta,
            )

    def on_key(self, keysym: str) -> None:
        """Handle key presses forwarded from the canvas while Preview is active."""
        if self._key_mask is None:
            return
        k = keysym.lower()
        if k == "bracketleft":
            self._key_mask.start_midi = max(0, self._key_mask.start_midi - 2)
            self._key_mask.mark_dirty()
        elif k == "bracketright":
            self._key_mask.start_midi = min(108, self._key_mask.start_midi + 2)
            self._key_mask.mark_dirty()
        elif k == "minus":
            self._key_mask.wkw = max(6, self._key_mask.wkw - 1)
            self._key_mask.mark_dirty()
        elif k in ("equal", "plus"):
            self._key_mask.wkw += 1
            self._key_mask.mark_dirty()
        elif k == "v":
            self._key_mask.reset_warp()

    def deactivate(self, app: "StudioApp") -> None:
        if self._mask_panel and self._mask_panel.visible:
            self._mask_panel.toggle()
        if self._mm:
            self._mm.close_all()
            self._mm = None
        self._fitz_frames = []
        self._fitz_label  = None

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        if self._mm is None:
            return frame
        out = frame.copy()
        # Key mask overlay + handle drawing
        if self._key_mask is not None:
            try:
                self._key_mask.draw(out)
                from scripts.key_calibration import draw_mask_handles
                draw_mask_handles(out, self._key_mask)
            except Exception:
                pass
        # Render mask control panel (separate OpenCV window)
        if self._mask_panel:
            self._mask_panel.render()
        # MediaPipe inference + draw
        try:
            result = self._mm.current.infer(out)
            out    = self._mm.current.draw(out, result)
            # Update hands indicator
            if self._hands_var:
                try:
                    n = len(result.multi_hand_landmarks) if result.multi_hand_landmarks else 0
                    self._hands_var.set(str(n))
                except Exception:
                    self._hands_var.set("—")
            # Fitzpatrick accumulation (20 frames)
            if len(self._fitz_frames) < 20:
                try:
                    lms = result.multi_hand_landmarks
                    if lms:
                        h_px, w_px = out.shape[:2]
                        from scripts.fitzpatrick_detector import detect_skin_type
                        skin = detect_skin_type(frame, lms[0], w_px, h_px)
                        if skin:
                            self._fitz_frames.append(skin)
                            if len(self._fitz_frames) == 20:
                                # majority vote
                                from collections import Counter
                                self._fitz_label = Counter(self._fitz_frames).most_common(1)[0][0]
                                if self._fitz_var:
                                    self._fitz_var.set(self._fitz_label)
                except Exception:
                    pass
        except Exception:
            pass
        return out

    def build_properties(self, panel: tk.Frame, app: "StudioApp") -> None:
        sec = _psec(panel, "Model")
        self._model_var = tk.StringVar(value=cfg.MODEL_MEDIAPIPE)

        def _switch():
            if self._mm:
                self._mm.cycle()
                self._model_var.set(self._mm.current.name)

        _pbtn(sec, "Switch Model  (M)", _switch)
        _prow(sec, "Active model", var=self._model_var)

        sec2 = _psec(panel, "Stats")
        _prow(sec2, "FPS",     var=app.fps_var)
        _prow(sec2, "Infer ms", var=app.infer_var)
        self._hands_var = tk.StringVar(value="—")
        _prow(sec2, "Hands",   var=self._hands_var)

        sec3 = _psec(panel, "Participant")
        self._fitz_var = tk.StringVar(value="Detecting…")
        _prow(sec3, "Fitzpatrick", var=self._fitz_var)

        if self._key_mask is not None:
            sec4 = _psec(panel, "Key Mask  (N)")
            _pbtn(sec4, "Toggle Mask Panel  (N)", lambda: self.toggle_mask_panel())
            _pbtn(sec4, "Reset Warp  (V)", lambda: (
                self._key_mask.reset_warp() if self._key_mask else None
            ))


# ---------------------------------------------------------------------------
# Record tool
# ---------------------------------------------------------------------------

class RecordTool(BaseTool):
    name = "Record"
    icon = "⏺"
    uses_camera = True

    def __init__(self):
        self._mm        = None
        self._recording = False
        self._writer    = None
        self._midi_rec  = None
        self._frame_log = None
        self._t0        = None
        self._frame_cnt = 0
        self._pid_str   = None
        # UI vars
        self._lux_var       = None
        self._hand_var      = None
        self._port_var      = None
        self._timer_var     = None
        self._frames_var    = None
        self._status_var    = None
        self._rec_btn       = None
        self._ports         = []
        self._timer_after   = None

    def activate(self, app: "StudioApp") -> None:
        from scripts.model_manager import ModelManager
        self._mm = ModelManager(cfg, str(ROOT))
        self._mm.ensure_loaded()
        # Enumerate MIDI ports
        try:
            from scripts.midi_recorder import list_midi_input_ports
            self._ports = list_midi_input_ports()
        except Exception:
            self._ports = []

    def deactivate(self, app: "StudioApp") -> None:
        if self._recording:
            self._stop(app)
        if self._mm:
            self._mm.close_all()
            self._mm = None

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        if self._mm is None:
            return frame
        out = frame.copy()
        try:
            result = self._mm.current.infer(out)
            out    = self._mm.current.draw(out, result)
        except Exception:
            pass
        if self._recording and self._writer:
            self._writer.write(frame)
            if self._frame_log:
                self._frame_log.log()
            self._frame_cnt += 1
            if self._frames_var:
                self._frames_var.set(str(self._frame_cnt))
        # Red REC border
        if self._recording:
            h, w = out.shape[:2]
            cv2.rectangle(out, (0, 0), (w - 1, h - 1), (0, 0, 200), 4)
        return out

    def _next_pid(self) -> str:
        raw = ROOT / "data" / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        existing = [d for d in raw.iterdir() if d.is_dir() and d.name.startswith("p")]
        return f"p{len(existing) + 1:03d}"

    def _start(self, app: "StudioApp") -> None:
        lux_str  = self._lux_var.get().strip() if self._lux_var else ""
        hand_str = self._hand_var.get().strip() if self._hand_var else ""
        try:
            lux_val  = float(lux_str)  if lux_str  else 0.0
        except ValueError:
            lux_val  = 0.0
        try:
            hand_val = float(hand_str) if hand_str else 0.0
        except ValueError:
            hand_val = 0.0

        pid = self._next_pid()
        self._pid_str = pid
        out_dir = ROOT / "data" / "raw" / pid
        out_dir.mkdir(parents=True, exist_ok=True)

        # VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vpath  = out_dir / f"{pid}.mp4"
        self._writer = cv2.VideoWriter(
            str(vpath), fourcc, cfg.CAP_FPS,
            (cfg.CAP_WIDTH, cfg.CAP_HEIGHT)
        )

        self._t0        = time.perf_counter()
        self._frame_cnt = 0

        # FrameLogger
        try:
            from scripts.midi_recorder import FrameLogger
            self._frame_log = FrameLogger(self._t0)
        except Exception:
            self._frame_log = None

        # MidiRecorder
        port_name = self._port_var.get() if self._port_var else ""
        try:
            if port_name and port_name != "No ports":
                from scripts.midi_recorder import MidiRecorder
                self._midi_rec = MidiRecorder(port_name)
                self._midi_rec.start(self._t0)
        except Exception:
            self._midi_rec = None

        self._recording = True
        if self._rec_btn:
            self._rec_btn.config(text="⏹ STOP", bg=REC_RED)
        if self._status_var:
            self._status_var.set(f"● REC  {pid}")
        app.set_status(f"Recording {pid}")
        self._tick_timer(app)

    def _stop(self, app: "StudioApp") -> None:
        self._recording = False
        if self._timer_after and app:
            try:
                app.root.after_cancel(self._timer_after)
            except Exception:
                pass
        pid     = self._pid_str or "p000"
        out_dir = ROOT / "data" / "raw" / pid

        if self._writer:
            self._writer.release()
            self._writer = None

        jsonl_path = mid_path = frames_path = None
        if self._midi_rec:
            try:
                jsonl_path, mid_path = self._midi_rec.stop()
                if jsonl_path:
                    import shutil
                    for p in [jsonl_path, mid_path]:
                        if p and Path(p).exists():
                            shutil.copy(p, out_dir / Path(p).name)
            except Exception:
                pass
            self._midi_rec = None

        if self._frame_log:
            try:
                frames_path = self._frame_log.save(str(out_dir), pid)
            except Exception:
                pass
            self._frame_log = None

        # Save session JSON
        lux_val  = 0.0
        hand_val = 0.0
        try:
            lux_val  = float(self._lux_var.get())  if self._lux_var  else 0.0
        except Exception:
            pass
        try:
            hand_val = float(self._hand_var.get()) if self._hand_var else 0.0
        except Exception:
            pass

        session = {
            "pid":         pid,
            "lux":         lux_val,
            "lux_label":   lux_to_label(lux_val),
            "hand_size_cm": hand_val,
            "midi_port":   self._port_var.get() if self._port_var else "",
            "video":       f"{pid}.mp4",
            "frames_csv":  f"{pid}_frames.csv" if frames_path else None,
            "midi_jsonl":  f"{pid}_midi.jsonl" if jsonl_path  else None,
            "midi_mid":    f"{pid}_midi.mid"   if mid_path    else None,
        }
        with open(out_dir / f"{pid}_session.json", "w") as f:
            json.dump(session, f, indent=2)

        if self._rec_btn:
            self._rec_btn.config(text="⏺ REC", bg=REC_DARK)
        if self._status_var:
            self._status_var.set(f"Saved → data/raw/{pid}/")
        if self._timer_var:
            self._timer_var.set("00:00")
        if app:
            app.set_status(f"Saved {pid}")

    def _tick_timer(self, app: "StudioApp") -> None:
        if not self._recording:
            return
        elapsed = int(time.perf_counter() - self._t0)
        m, s = divmod(elapsed, 60)
        if self._timer_var:
            self._timer_var.set(f"{m:02d}:{s:02d}")
        self._timer_after = app.root.after(1000, lambda: self._tick_timer(app))

    def build_properties(self, panel: tk.Frame, app: "StudioApp") -> None:
        sec = _psec(panel, "Session")
        self._lux_var   = tk.StringVar(value="")
        self._hand_var  = tk.StringVar(value="")
        _pentry(sec, "Lux (lx)",     self._lux_var)
        _pentry(sec, "Hand size (cm)", self._hand_var)

        sec2 = _psec(panel, "MIDI Port")
        self._port_var = tk.StringVar()
        port_names = self._ports if self._ports else ["No ports"]
        self._port_var.set(port_names[0])
        om = tk.OptionMenu(sec2, self._port_var, *port_names)
        om.config(bg=ENTRY_BG, fg=FG, activebackground=BTN_HOVER,
                  activeforeground=FG_BRIGHT, font=FONT_SMALL,
                  highlightthickness=0, relief="flat")
        om["menu"].config(bg=ENTRY_BG, fg=FG)
        om.pack(fill="x", padx=6, pady=4)

        sec3 = _psec(panel, "Controls")
        self._status_var = tk.StringVar(value="Ready")
        _prow(sec3, "Status", var=self._status_var)
        self._timer_var  = tk.StringVar(value="00:00")
        _prow(sec3, "Duration", var=self._timer_var)
        self._frames_var = tk.StringVar(value="0")
        _prow(sec3, "Frames", var=self._frames_var)

        def _toggle():
            if not self._recording:
                self._start(app)
            else:
                self._stop(app)

        self._rec_btn = tk.Button(
            sec3, text="⏺ REC", command=_toggle,
            bg=REC_DARK, fg=FG_BRIGHT, activebackground=REC_RED,
            activeforeground=FG_BRIGHT, relief="flat",
            font=("Segoe UI", 11, "bold"), cursor="hand2", pady=6
        )
        self._rec_btn.pack(fill="x", padx=6, pady=6)


# ---------------------------------------------------------------------------
class RecalibrateTool(BaseTool):
    """
    Releases the studio camera, launches key_calibration.py in its own
    OpenCV window (which auto-loads key_centers.json if it exists), then
    reopens the studio camera when the calibration window is closed.
    """
    name        = "Recalibrate"
    icon        = "🔄"
    uses_camera = False   # studio viewport shows placeholder while cal window is open

    def __init__(self):
        self._proc       = None
        self._status_var = None
        self._dot_lbl    = None

    def activate(self, app: "StudioApp") -> None:
        # Hand the camera to the subprocess — release studio's capture first
        app.release_camera()
        self._launch(app)

    def deactivate(self, app: "StudioApp") -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            self._proc = None
        app.reopen_camera()

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        return frame

    def _launch(self, app: "StudioApp") -> None:
        cmd = [str(PYTHON), "-m", "scripts.recalibrate"]
        self._proc = subprocess.Popen(cmd, cwd=str(ROOT))
        if self._status_var:
            self._status_var.set("Running…")
        if self._dot_lbl:
            self._dot_lbl.config(fg=DOT_RUN)
        app.root.after(500, lambda: self._poll(app))

    def _poll(self, app: "StudioApp") -> None:
        if self._proc and self._proc.poll() is None:
            app.root.after(500, lambda: self._poll(app))
            return
        rc = self._proc.returncode if self._proc else -1
        if self._status_var:
            self._status_var.set("Done ✓ — camera restored" if rc == 0
                                 else f"Exited ({rc}) — camera restored")
        if self._dot_lbl:
            self._dot_lbl.config(fg=DOT_OK if rc == 0 else DOT_ERR)
        # Reopen studio camera now that subprocess has released it
        app.reopen_camera()

    def build_properties(self, panel: tk.Frame, app: "StudioApp") -> None:
        sec = _psec(panel, "Recalibrate")
        row = tk.Frame(sec, bg=PANEL_BG)
        row.pack(fill="x", padx=6, pady=2)
        tk.Label(row, text="Status", bg=PANEL_BG, fg=FG_DIM,
                 font=FONT_SMALL, width=14, anchor="w").pack(side="left")
        self._dot_lbl = tk.Label(row, text="●", bg=PANEL_BG,
                                 fg=DOT_IDLE, font=FONT_SMALL)
        self._dot_lbl.pack(side="left")
        self._status_var = tk.StringVar(value="Idle")
        tk.Label(row, textvariable=self._status_var, bg=PANEL_BG, fg=FG,
                 font=FONT_SMALL, anchor="w").pack(side="left", padx=4)

        sec2 = _psec(panel, "Info")
        tk.Label(sec2,
                 text="Opens the key calibration window.\n"
                      "Existing key_centers.json is loaded\n"
                      "automatically. ENTER to save, ESC to quit.",
                 bg=PANEL_BG, fg=FG_DIM, font=FONT_SMALL,
                 wraplength=200, justify="left").pack(padx=6, pady=4)


# ---------------------------------------------------------------------------
# Streaming analysis tools
# ---------------------------------------------------------------------------

class _StreamTool(BaseTool):
    uses_camera = False
    _cmd: list  = []

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        return frame

    def activate(self, app: "StudioApp") -> None:
        app.console_expand()
        self._launch_stream(app)

    def deactivate(self, app: "StudioApp") -> None:
        pass

    def _launch_stream(self, app: "StudioApp") -> None:
        cmd = [str(PYTHON)] + self._cmd

        def _reader():
            try:
                proc = subprocess.Popen(
                    cmd, cwd=str(ROOT),
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                    text=True, encoding="utf-8", errors="replace"
                )
                for line in proc.stdout:
                    app.console_write(line.rstrip())
                proc.wait()
                app.console_write(f"[exit {proc.returncode}]")
            except Exception as e:
                app.console_write(f"[error] {e}")

        threading.Thread(target=_reader, daemon=True).start()

    def build_properties(self, panel: tk.Frame, app: "StudioApp") -> None:
        sec = _psec(panel, self.name)
        tk.Label(sec, text="Output in console below.",
                 bg=PANEL_BG, fg=FG_DIM, font=FONT_SMALL,
                 wraplength=200, justify="left").pack(padx=6, pady=6)
        _pbtn(sec, f"Re-run {self.name}", lambda: self._launch_stream(app))


class AnalyseTool(_StreamTool):
    name = "Analyse"
    icon = "📊"
    _cmd = ["-m", "scripts.analyse", "--ui"]


class PlotTool(_StreamTool):
    name = "Plot"
    icon = "📈"
    _cmd = ["-m", "scripts.plot_results"]


class ExportTool(_StreamTool):
    name = "Export"
    icon = "📤"
    _cmd = ["-m", "scripts.export_csv"]


# ---------------------------------------------------------------------------
# Viewer tools (open Toplevel windows)
# ---------------------------------------------------------------------------

class _ViewerTool(BaseTool):
    uses_camera = False

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        return frame

    def activate(self, app: "StudioApp") -> None:
        self._open(app)

    def deactivate(self, app: "StudioApp") -> None:
        pass

    def _open(self, app: "StudioApp") -> None:
        raise NotImplementedError

    def build_properties(self, panel: tk.Frame, app: "StudioApp") -> None:
        sec = _psec(panel, self.name)
        _pbtn(sec, f"Open {self.name}", lambda: self._open(app))


class SummaryTool(_ViewerTool):
    name = "Summary"
    icon = "📋"

    def _open(self, app: "StudioApp") -> None:
        win = tk.Toplevel(app.root)
        win.title("Data Summary")
        win.configure(bg=BG)
        win.geometry("700x480")
        txt = scrolledtext.ScrolledText(
            win, bg=CONSOLE_BG, fg=FG, font=FONT_MONO,
            relief="flat", wrap="word"
        )
        txt.pack(fill="both", expand=True, padx=8, pady=8)
        cmd = [str(PYTHON), "-m", "scripts.data_summary"]

        def _run():
            try:
                result = subprocess.run(
                    cmd, cwd=str(ROOT), capture_output=True,
                    text=True, encoding="utf-8", errors="replace"
                )
                out = result.stdout + (result.stderr or "")
            except Exception as e:
                out = str(e)
            app.root.after(0, lambda: (
                txt.insert("end", out),
                txt.see("end")
            ))

        threading.Thread(target=_run, daemon=True).start()


class TableTool(_ViewerTool):
    name = "Table Data"
    icon = "🗄"

    def _open(self, app: "StudioApp") -> None:
        win = tk.Toplevel(app.root)
        win.title("Table Data")
        win.configure(bg=BG)
        win.geometry("820x560")

        toolbar = tk.Frame(win, bg=HEADER_BG)
        toolbar.pack(fill="x")
        _pbtn(toolbar, "Expand All", lambda: self._expand_all(tree))
        _pbtn(toolbar, "Refresh",    lambda: self._load(tree))

        style = ttk.Style(win)
        style.theme_use("clam")
        style.configure("Dark.Treeview",
                        background=CONSOLE_BG, fieldbackground=CONSOLE_BG,
                        foreground=FG, rowheight=22, font=FONT_MONO)
        style.configure("Dark.Treeview.Heading",
                        background=HEADER_BG, foreground=ACCENT, font=FONT_SMALL)
        style.map("Dark.Treeview", background=[("selected", BTN_BG)])

        cols = ("key", "value")
        tree = ttk.Treeview(win, columns=cols, show="tree headings",
                            style="Dark.Treeview")
        tree.heading("key",   text="Key")
        tree.heading("value", text="Value")
        tree.column("#0",     width=20,  stretch=False)
        tree.column("key",    width=280, anchor="w")
        tree.column("value",  width=440, anchor="w")
        sb = ttk.Scrollbar(win, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        tree.pack(fill="both", expand=True, padx=4, pady=4)
        self._load(tree)

    def _load(self, tree: ttk.Treeview) -> None:
        for i in tree.get_children():
            tree.delete(i)
        data_dir = ROOT / "data" / "raw"
        if not data_dir.exists():
            tree.insert("", "end", values=("No data/raw/ directory", ""))
            return
        for session_dir in sorted(data_dir.iterdir()):
            json_files = list(session_dir.glob("*.json"))
            if not json_files:
                continue
            pid_node = tree.insert("", "end", values=(session_dir.name, ""))
            for jf in json_files:
                try:
                    with open(jf) as f:
                        data = json.load(f)
                    file_node = tree.insert(pid_node, "end", values=(jf.name, ""))
                    self._populate(tree, file_node, data)
                except Exception as e:
                    tree.insert(pid_node, "end", values=(jf.name, str(e)))

    def _populate(self, tree: ttk.Treeview, parent: str, data) -> None:
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, (dict, list)):
                    node = tree.insert(parent, "end", values=(k, ""))
                    self._populate(tree, node, v)
                else:
                    tree.insert(parent, "end", values=(k, str(v)))
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, (dict, list)):
                    node = tree.insert(parent, "end", values=(f"[{i}]", ""))
                    self._populate(tree, node, item)
                else:
                    tree.insert(parent, "end", values=(f"[{i}]", str(item)))

    def _expand_all(self, tree: ttk.Treeview) -> None:
        def _expand(node):
            tree.item(node, open=True)
            for child in tree.get_children(node):
                _expand(child)
        for node in tree.get_children():
            _expand(node)


# ---------------------------------------------------------------------------
# Tools list and separators
# ---------------------------------------------------------------------------

TOOLS: list[BaseTool] = [
    HomeTool(),
    PreviewTool(),
    RecordTool(),
    RecalibrateTool(),
    AnalyseTool(),
    PlotTool(),
    ExportTool(),
    SummaryTool(),
    TableTool(),
]

TOOL_SEPARATORS = {3, 4}


# ---------------------------------------------------------------------------
# StudioApp
# ---------------------------------------------------------------------------

class StudioApp:

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Piano Finger Tracking Studio")
        self.root.configure(bg=BG)
        self.root.minsize(900, 600)

        # State
        self._cap         = None
        self._active_tool: BaseTool | None = None
        self._tool_btns:   list[tk.Button] = []
        self._console_open = False
        self._console_h    = CONSOLE_H

        # Shared stat vars (updated by camera loop)
        self.fps_var   = tk.StringVar(value="—")
        self.infer_var = tk.StringVar(value="—")
        self._status_var = tk.StringVar(value="Ready")
        self._fps_times: list[float] = []
        self._img_id    = None
        self._photo_ref = None

        # Letterbox display params — updated each frame for canvas→frame coord mapping
        self._disp_scale = 1.0
        self._disp_x0    = 0
        self._disp_y0    = 0
        self._placeholder_tool: BaseTool | None = None

        self._build_ui()
        self._open_camera()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._select_tool(0)
        self.root.after(33, self._camera_loop)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.root.rowconfigure(0, weight=0)  # title
        self.root.rowconfigure(1, weight=1)  # main area
        self.root.rowconfigure(2, weight=0)  # console toggle
        self.root.rowconfigure(3, weight=0)  # status bar
        self.root.columnconfigure(0, weight=1)

        self._build_title()
        self._build_main()
        self._build_console_toggle()
        self._build_status_bar()

    def _build_title(self) -> None:
        bar = tk.Frame(self.root, bg=HEADER_BG, height=TITLE_H)
        bar.pack(fill="x")
        bar.pack_propagate(False)
        # Gold accent line
        tk.Frame(bar, bg=ACCENT, height=2).pack(fill="x", side="bottom")
        tk.Label(bar, text="🎹  Piano Finger Tracking Studio",
                 bg=HEADER_BG, fg=ACCENT, font=("Segoe UI", 13, "bold"),
                 pady=10).pack(side="left", padx=16)

    def _build_main(self) -> None:
        main = tk.Frame(self.root, bg=BG)
        main.pack(fill="both", expand=True)
        main.rowconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)

        # Left toolbar
        self._toolbar = tk.Frame(main, bg=TOOLBAR_BG, width=TOOLBAR_W)
        self._toolbar.pack(side="left", fill="y")
        self._toolbar.pack_propagate(False)
        self._build_toolbar()

        # Right props panel
        self._props_outer = tk.Frame(main, bg=PANEL_BG, width=PROPS_W)
        self._props_outer.pack(side="right", fill="y")
        self._props_outer.pack_propagate(False)
        self._build_props_header()

        # Viewport (centre)
        self._viewport = tk.Canvas(main, bg="#000000", highlightthickness=0)
        self._viewport.pack(fill="both", expand=True)
        self._img_id = self._viewport.create_image(0, 0, anchor="nw")
        self._show_placeholder("🏠", "Home")

        # Canvas mouse → mask interaction
        self._viewport.bind("<ButtonPress-1>",   self._on_canvas_btn)
        self._viewport.bind("<ButtonRelease-1>", self._on_canvas_release)
        self._viewport.bind("<B1-Motion>",       self._on_canvas_motion)
        self._viewport.bind("<MouseWheel>",      self._on_canvas_scroll)
        self._viewport.bind("<Configure>",       self._on_viewport_resize)

        # Key bindings (N, V, brackets, etc.) for Preview mask editing
        self.root.bind("<n>", lambda e: self._on_preview_key("n"))
        self.root.bind("<N>", lambda e: self._on_preview_key("n"))
        self.root.bind("<v>", lambda e: self._on_preview_key("v"))
        self.root.bind("<V>", lambda e: self._on_preview_key("v"))
        self.root.bind("<bracketleft>",  lambda e: self._on_preview_key("bracketleft"))
        self.root.bind("<bracketright>", lambda e: self._on_preview_key("bracketright"))
        self.root.bind("<minus>",  lambda e: self._on_preview_key("minus"))
        self.root.bind("<equal>",  lambda e: self._on_preview_key("equal"))
        self.root.bind("<s>",      lambda e: self._on_preview_key("s"))
        self.root.bind("<S>",      lambda e: self._on_preview_key("s"))
        self.root.bind("<Escape>", lambda e: None)

    def _build_toolbar(self) -> None:
        tk.Frame(self._toolbar, bg=TOOLBAR_BG, height=8).pack()
        for i, tool in enumerate(TOOLS):
            if i in TOOL_SEPARATORS:
                tk.Frame(self._toolbar, bg=BORDER, height=1).pack(fill="x", padx=8, pady=4)
            btn = tk.Button(
                self._toolbar, text=tool.icon,
                command=lambda idx=i: self._select_tool(idx),
                bg=TOOLBAR_BG, fg=FG, activebackground=ACCENT2,
                activeforeground=BG, relief="flat",
                font=FONT_TOOL, width=3, height=1,
                cursor="hand2"
            )
            btn.pack(pady=2)
            _add_tooltip(btn, tool.name)
            self._tool_btns.append(btn)

    def _build_props_header(self) -> None:
        hdr = tk.Frame(self._props_outer, bg=HEADER_BG, height=32)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        self._props_title_var = tk.StringVar(value="Properties")
        tk.Label(hdr, textvariable=self._props_title_var,
                 bg=HEADER_BG, fg=ACCENT, font=FONT_LABEL,
                 anchor="w").pack(fill="x", padx=10, pady=6)
        # Scrollable props area
        self._props_canvas = tk.Canvas(self._props_outer, bg=PANEL_BG,
                                       highlightthickness=0)
        sb = tk.Scrollbar(self._props_outer, orient="vertical",
                          command=self._props_canvas.yview)
        self._props_canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self._props_canvas.pack(fill="both", expand=True)
        self._props_frame = tk.Frame(self._props_canvas, bg=PANEL_BG)
        self._props_window = self._props_canvas.create_window(
            (0, 0), window=self._props_frame, anchor="nw"
        )
        self._props_frame.bind("<Configure>", self._on_props_resize)
        self._props_canvas.bind("<Configure>", self._on_props_canvas_resize)
        self._props_canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _on_props_resize(self, event) -> None:
        self._props_canvas.configure(scrollregion=self._props_canvas.bbox("all"))

    def _on_props_canvas_resize(self, event) -> None:
        self._props_canvas.itemconfig(self._props_window, width=event.width)

    def _on_mousewheel(self, event) -> None:
        self._props_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _build_console_toggle(self) -> None:
        self._console_frame = tk.Frame(self.root, bg=CONSOLE_BG)
        # Toggle tab
        tab_bar = tk.Frame(self.root, bg=STATUS_BG, height=20)
        tab_bar.pack(fill="x")
        tab_bar.pack_propagate(False)
        self._console_tab = tk.Button(
            tab_bar, text="▲  CONSOLE",
            command=self._toggle_console,
            bg=STATUS_BG, fg=FG_DIM, activebackground=HEADER_BG,
            activeforeground=FG, relief="flat",
            font=("Segoe UI", 8), cursor="hand2"
        )
        self._console_tab.pack(side="left", padx=8)

        # Console text (not packed yet — collapsed by default)
        self._console_txt = scrolledtext.ScrolledText(
            self._console_frame, bg=CONSOLE_BG, fg=FG,
            font=FONT_MONO, relief="flat", height=8,
            state="disabled"
        )
        self._console_txt.pack(fill="both", expand=True, padx=2, pady=2)

    def _build_status_bar(self) -> None:
        bar = tk.Frame(self.root, bg=STATUS_BG, height=STATUS_H)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)
        tk.Label(bar, textvariable=self._status_var,
                 bg=STATUS_BG, fg=FG_DIM, font=FONT_SMALL,
                 anchor="w").pack(side="left", padx=10)
        self._fps_status = tk.Label(bar, textvariable=self.fps_var,
                                    bg=STATUS_BG, fg=FG_DIM, font=FONT_SMALL)
        self._fps_status.pack(side="right", padx=10)
        tk.Label(bar, text="FPS:", bg=STATUS_BG, fg=FG_DIM,
                 font=FONT_SMALL).pack(side="right")

    # ------------------------------------------------------------------
    # Tool switching
    # ------------------------------------------------------------------

    def _select_tool(self, index: int) -> None:
        tool = TOOLS[index]
        if self._active_tool is tool:
            return
        # Deactivate previous
        if self._active_tool:
            self._active_tool.deactivate(self)
        self._active_tool = tool
        # Update toolbar highlights
        for i, btn in enumerate(self._tool_btns):
            if i == index:
                btn.config(bg=ACCENT, fg=BG)
            else:
                btn.config(bg=TOOLBAR_BG, fg=FG)
        # Rebuild properties panel
        for w in self._props_frame.winfo_children():
            w.destroy()
        self._props_title_var.set(tool.name)
        tool.build_properties(self._props_frame, self)
        # Activate new tool
        tool.activate(self)
        # Viewport
        if not tool.uses_camera:
            self._show_placeholder(tool.icon, tool.name)
        self.set_status(tool.name)

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------

    def _open_camera(self) -> None:
        try:
            self._cap = init_camera(cfg)
        except Exception:
            self._cap = cv2.VideoCapture(cfg.CAMERA_INDEX, cv2.CAP_DSHOW)

    def release_camera(self) -> None:
        """Hand the camera to a subprocess — release our capture handle."""
        if self._cap:
            self._cap.release()
            self._cap = None

    def reopen_camera(self) -> None:
        """Reclaim the camera after a subprocess has finished with it."""
        if not self._cap or not self._cap.isOpened():
            self._open_camera()

    def _camera_loop(self) -> None:
        tool = self._active_tool
        if tool and tool.uses_camera and self._cap and self._cap.isOpened():
            t0 = time.perf_counter()
            ret, frame = self._cap.read()
            if ret:
                infer_t0 = time.perf_counter()
                frame = tool.process_frame(frame)
                infer_ms = (time.perf_counter() - infer_t0) * 1000
                self.infer_var.set(f"{infer_ms:.1f}")
                self._display_frame(frame)
                # FPS rolling average
                now = time.perf_counter()
                self._fps_times.append(now)
                self._fps_times = [t for t in self._fps_times if now - t < 1.0]
                self.fps_var.set(f"{len(self._fps_times)}")
        self.root.after(33, self._camera_loop)

    def _display_frame(self, frame: np.ndarray) -> None:
        try:
            from PIL import Image, ImageTk
        except ImportError:
            return
        cw = self._viewport.winfo_width()
        ch = self._viewport.winfo_height()
        if cw < 2 or ch < 2:
            return
        fh, fw = frame.shape[:2]
        scale = min(cw / fw, ch / fh)
        nw = int(fw * scale)
        nh = int(fh * scale)
        x0 = (cw - nw) // 2
        y0 = (ch - nh) // 2
        # Store for canvas→frame coord conversion
        self._disp_scale = scale
        self._disp_x0    = x0
        self._disp_y0    = y0
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        photo   = ImageTk.PhotoImage(Image.fromarray(rgb))
        self._viewport.itemconfig(self._img_id, image=photo)
        self._viewport.coords(self._img_id, x0, y0)
        self._photo_ref = photo

    def _show_placeholder(self, icon: str, label: str) -> None:
        self._placeholder_tool = self._active_tool
        cw = self._viewport.winfo_width()  or 600
        ch = self._viewport.winfo_height() or 400
        # Clear camera image
        self._viewport.itemconfig(self._img_id, image="")
        self._photo_ref = None
        # Delete previous placeholder items
        self._viewport.delete("placeholder")
        # Dark fill rectangle so it's never just black
        self._viewport.create_rectangle(
            0, 0, cw, ch, fill="#0A1520", outline="", tags="placeholder"
        )
        # Thin accent border
        self._viewport.create_rectangle(
            2, 2, cw - 2, ch - 2, outline=BORDER, fill="", tags="placeholder"
        )
        # Plain text icon (avoid emoji font issues on Windows Canvas)
        self._viewport.create_text(
            cw // 2, ch // 2 - 28,
            text=label,
            fill="#8AABB0", font=("Segoe UI", 28, "bold"),
            tags="placeholder"
        )
        self._viewport.create_text(
            cw // 2, ch // 2 + 20,
            text="Camera paused",
            fill=FG_DIM, font=("Segoe UI", 11),
            tags="placeholder"
        )

    def _on_viewport_resize(self, event=None) -> None:
        """Redraw placeholder when viewport is resized and no camera is running."""
        if self._placeholder_tool is self._active_tool and self._active_tool is not None:
            if not self._active_tool.uses_camera:
                self._show_placeholder("", self._active_tool.name)

    # ------------------------------------------------------------------
    # Canvas → frame coordinate conversion + mouse routing
    # ------------------------------------------------------------------

    def _canvas_to_frame(self, cx: int, cy: int):
        """Convert canvas pixel coords to frame pixel coords."""
        if self._disp_scale == 0:
            return cx, cy
        fx = int((cx - self._disp_x0) / self._disp_scale)
        fy = int((cy - self._disp_y0) / self._disp_scale)
        return fx, fy

    def _mask_tool(self):
        """Return active tool if it supports mask interaction, else None."""
        t = self._active_tool
        if isinstance(t, PreviewTool):
            return t
        return None

    def _on_canvas_btn(self, event) -> None:
        fx, fy = self._canvas_to_frame(event.x, event.y)
        t = self._mask_tool()
        if t:
            t.on_canvas_mouse(cv2.EVENT_LBUTTONDOWN, fx, fy)

    def _on_canvas_release(self, event) -> None:
        fx, fy = self._canvas_to_frame(event.x, event.y)
        t = self._mask_tool()
        if t:
            t.on_canvas_mouse(cv2.EVENT_LBUTTONUP, fx, fy)

    def _on_canvas_motion(self, event) -> None:
        fx, fy = self._canvas_to_frame(event.x, event.y)
        t = self._mask_tool()
        if t:
            t.on_canvas_mouse(cv2.EVENT_MOUSEMOVE, fx, fy)

    def _on_canvas_scroll(self, event) -> None:
        fx, fy = self._canvas_to_frame(event.x, event.y)
        t = self._mask_tool()
        if t:
            t.on_canvas_scroll(event.delta, fx, fy)

    def _on_preview_key(self, keysym: str) -> None:
        t = self._mask_tool()
        if not t:
            return
        if keysym == "n":
            t.toggle_mask_panel()
        else:
            t.on_key(keysym)

    # ------------------------------------------------------------------
    # Window close
    # ------------------------------------------------------------------

    def _on_close(self) -> None:
        try:
            if self._active_tool:
                self._active_tool.deactivate(self)
        except Exception:
            pass
        try:
            for tool in TOOLS:
                if hasattr(tool, "_mm") and tool._mm:
                    tool._mm.close_all()
                if hasattr(tool, "_mask_panel") and tool._mask_panel and tool._mask_panel.visible:
                    tool._mask_panel.toggle()
        except Exception:
            pass
        try:
            if self._cap:
                self._cap.release()
        except Exception:
            pass
        self.root.destroy()

    # ------------------------------------------------------------------
    # Console
    # ------------------------------------------------------------------

    def _toggle_console(self) -> None:
        if self._console_open:
            self._console_frame.pack_forget()
            self._console_tab.config(text="▲  CONSOLE")
        else:
            self._console_frame.pack(fill="x", before=self.root.nametowidget(
                self._console_tab.winfo_parent()))
            self._console_frame.config(height=self._console_h)
            self._console_tab.config(text="▼  CONSOLE")
        self._console_open = not self._console_open

    def console_expand(self) -> None:
        if not self._console_open:
            self._toggle_console()

    def console_write(self, text: str) -> None:
        def _do():
            self._console_txt.config(state="normal")
            self._console_txt.insert("end", text + "\n")
            self._console_txt.see("end")
            self._console_txt.config(state="disabled")
        self.root.after(0, _do)

    # ------------------------------------------------------------------
    # Status bar
    # ------------------------------------------------------------------

    def set_status(self, msg: str) -> None:
        self._status_var.set(msg)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        try:
            self.root.mainloop()
        finally:
            if self._active_tool:
                self._active_tool.deactivate(self)
            for tool in TOOLS:
                if hasattr(tool, "_mm") and tool._mm:
                    try:
                        tool._mm.close_all()
                    except Exception:
                        pass
            if self._cap:
                self._cap.release()


# ---------------------------------------------------------------------------
# Tooltip helper
# ---------------------------------------------------------------------------

def _add_tooltip(widget: tk.Widget, text: str) -> None:
    tip = None

    def _enter(event):
        nonlocal tip
        x = widget.winfo_rootx() + widget.winfo_width() + 4
        y = widget.winfo_rooty() + widget.winfo_height() // 2
        tip = tk.Toplevel(widget)
        tip.wm_overrideredirect(True)
        tip.wm_geometry(f"+{x}+{y}")
        tk.Label(tip, text=text, bg=ACCENT, fg=BG,
                 font=FONT_SMALL, padx=6, pady=2).pack()

    def _leave(event):
        nonlocal tip
        if tip:
            tip.destroy()
            tip = None

    widget.bind("<Enter>", _enter)
    widget.bind("<Leave>", _leave)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.chdir(str(ROOT))
    app = StudioApp()
    app.run()
