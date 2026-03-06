"""key_calibration.py
Interactive piano keyboard overlay for MJMPE key-centre calibration.

Drag any of the 8 handles to resize (Photoshop-style), drag the interior to move.
Press ENTER to save key centre coordinates to data/calibration/key_centers.json.

Controls:
  Drag interior / top   – move keyboard overlay
  Drag left / right     – resize key width (opposite edge stays fixed)
  Drag bottom edge      – resize key height
  Drag any corner ◼     – resize width + height simultaneously
  Scroll up / down      – add / remove a white key on the right
  [ / ]                 – shift starting note one white key down / up
  - / =                 – shrink / grow white key width (fine, 1 px)
  H                     – toggle help
  ENTER or S            – save calibration
  ESC                   – quit without saving
"""

import json
import math
import os
import time
from pathlib import Path

import cv2
import numpy as np

try:
    from . import config
    from .camera_setup import init_camera
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import config
    from camera_setup import init_camera

# ---------------------------------------------------------------------------
# Piano constants
# ---------------------------------------------------------------------------

_NOTE_NAMES  = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
_WHITE_SEMIS = {0, 2, 4, 5, 7, 9, 11}

_CAL_DIR  = Path(__file__).parent.parent / "data" / "calibration"
_WIN_NAME = "Piano Key Calibration"


def _midi_to_name(midi: int) -> str:
    return _NOTE_NAMES[midi % 12] + str(midi // 12 - 1)


# ---------------------------------------------------------------------------
# Key layout builder  (exported — used by analysis scripts)
# ---------------------------------------------------------------------------

def build_key_layout(start_midi: int, num_white: int,
                     ox: int, oy: int, wkw: int, wkh: int) -> list:
    """
    Compute pixel rects and centre positions for every visible key.

    White key centre: 82 % down key height (finger-strike zone).
    Black key centre: vertical mid-point of the black key body.

    Returns list of dicts:
      midi_note, note_name, key_type, rect=(x,y,w,h), center=(cx,cy)
    """
    bkw = max(4, int(wkw * 0.55))
    bkh = int(wkh * 0.62)
    keys  = []
    white = 0

    for midi in range(start_midi, 128):
        semi = midi % 12
        name = _midi_to_name(midi)
        if semi in _WHITE_SEMIS:
            if white >= num_white:
                break
            x  = ox + white * wkw
            keys.append({'midi_note': midi, 'note_name': name,
                         'key_type': 'white',
                         'rect':   (x, oy, wkw, wkh),
                         'center': (x + wkw // 2, oy + int(wkh * 0.82))})
            white += 1
        else:
            if 0 < white < num_white:
                cx = ox + white * wkw
                keys.append({'midi_note': midi, 'note_name': name,
                             'key_type': 'black',
                             'rect':   (cx - bkw // 2, oy, bkw, bkh),
                             'center': (cx, oy + bkh // 2)})
    return keys


# ---------------------------------------------------------------------------
# KeyMask  (exported — used by record.py)
# ---------------------------------------------------------------------------

class KeyMask:
    """
    Encapsulates piano key overlay state, 8-handle drag-resize, and rendering.

    Keys list is rebuilt lazily (only when layout parameters change).
    draw() operates only on the keyboard bounding-box ROI for performance.
    """

    EDGE_TOL = 12   # pixels from edge that triggers a resize handle

    CORNER_TOL = 20   # px — click radius that triggers a corner warp handle

    def __init__(self, ox: int, oy: int, wkw: int, wkh: int,
                 num_white: int = 29, start_midi: int = 48,
                 corners: np.ndarray = None):
        self.ox          = ox
        self.oy          = oy
        self.wkw         = wkw
        self.wkh         = wkh
        self.num_white   = num_white
        self.start_midi  = start_midi
        self._keys       = None
        self._dirty      = True
        # Perspective warp — 4 corners in screen space (TL, TR, BR, BL)
        if corners is not None:
            self.corners = np.array(corners, dtype=np.float32)
        else:
            self._init_corners()
        self._H       = None    # cached homography
        self._H_dirty = True
        # drag state
        self._drag       = None
        self._mx0 = self._my0 = 0
        self._ox0 = self._oy0 = 0
        self._wkw0 = self._wkh0 = 0
        self._corners0: np.ndarray = None   # snapshot at drag-start

    # -- Key layout (lazy) --

    @property
    def keys(self) -> list:
        if self._dirty or self._H_dirty:
            flat = build_key_layout(self.start_midi, self.num_white,
                                    self.ox, self.oy, self.wkw, self.wkh)
            H  = self._homography()   # resets _H_dirty
            # Warp every key centre through H
            flat_pts = np.array([[k['center']] for k in flat], dtype=np.float32)
            warped   = cv2.perspectiveTransform(flat_pts, H).reshape(-1, 2)
            for k, (cx, cy) in zip(flat, warped):
                k['center'] = (int(round(cx)), int(round(cy)))
            # Warp each key rect into a polygon (for pointPolygonTest in test.py)
            for k in flat:
                x, y, w, h = k['rect']
                src_pts = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]],
                                   dtype=np.float32).reshape(4, 1, 2)
                k['polygon'] = cv2.perspectiveTransform(src_pts, H).reshape(4, 2).astype(np.int32)
            self._keys  = flat
            self._dirty = False
        return self._keys

    def mark_dirty(self) -> None:
        self._dirty   = True
        self._H_dirty = True
        self._H       = None

    # -- Perspective warp helpers --

    def _init_corners(self) -> None:
        """Set corners to the current rectangular bounding box."""
        total_w = self.num_white * self.wkw
        self.corners = np.array([
            [self.ox,           self.oy          ],
            [self.ox + total_w, self.oy          ],
            [self.ox + total_w, self.oy + self.wkh],
            [self.ox,           self.oy + self.wkh],
        ], dtype=np.float32)

    def reset_warp(self) -> None:
        """Snap corners back to a perfect rectangle (public API for V key / button)."""
        self._init_corners()
        self._H_dirty = True
        self._H       = None
        self._dirty   = True

    def _homography(self) -> np.ndarray:
        """Return the cached perspective matrix (flat layout → warped corners)."""
        if self._H is None or self._H_dirty:
            total_w = self.num_white * self.wkw
            src = np.array([
                [self.ox,           self.oy          ],
                [self.ox + total_w, self.oy          ],
                [self.ox + total_w, self.oy + self.wkh],
                [self.ox,           self.oy + self.wkh],
            ], dtype=np.float32)
            self._H       = cv2.getPerspectiveTransform(src, self.corners)
            self._H_dirty = False
        return self._H

    # -- Drawing --

    def draw(self, frame: np.ndarray, alpha: float = None) -> None:
        """
        Blend perspective-warped keyboard overlay onto frame in-place.
        Operates on the bounding-box ROI of the warped quad for performance.
        """
        if alpha is None:
            alpha = config.MASK_ALPHA

        fh, fw = frame.shape[:2]
        H = self._homography()

        # Bounding box of the warped quad
        bx1 = max(0, int(np.floor(self.corners[:, 0].min())))
        by1 = max(0, int(np.floor(self.corners[:, 1].min())))
        bx2 = min(fw, int(np.ceil(self.corners[:, 0].max())) + 1)
        by2 = min(fh, int(np.ceil(self.corners[:, 1].max())) + 1)
        if bx2 <= bx1 or by2 <= by1:
            return

        region = frame[by1:by2, bx1:bx2]
        ov     = region.copy()
        off    = np.array([bx1, by1])

        def _warp_pts(x, y, w, h) -> np.ndarray:
            """Perspective-transform a flat key rect into a warped polygon (int32)."""
            src = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]],
                           dtype=np.float32).reshape(4, 1, 2)
            dst = cv2.perspectiveTransform(src, H).reshape(4, 2)
            return (dst - off).astype(np.int32)

        # White keys first (behind black)
        for k in self.keys:
            if k['key_type'] == 'white':
                pts = _warp_pts(*k['rect'])
                cv2.fillPoly(ov,   [pts], config.MASK_WHITE_FILL)
                cv2.polylines(ov,  [pts], True, config.MASK_WHITE_BORDER, 2)
                if k['rect'][2] >= 16:
                    semi  = k['midi_note'] % 12
                    label = k['note_name'] if semi == 0 else _NOTE_NAMES[semi]
                    col   = config.MASK_C_NOTE_LABEL if semi == 0 else (70, 70, 70)
                    fs    = 0.33 if k['rect'][2] >= 22 else 0.26
                    # Place label near the warped bottom-left corner of the key
                    lx = int(pts[3][0]) + 2
                    ly = int(pts[3][1]) - 4
                    cv2.putText(ov, label, (lx, ly),
                                cv2.FONT_HERSHEY_SIMPLEX, fs, col, 1)

        # Black keys on top
        for k in self.keys:
            if k['key_type'] == 'black':
                pts = _warp_pts(*k['rect'])
                cv2.fillPoly(ov,  [pts], config.MASK_BLACK_FILL)
                cv2.polylines(ov, [pts], True, config.MASK_BLACK_BORDER, 2)

        # Blend ROI in-place
        cv2.addWeighted(ov, alpha, region, 1.0 - alpha, 0, region)

        # Centre dots: drawn on full frame, fully opaque (centres already warped)
        for k in self.keys:
            cx, cy = k['center']
            col = config.MASK_DOT_WHITE if k['key_type'] == 'white' else config.MASK_DOT_BLACK
            cv2.circle(frame, (cx, cy), 4, col, -1)
            cv2.circle(frame, (cx, cy), 4, (255, 255, 255), 1)

    # -- Mouse interaction --

    def on_mouse(self, event: int, x: int, y: int, flags: int) -> bool:
        """
        Wire to cv2.setMouseCallback.
        Returns True the moment a drag ends (caller should auto-save).
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            dtype = self._hit_test(x, y)
            if dtype:
                self._drag = dtype
                self._mx0, self._my0   = x, y
                self._ox0, self._oy0   = self.ox, self.oy
                self._wkw0, self._wkh0 = self.wkw, self.wkh
                self._corners0         = self.corners.copy()

        elif event == cv2.EVENT_LBUTTONUP:
            ended      = self._drag is not None
            self._drag = None
            return ended

        elif event == cv2.EVENT_MOUSEMOVE and self._drag:
            self._apply_drag(x - self._mx0, y - self._my0)

        elif event == cv2.EVENT_MOUSEWHEEL:
            self.num_white = max(4, min(52, self.num_white + (1 if flags > 0 else -1)))
            self._init_corners()    # reset warp when key count changes via scroll
            self.mark_dirty()

        return False

    def _hit_test(self, x: int, y: int):
        """Return drag-type string or None if outside overlay."""
        # 1. Corner warp handles — highest priority
        for i, (cx, cy) in enumerate(self.corners):
            if math.hypot(x - cx, y - cy) <= self.CORNER_TOL:
                return f'corner_{i}'

        # 2. Move / edge-resize — based on rectangular bounding box
        right  = self.ox + self.num_white * self.wkw
        bottom = self.oy + self.wkh
        t      = self.EDGE_TOL

        if not (self.ox - t <= x <= right + t and self.oy - t <= y <= bottom + t):
            return None

        nl = x <= self.ox + t
        nr = x >= right  - t
        nt = y <= self.oy + t
        nb = y >= bottom - t

        # Edges (corners now reserved for perspective drag, not resize)
        if nt: return 't'
        if nb: return 'b'
        if nl: return 'l'
        if nr: return 'r'
        return 'move'

    def _apply_drag(self, dx: int, dy: int) -> None:
        t = self._drag

        # -- Perspective corner warp --
        if t.startswith('corner_'):
            i = int(t[7:])
            self.corners[i] = self._corners0[i] + np.array([dx, dy], dtype=np.float32)
            self._H_dirty = True
            self._dirty   = True
            return

        right0  = self._ox0 + self.num_white * self._wkw0
        bottom0 = self._oy0 + self._wkh0

        if t == 'move':
            # Translate both rect origin and all 4 corners
            self.ox      = self._ox0 + dx
            self.oy      = self._oy0 + dy
            self.corners = self._corners0 + np.array([dx, dy], dtype=np.float32)
            self._H_dirty = True
        else:
            # Edge resize — resets warp to rectangle
            if t in ('t', 'tl', 'tr'):
                new_wkh  = max(20, bottom0 - (self._oy0 + dy))
                self.oy  = bottom0 - new_wkh
                self.wkh = new_wkh

            if t in ('b', 'bl', 'br'):
                self.wkh = max(20, self._wkh0 + dy)

            if t in ('l', 'tl', 'bl'):
                new_wkw  = max(6.0, (right0 - self._ox0 - dx) / self.num_white)
                self.ox  = round(right0 - self.num_white * new_wkw)
                self.wkw = max(6, round(new_wkw))

            if t in ('r', 'tr', 'br'):
                new_right = right0 + dx
                self.wkw  = max(6, round((new_right - self._ox0) / self.num_white))

            self._init_corners()    # reset warp on resize

        self.mark_dirty()

    # -- Persistence --

    def save(self, cal_dir: Path, frame_w: int, frame_h: int) -> Path:
        cal_dir = Path(cal_dir)
        cal_dir.mkdir(parents=True, exist_ok=True)
        out = {
            "frame_width":         frame_w,
            "frame_height":        frame_h,
            "start_midi":          self.start_midi,
            "num_white_keys":      self.num_white,
            "white_key_width_px":  self.wkw,
            "white_key_height_px": self.wkh,
            "overlay_x":           self.ox,
            "overlay_y":           self.oy,
            "corners": self.corners.tolist(),   # [[x,y], …] TL TR BR BL
            "keys": [
                {"midi_note": k['midi_note'], "note_name": k['note_name'],
                 "key_type":  k['key_type'],
                 "center_x":  k['center'][0], "center_y": k['center'][1]}
                for k in self.keys
            ],
        }
        path = cal_dir / "key_centers.json"
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        return path

    @classmethod
    def load(cls, path) -> "KeyMask":
        with open(path) as f:
            c = json.load(f)
        corners = np.array(c['corners'], dtype=np.float32) if 'corners' in c else None
        return cls(c['overlay_x'], c['overlay_y'],
                   c['white_key_width_px'], c['white_key_height_px'],
                   c['num_white_keys'], c['start_midi'],
                   corners=corners)

    @classmethod
    def default(cls, frame_w: int, frame_h: int) -> "KeyMask":
        """Default 29-key layout sized to fill ~3/4 of the frame width."""
        wkw = max(10, (frame_w * 3 // 4) // 29)
        return cls(frame_w // 8, int(frame_h * 0.52), wkw, int(frame_h * 0.28))


# ---------------------------------------------------------------------------
# Handle drawing  (exported — used by record.py)
# ---------------------------------------------------------------------------

def draw_mask_handles(frame: np.ndarray, mask: KeyMask) -> None:
    """
    Draw the warped-quad outline, 4 corner warp handles (circles), and
    4 edge-midpoint resize handles (squares) on the mask.
    """
    # Warped quad outline
    corners_i = mask.corners.astype(np.int32)
    cv2.polylines(frame, [corners_i], True, (180, 180, 180), 1)

    # 4 corner circles — warp handles (white fill, black outline)
    cr = 9   # circle radius
    for cx, cy in corners_i:
        cv2.circle(frame, (cx, cy), cr + 1, (0, 0, 0),       -1)
        cv2.circle(frame, (cx, cy), cr,     (255, 255, 255),  -1)

    # 4 edge-midpoint squares — resize handles (white fill, black outline)
    ox, oy = mask.ox, mask.oy
    right  = ox + mask.num_white * mask.wkw
    bottom = oy + mask.wkh
    mid_y  = (oy + bottom) // 2
    mid_x  = (ox + right)  // 2
    ks     = 5  # half-size of handle square

    for hx, hy in [
        (mid_x, oy),      # top edge
        (ox,    mid_y),   # left edge
        (right, mid_y),   # right edge
        (mid_x, bottom),  # bottom edge
    ]:
        cv2.rectangle(frame, (hx - ks - 1, hy - ks - 1),
                      (hx + ks + 1, hy + ks + 1), (0, 0, 0),         -1)
        cv2.rectangle(frame, (hx - ks,     hy - ks),
                      (hx + ks,     hy + ks),     (255, 255, 255),    -1)


# ---------------------------------------------------------------------------
# Mask Control Panel  (exported — used by record.py)
# ---------------------------------------------------------------------------

class MaskControlPanel:
    """
    Floating OpenCV window with a controls reference table (left half)
    and interactive buttons (right half) that act on a KeyMask in real time.
    Toggle visibility with N in record.py.
    """

    WIN_NAME = "Mask Controls"
    PW, PH   = 640, 480

    _TABLE = [
        ("Drag interior",     "Move whole keyboard"),
        ("Drag left edge",    "Resize (right stays fixed)"),
        ("Drag right edge",   "Resize (left stays fixed)"),
        ("Drag bottom edge",  "Change key height"),
        ("Drag corner \u25ef", "Warp perspective"),
        ("Scroll wheel",      "Add / remove a key"),
        ("[ / ]",             "Shift starting note"),
        ("- / =",             "Fine key width"),
        ("V key",             "Reset warp to rectangle"),
    ]

    _MOVE_STEP   = 5
    _WIDTH_STEP  = 1
    _HEIGHT_STEP = 4

    def __init__(self, mask: KeyMask, save_fn=None):
        self._mask           = mask
        self._save_fn        = save_fn   # callable() — persists calibration to disk
        self._visible        = False
        self._btns: list     = []        # [(x1, y1, x2, y2), callback]
        self._save_msg_until = 0.0       # monotonic() deadline for "saved" banner

    # ------------------------------------------------------------------
    # Visibility
    # ------------------------------------------------------------------

    def toggle(self) -> None:
        if self._visible:
            try:
                cv2.destroyWindow(self.WIN_NAME)
            except Exception:
                pass
            self._visible = False
        else:
            cv2.namedWindow(self.WIN_NAME, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(self.WIN_NAME, self._on_click)
            self._visible = True

    @property
    def visible(self) -> bool:
        return self._visible

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render(self) -> None:
        """Redraw and show the panel.  Call every main-loop iteration."""
        if not self._visible:
            return
        # Detect if user closed the window via the X button
        try:
            if cv2.getWindowProperty(self.WIN_NAME, cv2.WND_PROP_VISIBLE) < 1:
                self._visible = False
                return
        except Exception:
            self._visible = False
            return

        panel = np.full((self.PH, self.PW, 3), 18, dtype=np.uint8)
        self._btns = []
        self._draw_table(panel)
        cv2.line(panel, (335, 6), (335, self.PH - 6), (55, 55, 55), 1)
        self._draw_buttons(panel)

        # "Calibration saved!" confirmation banner (shown for 2 s after save)
        if time.monotonic() < self._save_msg_until:
            cv2.rectangle(panel, (0, self.PH - 34), (self.PW, self.PH), (20, 90, 30), -1)
            cv2.putText(panel, "Calibration saved!", (10, self.PH - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 255, 180), 2)

        cv2.imshow(self.WIN_NAME, panel)

    # ------------------------------------------------------------------
    # Mouse
    # ------------------------------------------------------------------

    def _on_click(self, event, x, y, _flags, _):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for (x1, y1, x2, y2), cb in self._btns:
            if x1 <= x <= x2 and y1 <= y <= y2:
                cb()
                break

    def _do_save(self) -> None:
        """Call the save callback and arm the confirmation banner."""
        if self._save_fn:
            self._save_fn()
            self._save_msg_until = time.monotonic() + 2.0

    # ------------------------------------------------------------------
    # Drawing primitives
    # ------------------------------------------------------------------

    def _btn(self, panel, x, y, w, h, label, cb, color=(42, 42, 42)):
        x2, y2 = x + w, y + h
        cv2.rectangle(panel, (x, y), (x2, y2), color, -1)
        cv2.rectangle(panel, (x, y), (x2, y2), (130, 130, 130), 1)
        if label:
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
            cv2.putText(panel, label,
                        (x + (w - tw) // 2, y + (h + th) // 2 - 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1)
        self._btns.append(((x, y, x2, y2), cb))

    @staticmethod
    def _lbl(panel, x, y, text, scale=0.50, color=(160, 160, 160)):
        cv2.putText(panel, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)

    @staticmethod
    def _tri(panel, cx, cy, direction, size=6, color=(220, 220, 220)):
        """Draw a filled arrow triangle (direction: 'u' 'd' 'l' 'r')."""
        if direction == 'u':
            pts = [[cx, cy - size], [cx - size, cy + size // 2], [cx + size, cy + size // 2]]
        elif direction == 'd':
            pts = [[cx, cy + size], [cx - size, cy - size // 2], [cx + size, cy - size // 2]]
        elif direction == 'l':
            pts = [[cx - size, cy], [cx + size // 2, cy - size], [cx + size // 2, cy + size]]
        else:  # 'r'
            pts = [[cx + size, cy], [cx - size // 2, cy - size], [cx - size // 2, cy + size]]
        cv2.fillPoly(panel, [np.array(pts, np.int32)], color)

    # ------------------------------------------------------------------
    # Controls table (left half)
    # ------------------------------------------------------------------

    def _draw_table(self, panel) -> None:
        cv2.putText(panel, "Controls Reference", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 200, 50), 1)
        hdr_y = 30
        cv2.rectangle(panel, (8, hdr_y), (328, hdr_y + 24), (40, 40, 40), -1)
        cv2.putText(panel, "Action", (14, hdr_y + 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (210, 210, 210), 1)
        cv2.putText(panel, "Result", (162, hdr_y + 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (210, 210, 210), 1)
        cv2.line(panel, (158, hdr_y), (158, hdr_y + 24), (70, 70, 70), 1)

        rh = 30
        for i, (act, res) in enumerate(self._TABLE):
            y0 = hdr_y + 24 + i * rh
            cv2.rectangle(panel, (8, y0), (328, y0 + rh),
                          (28, 28, 28) if i % 2 == 0 else (22, 22, 22), -1)
            cv2.putText(panel, act, (14, y0 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.43, (200, 200, 200), 1)
            cv2.putText(panel, res, (162, y0 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.39, (145, 145, 145), 1)
            cv2.line(panel, (158, y0), (158, y0 + rh), (50, 50, 50), 1)
            cv2.line(panel, (8,   y0), (328, y0),      (45, 45, 45), 1)

        cv2.rectangle(panel, (8, hdr_y),
                      (328, hdr_y + 24 + len(self._TABLE) * rh), (70, 70, 70), 1)

    # ------------------------------------------------------------------
    # Buttons (right half)
    # ------------------------------------------------------------------

    def _draw_buttons(self, panel) -> None:
        m   = self._mask
        BW  = 46
        BH  = 30
        rx  = 348
        rw  = self.PW - rx - 8
        cx  = rx + rw // 2   # horizontal centre of right panel
        ws  = self._WIDTH_STEP
        hs  = self._HEIGHT_STEP
        mv  = self._MOVE_STEP
        y   = 16

        # Title
        cv2.putText(panel, "Buttons", (rx, y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 200, 50), 1)
        y += 18

        # ── Move (cross layout) ───────────────────────────────────────
        cv2.rectangle(panel, (rx, y), (self.PW - 4, y + 120), (26, 26, 26), -1)
        cv2.rectangle(panel, (rx, y), (self.PW - 4, y + 120), (55, 55, 55), 1)
        self._lbl(panel, rx + 8, y + 14, "Move", 0.48, (180, 180, 180))

        def _mv(ddx, ddy):
            m.ox += ddx; m.oy += ddy
            m.corners += np.array([ddx, ddy], dtype=np.float32)
            m.mark_dirty()

        ux,  uy  = cx - BW // 2, y + 22      # up
        lx,  ly  = cx - BW - 4,  y + 55      # left
        rx2, ry2 = cx + 4,        y + 55      # right
        dx,  dy  = cx - BW // 2, y + 88      # down

        self._btn(panel, ux,  uy,  BW, BH, "", lambda: _mv(  0, -mv))
        self._tri(panel, ux  + BW // 2, uy  + BH // 2, 'u')
        self._btn(panel, lx,  ly,  BW, BH, "", lambda: _mv(-mv,   0))
        self._tri(panel, lx  + BW // 2, ly  + BH // 2, 'l')
        self._btn(panel, rx2, ry2, BW, BH, "", lambda: _mv( mv,   0))
        self._tri(panel, rx2 + BW // 2, ry2 + BH // 2, 'r')
        self._btn(panel, dx,  dy,  BW, BH, "", lambda: _mv(  0,  mv))
        self._tri(panel, dx  + BW // 2, dy  + BH // 2, 'd')

        y += 126

        # ── Section helper ────────────────────────────────────────────
        def _section(label, minus_cb, plus_cb):
            nonlocal y
            cv2.rectangle(panel, (rx, y), (self.PW - 4, y + 52), (26, 26, 26), -1)
            cv2.rectangle(panel, (rx, y), (self.PW - 4, y + 52), (55, 55, 55), 1)
            self._lbl(panel, rx + 8, y + 16, label, 0.48, (200, 200, 200))
            bx0 = cx - BW - 5
            self._btn(panel, bx0,           y + 20, BW, BH, "-", minus_cb)
            self._btn(panel, bx0 + BW + 10, y + 20, BW, BH, "+", plus_cb)
            y += 56

        # Keys +/-
        _section(
            f"Keys: {m.num_white}",
            lambda: (setattr(m, 'num_white', max(4,  m.num_white - 1)), m.reset_warp()),
            lambda: (setattr(m, 'num_white', min(52, m.num_white + 1)), m.reset_warp()),
        )

        # Start note
        def _shift(delta):
            mi   = m.start_midi + delta
            step = 1 if delta > 0 else -1
            while 0 <= mi <= 127 and mi % 12 not in _WHITE_SEMIS:
                mi += step
            if 0 <= mi <= 127:
                m.start_midi = mi
                m.mark_dirty()

        _section(
            f"Start: {_midi_to_name(m.start_midi)}",
            lambda: _shift(-1),
            lambda: _shift(+1),
        )

        # Key width
        _section(
            f"Width: {m.wkw} px",
            lambda: (setattr(m, 'wkw', max(6, m.wkw - ws)), m.reset_warp()),
            lambda: (setattr(m, 'wkw', m.wkw + ws),         m.reset_warp()),
        )

        # Key height
        _section(
            f"Height: {m.wkh} px",
            lambda: (setattr(m, 'wkh', max(20, m.wkh - hs)), m.reset_warp()),
            lambda: (setattr(m, 'wkh', m.wkh + hs),          m.reset_warp()),
        )

        # Save calibration
        self._btn(panel, rx, y + 4, rw, BH, "SAVE CALIBRATION",
                  self._do_save, color=(20, 90, 30))

        # Reset warp
        def _reset():
            m.reset_warp()
        self._btn(panel, rx, y + 40, rw, BH, "RESET WARP",
                  _reset, color=(100, 55, 10))


# ---------------------------------------------------------------------------
# HUD
# ---------------------------------------------------------------------------

def _draw_hud(frame: np.ndarray, mask: KeyMask, total_keys: int, show_help: bool) -> None:
    fh         = frame.shape[0]
    start_name = _midi_to_name(mask.start_midi)

    status = (f"Keys: {mask.num_white} white / {total_keys} total  |  "
              f"Start: {start_name} (MIDI {mask.start_midi})  |  "
              f"Key: {mask.wkw}x{mask.wkh} px  |  "
              f"Pos: ({mask.ox}, {mask.oy})")
    cv2.putText(frame, status, (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 240, 240), 1)

    if show_help:
        lines = [
            "Drag interior         move keyboard",
            "Drag left/right edge  resize key width",
            "Drag bottom edge      resize key height",
            "Drag corner \u25ef         warp perspective",
            "V                     reset warp to rectangle",
            "Scroll                add / remove key",
            "[ / ]                 shift start note",
            "- / =                 fine key width",
            "ENTER or S            save calibration",
            "ESC                   quit without saving",
        ]
        by = fh - 12 - len(lines) * 20
        for i, l in enumerate(lines):
            cv2.putText(frame, l, (10, by + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (150, 150, 150), 1)
    else:
        cv2.putText(frame, "H: help   ENTER: save   ESC: quit",
                    (10, fh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (150, 150, 150), 1)


# ---------------------------------------------------------------------------
# Main calibration runner
# ---------------------------------------------------------------------------

def run_calibration():
    cap = init_camera(config)

    ret, frame = cap.read()
    if not ret:
        print("ERROR: Could not read from camera.")
        cap.release()
        return

    fh, fw = frame.shape[:2]

    # Load existing calibration or start with default
    cal_file = _CAL_DIR / "key_centers.json"
    if cal_file.exists():
        try:
            mask = KeyMask.load(cal_file)
            print(f"Loaded existing calibration ({mask.num_white} white keys).")
        except Exception as e:
            print(f"Could not load calibration ({e}), using default.")
            mask = KeyMask.default(fw, fh)
    else:
        mask = KeyMask.default(fw, fh)

    # Snap start_midi to a white key (safety)
    while mask.start_midi % 12 not in _WHITE_SEMIS:
        mask.start_midi += 1

    show_help = True

    def on_mouse(event, x, y, flags, _):
        mask.on_mouse(event, x, y, flags)

    cv2.namedWindow(_WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(_WIN_NAME, min(fw, 1400), min(fh, 820))
    cv2.setMouseCallback(_WIN_NAME, on_mouse)

    print("Drag edges / corners to resize, drag interior to move.")
    print("Press H for help, ENTER to save.\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        mask.draw(frame)
        draw_mask_handles(frame, mask)
        _draw_hud(frame, mask, len(mask.keys), show_help)

        cv2.imshow(_WIN_NAME, frame)
        key = cv2.waitKey(20) & 0xFF

        if key == 27:
            print("Quit without saving.")
            break

        elif key in (13, 10):
            path = mask.save(_CAL_DIR, fw, fh)
            print(f"Saved → {path}")
            break

        elif key in (ord('s'), ord('S')):
            path = mask.save(_CAL_DIR, fw, fh)
            print(f"Saved → {path}")

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
            print("Warp reset to rectangle.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_calibration()
