# model_manager.py
# Abstract interface and concrete wrappers for MediaPipe Hands and OpenPose Hand.
# Used by live_preview.py for live model switching.
#
# All models share the same interface:
#   load()           – lazy one-time initialisation (may block on first use)
#   infer(bgr)       – run inference, return model-specific result object
#   draw(frame, res) – draw landmarks on frame in-place, return frame
#   close()          – release resources

import abc
import cv2
import numpy as np


# ---------------------------------------------------------------------------
# OpenPose Hand skeleton — 21 keypoints
# ---------------------------------------------------------------------------
#
# Keypoint index → joint:
#   0: Wrist
#   1-4:   Thumb  (CMC, MCP, IP, Tip)
#   5-8:   Index  (MCP, PIP, DIP, Tip)
#   9-12:  Middle (MCP, PIP, DIP, Tip)
#   13-16: Ring   (MCP, PIP, DIP, Tip)
#   17-20: Pinky  (MCP, PIP, DIP, Tip)

_OPENPOSE_HAND_EDGES = [
    (0, 1),  (1, 2),  (2, 3),  (3, 4),           # thumb
    (0, 5),  (5, 6),  (6, 7),  (7, 8),            # index
    (0, 9),  (9, 10), (10, 11), (11, 12),          # middle
    (0, 13), (13, 14), (14, 15), (15, 16),         # ring
    (0, 17), (17, 18), (18, 19), (19, 20),         # pinky
    (5, 9),  (9, 13), (13, 17),                    # palm knuckle bar
]
# Fingertips highlighted in red
_OPENPOSE_HAND_TIPS = {4, 8, 12, 16, 20}


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class PoseModel(abc.ABC):
    """Common interface every model wrapper must implement."""

    _loaded: bool = False

    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @property
    def loaded(self) -> bool:
        return self._loaded

    @abc.abstractmethod
    def load(self) -> None:
        """One-time initialisation. May block on first use (downloads, disk reads)."""

    @abc.abstractmethod
    def infer(self, bgr: np.ndarray) -> object:
        """Run inference on a BGR frame. Returns a model-specific result object."""

    @abc.abstractmethod
    def draw(self, frame: np.ndarray, result: object) -> np.ndarray:
        """Draw landmarks/skeleton on frame in-place. Returns frame."""

    @abc.abstractmethod
    def close(self) -> None:
        """Release model resources."""


# ---------------------------------------------------------------------------
# MediaPipe Hands
# ---------------------------------------------------------------------------

class MediaPipeHandsModel(PoseModel):
    """Wraps mp.solutions.hands.Hands — 21 hand landmarks per hand."""

    def __init__(self, config):
        self._cfg = config
        self._hands = None
        self._mp_hands = None
        self._mp_drawing = None
        self._landmark_style = None
        self._connection_style = None

    @property
    def name(self) -> str:
        return "MediaPipe Hands"

    def load(self) -> None:
        import mediapipe as mp
        self._mp_hands   = mp.solutions.hands
        self._mp_drawing = mp.solutions.drawing_utils
        mp_styles = mp.solutions.drawing_styles
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self._cfg.MAX_NUM_HANDS,
            model_complexity=self._cfg.MODEL_COMPLEXITY,
            min_detection_confidence=self._cfg.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=self._cfg.MIN_TRACKING_CONFIDENCE,
        )
        self._landmark_style   = mp_styles.get_default_hand_landmarks_style()
        self._connection_style = mp_styles.get_default_hand_connections_style()
        self._loaded = True

    def infer(self, bgr: np.ndarray) -> object:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        return self._hands.process(rgb)

    def draw(self, frame: np.ndarray, result: object) -> np.ndarray:
        if result.multi_hand_landmarks:
            for lm in result.multi_hand_landmarks:
                self._mp_drawing.draw_landmarks(
                    frame, lm,
                    self._mp_hands.HAND_CONNECTIONS,
                    self._landmark_style,
                    self._connection_style,
                )
        return frame

    def close(self) -> None:
        if self._hands:
            self._hands.close()
        self._loaded = False


# ---------------------------------------------------------------------------
# OpenPose Hand (via OpenCV DNN + Caffe model)
# ---------------------------------------------------------------------------

class OpenPoseHandModel(PoseModel):
    """
    Wraps the OpenPose hand model using cv2.dnn.readNetFromCaffe().
    Detects 21 finger keypoints per hand (same joints as MediaPipe Hands).

    Model files are downloaded by download_openpose_model.py:
        models/openpose_hand.prototxt
        models/openpose_hand.caffemodel  (~55 MB)

    infer() returns: list[tuple(x, y, conf) | None] — 21 keypoints in frame pixels.
    """

    def __init__(self, config, project_root: str):
        self._cfg  = config
        self._root = project_root
        self._net  = None

    @property
    def name(self) -> str:
        return "OpenPose Hand"

    def load(self) -> None:
        from pathlib import Path
        root  = Path(self._root)
        proto = root / self._cfg.OPENPOSE_HAND_PROTOTXT
        caffe = root / self._cfg.OPENPOSE_HAND_CAFFEMODEL

        if not caffe.exists():
            raise FileNotFoundError(
                f"\nOpenPose hand caffemodel not found:\n  {caffe}\n"
                f"Run:  python download_openpose_model.py\n"
            )
        if not proto.exists():
            raise FileNotFoundError(
                f"\nOpenPose hand prototxt not found:\n  {proto}\n"
                f"Run:  python download_openpose_model.py\n"
            )

        print("Loading OpenPose Hand from disk (~55 MB, may take 1 s)...")
        self._net = cv2.dnn.readNetFromCaffe(str(proto), str(caffe))

        # Backend: CUDA (RTX) → OpenCL → CPU
        cuda_ok = hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0
        if cuda_ok:
            self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
            print("OpenPose Hand: CUDA FP16 backend")
        else:
            self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
            print("OpenPose Hand: OpenCL/CPU backend")

        self._loaded = True
        print("OpenPose Hand ready.")

    def infer(self, bgr: np.ndarray) -> list:
        h, w  = bgr.shape[:2]
        iw    = self._cfg.OPENPOSE_HAND_INPUT_WIDTH
        ih    = self._cfg.OPENPOSE_HAND_INPUT_HEIGHT
        thr   = self._cfg.OPENPOSE_HAND_CONFIDENCE_THRESHOLD

        blob = cv2.dnn.blobFromImage(
            bgr, scalefactor=1.0 / 255,
            size=(iw, ih), mean=(0, 0, 0),
            swapRB=False, crop=False,
        )
        self._net.setInput(blob)
        out = self._net.forward()   # [1, 22, out_h, out_w]  (21 kp + background)

        out_h, out_w = out.shape[2], out.shape[3]
        keypoints = []
        for i in range(21):
            heatmap = out[0, i]
            _, conf, _, peak = cv2.minMaxLoc(heatmap)
            if conf > thr:
                x = int((peak[0] / out_w) * w)
                y = int((peak[1] / out_h) * h)
                keypoints.append((x, y, float(conf)))
            else:
                keypoints.append(None)
        return keypoints

    def draw(self, frame: np.ndarray, result: list) -> np.ndarray:
        for (a, b) in _OPENPOSE_HAND_EDGES:
            if result[a] is not None and result[b] is not None:
                ax, ay, _ = result[a]
                bx, by, _ = result[b]
                cv2.line(frame, (ax, ay), (bx, by), (0, 255, 255), 2)

        for i, kp in enumerate(result):
            if kp is not None:
                x, y, _ = kp
                tip    = i in _OPENPOSE_HAND_TIPS
                colour = (0, 0, 255) if tip else (0, 255, 0)
                radius = 8 if tip else 5
                cv2.circle(frame, (x, y), radius, colour, -1)

        return frame

    def close(self) -> None:
        self._net    = None
        self._loaded = False


# ---------------------------------------------------------------------------
# ModelManager — owns all models, handles lazy loading and cycling
# ---------------------------------------------------------------------------

class ModelManager:
    """
    Manages a list of PoseModel instances. Models are not loaded until
    first selected (lazy init). The M key calls cycle() to advance.
    """

    def __init__(self, config, project_root: str):
        self._order = config.MODEL_ORDER
        self._index = 0

        self._models = {
            config.MODEL_MEDIAPIPE:     MediaPipeHandsModel(config),
            config.MODEL_OPENPOSE_HAND: OpenPoseHandModel(config, project_root),
        }

    @property
    def current(self) -> PoseModel:
        return self._models[self._order[self._index]]

    def cycle(self) -> None:
        """Advance to the next model. Prints a status line."""
        prev = self.current.name
        self._index = (self._index + 1) % len(self._order)
        print(f"Model: {prev}  ->  {self.current.name}")

    def ensure_loaded(self) -> None:
        """Load the current model if it hasn't been loaded yet."""
        if not self.current.loaded:
            self.current.load()

    def close_all(self) -> None:
        for model in self._models.values():
            if model.loaded:
                model.close()
